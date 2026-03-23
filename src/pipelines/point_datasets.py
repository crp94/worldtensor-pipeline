"""Generic yearly point-dataset raster pipeline (events + stocks).

Supports:
- event datasets: yearly counts/intensity + cumulative + distance to nearest event
- stock datasets: active/added/retired counts + value stocks + distance + accessibility

Configured via `config/point_datasets.yml`.
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import yaml
from scipy.spatial import cKDTree
from tqdm import tqdm

from src.grid import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, N_LAT, N_LON, RESOLUTION, make_template
from src.utils import add_cyclic_point_xr, get_logger
from src.year_policy import resolve_year_bounds

logger = get_logger("pipeline.point_datasets")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "point_datasets.yml"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)
EARTH_RADIUS_KM = 6371.0088

LAT_RAD_1D = np.deg2rad(TARGET_LAT)
LON_RAD_1D = np.deg2rad(TARGET_LON)
LAT_RAD_2D = LAT_RAD_1D[:, np.newaxis]
LON_RAD_2D = LON_RAD_1D[np.newaxis, :]
COS_LAT_2D = np.cos(LAT_RAD_2D)
SIN_LAT_2D = np.sin(LAT_RAD_2D)
COS_LON_2D = np.cos(LON_RAD_2D)
SIN_LON_2D = np.sin(LON_RAD_2D)
GRID_XYZ = np.stack(
    [
        (COS_LAT_2D * COS_LON_2D).ravel(),
        (COS_LAT_2D * SIN_LON_2D).ravel(),
        np.broadcast_to(SIN_LAT_2D, (N_LAT, N_LON)).ravel(),
    ],
    axis=1,
)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _to_year(series: pd.Series) -> pd.Series:
    y = pd.to_numeric(series, errors="coerce")
    y = y.where((y >= 1000) & (y <= 2200))
    return y.round().astype("Int64")


def _resolve_column(df: pd.DataFrame, candidates: list[str] | str, required: bool = True) -> str | None:
    cand = [candidates] if isinstance(candidates, str) else list(candidates)
    for c in cand:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f"Missing required column; candidates={cand}")
    return None


def _to_grid_indices(lat: pd.Series, lon: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lat_idx = np.floor((lat.to_numpy(dtype=np.float64) - LAT_MIN) / RESOLUTION).astype(np.int32)
    lon_0360 = np.mod(lon.to_numpy(dtype=np.float64), 360.0)
    lon_idx = np.floor(lon_0360 / RESOLUTION).astype(np.int32)
    lat_idx = np.clip(lat_idx, 0, N_LAT - 1)
    lon_idx = np.clip(lon_idx, 0, N_LON - 1)
    return lat_idx, lon_idx


def _aggregate_grid(lat_idx: np.ndarray, lon_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    out = np.zeros((N_LAT, N_LON), dtype=np.float32)
    if values.size == 0:
        return out
    np.add.at(out, (lat_idx, lon_idx), values.astype(np.float32))
    return out


def _load_land_mask(mask_nc: Path) -> np.ndarray:
    if not mask_nc.exists():
        raise FileNotFoundError(f"Land-mask file not found: {mask_nc}")
    ds = xr.open_dataset(mask_nc)
    var = list(ds.data_vars)[0]
    da = ds[var]
    if "time" in da.dims:
        da = da.isel(time=0)
    vals = da.values
    ds.close()
    if vals.shape != (N_LAT, N_LON):
        raise ValueError(f"Land-mask shape must be {(N_LAT, N_LON)}, got {vals.shape}")
    return np.isfinite(vals) & (vals <= 0)


def _apply_land_mask(data: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    out = data.astype(np.float32, copy=True)
    out[~land_mask] = np.nan
    return out


def _distance_to_nearest_occupied_km(occupied: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    if not occupied.any():
        return np.full((N_LAT, N_LON), np.nan, dtype=np.float32)

    src_idx = np.flatnonzero(occupied.ravel())
    tree = cKDTree(GRID_XYZ[src_idx])
    qry_idx = np.flatnonzero(valid_mask.ravel())
    chord, _ = tree.query(GRID_XYZ[qry_idx], k=1, workers=-1)
    angle = 2.0 * np.arcsin(np.clip(chord * 0.5, 0.0, 1.0))
    dist = (EARTH_RADIUS_KM * angle).astype(np.float32)

    out = np.full(N_LAT * N_LON, np.nan, dtype=np.float32)
    out[qry_idx] = dist
    return out.reshape(N_LAT, N_LON)


def _read_tabular(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path, engine="openpyxl")
    if ext == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported input format: {path}")


def _norm_from_metric(var_name: str, data: np.ndarray):
    finite = data[np.isfinite(data)]
    if finite.size == 0:
        return None

    lower = var_name.lower()
    if "distance" in lower:
        vmax = max(float(np.nanpercentile(finite, 99)), 1.0)
        return mcolors.Normalize(vmin=0.0, vmax=vmax)

    if "accessibility" in lower:
        vmax = max(float(np.nanpercentile(finite, 99)), 1e-6)
        return mcolors.Normalize(vmin=0.0, vmax=vmax)

    if np.nanmin(finite) < 0:
        vmax = max(float(np.nanpercentile(np.abs(finite), 99)), 1e-6)
        return mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)

    positive = finite[finite > 0]
    if positive.size:
        vmin = max(float(np.nanpercentile(positive, 1)), 1e-6)
        vmax = max(float(np.nanpercentile(positive, 99)), vmin * 10)
        return mcolors.LogNorm(vmin=vmin, vmax=vmax)
    return mcolors.Normalize(vmin=0.0, vmax=1.0)


def _plot_map(data: np.ndarray, var_name: str, year: int, units: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    da = xr.DataArray(data, dims=("lat", "lon"), coords={"lat": TARGET_LAT, "lon": TARGET_LON})
    data_cyclic, lon_cyclic, lat_values = add_cyclic_point_xr(da)
    plot_data = data_cyclic.filled(np.nan) if np.ma.isMaskedArray(data_cyclic) else np.asarray(data_cyclic)

    if lat_values.size > 2 and np.isclose(lat_values[0], -90.0):
        lat_values = lat_values[1:]
        plot_data = plot_data[1:, :]
    if lat_values.size > 2 and np.isclose(lat_values[-1], 90.0):
        lat_values = lat_values[:-1]
        plot_data = plot_data[:-1, :]

    lon_step = float(np.median(np.diff(lon_cyclic)))
    lat_step = float(np.median(np.diff(lat_values)))
    lon_edges = np.concatenate(([lon_cyclic[0] - lon_step / 2.0], 0.5 * (lon_cyclic[:-1] + lon_cyclic[1:]), [lon_cyclic[-1] + lon_step / 2.0]))
    lat_edges = np.concatenate(([lat_values[0] - lat_step / 2.0], 0.5 * (lat_values[:-1] + lat_values[1:]), [lat_values[-1] + lat_step / 2.0]))

    norm = _norm_from_metric(var_name, plot_data)
    cmap = "viridis" if "distance" in var_name else "YlOrRd"

    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": ccrs.Robinson()})
    im = ax.pcolormesh(
        lon_edges,
        lat_edges,
        plot_data,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading="flat",
        antialiased=False,
    )
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, aspect=45, shrink=0.8, label=units)
    ax.coastlines(linewidth=0.5, color="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.set_title(f"{var_name} - {year}")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_timeseries(var_name: str, years: list[int], values: list[float], units: str, out_path: Path) -> None:
    if not years:
        return
    vals = np.asarray(values, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(years, values, color="#1f77b4", linewidth=1.2)
    ax.set_xlabel("Year")
    ax.set_ylabel(units)
    ax.set_title(f"Global summary: {var_name}")
    ax.grid(True, alpha=0.3)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _save_grid(
    grid: np.ndarray,
    var_name: str,
    long_name: str,
    units: str,
    year: int,
    out_dir: Path,
    source_name: str,
    overwrite: bool,
) -> Path | None:
    out_path = out_dir / var_name / f"{year}.nc"
    if out_path.exists() and not overwrite:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = make_template(year)
    ds[var_name] = (
        ["time", "lat", "lon"],
        grid.astype(np.float32)[np.newaxis, :, :],
        {"units": units, "long_name": long_name},
    )
    ds.attrs.update({"title": f"WorldTensor {var_name}", "source": source_name, "year": int(year)})
    ds.to_netcdf(out_path, encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}})
    return out_path


def _dataset_years_event(df: pd.DataFrame, year_col: str, max_output_year: int, start_year: int | None, end_year: int | None) -> list[int]:
    y = _to_year(df[year_col]).dropna()
    if y.empty:
        raise ValueError("No valid event years")
    y_min = int(y.min())
    y_max = min(int(y.max()), max_output_year)
    if start_year is not None:
        y_min = max(y_min, start_year)
    if end_year is not None:
        y_max = min(y_max, end_year)
    if y_max < y_min:
        raise ValueError(f"Invalid year range {y_min}..{y_max}")
    return list(range(y_min, y_max + 1))


def _dataset_years_stock(
    df: pd.DataFrame,
    start_col: str,
    end_col: str | None,
    default_start_year: int,
    max_output_year: int,
    start_year: int | None,
    end_year: int | None,
) -> list[int]:
    start_vals = _to_year(df[start_col]).fillna(default_start_year).astype("Int64")

    y_min = int(start_vals.min()) - 1
    y_max = max_output_year

    if start_year is not None:
        y_min = max(y_min, start_year)
    if end_year is not None:
        y_max = min(y_max, end_year)
    if y_max < y_min:
        raise ValueError(f"Invalid year range {y_min}..{y_max}")
    return list(range(y_min, y_max + 1))


def _prepare_values(df: pd.DataFrame, values_cfg: list[dict]) -> list[dict]:
    out: list[dict] = []
    for vcfg in values_cfg or []:
        col = _resolve_column(df, vcfg.get("columns", []), required=False)
        if not col:
            continue
        key = str(vcfg["key"])
        arr = pd.to_numeric(df[col], errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
        out.append(
            {
                "key": key,
                "column": col,
                "values": arr,
                "units": str(vcfg.get("units", "units per cell")),
                "long_name": str(vcfg.get("long_name", key)),
                "include_cumulative": bool(vcfg.get("include_cumulative", True)),
                "include_accessibility": bool(vcfg.get("include_accessibility", False)),
                "start_year": int(vcfg["start_year"]) if vcfg.get("start_year") is not None else None,
                "end_year": int(vcfg["end_year"]) if vcfg.get("end_year") is not None else None,
            }
        )
    return out


def _value_year_mask(year_vals: np.ndarray, start_year: int | None, end_year: int | None) -> np.ndarray:
    mask = np.ones(year_vals.shape, dtype=bool)
    if start_year is not None:
        mask &= year_vals >= float(start_year)
    if end_year is not None:
        mask &= year_vals <= float(end_year)
    return mask


def _value_active_in_year(year: int, start_year: int | None, end_year: int | None) -> bool:
    if start_year is not None and year < start_year:
        return False
    if end_year is not None and year > end_year:
        return False
    return True


def _process_event_dataset(
    key: str,
    ds_cfg: dict,
    years: list[int],
    frame: pd.DataFrame,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    year_vals: np.ndarray,
    value_defs: list[dict],
    land_mask: np.ndarray,
    final_root: Path,
    plots_root: Path,
    overwrite: bool,
    plot: bool,
    plot_years: set[int],
) -> None:
    var_prefix = ds_cfg["var_prefix"]
    domain = ds_cfg["domain"]
    source_name = ds_cfg["source_name"]

    summaries: dict[str, list[float]] = {}
    years_by_var: dict[str, list[int]] = {}
    units_by_var: dict[str, str] = {}

    include_count = bool(ds_cfg.get("include_count", True))
    include_distance = bool(ds_cfg.get("include_distance", True))
    include_cum_count = bool(ds_cfg.get("include_cumulative_count", True))
    missing_years = {int(y) for y in ds_cfg.get("missing_years", [])}

    for year in tqdm(years, desc=f"{key}:year"):
        y_mask = year_vals == year
        y_cum_mask = year_vals <= year

        grids: dict[str, tuple[np.ndarray, str, str, str]] = {}

        if year in missing_years:
            nan_grid = np.full((N_LAT, N_LON), np.nan, dtype=np.float32)
            if include_count:
                v = f"{var_prefix}_event_count_total"
                grids[v] = (nan_grid.copy(), "Event count", "events per cell", "sum")
            if include_cum_count:
                v = f"{var_prefix}_cumulative_event_count_total"
                grids[v] = (nan_grid.copy(), "Cumulative event count", "events per cell", "sum")
            if include_distance:
                v = f"{var_prefix}_distance_to_nearest_event_km_total"
                grids[v] = (nan_grid.copy(), "Distance to nearest event", "km", "mean")

            for vdef in value_defs:
                if not _value_active_in_year(year, vdef["start_year"], vdef["end_year"]):
                    continue
                key_name = vdef["key"]
                v = f"{var_prefix}_{key_name}_total"
                grids[v] = (nan_grid.copy(), vdef["long_name"], vdef["units"], "sum")
                if vdef["include_cumulative"]:
                    v2 = f"{var_prefix}_cumulative_{key_name}_total"
                    grids[v2] = (nan_grid.copy(), f"Cumulative {vdef['long_name'].lower()}", vdef["units"], "sum")
        else:
            if include_count:
                count = _aggregate_grid(lat_idx[y_mask], lon_idx[y_mask], np.ones(np.count_nonzero(y_mask), dtype=np.float32))
                count = _apply_land_mask(count, land_mask)
                v = f"{var_prefix}_event_count_total"
                grids[v] = (count, "Event count", "events per cell", "sum")

            if include_cum_count:
                cum_count = _aggregate_grid(
                    lat_idx[y_cum_mask], lon_idx[y_cum_mask], np.ones(np.count_nonzero(y_cum_mask), dtype=np.float32)
                )
                cum_count = _apply_land_mask(cum_count, land_mask)
                v = f"{var_prefix}_cumulative_event_count_total"
                grids[v] = (cum_count, "Cumulative event count", "events per cell", "sum")

            if include_distance:
                occ = np.zeros((N_LAT, N_LON), dtype=bool)
                if np.any(y_mask):
                    occ[lat_idx[y_mask], lon_idx[y_mask]] = True
                dist = _distance_to_nearest_occupied_km(occ, land_mask)
                dist = _apply_land_mask(dist, land_mask)
                v = f"{var_prefix}_distance_to_nearest_event_km_total"
                grids[v] = (dist, "Distance to nearest event", "km", "mean")

            for vdef in value_defs:
                if not _value_active_in_year(year, vdef["start_year"], vdef["end_year"]):
                    continue
                key_name = vdef["key"]
                vals = vdef["values"]
                value_mask = _value_year_mask(year_vals, vdef["start_year"], vdef["end_year"])
                yr_mask = y_mask & value_mask
                cum_mask = y_cum_mask & value_mask
                yr = _aggregate_grid(lat_idx[yr_mask], lon_idx[yr_mask], vals[yr_mask])
                yr = _apply_land_mask(yr, land_mask)
                v = f"{var_prefix}_{key_name}_total"
                grids[v] = (yr, vdef["long_name"], vdef["units"], "sum")

                if vdef["include_cumulative"]:
                    cum = _aggregate_grid(lat_idx[cum_mask], lon_idx[cum_mask], vals[cum_mask])
                    cum = _apply_land_mask(cum, land_mask)
                    v2 = f"{var_prefix}_cumulative_{key_name}_total"
                    grids[v2] = (cum, f"Cumulative {vdef['long_name'].lower()}", vdef["units"], "sum")

        for var_name, (grid, long_name, units, agg) in grids.items():
            _save_grid(
                grid=grid,
                var_name=var_name,
                long_name=long_name,
                units=units,
                year=year,
                out_dir=final_root / domain,
                source_name=source_name,
                overwrite=overwrite,
            )

            if agg == "mean":
                summary = float(np.nanmean(grid)) if np.isfinite(grid).any() else float("nan")
                units_by_var[var_name] = f"{units} (global mean)"
            else:
                summary = float(np.nansum(grid)) if np.isfinite(grid).any() else float("nan")
                units_by_var[var_name] = f"{units} (global sum)"
            summaries.setdefault(var_name, []).append(summary)
            years_by_var.setdefault(var_name, []).append(year)

            if plot and year in plot_years:
                finite = grid[np.isfinite(grid)]
                if finite.size == 0:
                    continue
                if var_name.endswith("_distance_to_nearest_event_km_total") and float(np.nanmax(finite)) <= 0:
                    continue
                if not var_name.endswith("_distance_to_nearest_event_km_total") and np.allclose(finite, 0.0):
                    continue
                _plot_map(
                    data=grid,
                    var_name=var_name,
                    year=year,
                    units=units,
                    out_path=plots_root / key / "maps" / var_name / f"{year}.png",
                )

        gc.collect()
        plt.close("all")

    if plot:
        for var_name, vals in summaries.items():
            _plot_timeseries(
                var_name=var_name,
                years=years_by_var[var_name],
                values=vals,
                units=units_by_var.get(var_name, ""),
                out_path=plots_root / key / "timeseries" / f"{var_name}.png",
            )


def _process_stock_dataset(
    key: str,
    ds_cfg: dict,
    years: list[int],
    frame: pd.DataFrame,
    lat_idx: np.ndarray,
    lon_idx: np.ndarray,
    start_vals: np.ndarray,
    end_vals: np.ndarray,
    value_defs: list[dict],
    land_mask: np.ndarray,
    final_root: Path,
    plots_root: Path,
    overwrite: bool,
    plot: bool,
    plot_years: set[int],
) -> None:
    var_prefix = ds_cfg["var_prefix"]
    domain = ds_cfg["domain"]
    source_name = ds_cfg["source_name"]

    summaries: dict[str, list[float]] = {}
    years_by_var: dict[str, list[int]] = {}
    units_by_var: dict[str, str] = {}

    include_distance = bool(ds_cfg.get("include_distance", True))
    include_accessibility = bool(ds_cfg.get("include_accessibility", True))

    for year in tqdm(years, desc=f"{key}:year"):
        valid_start = ~np.isnan(start_vals)
        valid_end = ~np.isnan(end_vals)

        active_mask = valid_start & (start_vals <= year) & (~valid_end | (end_vals > year))
        added_mask = valid_start & (start_vals == year)
        retired_mask = valid_end & (end_vals == year)
        cum_added_mask = valid_start & (start_vals <= year)

        active_count = _apply_land_mask(
            _aggregate_grid(lat_idx[active_mask], lon_idx[active_mask], np.ones(np.count_nonzero(active_mask), dtype=np.float32)),
            land_mask,
        )
        added_count = _apply_land_mask(
            _aggregate_grid(lat_idx[added_mask], lon_idx[added_mask], np.ones(np.count_nonzero(added_mask), dtype=np.float32)),
            land_mask,
        )
        retired_count = _apply_land_mask(
            _aggregate_grid(lat_idx[retired_mask], lon_idx[retired_mask], np.ones(np.count_nonzero(retired_mask), dtype=np.float32)),
            land_mask,
        )
        cum_added_count = _apply_land_mask(
            _aggregate_grid(lat_idx[cum_added_mask], lon_idx[cum_added_mask], np.ones(np.count_nonzero(cum_added_mask), dtype=np.float32)),
            land_mask,
        )

        grids: dict[str, tuple[np.ndarray, str, str, str]] = {
            f"{var_prefix}_active_site_count_total": (active_count, "Active site count", "sites per cell", "sum"),
            f"{var_prefix}_added_site_count_total": (added_count, "Added site count", "sites per cell", "sum"),
            f"{var_prefix}_retired_site_count_total": (retired_count, "Retired site count", "sites per cell", "sum"),
            f"{var_prefix}_cumulative_added_site_count_total": (
                cum_added_count,
                "Cumulative added site count",
                "sites per cell",
                "sum",
            ),
        }

        distance_grid = None
        if include_distance:
            occ = np.nan_to_num(active_count, nan=0.0) > 0
            distance_grid = _apply_land_mask(_distance_to_nearest_occupied_km(occ, land_mask), land_mask)
            grids[f"{var_prefix}_distance_to_nearest_site_km_total"] = (
                distance_grid,
                "Distance to nearest active site",
                "km",
                "mean",
            )

        if include_accessibility and distance_grid is not None:
            with np.errstate(divide="ignore", invalid="ignore"):
                acc_count = np.where(np.isfinite(distance_grid), active_count / (distance_grid + 1.0), np.nan).astype(np.float32)
            grids[f"{var_prefix}_accessibility_index_sites_per_km_total"] = (
                acc_count,
                "Site accessibility index",
                "sites/km",
                "mean",
            )

        for vdef in value_defs:
            vals = vdef["values"]
            key_name = vdef["key"]
            units = vdef["units"]
            long_name = vdef["long_name"]

            active_val = _apply_land_mask(_aggregate_grid(lat_idx[active_mask], lon_idx[active_mask], vals[active_mask]), land_mask)
            added_val = _apply_land_mask(_aggregate_grid(lat_idx[added_mask], lon_idx[added_mask], vals[added_mask]), land_mask)
            retired_val = _apply_land_mask(
                _aggregate_grid(lat_idx[retired_mask], lon_idx[retired_mask], vals[retired_mask]), land_mask
            )
            cum_val = _apply_land_mask(
                _aggregate_grid(lat_idx[cum_added_mask], lon_idx[cum_added_mask], vals[cum_added_mask]), land_mask
            )

            grids[f"{var_prefix}_active_{key_name}_total"] = (active_val, f"Active {long_name.lower()}", units, "sum")
            grids[f"{var_prefix}_added_{key_name}_total"] = (added_val, f"Added {long_name.lower()}", units, "sum")
            grids[f"{var_prefix}_retired_{key_name}_total"] = (
                retired_val,
                f"Retired {long_name.lower()}",
                units,
                "sum",
            )
            grids[f"{var_prefix}_cumulative_{key_name}_total"] = (
                cum_val,
                f"Cumulative {long_name.lower()}",
                units,
                "sum",
            )

            if include_accessibility and distance_grid is not None and vdef["include_accessibility"]:
                with np.errstate(divide="ignore", invalid="ignore"):
                    acc_val = np.where(np.isfinite(distance_grid), active_val / (distance_grid + 1.0), np.nan).astype(np.float32)
                grids[f"{var_prefix}_accessibility_{key_name}_per_km_total"] = (
                    acc_val,
                    f"Accessibility-adjusted {long_name.lower()}",
                    f"{units}/km",
                    "mean",
                )

        for var_name, (grid, long_name, units, agg) in grids.items():
            _save_grid(
                grid=grid,
                var_name=var_name,
                long_name=long_name,
                units=units,
                year=year,
                out_dir=final_root / domain,
                source_name=source_name,
                overwrite=overwrite,
            )

            if agg == "mean":
                summary = float(np.nanmean(grid)) if np.isfinite(grid).any() else float("nan")
                units_by_var[var_name] = f"{units} (global mean)"
            else:
                summary = float(np.nansum(grid))
                units_by_var[var_name] = f"{units} (global sum)"
            summaries.setdefault(var_name, []).append(summary)
            years_by_var.setdefault(var_name, []).append(year)

            if plot and year in plot_years:
                finite = grid[np.isfinite(grid)]
                if finite.size == 0:
                    continue
                if "distance_to_nearest" in var_name and float(np.nanmax(finite)) <= 0:
                    continue
                if "distance_to_nearest" not in var_name and np.allclose(finite, 0.0):
                    continue
                _plot_map(
                    data=grid,
                    var_name=var_name,
                    year=year,
                    units=units,
                    out_path=plots_root / key / "maps" / var_name / f"{year}.png",
                )

        gc.collect()
        plt.close("all")

    if plot:
        for var_name, vals in summaries.items():
            _plot_timeseries(
                var_name=var_name,
                years=years_by_var[var_name],
                values=vals,
                units=units_by_var.get(var_name, ""),
                out_path=plots_root / key / "timeseries" / f"{var_name}.png",
            )


def _run_dataset(
    key: str,
    ds_cfg: dict,
    final_root: Path,
    plots_root: Path,
    min_output_year: int,
    max_output_year: int,
    land_mask: np.ndarray,
    overwrite: bool,
    start_year: int | None,
    end_year: int | None,
    plot: bool,
    default_plot_every: int,
) -> None:
    raw_path = PROJECT_ROOT / ds_cfg["raw_path"]
    if not raw_path.exists():
        logger.warning("[%s] Raw file missing, skipping: %s", key, raw_path)
        return

    df = _read_tabular(raw_path)
    cols = ds_cfg["columns"]

    lat_col = _resolve_column(df, cols["latitude"], required=True)
    lon_col = _resolve_column(df, cols["longitude"], required=True)

    df["latitude"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["longitude"] = pd.to_numeric(df[lon_col], errors="coerce")
    valid = np.isfinite(df["latitude"]) & np.isfinite(df["longitude"])
    df = df.loc[valid].copy()

    land_only = bool(ds_cfg.get("land_only", True))
    spatial_mask = land_mask if land_only else np.ones_like(land_mask, dtype=bool)

    lat_idx, lon_idx = _to_grid_indices(df["latitude"], df["longitude"])
    df["lat_idx"] = lat_idx
    df["lon_idx"] = lon_idx
    if land_only:
        on_land = land_mask[lat_idx, lon_idx]
        df = df.loc[on_land].copy()
    lat_idx = df["lat_idx"].to_numpy(dtype=np.int32)
    lon_idx = df["lon_idx"].to_numpy(dtype=np.int32)

    value_defs = _prepare_values(df, ds_cfg.get("values", []))
    plot_every = int(ds_cfg.get("plot_every", default_plot_every))
    dataset_start_year = start_year if start_year is not None else min_output_year
    if ds_cfg.get("start_year") is not None:
        dataset_start_year = max(dataset_start_year, int(ds_cfg["start_year"]))
    dataset_end_year = end_year
    if ds_cfg.get("end_year") is not None:
        configured_end = int(ds_cfg["end_year"])
        dataset_end_year = configured_end if dataset_end_year is None else min(dataset_end_year, configured_end)
    if dataset_end_year is not None and dataset_end_year < dataset_start_year:
        logger.warning("[%s] no valid years after bounds %s-%s", key, dataset_start_year, dataset_end_year)
        return
    logger.info("[%s] spatial_mask=%s", key, "land-only" if land_only else "land+ocean")

    if ds_cfg["mode"] == "event":
        year_col = _resolve_column(df, cols["year"], required=True)
        df["year"] = _to_year(df[year_col])
        df = df.loc[df["year"].notna()].copy()
        year_vals = df["year"].to_numpy(dtype=np.float64)
        lat_idx = df["lat_idx"].to_numpy(dtype=np.int32)
        lon_idx = df["lon_idx"].to_numpy(dtype=np.int32)

        years = _dataset_years_event(
            df,
            "year",
            max_output_year=max_output_year,
            start_year=dataset_start_year,
            end_year=dataset_end_year,
        )
        plot_years = set(range(years[0], years[-1] + 1, max(plot_every, 1)))
        plot_years.add(years[-1])

        logger.info("[%s] rows=%d years=%d-%d", key, len(df), years[0], years[-1])
        _process_event_dataset(
            key=key,
            ds_cfg=ds_cfg,
            years=years,
            frame=df,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            year_vals=year_vals,
            value_defs=value_defs,
            land_mask=spatial_mask,
            final_root=final_root,
            plots_root=plots_root,
            overwrite=overwrite,
            plot=plot,
            plot_years=plot_years,
        )

    elif ds_cfg["mode"] == "stock":
        start_col = _resolve_column(df, cols["start_year"], required=False)
        end_col = _resolve_column(df, cols.get("end_year", []), required=False)

        default_start_year = int(ds_cfg.get("default_start_year", 2000))
        if start_col is None:
            df["start_year"] = pd.Series(default_start_year, index=df.index, dtype="Int64")
        else:
            df["start_year"] = _to_year(df[start_col]).fillna(default_start_year).astype("Int64")

        if end_col is None:
            df["end_year"] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        else:
            df["end_year"] = _to_year(df[end_col])

        start_vals = df["start_year"].to_numpy(dtype=np.float64)
        end_vals = df["end_year"].to_numpy(dtype=np.float64)

        years = _dataset_years_stock(
            df,
            start_col="start_year",
            end_col="end_year",
            default_start_year=default_start_year,
            max_output_year=max_output_year,
            start_year=dataset_start_year,
            end_year=dataset_end_year,
        )
        plot_years = set(range(years[0], years[-1] + 1, max(plot_every, 1)))
        plot_years.add(years[-1])

        logger.info("[%s] rows=%d years=%d-%d", key, len(df), years[0], years[-1])
        _process_stock_dataset(
            key=key,
            ds_cfg=ds_cfg,
            years=years,
            frame=df,
            lat_idx=lat_idx,
            lon_idx=lon_idx,
            start_vals=start_vals,
            end_vals=end_vals,
            value_defs=value_defs,
            land_mask=spatial_mask,
            final_root=final_root,
            plots_root=plots_root,
            overwrite=overwrite,
            plot=plot,
            plot_years=plot_years,
        )
    else:
        raise ValueError(f"Unsupported mode for {key}: {ds_cfg['mode']}")


@click.command()
@click.option("--all", "run_all", is_flag=True, help="Run all enabled point datasets.")
@click.option("--dataset", "datasets", multiple=True, help="Dataset key(s) from config, e.g. --dataset ucdp_ged")
@click.option("--start-year", type=int, default=None, help="Optional global start-year override.")
@click.option("--end-year", type=int, default=None, help="Optional global end-year override.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
@click.option("--plot/--no-plot", default=True, show_default=True, help="Generate map/timeseries plots.")
def main(run_all, datasets, start_year, end_year, overwrite, plot):
    if not run_all:
        click.echo("Specify --all")
        return

    cfg = load_config()
    out_cfg = cfg.get("output", {})
    filters_cfg = cfg.get("filters", {})

    final_root = PROJECT_ROOT / out_cfg.get("final_root", "data/final")
    plots_root = PROJECT_ROOT / out_cfg.get("plots_root", "plots/point_datasets")
    default_plot_every = int(out_cfg.get("plot_every", 10))
    min_output_year = int(cfg.get("min_output_year", 1900))
    max_output_year = int(cfg.get("max_output_year", 2025))
    requested_start_year, requested_end_year = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=min_output_year,
        default_end=max_output_year,
        label="point datasets years",
    )

    land_mask_path = PROJECT_ROOT / filters_cfg.get("land_mask_path", "data/final/static/geography/dist_to_coast.nc")
    land_mask = _load_land_mask(land_mask_path)

    all_datasets = cfg.get("datasets", {})
    if datasets:
        selected = {k: all_datasets[k] for k in datasets if k in all_datasets}
    else:
        selected = {k: v for k, v in all_datasets.items() if v.get("enabled", True)}

    if not selected:
        click.echo("No datasets selected.")
        return

    click.echo(f"Running point datasets: {', '.join(selected.keys())}")
    for key, ds_cfg in selected.items():
        try:
            _run_dataset(
                key=key,
                ds_cfg=ds_cfg,
                final_root=final_root,
                plots_root=plots_root,
                min_output_year=min_output_year,
                max_output_year=max_output_year,
                land_mask=land_mask,
                overwrite=overwrite,
                start_year=requested_start_year,
                end_year=requested_end_year,
                plot=plot,
                default_plot_every=default_plot_every,
            )
        except Exception as e:
            logger.exception("[%s] failed: %s", key, e)

    click.echo("\nPoint datasets pipeline complete.")


if __name__ == "__main__":
    main()
