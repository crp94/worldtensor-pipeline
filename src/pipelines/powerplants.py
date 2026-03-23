"""Yearly power-plant raster pipeline from GEM Global Integrated Power Tracker.

Reads `Global-Integrated-Power-March-2026.xlsx` and generates 0.25° yearly grids
for totals and by-type, using:
    - active capacity stock
    - added capacity
    - retired capacity
    - net capacity change

Output layout:
    data/final/energy/power_<metric>_capacity_mw_<type_or_total>/{YYYY}.nc
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

logger = get_logger("pipeline.powerplants")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "powerplants.yml"
TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)
MAX_OUTPUT_YEAR = 2025

CORE_METRICS = ("active", "added", "retired", "net", "distance")
DERIVED_METRICS = (
    "renewable_proximity_advantage",
    "clean_accessibility",
    "active_plant_count",
    "avg_active_plant_size",
    "active_mean_age",
    "type_diversity_entropy",
    "cumulative_retired",
)
METRICS = CORE_METRICS + DERIVED_METRICS
METRIC_LONG = {
    "active": "Active power generating capacity",
    "added": "Added power generating capacity",
    "retired": "Retired power generating capacity",
    "net": "Net power generating capacity change",
    "distance": "Distance to nearest active power plant",
    "renewable_proximity_advantage": "Renewable proximity advantage (distance fossil - distance renewables)",
    "clean_accessibility": "Clean power accessibility index (renewable capacity / (distance+1))",
    "active_plant_count": "Active power plant count",
    "avg_active_plant_size": "Average active plant size",
    "active_mean_age": "Capacity-weighted mean age of active plants",
    "type_diversity_entropy": "Active generation type diversity entropy",
    "cumulative_retired": "Cumulative retired power capacity",
}
METRIC_CMAP = {
    "active": "YlOrBr",
    "added": "YlGnBu",
    "retired": "OrRd",
    "net": "RdBu_r",
    "distance": "viridis",
    "renewable_proximity_advantage": "RdBu",
    "clean_accessibility": "Greens",
    "active_plant_count": "plasma",
    "avg_active_plant_size": "cividis",
    "active_mean_age": "viridis",
    "type_diversity_entropy": "PuBuGn",
    "cumulative_retired": "Reds",
}
LINEAR_METRICS = {
    "distance",
    "clean_accessibility",
    "active_plant_count",
    "avg_active_plant_size",
    "active_mean_age",
    "type_diversity_entropy",
}
DIVERGING_METRICS = {"net", "renewable_proximity_advantage"}

SUMMARY_METHOD = {
    "active": "sum",
    "added": "sum",
    "retired": "sum",
    "net": "sum",
    "distance": "mean",
    "renewable_proximity_advantage": "mean",
    "clean_accessibility": "mean",
    "active_plant_count": "sum",
    "avg_active_plant_size": "mean",
    "active_mean_age": "mean",
    "type_diversity_entropy": "mean",
    "cumulative_retired": "sum",
}

SERIES_UNITS = {
    "active": "MW (global sum)",
    "added": "MW (global sum)",
    "retired": "MW (global sum)",
    "net": "MW (global sum)",
    "distance": "km (global mean)",
    "renewable_proximity_advantage": "km (global mean)",
    "clean_accessibility": "MW/km (global mean)",
    "active_plant_count": "plants (global sum)",
    "avg_active_plant_size": "MW/plant (global mean)",
    "active_mean_age": "years (global mean)",
    "type_diversity_entropy": "nats (global mean)",
    "cumulative_retired": "MW (global sum)",
}

METRIC_UNITS = {
    "active": "MW per cell",
    "added": "MW per cell",
    "retired": "MW per cell",
    "net": "MW per cell",
    "distance": "km",
    "renewable_proximity_advantage": "km",
    "clean_accessibility": "MW/km",
    "active_plant_count": "plants per cell",
    "avg_active_plant_size": "MW/plant",
    "active_mean_age": "years",
    "type_diversity_entropy": "nats",
    "cumulative_retired": "MW per cell",
}
CAPACITY_UNIT_METRICS = {"active", "added", "retired", "net", "cumulative_retired"}

RENEWABLE_TYPE_SLUGS = {"bioenergy", "geothermal", "hydropower", "solar", "wind"}
FOSSIL_TYPE_SLUGS = {"coal", "oil_gas"}
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


def slugify_type(value: str) -> str:
    text = str(value).strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "unknown"


def _to_year(series: pd.Series) -> pd.Series:
    year = pd.to_numeric(series, errors="coerce")
    year = year.where((year >= 1000) & (year <= 2200))
    return year.round().astype("Int64")


def _to_grid_indices(lat: pd.Series, lon: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    lat_idx = np.floor((lat.to_numpy(dtype=np.float64) - LAT_MIN) / RESOLUTION).astype(np.int32)
    lon_0360 = np.mod(lon.to_numpy(dtype=np.float64), 360.0)
    lon_idx = np.floor(lon_0360 / RESOLUTION).astype(np.int32)
    lat_idx = np.clip(lat_idx, 0, N_LAT - 1)
    lon_idx = np.clip(lon_idx, 0, N_LON - 1)
    return lat_idx, lon_idx


def _aggregate_grid(lat_idx: np.ndarray, lon_idx: np.ndarray, values: np.ndarray) -> np.ndarray:
    grid = np.zeros((N_LAT, N_LON), dtype=np.float32)
    if values.size == 0:
        return grid
    np.add.at(grid, (lat_idx, lon_idx), values.astype(np.float32))
    return grid


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
    # dist_to_coast sign convention: <= 0 is land/coast, >0 ocean.
    return np.isfinite(vals) & (vals <= 0)


def _apply_land_mask(data: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
    out = data.astype(np.float32, copy=True)
    out[~land_mask] = np.nan
    return out


def _distance_to_nearest_occupied_km(occupied: np.ndarray, valid_mask: np.ndarray | None = None) -> np.ndarray:
    """Distance (km) to nearest occupied cell on the unit sphere."""
    if not occupied.any():
        return np.full((N_LAT, N_LON), np.nan, dtype=np.float32)

    src_idx = np.flatnonzero(occupied.ravel())
    src_xyz = GRID_XYZ[src_idx]
    tree = cKDTree(src_xyz)

    if valid_mask is None:
        qry_idx = np.arange(N_LAT * N_LON, dtype=np.int64)
    else:
        qry_idx = np.flatnonzero(valid_mask.ravel())

    qry_xyz = GRID_XYZ[qry_idx]
    chord, _ = tree.query(qry_xyz, k=1, workers=-1)
    # For unit vectors, chord length c -> central angle theta = 2*asin(c/2).
    angle = 2.0 * np.arcsin(np.clip(chord * 0.5, 0.0, 1.0))
    dist_km = (EARTH_RADIUS_KM * angle).astype(np.float32)

    out = np.full(N_LAT * N_LON, np.nan, dtype=np.float32)
    out[qry_idx] = dist_km
    return out.reshape(N_LAT, N_LON)


def _var_name(metric: str, scope: str) -> str:
    custom_names = {
        "distance": f"power_distance_to_nearest_plant_km_{scope}",
        "renewable_proximity_advantage": f"power_renewable_proximity_advantage_km_{scope}",
        "clean_accessibility": f"power_clean_accessibility_index_{scope}",
        "active_plant_count": f"power_active_plant_count_{scope}",
        "avg_active_plant_size": f"power_avg_active_plant_size_mw_{scope}",
        "active_mean_age": f"power_active_mean_age_years_{scope}",
        "type_diversity_entropy": f"power_type_diversity_entropy_{scope}",
        "cumulative_retired": f"power_cumulative_retired_capacity_mw_{scope}",
    }
    if metric in custom_names:
        return custom_names[metric]
    return f"power_{metric}_capacity_mw_{scope}"


def _build_dataset(
    data: np.ndarray,
    metric: str,
    scope: str,
    year: int,
    source_name: str,
    units: str,
    excluded_status_substrings: list[str],
) -> xr.Dataset:
    var_name = _var_name(metric, scope)
    scope_label = "all types" if scope == "total" else scope.replace("_", " ")
    long_name = f"{METRIC_LONG[metric]} ({scope_label})"

    ds = make_template(year)
    ds[var_name] = (
        ["time", "lat", "lon"],
        data.astype(np.float32)[np.newaxis, :, :],
        {"units": units, "long_name": long_name},
    )
    ds.attrs.update(
        {
            "title": f"WorldTensor Power Plants {var_name}",
            "source": source_name,
            "year": int(year),
            "metric": metric,
            "type_scope": scope,
            "status_exclusions": ", ".join(excluded_status_substrings),
        }
    )
    return ds


def _save_yearly_grid(
    data: np.ndarray,
    metric: str,
    scope: str,
    year: int,
    final_dir: Path,
    source_name: str,
    units: str,
    excluded_status_substrings: list[str],
    overwrite: bool,
) -> Path | None:
    var_name = _var_name(metric, scope)
    out_path = final_dir / var_name / f"{year}.nc"
    if out_path.exists() and not overwrite:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = _build_dataset(data, metric, scope, year, source_name, units, excluded_status_substrings)
    ds.to_netcdf(
        out_path,
        encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    return out_path


def _plot_map(
    data: np.ndarray,
    metric: str,
    scope: str,
    year: int,
    units: str,
    out_path: Path,
) -> None:
    if out_path.exists():
        return

    da = xr.DataArray(
        data,
        dims=("lat", "lon"),
        coords={"lat": TARGET_LAT, "lon": TARGET_LON},
    )
    data_cyclic, lon_cyclic, lat_values = add_cyclic_point_xr(da)
    if np.ma.isMaskedArray(data_cyclic):
        plot_data = data_cyclic.filled(np.nan)
    else:
        plot_data = np.asarray(data_cyclic)

    # Drop singular pole rows for plotting stability in global projections.
    if lat_values.size > 2 and np.isclose(lat_values[0], -90.0):
        lat_values = lat_values[1:]
        plot_data = plot_data[1:, :]
    if lat_values.size > 2 and np.isclose(lat_values[-1], 90.0):
        lat_values = lat_values[:-1]
        plot_data = plot_data[:-1, :]

    # Build cell edges from center coordinates to avoid meridian strip artifacts.
    lon_step = float(np.median(np.diff(lon_cyclic)))
    lat_step = float(np.median(np.diff(lat_values)))
    lon_edges = np.concatenate(
        ([lon_cyclic[0] - lon_step / 2.0], 0.5 * (lon_cyclic[:-1] + lon_cyclic[1:]), [lon_cyclic[-1] + lon_step / 2.0])
    )
    lat_edges = np.concatenate(
        ([lat_values[0] - lat_step / 2.0], 0.5 * (lat_values[:-1] + lat_values[1:]), [lat_values[-1] + lat_step / 2.0])
    )

    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": ccrs.Robinson()})
    cmap = METRIC_CMAP.get(metric, "viridis")

    if metric in DIVERGING_METRICS:
        finite = plot_data[np.isfinite(plot_data)]
        vmax = float(np.nanpercentile(np.abs(finite), 99)) if finite.size else 1.0
        vmax = max(vmax, 1.0)
        norm = mcolors.TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    elif metric in LINEAR_METRICS:
        finite = plot_data[np.isfinite(plot_data)]
        vmax = float(np.nanpercentile(finite, 99)) if finite.size else 1.0
        vmax = max(vmax, 1.0)
        vmin = 0.0
        if metric == "renewable_proximity_advantage":
            vmin = float(np.nanpercentile(finite, 1)) if finite.size else -1.0
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
    else:
        positive = plot_data[np.isfinite(plot_data) & (plot_data > 0)]
        if positive.size:
            vmin = max(float(np.nanpercentile(positive, 1)), 1e-3)
            vmax = max(float(np.nanpercentile(positive, 99)), vmin * 10.0)
            norm = mcolors.LogNorm(vmin=vmin, vmax=vmax)
        else:
            norm = None

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
    plt.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        aspect=45,
        shrink=0.8,
        label=units,
    )
    ax.coastlines(linewidth=0.5, color="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()
    title = METRIC_LONG.get(metric, metric).replace("(", "").replace(")", "")
    ax.set_title(f"{title} ({scope}) - {year}", fontsize=13)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _has_plottable_signal(metric: str, grid: np.ndarray) -> bool:
    finite = grid[np.isfinite(grid)]
    if finite.size == 0:
        return False
    if metric == "distance":
        return float(np.nanmax(finite)) > 0.0
    return np.any(np.abs(finite) > 0.0)


def _summarize_grid(metric: str, grid: np.ndarray) -> float:
    method = SUMMARY_METHOD.get(metric, "sum")
    if method == "mean":
        finite = grid[np.isfinite(grid)]
        if finite.size == 0:
            return float("nan")
        return float(np.nanmean(finite))
    return float(np.nansum(grid))


def _plot_timeseries(var_name: str, years: list[int], totals: list[float], units: str, out_path: Path) -> None:
    if out_path.exists() or not years:
        return
    vals = np.asarray(totals, dtype=np.float64)
    finite = vals[np.isfinite(vals)]
    if finite.size == 0:
        return
    if "distance_to_nearest_plant_km" not in var_name and np.allclose(finite, 0.0):
        return
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(years, totals, color="#1f77b4", linewidth=1.2)
    ax.set_xlabel("Year")
    ax.set_ylabel(units)
    ax.set_title(f"Global total: {var_name}")
    ax.grid(True, alpha=0.3)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _select_years(df: pd.DataFrame, start_year: int | None, end_year: int | None) -> list[int]:
    start_candidates = df["start_year"].dropna()
    year_candidates = pd.concat([df["start_year"], df["retired_year"]], axis=0).dropna()
    if start_candidates.empty or year_candidates.empty:
        raise ValueError("No valid start/retired years found in input data.")
    y_min = int(start_candidates.min()) - 1
    y_max = min(int(year_candidates.max()), MAX_OUTPUT_YEAR)
    if start_year is not None:
        y_min = max(y_min, start_year)
    if end_year is not None:
        y_max = min(y_max, end_year, MAX_OUTPUT_YEAR)
    if y_max < y_min:
        raise ValueError(f"Invalid year window: {y_min}..{y_max}")
    # Explicitly produce a continuous year range with no gaps.
    return list(range(y_min, y_max + 1))


def _scope_start_years(scope_frames: dict[str, pd.DataFrame], global_first_start: int) -> dict[str, dict[str, int]]:
    """Per-scope start windows for capacity metrics and distance."""
    starts: dict[str, dict[str, int]] = {}
    for scope, frame in scope_frames.items():
        valid = frame["start_year"].dropna()
        if valid.empty:
            first = global_first_start
        else:
            first = int(valid.min())
        cap_start = first - 1
        dist_start = first
        starts[scope] = {"capacity_start": cap_start, "distance_start": dist_start}
    return starts


def load_power_dataframe(
    xlsx_path: Path,
    sheet_name: str,
    columns_cfg: dict,
    min_capacity_mw: float,
    excluded_status_substrings: list[str],
    land_mask: np.ndarray,
) -> pd.DataFrame:
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, engine="openpyxl")

    missing_cols = [col for col in columns_cfg.values() if col not in df.columns]
    if missing_cols:
        raise KeyError(f"Missing required columns in XLSX: {missing_cols}")

    df = df.rename(columns={v: k for k, v in columns_cfg.items()})
    df["capacity_mw"] = pd.to_numeric(df["capacity_mw"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["start_year"] = _to_year(df["start_year"])
    df["retired_year"] = _to_year(df["retired_year"])
    df["status"] = df["status"].astype(str).str.strip().str.lower()
    df["type"] = df["type"].astype(str).str.strip()
    df["type_slug"] = df["type"].apply(slugify_type)

    base_valid = (
        np.isfinite(df["capacity_mw"])
        & np.isfinite(df["latitude"])
        & np.isfinite(df["longitude"])
        & (df["capacity_mw"] > float(min_capacity_mw))
    )
    df = df.loc[base_valid].copy()

    if excluded_status_substrings:
        pat = "|".join(re.escape(s.lower()) for s in excluded_status_substrings if s)
        if pat:
            df = df.loc[~df["status"].str.contains(pat, na=False)].copy()

    lat_idx, lon_idx = _to_grid_indices(df["latitude"], df["longitude"])
    df["lat_idx"] = lat_idx
    df["lon_idx"] = lon_idx
    # Keep land cells only.
    on_land = land_mask[df["lat_idx"].to_numpy(dtype=np.int32), df["lon_idx"].to_numpy(dtype=np.int32)]
    df = df.loc[on_land].copy()

    return df


@click.command()
@click.option("--all", "run_all", is_flag=True, help="Run full yearly gridding.")
@click.option(
    "--input-xlsx",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Path to Global-Integrated-Power-March-2026.xlsx.",
)
@click.option("--start-year", type=int, default=None, help="Start year override.")
@click.option("--end-year", type=int, default=None, help="End year override.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing NetCDF outputs.")
@click.option("--plot/--no-plot", default=True, show_default=True, help="Generate plot outputs.")
@click.option(
    "--plot-all-variables/--plot-totals-only",
    default=True,
    show_default=True,
    help="Plot all by-type variables or only totals.",
)
@click.option("--plot-every", type=int, default=None, help="Spatial map interval in years.")
def main(run_all, input_xlsx, start_year, end_year, overwrite, plot, plot_all_variables, plot_every):
    if not run_all:
        click.echo("Specify --all.")
        return

    cfg = load_config()
    source_file = cfg["source_file"]
    source_name = cfg["source_name"]
    sheet_name = cfg["sheet_name"]
    columns_cfg = cfg["columns"]

    out_cfg = cfg["output"]
    final_dir = PROJECT_ROOT / out_cfg["final_dir"]
    plots_dir = PROJECT_ROOT / out_cfg["plots_dir"]
    units = out_cfg.get("units", "MW per cell")
    plot_every = int(plot_every or out_cfg.get("plot_every", 10))

    flt = cfg.get("filters", {})
    min_capacity_mw = float(flt.get("min_capacity_mw", 0.0))
    land_mask_path = PROJECT_ROOT / flt.get("land_mask_path", "data/final/static/geography/dist_to_coast.nc")
    excluded_status_substrings = [s.strip().lower() for s in flt.get("excluded_status_substrings", [])]
    land_mask = _load_land_mask(land_mask_path)

    xlsx_path = input_xlsx or (PROJECT_ROOT / "data" / "raw" / source_file)
    if not xlsx_path.exists():
        raise FileNotFoundError(f"Input XLSX not found: {xlsx_path}")

    click.echo(f"Input: {xlsx_path}")
    click.echo("Loading and cleaning rows ...")
    df = load_power_dataframe(
        xlsx_path=xlsx_path,
        sheet_name=sheet_name,
        columns_cfg=columns_cfg,
        min_capacity_mw=min_capacity_mw,
        excluded_status_substrings=excluded_status_substrings,
        land_mask=land_mask,
    )

    years = _select_years(df, start_year=start_year, end_year=end_year)
    global_first_start = int(df["start_year"].dropna().min())
    type_list = sorted(t for t in df["type_slug"].dropna().unique() if t)
    scopes = ["total"] + type_list
    scope_frames = {"total": df}
    for t in type_list:
        scope_frames[t] = df.loc[df["type_slug"] == t].copy()
    scope_starts = _scope_start_years(scope_frames, global_first_start=global_first_start)

    click.echo(
        f"Rows={len(df):,}, scopes={len(scopes)} (total + {len(type_list)} types), "
        f"years={years[0]}-{years[-1]} ({len(years)} years, no gaps)"
    )

    # Track global totals for timeseries plots.
    totals_by_var: dict[str, list[float]] = {}
    years_by_var: dict[str, list[int]] = {}
    series_units: dict[str, str] = {}

    plot_years = set(range(years[0], years[-1] + 1, max(plot_every, 1)))
    plot_years.add(years[-1])
    plot_scopes = scopes if plot_all_variables else ["total"]

    total_frame = scope_frames["total"]
    total_cap = total_frame["capacity_mw"].to_numpy(dtype=np.float32)
    total_lat_idx = total_frame["lat_idx"].to_numpy(dtype=np.int32)
    total_lon_idx = total_frame["lon_idx"].to_numpy(dtype=np.int32)
    total_start = total_frame["start_year"].to_numpy(dtype=np.float64)
    total_retired = total_frame["retired_year"].to_numpy(dtype=np.float64)
    total_type = total_frame["type_slug"].to_numpy(dtype=object)
    renewable_row_mask = np.isin(total_type, list(RENEWABLE_TYPE_SLUGS))
    fossil_row_mask = np.isin(total_type, list(FOSSIL_TYPE_SLUGS))

    total_cap_start = scope_starts["total"]["capacity_start"]
    total_dist_start = scope_starts["total"]["distance_start"]
    renewable_dist_start = min(
        [scope_starts[t]["distance_start"] for t in type_list if t in RENEWABLE_TYPE_SLUGS] or [total_dist_start]
    )
    fossil_dist_start = min(
        [scope_starts[t]["distance_start"] for t in type_list if t in FOSSIL_TYPE_SLUGS] or [total_dist_start]
    )
    derived_starts = {
        "renewable_proximity_advantage": max(renewable_dist_start, fossil_dist_start),
        "clean_accessibility": renewable_dist_start,
        "active_plant_count": total_cap_start,
        "avg_active_plant_size": total_cap_start,
        "active_mean_age": total_cap_start,
        "type_diversity_entropy": total_cap_start,
        "cumulative_retired": total_cap_start,
    }

    for year in tqdm(years, desc="year"):
        total_active_grid = None
        total_distance_grid = None

        for scope in scopes:
            frame = scope_frames[scope]
            cap = frame["capacity_mw"].to_numpy(dtype=np.float32)
            lat_idx = frame["lat_idx"].to_numpy(dtype=np.int32)
            lon_idx = frame["lon_idx"].to_numpy(dtype=np.int32)
            start = frame["start_year"].to_numpy(dtype=np.float64)
            retired = frame["retired_year"].to_numpy(dtype=np.float64)

            valid_start = ~np.isnan(start)
            valid_retired = ~np.isnan(retired)
            added_mask = valid_start & (start == year)
            retired_mask = valid_retired & (retired == year)
            active_mask = valid_start & (start <= year) & (~valid_retired | (retired > year))

            added = _aggregate_grid(lat_idx[added_mask], lon_idx[added_mask], cap[added_mask])
            retired_grid = _aggregate_grid(lat_idx[retired_mask], lon_idx[retired_mask], cap[retired_mask])
            active = _aggregate_grid(lat_idx[active_mask], lon_idx[active_mask], cap[active_mask])
            net = added - retired_grid
            distance = _distance_to_nearest_occupied_km(active > 0, valid_mask=land_mask)

            grids = {
                "active": _apply_land_mask(active, land_mask),
                "added": _apply_land_mask(added, land_mask),
                "retired": _apply_land_mask(retired_grid, land_mask),
                "net": _apply_land_mask(net, land_mask),
                "distance": _apply_land_mask(distance, land_mask),
            }

            for metric, grid in grids.items():
                cap_start = scope_starts[scope]["capacity_start"]
                dist_start = scope_starts[scope]["distance_start"]
                metric_start = dist_start if metric == "distance" else cap_start
                if year < metric_start:
                    continue

                var_name = _var_name(metric, scope)
                metric_units = units if metric in CAPACITY_UNIT_METRICS else METRIC_UNITS.get(metric, units)
                _save_yearly_grid(
                    data=grid,
                    metric=metric,
                    scope=scope,
                    year=year,
                    final_dir=final_dir,
                    source_name=source_name,
                    units=metric_units,
                    excluded_status_substrings=excluded_status_substrings,
                    overwrite=overwrite,
                )

                summary_val = _summarize_grid(metric, grid)
                totals_by_var.setdefault(var_name, []).append(summary_val)
                years_by_var.setdefault(var_name, []).append(year)
                series_units[var_name] = SERIES_UNITS.get(metric, "")

                if plot and (scope in plot_scopes) and (year in plot_years):
                    if not _has_plottable_signal(metric, grid):
                        continue
                    map_out = plots_dir / "maps" / var_name / f"{year}.png"
                    _plot_map(
                        data=grid,
                        metric=metric,
                        scope=scope,
                        year=year,
                        units=metric_units,
                        out_path=map_out,
                    )

                if scope == "total" and metric == "active":
                    total_active_grid = grid
                if scope == "total" and metric == "distance":
                    total_distance_grid = grid

        # Derived metrics (total scope only).
        valid_start_total = ~np.isnan(total_start)
        valid_retired_total = ~np.isnan(total_retired)
        active_mask_total = valid_start_total & (total_start <= year) & (~valid_retired_total | (total_retired > year))
        retired_cum_mask = valid_retired_total & (total_retired <= year)

        if total_active_grid is None:
            total_active_grid = _apply_land_mask(
                _aggregate_grid(
                    total_lat_idx[active_mask_total],
                    total_lon_idx[active_mask_total],
                    total_cap[active_mask_total],
                ),
                land_mask,
            )
        if total_distance_grid is None:
            total_distance_grid = _apply_land_mask(
                _distance_to_nearest_occupied_km(np.nan_to_num(total_active_grid, nan=0.0) > 0.0, valid_mask=land_mask),
                land_mask,
            )

        renewable_active_mask = active_mask_total & renewable_row_mask
        fossil_active_mask = active_mask_total & fossil_row_mask
        renewable_active_grid = _apply_land_mask(
            _aggregate_grid(
                total_lat_idx[renewable_active_mask],
                total_lon_idx[renewable_active_mask],
                total_cap[renewable_active_mask],
            ),
            land_mask,
        )
        fossil_active_grid = _apply_land_mask(
            _aggregate_grid(
                total_lat_idx[fossil_active_mask],
                total_lon_idx[fossil_active_mask],
                total_cap[fossil_active_mask],
            ),
            land_mask,
        )
        dist_renewables = _apply_land_mask(
            _distance_to_nearest_occupied_km(np.nan_to_num(renewable_active_grid, nan=0.0) > 0.0, valid_mask=land_mask),
            land_mask,
        )
        dist_fossil = _apply_land_mask(
            _distance_to_nearest_occupied_km(np.nan_to_num(fossil_active_grid, nan=0.0) > 0.0, valid_mask=land_mask),
            land_mask,
        )

        active_count_grid = _apply_land_mask(
            _aggregate_grid(
                total_lat_idx[active_mask_total],
                total_lon_idx[active_mask_total],
                np.ones(int(np.count_nonzero(active_mask_total)), dtype=np.float32),
            ),
            land_mask,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            avg_active_size = np.where(active_count_grid > 0, total_active_grid / active_count_grid, np.nan).astype(np.float32)

        active_ages = (year - total_start[active_mask_total]).astype(np.float32)
        age_weighted_sum = _apply_land_mask(
            _aggregate_grid(
                total_lat_idx[active_mask_total],
                total_lon_idx[active_mask_total],
                total_cap[active_mask_total] * active_ages,
            ),
            land_mask,
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            active_mean_age = np.where(total_active_grid > 0, age_weighted_sum / total_active_grid, np.nan).astype(np.float32)

        entropy = np.zeros((N_LAT, N_LON), dtype=np.float32)
        active_cap_nonan = np.nan_to_num(total_active_grid, nan=0.0)
        active_cells = active_cap_nonan > 0.0
        for t in type_list:
            type_mask = active_mask_total & (total_type == t)
            type_cap_grid = _aggregate_grid(total_lat_idx[type_mask], total_lon_idx[type_mask], total_cap[type_mask])
            share = np.zeros((N_LAT, N_LON), dtype=np.float32)
            share[active_cells] = (type_cap_grid[active_cells] / active_cap_nonan[active_cells]).astype(np.float32)
            valid_share = share > 0
            entropy[valid_share] -= (share[valid_share] * np.log(share[valid_share])).astype(np.float32)
        entropy = _apply_land_mask(entropy, land_mask)

        cumulative_retired_grid = _apply_land_mask(
            _aggregate_grid(
                total_lat_idx[retired_cum_mask],
                total_lon_idx[retired_cum_mask],
                total_cap[retired_cum_mask],
            ),
            land_mask,
        )

        with np.errstate(divide="ignore", invalid="ignore"):
            renewable_proximity_advantage = np.where(
                np.isfinite(dist_renewables) & np.isfinite(dist_fossil), dist_fossil - dist_renewables, np.nan
            ).astype(np.float32)
            clean_accessibility = np.where(
                np.isfinite(dist_renewables), renewable_active_grid / (dist_renewables + 1.0), np.nan
            ).astype(np.float32)

        derived_grids = {
            "renewable_proximity_advantage": renewable_proximity_advantage,
            "clean_accessibility": clean_accessibility,
            "active_plant_count": active_count_grid,
            "avg_active_plant_size": avg_active_size,
            "active_mean_age": active_mean_age,
            "type_diversity_entropy": entropy,
            "cumulative_retired": cumulative_retired_grid,
        }

        for metric, grid in derived_grids.items():
            if year < derived_starts[metric]:
                continue
            scope = "total"
            var_name = _var_name(metric, scope)
            metric_units = units if metric in CAPACITY_UNIT_METRICS else METRIC_UNITS.get(metric, units)
            _save_yearly_grid(
                data=grid,
                metric=metric,
                scope=scope,
                year=year,
                final_dir=final_dir,
                source_name=source_name,
                units=metric_units,
                excluded_status_substrings=excluded_status_substrings,
                overwrite=overwrite,
            )
            totals_by_var.setdefault(var_name, []).append(_summarize_grid(metric, grid))
            years_by_var.setdefault(var_name, []).append(year)
            series_units[var_name] = SERIES_UNITS.get(metric, "")

            if plot and ("total" in plot_scopes) and (year in plot_years):
                if not _has_plottable_signal(metric, grid):
                    continue
                map_out = plots_dir / "maps" / var_name / f"{year}.png"
                _plot_map(
                    data=grid,
                    metric=metric,
                    scope="total",
                    year=year,
                    units=metric_units,
                    out_path=map_out,
                )

        gc.collect()
        plt.close("all")

    if plot:
        for scope in plot_scopes:
            for metric in METRICS:
                var_name = _var_name(metric, scope)
                ts_out = plots_dir / "timeseries" / f"{var_name}.png"
                _plot_timeseries(
                    var_name=var_name,
                    years=years_by_var.get(var_name, []),
                    totals=totals_by_var.get(var_name, []),
                    units=series_units.get(var_name, ""),
                    out_path=ts_out,
                )

    click.echo(f"\nPower pipeline complete. Outputs in {final_dir}")


if __name__ == "__main__":
    main()
