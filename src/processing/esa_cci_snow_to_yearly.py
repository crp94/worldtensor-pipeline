"""Aggregate ESA Snow CCI daily SWE files to yearly 0.25 degree grids.

Outputs yearly statistics per variable:
- mean
- std
- min
- max

Also supports linear interpolation for missing years between valid anchor years.
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from src.grid import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, N_LAT, N_LON, make_template
from src.utils import get_logger
from src.year_policy import resolve_year_list

logger = get_logger("processing.esa_cci_snow")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "esa_cci_snow.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "esa_cci_snow"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _normalize_da_coords(da: xr.DataArray) -> xr.DataArray:
    rename: dict[str, str] = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if rename:
        da = da.rename(rename)

    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Expected lat/lon dimensions, got {da.dims}")

    if da.lat.values[0] > da.lat.values[-1]:
        da = da.isel(lat=slice(None, None, -1))

    if float(np.nanmin(da.lon.values)) < 0:
        da = da.assign_coords(lon=(da.lon.values % 360.0))
    da = da.sortby("lon")

    # Drop duplicate coordinates if present.
    _, lat_idx = np.unique(da.lat.values, return_index=True)
    _, lon_idx = np.unique(da.lon.values, return_index=True)
    da = da.isel(lat=np.sort(lat_idx), lon=np.sort(lon_idx))
    return da


def _extend_periodic_lon(da: xr.DataArray, pad: int = 2) -> xr.DataArray:
    left = da.isel(lon=slice(-pad, None)).copy()
    right = da.isel(lon=slice(0, pad)).copy()
    left = left.assign_coords(lon=(left.lon.values - 360.0))
    right = right.assign_coords(lon=(right.lon.values + 360.0))
    return xr.concat([left, da, right], dim="lon").sortby("lon")


def _to_target_grid(da: xr.DataArray) -> xr.DataArray:
    da_ext = _extend_periodic_lon(da, pad=2)
    out = da_ext.interp(lon=TARGET_LON, method="linear")
    out = out.interp(lat=TARGET_LAT, method="linear", kwargs={"fill_value": "extrapolate"})
    return out


def _collect_year_files(raw_dir: Path, year: int) -> list[Path]:
    year_dir = raw_dir / str(year)
    if not year_dir.exists():
        return []

    nested = sorted(year_dir.glob("*/*.nc"))
    if nested:
        return nested
    return sorted(year_dir.glob("*.nc"))


def _prepare_daily_array(ds: xr.Dataset, source_var: str) -> xr.DataArray | None:
    if source_var not in ds.data_vars:
        return None
    da = ds[source_var]

    if "time" in da.dims:
        da = da.isel(time=0, drop=True)

    squeeze_dims = [d for d in da.dims if d not in ("lat", "lon", "latitude", "longitude") and da.sizes.get(d, 0) == 1]
    if squeeze_dims:
        da = da.isel({d: 0 for d in squeeze_dims}, drop=True)

    da = _normalize_da_coords(da)
    # Product flag values are negative for non-valid categories.
    da = da.where(da >= 0.0)
    return da.astype(np.float32)


def _aggregate_year_source_var(year_files: list[Path], source_var: str) -> dict[str, xr.DataArray]:
    acc_sum: np.ndarray | None = None
    acc_sumsq: np.ndarray | None = None
    acc_count: np.ndarray | None = None
    acc_min: np.ndarray | None = None
    acc_max: np.ndarray | None = None
    lat_vals: np.ndarray | None = None
    lon_vals: np.ndarray | None = None

    for path in year_files:
        try:
            with xr.open_dataset(path, engine="netcdf4") as ds:
                da = _prepare_daily_array(ds, source_var)
                if da is None:
                    continue
                arr = da.values.astype(np.float32, copy=False)
                finite = np.isfinite(arr)
                if not np.any(finite):
                    continue

                if acc_sum is None:
                    shape = arr.shape
                    acc_sum = np.zeros(shape, dtype=np.float64)
                    acc_sumsq = np.zeros(shape, dtype=np.float64)
                    acc_count = np.zeros(shape, dtype=np.uint16)
                    acc_min = np.full(shape, np.inf, dtype=np.float32)
                    acc_max = np.full(shape, -np.inf, dtype=np.float32)
                    lat_vals = da["lat"].values
                    lon_vals = da["lon"].values

                valid = np.where(finite, arr, 0.0)
                acc_sum += valid
                acc_sumsq += valid * valid
                acc_count += finite.astype(np.uint16)
                acc_min = np.minimum(acc_min, np.where(finite, arr, np.inf))
                acc_max = np.maximum(acc_max, np.where(finite, arr, -np.inf))
        except Exception as e:
            logger.warning("Skipping file %s (%s)", path, e)
            continue

    if acc_sum is None or lat_vals is None or lon_vals is None or acc_count is None:
        return {}

    count = acc_count.astype(np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        mean = np.where(count > 0, acc_sum / count, np.nan)
        var = np.where(count > 0, (acc_sumsq / count) - (mean * mean), np.nan)
    var = np.where(np.isfinite(var), np.maximum(var, 0.0), np.nan)
    std = np.sqrt(var, dtype=np.float64)
    min_arr = np.where(np.isfinite(acc_min), acc_min, np.nan)
    max_arr = np.where(np.isfinite(acc_max), acc_max, np.nan)

    coords = {"lat": lat_vals, "lon": lon_vals}
    return {
        "mean": xr.DataArray(mean.astype(np.float32), dims=("lat", "lon"), coords=coords),
        "std": xr.DataArray(std.astype(np.float32), dims=("lat", "lon"), coords=coords),
        "min": xr.DataArray(min_arr.astype(np.float32), dims=("lat", "lon"), coords=coords),
        "max": xr.DataArray(max_arr.astype(np.float32), dims=("lat", "lon"), coords=coords),
    }


def _save_year_grid(
    grid: np.ndarray,
    year: int,
    var_name: str,
    long_name: str,
    units: str,
    final_root: Path,
    domain: str,
    source: str,
    overwrite: bool,
    interpolation_note: str | None = None,
) -> Path | None:
    out_path = final_root / domain / var_name / f"{year}.nc"
    if out_path.exists() and not overwrite:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = make_template(year)
    attrs = {"units": units, "long_name": long_name}
    if interpolation_note:
        attrs["interpolation"] = interpolation_note
    ds[var_name] = (
        ["time", "lat", "lon"],
        grid.astype(np.float32, copy=False)[np.newaxis, :, :],
        attrs,
    )
    ds.attrs.update(
        {
            "title": f"WorldTensor {var_name}",
            "source": source,
            "aggregation_method": "yearly statistic from daily SWE",
            "year": int(year),
        }
    )
    ds.to_netcdf(out_path, encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}})
    ds.close()
    return out_path


def _interpolate_missing_years(
    grids_by_year: dict[int, np.ndarray],
    years: list[int],
) -> dict[int, tuple[np.ndarray, tuple[int, int]]]:
    out: dict[int, tuple[np.ndarray, tuple[int, int]]] = {}
    existing = sorted(grids_by_year.keys())
    if len(existing) < 2:
        return out

    for year in years:
        if year in grids_by_year:
            continue
        prev = [y for y in existing if y < year]
        nxt = [y for y in existing if y > year]
        if not prev or not nxt:
            continue
        y0 = max(prev)
        y1 = min(nxt)
        g0 = grids_by_year[y0]
        g1 = grids_by_year[y1]
        w = (year - y0) / float(y1 - y0)
        interp = np.where(
            np.isfinite(g0) & np.isfinite(g1),
            g0 + (g1 - g0) * w,
            np.nan,
        ).astype(np.float32)
        out[year] = (interp, (y0, y1))
    return out


def process_esa_cci_snow(
    years: list[int],
    variable_keys: list[str],
    raw_dir: Path = RAW_DIR,
    overwrite: bool = False,
    interpolate_missing: bool = True,
) -> dict[str, int]:
    cfg = load_config()
    domain = str(cfg["output"]["domain"])
    final_root = PROJECT_ROOT / str(cfg["output"]["final_root"])
    stats = list(cfg.get("stats", ["mean", "std", "min", "max"]))
    variables_cfg = cfg["variables"]

    source_vars = []
    for key in variable_keys:
        if key not in variables_cfg:
            continue
        source_vars.append(variables_cfg[key]["source_variable"])
    source_vars = sorted(set(source_vars))

    # Store computed target-grid arrays first, then write and interpolate.
    computed: dict[str, dict[int, np.ndarray]] = {}
    meta: dict[str, tuple[str, str]] = {}

    for year in tqdm(years, desc="ESA Snow yearly aggregation"):
        year_files = _collect_year_files(raw_dir, year)
        if not year_files:
            continue

        agg_by_source: dict[str, dict[str, xr.DataArray]] = {}
        for src_var in source_vars:
            agg_by_source[src_var] = _aggregate_year_source_var(year_files, src_var)

        for key in variable_keys:
            if key not in variables_cfg:
                continue
            info = variables_cfg[key]
            src_var = info["source_variable"]
            stat_map = agg_by_source.get(src_var, {})
            if not stat_map:
                continue

            for stat in stats:
                if stat not in stat_map:
                    continue
                output_name = f"{info['output_prefix']}_{stat}"
                regridded = _to_target_grid(stat_map[stat])
                computed.setdefault(output_name, {})[year] = regridded.values.astype(np.float32)
                meta[output_name] = (
                    f"{info['long_name']} ({stat})",
                    str(info["units"]),
                )

    files_written = 0
    interpolated_written = 0
    source_name = "ESA Snow CCI v4.0 (CEDA)"

    for output_name, grids in computed.items():
        long_name, units = meta[output_name]

        for year, grid in sorted(grids.items()):
            p = _save_year_grid(
                grid=grid,
                year=year,
                var_name=output_name,
                long_name=long_name,
                units=units,
                final_root=final_root,
                domain=domain,
                source=source_name,
                overwrite=overwrite,
                interpolation_note=None,
            )
            if p is not None:
                files_written += 1

        if interpolate_missing:
            interp = _interpolate_missing_years(grids, years)
            for year, (grid, (y0, y1)) in sorted(interp.items()):
                p = _save_year_grid(
                    grid=grid,
                    year=year,
                    var_name=output_name,
                    long_name=long_name,
                    units=units,
                    final_root=final_root,
                    domain=domain,
                    source=source_name,
                    overwrite=overwrite,
                    interpolation_note=f"linear interpolation between {y0} and {y1}",
                )
                if p is not None:
                    files_written += 1
                    interpolated_written += 1

    return {
        "years_requested": len(years),
        "variables_requested": len(variable_keys),
        "files_written": files_written,
        "interpolated_files_written": interpolated_written,
    }


@click.command()
@click.option("--variables", "-v", multiple=True, help="Configured variable keys (swe, swe_std).")
@click.option("--all", "run_all", is_flag=True, help="Process all configured variables.")
@click.option("--years", "-y", multiple=True, type=int, help="Specific years.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing yearly outputs.")
@click.option("--interpolate-missing/--no-interpolate-missing", default=True, show_default=True)
def main(variables, run_all, years, start_year, end_year, overwrite, interpolate_missing):
    """Aggregate ESA Snow daily files to yearly 0.25 degree grids."""
    cfg = load_config()
    y0, y1 = [int(v) for v in cfg["temporal_range"]]

    if variables:
        var_list = [v for v in variables if v in cfg["variables"]]
    elif run_all:
        var_list = list(cfg["variables"].keys())
    else:
        click.echo("Specify --variables or --all")
        return

    year_list = resolve_year_list(
        years,
        start_year=start_year,
        end_year=end_year,
        default_start=y0,
        default_end=y1,
        label="ESA CCI Snow processing years",
    )

    if not year_list:
        click.echo("No valid years selected.")
        return

    summary = process_esa_cci_snow(
        years=year_list,
        variable_keys=var_list,
        raw_dir=RAW_DIR,
        overwrite=overwrite,
        interpolate_missing=interpolate_missing,
    )
    click.echo(
        "Done. "
        f"files_written={summary['files_written']}, "
        f"interpolated={summary['interpolated_files_written']}"
    )


if __name__ == "__main__":
    main()
