"""Extract and regrid LUH3 states.nc to individual yearly NetCDF files.

Reads LUH3 states.nc from input4MIPs, interpolates each variable/year to the
master 0.25 degree grid (721x1440, 0..360 lon), and writes CF-1.8 yearly files.

Usage:
    python -m src.processing.luh3_states_to_yearly --all
    python -m src.processing.luh3_states_to_yearly -v primf -v urban --start-year 2000 --end-year 2024
"""

from __future__ import annotations

import re
from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.utils import get_logger
from src.year_policy import resolve_year_bounds

logger = get_logger("processing.luh3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "luh3.yml"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "luh3" / "states.nc"
STATIC_PATH = PROJECT_ROOT / "data" / "raw" / "luh3" / "static.nc"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "land_use" / "states"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(0, 360, N_LON, endpoint=False)

FRACTION_VARS = {
    "primf", "primn", "secdf", "secdn", "pastr", "range", "urban",
    "c3ann", "c3per", "c4ann", "c4per", "c3nfx",
}


def load_luh3_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _parse_base_year(units: str) -> int | None:
    match = re.search(r"since\s+(\d{4})", units)
    if match:
        return int(match.group(1))
    return None


def _numeric_time_to_years(time_vals: np.ndarray, units: str, calendar: str | None) -> list[int]:
    """Convert numeric CF-style time values to calendar years."""
    # Preferred path: decode via cftime for correct calendar handling.
    if units and "since" in units:
        try:
            import cftime  # type: ignore

            dates = cftime.num2date(time_vals, units=units, calendar=calendar or "standard")
            return [int(getattr(d, "year")) for d in dates]
        except Exception:
            pass

    # Fallback: coarse conversion from offset units to years.
    base_year = _parse_base_year(units)
    unit_l = (units or "").lower()
    if "day" in unit_l:
        scale = 365.0
    elif "month" in unit_l:
        scale = 12.0
    else:
        scale = 1.0

    if base_year is not None:
        return [base_year + int(np.floor(float(v) / scale + 1e-9)) for v in time_vals]
    return [int(round(float(v))) for v in time_vals]


def build_year_to_index(ds: xr.Dataset) -> dict[int, int]:
    """Build year -> time-index mapping from LUH3 time coordinate."""
    time_coord = ds["time"]
    time_vals = np.asarray(time_coord.values)

    if np.issubdtype(time_vals.dtype, np.number):
        units = str(time_coord.attrs.get("units", ""))
        calendar = time_coord.attrs.get("calendar")
        years = _numeric_time_to_years(time_vals, units, str(calendar) if calendar else None)
    else:
        years = []
        for value in time_vals:
            if hasattr(value, "year"):
                years.append(int(value.year))
            else:
                years.append(int(str(value)[:4]))

    year_to_idx: dict[int, int] = {}
    for i, year in enumerate(years):
        year_to_idx.setdefault(year, i)
    return year_to_idx


def _standardize_longitude(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    if "lon" not in obj.coords:
        return obj

    if float(obj.lon.min()) < 0:
        obj = obj.assign_coords(lon=(obj.lon % 360.0))
    obj = obj.sortby("lon")

    lon_vals = np.asarray(obj["lon"].values, dtype=float)
    _, unique_idx = np.unique(lon_vals, return_index=True)
    return obj.isel(lon=np.sort(unique_idx))


def _pad_periodic_longitude(obj: xr.DataArray | xr.Dataset, pad_cells: int = 2) -> xr.DataArray | xr.Dataset:
    obj = _standardize_longitude(obj)
    if "lon" not in obj.coords or obj.sizes.get("lon", 0) < 4:
        return obj

    left = obj.isel(lon=slice(-pad_cells, None)).copy()
    right = obj.isel(lon=slice(0, pad_cells)).copy()
    left = left.assign_coords(lon=left.lon - 360.0)
    right = right.assign_coords(lon=right.lon + 360.0)
    return xr.concat([left, obj, right], dim="lon")


def _interp_to_target(obj: xr.DataArray | xr.Dataset) -> xr.DataArray | xr.Dataset:
    return _pad_periodic_longitude(obj, pad_cells=2).interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )


def load_land_budget(static_path: Path | None = None) -> xr.DataArray | None:
    src_path = Path(static_path) if static_path else STATIC_PATH
    if not src_path.exists():
        logger.warning("LUH3 static file not found; states will only cap overshoot, not renormalize to land budget.")
        return None

    ds_static = xr.open_dataset(src_path)
    if "icwtr" not in ds_static.data_vars:
        ds_static.close()
        logger.warning("LUH3 static file lacks icwtr; states will only cap overshoot, not renormalize to land budget.")
        return None

    da = ds_static["icwtr"].load()
    ds_static.close()
    da = _interp_to_target(da).clip(0, 1)
    return (1.0 - da).clip(0, 1).rename("land_budget")


def _write_output(var_name: str, var_info: dict, data: np.ndarray, year: int, source: str) -> Path:
    out_path = FINAL_DIR / var_name / f"{year:04d}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    out_ds = xr.Dataset(
        {
            var_name: (
                ["lat", "lon"],
                data.astype(np.float32),
                {"units": var_info["units"], "long_name": var_info["long_name"]},
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor LUH3 {var_name}",
            "source": source,
            "year": year,
        },
    )
    out_ds.to_netcdf(
        out_path,
        encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    return out_path


def process_fraction_year(
    ds: xr.Dataset,
    requested_vars: dict[str, dict],
    year_idx: int,
    year: int,
    overwrite: bool = False,
    land_budget: xr.DataArray | None = None,
) -> dict[str, Path | None]:
    out_paths = {name: FINAL_DIR / name / f"{year:04d}.nc" for name in requested_vars}
    if not overwrite and all(path.exists() for path in out_paths.values()):
        return {name: None for name in requested_vars}

    stack_vars = [name for name in sorted(FRACTION_VARS) if name in ds.data_vars]
    stack = xr.concat(
        [ds[name].isel(time=year_idx).load() for name in stack_vars],
        dim=xr.IndexVariable("component", stack_vars),
    )
    stack = _interp_to_target(stack).clip(min=0)

    total = stack.sum(dim="component")
    if land_budget is not None:
        budget = land_budget.broadcast_like(total)
        scale = xr.where(total > 0, budget / total, 0.0)
        source = "input4MIPs LUH3 v3.1.1 (UofMD); periodic-lon regrid; states normalized to 1-icwtr"
    else:
        scale = xr.where(total > 1.0, 1.0 / total, 1.0)
        source = "input4MIPs LUH3 v3.1.1 (UofMD); periodic-lon regrid; states capped to avoid overshoot"
    stack = (stack * scale).clip(min=0, max=1)

    written: dict[str, Path | None] = {}
    for name, info in requested_vars.items():
        path = out_paths[name]
        if path.exists() and not overwrite:
            written[name] = None
            continue
        data = stack.sel(component=name).values
        written[name] = _write_output(name, info, data, year, source=source)
    return written


def process_nonfraction_year(
    ds: xr.Dataset,
    var_name: str,
    var_info: dict,
    year_idx: int,
    year: int,
    overwrite: bool = False,
) -> Path | None:
    out_path = FINAL_DIR / var_name / f"{year:04d}.nc"
    if out_path.exists() and not overwrite:
        return None

    da = ds[var_name].isel(time=year_idx).load()
    da = _interp_to_target(da).clip(min=0)
    return _write_output(
        var_name,
        var_info,
        da.values,
        year,
        source="input4MIPs LUH3 v3.1.1 (UofMD); periodic-lon regrid",
    )


def process_variable_year(
    ds: xr.Dataset,
    var_name: str,
    var_info: dict,
    year_idx: int,
    year: int,
    overwrite: bool = False,
    land_budget: xr.DataArray | None = None,
) -> Path | None:
    if var_name in FRACTION_VARS:
        return process_fraction_year(
            ds,
            requested_vars={var_name: var_info},
            year_idx=year_idx,
            year=year,
            overwrite=overwrite,
            land_budget=land_budget,
        )[var_name]
    return process_nonfraction_year(ds, var_name, var_info, year_idx, year, overwrite)


@click.command()
@click.option("--variables", "-v", multiple=True, help="Variable names (e.g. primf urban).")
@click.option("--all", "run_all", is_flag=True, help="Process all configured variables.")
@click.option("--start-year", type=int, default=None, help="Start year (default: from config).")
@click.option("--end-year", type=int, default=None, help="End year (default: from config).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
@click.option("--raw-path", type=click.Path(exists=True), default=None,
              help=f"Path to LUH3 states.nc (default: {RAW_PATH})")
def main(variables, run_all, start_year, end_year, overwrite, raw_path):
    """Extract and regrid LUH3 states.nc to yearly files."""
    config = load_luh3_config()
    all_vars = config["variables"]
    t_range = config["temporal_range"]

    if variables:
        var_list = {v: all_vars[v] for v in variables if v in all_vars}
        missing = [v for v in variables if v not in all_vars]
        if missing:
            logger.warning("Unknown variables (skipped): %s", missing)
    elif run_all:
        var_list = all_vars
    else:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    y_start, y_end = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=t_range[0],
        default_end=t_range[1],
        label="LUH3 processing years",
    )

    src_path = Path(raw_path) if raw_path else RAW_PATH
    if not src_path.exists():
        click.echo(f"Source file not found: {src_path}")
        click.echo("Run: python -m src.download.luh3 --all --source-dir <download_dir>")
        return

    click.echo(f"Opening {src_path} ...")
    ds = xr.open_dataset(src_path, decode_times=False, chunks={"time": 1})

    year_to_idx = build_year_to_index(ds)
    years = [y for y in range(y_start, y_end + 1) if y in year_to_idx]
    if not years:
        click.echo(f"No matching years in range {y_start}-{y_end}")
        ds.close()
        return

    click.echo(f"Processing {len(var_list)} variables x {len(years)} years ({years[0]}-{years[-1]})")

    fraction_vars = {v: info for v, info in var_list.items() if v in FRACTION_VARS}
    scalar_vars = {v: info for v, info in var_list.items() if v not in FRACTION_VARS}
    counts = {v: {"processed": 0, "skipped": 0} for v in var_list}

    land_budget = load_land_budget() if fraction_vars else None

    if fraction_vars:
        for missing_var in [v for v in fraction_vars if v not in ds.data_vars]:
            logger.warning("Fraction variable %s not found in dataset, skipping", missing_var)
            counts.pop(missing_var, None)
        fraction_vars = {v: info for v, info in fraction_vars.items() if v in ds.data_vars}

        for year in tqdm(years, desc="fraction_states"):
            idx = year_to_idx[year]
            results = process_fraction_year(ds, fraction_vars, idx, year, overwrite, land_budget=land_budget)
            for var_name, result in results.items():
                if result:
                    counts[var_name]["processed"] += 1
                else:
                    counts[var_name]["skipped"] += 1

    for var_name, var_info in scalar_vars.items():
        if var_name not in ds.data_vars:
            logger.warning("Variable %s not found in dataset, skipping", var_name)
            continue

        for year in tqdm(years, desc=var_name):
            idx = year_to_idx[year]
            result = process_nonfraction_year(ds, var_name, var_info, idx, year, overwrite)
            if result:
                counts[var_name]["processed"] += 1
            else:
                counts[var_name]["skipped"] += 1

    for var_name in var_list:
        if var_name in counts:
            click.echo(f"  {var_name}: {counts[var_name]['processed']} new, {counts[var_name]['skipped']} skipped")

    ds.close()
    click.echo(f"Done. Output in {FINAL_DIR}")


if __name__ == "__main__":
    main()
