"""Extract and regrid LUH3 transitions.nc to individual yearly NetCDF files.

Usage:
    python -m src.processing.luh3_transitions_to_yearly --all
    python -m src.processing.luh3_transitions_to_yearly -v primf_to_secdn --start-year 1800 --end-year 2023
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.processing.luh3_states_to_yearly import build_year_to_index, _interp_to_target
from src.utils import get_logger
from src.year_policy import resolve_year_bounds

logger = get_logger("processing.luh3_transitions")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "luh3.yml"
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "luh3" / "transitions.nc"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "land_use" / "transitions"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(0, 360, N_LON, endpoint=False)


def load_luh3_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def list_transition_variables(ds: xr.Dataset) -> list[str]:
    vars_out: list[str] = []
    for name, da in ds.data_vars.items():
        if {"time", "lat", "lon"}.issubset(set(da.dims)):
            vars_out.append(name)
    return sorted(vars_out)


def process_transition_year(
    ds: xr.Dataset,
    var_name: str,
    year_idx: int,
    year: int,
    overwrite: bool = False,
) -> Path | None:
    out_path = FINAL_DIR / var_name / f"{year:04d}.nc"
    if out_path.exists() and not overwrite:
        return None

    da = ds[var_name].isel(time=year_idx).load()
    da = _interp_to_target(da).clip(min=0, max=1)

    src = ds[var_name]
    units = src.attrs.get("units", "1")
    long_name = src.attrs.get("long_name", var_name)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds = xr.Dataset(
        {
            var_name: (
                ["lat", "lon"],
                da.values.astype(np.float32),
                {"units": units, "long_name": long_name},
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor LUH3 transition {var_name}",
            "source": "input4MIPs LUH3 v3.1.1 (UofMD) transitions; periodic-lon regrid",
            "year": year,
        },
    )
    out_ds.to_netcdf(
        out_path,
        encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    return out_path


@click.command()
@click.option("--variables", "-v", multiple=True, help="Transition variable names.")
@click.option("--all", "run_all", is_flag=True, help="Process all transition variables.")
@click.option("--start-year", type=int, default=None, help="Start year (default: from config).")
@click.option("--end-year", type=int, default=None, help="End year (default: from config and file availability).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
@click.option("--raw-path", type=click.Path(exists=True), default=None,
              help=f"Path to LUH3 transitions.nc (default: {RAW_PATH})")
def main(variables, run_all, start_year, end_year, overwrite, raw_path):
    """Extract and regrid LUH3 transitions.nc to yearly files."""
    config = load_luh3_config()
    t_range = config["temporal_range"]
    y_start, y_end_req = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=t_range[0],
        default_end=t_range[1],
        label="LUH3 transitions processing years",
    )

    src_path = Path(raw_path) if raw_path else RAW_PATH
    if not src_path.exists():
        click.echo(f"Source file not found: {src_path}")
        click.echo("Run: python -m src.download.luh3 --all --source-dir <download_dir>")
        return

    click.echo(f"Opening {src_path} ...")
    ds = xr.open_dataset(src_path, decode_times=False, chunks={"time": 1})

    all_vars = list_transition_variables(ds)
    if variables:
        var_list = [v for v in variables if v in all_vars]
        missing = [v for v in variables if v not in all_vars]
        if missing:
            logger.warning("Unknown transition variables (skipped): %s", missing)
    elif run_all:
        var_list = all_vars
    else:
        click.echo("Specify --variables or --all. Use --help for usage.")
        ds.close()
        return

    year_to_idx = build_year_to_index(ds)
    year_max_avail = max(year_to_idx) if year_to_idx else y_end_req
    y_end = min(y_end_req, year_max_avail)
    years = [y for y in range(y_start, y_end + 1) if y in year_to_idx]
    if not years:
        click.echo(f"No matching years in range {y_start}-{y_end}")
        ds.close()
        return

    click.echo(
        f"Processing {len(var_list)} transition variables x {len(years)} years "
        f"({years[0]}-{years[-1]})"
    )

    for var_name in var_list:
        processed = 0
        skipped = 0
        for year in tqdm(years, desc=var_name):
            idx = year_to_idx[year]
            result = process_transition_year(ds, var_name, idx, year, overwrite)
            if result:
                processed += 1
            else:
                skipped += 1
        click.echo(f"  {var_name}: {processed} new, {skipped} skipped")

    ds.close()
    click.echo(f"Done. Output in {FINAL_DIR}")


if __name__ == "__main__":
    main()
