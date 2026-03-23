"""Aggregate climate extremes monthly files to yearly mean maps.

For each variable/accumulation and year, produces 1 output file under
`data/final/extremes/`:
    {combo}_mean/{year}.nc

Where combo = {variable}_{accumulation:02d}, e.g. spi_12.

Usage:
    python -m src.processing.climate_extremes_monthly_to_yearly --all
    python -m src.processing.climate_extremes_monthly_to_yearly -v spi -a 12 --start-year 2000 --end-year 2020
"""

from __future__ import annotations

from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.utils import get_logger
from src.year_policy import resolve_year_list

logger = get_logger("processing.climate_extremes")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "climate_extremes.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "climate_extremes"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "extremes"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)
STATS = ("mean",)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def combo_name(short_name: str, accumulation: int) -> str:
    return f"{short_name}_{accumulation:02d}"


def raw_path_for(raw_dir: Path, short_name: str, accumulation: int, year: int) -> Path:
    return raw_dir / combo_name(short_name, accumulation) / f"{year}.nc"


def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename["longitude"] = "lon"
    if "valid_time" in ds.dims:
        rename["valid_time"] = "time"
    if rename:
        ds = ds.rename(rename)

    if "lat" in ds.coords and ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))

    if "lon" in ds.coords and float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon.values % 360.0))
        ds = ds.sortby("lon")

    return ds


def _find_data_var(ds: xr.Dataset, short_name: str) -> str:
    if short_name in ds.data_vars:
        return short_name

    data_vars = [v for v in ds.data_vars if v not in ("time_bnds", "expver")]
    if data_vars:
        return data_vars[0]

    raise ValueError(f"No data variables found for {short_name}")


def _to_target_grid(da: xr.DataArray) -> xr.DataArray:
    """Interpolate to master 0.25° grid if source grid differs."""
    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError("DataArray must have lat/lon dimensions")

    same_shape = da.sizes.get("lat") == N_LAT and da.sizes.get("lon") == N_LON
    if same_shape:
        lat_ok = np.isclose(float(da.lat.min()), LAT_MIN) and np.isclose(float(da.lat.max()), LAT_MAX)
        lon_ok = np.isclose(float(da.lon.min()), LON_MIN) and np.isclose(float(da.lon.max()), LON_MAX)
        if lat_ok and lon_ok:
            return da

    return da.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )


def _save_derived(
    data: np.ndarray,
    combo: str,
    stat_name: str,
    long_name: str,
    units: str,
    year: int,
    output_dir: Path,
    overwrite: bool,
) -> Path | None:
    folder = f"{combo}_{stat_name}"
    out_path = output_dir / folder / f"{year}.nc"
    if out_path.exists() and not overwrite:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            folder: (
                ["lat", "lon"],
                data.astype(np.float32),
                {"units": units, "long_name": f"{long_name} ({stat_name})"},
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor Climate Extremes {folder}",
            "source": "CDS derived-drought-historical-monthly",
            "aggregation_method": stat_name,
            "year": year,
        },
    )
    ds.to_netcdf(out_path, encoding={folder: {"zlib": True, "complevel": 4, "dtype": "float32"}})
    return out_path


def process_combo_year(
    short_name: str,
    accumulation: int,
    var_info: dict,
    year: int,
    raw_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> int:
    combo = combo_name(short_name, accumulation)
    raw_path = raw_path_for(raw_dir, short_name, accumulation, year)
    if not raw_path.exists():
        return 0

    if not overwrite:
        all_exist = all((output_dir / f"{combo}_{s}" / f"{year}.nc").exists() for s in STATS)
        if all_exist:
            return 0

    try:
        ds = xr.open_dataset(raw_path)
        ds = _normalize_coords(ds)
        data_var = _find_data_var(ds, short_name)
        da = ds[data_var]

        time_dim = "time" if "time" in da.dims else da.dims[0]
        da = _to_target_grid(da)

        derived = {"mean": da.mean(dim=time_dim).values}

        ds.close()

        long_name = f"{var_info['long_name']} ({accumulation}-month accumulation)"
        units = var_info["units"]

        written = 0
        for stat_name, data in derived.items():
            result = _save_derived(
                data=data,
                combo=combo,
                stat_name=stat_name,
                long_name=long_name,
                units=units,
                year=year,
                output_dir=output_dir,
                overwrite=overwrite,
            )
            if result:
                written += 1

        return written

    except Exception as e:
        logger.error("Failed %s/%d: %s", combo, year, e)
        return 0


@click.command()
@click.option("--variables", "variables", "-v", multiple=True, help="Variables (spi, spei).")
@click.option(
    "--accumulation-periods",
    "accumulation_periods",
    "-a",
    multiple=True,
    type=int,
    help="Accumulation period(s) in months.",
)
@click.option("--all", "run_all", is_flag=True, help="Process all configured variables.")
@click.option("--years", "years", "-y", multiple=True, type=int, help="Specific years.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
def main(variables, accumulation_periods, run_all, years, start_year, end_year, overwrite):
    """Aggregate monthly climate extremes files to yearly maps."""
    config = load_config()

    if variables:
        var_list = [v for v in variables if v in config["variables"]]
        missing = [v for v in variables if v not in config["variables"]]
        if missing:
            logger.warning("Unknown variables skipped: %s", missing)
    elif run_all:
        var_list = list(config["variables"].keys())
    else:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    allowed_acc = {int(v) for v in config["accumulation_periods"]}
    if accumulation_periods:
        acc_list = sorted(set(int(v) for v in accumulation_periods if int(v) in allowed_acc))
        missing_acc = sorted(set(int(v) for v in accumulation_periods) - allowed_acc)
        if missing_acc:
            logger.warning("Unsupported accumulation periods skipped: %s", missing_acc)
    else:
        acc_list = sorted(allowed_acc)
    if not acc_list:
        click.echo("No valid accumulation periods selected.")
        return

    year_list = resolve_year_list(
        years,
        start_year=start_year,
        end_year=end_year,
        default_start=config["temporal_range"][0],
        default_end=config["temporal_range"][1],
        label="climate extremes processing years",
    )
    if not year_list:
        click.echo("No valid years selected.")
        return

    tasks = []
    for short_name in var_list:
        info = config["variables"][short_name]
        for accumulation in acc_list:
            for year in year_list:
                raw_path = raw_path_for(RAW_DIR, short_name, accumulation, year)
                if raw_path.exists():
                    tasks.append((short_name, accumulation, info, year))

    click.echo(f"Tasks found: {len(tasks)}")
    if not tasks:
        click.echo("No raw files found. Run download first.")
        return

    total_written = 0
    with tqdm(total=len(tasks), desc="Processing Climate Extremes") as pbar:
        for short_name, accumulation, info, year in tasks:
            n = process_combo_year(
                short_name=short_name,
                accumulation=accumulation,
                var_info=info,
                year=year,
                raw_dir=RAW_DIR,
                output_dir=FINAL_DIR,
                overwrite=overwrite,
            )
            total_written += n
            pbar.update(1)
            pbar.set_postfix(written=total_written)

    click.echo(f"Done. Files written: {total_written}")


if __name__ == "__main__":
    main()
