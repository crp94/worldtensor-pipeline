"""Aggregate ERA5 monthly data to yearly, producing derived maps.

For each variable and year, produces 4 output files under data/final/climate/:
    {var}_{mean|sum}/{year}.nc  — primary aggregate (mean or sum)
    {var}_std/{year}.nc         — standard deviation across months
    {var}_max/{year}.nc         — maximum monthly value
    {var}_min/{year}.nc         — minimum monthly value

Excluded variables (sentinel-value contamination)
-------------------------------------------------
The following ERA5 variables are excluded from processing because their
yearly _max, _min, _std, and/or _sum aggregations contain unmasked
fill/sentinel values (32767, 9999, -999, -9999, -32768), predominantly
in early reanalysis years (1940s–1960s):

    bld   — boundary layer dissipation
    gwd   — gravity wave dissipation
    ewss  — eastward turbulent surface stress
    nsss  — northward turbulent surface stress
    lgws  — eastward gravity wave surface stress
    mgws  — northward gravity wave surface stress
    lspf  — large-scale precipitation fraction
    slhf  — surface latent heat flux
    sshf  — surface sensible heat flux
    rsn   — snow density (max < mean inconsistency at 761 k pixels)

These are listed in the ``excluded_variables`` key in ``config/era5.yml``
and are automatically filtered out by ``load_era5_config()``.

Usage:
    python -m src.processing.era5_monthly_to_yearly --variables t2m tp
    python -m src.processing.era5_monthly_to_yearly --all
"""

import gc
from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from src.utils import get_logger
from src.year_policy import resolve_year_list

logger = get_logger("processing.era5")

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config" / "era5.yml"
RAW_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "raw" / "era5"
FINAL_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "final" / "climate"


def load_era5_config() -> dict:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    excluded = {str(v) for v in config.get("excluded_variables", [])}
    if excluded:
        config["variables"] = {k: v for k, v in config.get("variables", {}).items() if k not in excluded}
    return config


def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Ensure lat is ascending (-90→90) and lon is 0→360."""
    # Rename latitude/longitude if needed
    rename = {}
    if "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename["longitude"] = "lon"
    if "valid_time" in ds.dims:
        rename["valid_time"] = "time"
    if rename:
        ds = ds.rename(rename)

    # Flip lat to ascending if needed
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))

    # Convert lon from -180..180 to 0..360 if needed
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon.values % 360))
        ds = ds.sortby("lon")

    return ds


def _find_data_var(ds: xr.Dataset, short_name: str) -> str:
    """Find the actual data variable name in the dataset.

    ERA5 NetCDFs may use the short name directly or a different internal name.
    """
    if short_name in ds.data_vars:
        return short_name

    # Try common ERA5 naming patterns
    data_vars = [v for v in ds.data_vars if v not in ("time_bnds", "expver")]
    if len(data_vars) == 1:
        return data_vars[0]

    # Last resort: return first data var
    if data_vars:
        logger.warning("Could not match '%s', using '%s'", short_name, data_vars[0])
        return data_vars[0]

    raise ValueError(f"No data variables found in dataset for '{short_name}'")


def _save_derived(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    var_name: str,
    stat_name: str,
    long_name: str,
    units: str,
    year: int,
    output_dir: Path,
    overwrite: bool = False,
) -> Path | None:
    """Save a single derived 2D map as a CF-compliant NetCDF."""
    folder_name = f"{var_name}_{stat_name}"
    out_path = output_dir / folder_name / f"{year}.nc"

    if out_path.exists() and not overwrite:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            folder_name: (
                ["lat", "lon"],
                data.astype(np.float32),
                {
                    "units": units,
                    "long_name": f"{long_name} ({stat_name})",
                },
            )
        },
        coords={
            "lat": ("lat", lat, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", lon, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor ERA5 {folder_name}",
            "source": "ERA5 reanalysis (ECMWF)",
            "aggregation_method": stat_name,
            "year": year,
        },
    )

    encoding = {
        folder_name: {"zlib": True, "complevel": 4, "dtype": "float32"},
    }
    ds.to_netcdf(out_path, encoding=encoding)
    return out_path


def process_variable_year(
    short_name: str,
    var_info: dict,
    year: int,
    raw_dir: Path,
    output_dir: Path,
    overwrite: bool = False,
) -> int:
    """Process one variable for one year. Returns number of files written."""
    raw_path = raw_dir / short_name / f"{year}.nc"
    if not raw_path.exists():
        return 0

    aggregation = var_info["aggregation"]
    long_name = var_info["long_name"]
    units = var_info["units"]

    # Map aggregation type to stat name for the primary output
    primary_stat = aggregation  # mean, sum, max, or min

    # Check if all outputs already exist
    stats_to_compute = [primary_stat, "std", "max", "min"]
    if not overwrite:
        all_exist = all(
            (output_dir / f"{short_name}_{s}" / f"{year}.nc").exists()
            for s in stats_to_compute
        )
        if all_exist:
            return 0

    try:
        ds = xr.open_dataset(raw_path)
        ds = _normalize_coords(ds)
        data_var = _find_data_var(ds, short_name)
        da = ds[data_var]

        # Ensure time dimension exists for aggregation
        time_dim = "time" if "time" in da.dims else da.dims[0]

        lat = ds.lat.values
        lon = ds.lon.values

        # Compute all derived maps
        derived = {}
        if primary_stat == "mean":
            derived["mean"] = da.mean(dim=time_dim).values
        elif primary_stat == "sum":
            derived["sum"] = da.sum(dim=time_dim).values
        elif primary_stat == "max":
            derived["max"] = da.max(dim=time_dim).values
        elif primary_stat == "min":
            derived["min"] = da.min(dim=time_dim).values

        derived["std"] = da.std(dim=time_dim).values

        # Only compute max/min if not already the primary stat
        if primary_stat != "max":
            derived["max"] = da.max(dim=time_dim).values
        if primary_stat != "min":
            derived["min"] = da.min(dim=time_dim).values

        ds.close()

        # Save each derived map
        written = 0
        for stat_name, data in derived.items():
            result = _save_derived(
                data=data,
                lat=lat,
                lon=lon,
                var_name=short_name,
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
        logger.error("Failed %s/%d: %s", short_name, year, e)
        return 0


@click.command()
@click.option("--variables", "-v", multiple=True, help="Short names to process (e.g. t2m tp).")
@click.option("--all", "process_all", is_flag=True, help="Process all variables in config.")
@click.option("--years", "-y", multiple=True, type=int, help="Specific years.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--raw-dir", type=click.Path(), default=None, help="Override raw data directory.")
@click.option("--output-dir", type=click.Path(), default=None, help="Override output directory.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
def main(variables, process_all, years, start_year, end_year, raw_dir, output_dir, overwrite):
    """Aggregate ERA5 monthly data to yearly derived maps."""
    config = load_era5_config()
    all_vars = config["variables"]
    t_range = config["temporal_range"]

    # Resolve variables
    if variables:
        var_list = {v: all_vars[v] for v in variables if v in all_vars}
        missing = [v for v in variables if v not in all_vars]
        if missing:
            logger.warning("Unknown variables (skipped): %s", missing)
    elif process_all:
        var_list = all_vars
    else:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    # Resolve years
    year_list = resolve_year_list(
        years,
        start_year=start_year,
        end_year=end_year,
        default_start=t_range[0],
        default_end=t_range[1],
        label="ERA5 processing years",
    )
    if not year_list:
        click.echo("No valid years selected.")
        return

    r_dir = Path(raw_dir) if raw_dir else RAW_DIR
    o_dir = Path(output_dir) if output_dir else FINAL_DIR

    # Discover which raw files actually exist
    tasks = []
    for short_name, info in var_list.items():
        for year in year_list:
            raw_path = r_dir / short_name / f"{year}.nc"
            if raw_path.exists():
                tasks.append((short_name, info, year))

    click.echo(f"Variables: {len(var_list)}, Raw files found: {len(tasks)}")

    if not tasks:
        click.echo("No raw files found. Run the download script first.")
        return

    total_written = 0
    current_var = None

    with tqdm(total=len(tasks), desc="Processing ERA5") as pbar:
        for short_name, info, year in tasks:
            # GC between variables (per notebook pattern)
            if short_name != current_var:
                if current_var is not None:
                    gc.collect()
                current_var = short_name

            written = process_variable_year(
                short_name=short_name,
                var_info=info,
                year=year,
                raw_dir=r_dir,
                output_dir=o_dir,
                overwrite=overwrite,
            )
            total_written += written
            pbar.update(1)
            pbar.set_postfix(var=short_name, written=total_written)

    click.echo(f"\nDone. Files written: {total_written}")


if __name__ == "__main__":
    main()
