"""Regrid GPWv4.11 2.5 arc-minute population data to the master 0.25° grid.

Handles both NetCDF (population count/density) and GeoTIFF (land area)
input files. Shifts longitude from -180..180 to 0..360, interpolates to
the 721x1440 master grid, clips negative values.

Anchor years (2000, 2005, 2010, 2015, 2020) are regridded from source data.
Intermediate years are filled via per-pixel linear interpolation.

Usage:
    python -m src.processing.gpw_to_yearly --all
    python -m src.processing.gpw_to_yearly --variables population_count population_density
"""

import re
from pathlib import Path

import click
import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
import yaml

from src.data_layout import output_path_for
from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.utils import get_logger, save_annual_variable, save_static_variable

logger = get_logger("processing.gpw")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "gpw.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "gpw"
FINAL_DIR = PROJECT_ROOT / "data" / "final"

# Master grid coordinates
TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)


def load_gpw_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _normalize_coords(da: xr.DataArray) -> xr.DataArray:
    """Rename latitude/longitude to lat/lon if needed."""
    rename_map = {}
    if "latitude" in da.dims:
        rename_map["latitude"] = "lat"
    if "longitude" in da.dims:
        rename_map["longitude"] = "lon"
    if rename_map:
        da = da.rename(rename_map)
    return da


def regrid(da: xr.DataArray) -> xr.DataArray:
    """Normalize coords, shift longitude to 0..360, interpolate to master grid."""
    da = _normalize_coords(da)

    lon_shifted = da.lon.values % 360
    da = da.assign_coords(lon=lon_shifted).sortby("lon")

    da = da.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    return da.clip(min=0)


def _build_dataset(
    da: xr.DataArray,
    var_name: str,
    units: str,
    long_name: str,
    year: int | None = None,
) -> xr.Dataset:
    """Build a CF-style dataset for one GPW variable."""
    # Ensure 2D (lat, lon)
    while da.ndim > 2:
        extra = [d for d in da.dims if d not in ("lat", "lon")]
        if extra:
            da = da.isel({extra[0]: 0})
        else:
            break

    attrs = {
        "Conventions": "CF-1.8",
        "title": f"WorldTensor GPW {var_name}",
        "source": "GPWv4.11 (CIESIN/SEDAC)",
    }
    if year is not None:
        attrs["year"] = int(year)

    return xr.Dataset(
        {
            var_name: (
                ["lat", "lon"],
                da.values.astype(np.float32),
                {
                    "units": units,
                    "long_name": long_name,
                    "source": "GPWv4.11 (CIESIN/SEDAC)",
                },
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs=attrs,
    )


def save_variable(
    da: xr.DataArray,
    var_name: str,
    year: int | None,
    units: str,
    long_name: str,
    overwrite: bool = False,
    is_static: bool = False,
) -> Path | None:
    """Save a regridded GPW variable through the canonical registry path."""
    out_path = output_path_for(var_name, year=year, base_dir=FINAL_DIR) if not is_static else output_path_for(var_name, base_dir=FINAL_DIR)
    if out_path.exists() and not overwrite:
        return None

    out_ds = _build_dataset(da, var_name, units, long_name, year=year if not is_static else None)
    if is_static:
        return save_static_variable(out_ds, var_name, base_dir=FINAL_DIR)
    return save_annual_variable(out_ds, var_name, int(year), base_dir=FINAL_DIR)


def extract_year_from_varname(name: str) -> int | None:
    """Extract year from variable names like 'Population Count, 2020'."""
    match = re.search(r"(\d{4})", name)
    if match:
        year = int(match.group(1))
        if 1990 <= year <= 2030:
            return year
    return None


def open_tif_as_dataarray(tif_path: Path, band: int = 1) -> xr.DataArray:
    """Open a GeoTIFF and return as xr.DataArray with lat/lon coords."""
    da = rioxarray.open_rasterio(tif_path)

    if "band" in da.dims:
        da = da.sel(band=band)

    da = da.rename({"x": "lon", "y": "lat"})

    nodata = da.rio.nodata
    if nodata is not None:
        da = da.where(da != nodata, np.nan)

    return da


def interpolate_years(
    anchor_grids: dict[int, np.ndarray],
    output_years: list[int],
    var_name: str,
    units: str,
    long_name: str,
    overwrite: bool = False,
) -> int:
    """Linearly interpolate anchor grids to fill all output years.

    Parameters
    ----------
    anchor_grids : dict[int, np.ndarray]
        Mapping of anchor year → 2D array (lat, lon).
    output_years : list[int]
        All years to produce output for.

    Returns number of new files written.
    """
    anchor_years = sorted(anchor_grids.keys())
    if len(anchor_years) < 2:
        return 0

    processed = 0
    for year in output_years:
        out_path = output_path_for(var_name, year=year, base_dir=FINAL_DIR)
        if out_path.exists() and not overwrite:
            continue

        if year in anchor_grids:
            # Anchor year — already saved during regridding
            continue

        # Find bracketing anchor years
        y_before = max(y for y in anchor_years if y <= year)
        y_after = min(y for y in anchor_years if y >= year)

        if y_before == y_after:
            continue

        # Linear interpolation weight
        t = (year - y_before) / (y_after - y_before)
        grid = (1 - t) * anchor_grids[y_before] + t * anchor_grids[y_after]
        grid = np.clip(grid, 0, None)

        # Save
        da = xr.DataArray(grid, dims=["lat", "lon"])
        result = save_variable(da, var_name, year, units, long_name, overwrite, is_static=False)
        if result:
            processed += 1
            logger.info("Interpolated %s/%d (%.0f%% between %d–%d)",
                        var_name, year, t * 100, y_before, y_after)

    return processed


def process_nc_variable(
    raw_dir: Path,
    var_name: str,
    var_info: dict,
    anchor_years: list[int],
    output_years: list[int],
    overwrite: bool = False,
) -> int:
    """Process a NetCDF-based GPW variable (population count/density).

    Regrids anchor years, then linearly interpolates intermediate years.
    Returns number of files processed.
    """
    nc_files = sorted(raw_dir.glob("*.nc"))
    if not nc_files:
        logger.warning("No NC files found for %s in %s", var_name, raw_dir)
        return 0

    units = var_info["units"]
    long_name = var_info["long_name"]
    processed = 0
    anchor_grids: dict[int, np.ndarray] = {}

    for nc_path in nc_files:
        ds = xr.open_dataset(nc_path)
        data_vars = list(ds.data_vars)

        if not data_vars:
            ds.close()
            continue

        da = ds[data_vars[0]].load()

        if "raster" in da.dims:
            for i, year in enumerate(anchor_years):
                if i >= da.sizes["raster"]:
                    break
                band_da = da.isel(raster=i)
                da_regrid = regrid(band_da)
                anchor_grids[year] = da_regrid.values.astype(np.float32)
                result = save_variable(
                    da_regrid,
                    var_name,
                    year,
                    units,
                    long_name,
                    overwrite,
                    is_static=False,
                )
                if result:
                    processed += 1
                    logger.info("Processed %s/%d (band %d)", var_name, year, i)
        else:
            year = extract_year_from_varname(data_vars[0])
            if year and year in anchor_years:
                da_regrid = regrid(da)
                anchor_grids[year] = da_regrid.values.astype(np.float32)
                result = save_variable(
                    da_regrid,
                    var_name,
                    year,
                    units,
                    long_name,
                    overwrite,
                    is_static=False,
                )
                if result:
                    processed += 1
                    logger.info("Processed %s/%d", var_name, year)

        ds.close()

    # Load any existing anchor grids that were skipped (already existed)
    for year in anchor_years:
        if year not in anchor_grids:
            existing = output_path_for(var_name, year=year, base_dir=FINAL_DIR)
            if existing.exists():
                ds = xr.open_dataset(existing)
                anchor_grids[year] = ds[var_name].values.astype(np.float32)
                ds.close()

    # Interpolate intermediate years
    processed += interpolate_years(
        anchor_grids, output_years, var_name, units, long_name, overwrite,
    )

    return processed


def process_tif_variable(
    raw_dir: Path,
    var_name: str,
    var_info: dict,
    anchor_years: list[int],
    output_years: list[int],
    overwrite: bool = False,
) -> int:
    """Process a GeoTIFF-based GPW variable (land area).

    Returns number of files processed.
    """
    tif_files = sorted(raw_dir.glob("*.tif"))
    if not tif_files:
        logger.warning("No TIF files found for %s in %s", var_name, raw_dir)
        return 0

    units = var_info["units"]
    long_name = var_info["long_name"]
    is_static = var_info.get("static", False)
    processed = 0

    for tif_path in tif_files:
        da = open_tif_as_dataarray(tif_path, band=1)
        da_regrid = regrid(da)

        if is_static:
            result = save_variable(
                da_regrid,
                var_name,
                None,
                units,
                long_name,
                overwrite,
                is_static=True,
            )
            if result:
                processed += 1
                logger.info("Processed static %s", var_name)
            break
        else:
            year = extract_year_from_varname(tif_path.stem)
            if year and year in anchor_years:
                result = save_variable(
                    da_regrid,
                    var_name,
                    year,
                    units,
                    long_name,
                    overwrite,
                    is_static=False,
                )
                if result:
                    processed += 1
                    logger.info("Processed %s/%d", var_name, year)

    return processed


@click.command()
@click.option("--variables", "-v", multiple=True,
              help="Variable(s) to process (e.g. population_count).")
@click.option("--all", "run_all", is_flag=True, help="Process all variables.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Raw data directory (default: {DEFAULT_RAW_DIR})")
def main(variables, run_all, overwrite, raw_dir):
    """Regrid GPW 2.5' population data to 0.25° master grid."""
    config = load_gpw_config()
    anchor_years = config["anchor_years"]
    output_years = config["output_years"]

    if not variables and not run_all:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    src_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    var_list = list(variables) if variables else list(config["variables"].keys())

    total_processed = 0

    for var_name in var_list:
        var_info = config["variables"].get(var_name)
        if var_info is None:
            logger.warning("Unknown variable: %s", var_name)
            continue

        var_raw_dir = src_dir / var_name
        if not var_raw_dir.exists():
            logger.warning("Raw directory not found: %s", var_raw_dir)
            continue

        click.echo(f"\n[{var_name}] {var_info['long_name']}")

        fmt = var_info.get("format", "nc")
        if fmt == "tif":
            n = process_tif_variable(
                var_raw_dir, var_name, var_info,
                anchor_years, output_years, overwrite,
            )
        else:
            n = process_nc_variable(
                var_raw_dir, var_name, var_info,
                anchor_years, output_years, overwrite,
            )

        total_processed += n
        click.echo(f"  {var_name}: {n} files processed")

    click.echo(f"\nDone. {total_processed} total files processed. Output in {FINAL_DIR}")


if __name__ == "__main__":
    main()
