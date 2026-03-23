"""Resample ETOPO 2022 to the master 0.25° grid as static NetCDFs.

Data source
-----------
Website : https://www.ncei.noaa.gov/products/etopo-global-relief-model
Citation: NOAA NCEI (2022). ETOPO 2022 15 Arc-Second Global Relief Model.

Source data is 60 arc-second (~1.8 km) bedrock elevation in WGS84.
Uses gdalwarp to resample to 0.25°, then regrids to the master 0..360 lon
grid. Produces two layers:
    - ocean_depth: positive depth below sea level (land = 0)
    - bathymetry_elevation: raw elevation (land + ocean)

Output: data/final/static/bathymetry/{variable_code}.nc

Usage:
    python -m src.processing.etopo2022_to_static --all
    python -m src.processing.etopo2022_to_static --variables ocean_depth
"""

import subprocess
import tempfile
from pathlib import Path

import click
import numpy as np
import rioxarray  # noqa: F401
import xarray as xr
import yaml

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.utils import get_logger

logger = get_logger("processing.etopo2022")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "etopo2022.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "etopo2022"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "static" / "bathymetry"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resample_to_025(src_path: Path, dst_path: Path) -> bool:
    """Resample ETOPO TIF to 0.25° WGS84 using gdalwarp."""
    cmd = [
        "gdalwarp",
        "-t_srs", "EPSG:4326",
        "-tr", "0.25", "0.25",
        "-te", "-180", "-90", "180", "90",
        "-r", "average",
        "-overwrite",
        str(src_path),
        str(dst_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("gdalwarp failed: %s", result.stderr.strip())
        return False
    return True


def open_tif_as_dataarray(tif_path: Path) -> xr.DataArray:
    """Open a GeoTIFF and return as xr.DataArray with lat/lon coords."""
    da = rioxarray.open_rasterio(tif_path)
    if "band" in da.dims:
        da = da.sel(band=1)
    da = da.rename({"x": "lon", "y": "lat"})
    nodata = da.rio.nodata
    if nodata is not None:
        da = da.where(da != nodata, np.nan)
    return da


def regrid_to_master(da: xr.DataArray) -> xr.DataArray:
    """Shift longitude to 0..360 and interpolate to the master grid."""
    lon_shifted = da.lon.values % 360
    da = da.assign_coords(lon=lon_shifted).sortby("lon")

    first = da.isel(lon=0).assign_coords(lon=da.lon.values[0] + 360.0)
    last = da.isel(lon=-1).assign_coords(lon=da.lon.values[-1] - 360.0)
    da = xr.concat([last, da, first], dim="lon")

    da = da.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="linear",
        kwargs={"fill_value": np.nan},
    )
    return da


def _save_variable(var_code: str, var_info: dict, values: np.ndarray,
                   source_label: str) -> Path:
    """Save a single variable as a CF-compliant NetCDF."""
    out_path = FINAL_DIR / f"{var_code}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            var_code: (
                ["lat", "lon"],
                values.astype(np.float32),
                {
                    "units": var_info["units"],
                    "long_name": var_info["long_name"],
                    "source": source_label,
                },
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor {var_info['long_name']}",
            "source": source_label,
            "resolution": "0.25 degree",
        },
    )
    ds.to_netcdf(
        out_path,
        encoding={var_code: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    logger.info("Saved %s → %s", var_code, out_path)
    return out_path


def process_etopo(
    variables: list[str] | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> int:
    """Process ETOPO 2022 to NetCDF. Returns count of files written."""
    config = load_config()
    all_vars = config["variables"]
    var_list = variables or list(all_vars.keys())
    source_label = "ETOPO 2022 (NOAA NCEI)"

    # Check which outputs already exist
    if not overwrite:
        needed = [v for v in var_list if not (FINAL_DIR / f"{v}.nc").exists()]
        if not needed:
            logger.info("All outputs already exist")
            return 0
    else:
        needed = var_list

    # Resample source TIF to 0.25°
    src_path = raw_dir / config["source_file"]
    if not src_path.exists():
        logger.error("Source file not found: %s", src_path)
        return 0

    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        logger.info("Resampling ETOPO 2022 → 0.25° ...")
        if not _resample_to_025(src_path, tmp_path):
            return 0

        da = open_tif_as_dataarray(tmp_path)
        da = regrid_to_master(da)
    finally:
        tmp_path.unlink(missing_ok=True)

    elevation = da.values.copy()
    processed = 0

    for var_code in needed:
        var_info = all_vars.get(var_code)
        if var_info is None:
            logger.warning("Unknown variable: %s", var_code)
            continue

        out_path = FINAL_DIR / f"{var_code}.nc"
        if out_path.exists() and not overwrite:
            logger.info("Already exists: %s", out_path.name)
            continue

        if var_code == "ocean_depth":
            # Ocean depth = negative elevation flipped to positive; land = 0
            values = np.where(elevation < 0, -elevation, 0.0)
            values = np.where(np.isnan(elevation), np.nan, values)
        elif var_code == "bathymetry_elevation":
            values = elevation.copy()
        else:
            logger.warning("Unknown variable: %s", var_code)
            continue

        _save_variable(var_code, var_info, values, source_label)
        processed += 1

    return processed


@click.command()
@click.option("--variables", "-v", multiple=True,
              help="Variable(s) to process (e.g. ocean_depth bathymetry_elevation).")
@click.option("--all", "run_all", is_flag=True, help="Process all variables.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Raw data directory (default: {DEFAULT_RAW_DIR})")
def main(variables, run_all, overwrite, raw_dir):
    """Resample ETOPO 2022 to 0.25° master grid NetCDFs."""
    if not variables and not run_all:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    src_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    var_list = list(variables) if variables else None

    n = process_etopo(variables=var_list, raw_dir=src_dir, overwrite=overwrite)
    click.echo(f"Processed {n} files. Output in {FINAL_DIR}")


if __name__ == "__main__":
    main()
