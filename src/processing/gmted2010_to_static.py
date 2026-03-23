"""Resample GMTED2010 grids to the master 0.25° grid as static NetCDFs.

Data source
-----------
Website : https://www.usgs.gov/coastal-changes-and-impacts/gmted2010
Citation: Danielson & Gesch (2011), USGS Open-File Report 2011-1073

Source data is at 30 arc-second (~1 km) in WGS84 geographic coordinates
(ESRI ArcGrid for elevation/stddev, GeoTIFF for derived slope).
Uses gdalwarp to resample to 0.25°, then regrids to the master 0..360 lon
grid with wrap-padding at the 0°/360° seam.

Output: data/final/static/topography/{variable_code}.nc

Usage:
    python -m src.processing.gmted2010_to_static --all
    python -m src.processing.gmted2010_to_static --variables elevation_mean slope_mean
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

logger = get_logger("processing.gmted2010")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "gmted2010.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "gmted2010"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "static" / "topography"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)


def load_gmted_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _find_raw_path(var_code: str, var_info: dict, raw_dir: Path) -> Path | None:
    """Resolve the raw data path for a variable."""
    if "grid_file" in var_info:
        # ArcGrid directory
        grid_path = raw_dir / var_info["grid_file"]
        if grid_path.exists():
            return grid_path
    elif "derived_from" in var_info:
        # Derived slope TIF
        slope_path = raw_dir / "slope_from_mn30.tif"
        if slope_path.exists():
            return slope_path
    return None


def _resample_to_025(src_path: Path, dst_path: Path,
                     resample_method: str = "average") -> bool:
    """Resample a raster to 0.25° WGS84 using gdalwarp."""
    cmd = [
        "gdalwarp",
        "-t_srs", "EPSG:4326",
        "-tr", "0.25", "0.25",
        "-te", "-180", "-90", "180", "90",
        "-r", resample_method,
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
    """Shift longitude to 0..360 and interpolate to the master grid.

    Pads the longitude axis so interpolation wraps correctly across
    the 0°/360° seam.
    """
    lon_shifted = da.lon.values % 360
    da = da.assign_coords(lon=lon_shifted).sortby("lon")

    # Wrap padding for 0°/360° seam
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


def process_one(
    var_code: str,
    var_info: dict,
    raw_dir: Path,
    overwrite: bool = False,
) -> Path | None:
    """Process a single GMTED2010 variable to NetCDF on the master grid."""
    out_path = FINAL_DIR / f"{var_code}.nc"

    if out_path.exists() and not overwrite:
        logger.info("Already exists: %s", out_path.name)
        return None

    raw_path = _find_raw_path(var_code, var_info, raw_dir)
    if raw_path is None:
        logger.warning("Raw data not found for %s", var_code)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Resample to 0.25° via gdalwarp (to a temp file)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        logger.info("Resampling %s → 0.25° ...", var_code)
        if not _resample_to_025(raw_path, tmp_path):
            return None

        da = open_tif_as_dataarray(tmp_path)
        da = regrid_to_master(da)
    finally:
        tmp_path.unlink(missing_ok=True)

    # GMTED2010 coverage ends at ~84°N and ~90°S; fill polar gaps with 0
    values = da.values.astype(np.float32)
    values = np.where(np.isnan(values), 0.0, values)

    ds = xr.Dataset(
        {
            var_code: (
                ["lat", "lon"],
                values,
                {
                    "units": var_info["units"],
                    "long_name": var_info["long_name"],
                    "source": "GMTED2010 (USGS)",
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
            "source": "GMTED2010 (USGS)",
            "resolution": "0.25 degree",
        },
    )

    ds.to_netcdf(
        out_path,
        encoding={var_code: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    logger.info("Saved %s → %s", var_code, out_path)
    return out_path


def process_gmted(
    variables: list[str] | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> int:
    """Process GMTED2010 grids to NetCDF. Returns count of files written."""
    config = load_gmted_config()
    all_vars = config["variables"]
    var_list = variables or list(all_vars.keys())
    processed = 0

    for var_code in var_list:
        var_info = all_vars.get(var_code)
        if var_info is None:
            logger.warning("Unknown variable: %s", var_code)
            continue

        result = process_one(var_code, var_info, raw_dir, overwrite)
        if result:
            processed += 1

    return processed


@click.command()
@click.option("--variables", "-v", multiple=True,
              help="Variable(s) to process (e.g. elevation_mean slope_mean).")
@click.option("--all", "run_all", is_flag=True, help="Process all variables.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Raw data directory (default: {DEFAULT_RAW_DIR})")
def main(variables, run_all, overwrite, raw_dir):
    """Resample GMTED2010 grids to 0.25° master grid NetCDFs."""
    if not variables and not run_all:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    src_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    var_list = list(variables) if variables else None

    n = process_gmted(
        variables=var_list,
        raw_dir=src_dir,
        overwrite=overwrite,
    )
    click.echo(f"Processed {n} files. Output in {FINAL_DIR}")


if __name__ == "__main__":
    main()
