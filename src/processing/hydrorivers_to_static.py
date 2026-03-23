"""Compute distance-to-nearest-river from HydroRIVERS vectors.

Data source
-----------
Website : https://www.hydrosheds.org/products/hydrorivers
Citation: Lehner & Grill (2013), Hydrol. Process. 27(15):2171-2186.

Processing steps:
    1. gdal_rasterize: burn river vectors into a 0.25° presence raster
    2. gdal_proximity: compute distance from each cell to nearest river pixel
    3. Convert pixel distances to km, regrid to master grid

Output: data/final/static/geography/dist_to_river.nc

Usage:
    python -m src.processing.hydrorivers_to_static --all
    python -m src.processing.hydrorivers_to_static --overwrite
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

logger = get_logger("processing.hydrorivers")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "hydrorivers.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hydrorivers"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "static" / "geography"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)

# Approximate km per degree at equator
KM_PER_DEG = 111.32


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _find_shapefile(raw_dir: Path, config: dict) -> Path | None:
    """Find the HydroRIVERS shapefile."""
    shp_path = raw_dir / config["shapefile"]
    if shp_path.exists():
        return shp_path
    # Search for any .shp file
    shps = list(raw_dir.rglob("*.shp"))
    return shps[0] if shps else None


def _rasterize_rivers(shp_path: Path, out_tif: Path) -> bool:
    """Rasterize river vectors to a 0.25° presence raster."""
    cmd = [
        "gdal_rasterize",
        "-burn", "1",
        "-te", "-180", "-90", "180", "90",
        "-tr", "0.25", "0.25",
        "-ot", "Byte",
        "-init", "0",
        "-a_nodata", "255",
        "-l", shp_path.stem,
        str(shp_path),
        str(out_tif),
    ]
    logger.info("Rasterizing rivers to 0.25° ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("gdal_rasterize failed: %s", result.stderr.strip())
        return False
    return True


def _compute_proximity(src_tif: Path, out_tif: Path) -> bool:
    """Compute proximity (distance to nearest river pixel)."""
    cmd = [
        "gdal_proximity.py",
        str(src_tif),
        str(out_tif),
        "-values", "1",
        "-distunits", "GEO",
        "-ot", "Float32",
        "-co", "COMPRESS=LZW",
    ]
    logger.info("Computing proximity distances ...")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("gdal_proximity failed: %s", result.stderr.strip())
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


def process_hydrorivers(
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> Path | None:
    """Compute distance-to-river and save as NetCDF."""
    config = load_config()
    var_name = "dist_to_river"
    var_info = config["variables"][var_name]

    out_path = FINAL_DIR / f"{var_name}.nc"
    if out_path.exists() and not overwrite:
        logger.info("Already exists: %s", out_path.name)
        return None

    shp_path = _find_shapefile(raw_dir, config)
    if shp_path is None:
        logger.error("Shapefile not found in %s", raw_dir)
        return None

    logger.info("Using shapefile: %s", shp_path)

    # Step 1: Rasterize rivers
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp1:
        raster_path = Path(tmp1.name)
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp2:
        prox_path = Path(tmp2.name)

    try:
        if not _rasterize_rivers(shp_path, raster_path):
            return None

        # Step 2: Compute proximity
        if not _compute_proximity(raster_path, prox_path):
            return None

        # Step 3: Open proximity, convert to km, regrid
        da = open_tif_as_dataarray(prox_path)

        # gdal_proximity with -distunits GEO gives distance in degrees
        # Convert to km (approximate using equatorial degree length)
        da = da * KM_PER_DEG

        da = regrid_to_master(da)
    finally:
        raster_path.unlink(missing_ok=True)
        prox_path.unlink(missing_ok=True)

    # Mask ocean cells using dist_to_coast (land = negative values)
    dist_coast_path = FINAL_DIR / "dist_to_coast.nc"
    if dist_coast_path.exists():
        ds_coast = xr.open_dataset(dist_coast_path)
        land_mask = ds_coast["dist_to_coast"].values < 0  # negative = land
        ds_coast.close()
        values = da.values.astype(np.float32)
        values[~land_mask] = np.nan
        da = da.copy(data=values)
        logger.info("Applied land-only mask from dist_to_coast")
    else:
        logger.warning("dist_to_coast.nc not found — ocean cells not masked")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            var_name: (
                ["lat", "lon"],
                da.values.astype(np.float32),
                {
                    "units": var_info["units"],
                    "long_name": var_info["long_name"],
                    "source": "HydroRIVERS v1.0 (Lehner & Grill 2013)",
                },
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "WorldTensor Distance to Nearest River",
            "source": "HydroRIVERS v1.0 (Lehner & Grill 2013)",
            "resolution": "0.25 degree",
        },
    )
    ds.to_netcdf(
        out_path,
        encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    logger.info("Saved %s → %s", var_name, out_path)
    return out_path


@click.command()
@click.option("--all", "run_all", is_flag=True, help="Process all variables.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Raw data directory (default: {DEFAULT_RAW_DIR})")
def main(run_all, overwrite, raw_dir):
    """Compute distance-to-river from HydroRIVERS on 0.25° master grid."""
    if not run_all:
        click.echo("Specify --all. Use --help for usage.")
        return

    src_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    result = process_hydrorivers(raw_dir=src_dir, overwrite=overwrite)
    if result:
        click.echo(f"Output: {result}")
    else:
        click.echo("Skipped or failed.")


if __name__ == "__main__":
    main()
