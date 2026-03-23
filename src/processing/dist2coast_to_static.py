"""Regrid distance-to-coastline to the master 0.25° grid as a static NetCDF.

Data source
-----------
ERDDAP   : https://upwell.pfeg.noaa.gov/erddap/griddap/dist2coast_1deg.html
Docs     : https://oceancolor.gsfc.nasa.gov/resources/docs/distfromcoast/
Citation : NASA OBPG (2012). Distance to nearest coastline.

Processing steps
----------------
The ERDDAP download gives us a 720×1440 grid at 0.25° resolution:
    - Latitude:  90.0 to -89.75  (720 points, descending)
    - Longitude: -180.0 to 179.75 (1440 points)

The master WorldTensor grid is:
    - Latitude:  -90.0 to 90.0  (721 points, ascending)
    - Longitude: 0.0 to 359.75  (1440 points)

Transformation:
    1. Shift longitude from [-180, 180) to [0, 360) via modulo
    2. Wrap-pad the 0°/360° seam for smooth interpolation
    3. Interpolate to master grid (linear, with extrapolation for ±90°)

The source grid is missing -90.0°, so we use fill_value="extrapolate" to
extend coverage to the South Pole. The distance field is smooth and
continuous, so linear interpolation is appropriate (unlike categorical data).

Sign convention (preserved from source):
    positive = over ocean (distance to nearest land, km)
    negative = over land  (distance to nearest coast, km)
    zero     = exactly on coastline

Output: data/final/static/geography/dist_to_coast.nc

Usage:
    python -m src.processing.dist2coast_to_static
    python -m src.processing.dist2coast_to_static --overwrite
"""

from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.utils import get_logger

logger = get_logger("processing.dist2coast")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "dist2coast.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "dist2coast"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "static" / "geography"

# Master grid coordinates
TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)   # -90 .. 90, 721 pts
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)   # 0 .. 359.75, 1440 pts


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def regrid_to_master(da: xr.DataArray) -> xr.DataArray:
    """Shift longitude to 0..360 and interpolate to the master grid.

    Adds wrap-padding at the 0°/360° seam so linear interpolation works
    correctly across the date line. Uses extrapolation for latitudes
    beyond the source extent (the source stops at -89.75°, but the
    master grid extends to -90.0°).
    """
    # Shift longitude from [-180, 180) → [0, 360)
    lon_shifted = da.lon.values % 360
    da = da.assign_coords(lon=lon_shifted).sortby("lon")

    # Wrap-pad: copy edge slices across the 0°/360° seam
    # This ensures linear interpolation covers the full 0..359.75 range
    first = da.isel(lon=0).assign_coords(lon=da.lon.values[0] + 360.0)
    last = da.isel(lon=-1).assign_coords(lon=da.lon.values[-1] - 360.0)
    da = xr.concat([last, da, first], dim="lon")

    # Interpolate to master grid
    # - Linear interpolation is appropriate for this smooth distance field
    # - "extrapolate" handles the poles (source may not reach exactly ±90°)
    da = da.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )
    return da


def process_dist2coast(
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> Path | None:
    """Regrid the distance-to-coast NetCDF to the master grid.

    Parameters
    ----------
    raw_dir : Path
        Directory containing the downloaded ERDDAP file.
    overwrite : bool
        Overwrite existing output file.

    Returns
    -------
    Path or None
        Output path if written, None if skipped.
    """
    config = load_config()
    var_info = config["variables"]["dist_to_coast"]
    var_name = "dist_to_coast"

    out_path = FINAL_DIR / f"{var_name}.nc"
    if out_path.exists() and not overwrite:
        logger.info("Already exists: %s", out_path.name)
        return None

    # Open source file
    src_file = raw_dir / config["source_file"]
    if not src_file.exists():
        raise FileNotFoundError(
            f"Source file not found: {src_file}\n"
            f"Run: python -m src.download.dist2coast"
        )

    ds = xr.open_dataset(src_file)
    da = ds[config["source_variable"]]

    # The ERDDAP file uses "latitude"/"longitude" as coordinate names.
    # Rename to "lat"/"lon" for consistency with our regridding code.
    da = da.rename({"latitude": "lat", "longitude": "lon"})

    # Drop time dimension if present (it's a static dataset)
    if "time" in da.dims:
        da = da.squeeze("time", drop=True)

    # Replace fill value (0 in source) with NaN
    # Note: In the ERDDAP data, _FillValue=0 is set, but xarray handles this
    # automatically when opening. However, true coastline cells may also be 0.
    # We keep 0 as a valid value since it means "exactly on the coast".

    logger.info(
        "Source: shape=%s, lat=[%.2f, %.2f], lon=[%.2f, %.2f]",
        da.shape,
        float(da.lat.min()), float(da.lat.max()),
        float(da.lon.min()), float(da.lon.max()),
    )

    # Regrid to master grid
    da = regrid_to_master(da)

    # Build CF-compliant output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_ds = xr.Dataset(
        {
            var_name: (
                ["lat", "lon"],
                da.values.astype(np.float32),
                {
                    "units": var_info["units"],
                    "long_name": var_info["long_name"],
                    "sign_convention": var_info["sign_convention"],
                    "source": "NASA GSFC OBPG via NOAA ERDDAP",
                },
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT,
                    {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON,
                    {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "WorldTensor Distance to Nearest Coastline",
            "source": "NASA GSFC OBPG (2012), via NOAA ERDDAP",
            "resolution": "0.25 degree",
            "note": "Positive = ocean, negative = land. Landlocked water = land.",
        },
    )

    out_ds.to_netcdf(
        out_path,
        encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )

    # Log summary statistics
    vals = da.values
    land = vals[vals < 0]
    ocean = vals[vals > 0]
    logger.info("Saved %s → %s", var_name, out_path)
    logger.info("  Land cells:  %d (max inland: %.0f km)", land.size, abs(land.min()) if land.size else 0)
    logger.info("  Ocean cells: %d (max offshore: %.0f km)", ocean.size, ocean.max() if ocean.size else 0)

    ds.close()
    return out_path


@click.command()
@click.option("--overwrite", is_flag=True, help="Overwrite existing output file.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Raw data directory (default: {DEFAULT_RAW_DIR})")
def main(overwrite, raw_dir):
    """Regrid distance-to-coastline to 0.25° master grid."""
    src_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    result = process_dist2coast(raw_dir=src_dir, overwrite=overwrite)
    if result:
        click.echo(f"Output: {result}")
    else:
        click.echo("Skipped (already exists). Use --overwrite to regenerate.")


if __name__ == "__main__":
    main()
