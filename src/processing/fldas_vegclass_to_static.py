"""Convert FLDAS vegetation class to one-hot encoded static NetCDF layers.

Data source
-----------
Website : https://ldas.gsfc.nasa.gov/fldas/vegetation-class
Citation: McNally et al. (2017), Scientific Data 4:170012.

Overview
--------
The FLDAS dominant vegetation type is a categorical raster where each cell
holds an IGBP-modified MODIS class ID (1–20). This script converts it into
18 binary (one-hot) layers — one per land surface type — on the master 0.25°
grid (721 lat × 1440 lon, 0..360° longitude).

One-Hot Encoding
----------------
For each land surface type, a binary layer is created where:
    1 = cell is classified as this type
    0 = cell is classified as a different type
    NaN = ocean / no data

This preserves all information from the categorical map while making it
usable as independent features in ML models.

Excluded classes:
    11 = Permanent Wetlands  (absent from the data — 0 cells)
    17 = Water bodies         (excluded by config — captured by other layers)

Regridding Strategy
-------------------
The source grid is 0.1° (1500×3600, -59.95°S to 89.95°N). The master grid
is 0.25° (721×1440, -90° to 90°, 0° to 359.75°).

Nearest-neighbor interpolation is used because:
    - This is categorical data — linear interpolation would produce meaningless
      fractional values between class boundaries
    - Nearest-neighbor preserves crisp type boundaries
    - Areas outside the source extent (Antarctica < -59.95°S) become NaN

Output: data/final/static/vegclass/{var_name}.nc

Usage:
    python -m src.processing.fldas_vegclass_to_static --all
    python -m src.processing.fldas_vegclass_to_static --variables vegclass_ebf vegclass_cropland
"""

from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.utils import get_logger

logger = get_logger("processing.fldas_vegclass")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "fldas_vegclass.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fldas_vegclass"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "static" / "vegclass"

# Master grid coordinates
TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def open_source(raw_dir: Path) -> xr.DataArray:
    """Open the FLDAS vegetation class NetCDF and return as a 2D DataArray.

    Squeezes the time dimension and shifts longitude to 0..360.
    Replaces the missing value sentinel (-9999) with NaN.
    """
    config = load_config()
    src_file = raw_dir / config["source_file"]

    if not src_file.exists():
        raise FileNotFoundError(
            f"Source file not found: {src_file}\n"
            f"Run: python -m src.download.fldas_vegclass"
        )

    ds = xr.open_dataset(src_file)
    da = ds[config["source_variable"]].squeeze("time", drop=True)

    # Replace missing values with NaN
    da = da.where(da != -9999.0, np.nan)

    # Shift longitude: -180..180 → 0..360
    lon_shifted = da.lon.values % 360
    da = da.assign_coords(lon=lon_shifted).sortby("lon")

    logger.info(
        "Opened source: %s, shape=%s, lat=[%.2f, %.2f], lon=[%.2f, %.2f]",
        src_file.name, da.shape,
        float(da.lat.min()), float(da.lat.max()),
        float(da.lon.min()), float(da.lon.max()),
    )
    return da


def create_onehot_layer(source: xr.DataArray, class_id: int) -> xr.DataArray:
    """Create a binary one-hot layer for a single vegetation class.

    Parameters
    ----------
    source : xr.DataArray
        The categorical vegetation raster (integer class IDs).
    class_id : int
        The IGBP class ID to extract.

    Returns
    -------
    xr.DataArray
        Binary layer (1.0 where match, 0.0 elsewhere, NaN for no-data).
        Interpolated to the master grid via nearest-neighbor.
    """
    # Build binary mask
    mask = xr.where(source == class_id, 1.0, 0.0)
    mask = mask.where(source.notnull())  # preserve NaN for ocean/no-data

    # Wrap-pad at 0°/360° meridian seam to prevent NaN strip.
    # Copy the first few columns to lon > 360 and last few to lon < 0
    # so nearest-neighbor interpolation has data on both sides of the seam.
    pad_width = 5  # number of source cells to duplicate
    lon = mask.lon.values
    left_slice = mask.isel(lon=slice(0, pad_width))
    right_slice = mask.isel(lon=slice(-pad_width, None))
    left_pad = left_slice.assign_coords(lon=left_slice.lon.values + 360.0)
    right_pad = right_slice.assign_coords(lon=right_slice.lon.values - 360.0)
    mask = xr.concat([right_pad, mask, left_pad], dim="lon")

    # Regrid via nearest-neighbor (categorical data)
    regridded = mask.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="nearest",
        kwargs={"fill_value": np.nan},
    )

    # Fill southern gap: source stops at ~-59.95°S but master grid extends to
    # -90°S (Antarctica). No vegetation exists there, so fill with 0.
    src_lat_min = float(source.lat.min())
    south_mask = regridded.lat.values < src_lat_min
    vals = regridded.values.copy()
    vals[south_mask, :] = 0.0
    regridded = regridded.copy(data=vals)

    return regridded


def process_one(
    source: xr.DataArray,
    var_name: str,
    var_info: dict,
    overwrite: bool = False,
) -> Path | None:
    """Process a single vegetation class to a one-hot NetCDF file."""
    out_path = FINAL_DIR / f"{var_name}.nc"
    if out_path.exists() and not overwrite:
        logger.info("Already exists: %s", out_path.name)
        return None

    class_id = var_info["class_id"]
    logger.info("Creating one-hot layer: %s (class %d = %s)",
                var_name, class_id, var_info["long_name"])

    da = create_onehot_layer(source, class_id)

    # Log coverage
    valid = da.values[np.isfinite(da.values)]
    n_ones = int(np.sum(valid == 1.0))
    logger.info("  class %d: %d cells present", class_id, n_ones)

    # Build CF-compliant output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            var_name: (
                ["lat", "lon"],
                da.values.astype(np.float32),
                {
                    "units": var_info["units"],
                    "long_name": f"{var_info['long_name']} (one-hot)",
                    "source": "FLDAS IGBP-modified MODIS",
                    "igbp_class_id": class_id,
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
            "title": f"WorldTensor FLDAS vegetation — {var_info['long_name']}",
            "source": "FLDAS (McNally et al. 2017), IGBP-modified MODIS",
            "encoding": "one-hot (binary: 1 = this type, 0 = other)",
            "resolution": "0.25 degree",
        },
    )

    ds.to_netcdf(
        out_path,
        encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    logger.info("Saved %s → %s", var_name, out_path)
    return out_path


def process_vegclass(
    variables: list[str] | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> int:
    """Process FLDAS vegetation class to one-hot NetCDFs.

    Opens the source file once and creates all one-hot layers from it.

    Returns
    -------
    int
        Number of files written.
    """
    config = load_config()
    all_vars = config["variables"]
    var_list = variables or list(all_vars.keys())

    source = open_source(raw_dir)

    processed = 0
    for var_name in var_list:
        var_info = all_vars.get(var_name)
        if var_info is None:
            logger.warning("Unknown variable: %s", var_name)
            continue
        result = process_one(source, var_name, var_info, overwrite)
        if result:
            processed += 1

    return processed


@click.command()
@click.option("--variables", "-v", multiple=True,
              help="Variable(s) to process (e.g. vegclass_ebf).")
@click.option("--all", "run_all", is_flag=True, help="Process all vegetation types.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Raw data directory (default: {DEFAULT_RAW_DIR})")
def main(variables, run_all, overwrite, raw_dir):
    """Convert FLDAS vegetation class to one-hot encoded 0.25° NetCDFs."""
    if not variables and not run_all:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    src_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    var_list = list(variables) if variables else None

    n = process_vegclass(variables=var_list, raw_dir=src_dir, overwrite=overwrite)
    click.echo(f"Processed {n} files. Output in {FINAL_DIR}")


if __name__ == "__main__":
    main()
