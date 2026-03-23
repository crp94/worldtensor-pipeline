"""Convert GLDAS soil texture to one-hot encoded static NetCDF layers.

Overview
--------
The GLDAS/Noah soil texture file contains a single categorical raster where
each cell holds an integer FAO class ID (1–16). This script converts it into
12 binary (one-hot) layers — one per soil texture type — on the master 0.25°
grid (721 lat × 1440 lon, 0..360° longitude).

One-Hot Encoding
----------------
For each soil type, a binary layer is created where:
    1 = cell matches this soil type
    0 = cell does not match (or is NaN / ocean / no data)

This encoding allows the soil types to be used as independent features in
machine learning models without imposing ordinal relationships.

Excluded classes:
    5  = Silt          (absent from the data)
    14 = (unused)      (absent from the data)
    15 = Bedrock       (non-soil, excluded by config)
    16 = Other         (non-soil, excluded by config)

Regridding Strategy
-------------------
The source grid is 600×1440 (-59.875°S..89.875°N, -179.875°..179.875° lon).
The master grid is 721×1440 (-90°..90°N, 0°..359.75° lon).

Steps:
    1. Shift source longitude from [-180, 180) to [0, 360)
    2. For each class: create binary mask (source_value == class_id)
    3. Interpolate to master grid with nearest-neighbor (preserves crisp
       categorical boundaries — linear interpolation would blur them)
    4. Areas outside the source extent (Antarctica, <-59.875°S) are NaN

Output: data/final/static/soiltex/{var_name}.nc  (e.g. soiltex_sand.nc)

Usage:
    python -m src.processing.gldas_soiltex_to_static --all
    python -m src.processing.gldas_soiltex_to_static --variables soiltex_sand soiltex_clay
"""

from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.utils import get_logger

logger = get_logger("processing.gldas_soiltex")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "gldas_soiltex.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "gldas_soils"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "static" / "soiltex"

# Master grid coordinates (0.25° spacing)
TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)   # -90 .. 90, 721 pts
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)   # 0 .. 359.75, 1440 pts


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def open_source(raw_dir: Path) -> xr.DataArray:
    """Open the GLDAS soil texture NetCDF and return as a 2D DataArray.

    The source file has dims (time=1, lat=600, lon=1440). We squeeze out
    the time dimension and shift longitude to 0..360 to match the master grid.
    """
    config = load_config()
    src_file = raw_dir / config["source_file"]

    if not src_file.exists():
        raise FileNotFoundError(
            f"Source file not found: {src_file}\n"
            f"Run: python -m src.download.gldas_soiltex"
        )

    ds = xr.open_dataset(src_file)
    da = ds[config["source_variable"]].squeeze("time", drop=True)

    # Replace missing values (-9999) with NaN
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


def create_onehot_layer(
    source: xr.DataArray,
    class_id: int,
) -> xr.DataArray:
    """Create a binary one-hot layer for a single soil class.

    Parameters
    ----------
    source : xr.DataArray
        The categorical soil texture raster (integer class IDs).
    class_id : int
        The FAO class ID to extract (e.g. 1=Sand, 12=Clay).

    Returns
    -------
    xr.DataArray
        Binary layer (1.0 where source == class_id, 0.0 elsewhere, NaN for
        ocean/no-data). Interpolated to the master grid via nearest-neighbor.
    """
    # Build binary mask: 1.0 where class matches, 0.0 where different class,
    # NaN where source is NaN (ocean / no data)
    mask = xr.where(source == class_id, 1.0, 0.0)
    mask = mask.where(source.notnull())  # preserve NaN for ocean pixels

    # Wrap-pad at 0°/360° meridian seam to prevent NaN strip.
    # Copy the first few columns to lon > 360 and last few to lon < 0
    # so nearest-neighbor interpolation has data on both sides of the seam.
    pad_width = 5
    left_slice = mask.isel(lon=slice(0, pad_width))
    right_slice = mask.isel(lon=slice(-pad_width, None))
    left_pad = left_slice.assign_coords(lon=left_slice.lon.values + 360.0)
    right_pad = right_slice.assign_coords(lon=right_slice.lon.values - 360.0)
    mask = xr.concat([right_pad, mask, left_pad], dim="lon")

    # Regrid to master grid via nearest-neighbor interpolation.
    # Nearest-neighbor is the correct choice for categorical data — it
    # preserves crisp soil type boundaries without blurring.
    regridded = mask.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="nearest",
        kwargs={"fill_value": np.nan},
    )

    return regridded


def process_one(
    source: xr.DataArray,
    var_name: str,
    var_info: dict,
    overwrite: bool = False,
) -> Path | None:
    """Process a single soil texture class to a one-hot NetCDF file.

    Parameters
    ----------
    source : xr.DataArray
        The categorical soil texture raster.
    var_name : str
        Output variable name (e.g. "soiltex_sand").
    var_info : dict
        Config entry with class_id, long_name, units.
    overwrite : bool
        Whether to overwrite existing files.

    Returns
    -------
    Path or None
        Output path if written, None if skipped.
    """
    out_path = FINAL_DIR / f"{var_name}.nc"
    if out_path.exists() and not overwrite:
        logger.info("Already exists: %s", out_path.name)
        return None

    class_id = var_info["class_id"]
    logger.info("Creating one-hot layer: %s (class %d = %s)",
                var_name, class_id, var_info["long_name"])

    # Create the binary one-hot mask and regrid
    da = create_onehot_layer(source, class_id)

    # Log coverage statistics
    valid = da.values[np.isfinite(da.values)]
    n_ones = int(np.sum(valid == 1.0))
    n_zeros = int(np.sum(valid == 0.0))
    logger.info("  class %d: %d cells = 1, %d cells = 0, %d NaN",
                class_id, n_ones, n_zeros, da.size - valid.size)

    # Build CF-compliant output dataset
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            var_name: (
                ["lat", "lon"],
                da.values.astype(np.float32),
                {
                    "units": var_info["units"],
                    "long_name": f"{var_info['long_name']} (one-hot)",
                    "source": "GLDAS/Noah FAO soil texture",
                    "fao_class_id": class_id,
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
            "title": f"WorldTensor GLDAS soil texture — {var_info['long_name']}",
            "source": "GLDAS/Noah (STATSGO + FAO Soil Map of the World)",
            "encoding": "one-hot (binary: 1 = this soil type, 0 = other)",
            "resolution": "0.25 degree",
        },
    )

    ds.to_netcdf(
        out_path,
        encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    logger.info("Saved %s → %s", var_name, out_path)
    return out_path


def process_soiltex(
    variables: list[str] | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> int:
    """Process GLDAS soil texture to one-hot NetCDFs.

    Opens the source file once and creates all one-hot layers from it.

    Parameters
    ----------
    variables : list of str, optional
        Specific variables to process. None = all from config.
    raw_dir : Path
        Directory containing the source NetCDF.
    overwrite : bool
        Overwrite existing output files.

    Returns
    -------
    int
        Number of files written.
    """
    config = load_config()
    all_vars = config["variables"]
    var_list = variables or list(all_vars.keys())

    # Open the source raster once (it's small, ~150 KB)
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
              help="Variable(s) to process (e.g. soiltex_sand).")
@click.option("--all", "run_all", is_flag=True, help="Process all soil types.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Raw data directory (default: {DEFAULT_RAW_DIR})")
def main(variables, run_all, overwrite, raw_dir):
    """Convert GLDAS soil texture to one-hot encoded 0.25° NetCDFs."""
    if not variables and not run_all:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    src_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    var_list = list(variables) if variables else None

    n = process_soiltex(
        variables=var_list,
        raw_dir=src_dir,
        overwrite=overwrite,
    )
    click.echo(f"Processed {n} files. Output in {FINAL_DIR}")


if __name__ == "__main__":
    main()
