"""Regrid SoilGrids 2.0 GeoTIFFs to the master 0.25° grid as static NetCDFs.

Data source
-----------
Website : https://www.isric.org/explore/soilgrids
Citation: Poggio et al. (2021), Soil 7:217-240, doi:10.5194/soil-7-217-2021

Reads gdalwarp output (1440×720, -180..180 lon), shifts longitude to 0..360,
interpolates to the 721×1440 master grid with wrap-padding at the 0°/360°
seam, and saves as CF-1.8 NetCDF.

Output: data/final/static/soilgrids/{property}_{depth}_mean.nc

Usage:
    python -m src.processing.soilgrids_to_static --all
    python -m src.processing.soilgrids_to_static --properties bdod clay
"""

from pathlib import Path

import click
import numpy as np
import rioxarray  # noqa: F401 — registers .rio accessor
import xarray as xr
import yaml

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.utils import get_logger

logger = get_logger("processing.soilgrids")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "soilgrids.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "soilgrids"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "static" / "soilgrids"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)


def load_soilgrids_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


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
    the 0°/360° seam (source grid may not have values at exactly 0 or 360).
    """
    lon_shifted = da.lon.values % 360
    da = da.assign_coords(lon=lon_shifted).sortby("lon")

    # Wrap: copy the last slice to lon=0-ε and first slice to lon=360+ε
    # so linear interpolation covers the full 0..359.75 target range.
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
    tif_path: Path,
    prop: str,
    depth: str,
    statistic: str,
    units: str,
    long_name: str,
    overwrite: bool = False,
) -> Path | None:
    """Process a single SoilGrids TIF to NetCDF on the master grid."""
    var_name = f"{prop}_{depth}_{statistic}"
    out_path = FINAL_DIR / f"{var_name}.nc"

    if out_path.exists() and not overwrite:
        logger.info("Already exists: %s", out_path.name)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    da = open_tif_as_dataarray(tif_path)
    da = regrid_to_master(da)

    ds = xr.Dataset(
        {
            var_name: (
                ["lat", "lon"],
                da.values.astype(np.float32),
                {
                    "units": units,
                    "long_name": f"{long_name} ({depth})",
                    "source": "SoilGrids 2.0 (ISRIC)",
                    "depth": depth,
                },
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor SoilGrids {prop} {depth}",
            "source": "SoilGrids 2.0 (ISRIC)",
            "resolution": "0.25 degree",
        },
    )

    ds.to_netcdf(
        out_path,
        encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    logger.info("Saved %s → %s", var_name, out_path)
    return out_path


def process_soilgrids(
    properties: list[str] | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> int:
    """Process all SoilGrids TIFs to NetCDF. Returns count of files written."""
    config = load_soilgrids_config()
    statistic = config["statistic"]
    depths = config["depths"]
    all_props = config["properties"]

    prop_list = properties or list(all_props.keys())
    processed = 0

    for prop in prop_list:
        prop_info = all_props.get(prop)
        if prop_info is None:
            logger.warning("Unknown property: %s", prop)
            continue

        for depth in depths:
            tif_name = f"{prop}_{depth}_{statistic}.tif"
            tif_path = raw_dir / tif_name

            if not tif_path.exists():
                logger.warning("TIF not found: %s", tif_path)
                continue

            result = process_one(
                tif_path, prop, depth, statistic,
                prop_info["units"], prop_info["long_name"],
                overwrite,
            )
            if result:
                processed += 1

    return processed


@click.command()
@click.option("--properties", "-p", multiple=True,
              help="Property(ies) to process (e.g. bdod clay).")
@click.option("--all", "run_all", is_flag=True, help="Process all properties.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Raw data directory (default: {DEFAULT_RAW_DIR})")
def main(properties, run_all, overwrite, raw_dir):
    """Regrid SoilGrids GeoTIFFs to 0.25° master grid NetCDFs."""
    if not properties and not run_all:
        click.echo("Specify --properties or --all. Use --help for usage.")
        return

    src_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    prop_list = list(properties) if properties else None

    n = process_soilgrids(
        properties=prop_list,
        raw_dir=src_dir,
        overwrite=overwrite,
    )
    click.echo(f"Processed {n} files. Output in {FINAL_DIR}")


if __name__ == "__main__":
    main()
