"""Full NTL pipeline: download → regrid → visualize.

Downloads global Nighttime Light (1992-2018), regrids to 0.25° master grid,
and generates spatial maps and temporal trends.

Usage:
    python -m src.pipelines.ntl
"""

from pathlib import Path

import click
import matplotlib.pyplot as plt
import xarray as xr
import yaml
from tqdm import tqdm

from src.download.ntl import download_ntl, load_ntl_config, DEFAULT_RAW_DIR as RAW_DIR
from src.processing.raster_to_grid import load_raster, regrid_raster
try:
    from src.visualization.plot_raster import plot_variable
except ImportError:
    plot_variable = None
try:
    from src.visualization.statistics import plot_temporal_trend
except ImportError:
    plot_temporal_trend = None
from src.utils import get_logger

logger = get_logger("pipeline.ntl")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "human_systems"
PLOTS_DIR = PROJECT_ROOT / "plots" / "ntl"


def process_ntl(overwrite: bool = False, cleanup: bool = True):
    """Run the full NTL pipeline."""
    config = load_ntl_config()
    var_key = "ntl_harmonized"
    var_info = config["variables"][var_key]
    t_range = config["temporal_range"]
    years = list(range(t_range[0], t_range[1] + 1))
    
    # 1. Download
    raw_files = download_ntl(overwrite=overwrite)
    if not raw_files:
        logger.error("No raw files available for processing.")
        return

    # 2. Process each year
    logger.info("Processing NTL variable: %s", var_key)
    out_var_dir = FINAL_DIR / var_key
    out_var_dir.mkdir(parents=True, exist_ok=True)
    
    for year in tqdm(years, desc="Regridding NTL"):
        out_path = out_var_dir / f"{year}.nc"
        if out_path.exists() and not overwrite:
            continue
            
        # DMSP vs VIIRS pattern
        if year <= 2013:
            fname = var_info["filename_pattern_dmsp"].format(year=year)
        else:
            fname = var_info["filename_pattern_viirs"].format(year=year)
            
        raw_path = RAW_DIR / fname
        if not raw_path.exists():
            # Try recursive search if naming differs slightly
            candidates = list(RAW_DIR.glob(f"*{year}*.tif"))
            if candidates:
                raw_path = candidates[0]
            else:
                logger.warning("File not found for year %d: %s", year, fname)
                continue
                
        # Load and regrid
        da = load_raster(raw_path)
        da.attrs["units"] = var_info["units"]
        da.attrs["long_name"] = var_info["long_name"]
        da.name = var_key
        
        ds_regridded = regrid_raster(da, year=year, var_name=var_key)
        
        # Save to NetCDF
        encoding = {var_key: {"zlib": True, "complevel": 4, "dtype": "float32"}}
        ds_regridded.to_netcdf(out_path, encoding=encoding)
        ds_regridded.close()
        
    # 3. Visualize
    # Maps every 5 years
    plot_years = [y for y in years if y % 5 == 0 or y == years[-1]]
    for year in plot_years:
        nc_path = out_var_dir / f"{year}.nc"
        if not nc_path.exists():
            continue
        ds = xr.open_dataset(nc_path)
        map_out = PLOTS_DIR / "maps" / f"{var_key}_{year}.png"
        if plot_variable is not None:
            plot_variable(ds, var_name=var_key, year=year, output_path=map_out, log_scale=True)
        ds.close()
        
    # Temporal trend
    trend_out = PLOTS_DIR / "trends" / f"{var_key}_trend.png"
    trend_out.parent.mkdir(parents=True, exist_ok=True)
    if plot_temporal_trend is not None:
        plot_temporal_trend(var_key, data_dir=FINAL_DIR, output_path=trend_out)

    # 4. Cleanup
    if cleanup:
        logger.info("Cleaning up raw NTL files...")
        for p in RAW_DIR.glob("*.tif"):
            p.unlink()
        if RAW_DIR.exists() and not any(RAW_DIR.iterdir()):
            RAW_DIR.rmdir()


@click.command()
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
@click.option("--no-cleanup", is_flag=True, help="Keep raw files after processing.")
def main(overwrite, no_cleanup):
    """NTL Pipeline: Download → Regrid → Visualize."""
    process_ntl(overwrite=overwrite, cleanup=not no_cleanup)
    click.echo("\nNTL pipeline complete.")


if __name__ == "__main__":
    main()
