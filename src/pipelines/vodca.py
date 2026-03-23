"""VODCA L-band Pipeline.

Processes 0.25° 10-daily VOD to yearly mean.
"""

import click
import numpy as np
import xarray as xr
from pathlib import Path
import yaml
import zipfile
import os

from src.utils import get_logger, plot_global_map, plot_time_series, save_annual_variable
from src.grid import make_template
from src.processing.raster_to_grid import regrid_raster

logger = get_logger("pipeline.vodca")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "vodca.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "vodca"
PLOTS_DIR = PROJECT_ROOT / "plots" / "vodca"

def process_vodca():
    # 1. Extract if needed
    zip_path = RAW_DIR / "VODCA_L.zip"
    if not zip_path.exists():
        logger.error(f"Raw zip not found: {zip_path}")
        return

    extract_dir = RAW_DIR / "extracted"
    if not extract_dir.exists():
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_dir)

    # 2. Years available
    years = range(2010, 2022)
    ts_years = []
    ts_means = []

    for year in years:
        year_dir = extract_dir / "daily_images_VODCA_L" / str(year)
        if not year_dir.exists():
            logger.warning(f"No directory for year {year}")
            continue
            
        logger.info(f"Processing year {year}...")
        nc_files = list(year_dir.glob("*.nc"))
        if not nc_files:
            continue
            
        # Load all 10-daily files for the year
        # Variable name in VODCA is 'VODCA_L'
        datasets = []
        for f in nc_files:
            try:
                ds_tmp = xr.open_dataset(f)
                if 'VODCA_L' in ds_tmp.data_vars:
                    datasets.append(ds_tmp['VODCA_L'])
                ds_tmp.close()
            except Exception as e:
                logger.warning(f"Failed to load {f.name}: {e}")
        
        if not datasets:
            continue
            
        # Concatenate and mean
        da_year_raw = xr.concat(datasets, dim='time').mean(dim='time')
        
        # 3. Regrid to master grid (VODCA is already 0.25, but may have different orientation/alignment)
        ds_out = regrid_raster(da_year_raw, year, var_name="vod", method="linear")
        
        # Update metadata
        ds_out["vod"].attrs = {
            "units": "1",
            "long_name": "Vegetation Optical Depth (L-band)",
            "description": f"Annual mean VOD from VODCA v2 L-band for year {year}"
        }
        
        # 4. Save to final
        save_annual_variable(ds_out, "vod", year)
        
        # Accumulate for time series
        ts_years.append(year)
        ts_means.append(float(ds_out["vod"].mean()))

        # Plot spatial map every 3 years or first/last
        if year % 3 == 0 or year == 2010 or year == 2021:
            plot_global_map(
                ds_out["vod"],
                f"VODCA L-band VOD — {year}",
                PLOTS_DIR / f"vod_{year}.png",
                cmap="YlGn"
            )
            
    # 5. Plot time series
    if ts_years:
        plot_time_series(
            ts_years, ts_means,
            "VODCA Global Mean Vegetation Optical Depth (L-band)",
            "VOD (unitless)",
            PLOTS_DIR / "timeseries_vod.png",
            color="#2ca02c"
        )

    logger.info("VODCA Pipeline complete.")

if __name__ == "__main__":
    process_vodca()
