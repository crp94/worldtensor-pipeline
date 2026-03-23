"""WAD2M Wetlands Pipeline.

Processes 0.25° monthly inundation fraction to yearly mean/max.
Source: Zhang et al. (2021), ESSD.
"""

import click
import numpy as np
import xarray as xr
from pathlib import Path
import yaml
import zipfile

from src.utils import get_logger, save_annual_variable, plot_global_map
from src.grid import make_template
from src.processing.raster_to_grid import regrid_raster

logger = get_logger("pipeline.wad2m")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "wetlands_wad2m.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "wad2m"
PLOTS_DIR = PROJECT_ROOT / "plots" / "wad2m"

def process_wad2m():
    # 1. Locate and extract zip if needed
    zip_path = RAW_DIR / "WAD2M_wetlands_2000-2020_025deg_Ver2.0.nc.zip"
    if not zip_path.exists():
        logger.error(f"Raw zip not found: {zip_path}")
        return

    extracted_nc = RAW_DIR / "WAD2M_wetlands_2000-2020_025deg_Ver2.0.nc"
    if not extracted_nc.exists():
        logger.info(f"Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DIR)

    # 2. Load the multi-year monthly dataset
    ds = xr.open_dataset(extracted_nc)
    
    # Standardize coordinate names to lat/lon
    if 'latitude' in ds.coords:
        ds = ds.rename({'latitude': 'lat', 'longitude': 'lon'})
    
    # 3. Process each year
    years = range(2000, 2021)
    ts_years = []
    ts_means = []
    ts_maxes = []

    for year in years:
        logger.info(f"Processing year {year}...")
        
        # Slice the year
        ds_year = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        
        # Calculate yearly stats
        # The variable in WAD2M is 'Fw' (inundation fraction)
        da_mean_raw = ds_year['Fw'].mean(dim='time')
        da_max_raw = ds_year['Fw'].max(dim='time')
        
        # 4. Use regrid_raster to align with master template (721x1440)
        # This handles the 720 vs 721 lat difference safely via interpolation.
        ds_out_mean = regrid_raster(da_mean_raw, year, var_name="inundation_fraction_mean", method="linear")
        ds_out_max = regrid_raster(da_max_raw, year, var_name="inundation_fraction_max", method="linear")
        
        # Update metadata
        ds_out_mean["inundation_fraction_mean"].attrs = {
            "units": "1",
            "long_name": "Yearly mean inundation fraction (WAD2M)",
            "description": "Mean of monthly inundation fractions for vegetated wetlands"
        }
        ds_out_max["inundation_fraction_max"].attrs = {
            "units": "1",
            "long_name": "Yearly maximum inundation fraction (WAD2M)",
            "description": "Maximum of monthly inundation fractions for vegetated wetlands"
        }
        
        # Save both
        save_annual_variable(ds_out_mean, "inundation_fraction_mean", year)
        save_annual_variable(ds_out_max, "inundation_fraction_max", year)
        
        # Accumulate for time series
        ts_years.append(year)
        ts_means.append(float(ds_out_mean["inundation_fraction_mean"].mean()))
        ts_maxes.append(float(ds_out_max["inundation_fraction_max"].mean()))

        # Plot spatial map every 5 years or first/last
        if year % 5 == 0 or year == 2000 or year == 2020:
            plot_global_map(
                ds_out_mean["inundation_fraction_mean"],
                f"WAD2M Mean Inundation Fraction — {year}",
                PLOTS_DIR / f"mean_{year}.png",
                cmap="Blues"
            )
            plot_global_map(
                ds_out_max["inundation_fraction_max"],
                f"WAD2M Max Inundation Fraction — {year}",
                PLOTS_DIR / f"max_{year}.png",
                cmap="Blues"
            )
        
    # 5. Plot time series
    from src.utils import plot_time_series
    plot_time_series(
        ts_years, ts_means,
        "WAD2M Global Mean Inundation Fraction (Yearly Mean)",
        "Fraction",
        PLOTS_DIR / "timeseries_mean.png",
        color="#1f77b4"
    )
    plot_time_series(
        ts_years, ts_maxes,
        "WAD2M Global Mean Inundation Fraction (Yearly Max)",
        "Fraction",
        PLOTS_DIR / "timeseries_max.png",
        color="#d62728"
    )

    ds.close()
    logger.info("WAD2M Pipeline complete.")

if __name__ == "__main__":
    process_wad2m()
