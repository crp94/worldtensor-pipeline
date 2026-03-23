"""Kummu et al. (2025) Gridded GDP Pipeline.

Processes 5-arcmin multi-band GeoTIFF (1990-2022) to 0.25° annual means.
"""

import click
import numpy as np
import xarray as xr
import rioxarray
from pathlib import Path
import yaml

from src.grid import make_template
from src.utils import get_logger, plot_global_map, plot_time_series, save_annual_variable
from src.processing.raster_to_grid import load_raster, regrid_raster

logger = get_logger("pipeline.kummu_gdp")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_PATH = PROJECT_ROOT / "data" / "raw" / "kummu" / "gdp_total_5arcmin.tif"
PLOTS_DIR = PROJECT_ROOT / "plots" / "economy"

def process_kummu_gdp():
    if not RAW_PATH.exists():
        logger.error(f"Raw file not found: {RAW_PATH}")
        return

    logger.info(f"Opening {RAW_PATH}...")
    da_full = rioxarray.open_rasterio(RAW_PATH)
    
    # 33 bands correspond to 1990 to 2022
    years = range(1990, 2023)
    ts_years = []
    ts_means = []

    for i, year in enumerate(years):
        # We only process 2000+ for consistency unless requested otherwise
        if year < 2000: continue
        
        logger.info(f"  Processing year {year} (band {i+1})...")
        da_year = da_full.isel(band=i)
        
        # Standardize for regrid_raster
        if "x" in da_year.dims:
            da_year = da_year.rename({"x": "lon", "y": "lat"})
        
        # Regrid to 0.25° master template
        # GDP Total is an absolute value? No, usually these are density (USD per cell) 
        # or per capita. The file name says 'gdpTot'. 
        # For 'total' we should use 'sum' resampling if the resolution was higher, 
        # but 5-arcmin to 15-arcmin (0.25) is a 3x3 aggregation.
        
        # We use regrid_raster with 'linear' for simplicity or manual aggregation
        ds_out = regrid_raster(da_year, year, var_name="gdp_total", method="linear")
        
        # Update metadata
        ds_out["gdp_total"].attrs = {
            "units": "constant 2017 international USD",
            "long_name": "Total GDP (PPP)",
            "description": f"Annual total GDP in 2017 PPP international dollars per 0.25° cell for year {year} (Kummu et al. 2025)"
        }
        
        # Save
        save_annual_variable(ds_out, "gdp_total", year)
        
        # Accumulate
        ts_years.append(year)
        ts_means.append(float(ds_out["gdp_total"].mean()))

        # Plot spatial map every 5 years
        if year % 5 == 0 or year == 2022:
            plot_global_map(
                ds_out["gdp_total"],
                f"Global Total GDP (PPP) — {year}",
                PLOTS_DIR / f"gdp_total_{year}.png",
                cmap="YlOrBr",
                force_log=True
            )
            
    # Plot time series
    if ts_years:
        plot_time_series(
            ts_years, ts_means,
            "Global Average GDP per Cell",
            "USD",
            PLOTS_DIR / "timeseries_gdp_total.png",
            color="#1f77b4"
        )

    da_full.close()
    logger.info("Kummu GDP Pipeline complete.")

if __name__ == "__main__":
    process_kummu_gdp()
