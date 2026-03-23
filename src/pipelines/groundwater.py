"""NASA GLDAS Groundwater and Soil Moisture Pipeline.

Processes 0.25° monthly GLDAS NOAH componentsto 0.25° annual means.
"""

import click
import numpy as np
import xarray as xr
from pathlib import Path
import yaml

from src.utils import get_logger, save_annual_variable, plot_global_map, plot_time_series
from src.grid import make_template
from src.processing.raster_to_grid import regrid_raster

logger = get_logger("pipeline.groundwater")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_BASE_DIR = PROJECT_ROOT / "data" / "raw" / "gldas"
PLOTS_DIR = PROJECT_ROOT / "plots" / "groundwater"

def process_year(year):
    year_dir = RAW_BASE_DIR / str(year)
    if not year_dir.exists():
        logger.warning(f"No data for year {year}")
        return None

    nc_files = sorted(list(year_dir.glob("*.nc4")))
    if not nc_files:
        return None

    logger.info(f"Processing GLDAS for year {year} ({len(nc_files)} months)...")
    
    # Load all months
    datasets = []
    for f in nc_files:
        ds_tmp = xr.open_dataset(f)
        # RootMoist_inst (0-100cm) and SWE_inst (Snow Water Equivalent)
        vars_to_keep = ['RootMoist_inst', 'SWE_inst']
        datasets.append(ds_tmp[vars_to_keep])
        ds_tmp.close()

    ds_year = xr.concat(datasets, dim='time').mean(dim='time')
    
    # Standardize names for WorldTensor
    ds_year = ds_year.rename({
        'RootMoist_inst': 'root_zone_sm',
        'SWE_inst': 'gldas_snow_water_equivalent'
    })

    # Regrid to master grid template
    ds_out_sm = regrid_raster(ds_year['root_zone_sm'], year, var_name="root_zone_sm")
    ds_out_swe = regrid_raster(ds_year['gldas_snow_water_equivalent'], year, var_name="gldas_snow_water_equivalent")
    
    # Update metadata
    ds_out_sm["root_zone_sm"].attrs = {
        "units": "kg m-2",
        "long_name": "Root zone soil moisture (0-100cm)",
        "description": f"Annual mean root zone soil moisture from GLDAS v2.1 for year {year}"
    }
    ds_out_swe["gldas_snow_water_equivalent"].attrs = {
        "units": "kg m-2",
        "long_name": "GLDAS snow water equivalent",
        "description": f"Annual mean snow water equivalent from GLDAS v2.1 for year {year}"
    }

    # Save to final
    save_annual_variable(ds_out_sm, "root_zone_sm", year)
    save_annual_variable(ds_out_swe, "gldas_snow_water_equivalent", year)

    # Plot maps for benchmark years
    if year % 5 == 0 or year == 2022:
        plot_global_map(ds_out_sm["root_zone_sm"], f"GLDAS Root Zone SM — {year}", PLOTS_DIR / f"sm_{year}.png", cmap="YlGnBu")
        plot_global_map(ds_out_swe["gldas_snow_water_equivalent"], f"GLDAS Snow Water — {year}", PLOTS_DIR / f"swe_{year}.png", cmap="Blues")

    return {
        "sm": float(ds_out_sm["root_zone_sm"].mean()),
        "swe": float(ds_out_swe["gldas_snow_water_equivalent"].mean())
    }

@click.command()
@click.option("--start-year", type=int, default=2000)
@click.option("--end-year", type=int, default=2023)
def main(start_year, end_year):
    import gc
    ts_years = []
    ts_sm = []
    ts_swe = []

    # Track existing processed years from data/final to avoid re-running if memory failed
    final_dir_sm = Path("data/final/hydrology/root_zone_sm")
    final_dir_swe = Path("data/final/hydrology/gldas_snow_water_equivalent")

    for year in range(start_year, end_year + 1):
        res = process_year(year)
        if res:
            ts_years.append(year)
            ts_sm.append(res["sm"])
            ts_swe.append(res["swe"])

        # Explicit cleanup after each year
        gc.collect()

    if ts_years:
        plot_time_series(ts_years, ts_sm, "Global Mean Root Zone SM (GLDAS)", "kg m-2", PLOTS_DIR / "timeseries_sm.png")
        plot_time_series(ts_years, ts_swe, "Global Mean Snow Water Equivalent (GLDAS)", "kg m-2", PLOTS_DIR / "timeseries_swe.png")


if __name__ == "__main__":
    main()
