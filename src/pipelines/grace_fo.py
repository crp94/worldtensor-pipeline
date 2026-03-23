"""GRACE / GRACE-FO Pipeline.

Processes 0.5° monthly TWS anomalies to 0.25° annual aggregates (mean, max, min, std).
"""

import click
import numpy as np
import xarray as xr
from pathlib import Path
import yaml

from src.utils import get_logger, save_annual_variable, plot_global_map, plot_time_series
from src.grid import make_template
from src.processing.raster_to_grid import regrid_raster, load_raster

logger = get_logger("pipeline.grace")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "grace"
PLOTS_DIR = PROJECT_ROOT / "plots" / "grace"

def process_grace():
    # 1. Load the single large Mascon file
    nc_files = list(RAW_DIR.glob("*.nc"))
    if not nc_files:
        logger.error(f"Raw GRACE file not found in {RAW_DIR}")
        return

    ds = xr.open_dataset(nc_files[0])
    
    # 2. Extract years and identify missing ones
    available_years = sorted(list(np.unique(ds.time.dt.year.values)))
    ts_data = {"mean": [], "max": [], "min": [], "std": []}
    ts_years = []
    
    # We will store annual maps in a dict to handle interpolation
    annual_maps = {stat: {} for stat in ts_data.keys()}

    for year in available_years:
        # Calculate annual aggregates
        ds_year = ds.sel(time=slice(f"{year}-01-01", f"{year}-12-31"))
        if ds_year.time.size < 6:
            continue
            
        ts_years.append(int(year))
        for stat_name in ts_data.keys():
            da_stat = getattr(ds_year['lwe_thickness'], stat_name)(dim='time')
            # Regrid and store
            var_name = f"lwe_thickness_{stat_name}"
            ds_out = regrid_raster(da_stat, year, var_name=var_name, method="linear")
            annual_maps[stat_name][int(year)] = ds_out[var_name]
            
    # 3. Interpolate missing years
    full_range = range(min(ts_years), max(ts_years) + 1)
    final_ts_years = list(full_range)
    final_ts_data = {stat: [] for stat in ts_data.keys()}

    for year in full_range:
        logger.info(f"Finalizing GRACE for year {year}...")
        for stat_name in ts_data.keys():
            if year in annual_maps[stat_name]:
                da_year = annual_maps[stat_name][year]
            else:
                logger.info(f"  Interpolating missing year {year} for {stat_name}")
                # Find nearest neighbors for linear interpolation: (y - y0) / (y1 - y0)
                y0 = max([y for y in ts_years if y < year])
                y1 = min([y for y in ts_years if y > year])
                da0 = annual_maps[stat_name][y0]
                da1 = annual_maps[stat_name][y1]
                weight = (year - y0) / (y1 - y0)
                da_year = da0 * (1 - weight) + da1 * weight
                da_year.attrs = da0.attrs
                da_year.attrs["description"] = f"Linearly interpolated from {y0} and {y1}"

            # Save to final
            var_name = f"lwe_thickness_{stat_name}"
            ds_to_save = da_year.to_dataset(name=var_name)
            save_annual_variable(ds_to_save, var_name, year)
            
            # Accumulate for time series plot
            final_ts_data[stat_name].append(float(da_year.mean()))

            # Plot spatial map for 'mean' for benchmark or interpolated years
            if stat_name == "mean" and (year % 5 == 0 or year not in ts_years or year == full_range[-1]):
                plot_global_map(
                    da_year,
                    f"GRACE TWS Anomaly (Annual Mean) — {year} {'(Interp)' if year not in ts_years else ''}",
                    PLOTS_DIR / f"grace_tws_mean_{year}.png",
                    cmap="RdBu"
                )
            
    # 5. Plot time series for all stats
    colors = {"mean": "#1f77b4", "max": "#d62728", "min": "#ff7f0e", "std": "#2ca02c"}
    for stat_name, values in final_ts_data.items():
        plot_time_series(
            final_ts_years, values,
            f"Global Summary: GRACE TWS Anomaly ({stat_name})",
            "LWE Thickness (cm)",
            PLOTS_DIR / f"timeseries_grace_{stat_name}.png",
            color=colors[stat_name]
        )

    ds.close()
    logger.info("GRACE Pipeline complete.")

if __name__ == "__main__":
    process_grace()
