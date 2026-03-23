"""Ecosystem Service Value (ESV) Pipeline.

Calculates annual ESV by combining LUH3 land use fractions and WAD2M wetlands
with monetary coefficients from Costanza et al. (2014).
"""

import click
import numpy as np
import xarray as xr
from pathlib import Path
import yaml

from src.utils import (
    calculate_cell_area_ha,
    get_logger,
    plot_global_map,
    plot_time_series,
    save_annual_variable,
)
from src.grid import make_template

logger = get_logger("pipeline.esv")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
LUH3_DIR = PROJECT_ROOT / "data" / "final" / "land_use" / "states"
WETLANDS_DIR = PROJECT_ROOT / "data" / "final" / "hydrology" / "inundation_fraction_mean"
PLOTS_DIR = PROJECT_ROOT / "plots" / "economy"

# Coefficients (USD/ha/yr, 2011 values from Costanza et al. 2014)
# We map LUH3 variables to these major biomes
COEFFICIENTS = {
    "forest": 3800.0,    # primf, secdf, secmb (tropical/temperate forest)
    "grassland": 4166.0, # primn, secdn, range, pastr (grassland/pasture)
    "cropland": 5567.0,  # c3ann, c4ann, c3per, c4per, c3nfx
    "urban": 6661.0,     # urban
    "wetland": 25682.0   # inundation_fraction_mean (WAD2M)
}

def calculate_esv(year):
    logger.info(f"Calculating ESV for {year}...")
    
    # 1. Load LUH3 fractions
    # Major categories
    forest_vars = ["primf", "secdf", "secmb"]
    grass_vars = ["primn", "secdn", "range", "pastr"]
    crop_vars = ["c3ann", "c4ann", "c3per", "c4per", "c3nfx"]
    
    fractions = {}
    for group, vars in [("forest", forest_vars), ("grass", grass_vars), ("crop", crop_vars), ("urban", ["urban"])]:
        group_sum = None
        for v in vars:
            p = LUH3_DIR / v / f"{year}.nc"
            if not p.exists(): continue
            ds_v = xr.open_dataset(p)
            # Use .values to avoid xarray ambiguity in arithmetic
            val = ds_v[v].values
            group_sum = val if group_sum is None else group_sum + val
            ds_v.close()
        fractions[group] = group_sum

    # 2. Load Wetlands (WAD2M)
    p_wet = WETLANDS_DIR / f"{year}.nc"
    if p_wet.exists():
        ds_wet = xr.open_dataset(p_wet)
        # The variable in our processed WAD2M is 'inundation_fraction_mean'
        fractions["wetland"] = ds_wet["inundation_fraction_mean"].values
        ds_wet.close()
    else:
        logger.warning(f"  Wetlands missing for {year}")
        fractions["wetland"] = 0.0

    # 3. Calculate ESV Density (USD/ha)
    # ESV = sum(Fraction_i * Coefficient_i)
    # Initialize with zeros of the master shape
    esv_density = np.zeros((721, 1440), dtype=np.float32)
    for group, coeff in [("forest", COEFFICIENTS["forest"]), 
                         ("grass", COEFFICIENTS["grassland"]), 
                         ("crop", COEFFICIENTS["cropland"]), 
                         ("urban", COEFFICIENTS["urban"]),
                         ("wetland", COEFFICIENTS["wetland"])]:
        if group in fractions and fractions[group] is not None:
            esv_density += fractions[group] * coeff

    # 4. Convert to Total Value (USD/cell)
    template = make_template(year)
    cell_area_ha = calculate_cell_area_ha(template.lat.values, template.lon.values)
    esv_total = esv_density * cell_area_ha
    
    ds_out = template.copy()
    ds_out["esv_total"] = (("lat", "lon"), esv_total.astype(np.float32))
    ds_out["esv_total"].attrs = {
        "units": "USD",
        "long_name": "Total Ecosystem Service Value",
        "description": f"Annual ESV calculated from LUH3 and WAD2M using Costanza et al. (2014) coefficients for year {year}"
    }
    
    # 5. Save
    save_annual_variable(ds_out, "esv_total", year)
    
    # 6. Plot
    if year % 5 == 0 or year == 2020:
        plot_global_map(
            ds_out["esv_total"],
            f"Global Ecosystem Service Value — {year}",
            PLOTS_DIR / f"esv_{year}.png",
            cmap="YlGn",
            force_log=True
        )
        
    return float(ds_out["esv_total"].mean())

@click.command()
def main():
    # LUH3 and Wetlands overlap period
    years = range(2000, 2021)
    ts_years = []
    ts_means = []
    
    for year in years:
        try:
            m = calculate_esv(year)
            ts_years.append(year)
            ts_means.append(m)
        except Exception as e:
            logger.exception(f"Failed {year}: {e}")
            
    if ts_years:
        plot_time_series(
            ts_years, ts_means,
            "Global Average Ecosystem Service Value per Cell",
            "USD",
            PLOTS_DIR / "timeseries_esv.png",
            color="#2ca02c"
        )

if __name__ == "__main__":
    main()
