"""Market Accessibility Pipeline (Weiss et al. 2018).

Regrids the 2015 1 km travel-time surface to the WorldTensor 0.25° grid and
stores it as a static contextual layer.
"""

import numpy as np
from pathlib import Path

from src.processing.raster_to_grid import load_raster, regrid_raster
from src.utils import get_logger, plot_global_map, save_static_variable

logger = get_logger("pipeline.accessibility")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "accessibility"
PLOTS_DIR = PROJECT_ROOT / "plots" / "accessibility"

def process_accessibility(year: int = 2015):
    logger.info("Processing Accessibility static snapshot for %d...", year)
    
    # 1. Load the 1km GeoTIFF
    tif_path = RAW_DIR / "accessibility_2015.tif"
    if not tif_path.exists():
        logger.error(f"GeoTIFF not found in {RAW_DIR}")
        return

    da_raw = load_raster(tif_path)
    
    # Regrid to 0.25°. The helper uses bilinear/linear interpolation, which is
    # the current publication choice for this continuous travel-time surface.
    ds_out = regrid_raster(da_raw, year, var_name="travel_time_to_cities", method="linear")
    ds_out["travel_time_to_cities"] = ds_out["travel_time_to_cities"].astype(np.float32)
    
    # Update metadata
    ds_out["travel_time_to_cities"].attrs = {
        "units": "min",
        "long_name": "Travel time to the nearest city",
        "description": f"Static 2015 travel time to the nearest city aggregated to 0.25° from the Weiss et al. source raster"
    }
    
    # Save as a static contextual layer according to the registry-backed layout.
    save_static_variable(ds_out, "travel_time_to_cities")
    
    # 4. Plot
    plot_global_map(
        ds_out["travel_time_to_cities"],
        "Global Accessibility (Travel Time to Cities) — 2015 static snapshot",
        PLOTS_DIR / "travel_time_to_cities.png",
        cmap="viridis_r",
        force_log=True
    )
    
    return float(ds_out["travel_time_to_cities"].mean())

if __name__ == "__main__":
    process_accessibility()
