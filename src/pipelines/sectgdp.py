"""Full SectGDP30 pipeline: download → regrid → interpolate → visualize.

Downloads global sectoral GDP (2010, 2015, 2020), regrids to 0.25° master grid,
interpolates linearly across years to fill gaps, and generates spatial maps.

Usage:
    python -m src.pipelines.sectgdp
    python -m src.pipelines.sectgdp --overwrite
    python -m src.pipelines.sectgdp --no-cleanup
"""

import gc
from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from src.download.sectgdp import download_sectgdp, load_sectgdp_config, DEFAULT_RAW_DIR as RAW_DIR
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

logger = get_logger("pipeline.sectgdp")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "human_systems"
PLOTS_DIR = PROJECT_ROOT / "plots" / "sectgdp"


def process_sectgdp(overwrite: bool = False, cleanup: bool = True):
    """Run the full SectGDP30 pipeline."""
    config = load_sectgdp_config()
    variables = config["variables"]
    anchor_years = config["anchor_years"]
    t_range = config["temporal_range"]
    all_years = list(range(t_range[0], t_range[1] + 1))
    
    # 1. Download anchor years
    raw_files = download_sectgdp(overwrite=overwrite)
    if not raw_files:
        logger.error("No raw files available for processing.")
        return

    # 2. Process each variable (Aggregate across years)
    for var_key, var_info in variables.items():
        logger.info("Processing SectGDP30 variable: %s", var_key)
        
        # Step A: Regrid anchor years
        anchor_datasets = {}
        for year in anchor_years:
            fname = var_info["filename_pattern"].format(year=year)
            raw_path = RAW_DIR / fname
            
            if not raw_path.exists():
                logger.warning("Anchor year file not found: %s", raw_path)
                continue
                
            # Load and regrid
            logger.info("  Regridding anchor year %d...", year)
            da = load_raster(raw_path)
            da.attrs["units"] = var_info["units"]
            da.attrs["long_name"] = var_info["long_name"]
            da.name = var_key
            
            ds_regridded = regrid_raster(da, year=year, var_name=var_key)
            anchor_datasets[year] = ds_regridded[var_key]
            
        if not anchor_datasets:
            continue

        # Step B: Interpolate across all years
        logger.info("  Interpolating across years %d-%d...", t_range[0], t_range[1])
        
        # Combine anchor years into a single DataArray with a time dimension
        times = [np.datetime64(f"{y}-01-01") for y in anchor_years]
        combined = xr.concat(
            [anchor_datasets[y] for y in anchor_years],
            dim=xr.DataArray(times, dims="time", name="time")
        )
        
        # Reindex to all years and interpolate
        target_times = [np.datetime64(f"{y}-01-01") for y in all_years]
        full_series = combined.reindex(time=target_times).interpolate_na(
            dim="time", method="linear"
        )
        # Restore attributes
        full_series.attrs = combined.attrs
        
        # Step C: Save each year to NetCDF
        out_var_dir = FINAL_DIR / var_key
        out_var_dir.mkdir(parents=True, exist_ok=True)
        
        for i, year in enumerate(all_years):
            out_path = out_var_dir / f"{year}.nc"
            if out_path.exists() and not overwrite:
                continue
                
            # Extract single year, keeping attributes
            ds_year = full_series.isel(time=i).to_dataset(name=var_key)
            
            encoding = {var_key: {"zlib": True, "complevel": 4, "dtype": "float32"}}
            ds_year.to_netcdf(out_path, encoding=encoding)
            
        logger.info("  Saved all years to %s", out_var_dir)

        # 3. Visualize
        # Map for 2020 (latest anchor)
        ds_2020 = xr.open_dataset(out_var_dir / "2020.nc")
        map_out = PLOTS_DIR / "maps" / f"{var_key}_2020.png"
        if plot_variable is not None:
            plot_variable(
            ds_2020, 
            var_name=var_key, 
            year=2020, 
            output_path=map_out,
            cmap="inferno"
        )
        ds_2020.close()
        
        # Temporal trend
        trend_out = PLOTS_DIR / "trends" / f"{var_key}_trend.png"
        trend_out.parent.mkdir(parents=True, exist_ok=True)
        # Point to Economy/ directory
        if plot_temporal_trend is not None:
            plot_temporal_trend(var_key, data_dir=FINAL_DIR, output_path=trend_out)
        
        gc.collect()

    # 4. Cleanup raw files
    if cleanup:
        logger.info("Cleaning up raw SectGDP30 files...")
        for p in RAW_DIR.glob("*"):
            if p.is_file():
                p.unlink()
        if RAW_DIR.exists() and not any(RAW_DIR.iterdir()):
            RAW_DIR.rmdir()


@click.command()
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
@click.option("--no-cleanup", is_flag=True, help="Keep raw files after processing.")
def main(overwrite, no_cleanup):
    """SectGDP30 Pipeline: Download → Regrid → Interpolate → Visualize."""
    process_sectgdp(overwrite=overwrite, cleanup=not no_cleanup)
    click.echo("\nSectGDP30 pipeline complete.")


if __name__ == "__main__":
    main()
