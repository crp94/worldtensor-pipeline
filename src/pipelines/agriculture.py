"""Full GGCP10 Agriculture pipeline: download → regrid → visualize.

Data source
-----------
He et al. (2024), Scientific Data

Regrids 10km yearly GeoTIFFs to 0.25° WorldTensor grid.
Focuses on production (tonnes) for major staples 2010–2020.

Usage:
    python -m src.pipelines.agriculture --all
"""

import gc
import re
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from src.download.agriculture import download_agriculture, load_config as load_agri_config
from src.grid import make_template
from src.utils import get_logger, add_cyclic_point_xr
from src.processing.raster_to_grid import regrid_raster, load_raster

logger = get_logger("pipeline.agriculture")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "agriculture"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "agriculture"
PLOTS_DIR = PROJECT_ROOT / "plots" / "agriculture"


def plot_agriculture(var_code: str, year: int, long_name: str, units: str, cmap: str):
    """Plot a Robinson projection spatial map for crop production."""
    out = PLOTS_DIR / "maps" / var_code / f"{year}.png"
    out.parent.mkdir(parents=True, exist_ok=True)

    nc_path = FINAL_DIR / var_code / f"{year}.nc"
    if not nc_path.exists(): return

    ds = xr.open_dataset(nc_path)
    da = ds[var_code]

    # Robust scale
    nonzero = da.values[np.isfinite(da.values) & (da.values > 0)]
    if nonzero.size > 0:
        vmax = float(np.percentile(nonzero, 99))
        vmin = max(0.1, float(np.percentile(nonzero, 1)))
    else:
        vmax, vmin = 100.0, 0.1

    fig, ax = plt.subplots(figsize=(14, 7), subplot_kw={"projection": ccrs.Robinson()})
    
    # Fix meridian seam
    data_cyclic, lon_cyclic, lat_values = add_cyclic_point_xr(da)

    im = ax.pcolormesh(
        lon_cyclic, lat_values, data_cyclic,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=mcolors.LogNorm(vmin=vmin, vmax=vmax),
        shading='auto'
    )
    
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, aspect=50, shrink=0.8,
                 label=f"{units}")
    
    ax.coastlines(linewidth=0.5, color="gray")
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_title(f"GGCP10 Agriculture — {long_name} — {year}", fontsize=14)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()


def plot_time_series(var_code: str, long_name: str, units: str, years: list[int], means: list[float]):
    """Plot global mean production over time."""
    out = PLOTS_DIR / "timeseries" / f"{var_code}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(years, means, marker=".", color="#1f77b4", linewidth=1.5)
    ax.set_ylabel(f"Global mean ({units})")
    ax.set_title(f"GGCP10 {long_name} — Global Trend (2010-2020)")
    ax.grid(True, alpha=0.3)
    ax.set_xlabel("Year")
    
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option("--all", "run_all", is_flag=True)
@click.option("--skip-download", is_flag=True)
@click.option("--overwrite", is_flag=True)
def main(run_all, skip_download, overwrite):
    config = load_agri_config()
    all_vars = config["variables"]

    if not run_all:
        click.echo("Specify --all")
        return

    # 1. Download
    if not skip_download:
        download_agriculture(overwrite=overwrite)

    # 2. Process
    FINAL_DIR.mkdir(parents=True, exist_ok=True)
    years = list(range(2010, 2021))
    
    for var_code, info in all_vars.items():
        logger.info("Processing %s...", var_code)
        ts_years, ts_means = [], []
        crop = info["crop_name"]
        
        for year in tqdm(years, desc=var_code):
            out_path = FINAL_DIR / var_code / f"{year}.nc"
            
            if out_path.exists() and not overwrite:
                try:
                    ds_out = xr.open_dataset(out_path)
                    ts_years.append(year)
                    ts_means.append(float(ds_out[var_code].mean(skipna=True)))
                    ds_out.close()
                    plot_agriculture(var_code, year, info["long_name"], info["units"], info["cmap"])
                    continue
                except Exception:
                    pass

            # GGCP10 GeoTIFF names: GGCP10_Production_{year}_{Crop}.tif
            raw_name = f"GGCP10_Production_{year}_{crop}.tif"
            raw_path = list(RAW_DIR.rglob(raw_name))
            
            if not raw_path:
                logger.warning("Raw file not found: %s", raw_name)
                continue
            
            raw_path = raw_path[0]
            template = make_template(year)
            
            try:
                # Load and regrid
                da = load_raster(raw_path)
                ds_regridded = regrid_raster(da, year=year, var_name=var_code)
                data = ds_regridded[var_code].values
                data[data < 0] = 0
                
                ds_final = xr.Dataset(
                    {var_code: (["lat", "lon"], data.astype(np.float32), 
                                {"units": info["units"], "long_name": info["long_name"]})},
                    coords={"lat": ds_regridded.lat, "lon": ds_regridded.lon, "time": [np.datetime64(f"{year}-01-01")]},
                    attrs={"title": f"WorldTensor Crop Production {var_code}", "source": "GGCP10 He et al. 2024"}
                )
                
                out_path.parent.mkdir(parents=True, exist_ok=True)
                ds_final.to_netcdf(out_path, encoding={var_code: {"zlib": True, "complevel": 4}})
                
                ts_years.append(year)
                ts_means.append(float(ds_final[var_code].mean(skipna=True)))
                
                plot_agriculture(var_code, year, info["long_name"], info["units"], info["cmap"])
                
            except Exception as e:
                logger.error("Failed processing %s/%d: %s", var_code, year, e)

        if ts_years:
            plot_time_series(var_code, info["long_name"], info["units"], ts_years, ts_means)

    click.echo("\nAgriculture pipeline complete.")


if __name__ == "__main__":
    main()
