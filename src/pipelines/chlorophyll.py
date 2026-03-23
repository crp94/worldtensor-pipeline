"""Full Chlorophyll-a pipeline: download → process → visualize.

Includes robust meridian padding and 4-panel plots (mean, std, max, min).
Source: MODIS-Aqua L3m Monthly 4km Chlorophyll-a
"""

import gc
import os
import zipfile
import tempfile
import shutil
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from src.utils import get_logger, add_cyclic_point_xr
from src.grid import make_template
from src.year_policy import resolve_year_list

logger = get_logger("pipeline.chlorophyll")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "chlorophyll.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chlorophyll"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "ocean"
PLOTS_DIR = PROJECT_ROOT / "plots" / "chlorophyll"

STATS = ("mean", "std", "max", "min")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def outputs_exist(year: int) -> bool:
    return all((FINAL_DIR / f"chl_{s}" / f"{year}.nc").exists() for s in STATS)


def read_means_from_outputs(year: int) -> dict | None:
    means = {}
    for s in STATS:
        p = FINAL_DIR / f"chl_{s}" / f"{year}.nc"
        if not p.exists(): return None
        ds = xr.open_dataset(p)
        means[s] = float(ds[list(ds.data_vars)[0]].mean(skipna=True))
        ds.close()
    return means


def _save_derived(data, lat, lon, stat_name, year):
    folder = f"chl_{stat_name}"
    out = FINAL_DIR / folder / f"{year}.nc"
    if out.exists(): out.unlink() # Force fresh save
    out.parent.mkdir(parents=True, exist_ok=True)
    
    if data.ndim > 2: data = np.squeeze(data)
    time_coords = [np.datetime64(f"{year}-01-01")]
    
    ds = xr.Dataset(
        {folder: (["lat", "lon"], data.astype(np.float32), 
                  {"units": "mg m-3", "long_name": f"Chlorophyll-a concentration ({stat_name})"})},
        coords={
            "lat": ("lat", lat, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", lon, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "title": f"WorldTensor Chlorophyll-a {stat_name}", 
            "source": "MODIS-Aqua L3m Monthly 4km", 
            "year": year,
            "Conventions": "CF-1.8"
        }
    )
    ds = ds.expand_dims(time=time_coords)
    ds.to_netcdf(out, encoding={folder: {"zlib": True, "complevel": 4, "dtype": "float32"}})
    return out


def process_year(year: int, overwrite: bool = False):
    """Aggregate monthly 4km CHL to yearly 0.25° grid mean/std/max/min."""
    out_path_check = FINAL_DIR / "chl_mean" / f"{year}.nc"
    if out_path_check.exists() and not overwrite:
        return read_means_from_outputs(year)

    monthly_files = list((RAW_DIR / str(year)).glob("*.nc"))
    if not monthly_files:
        logger.warning("No raw files found for %d", year)
        return None

    template = make_template(year)
    monthly_data = []
    
    for f in sorted(monthly_files):
        try:
            ds = xr.open_dataset(f)
            da = ds["chlor_a"]
            
            # Standardize coordinates
            da = da.rename({"lat": "lat", "lon": "lon"})
            da = da.sortby("lat")
            da = da.assign_coords(lon=(da.lon.values % 360))
            da = da.sortby("lon")
            
            # Robust Padding: Add 1 degree of overlap on each side
            da_first = da.isel(lon=slice(0, 30)).copy()
            da_first = da_first.assign_coords(lon=da_first.lon + 360)
            da_last = da.isel(lon=slice(-30, None)).copy()
            da_last = da_last.assign_coords(lon=da_last.lon - 360)
            da_padded = xr.concat([da_last, da, da_first], dim="lon")
            
            # Interp to target grid
            regridded = da_padded.interp(
                lat=template.lat, 
                lon=template.lon, 
                method="linear"
            ).values
            monthly_data.append(regridded)
            ds.close()
        except Exception as e:
            logger.error("Failed to process %s: %s", f.name, e)

    if not monthly_data:
        return None

    stacked = np.stack(monthly_data)
    lat, lon = template.lat.values, template.lon.values
    
    res = {}
    stats_map = {
        "mean": np.nanmean(stacked, axis=0),
        "std": np.nanstd(stacked, axis=0),
        "max": np.nanmax(stacked, axis=0),
        "min": np.nanmin(stacked, axis=0)
    }
    
    for s in STATS:
        data = stats_map[s]
        _save_derived(data, lat, lon, s, year)
        res[s] = float(np.nanmean(data))
        
    return res


def plot_spatial_map(year: int, overwrite: bool = False):
    """Plot 4-panel Robinson map for CHL stats."""
    out = PLOTS_DIR / "maps" / f"{year}.png"
    if out.exists() and not overwrite:
        return
    
    out.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 10), subplot_kw={"projection": ccrs.Robinson()})
    
    for idx, stat in enumerate(STATS):
        ax = axes.flat[idx]
        nc = FINAL_DIR / f"chl_{stat}" / f"{year}.nc"
        if not nc.exists(): continue
        
        ds = xr.open_dataset(nc)
        da = ds[list(ds.data_vars)[0]]
        
        # Fix meridian seam robustly
        data_cyclic, lon_cyclic, lat_values = add_cyclic_point_xr(da)
        
        cmap = "plasma" if stat == "std" else "YlGn"
        
        if stat == "std":
            vmax = float(np.nanpercentile(data_cyclic, 99))
            norm = mcolors.PowerNorm(gamma=0.5, vmin=0, vmax=vmax)
        else:
            norm = mcolors.LogNorm(vmin=0.01, vmax=10)
            
        im = ax.pcolormesh(
            lon_cyclic, lat_values, data_cyclic,
            transform=ccrs.PlateCarree(),
            norm=norm,
            cmap=cmap, 
            shading='auto'
        )
        
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, aspect=40, shrink=0.8,
                     label=f"mg m-3")
        
        ax.coastlines(linewidth=0.4, color="gray")
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.set_title(f"CHL {stat.upper()}", fontsize=11)
        ds.close()
        
    plt.suptitle(f"MODIS-Aqua Chlorophyll-a — {year}", fontsize=14)
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_series(years, all_means):
    out = PLOTS_DIR / "timeseries" / "chlor_a.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(len(STATS), 1, figsize=(14, 4 * len(STATS)), sharex=True)
    colors = {"mean": "#1f77b4", "std": "#ff7f0e", "max": "#d62728", "min": "#2ca02c"}
    
    for ax, stat in zip(axes, STATS):
        if stat in all_means and all_means[stat]:
            ax.plot(years, all_means[stat], marker=".", markersize=3, color=colors[stat])
            ax.set_ylabel(f"{stat.upper()} (mg m-3)")
            ax.set_title(f"Chlorophyll-a — {stat.upper()}")
            ax.grid(True, alpha=0.3)
            
    axes[-1].set_xlabel("Year")
    plt.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option("--years", "-y", multiple=True, type=int)
@click.option("--all", "run_all", is_flag=True)
@click.option("--overwrite", is_flag=True)
def main(years, run_all, overwrite):
    config = load_config()
    t_range = config["temporal_range"]
    
    if years or run_all:
        year_list = resolve_year_list(
            years,
            default_start=t_range[0],
            default_end=t_range[1],
            start_year=t_range[0] if run_all and not years else None,
            end_year=t_range[1] if run_all and not years else None,
            label="chlorophyll pipeline years",
        )
    else:
        click.echo("Specify -y or --all")
        return

    ts_years, ts_means = [], {s: [] for s in STATS}
    
    for year in year_list:
        click.echo(f"\n--- Year {year} ---")
        from src.download.chlorophyll import DEFAULT_RAW_DIR as R, download_chlorophyll
        download_chlorophyll([year], raw_dir=R)
            
        means = process_year(year, overwrite)
        if means:
            ts_years.append(year)
            for s in STATS:
                ts_means[s].append(means[s])
            plot_spatial_map(year, overwrite)
            
    if ts_years:
        plot_time_series(ts_years, ts_means)
            
    click.echo("\nPipeline complete.")


if __name__ == "__main__":
    main()
