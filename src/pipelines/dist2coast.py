"""Full distance-to-coast pipeline: download → process → visualize.

Data source
-----------
Producer : NASA Goddard Space Flight Center (GSFC), Ocean Biology Processing
           Group (OBPG)
ERDDAP   : https://upwell.pfeg.noaa.gov/erddap/griddap/dist2coast_1deg.html
Docs     : https://oceancolor.gsfc.nasa.gov/resources/docs/distfromcoast/
Citation : NASA OBPG (2012). Distance to nearest coastline.

Pipeline steps:
    1. Download: Fetches 0.25° subsampled NetCDF from ERDDAP (~2 MB)
    2. Process:  Shifts longitude to 0..360, interpolates to master grid
    3. Visualize: Robinson projection map with diverging colormap
                  (brown = deep inland, white = coast, blue = open ocean)

The output is a single signed distance layer (km):
    - Positive values = over ocean (distance to nearest land)
    - Negative values = over land  (distance to nearest coast)
    - Zero = on the coastline

Usage:
    python -m src.pipelines.dist2coast
    python -m src.pipelines.dist2coast --skip-download
    python -m src.pipelines.dist2coast --overwrite
"""

import gc
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.download.dist2coast import download_dist2coast
from src.processing.dist2coast_to_static import process_dist2coast, FINAL_DIR
from src.utils import get_logger

logger = get_logger("pipeline.dist2coast")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots" / "dist2coast"


def plot_dist2coast():
    """Plot a Robinson projection map of distance to coastline.

    Uses a diverging colormap centered at zero (coastline):
        - Brown shades = land (negative distance, inland)
        - Blue shades  = ocean (positive distance, offshore)
        - White        = near coastline (0 km)

    The color scale is symmetric and clipped at the 99th percentile
    of absolute values to avoid outliers (remote Pacific / deep Sahara)
    dominating the scale.
    """
    var_name = "dist_to_coast"
    out = PLOTS_DIR / f"{var_name}.png"
    if out.exists():
        return

    nc_path = FINAL_DIR / f"{var_name}.nc"
    if not nc_path.exists():
        return

    ds = xr.open_dataset(nc_path)
    da = ds[var_name]

    # Symmetric color range: clip at 99th percentile of |distance|
    abs_vals = np.abs(da.values[np.isfinite(da.values)])
    vmax = float(np.percentile(abs_vals, 99))

    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={"projection": ccrs.Robinson()},
    )

    # Diverging colormap: brown (land/inland) → white (coast) → blue (ocean)
    im = da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="BrBG",
        vmin=-vmax,
        vmax=vmax,
        add_colorbar=False,
    )
    plt.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.05, aspect=40, shrink=0.8,
        label="Distance to coastline (km) — negative = land, positive = ocean",
    )
    ax.coastlines(linewidth=0.5)
    ax.set_global()
    ax.set_title("Distance to Nearest Coastline", fontsize=12)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    logger.info("Saved map → %s", out)


@click.command()
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
def main(skip_download, overwrite):
    """Distance-to-coast pipeline: download → process → visualize."""
    click.echo("Pipeline: distance to nearest coastline")

    # ── Step 1: Download ─────────────────────────────────────────────────
    if not skip_download:
        click.echo("\n── Download ──")
        download_dist2coast(overwrite=overwrite)
    else:
        click.echo("\n── Download (skipped) ──")

    # ── Step 2: Process (regrid to master) ───────────────────────────────
    click.echo("\n── Processing ──")
    result = process_dist2coast(overwrite=overwrite)
    if result:
        click.echo(f"  Output: {result}")
    else:
        click.echo("  Skipped (already exists)")

    # ── Step 3: Visualize ────────────────────────────────────────────────
    click.echo("\n── Visualization ──")
    try:
        plot_dist2coast()
    except Exception as e:
        logger.warning("Plot failed: %s", e)

    gc.collect()
    plt.close("all")

    click.echo(f"\nPipeline complete. Output in {FINAL_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
