"""Full FLDAS vegetation class pipeline: download → one-hot encode → visualize.

Data source
-----------
Website : https://ldas.gsfc.nasa.gov/fldas/vegetation-class
Citation: McNally et al. (2017), Scientific Data 4:170012.

Downloads the FLDAS/Noah IGBP-modified MODIS vegetation classification,
converts each of 18 land surface types to a binary (one-hot) layer on the
0.25° master grid, and generates Robinson projection maps.

Pipeline steps:
    1. Download: Fetches FLDAS-global_domveg.nc4 (~5 MB, no auth)
    2. Process:  Creates 18 one-hot NetCDF layers via nearest-neighbor regrid
    3. Visualize: Robinson projection maps with binary colormap

Output directories:
    - NetCDF: data/final/static/vegclass/{var_name}.nc
    - Plots:  plots/fldas_vegclass/{var_name}.png

Usage:
    python -m src.pipelines.fldas_vegclass --all
    python -m src.pipelines.fldas_vegclass --all --skip-download
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

from src.download.fldas_vegclass import download_vegclass
from src.processing.fldas_vegclass_to_static import (
    process_vegclass, FINAL_DIR, load_config,
)
from src.utils import get_logger

logger = get_logger("pipeline.fldas_vegclass")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots" / "fldas_vegclass"

# Binary colormap: light gray (absent) → forest green (present)
BINARY_CMAP = mcolors.ListedColormap(["#f0ede4", "#2d7d46"])
BINARY_NORM = mcolors.BoundaryNorm([0, 0.5, 1], BINARY_CMAP.N)


def plot_onehot_map(var_name: str, long_name: str):
    """Plot a Robinson projection map for a single one-hot vegetation layer."""
    out = PLOTS_DIR / f"{var_name}.png"
    if out.exists():
        return

    nc_path = FINAL_DIR / f"{var_name}.nc"
    if not nc_path.exists():
        return

    ds = xr.open_dataset(nc_path)
    da = ds[var_name]

    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={"projection": ccrs.Robinson()},
    )

    im = da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=BINARY_CMAP,
        norm=BINARY_NORM,
        add_colorbar=False,
    )

    cbar = plt.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.05, aspect=40, shrink=0.5,
        ticks=[0.25, 0.75],
    )
    cbar.ax.set_xticklabels(["Absent", "Present"])

    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.set_title(f"Vegetation: {long_name}", fontsize=12)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    logger.info("Saved map → %s", out)


@click.command()
@click.option("--variables", "-v", multiple=True,
              help="Variable(s) (e.g. vegclass_ebf vegclass_cropland).")
@click.option("--all", "run_all", is_flag=True, help="Process all vegetation types.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
def main(variables, run_all, skip_download, overwrite):
    """FLDAS vegetation class pipeline: download → one-hot encode → visualize."""
    config = load_config()
    all_vars = config["variables"]

    if not variables and not run_all:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    var_list = list(variables) if variables else list(all_vars.keys())
    click.echo(f"Pipeline: {len(var_list)} vegetation types")

    # ── Step 1: Download ─────────────────────────────────────────────────
    if not skip_download:
        click.echo("\n── Download ──")
        download_vegclass(overwrite=overwrite)
    else:
        click.echo("\n── Download (skipped) ──")

    # ── Step 2: One-hot encode + regrid ──────────────────────────────────
    click.echo("\n── Processing ──")
    n = process_vegclass(variables=var_list, overwrite=overwrite)
    click.echo(f"  Processed {n} files")

    # ── Step 3: Visualize ────────────────────────────────────────────────
    click.echo("\n── Visualization ──")
    for var_name in var_list:
        var_info = all_vars.get(var_name)
        if var_info is None:
            continue
        try:
            plot_onehot_map(var_name, var_info["long_name"])
        except Exception as e:
            logger.warning("Plot failed %s: %s", var_name, e)
        gc.collect()
        plt.close("all")

    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
