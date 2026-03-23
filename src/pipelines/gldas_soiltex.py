"""Full GLDAS soil texture pipeline: download → one-hot encode → visualize.

Downloads the GLDAS/Noah FAO soil texture classification, converts each of
the 12 soil types to a binary (one-hot) layer on the 0.25° master grid, and
generates Robinson projection maps.

The FAO 16-category system classifies soil into texture types based on the
relative proportions of sand, silt, and clay. Each output layer is a binary
mask (1 = this type, 0 = other type, NaN = ocean/no data).

Pipeline steps:
    1. Download: Fetches GLDASp4_soiltexture_025d.nc4 (~150 KB, no auth)
    2. Process:  Creates 12 one-hot NetCDF layers via nearest-neighbor regrid
    3. Visualize: Robinson projection maps showing spatial distribution
                  Uses binary colormap (tan = absent, green = present)

Output directories:
    - NetCDF: data/final/static/soiltex/{var_name}.nc
    - Plots:  plots/gldas_soiltex/{var_name}.png

Usage:
    python -m src.pipelines.gldas_soiltex --all
    python -m src.pipelines.gldas_soiltex --all --skip-download
    python -m src.pipelines.gldas_soiltex --variables soiltex_sand soiltex_clay
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

from src.download.gldas_soiltex import download_soiltex, DEFAULT_RAW_DIR as RAW_DIR
from src.processing.gldas_soiltex_to_static import (
    process_soiltex, FINAL_DIR, load_config,
)
from src.utils import get_logger

logger = get_logger("pipeline.gldas_soiltex")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots" / "gldas_soiltex"

# Binary colormap for one-hot layers: tan (0) → green (1)
BINARY_CMAP = mcolors.ListedColormap(["#f5f0e1", "#2d8659"])
BINARY_NORM = mcolors.BoundaryNorm([0, 0.5, 1], BINARY_CMAP.N)


def plot_onehot_map(var_name: str, long_name: str):
    """Plot a Robinson projection map for a single one-hot soil type layer.

    Uses a binary colormap to clearly show presence (green) vs absence (tan)
    of each soil type. Ocean and no-data areas are left white.
    """
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

    # Plot with binary colormap
    im = da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=BINARY_CMAP,
        norm=BINARY_NORM,
        add_colorbar=False,
    )

    # Add a simple legend-style colorbar with "absent" / "present" labels
    cbar = plt.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.05, aspect=40, shrink=0.5,
        ticks=[0.25, 0.75],
    )
    cbar.ax.set_xticklabels(["Absent", "Present"])

    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.set_title(f"Soil texture: {long_name}", fontsize=12)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    logger.info("Saved map → %s", out)


@click.command()
@click.option("--variables", "-v", multiple=True,
              help="Variable(s) (e.g. soiltex_sand soiltex_clay).")
@click.option("--all", "run_all", is_flag=True, help="Process all soil types.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
def main(variables, run_all, skip_download, overwrite):
    """GLDAS soil texture pipeline: download → one-hot encode → visualize."""
    config = load_config()
    all_vars = config["variables"]

    if not variables and not run_all:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    var_list = list(variables) if variables else list(all_vars.keys())

    click.echo(f"Pipeline: {len(var_list)} soil texture types")

    # ── Step 1: Download ─────────────────────────────────────────────────
    if not skip_download:
        click.echo("\n── Download ──")
        download_soiltex(overwrite=overwrite)
    else:
        click.echo("\n── Download (skipped) ──")

    # ── Step 2: One-hot encode + regrid ──────────────────────────────────
    click.echo("\n── Processing ──")
    n = process_soiltex(
        variables=var_list,
        overwrite=overwrite,
    )
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
