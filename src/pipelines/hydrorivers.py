"""Full HydroRIVERS pipeline: download → process → visualize.

Data source
-----------
Website : https://www.hydrosheds.org/products/hydrorivers
Citation: Lehner & Grill (2013), Hydrol. Process. 27(15):2171-2186.
License : Free for non-commercial use

Downloads HydroRIVERS v1.0 global river shapefile, rasterizes, computes
distance-to-nearest-river, and generates a Robinson map.

Usage:
    python -m src.pipelines.hydrorivers --all
    python -m src.pipelines.hydrorivers --all --skip-download
"""

import gc
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from src.download.hydrorivers import download_hydrorivers, load_config
from src.processing.hydrorivers_to_static import process_hydrorivers, FINAL_DIR
from src.utils import get_logger

logger = get_logger("pipeline.hydrorivers")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots" / "hydrorivers"


def plot_dist_to_river():
    """Plot distance-to-river Robinson map."""
    var_name = "dist_to_river"
    out = PLOTS_DIR / f"{var_name}.png"
    if out.exists():
        return

    nc_path = FINAL_DIR / f"{var_name}.nc"
    if not nc_path.exists():
        return

    ds = xr.open_dataset(nc_path)
    da = ds[var_name]

    finite = da.values[np.isfinite(da.values)]
    if finite.size > 0:
        vmin = 0.0
        vmax = float(np.percentile(finite, 99))
    else:
        vmin, vmax = 0.0, 500.0

    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={"projection": ccrs.Robinson()},
    )

    im = da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="Blues_r",
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    plt.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.05, aspect=40, shrink=0.8,
        label="Distance to nearest river (km)",
    )
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.set_title("Distance to Nearest River", fontsize=12)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    logger.info("Saved map → %s", out)


@click.command()
@click.option("--all", "run_all", is_flag=True, help="Run full pipeline.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
def main(run_all, skip_download, overwrite):
    """HydroRIVERS pipeline: download → process → visualize."""
    if not run_all:
        click.echo("Specify --all. Use --help for usage.")
        return

    click.echo("Pipeline: distance to nearest river (HydroRIVERS)")

    if not skip_download:
        click.echo("\n── Download ──")
        download_hydrorivers(overwrite=overwrite)

    click.echo("\n── Processing ──")
    result = process_hydrorivers(overwrite=overwrite)
    if result:
        click.echo(f"  Output: {result}")
    else:
        click.echo("  Skipped (already exists)")

    click.echo("\n── Visualization ──")
    try:
        plot_dist_to_river()
    except Exception as e:
        logger.warning("Plot failed: %s", e)

    gc.collect()
    plt.close("all")

    click.echo(f"\nPipeline complete. Output in {FINAL_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
