"""Full GMTED2010 pipeline: download → derive slope → process → visualize.

Data source
-----------
Website : https://www.usgs.gov/coastal-changes-and-impacts/gmted2010
Download: https://topotools.cr.usgs.gov/gmted_viewer/gmted2010_global_grids.php
Citation: Danielson & Gesch (2011), USGS Open-File Report 2011-1073
License : Public domain (USGS)

Downloads 30 arc-second mean elevation and standard deviation grids,
derives slope via gdaldem, resamples to 0.25° WGS84, and generates
Robinson projection maps.

Usage:
    python -m src.pipelines.gmted2010 --all
    python -m src.pipelines.gmted2010 --all --skip-download
    python -m src.pipelines.gmted2010 --variables elevation_mean slope_mean
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

from src.download.gmted2010 import download_gmted, load_gmted_config
from src.processing.gmted2010_to_static import process_gmted, FINAL_DIR
from src.utils import get_logger

logger = get_logger("pipeline.gmted2010")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots" / "gmted2010"


def plot_map(var_code: str, long_name: str, units: str, cmap: str):
    """Plot a Robinson projection spatial map for one variable."""
    out = PLOTS_DIR / f"{var_code}.png"
    if out.exists():
        return

    nc_path = FINAL_DIR / f"{var_code}.nc"
    if not nc_path.exists():
        return

    ds = xr.open_dataset(nc_path)
    da = ds[var_code]

    finite = da.values[np.isfinite(da.values)]
    if finite.size > 0:
        vmin = float(np.percentile(finite, 1))
        vmax = float(np.percentile(finite, 99))
    else:
        vmin, vmax = 0.0, 1.0

    # For slope/stddev, floor at 0
    if var_code != "elevation_mean":
        vmin = 0.0

    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={"projection": ccrs.Robinson()},
    )

    im = da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        add_colorbar=False,
    )
    plt.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.05, aspect=40, shrink=0.8,
        label=f"{long_name} ({units})",
    )
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.set_title(long_name, fontsize=12)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    logger.info("Saved map → %s", out)


@click.command()
@click.option("--variables", "-v", multiple=True,
              help="Variable(s) (e.g. elevation_mean slope_mean).")
@click.option("--all", "run_all", is_flag=True, help="Process all variables.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
def main(variables, run_all, skip_download, overwrite):
    """GMTED2010 full pipeline: download → process → visualize."""
    config = load_gmted_config()
    all_vars = config["variables"]

    if not variables and not run_all:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    var_list = list(variables) if variables else list(all_vars.keys())

    click.echo(f"Pipeline: {len(var_list)} GMTED2010 variables")

    # ── Step 1: Download + derive slope ──────────────────────────────────
    if not skip_download:
        click.echo("\n── Download ──")
        download_gmted(overwrite=overwrite)

    # ── Step 2: Process (resample + regrid) ──────────────────────────────
    click.echo("\n── Processing ──")
    n = process_gmted(
        variables=var_list,
        overwrite=overwrite,
    )
    click.echo(f"  Processed {n} files")

    # ── Step 3: Visualize ────────────────────────────────────────────────
    click.echo("\n── Visualization ──")

    for var_code in var_list:
        var_info = all_vars.get(var_code)
        if var_info is None:
            continue

        try:
            plot_map(
                var_code,
                var_info["long_name"],
                var_info["units"],
                var_info.get("cmap", "viridis"),
            )
        except Exception as e:
            logger.warning("Plot failed %s: %s", var_code, e)

        gc.collect()
        plt.close("all")

    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
