"""Full SoilGrids pipeline: download → process → visualize.

Data source
-----------
Website : https://www.isric.org/explore/soilgrids
Data    : https://files.isric.org/soilgrids/latest/data/
Citation: Poggio et al. (2021), Soil 7:217-240, doi:10.5194/soil-7-217-2021

Downloads SoilGrids 2.0 soil properties via gdalwarp, regrids to 0.25°,
and generates Robinson projection maps and depth profile plots.

Usage:
    python -m src.pipelines.soilgrids --all
    python -m src.pipelines.soilgrids --properties bdod clay --skip-download
    python -m src.pipelines.soilgrids --all --skip-download
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

from src.download.soilgrids import download_soilgrids, load_soilgrids_config, DEFAULT_RAW_DIR as RAW_DIR
from src.processing.soilgrids_to_static import process_soilgrids, FINAL_DIR
from src.utils import get_logger

logger = get_logger("pipeline.soilgrids")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots" / "soilgrids"

# Colormaps per property
CMAP_MAP = {
    "bdod": "YlOrBr",
    "cec": "YlOrBr",
    "clay": "YlGnBu",
    "nitrogen": "YlGn",
    "ocd": "YlOrBr",
    "phh2o": "RdYlGn_r",
    "sand": "YlGnBu",
    "silt": "YlGnBu",
}

# Depth labels ordered shallow → deep for profile plots
DEPTH_ORDER = ["0-5cm", "5-15cm", "15-30cm", "30-60cm", "60-100cm", "100-200cm"]
DEPTH_MIDPOINTS = [2.5, 10.0, 22.5, 45.0, 80.0, 150.0]  # cm


def get_cmap(prop: str) -> str:
    return CMAP_MAP.get(prop, "viridis")


# ── Visualization ────────────────────────────────────────────────────────────

def plot_spatial_map(prop: str, depth: str, statistic: str,
                     long_name: str, units: str):
    """Plot a Robinson projection spatial map for one property/depth."""
    out = PLOTS_DIR / "maps" / f"{prop}_{depth}_{statistic}.png"
    if out.exists():
        return

    var_name = f"{prop}_{depth}_{statistic}"
    nc_path = FINAL_DIR / f"{var_name}.nc"
    if not nc_path.exists():
        return

    ds = xr.open_dataset(nc_path)
    da = ds[var_name]

    nonzero = da.values[np.isfinite(da.values) & (da.values > 0)]
    if nonzero.size > 0:
        vmax = float(np.percentile(nonzero, 99))
    else:
        vmax = float(np.nanmax(da.values)) or 1.0

    cmap = get_cmap(prop)

    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={"projection": ccrs.Robinson()},
    )

    im = da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=0,
        vmax=vmax,
        add_colorbar=False,
    )
    plt.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.05, aspect=40, shrink=0.8,
        label=units,
    )
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.set_title(f"{long_name} ({depth})", fontsize=12)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    logger.info("Saved map → %s", out)


def plot_depth_profile(prop: str, statistic: str, long_name: str, units: str):
    """Plot global mean vs depth for a single property."""
    out = PLOTS_DIR / "depth_profiles" / f"{prop}.png"
    if out.exists():
        return

    means = []
    valid_depths = []
    valid_midpoints = []

    for depth, midpoint in zip(DEPTH_ORDER, DEPTH_MIDPOINTS):
        var_name = f"{prop}_{depth}_{statistic}"
        nc_path = FINAL_DIR / f"{var_name}.nc"
        if not nc_path.exists():
            continue

        ds = xr.open_dataset(nc_path)
        val = float(ds[var_name].mean(skipna=True).values)
        ds.close()

        means.append(val)
        valid_depths.append(depth)
        valid_midpoints.append(midpoint)

    if len(means) < 2:
        return

    fig, ax = plt.subplots(figsize=(6, 8))
    ax.plot(means, valid_midpoints, "o-", linewidth=2, markersize=8, color="#2c7bb6")
    ax.invert_yaxis()
    ax.set_xlabel(f"Global mean ({units})", fontsize=11)
    ax.set_ylabel("Depth (cm)", fontsize=11)
    ax.set_yticks(valid_midpoints)
    ax.set_yticklabels(valid_depths)
    ax.set_title(f"SoilGrids {long_name}", fontsize=12)
    ax.grid(True, alpha=0.3)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved depth profile → %s", out)


# ── Pipeline ─────────────────────────────────────────────────────────────────

@click.command()
@click.option("--properties", "-p", multiple=True,
              help="Property(ies) (e.g. bdod clay).")
@click.option("--all", "run_all", is_flag=True, help="Process all properties.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
@click.option("--workers", "-w", type=int, default=4,
              help="Parallel download workers (default: 4).")
def main(properties, run_all, skip_download, overwrite, workers):
    """SoilGrids full pipeline: download → process → visualize."""
    config = load_soilgrids_config()
    depths = config["depths"]
    statistic = config["statistic"]
    all_props = config["properties"]

    if not properties and not run_all:
        click.echo("Specify --properties or --all. Use --help for usage.")
        return

    prop_list = list(properties) if properties else list(all_props.keys())

    click.echo(f"Pipeline: {len(prop_list)} properties × {len(depths)} depths "
               f"= {len(prop_list) * len(depths)} layers")

    # ── Step 1: Download ─────────────────────────────────────────────────
    if not skip_download:
        click.echo("\n── Download ──")
        download_soilgrids(
            properties=prop_list,
            overwrite=overwrite,
            workers=workers,
        )

    # ── Step 2: Process (regrid to master grid) ──────────────────────────
    click.echo("\n── Processing ──")
    n = process_soilgrids(
        properties=prop_list,
        overwrite=overwrite,
    )
    click.echo(f"  Processed {n} files")

    # ── Step 3: Visualize ────────────────────────────────────────────────
    click.echo("\n── Visualization ──")

    for prop in prop_list:
        prop_info = all_props.get(prop)
        if prop_info is None:
            continue

        units = prop_info["units"]
        long_name = prop_info["long_name"]

        # Spatial maps (one per depth)
        for depth in depths:
            try:
                plot_spatial_map(prop, depth, statistic, long_name, units)
            except Exception as e:
                logger.warning("Map plot failed %s_%s: %s", prop, depth, e)

        # Depth profile
        try:
            plot_depth_profile(prop, statistic, long_name, units)
        except Exception as e:
            logger.warning("Depth profile failed %s: %s", prop, e)

        gc.collect()
        plt.close("all")

    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
