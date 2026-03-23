"""Full GPW pipeline: download → regrid + interpolate → visualize.

Downloads GPWv4.11 population data from NASA Earthdata, regrids 2.5' → 0.25°,
linearly interpolates between anchor years (2000-2020), then generates
Robinson projection spatial maps and time series plots.

Usage:
    python -m src.pipelines.gpw --all
    python -m src.pipelines.gpw --variables population_count --skip-download
    python -m src.pipelines.gpw --all --skip-download
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

from src.data_layout import output_dir_for, output_path_for
from src.download.gpw import download_gpw, load_gpw_config, DEFAULT_RAW_DIR as RAW_DIR
from src.processing.gpw_to_yearly import (
    FINAL_DIR,
    process_nc_variable, process_tif_variable,
)
from src.utils import get_logger

logger = get_logger("pipeline.gpw")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots" / "gpw"

# Colormaps per variable type
CMAP_MAP = {
    "population_count": "YlOrRd",
    "population_density": "YlOrRd",
    "unwpp_adjusted_density": "YlOrRd",
    "land_area": "YlGn",
}


def get_cmap(var_name: str) -> str:
    """Return appropriate colormap for a variable."""
    return CMAP_MAP.get(var_name, "YlOrRd")


# ── Visualization ────────────────────────────────────────────────────────────

def plot_spatial_map(var_name: str, long_name: str, units: str, year: int | None = None):
    """Plot a Robinson projection spatial map for one GPW variable."""
    suffix = "static" if year is None else str(year)
    out = PLOTS_DIR / "maps" / var_name / f"{suffix}.png"
    if out.exists():
        return

    nc_path = output_path_for(var_name, base_dir=FINAL_DIR) if year is None else output_path_for(var_name, year=year, base_dir=FINAL_DIR)
    if not nc_path.exists():
        return

    ds = xr.open_dataset(nc_path)
    dv = list(ds.data_vars)[0]
    da = ds[dv]

    nonzero = da.values[da.values > 0]
    if nonzero.size > 0:
        vmax = float(np.percentile(nonzero, 99))
    else:
        vmax = float(da.max()) or 1.0

    cmap = get_cmap(var_name)

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
    title = long_name if year is None else f"{long_name} — {year}"
    ax.set_title(title, fontsize=12)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    logger.info("Saved map → %s", out)


def plot_time_series(var_name: str, long_name: str, units: str,
                     years: list[int], means: list[float]):
    """Plot global mean over time with markers."""
    out = PLOTS_DIR / "timeseries" / f"{var_name}.png"
    if out.exists():
        return
    if not means:
        return

    y = np.array(years[:len(means)])
    v = np.array(means[:len(y)])

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(y, v, linewidth=2, color="#1f77b4", marker="o", markersize=5)
    ax.fill_between(y, 0, v, alpha=0.15, color="#1f77b4")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Global mean ({units})", fontsize=10)
    ax.set_title(f"GPW {long_name}", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(y[0] - 0.5, y[-1] + 0.5)

    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved time series → %s", out)


# ── Pipeline ─────────────────────────────────────────────────────────────────

@click.command()
@click.option("--variables", "-v", multiple=True,
              help="Variable(s) (e.g. population_count).")
@click.option("--all", "run_all", is_flag=True, help="Process all variables.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--plot-every", type=int, default=5,
              help="Plot spatial maps every N years (default: 5).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
def main(variables, run_all, skip_download, plot_every, overwrite):
    """GPW full pipeline: download → regrid → interpolate → visualize."""
    config = load_gpw_config()
    anchor_years = config["anchor_years"]
    output_years = config["output_years"]

    if not variables and not run_all:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    var_list = list(variables) if variables else list(config["variables"].keys())

    click.echo(f"Pipeline: {len(var_list)} variables, "
               f"anchors {anchor_years}, output {output_years[0]}-{output_years[-1]}")

    # ── Step 1: Download ─────────────────────────────────────────────────
    if not skip_download:
        click.echo("\n── Download ──")
        download_gpw(variables=var_list, overwrite=overwrite)

    # ── Step 2: Process (regrid + interpolate) ───────────────────────────
    click.echo("\n── Processing ──")
    for var_name in var_list:
        var_info = config["variables"].get(var_name)
        if var_info is None:
            continue

        var_raw_dir = RAW_DIR / var_name
        if not var_raw_dir.exists():
            logger.warning("Raw directory not found: %s", var_raw_dir)
            continue

        click.echo(f"\n[{var_name}] {var_info['long_name']}")

        fmt = var_info.get("format", "nc")
        if fmt == "tif":
            process_tif_variable(
                var_raw_dir, var_name, var_info,
                anchor_years, output_years, overwrite,
            )
        else:
            process_nc_variable(
                var_raw_dir, var_name, var_info,
                anchor_years, output_years, overwrite,
            )

    # ── Step 3: Visualize ────────────────────────────────────────────────
    click.echo("\n── Visualization ──")

    # Pick which years get spatial map plots
    plot_year_set = set(range(output_years[0], output_years[-1] + 1, plot_every))
    plot_year_set.add(output_years[-1])

    for var_name in var_list:
        var_info = config["variables"].get(var_name)
        if var_info is None:
            continue

        units = var_info["units"]
        long_name = var_info["long_name"]
        is_static = var_info.get("static", False)

        ts_years = []
        ts_means = []

        if is_static:
            try:
                plot_spatial_map(var_name, long_name, units, year=None)
            except Exception as e:
                logger.warning("Static map plot failed %s: %s", var_name, e)
            continue

        var_dir = output_dir_for(var_name, base_dir=FINAL_DIR)
        if not var_dir.exists():
            continue

        for year in output_years:
            nc_path = output_path_for(var_name, year=year, base_dir=FINAL_DIR)
            if not nc_path.exists():
                continue

            # Collect stats for time series
            try:
                ds = xr.open_dataset(nc_path)
                dv = list(ds.data_vars)[0]
                ts_years.append(year)
                ts_means.append(float(ds[dv].mean(skipna=True).values))
                ds.close()
            except Exception:
                pass

            # Spatial map (only for selected years)
            if year in plot_year_set:
                try:
                    plot_spatial_map(var_name, long_name, units, year=year)
                except Exception as e:
                    logger.warning("Map plot failed %s/%d: %s", var_name, year, e)

        # Time series plot (skip for static vars)
        if ts_years and not is_static:
            try:
                plot_time_series(var_name, long_name, units, ts_years, ts_means)
            except Exception as e:
                logger.warning("Time series plot failed %s: %s", var_name, e)

        gc.collect()
        plt.close("all")

    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
