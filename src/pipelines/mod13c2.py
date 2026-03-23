"""Full MOD13C2 pipeline: download -> process -> cleanup -> visualize.

Produces ERA5-style yearly derived maps for NDVI/EVI:
    {var}_{mean|std|max|min}/{year}.nc

Usage:
    python -m src.pipelines.mod13c2 --all
    python -m src.pipelines.mod13c2 --all --start-year 2010 --end-year 2020
    python -m src.pipelines.mod13c2 -v ndvi --skip-download --plot-every 5
"""

from __future__ import annotations

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

from src.download.mod13c2 import download_mod13c2, DEFAULT_RAW_DIR
from src.processing.mod13c2_monthly_to_yearly import (
    DEFAULT_FINAL_DIR,
    load_mod13c2_config,
    parse_granule_date,
    list_raw_granules,
    process_mod13c2,
)
from src.utils import get_logger, add_cyclic_point_xr

logger = get_logger("pipeline.mod13c2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots" / "mod13c2"

CMAPS = {
    "ndvi": "YlGn",
    "evi": "YlGnBu",
    "std": "magma",
}
STATS = ("mean", "std", "max", "min")


def _available_years(var_name: str, final_dir: Path) -> list[int]:
    var_dir = final_dir / f"{var_name}_mean"
    if not var_dir.exists():
        return []
    years = []
    for p in var_dir.glob("*.nc"):
        try:
            years.append(int(p.stem))
        except ValueError:
            continue
    return sorted(years)


def plot_spatial_map(var_name: str, var_info: dict, year: int, final_dir: Path):
    """Plot a 4-panel map for yearly mean/std/max/min outputs."""
    out = PLOTS_DIR / "maps" / var_name / f"{year}.png"
    if out.exists():
        return

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 10),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    axes = np.atleast_2d(axes)

    base_cmap = CMAPS.get(var_name, "YlGn")

    for idx, stat in enumerate(STATS):
        ax = axes.flat[idx]
        nc = final_dir / f"{var_name}_{stat}" / f"{year}.nc"
        if not nc.exists():
            ax.set_title(f"{stat.upper()} - not available")
            ax.set_global()
            continue

        ds = xr.open_dataset(nc)
        dv = list(ds.data_vars)[0]
        da = ds[dv]

        # Fix meridian seam robustly
        data_cyclic, lon_cyclic, lat_values = add_cyclic_point_xr(da)

        cmap = CMAPS.get(stat, base_cmap)
        
        # Use pcolormesh for seamless plotting
        vmin, vmax = None, None
        if stat in ("mean", "max", "min"):
            vmin, vmax = -0.2, 1.0
        elif stat == "std":
            vmin = float(np.nanpercentile(data_cyclic, 2))
            vmax = float(np.nanpercentile(data_cyclic, 98))
        
        im = ax.pcolormesh(
            lon_cyclic, lat_values, data_cyclic,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            shading='auto'
        )

        plt.colorbar(
            im,
            ax=ax,
            orientation="horizontal",
            pad=0.05,
            aspect=40,
            shrink=0.8,
            label=da.attrs.get("units", ""),
        )
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.set_global()
        ax.set_title(f"{var_info['long_name']} - {stat.upper()}", fontsize=11)
        ds.close()

    plt.suptitle(f"MOD13C2 {var_name.upper()} - {year}", fontsize=14, y=1.01)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved map -> %s", out)


def plot_time_series(var_name: str, var_info: dict, series_by_stat: dict[str, dict[str, list[float] | list[int]]]):
    """Plot spatial-mean yearly time series for all stats."""
    out = PLOTS_DIR / "timeseries" / f"{var_name}.png"
    if out.exists():
        return

    available = [s for s in STATS if series_by_stat.get(s, {}).get("values")]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = {"mean": "#2a9d8f", "std": "#e76f51", "max": "#264653", "min": "#8ab17d"}

    for ax, stat in zip(axes, available):
        vals = series_by_stat[stat]["values"]
        x = series_by_stat[stat]["years"]
        ax.plot(x, vals, marker=".", markersize=3, linewidth=1.2, color=colors[stat])
        ax.set_ylabel(f"{stat.upper()} ({var_info['units']})")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{var_info['long_name']} - {stat.upper()}", fontsize=11)

    axes[-1].set_xlabel("Year")
    plt.suptitle(f"MOD13C2 {var_name.upper()} - Time Series", fontsize=13, y=1.01)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved time series -> %s", out)


@click.command()
@click.option("--variables", "variables", "-v", multiple=True, help="Variables (ndvi, evi).")
@click.option("--all", "run_all", is_flag=True, help="Process all configured variables.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--plot-every", type=int, default=5, show_default=True, help="Spatial map interval in years.")
@click.option("--skip-download", is_flag=True, help="Skip earthaccess download step.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing yearly outputs.")
@click.option(
    "--qa-mode",
    type=click.Choice(["strict", "moderate", "none"], case_sensitive=False),
    default="moderate",
    show_default=True,
    help="Quality mask mode.",
)
@click.option("--cleanup-raw/--keep-raw", default=True, show_default=True, help="Delete raw granules after processing.")
def main(variables, run_all, start_year, end_year, plot_every, skip_download, overwrite, qa_mode, cleanup_raw):
    """MOD13C2 full pipeline: download -> process -> cleanup -> visualize."""
    config = load_mod13c2_config()
    all_vars = config["variables"]
    cfg_start, cfg_end = config["temporal_range"]

    if variables:
        var_list = [v for v in variables if v in all_vars]
        missing = [v for v in variables if v not in all_vars]
        if missing:
            logger.warning("Unknown variables skipped: %s", missing)
    elif run_all:
        var_list = list(all_vars.keys())
    else:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    effective_start = start_year or cfg_start
    effective_end = end_year or cfg_end

    click.echo(f"Pipeline: vars={var_list}, years={effective_start}-{effective_end}, plot_every={plot_every}")

    # 1) Download
    if not skip_download:
        click.echo("\n-- Download --")
        paths = download_mod13c2(
            start_year=effective_start,
            end_year=effective_end,
            raw_dir=DEFAULT_RAW_DIR,
        )
        click.echo(f"Downloaded/available granules: {len(paths)}")

    # 2) Process monthly -> yearly stats
    click.echo("\n-- Processing --")
    summary = process_mod13c2(
        variables=var_list,
        start_year=effective_start,
        end_year=effective_end,
        raw_dir=DEFAULT_RAW_DIR,
        output_dir=DEFAULT_FINAL_DIR,
        overwrite=overwrite,
        qa_mode=qa_mode.lower(),
    )
    click.echo(
        f"Processed granules={summary.get('granules', 0)}, "
        f"years_written={summary.get('years_written', 0)}, "
        f"files_written={summary.get('files_written', 0)}"
    )

    # 3) Optional cleanup
    if cleanup_raw:
        granules = list_raw_granules(DEFAULT_RAW_DIR)
        removed = 0
        for p in granules:
            dt = parse_granule_date(p)
            if dt is None:
                continue
            if dt.year < effective_start:
                continue
            if dt.year > effective_end:
                continue
            p.unlink(missing_ok=True)
            removed += 1
        click.echo(f"Cleanup: removed {removed} raw granules")

        # remove empty raw dir
        if DEFAULT_RAW_DIR.exists() and not any(DEFAULT_RAW_DIR.iterdir()):
            DEFAULT_RAW_DIR.rmdir()

    # 4) Visualization
    click.echo("\n-- Visualization --")
    for var_name in var_list:
        info = all_vars[var_name]

        years = _available_years(var_name, DEFAULT_FINAL_DIR)
        years = [y for y in years if effective_start <= y <= effective_end]
        if not years:
            continue

        plot_years = set(range(years[0], years[-1] + 1, plot_every))
        plot_years.add(years[-1])

        series_by_stat = {s: {"years": [], "values": []} for s in STATS}

        for year in years:
            for stat in STATS:
                nc_path = DEFAULT_FINAL_DIR / f"{var_name}_{stat}" / f"{year}.nc"
                if not nc_path.exists():
                    continue
                ds = xr.open_dataset(nc_path)
                dv = list(ds.data_vars)[0]
                series_by_stat[stat]["years"].append(year)
                series_by_stat[stat]["values"].append(float(ds[dv].mean(skipna=True).values))
                ds.close()

            if year in plot_years:
                try:
                    plot_spatial_map(var_name, info, year, DEFAULT_FINAL_DIR)
                except Exception as e:
                    logger.warning("Map plot failed %s/%d: %s", var_name, year, e)

        try:
            plot_time_series(var_name, info, series_by_stat)
        except Exception as e:
            logger.warning("Time series plot failed %s: %s", var_name, e)

        gc.collect()
        plt.close("all")

    click.echo(f"\nPipeline complete. Outputs in {DEFAULT_FINAL_DIR} and plots in {PLOTS_DIR}")


if __name__ == "__main__":
    main()
