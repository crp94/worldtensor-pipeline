"""Full MCD64A1 pipeline: download -> process -> cleanup -> visualize.

Keeps only the yearly burned-area sum product.
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

from src.download.mcd64a1 import (
    download_mcd64a1,
    download_granules_direct,
    login_earthaccess,
    search_granules,
    DEFAULT_RAW_DIR,
)
from src.processing.mcd64a1_v2 import (
    DEFAULT_FINAL_DIR,
    load_mcd64a1_config,
    parse_granule_date,
    list_raw_granules,
    process_mcd64a1,
)
from src.utils import get_logger
from src.year_policy import resolve_year_bounds

logger = get_logger("pipeline.mcd64a1")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
PLOTS_DIR = PROJECT_ROOT / "plots" / "mcd64a1"

CMAPS = {
    "burned_area": "hot_r",
    "burn_date": "viridis",
    "std": "magma",
}

STATS_BY_VAR = {
    "burned_area": ("sum",),
}

def _available_years(var_name: str, final_dir: Path) -> list[int]:
    stats = STATS_BY_VAR.get(var_name, ("mean",))
    primary = stats[0]
    var_dir = final_dir / f"{var_name}_{primary}"
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
    """Plot the primary yearly output map."""
    out = PLOTS_DIR / "maps" / var_name / f"{year}.png"
    if out.exists():
        return

    stats_to_plot = STATS_BY_VAR.get(var_name, ("mean", "std", "max", "min"))
    fig, axes = plt.subplots(
        1,
        len(stats_to_plot),
        figsize=(8 * len(stats_to_plot), 5),
        subplot_kw={"projection": ccrs.Robinson()},
    )
    axes = np.atleast_1d(axes)

    base_cmap = CMAPS.get(var_name, "hot_r")

    for idx, stat in enumerate(stats_to_plot):
        ax = axes[idx]
        nc = final_dir / f"{var_name}_{stat}" / f"{year}.nc"
        if not nc.exists():
            ax.set_title(f"{stat.upper()} - not available")
            ax.set_global()
            continue

        ds = xr.open_dataset(nc)
        dv = list(ds.data_vars)[0]
        da = ds[dv]

        cmap = CMAPS.get(stat, base_cmap)
        kwargs = {
            "ax": ax,
            "transform": ccrs.PlateCarree(),
            "cmap": cmap,
            "add_colorbar": False,
            "robust": True,
        }

        im = da.plot(**kwargs)

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

    plt.suptitle(f"MCD64A1 {var_name.upper()} - {year}", fontsize=14, y=1.01)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved map -> %s", out)

def plot_time_series(var_name: str, var_info: dict, series_by_stat: dict[str, dict[str, list[float] | list[int]]]):
    """Plot spatial-mean yearly time series for all stats."""
    out = PLOTS_DIR / "timeseries" / f"{var_name}.png"
    
    stats_to_plot = STATS_BY_VAR.get(var_name, ("mean", "std", "max", "min"))
    available = [s for s in stats_to_plot if series_by_stat.get(s, {}).get("values")]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = {"mean": "#2a9d8f", "sum": "#2a9d8f", "std": "#e76f51", "max": "#264653", "min": "#8ab17d"}

    for ax, stat in zip(axes, available):
        vals = series_by_stat[stat]["values"]
        x = series_by_stat[stat]["years"]
        ax.plot(x, vals, marker=".", markersize=3, linewidth=1.2, color=colors.get(stat, "#000000"))
        ax.set_ylabel(f"{stat.upper()} ({var_info['units']})")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{var_info['long_name']} - {stat.upper()}", fontsize=11)

    axes[-1].set_xlabel("Year")
    plt.suptitle(f"MCD64A1 {var_name.upper()} - Time Series", fontsize=13, y=1.01)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved time series -> %s", out)


@click.command()
@click.option("--variables", "variables", "-v", multiple=True, help="Variables (burned_area, burn_date).")
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
    """MCD64A1 full pipeline: download -> process -> cleanup -> visualize."""
    config = load_mcd64a1_config()
    all_vars = config["variables"]
    cfg_start, cfg_end = config["temporal_range"]

    if variables:
        var_list = [v for v in variables if v in all_vars]
    elif run_all:
        var_list = list(all_vars.keys())
    else:
        click.echo("Specify --variables or --all.")
        return

    effective_start, effective_end = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=cfg_start,
        default_end=cfg_end,
        label="MCD64A1 pipeline years",
    )

    click.echo(f"Pipeline: vars={var_list}, years={effective_start}-{effective_end}, plot_every={plot_every}")

    all_years = list(range(effective_start, effective_end + 1))

    # ── Single auth + search, then group granules by year ────────────────
    granules_by_year: dict[int, list] = {}
    if not skip_download:
        login_earthaccess()
        all_granules = search_granules(start_year=effective_start, end_year=effective_end)
        click.echo(f"  Total granules found: {len(all_granules)}")

        # Group granules by year using their temporal metadata
        for g in all_granules:
            try:
                ts = g.get("umm", {}).get("TemporalExtent", {}).get(
                    "RangeDateTime", {}
                ).get("BeginningDateTime", "")
                yr = int(ts[:4]) if ts else None
            except Exception:
                yr = None
            if yr is not None:
                granules_by_year.setdefault(yr, []).append(g)

    for year in all_years:
        click.echo(f"\n--- Processing Year {year} ---")

        if not skip_download:
            year_granules = granules_by_year.get(year, [])
            if year_granules:
                paths = download_granules_direct(year_granules, raw_dir=DEFAULT_RAW_DIR, threads=16)
                click.echo(f"  Downloaded/available granules: {len(paths)}")
            else:
                click.echo("  No granules found for this year")

        summary = process_mcd64a1(
            variables=var_list,
            start_year=year,
            end_year=year,
            raw_dir=DEFAULT_RAW_DIR,
            output_dir=DEFAULT_FINAL_DIR,
            overwrite=overwrite,
            qa_mode=qa_mode.lower(),
        )
        click.echo(
            f"  Processed granules={summary.get('granules', 0)}, "
            f"years_written={summary.get('years_written', 0)}, "
            f"files_written={summary.get('files_written', 0)}"
        )

        if cleanup_raw:
            granules = list_raw_granules(DEFAULT_RAW_DIR)
            removed = 0
            for p in granules:
                dt = parse_granule_date(p)
                if dt and dt.year == year:
                    p.unlink(missing_ok=True)
                    removed += 1
            click.echo(f"  Cleanup: removed {removed} raw granules for {year}")

        if (year - effective_start) % plot_every == 0 or year == effective_end:
            click.echo(f"  Visualization for {year}...")
            for var_name in var_list:
                try:
                    plot_spatial_map(var_name, all_vars[var_name], year, DEFAULT_FINAL_DIR)
                except Exception as e:
                    logger.warning("Map plot failed %s/%d: %s", var_name, year, e)

    click.echo("\n-- Final Visualization --")
    for var_name in var_list:
        info = all_vars[var_name]
        years = _available_years(var_name, DEFAULT_FINAL_DIR)
        if not years: continue

        stats_to_plot = STATS_BY_VAR.get(var_name, ("mean", "std", "max", "min"))
        series_by_stat = {s: {"years": [], "values": []} for s in stats_to_plot}

        for year in years:
            for stat in stats_to_plot:
                nc_path = DEFAULT_FINAL_DIR / f"{var_name}_{stat}" / f"{year}.nc"
                if not nc_path.exists(): continue
                ds = xr.open_dataset(nc_path)
                dv = list(ds.data_vars)[0]
                val = float(ds[dv].mean(skipna=True).values)
                series_by_stat[stat]["years"].append(year)
                series_by_stat[stat]["values"].append(val)
                ds.close()

        try:
            plot_time_series(var_name, info, series_by_stat)
        except Exception as e:
            logger.warning("Time series plot failed %s: %s", var_name, e)

        gc.collect()
        plt.close("all")

    click.echo(f"\nPipeline complete. Outputs in {DEFAULT_FINAL_DIR}")


if __name__ == "__main__":
    main()
