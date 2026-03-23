"""Full EDGAR pipeline: download → regrid → cleanup raw → visualize.

Downloads per-year/substance/sector zips, regrids 0.1° → 0.25°, deletes raw
files to save storage, then generates spatial maps and IQR time series plots.

Usage:
    python -m src.pipelines.edgar --all
    python -m src.pipelines.edgar --substances CO2 --sectors TOTALS --start-year 2015 --end-year 2024
    python -m src.pipelines.edgar --all --skip-download --plot-every 5
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
import yaml
from tqdm import tqdm

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.download.edgar import (
    download_file, build_url, load_edgar_config,
    DEFAULT_RAW_DIR as RAW_DIR,
)
from src.processing.edgar_to_yearly import (
    TARGET_LAT, TARGET_LON, process_file, expected_nc_name,
    aggregate_sectors, FINAL_DIR,
)
from src.utils import get_logger, add_cyclic_point_xr

logger = get_logger("pipeline.edgar")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "edgar.yml"
PLOTS_DIR = PROJECT_ROOT / "plots" / "edgar"


def _data_var_name(ds: xr.Dataset, preferred: str) -> str:
    if preferred in ds.data_vars:
        return preferred
    if len(ds.data_vars) == 1:
        return next(iter(ds.data_vars))
    raise KeyError(f"Could not resolve data variable '{preferred}' in {list(ds.data_vars)}")


# ── Visualization ────────────────────────────────────────────────────────────

def plot_spatial_map(substance: str, sector: str, year: int,
                     long_name: str, units: str):
    """Plot a Robinson projection spatial map for one substance/sector/year."""
    out = PLOTS_DIR / "maps" / f"{substance}_{sector}" / f"{year:04d}.png"
    if out.exists():
        return

    nc_path = FINAL_DIR / f"{substance}_{sector}" / f"{year:04d}.nc"
    if not nc_path.exists():
        return

    ds = xr.open_dataset(nc_path)
    var_name = _data_var_name(ds, f"{substance}_{sector}")
    da = ds[var_name]

    # Fix meridian seam robustly
    data_cyclic, lon_cyclic, lat_values = add_cyclic_point_xr(da)

    # Compute vmax from nonzero values to avoid flat-color maps
    nonzero = da.values[da.values > 0]
    if nonzero.size > 0:
        vmax = float(np.percentile(nonzero, 99))
    else:
        vmax = float(da.max()) or 1.0

    fig, ax = plt.subplots(
        figsize=(12, 6),
        subplot_kw={"projection": ccrs.Robinson()},
    )

    im = ax.pcolormesh(
        lon_cyclic, lat_values, data_cyclic,
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        shading='auto'
    )
    plt.colorbar(
        im, ax=ax, orientation="horizontal", pad=0.05, aspect=40, shrink=0.8,
        label=units,
    )
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.set_title(f"{long_name} [{sector}] — {year}", fontsize=12)

    out = PLOTS_DIR / "maps" / f"{substance}_{sector}" / f"{year:04d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    logger.info("Saved map → %s", out)


def plot_time_series(substance: str, sector: str, long_name: str, units: str,
                     years: list[int], means: list[float]):
    """Plot global mean flux over time."""
    out = PLOTS_DIR / "timeseries" / f"{substance}_{sector}.png"
    if out.exists():
        return
    if not means:
        return

    y = np.array(years[:len(means)])
    v = np.array(means[:len(y)])

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(y, v, linewidth=1.5, color="#d62728", marker="o", markersize=3)
    ax.fill_between(y, 0, v, alpha=0.15, color="#d62728")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"Global mean flux ({units})", fontsize=10)
    ax.set_title(f"EDGAR {substance} [{sector}] — {long_name}", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(y[0], y[-1])

    out = PLOTS_DIR / "timeseries" / f"{substance}_{sector}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved time series → %s", out)


# ── Pipeline ─────────────────────────────────────────────────────────────────

@click.command()
@click.option("--substances", "-s", multiple=True, help="Substance(s) (e.g. CO2 CH4).")
@click.option("--sectors", "-S", multiple=True, help="Sector(s) (e.g. TOTALS ENE).")
@click.option("--all", "run_all", is_flag=True, help="Process all substances/sectors.")
@click.option("--start-year", type=int, default=None, help="Start year (default: 1970).")
@click.option("--end-year", type=int, default=None, help="End year (default: 2024).")
@click.option("--plot-every", type=int, default=5,
              help="Plot spatial maps every N years (default: 5).")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--cleanup-raw", is_flag=True, default=True,
              help="Delete raw NC files after regridding (default: true).")
@click.option("--no-cleanup", is_flag=True, help="Keep raw NC files after regridding.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
def main(substances, sectors, run_all, start_year, end_year, plot_every,
         skip_download, cleanup_raw, no_cleanup, overwrite):
    """EDGAR full pipeline: download → regrid → cleanup → visualize."""
    config = load_edgar_config()
    base_url = config["source_url_base"]
    t_range = config["temporal_range"]
    all_sectors_info = config["sectors"]

    if no_cleanup:
        cleanup_raw = False

    # Resolve substances
    if not substances and not run_all:
        click.echo("Specify --substances or --all. Use --help for usage.")
        return

    sub_list = list(substances) if substances else list(config["substances"].keys())

    # Resolve years
    y_start = start_year or t_range[0]
    y_end = end_year or t_range[1]
    years = list(range(y_start, y_end + 1))

    # Pick which years get spatial map plots
    plot_years = set(range(y_start, y_end + 1, plot_every))
    plot_years.add(y_end)

    click.echo(f"Pipeline: {len(sub_list)} substances, years {y_start}-{y_end}, "
               f"maps every {plot_every} years")

    for sub in sub_list:
        sub_info = config["substances"].get(sub)
        if sub_info is None:
            logger.warning("Unknown substance: %s", sub)
            continue

        available_sectors = sub_info["sectors"]
        if sectors:
            sec_list = [s for s in sectors if s in available_sectors]
            skipped_sec = [s for s in sectors if s not in available_sectors]
            if skipped_sec:
                logger.info("Sectors not available for %s: %s", sub, skipped_sec)
        else:
            sec_list = available_sectors

        for sec in sec_list:
            sec_info = all_sectors_info.get(sec, {})
            sec_long_name = sec_info.get("long_name", sec)
            long_name = sub_info["long_name"]
            units = sub_info["units"]

            # Skip entirely if all outputs and plots already exist
            var_dir = FINAL_DIR / f"{sub}_{sec}"
            ts_plot = PLOTS_DIR / "timeseries" / f"{sub}_{sec}.png"
            all_final_exist = all(
                (var_dir / f"{y:04d}.nc").exists() for y in years
            )
            all_maps_exist = all(
                (PLOTS_DIR / "maps" / f"{sub}_{sec}" / f"{y:04d}.png").exists()
                for y in years if y in plot_years
            )
            if all_final_exist and all_maps_exist and ts_plot.exists():
                click.echo(f"\n[{sub}/{sec}] Already complete, skipping")
                continue

            click.echo(f"\n[{sub}/{sec}] {long_name} — {sec_long_name}")

            ts_years = []
            ts_means = []

            def _collect_mean(da):
                val = float(da.mean(skipna=True).values)
                ts_means.append(val)

            for year in tqdm(years, desc=f"  {sub}/{sec}", unit="yr"):
                final_path = FINAL_DIR / f"{sub}_{sec}" / f"{year:04d}.nc"

                # If final already exists, just collect stats
                if final_path.exists() and not overwrite:
                    try:
                        out_ds = xr.open_dataset(final_path)
                        var_name = _data_var_name(out_ds, f"{sub}_{sec}")
                        ts_years.append(year)
                        _collect_mean(out_ds[var_name])
                        out_ds.close()
                    except Exception:
                        pass
                else:
                    # Download
                    raw_path = None
                    if not skip_download:
                        raw_dir = RAW_DIR / sub / sec
                        url = build_url(base_url, sub, sec, year)
                        raw_path = download_file(url, raw_dir, sub, sec, year,
                                                 overwrite=overwrite)

                    # Find raw file (may already exist from previous download)
                    if raw_path is None:
                        nc_name = expected_nc_name(sub, sec, year)
                        candidate = RAW_DIR / sub / sec / nc_name
                        if candidate.exists():
                            raw_path = candidate

                    if raw_path is None:
                        continue

                    # Process (regrid)
                    result = process_file(
                        raw_path, sub, sec, year,
                        units=units,
                        long_name=long_name,
                        sector_long_name=sec_long_name,
                        overwrite=overwrite,
                    )

                    # Cleanup raw immediately to save storage
                    if cleanup_raw and raw_path.exists():
                        raw_path.unlink(missing_ok=True)

                    # Collect stats from processed file
                    if result and result.exists():
                        try:
                            out_ds = xr.open_dataset(result)
                            var_name = _data_var_name(out_ds, f"{sub}_{sec}")
                            ts_years.append(year)
                            _collect_mean(out_ds[var_name])
                            out_ds.close()
                        except Exception:
                            pass

                # Spatial map
                if year in plot_years:
                    try:
                        plot_spatial_map(sub, sec, year, long_name, units)
                    except Exception as e:
                        logger.warning("Map plot failed %s/%s/%d: %s", sub, sec, year, e)

            # Time series plot
            if ts_years:
                try:
                    plot_time_series(sub, sec, long_name, units, ts_years, ts_means)
                except Exception as e:
                    logger.warning("Time series plot failed %s/%s: %s", sub, sec, e)

            gc.collect()
            plt.close("all")

    # ── Aggregate virtual sectors (e.g. Aviation) ─────────────────────────
    aggregates = config.get("aggregates", {})
    for agg_name, agg_info in aggregates.items():
        components = agg_info["components"]
        for sub in sub_list:
            sub_info = config["substances"].get(sub)
            if sub_info is None:
                continue
            # Only aggregate if this substance has all component sectors
            available = sub_info["sectors"]
            if not all(c in available for c in components):
                continue

            click.echo(f"\n[{sub}/{agg_name}] Aggregating {', '.join(components)}")

            ts_years = []
            ts_means = []

            for year in tqdm(years, desc=f"  {sub}/{agg_name}", unit="yr"):
                result = aggregate_sectors(
                    sub, agg_name, agg_info, sub_info, year, overwrite=overwrite,
                )
                # Collect stats (from new or existing file)
                final_path = FINAL_DIR / f"{sub}_{agg_name}" / f"{year:04d}.nc"
                if final_path.exists():
                    try:
                        out_ds = xr.open_dataset(final_path)
                        var_name = _data_var_name(out_ds, f"{sub}_{agg_name}")
                        ts_years.append(year)
                        ts_means.append(float(out_ds[var_name].mean(skipna=True).values))
                        out_ds.close()
                    except Exception:
                        pass

                if year in plot_years:
                    try:
                        plot_spatial_map(sub, agg_name, year,
                                         sub_info["long_name"], sub_info["units"])
                    except Exception as e:
                        logger.warning("Map plot failed %s/%s/%d: %s",
                                       sub, agg_name, year, e)

            if ts_years:
                try:
                    plot_time_series(sub, agg_name, sub_info["long_name"],
                                     sub_info["units"], ts_years, ts_means)
                except Exception as e:
                    logger.warning("Time series failed %s/%s: %s", sub, agg_name, e)

            gc.collect()
            plt.close("all")

    # Clean up empty raw directories
    if cleanup_raw and RAW_DIR.exists():
        for d in sorted(RAW_DIR.rglob("*"), reverse=True):
            if d.is_dir():
                try:
                    d.rmdir()  # only removes empty dirs
                except OSError:
                    pass

    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
