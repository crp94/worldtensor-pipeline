"""Full LUH3 pipeline: ingest -> process -> visualize.

Stages LUH3 input4MIPs files, extracts/regrids each states variable/year to
the master grid, and generates spatial maps plus time-series plots.

Usage:
    python -m src.pipelines.luh3 --all --source-dir ~/Downloads
    python -m src.pipelines.luh3 -v primf -v urban --start-year 1950 --plot-every 10 --skip-download
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
import yaml
from tqdm import tqdm

from src.download.luh3 import (
    ingest_luh3_files,
    download_from_wget_scripts,
    download_from_esgf_discovery,
    download_from_config_fallback_urls,
    DEFAULT_ESGF_SEARCH_URLS,
    DEFAULT_OUTPUT_DIR as RAW_DIR,
)
from src.processing.luh3_states_to_yearly import (
    FRACTION_VARS,
    build_year_to_index,
    load_land_budget,
    process_fraction_year,
    process_nonfraction_year,
)
from src.utils import get_logger
from src.year_policy import resolve_year_bounds

logger = get_logger("pipeline.luh3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "luh3.yml"
RAW_STATES_PATH = RAW_DIR / "states.nc"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "land_use" / "states"
PLOTS_DIR = PROJECT_ROOT / "plots" / "luh3"

CMAPS = {
    "land_use_fraction": "YlGn",
    "secondary_forest": "YlOrBr",
}


def load_luh3_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def plot_spatial_map(var_name: str, var_info: dict, year: int) -> None:
    nc_path = FINAL_DIR / var_name / f"{year:04d}.nc"
    if not nc_path.exists():
        return

    ds = xr.open_dataset(nc_path)
    da = ds[var_name]
    category = var_info.get("category", "land_use_fraction")
    cmap = CMAPS.get(category, "YlGn")

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.Robinson()})

    plot_kwargs = {
        "ax": ax,
        "transform": ccrs.PlateCarree(),
        "cmap": cmap,
        "add_colorbar": False,
    }

    if var_name in FRACTION_VARS:
        plot_kwargs["vmin"] = 0
        plot_kwargs["vmax"] = 1
    else:
        plot_kwargs["robust"] = True

    im = da.plot(**plot_kwargs)
    plt.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        aspect=40,
        shrink=0.8,
        label=var_info.get("units", ""),
    )
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.set_title(f"{var_info['long_name']} - {year}", fontsize=12)

    out = PLOTS_DIR / "maps" / var_name / f"{year:04d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    logger.info("Saved map -> %s", out)


def plot_time_series(var_name: str, var_info: dict, years: list[int], quantiles: dict[str, list[float]]) -> None:
    if not quantiles.get("p50"):
        return

    y = np.array(years[:len(quantiles["p50"])])
    n = len(y)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(y, quantiles["p10"][:n], quantiles["p90"][:n], alpha=0.12, color="#1f77b4", label="10th-90th pctl")
    ax.fill_between(y, quantiles["p25"][:n], quantiles["p75"][:n], alpha=0.3, color="#1f77b4", label="IQR (25th-75th)")
    ax.plot(y, quantiles["p50"][:n], linewidth=1.2, color="#1f77b4", label="Median")
    ax.set_xlabel("Year")
    ax.set_ylabel(f"{var_info['long_name']} ({var_info['units']})", fontsize=10)
    ax.set_title(f"LUH3 {var_name} - Spatial Distribution Time Series", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    out = PLOTS_DIR / "timeseries" / f"{var_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved time series -> %s", out)


@click.command()
@click.option("--variables", "-v", multiple=True, help="Variable names (e.g. primf urban).")
@click.option("--all", "run_all", is_flag=True, help="Process all configured LUH3 states variables.")
@click.option("--start-year", type=int, default=None, help="Start year (default: from config).")
@click.option("--end-year", type=int, default=None, help="End year (default: from config).")
@click.option("--plot-every", type=int, default=20, show_default=True, help="Plot spatial maps every N years.")
@click.option("--wget-script", "wget_scripts", multiple=True, type=click.Path(exists=True),
              help="Path(s) to ESGF-generated wget scripts.")
@click.option("--discover/--no-discover", default=True, show_default=True,
              help="Auto-discover LUH3 file URLs from ESGF search API when wget scripts are not provided.")
@click.option(
    "--esgf-search-url",
    default=",".join(DEFAULT_ESGF_SEARCH_URLS),
    show_default=True,
    help="Comma-separated ESGF search API endpoint(s) used for LUH3 auto-discovery.",
)
@click.option("--source-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None,
              help="Directory containing manually downloaded LUH3 files.")
@click.option("--skip-download", is_flag=True, help="Skip LUH3 file ingest step.")
@click.option("--cleanup-raw", is_flag=True, help="Delete staged LUH3 files after processing.")
@click.option("--no-verify-checksum", is_flag=True, help="Skip checksum verification for network downloads.")
def main(
    variables,
    run_all,
    start_year,
    end_year,
    plot_every,
    wget_scripts,
    discover,
    esgf_search_url,
    source_dir,
    skip_download,
    cleanup_raw,
    no_verify_checksum,
):
    """LUH3 full pipeline: ingest -> process -> visualize."""
    config = load_luh3_config()
    all_vars = config["variables"]
    t_range = config["temporal_range"]

    if variables:
        var_list = {v: all_vars[v] for v in variables if v in all_vars}
        missing = [v for v in variables if v not in all_vars]
        if missing:
            logger.warning("Unknown variables (skipped): %s", missing)
    elif run_all:
        var_list = all_vars
    else:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    y_start, y_end = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=t_range[0],
        default_end=t_range[1],
        label="LUH3 pipeline years",
    )

    click.echo(f"Pipeline: {len(var_list)} variables, years {y_start}-{y_end}")

    if not skip_download:
        click.echo("Step 1: Ingesting LUH3 files ...")
        wanted_kinds = ("states", "transitions", "static")
        staged: dict[str, Path] = {}
        missing: list[str] = []
        if wget_scripts:
            script_paths = [Path(p).expanduser().resolve() for p in wget_scripts]
            dl_staged, dl_missing = download_from_wget_scripts(
                wget_scripts=script_paths,
                raw_dir=RAW_DIR,
                overwrite=False,
                verify_checksum=not no_verify_checksum,
                kinds=wanted_kinds,
            )
            staged.update(dl_staged)
            missing.extend(dl_missing)
        elif discover and source_dir is None:
            try:
                dl_staged, dl_missing = download_from_esgf_discovery(
                    raw_dir=RAW_DIR,
                    overwrite=False,
                    verify_checksum=not no_verify_checksum,
                    esgf_search_url=esgf_search_url,
                    kinds=wanted_kinds,
                )
            except click.ClickException as e:
                logger.warning("ESGF discovery failed; trying direct URL fallback: %s", e)
                dl_staged, dl_missing = download_from_config_fallback_urls(
                    raw_dir=RAW_DIR,
                    overwrite=False,
                    verify_checksum=not no_verify_checksum,
                    kinds=wanted_kinds,
                )
            staged.update(dl_staged)
            missing.extend(dl_missing)

        staged2, missing2 = ingest_luh3_files(
            raw_dir=RAW_DIR,
            source_dir=Path(source_dir).expanduser().resolve() if source_dir else None,
            explicit_paths={},
            overwrite=False,
            kinds=wanted_kinds,
        )
        staged.update(staged2)
        missing = sorted(set([k for k in (missing + missing2) if k not in staged]))
        click.echo(f"  staged {len(staged)}/{len(wanted_kinds)} files")
        if missing:
            raise click.ClickException(
                f"Missing LUH3 files: {missing}. Run: python -m src.download.luh3 --all --source-dir <download_dir>"
            )
    else:
        click.echo("Step 1: Ingest skipped")

    if not RAW_STATES_PATH.exists():
        click.echo(f"Source states file not found: {RAW_STATES_PATH}")
        return

    click.echo("Step 2: Opening states.nc ...")
    ds = xr.open_dataset(RAW_STATES_PATH, decode_times=False, chunks={"time": 1})
    year_to_idx = build_year_to_index(ds)

    years = [y for y in range(y_start, y_end + 1) if y in year_to_idx]
    if not years:
        click.echo(f"No matching years in range {y_start}-{y_end}")
        ds.close()
        return

    plot_years = set(range(y_start, y_end + 1, plot_every))
    plot_years.add(y_end)
    click.echo(f"  {len(years)} years, spatial maps every {plot_every} years")

    fraction_vars = {v: info for v, info in var_list.items() if v in FRACTION_VARS}
    land_budget = load_land_budget() if fraction_vars else None
    if fraction_vars:
        click.echo("Step 2a: Processing fraction states with periodic regrid + land-budget normalization ...")
        for year in tqdm(years, desc="fraction_states"):
            idx = year_to_idx[year]
            process_fraction_year(ds, fraction_vars, idx, year, overwrite=False, land_budget=land_budget)

    for var_idx, (var_name, var_info) in enumerate(var_list.items(), 1):
        click.echo(f"\n[{var_idx}/{len(var_list)}] {var_name} ({var_info['long_name']})")

        if var_name not in ds.data_vars:
            logger.warning("Variable %s not found in dataset, skipping", var_name)
            continue

        ts_years = []
        ts_quantiles = {k: [] for k in ("p10", "p25", "p50", "p75", "p90")}

        def _collect_quantiles(da: xr.DataArray) -> None:
            vals = da.values.ravel()
            vals = vals[~np.isnan(vals)]
            if len(vals) == 0:
                return
            p10, p25, p50, p75, p90 = np.percentile(vals, [10, 25, 50, 75, 90])
            ts_quantiles["p10"].append(float(p10))
            ts_quantiles["p25"].append(float(p25))
            ts_quantiles["p50"].append(float(p50))
            ts_quantiles["p75"].append(float(p75))
            ts_quantiles["p90"].append(float(p90))

        for year in tqdm(years, desc=f"  {var_name}"):
            idx = year_to_idx[year]
            out_path = FINAL_DIR / var_name / f"{year:04d}.nc"

            if out_path.exists():
                try:
                    out_ds = xr.open_dataset(out_path)
                    ts_years.append(year)
                    _collect_quantiles(out_ds[var_name])
                    out_ds.close()
                except Exception:
                    pass
            else:
                if var_name in FRACTION_VARS:
                    result = None
                else:
                    result = process_nonfraction_year(ds, var_name, var_info, idx, year, overwrite=False)
                if result:
                    try:
                        out_ds = xr.open_dataset(result)
                        ts_years.append(year)
                        _collect_quantiles(out_ds[var_name])
                        out_ds.close()
                    except Exception:
                        pass

            if year in plot_years:
                try:
                    plot_spatial_map(var_name, var_info, year)
                except Exception as e:
                    logger.warning("Map plot failed %s/%d: %s", var_name, year, e)

        if ts_years:
            try:
                plot_time_series(var_name, var_info, ts_years, ts_quantiles)
            except Exception as e:
                logger.warning("Time series plot failed %s: %s", var_name, e)

        gc.collect()
        plt.close("all")

    ds.close()

    if cleanup_raw and RAW_DIR.exists():
        for p in RAW_DIR.glob("*.nc"):
            p.unlink(missing_ok=True)
        click.echo(f"Cleaned up {RAW_DIR}/*.nc")

    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
