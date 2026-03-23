"""Full LUH3 transitions pipeline: ingest -> process -> visualize.

Usage:
    python -m src.pipelines.luh3_transitions --all
    python -m src.pipelines.luh3_transitions -v primf_to_secdn --start-year 1800 --end-year 2023 --plot-every 20
"""

from __future__ import annotations

import gc
import shutil
from pathlib import Path
from textwrap import dedent

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
from src.processing.luh3_states_to_yearly import build_year_to_index
from src.processing.luh3_transitions_to_yearly import process_transition_year, list_transition_variables
from src.utils import get_logger
from src.year_policy import resolve_year_bounds

logger = get_logger("pipeline.luh3_transitions")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "luh3.yml"
RAW_PATH = RAW_DIR / "transitions.nc"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "land_use" / "transitions"
PLOTS_DIR = PROJECT_ROOT / "plots" / "luh3_transitions"
INACTIVE_CACHE_PATH = FINAL_DIR / "_inactive_transitions.txt"
CURATION_MANIFEST_PATH = FINAL_DIR / "_curation_manifest.yml"
README_PATH = FINAL_DIR / "README.md"


def load_luh3_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_inactive_cache(path: Path = INACTIVE_CACHE_PATH) -> set[str]:
    if not path.exists():
        return set()
    return {line.strip() for line in path.read_text().splitlines() if line.strip()}


def save_inactive_cache(names: set[str], path: Path = INACTIVE_CACHE_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{name}\n" for name in sorted(names)))


def write_curation_metadata(
    *,
    all_transition_vars: list[str],
    inactive_vars: set[str],
    activity_threshold: float,
    year_range: tuple[int, int],
    plot_years: list[int],
    prune_inactive: bool,
    manifest_path: Path = CURATION_MANIFEST_PATH,
    readme_path: Path = README_PATH,
) -> None:
    """Write a machine-readable manifest and a readable note for curated LUH3 transitions."""
    active_names = sorted(p.name for p in FINAL_DIR.iterdir() if p.is_dir())
    inactive_names = sorted(inactive_vars)
    rel_raw = RAW_PATH.relative_to(PROJECT_ROOT)
    rel_final = FINAL_DIR.relative_to(PROJECT_ROOT)
    rel_cache = INACTIVE_CACHE_PATH.relative_to(PROJECT_ROOT)

    manifest = {
        "dataset": "LUH3 transitions",
        "curated_output_dir": str(rel_final),
        "raw_source_file": str(rel_raw),
        "raw_transition_variable_count": len(all_transition_vars),
        "curated_transition_variable_count": len(active_names),
        "inactive_transition_variable_count": len(inactive_names),
        "inactive_selection_rule": f"global max <= {activity_threshold}",
        "inactive_selection_applies_to_curated_final_only": True,
        "inactive_removed_from_curated_final": bool(prune_inactive),
        "inactive_retained_in_raw_source": True,
        "inactive_list_file": str(rel_cache),
        "year_range": [year_range[0], year_range[1]],
        "standard_plot_years": plot_years,
        "active_transition_variables": active_names,
        "inactive_transition_variables": inactive_names,
    }
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False))

    note = dedent(
        f"""\
        # LUH3 Transitions Curation Note

        This directory is a curated subset of the raw LUH3 transitions source.

        - Raw source file: `{rel_raw}`
        - Curated output directory: `{rel_final}`
        - Raw transition variables available: `{len(all_transition_vars)}`
        - Transition variables kept in `data/final`: `{len(active_names)}`
        - Transition variables omitted from `data/final`: `{len(inactive_names)}`
        - Omission rule: global max `<= {activity_threshold}` across the full raw LUH3 cube
        - Omitted variable ids: `{rel_cache}`
        - This omission applies only to the curated `data/final` layer. The raw LUH3 source file is kept intact.
        - Current standard plot years: `{", ".join(str(y) for y in plot_years)}`

        The machine-readable version of this note is in `_curation_manifest.yml`.
        """
    )
    readme_path.write_text(note)


def scan_transition_global_max(ds: xr.Dataset, var_name: str) -> float:
    """Compute global max for a transition variable robustly."""
    arr = ds[var_name].max(skipna=True)
    if hasattr(arr, "compute"):
        try:
            import dask
            with dask.config.set(scheduler="single-threaded"):
                return float(arr.compute())
        except Exception:
            return float(arr.compute())
    return float(arr.values)


def plot_spatial_map(var_name: str, var_info: dict, year: int) -> None:
    nc_path = FINAL_DIR / var_name / f"{year:04d}.nc"
    if not nc_path.exists():
        return

    ds = xr.open_dataset(nc_path)
    da = ds[var_name]
    vmax = float(np.nanpercentile(da.values, 99))
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 0.01
    vmax = min(max(vmax, 1e-4), 1.0)

    fig, ax = plt.subplots(figsize=(12, 6), subplot_kw={"projection": ccrs.Robinson()})
    im = da.plot(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap="YlOrRd",
        vmin=0,
        vmax=vmax,
        add_colorbar=False,
    )
    plt.colorbar(
        im,
        ax=ax,
        orientation="horizontal",
        pad=0.05,
        aspect=40,
        shrink=0.8,
        label=var_info.get("units", "1"),
    )
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.set_title(f"{var_name} - {year}", fontsize=11)

    out = PLOTS_DIR / "maps" / var_name / f"{year:04d}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    ds.close()
    logger.info("Saved map -> %s", out)


def plot_time_series(var_name: str, years: list[int], quantiles: dict[str, list[float]]) -> None:
    if not quantiles.get("p50"):
        return

    y = np.array(years[:len(quantiles["p50"])])
    n = len(y)

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.fill_between(y, quantiles["p10"][:n], quantiles["p90"][:n], alpha=0.12, color="#d62728", label="10th-90th pctl")
    ax.fill_between(y, quantiles["p25"][:n], quantiles["p75"][:n], alpha=0.3, color="#d62728", label="IQR (25th-75th)")
    ax.plot(y, quantiles["p50"][:n], linewidth=1.2, color="#d62728", label="Median")
    ax.set_xlabel("Year")
    ax.set_ylabel("Area fraction (1)", fontsize=10)
    ax.set_title(f"LUH3 transition {var_name} - Spatial Distribution Time Series", fontsize=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, alpha=0.3)

    out = PLOTS_DIR / "timeseries" / f"{var_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved time series -> %s", out)


@click.command()
@click.option("--variables", "-v", multiple=True, help="Transition variable names.")
@click.option("--all", "run_all", is_flag=True, help="Process all transition variables.")
@click.option("--start-year", type=int, default=None, help="Start year (default: from config).")
@click.option("--end-year", type=int, default=None, help="End year (default: from config and file availability).")
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
@click.option("--exclude-substring", multiple=True, help="Skip any transition variable containing this substring.")
@click.option("--skip-never-happens/--no-skip-never-happens", default=True, show_default=True,
              help="Skip transitions that are zero everywhere across all years/grid cells.")
@click.option("--activity-threshold", type=float, default=0.0, show_default=True,
              help="Transition considered active only if global max > threshold.")
@click.option("--prune-inactive/--no-prune-inactive", default=True, show_default=True,
              help="Delete stale outputs/plots for transitions identified as inactive.")
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
    exclude_substring,
    skip_never_happens,
    activity_threshold,
    prune_inactive,
):
    """LUH3 transitions full pipeline: ingest -> process -> visualize."""
    config = load_luh3_config()
    t_range = config["temporal_range"]
    y_start, y_end_req = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=t_range[0],
        default_end=t_range[1],
        label="LUH3 transitions pipeline years",
    )

    if not skip_download:
        click.echo("Step 1: Ingesting LUH3 files (states + transitions) ...")
        wanted_kinds = ("states", "transitions")
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

    if not RAW_PATH.exists():
        click.echo(f"Source transitions file not found: {RAW_PATH}")
        return

    click.echo("Step 2: Opening transitions.nc ...")
    ds = xr.open_dataset(RAW_PATH, decode_times=False, chunks={"time": 1})
    all_vars = list_transition_variables(ds)

    if variables:
        var_list = [v for v in variables if v in all_vars]
        missing = [v for v in variables if v not in all_vars]
        if missing:
            logger.warning("Unknown transition variables (skipped): %s", missing)
    elif run_all:
        var_list = all_vars
    else:
        click.echo("Specify --variables or --all. Use --help for usage.")
        ds.close()
        return

    if exclude_substring:
        var_list = [v for v in var_list if not any(s in v for s in exclude_substring)]

    inactive_cache = load_inactive_cache() if skip_never_happens else set()
    if skip_never_happens and inactive_cache:
        before = len(var_list)
        var_list = [v for v in var_list if v not in inactive_cache]
        skipped_cached = before - len(var_list)
        if skipped_cached > 0:
            click.echo(f"Step 2b: Skipping {skipped_cached} cached inactive transitions")

    if skip_never_happens and var_list:
        click.echo("Step 2c: Pre-scanning transition activity ...")
        active_vars: list[str] = []
        skipped_inactive = 0
        for var_name in tqdm(var_list, desc="  activity"):
            try:
                vmax = scan_transition_global_max(ds, var_name)
            except Exception as e:
                logger.warning("Activity scan failed for %s (%s); keeping variable.", var_name, e)
                active_vars.append(var_name)
                continue

            if np.isfinite(vmax) and vmax > activity_threshold:
                active_vars.append(var_name)
            else:
                skipped_inactive += 1
                inactive_cache.add(var_name)
                if prune_inactive:
                    shutil.rmtree(FINAL_DIR / var_name, ignore_errors=True)
                    shutil.rmtree(PLOTS_DIR / "maps" / var_name, ignore_errors=True)
                    (PLOTS_DIR / "timeseries" / f"{var_name}.png").unlink(missing_ok=True)

        var_list = active_vars
        if skipped_inactive > 0:
            click.echo(
                f"  skipped {skipped_inactive} inactive transitions "
                f"(global max <= {activity_threshold})"
            )

    year_to_idx = build_year_to_index(ds)
    year_max_avail = max(year_to_idx) if year_to_idx else y_end_req
    y_end = min(y_end_req, year_max_avail)
    years = [y for y in range(y_start, y_end + 1) if y in year_to_idx]
    if not years:
        click.echo(f"No matching years in range {y_start}-{y_end}")
        ds.close()
        return

    click.echo(f"Pipeline: {len(var_list)} transitions, years {years[0]}-{years[-1]}")
    plot_years = set(range(years[0], years[-1] + 1, plot_every))
    plot_years.add(years[-1])
    click.echo(f"  spatial maps every {plot_every} years")

    for var_idx, var_name in enumerate(var_list, 1):
        click.echo(f"\n[{var_idx}/{len(var_list)}] {var_name}")
        units = ds[var_name].attrs.get("units", "1")
        var_info = {"units": units}

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
                result = process_transition_year(ds, var_name, idx, year, overwrite=False)
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
                plot_time_series(var_name, ts_years, ts_quantiles)
            except Exception as e:
                logger.warning("Time series plot failed %s: %s", var_name, e)

        gc.collect()
        plt.close("all")

    ds.close()

    if skip_never_happens:
        save_inactive_cache(inactive_cache)

    write_curation_metadata(
        all_transition_vars=all_vars,
        inactive_vars=inactive_cache if skip_never_happens else load_inactive_cache(),
        activity_threshold=activity_threshold,
        year_range=(years[0], years[-1]),
        plot_years=sorted(plot_years),
        prune_inactive=prune_inactive,
    )

    if cleanup_raw and RAW_DIR.exists():
        for name in ("states.nc", "transitions.nc"):
            p = RAW_DIR / name
            if p.exists():
                p.unlink(missing_ok=True)
        click.echo(f"Cleaned up {RAW_DIR}/states.nc and transitions.nc")

    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
