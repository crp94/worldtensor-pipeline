"""Full ERA5 pipeline: download → process → cleanup → visualize.

Downloads each variable/year, processes to yearly derived maps, deletes raw files,
generates spatial map plots every N years, and plots full time series per variable.

Note: 10 ERA5 variables are excluded from processing due to sentinel-value
contamination (bld, gwd, ewss, nsss, lgws, mgws, lspf, slhf, sshf, rsn).
See ``config/era5.yml`` excluded_variables and the docstring in
``src/processing/era5_monthly_to_yearly.py`` for details.

Usage:
    python -m src.pipelines.era5 --all
    python -m src.pipelines.era5 --all --plot-every 10 --workers 6
    python -m src.pipelines.era5 -v t2m -v tp --start-year 2000
"""

import gc
import time as time_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import cdsapi
import matplotlib
matplotlib.use("Agg")
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from src.utils import get_logger, add_cyclic_point_xr
from src.year_policy import resolve_year_bounds

logger = get_logger("pipeline.era5")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "era5.yml"
SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "era5"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "climate"
PLOTS_DIR = PROJECT_ROOT / "plots" / "era5"


# ── Config / secrets ─────────────────────────────────────────────────────────

def load_era5_config() -> dict:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    excluded = {str(v) for v in config.get("excluded_variables", [])}
    if excluded:
        config["variables"] = {k: v for k, v in config.get("variables", {}).items() if k not in excluded}
    return config


def load_secrets() -> dict:
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            f"Secrets file not found: {SECRETS_PATH}\n"
            f"Copy config/secrets.yml.example → config/secrets.yml and add your CDS API key."
        )
    with open(SECRETS_PATH) as f:
        return yaml.safe_load(f)


# ── Download ─────────────────────────────────────────────────────────────────

def _make_client(secrets: dict) -> cdsapi.Client:
    return cdsapi.Client(url=secrets["cds"]["url"], key=secrets["cds"]["key"])


def download_one(secrets, cds_name, short_name, year, dataset, product_type, max_retries=5):
    """Download one variable/year. Thread-safe — creates its own CDS client.

    Retries with exponential backoff on rate-limit or transient errors.
    """
    out_path = RAW_DIR / short_name / f"{year}.nc"
    if out_path.exists():
        return year, out_path, False  # already existed

    out_path.parent.mkdir(parents=True, exist_ok=True)
    request = {
        "product_type": [product_type],
        "variable": [cds_name],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    for attempt in range(1, max_retries + 1):
        try:
            client = _make_client(secrets)
            client.retrieve(dataset, request, str(out_path))
            return year, out_path, True
        except Exception as e:
            if out_path.exists():
                out_path.unlink()
            err_str = str(e)
            is_rate_limit = "queued requests" in err_str or "temporarily limited" in err_str
            is_transient = "500" in err_str or "502" in err_str or "503" in err_str or "timeout" in err_str.lower()
            if (is_rate_limit or is_transient) and attempt < max_retries:
                wait = min(30 * attempt, 120)
                logger.warning("Retry %d/%d for %s/%d (waiting %ds): %s",
                               attempt, max_retries, short_name, year, wait, err_str[:120])
                time_mod.sleep(wait)
            else:
                logger.error("Failed %s/%d after %d attempts: %s", short_name, year, attempt, err_str[:200])
                return year, None, False

    return year, None, False


def download_years_parallel(
    secrets, cds_name, short_name, years, dataset, product_type, workers,
):
    """Download multiple years for one variable in parallel. Returns {year: path}."""
    results = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for year in years:
            fut = pool.submit(
                download_one, secrets, cds_name, short_name, year, dataset, product_type,
            )
            futures[fut] = year

        for fut in as_completed(futures):
            year, path, was_new = fut.result()
            if path:
                results[year] = path

    return results


# ── Processing ───────────────────────────────────────────────────────────────

def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename["longitude"] = "lon"
    if "valid_time" in ds.dims:
        rename["valid_time"] = "time"
    if rename:
        ds = ds.rename(rename)
    if ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon.values % 360))
        ds = ds.sortby("lon")
    return ds


def _find_data_var(ds: xr.Dataset, short_name: str) -> str:
    if short_name in ds.data_vars:
        return short_name
    data_vars = [v for v in ds.data_vars if v not in ("time_bnds", "expver")]
    if data_vars:
        return data_vars[0]
    raise ValueError(f"No data variables found for '{short_name}'")


def _save_derived(data, lat, lon, var_name, stat_name, long_name, units, year) -> Path:
    folder_name = f"{var_name}_{stat_name}"
    out_path = FINAL_DIR / folder_name / f"{year}.nc"
    if out_path.exists():
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {
            folder_name: (
                ["lat", "lon"],
                data.astype(np.float32),
                {"units": units, "long_name": f"{long_name} ({stat_name})"},
            )
        },
        coords={
            "lat": ("lat", lat, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", lon, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor ERA5 {folder_name}",
            "source": "ERA5 reanalysis (ECMWF)",
            "aggregation_method": stat_name,
            "year": year,
        },
    )
    ds.to_netcdf(out_path, encoding={folder_name: {"zlib": True, "complevel": 4, "dtype": "float32"}})
    return out_path


def outputs_exist(short_name: str, aggregation: str, year: int) -> bool:
    """Check if all derived outputs already exist for a variable/year."""
    return all(
        (FINAL_DIR / f"{short_name}_{s}" / f"{year}.nc").exists()
        for s in _get_stat_names(aggregation)
    )


def read_means_from_outputs(short_name: str, aggregation: str, year: int) -> dict | None:
    """Read spatial means from existing output files (no raw file needed)."""
    stat_names = _get_stat_names(aggregation)
    means = {}
    for s in stat_names:
        p = FINAL_DIR / f"{short_name}_{s}" / f"{year}.nc"
        if not p.exists():
            return None
        ds = xr.open_dataset(p)
        means[s] = float(ds[list(ds.data_vars)[0]].mean(skipna=True))
        ds.close()
    return means


def process_raw_file(short_name: str, var_info: dict, year: int) -> dict | None:
    """Process one raw file → 4 derived maps.

    Returns dict of {stat_name: spatial_mean} for time series, or None on failure.
    """
    raw_path = RAW_DIR / short_name / f"{year}.nc"
    if not raw_path.exists():
        return None

    aggregation = var_info["aggregation"]
    long_name = var_info["long_name"]
    units = var_info["units"]

    # All stat names this variable produces
    stat_names = _get_stat_names(aggregation)

    # Check if all outputs already exist
    if outputs_exist(short_name, aggregation, year):
        return read_means_from_outputs(short_name, aggregation, year)

    try:
        ds = xr.open_dataset(raw_path)
        ds = _normalize_coords(ds)
        data_var = _find_data_var(ds, short_name)
        da = ds[data_var]
        time_dim = "time" if "time" in da.dims else da.dims[0]
        lat, lon = ds.lat.values, ds.lon.values

        # Compute all stats
        computed = {}
        if aggregation == "mean":
            computed["mean"] = da.mean(dim=time_dim).values
        elif aggregation == "sum":
            computed["sum"] = da.sum(dim=time_dim).values
        elif aggregation == "max":
            computed["max"] = da.max(dim=time_dim).values
        elif aggregation == "min":
            computed["min"] = da.min(dim=time_dim).values

        computed["std"] = da.std(dim=time_dim).values
        if "max" not in computed:
            computed["max"] = da.max(dim=time_dim).values
        if "min" not in computed:
            computed["min"] = da.min(dim=time_dim).values
        ds.close()
    except Exception as e:
        logger.error("Corrupt file %s/%d, deleting: %s", short_name, year, e)
        raw_path.unlink(missing_ok=True)
        return None

    # Save and collect spatial means
    means = {}
    for stat_name, data in computed.items():
        _save_derived(data, lat, lon, short_name, stat_name, long_name, units, year)
        means[stat_name] = float(np.nanmean(data))

    return means


def _get_stat_names(aggregation: str) -> list[str]:
    """Return all stat names produced for a given aggregation type."""
    stats = [aggregation, "std"]
    if aggregation != "max":
        stats.append("max")
    if aggregation != "min":
        stats.append("min")
    return stats


# ── Visualization ────────────────────────────────────────────────────────────

CMAPS = {
    "temperature": "RdYlBu_r", "pressure": "viridis", "wind": "viridis",
    "cloud": "Blues", "atmospheric": "plasma", "precipitation": "Blues",
    "evaporation": "BrBG", "radiation": "inferno", "heat_flux": "RdBu_r",
    "stress": "PuOr", "moisture": "BrBG", "snow": "Blues", "soil": "YlGnBu",
    "vegetation": "Greens", "albedo": "YlOrBr", "lake": "coolwarm", "surface": "viridis",
}


def plot_spatial_map(short_name: str, var_info: dict, year: int):
    """Plot a 4-panel spatial map for one variable/year."""
    out = PLOTS_DIR / "maps" / short_name / f"{year}.png"
    if out.exists():
        return
    aggregation = var_info["aggregation"]
    category = var_info.get("category", "surface")
    stats = _get_stat_names(aggregation)
    cmap_base = CMAPS.get(category, "viridis")

    n = len(stats)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(18, 5 * rows),
                             subplot_kw={"projection": ccrs.Robinson()})
    axes = np.atleast_2d(axes)

    for idx, stat in enumerate(stats):
        ax = axes.flat[idx]
        nc = FINAL_DIR / f"{short_name}_{stat}" / f"{year}.nc"
        if not nc.exists():
            ax.set_title(f"{stat} — not available")
            continue
        ds = xr.open_dataset(nc)
        da = ds[list(ds.data_vars)[0]]
        
        # Fix meridian seam robustly
        data_cyclic, lon_cyclic, lat_values = add_cyclic_point_xr(da)
        
        cmap = "inferno" if stat == "std" else cmap_base
        
        # Robust color scaling
        vmin = float(np.nanpercentile(data_cyclic, 2))
        vmax = float(np.nanpercentile(data_cyclic, 98))
        
        im = ax.pcolormesh(
            lon_cyclic, lat_values, data_cyclic,
            transform=ccrs.PlateCarree(),
            cmap=cmap,
            vmin=vmin, vmax=vmax,
            shading='auto'
        )
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, aspect=40, shrink=0.8,
                     label=da.attrs.get("units", ""))
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.set_global()
        ax.set_title(f"{var_info['long_name']} — {stat.upper()}", fontsize=11)
        ds.close()

    # Hide unused axes
    for idx in range(n, rows * cols):
        axes.flat[idx].set_visible(False)

    plt.suptitle(f"ERA5 {short_name} — {year}", fontsize=14, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / "maps" / short_name / f"{year}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved map → %s", out)


def plot_time_series(short_name: str, var_info: dict, years: list[int],
                     all_means: dict[str, list[float]]):
    """Plot spatial-mean time series for ALL derived stats of a variable.

    Produces one figure with subplots for each stat (primary agg, std, max, min).
    """
    out = PLOTS_DIR / "timeseries" / f"{short_name}.png"
    if out.exists():
        return
    aggregation = var_info["aggregation"]
    stat_names = _get_stat_names(aggregation)
    available = [s for s in stat_names if s in all_means and all_means[s]]

    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = {"mean": "#1f77b4", "sum": "#1f77b4", "std": "#ff7f0e",
              "max": "#d62728", "min": "#2ca02c"}

    for ax, stat in zip(axes, available):
        vals = all_means[stat]
        ax.plot(years[:len(vals)], vals, marker=".", markersize=3, linewidth=1,
                color=colors.get(stat, "#333333"))
        ax.set_ylabel(f"{stat.upper()} ({var_info['units']})", fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{var_info['long_name']} — {stat.upper()}", fontsize=11)

    axes[-1].set_xlabel("Year")
    plt.suptitle(f"ERA5 {short_name} — Time Series (spatial mean)", fontsize=13, y=1.01)
    plt.tight_layout()
    out = PLOTS_DIR / "timeseries" / f"{short_name}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved time series → %s", out)


# ── Pipeline ─────────────────────────────────────────────────────────────────

@click.command()
@click.option("--variables", "-v", multiple=True, help="Short names (e.g. t2m tp).")
@click.option("--all", "run_all", is_flag=True, help="Process all variables.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--plot-every", type=int, default=10, help="Plot spatial maps every N years (default: 10).")
@click.option("--workers", "-w", type=int, default=6, help="Parallel download workers (default: 6).")
@click.option("--batch-size", type=int, default=10, help="Years to download per batch (default: 10).")
def main(variables, run_all, start_year, end_year, plot_every, workers, batch_size):
    """ERA5 full pipeline: download → process → cleanup → visualize."""
    config = load_era5_config()
    secrets = load_secrets()
    dataset = config["dataset"]
    product_type = config["product_type"]
    t_range = config["temporal_range"]
    all_vars = config["variables"]

    # Resolve variables
    if variables:
        var_list = {v: all_vars[v] for v in variables if v in all_vars}
    elif run_all:
        var_list = all_vars
    else:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    # Resolve years
    y_start, y_end = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=t_range[0],
        default_end=t_range[1],
        label="ERA5 pipeline years",
    )
    year_list = list(range(y_start, y_end + 1))

    # Pick which years get spatial map plots
    plot_years = set(range(y_start, y_end + 1, plot_every))
    plot_years.add(y_end)

    click.echo(f"Pipeline: {len(var_list)} variables × {len(year_list)} years ({y_start}–{y_end})")
    click.echo(f"Spatial maps every {plot_every} years: {sorted(plot_years)}")
    click.echo(f"Parallel workers: {workers}, batch size: {batch_size}")

    for var_idx, (short_name, info) in enumerate(var_list.items(), 1):
        click.echo(f"\n[{var_idx}/{len(var_list)}] {short_name} ({info['long_name']})")

        aggregation = info["aggregation"]
        stat_names = _get_stat_names(aggregation)

        # Time series accumulators: {stat_name: [mean_values]}
        ts_years = []
        ts_means = {s: [] for s in stat_names}

        # Process in batches of years
        for batch_start in range(0, len(year_list), batch_size):
            batch_years = year_list[batch_start : batch_start + batch_size]

            # Separate years into already-done vs needs-download
            need_download = []
            for year in batch_years:
                if outputs_exist(short_name, aggregation, year):
                    # Already processed — just collect time series means
                    means = read_means_from_outputs(short_name, aggregation, year)
                    if means:
                        ts_years.append(year)
                        for s in stat_names:
                            if s in means:
                                ts_means[s].append(means[s])
                else:
                    need_download.append(year)

            if not need_download:
                click.echo(f"  batch {batch_start // batch_size + 1}: "
                           f"years {batch_years[0]}–{batch_years[-1]} already done")
                continue

            # 1. Parallel download only for years that need processing
            downloaded = download_years_parallel(
                secrets, info["cds_name"], short_name, need_download,
                dataset, product_type, workers,
            )

            # 2. Process each year (sequential — fast, ~1s each)
            for year in sorted(need_download):
                if year not in downloaded:
                    continue

                means = process_raw_file(short_name, info, year)
                if means:
                    ts_years.append(year)
                    for s in stat_names:
                        if s in means:
                            ts_means[s].append(means[s])

                # 3. Cleanup raw
                raw_file = RAW_DIR / short_name / f"{year}.nc"
                if raw_file.exists():
                    raw_file.unlink()

                # 4. Spatial map (every N years)
                if year in plot_years:
                    try:
                        plot_spatial_map(short_name, info, year)
                    except Exception as e:
                        logger.warning("Map plot failed %s/%d: %s", short_name, year, e)

            click.echo(f"  batch {batch_start // batch_size + 1}: "
                       f"years {batch_years[0]}–{batch_years[-1]} done "
                       f"({len(downloaded)}/{len(batch_years)} downloaded)")

        # 5. Time series plot with all stats
        if ts_years:
            try:
                plot_time_series(short_name, info, ts_years, ts_means)
            except Exception as e:
                logger.warning("Time series plot failed %s: %s", short_name, e)

        # Cleanup empty raw dir
        raw_var_dir = RAW_DIR / short_name
        if raw_var_dir.exists() and not any(raw_var_dir.iterdir()):
            raw_var_dir.rmdir()

        gc.collect()
        plt.close("all")

    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
