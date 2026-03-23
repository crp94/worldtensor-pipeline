"""Full climate extremes pipeline: download -> process -> cleanup -> visualize.

Uses CDS dataset `derived-drought-historical-monthly` and produces yearly
mean maps for each variable + accumulation period combination.

Usage:
    python -m src.pipelines.climate_extremes --all
    python -m src.pipelines.climate_extremes --variables spi --accumulation-periods 12 --start-year 2000 --end-year 2020
"""

from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor, as_completed
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

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.download.climate_extremes import (
    clamp_years_to_available,
    download_one,
    fetch_available_years,
    load_secrets,
    raw_path_for,
    build_combo_name,
    make_cds_name,
)
from src.utils import get_logger
from src.year_policy import resolve_year_bounds

logger = get_logger("pipeline.climate_extremes")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "climate_extremes.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "climate_extremes"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "extremes"
PLOTS_DIR = PROJECT_ROOT / "plots" / "climate_extremes"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)
STATS = ("mean",)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


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

    if "lat" in ds.coords and ds.lat.values[0] > ds.lat.values[-1]:
        ds = ds.isel(lat=slice(None, None, -1))

    if "lon" in ds.coords and float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon.values % 360.0))
        ds = ds.sortby("lon")

    return ds


def _find_data_var(ds: xr.Dataset, short_name: str) -> str:
    if short_name in ds.data_vars:
        return short_name
    data_vars = [v for v in ds.data_vars if v not in ("time_bnds", "expver")]
    if data_vars:
        return data_vars[0]
    raise ValueError(f"No data variables found for {short_name}")


def _to_target_grid(da: xr.DataArray) -> xr.DataArray:
    same_shape = da.sizes.get("lat") == N_LAT and da.sizes.get("lon") == N_LON
    if same_shape:
        lat_ok = np.isclose(float(da.lat.min()), LAT_MIN) and np.isclose(float(da.lat.max()), LAT_MAX)
        lon_ok = np.isclose(float(da.lon.min()), LON_MIN) and np.isclose(float(da.lon.max()), LON_MAX)
        if lat_ok and lon_ok:
            return da

    return da.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )


def _save_derived(data, combo: str, stat_name: str, long_name: str, units: str, year: int) -> Path:
    folder = f"{combo}_{stat_name}"
    out_path = FINAL_DIR / folder / f"{year}.nc"
    if out_path.exists():
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            folder: (
                ["lat", "lon"],
                data.astype(np.float32),
                {"units": units, "long_name": f"{long_name} ({stat_name})"},
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor Climate Extremes {folder}",
            "source": "CDS derived-drought-historical-monthly",
            "aggregation_method": stat_name,
            "year": year,
        },
    )
    ds.to_netcdf(out_path, encoding={folder: {"zlib": True, "complevel": 4, "dtype": "float32"}})
    return out_path


def outputs_exist(combo: str, year: int) -> bool:
    return all((FINAL_DIR / f"{combo}_{s}" / f"{year}.nc").exists() for s in STATS)


def read_means_from_outputs(combo: str, year: int) -> dict | None:
    means = {}
    for stat in STATS:
        p = FINAL_DIR / f"{combo}_{stat}" / f"{year}.nc"
        if not p.exists():
            return None
        ds = xr.open_dataset(p)
        means[stat] = float(ds[list(ds.data_vars)[0]].mean(skipna=True).values)
        ds.close()
    return means


def process_raw_file(short_name: str, accumulation: int, var_info: dict, year: int) -> dict | None:
    combo = build_combo_name(short_name, accumulation)
    raw_path = raw_path_for(RAW_DIR, short_name, accumulation, year)
    if not raw_path.exists():
        return None

    if outputs_exist(combo, year):
        return read_means_from_outputs(combo, year)

    ds = xr.open_dataset(raw_path)
    ds = _normalize_coords(ds)
    data_var = _find_data_var(ds, short_name)
    da = ds[data_var]

    time_dim = "time" if "time" in da.dims else da.dims[0]
    da = _to_target_grid(da)

    computed = {"mean": da.mean(dim=time_dim).values}

    ds.close()

    long_name = f"{var_info['long_name']} ({accumulation}-month accumulation)"
    units = var_info["units"]

    means = {}
    for stat_name, data in computed.items():
        _save_derived(data, combo, stat_name, long_name, units, year)
        means[stat_name] = float(np.nanmean(data))

    return means


def download_years_parallel(
    secrets: dict,
    config: dict,
    short_name: str,
    cds_name: str,
    accumulation: int,
    years: list[int],
    workers: int,
    overwrite: bool,
) -> dict[int, Path]:
    results: dict[int, Path] = {}
    futures = {}

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for year in years:
            fut = pool.submit(
                download_one,
                secrets,
                config,
                cds_name,
                short_name,
                accumulation,
                year,
                RAW_DIR,
                overwrite,
                4,
            )
            futures[fut] = year

        for fut in as_completed(futures):
            year, path, _was_new = fut.result()
            if path:
                results[year] = path

    return results


def plot_spatial_map(combo: str, long_name: str, units: str, year: int):
    out = PLOTS_DIR / "maps" / combo / f"{year}.png"
    if out.exists():
        return

    fig, axes = plt.subplots(1, len(STATS), figsize=(6 * len(STATS), 5), subplot_kw={"projection": ccrs.Robinson()})
    if len(STATS) == 1:
        axes = [axes]

    for ax, stat in zip(axes, STATS):
        nc = FINAL_DIR / f"{combo}_{stat}" / f"{year}.nc"
        if not nc.exists():
            ax.set_title(f"{stat.upper()} - not available")
            ax.set_global()
            continue

        ds = xr.open_dataset(nc)
        da = ds[list(ds.data_vars)[0]]

        cmap = "YlOrRd" if stat in ("mean", "max") else "viridis"
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
            label=units,
        )
        ax.coastlines(linewidth=0.5)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
        ax.set_global()
        ax.set_title(f"{long_name} - {stat.upper()}", fontsize=11)
        ds.close()

    plt.suptitle(f"Climate Extremes {combo.upper()} - {year}", fontsize=14, y=1.01)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved map -> %s", out)


def plot_time_series(combo: str, long_name: str, units: str, years: list[int], means_by_stat: dict[str, list[float]]):
    out = PLOTS_DIR / "timeseries" / f"{combo}.png"
    if out.exists():
        return

    available = [s for s in STATS if means_by_stat.get(s)]
    if not available:
        return

    n = len(available)
    fig, axes = plt.subplots(n, 1, figsize=(14, 4 * n), sharex=True)
    if n == 1:
        axes = [axes]

    colors = {"mean": "#1f77b4", "max": "#d62728", "min": "#2ca02c"}

    for ax, stat in zip(axes, available):
        vals = means_by_stat[stat]
        x = years[:len(vals)]
        ax.plot(x, vals, marker=".", markersize=3, linewidth=1.2, color=colors[stat])
        ax.set_ylabel(f"{stat.upper()} ({units})")
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{long_name} - {stat.upper()}", fontsize=11)

    axes[-1].set_xlabel("Year")
    plt.suptitle(f"Climate Extremes {combo.upper()} - Time Series", fontsize=13, y=1.01)
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved time series -> %s", out)


@click.command()
@click.option("--variables", "variables", "-v", multiple=True, help="Variables (spi, spei).")
@click.option(
    "--accumulation-periods",
    "accumulation_periods",
    "-a",
    multiple=True,
    type=int,
    help="Accumulation period(s) in months.",
)
@click.option("--all", "run_all", is_flag=True, help="Process all configured variables.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--plot-every", type=int, default=10, show_default=True, help="Plot maps every N years.")
@click.option("--workers", "workers", "-w", type=int, default=6, show_default=True, help="Parallel download workers.")
@click.option("--batch-size", type=int, default=10, show_default=True, help="Years per batch.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--cleanup-raw/--keep-raw", default=True, show_default=True, help="Delete raw files after processing.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
def main(
    variables,
    accumulation_periods,
    run_all,
    start_year,
    end_year,
    plot_every,
    workers,
    batch_size,
    skip_download,
    cleanup_raw,
    overwrite,
):
    """Climate extremes full pipeline: download -> process -> cleanup -> visualize."""
    config = load_config()
    all_vars = config["variables"]

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

    allowed_acc = {int(v) for v in config["accumulation_periods"]}
    if accumulation_periods:
        acc_list = sorted(set(int(v) for v in accumulation_periods if int(v) in allowed_acc))
        missing_acc = sorted(set(int(v) for v in accumulation_periods) - allowed_acc)
        if missing_acc:
            logger.warning("Unsupported accumulation periods skipped: %s", missing_acc)
    else:
        acc_list = sorted(allowed_acc)
    if not acc_list:
        click.echo("No valid accumulation periods selected.")
        return

    requested_start, requested_end = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=config["temporal_range"][0],
        default_end=config["temporal_range"][1],
        label="climate extremes pipeline years",
    )
    year_list = list(range(requested_start, requested_end + 1))

    secrets = None
    if not skip_download:
        secrets = load_secrets()
        available_years = fetch_available_years(secrets, config["dataset"])
        year_list = clamp_years_to_available(year_list, available_years)
        if not year_list:
            raise click.ClickException("No requested years are available in CDS for this dataset.")

    y_start = year_list[0]
    y_end = year_list[-1]

    plot_years = set(range(y_start, y_end + 1, plot_every))
    plot_years.add(y_end)

    click.echo(
        f"Pipeline: vars={var_list}, accumulations={acc_list}, years={y_start}-{y_end}, "
        f"plot_every={plot_every}, workers={workers}, batch_size={batch_size}"
    )

    for short_name in var_list:
        info = all_vars[short_name]

        for accumulation in acc_list:
            cds_name = make_cds_name(info, accumulation)
            combo = build_combo_name(short_name, accumulation)
            long_name = f"{info['long_name']} ({accumulation}-month accumulation)"
            units = info["units"]

            click.echo(f"\n[{combo}] {long_name}")

            ts_years = []
            ts_means = {s: [] for s in STATS}

            for batch_start in range(0, len(year_list), batch_size):
                batch_years = year_list[batch_start : batch_start + batch_size]

                need_download = []
                for year in batch_years:
                    if outputs_exist(combo, year) and not overwrite:
                        means = read_means_from_outputs(combo, year)
                        if means:
                            ts_years.append(year)
                            for s in STATS:
                                if s in means:
                                    ts_means[s].append(means[s])
                    else:
                        need_download.append(year)

                if not need_download:
                    click.echo(
                        f"  batch {batch_start // batch_size + 1}: years {batch_years[0]}-{batch_years[-1]} already done"
                    )
                    continue

                downloaded: dict[int, Path] = {}
                if not skip_download:
                    downloaded = download_years_parallel(
                        secrets=secrets,
                        config=config,
                        short_name=short_name,
                        cds_name=cds_name,
                        accumulation=accumulation,
                        years=need_download,
                        workers=workers,
                        overwrite=overwrite,
                    )

                for year in sorted(need_download):
                    if not skip_download and year not in downloaded:
                        continue

                    means = process_raw_file(short_name, accumulation, info, year)
                    if means:
                        ts_years.append(year)
                        for s in STATS:
                            if s in means:
                                ts_means[s].append(means[s])

                    raw_file = raw_path_for(RAW_DIR, short_name, accumulation, year)
                    if cleanup_raw and raw_file.exists():
                        raw_file.unlink(missing_ok=True)

                    if year in plot_years:
                        try:
                            plot_spatial_map(combo, long_name, units, year)
                        except Exception as e:
                            logger.warning("Map plot failed %s/%d: %s", combo, year, e)

                click.echo(
                    f"  batch {batch_start // batch_size + 1}: years {batch_years[0]}-{batch_years[-1]} done "
                    f"({len(downloaded)}/{len(batch_years)} downloaded)"
                )

            if ts_years:
                try:
                    plot_time_series(combo, long_name, units, ts_years, ts_means)
                except Exception as e:
                    logger.warning("Time series plot failed %s: %s", combo, e)

            gc.collect()
            plt.close("all")

    if cleanup_raw and RAW_DIR.exists():
        for d in sorted(RAW_DIR.rglob("*"), reverse=True):
            if d.is_dir():
                try:
                    d.rmdir()
                except OSError:
                    pass

    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
