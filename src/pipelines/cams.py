"""Full CAMS pipeline: download → process → cleanup → visualize.

Usage:
    python -m src.pipelines.cams --all
    python -m src.pipelines.cams -v no2 -v pm2p5 --start-year 2010
"""

import gc
import zipfile
import tempfile
import shutil
import time as time_mod
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import cdsapi
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from scipy.interpolate import RegularGridInterpolator

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, YEAR_START, YEAR_END
from src.utils import add_cyclic_point_xr, enforce_periodic_edge_interp, get_logger

logger = get_logger("pipeline.cams")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "cams.yml"
SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cams"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "air_quality"
PLOTS_DIR = PROJECT_ROOT / "plots" / "cams"


def load_cams_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_secrets() -> dict:
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(f"Secrets file not found: {SECRETS_PATH}")
    with open(SECRETS_PATH) as f:
        secrets = yaml.safe_load(f)
    if "ads" in secrets:
        return secrets["ads"]
    elif "cds" in secrets:
        return {"url": "https://ads.atmosphere.copernicus.eu/api", "key": secrets["cds"]["key"]}
    raise KeyError("No 'ads' or 'cds' section in secrets.yml")


def _make_client(secrets: dict) -> cdsapi.Client:
    return cdsapi.Client(url=secrets["url"], key=secrets["key"])


def download_one(secrets, ads_name, short_name, year, dataset, product_type, overwrite=False, max_retries=5):
    out_path = RAW_DIR / short_name / f"{year}.nc"
    if out_path.exists() and not overwrite:
        return year, out_path, False

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(f"{out_path.suffix}.partial")
    if tmp_path.exists():
        tmp_path.unlink()
    request = {
        "variable": [ads_name],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }
    if not short_name.startswith("tc") and short_name not in ["aod550"]:
        request["pressure_level"] = ["1000"]

    for attempt in range(1, max_retries + 1):
        try:
            client = _make_client(secrets)
            client.retrieve(dataset, request, str(tmp_path))
            tmp_path.replace(out_path)
            return year, out_path, True
        except Exception as e:
            if tmp_path.exists():
                tmp_path.unlink()
            err_str = str(e)
            if ("queued" in err_str or "limited" in err_str or "50" in err_str) and attempt < max_retries:
                wait = min(30 * attempt, 120)
                time_mod.sleep(wait)
            else:
                logger.error("Failed %s/%d: %s", short_name, year, err_str[:200])
                return year, None, False
    return year, None, False


def download_years_parallel(secrets, ads_name, short_name, years, dataset, product_type, workers, overwrite=False):
    results = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(
                download_one,
                secrets,
                ads_name,
                short_name,
                y,
                dataset,
                product_type,
                overwrite,
            ): y
            for y in years
        }
        for fut in as_completed(futures):
            year, path, _ = fut.result()
            if path: results[year] = path
    return results


def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    rename = {d: "lat" for d in ["latitude", "lat"] if d in ds.dims}
    rename.update({d: "lon" for d in ["longitude", "lon"] if d in ds.dims})
    if "valid_time" in ds.dims: rename["valid_time"] = "time"
    if rename: ds = ds.rename(rename)
    if ds.lat.values[0] > ds.lat.values[-1]: ds = ds.isel(lat=slice(None, None, -1))
    if float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon.values % 360))
        ds = ds.sortby("lon")
    return ds


def _find_data_var(ds: xr.Dataset, short_name: str) -> str:
    if short_name in ds.data_vars: return short_name
    data_vars = [v for v in ds.data_vars if v not in ("time_bnds", "expver")]
    return data_vars[0] if data_vars else short_name


def _regrid_to_master(data: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Regrid a 2-D array from its native grid to the 0.25° master grid.

    Uses bilinear interpolation for valid data and nearest-neighbour to fill
    NaN regions so that coastlines / polar edges are not lost.

    Returns (regridded_data, target_lat, target_lon).
    """
    tgt_lat = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    tgt_lon = np.linspace(0, 359.75, N_LON)

    # Skip regridding when source already matches master grid
    if data.shape == (N_LAT, N_LON):
        return data, tgt_lat, tgt_lon

    src_lon_support = np.asarray(src_lon, dtype=np.float64).copy()

    # Ensure source coordinates are strictly ascending (required by RegularGridInterpolator)
    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        data = data[::-1, :]
    if src_lon[0] > src_lon[-1]:
        order = np.argsort(src_lon)
        src_lon = src_lon[order]
        data = data[:, order]
    if src_lon_support[0] > src_lon_support[-1]:
        src_lon_support = np.sort(src_lon_support)

    # Treat longitude as periodic so the dateline is interpolated from wrapped neighbours.
    if len(src_lon) >= 4 and float(src_lon.max() - src_lon.min()) > 300.0:
        pad_cells = min(2, len(src_lon) // 2)
        src_lon = np.concatenate([src_lon[-pad_cells:] - 360.0, src_lon, src_lon[:pad_cells] + 360.0])
        data = np.concatenate([data[:, -pad_cells:], data, data[:, :pad_cells]], axis=1)

    # Build meshgrid of target points (lat varies along rows, lon along cols)
    tgt_lon_grid, tgt_lat_grid = np.meshgrid(tgt_lon, tgt_lat)
    target_points = np.column_stack([tgt_lat_grid.ravel(), tgt_lon_grid.ravel()])

    # --- bilinear pass (primary) ---
    interp_linear = RegularGridInterpolator(
        (src_lat, src_lon), data,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    regridded = interp_linear(target_points).reshape(N_LAT, N_LON)

    # --- nearest pass to fill NaN holes left by bilinear ---
    nan_mask = np.isnan(regridded)
    if nan_mask.any():
        # Use the original data with NaNs replaced by 0 for nearest lookup,
        # but only write into cells that were NaN after bilinear.
        data_filled = np.where(np.isnan(data), 0.0, data)
        interp_nearest = RegularGridInterpolator(
            (src_lat, src_lon), data_filled,
            method="nearest", bounds_error=False, fill_value=np.nan,
        )
        nearest_vals = interp_nearest(target_points).reshape(N_LAT, N_LON)
        # Only fill where the source itself had valid data (nearest picked a real value)
        src_has_data = RegularGridInterpolator(
            (src_lat, src_lon), (~np.isnan(data)).astype(np.float32),
            method="nearest", bounds_error=False, fill_value=0.0,
        )(target_points).reshape(N_LAT, N_LON) > 0.5
        regridded = np.where(nan_mask & src_has_data, nearest_vals, regridded)

    regridded = enforce_periodic_edge_interp(regridded, tgt_lon, src_lon_support)

    return regridded, tgt_lat, tgt_lon


def _reduce_monthly_stack(stack: np.ndarray, stat: str) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        if stat == "mean":
            return np.nanmean(stack, axis=0)
        if stat == "sum":
            return np.nansum(stack, axis=0)
        if stat == "max":
            return np.nanmax(stack, axis=0)
        if stat == "min":
            return np.nanmin(stack, axis=0)
        if stat == "std":
            return np.nanstd(stack, axis=0)
    raise ValueError(f"Unknown stat: {stat}")


def _regrid_monthly_stack_to_master(
    monthly: np.ndarray,
    src_lat: np.ndarray,
    src_lon: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    monthly = np.asarray(monthly)
    if monthly.ndim == 2:
        monthly = monthly[np.newaxis, ...]

    regridded = []
    tgt_lat = tgt_lon = None
    for idx in range(monthly.shape[0]):
        arr, tgt_lat, tgt_lon = _regrid_to_master(monthly[idx], src_lat, src_lon)
        regridded.append(arr.astype(np.float32, copy=False))
    return np.stack(regridded, axis=0), tgt_lat, tgt_lon


def _save_derived(data, lat, lon, var_name, stat_name, long_name, units, year, overwrite=False):
    folder = f"{var_name}_{stat_name}"
    out = FINAL_DIR / folder / f"{year}.nc"
    if out.exists() and not overwrite:
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    if data.ndim > 2: data = np.squeeze(data)

    # Regrid from native resolution to 0.25° master grid
    data, lat, lon = _regrid_to_master(data, lat, lon)

    time_coords = [np.datetime64(f"{year}-01-01")]
    ds = xr.Dataset(
        {folder: (["lat", "lon"], data.astype(np.float32), {"units": units, "long_name": f"{long_name} ({stat_name})"})},
        coords={"lat": ("lat", lat, {"units": "degrees_north"}), "lon": ("lon", lon, {"units": "degrees_east"})},
        attrs={"title": f"WorldTensor CAMS {folder}", "source": "CAMS EAC4", "year": year, "Conventions": "CF-1.8"}
    )
    ds = ds.expand_dims(time=time_coords)
    ds.to_netcdf(out, encoding={folder: {"zlib": True, "complevel": 4, "dtype": "float32"}})
    return out


def _get_stat_names(aggregation: str) -> list[str]:
    stats = [aggregation, "std"]
    if aggregation != "max": stats.append("max")
    if aggregation != "min": stats.append("min")
    return stats


def outputs_exist(short_name, aggregation, year):
    return all((FINAL_DIR / f"{short_name}_{s}" / f"{year}.nc").exists() for s in _get_stat_names(aggregation))


def read_means_from_outputs(short_name, aggregation, year):
    stats = _get_stat_names(aggregation)
    means = {}
    for s in stats:
        p = FINAL_DIR / f"{short_name}_{s}" / f"{year}.nc"
        if not p.exists(): return None
        ds = xr.open_dataset(p)
        means[s] = float(ds[list(ds.data_vars)[0]].mean(skipna=True))
        ds.close()
    return means


def process_raw_file(short_name, var_info, year, overwrite=False):
    raw_path = RAW_DIR / short_name / f"{year}.nc"
    if not raw_path.exists(): return None
    agg = var_info["aggregation"]
    tmp_dir = None
    try:
        actual_path = raw_path
        if zipfile.is_zipfile(raw_path):
            tmp_dir = Path(tempfile.mkdtemp(prefix="cams_pipe_"))
            with zipfile.ZipFile(raw_path, 'r') as z: z.extractall(tmp_dir)
            nc_files = list(tmp_dir.glob("*.nc"))
            if not nc_files: raise ValueError("No .nc in zip")
            actual_path = nc_files[0]

        ds = xr.open_dataset(actual_path)
        ds = _normalize_coords(ds)
        da = ds[_find_data_var(ds, short_name)].squeeze(drop=True)
        non_spatial_dims = [dim for dim in da.dims if dim not in ("lat", "lon")]
        if not non_spatial_dims:
            raise ValueError(f"No time-like dimension found for {short_name}/{year}")
        time_dim = "time" if "time" in non_spatial_dims else non_spatial_dims[0]
        src_lat, src_lon = ds.lat.values, ds.lon.values
        monthly = da.transpose(time_dim, "lat", "lon").values
        ds.close()

        monthly_regridded, lat, lon = _regrid_monthly_stack_to_master(monthly, src_lat, src_lon)
        stats = {
            agg: _reduce_monthly_stack(monthly_regridded, agg),
            "std": _reduce_monthly_stack(monthly_regridded, "std"),
            "max": _reduce_monthly_stack(monthly_regridded, "max"),
            "min": _reduce_monthly_stack(monthly_regridded, "min"),
        }
        means = {}
        for s, d in stats.items():
            _save_derived(
                d,
                lat,
                lon,
                short_name,
                s,
                var_info["long_name"],
                var_info["units"],
                year,
                overwrite=overwrite,
            )
            means[s] = float(np.nanmean(d))
        return means
    except Exception as e:
        logger.error("Error processing %s/%d: %s", short_name, year, e)
        return None
    finally:
        if tmp_dir and tmp_dir.exists(): shutil.rmtree(tmp_dir)


def _plot_limits(data: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(data)
    if not finite.any():
        return 0.0, 1.0

    vals = data[finite]
    vmin = float(np.nanpercentile(vals, 2))
    vmax = float(np.nanpercentile(vals, 98))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmin = float(np.nanmin(vals))
        vmax = float(np.nanmax(vals))
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin == vmax:
        vmax = vmin + 1.0
    return vmin, vmax


def plot_spatial_map(short_name, var_info, year, overwrite=False):
    out = PLOTS_DIR / "maps" / short_name / f"{year}.png"
    if out.exists() and not overwrite:
        return

    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    stats = _get_stat_names(var_info["aggregation"])
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(18, 10),
        subplot_kw={"projection": ccrs.Robinson()},
        constrained_layout=True,
    )
    flat_axes = list(axes.flat)
    for idx, stat in enumerate(stats):
        ax = flat_axes[idx]
        nc = FINAL_DIR / f"{short_name}_{stat}" / f"{year}.nc"
        if not nc.exists():
            ax.axis("off")
            continue
        ds = xr.open_dataset(nc)
        da = ds[list(ds.data_vars)[0]].squeeze(drop=True)

        if da.sizes.get("lat", 0) > 240 and da.sizes.get("lon", 0) > 480:
            da = da.isel(lat=slice(None, None, 4), lon=slice(None, None, 4))

        data_cyclic, lon_cyclic, lat_values = add_cyclic_point_xr(da)
        data_cyclic = np.asarray(data_cyclic, dtype=np.float32)
        vmin, vmax = _plot_limits(data_cyclic)

        im = ax.pcolormesh(
            lon_cyclic,
            lat_values,
            data_cyclic,
            transform=ccrs.PlateCarree(),
            cmap="plasma" if stat == "std" else "viridis",
            vmin=vmin,
            vmax=vmax,
            shading="auto",
        )
        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.03, shrink=0.85, label=da.attrs.get("units", ""))
        ax.coastlines(linewidth=0.4, color="gray")
        ax.add_feature(cfeature.BORDERS, linewidth=0.25, linestyle=":")
        ax.set_global()
        ax.set_title(f"{stat.upper()}", fontsize=11)
        ds.close()

    for idx in range(len(stats), len(flat_axes)):
        flat_axes[idx].axis("off")

    plt.suptitle(f"CAMS {short_name} — {year}", fontsize=14)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


def plot_time_series(short_name, var_info, years, all_means, overwrite=False):
    out = PLOTS_DIR / "timeseries" / f"{short_name}.png"
    if out.exists() and not overwrite:
        return
    stats = _get_stat_names(var_info["aggregation"])
    fig, axes = plt.subplots(len(stats), 1, figsize=(14, 4 * len(stats)), sharex=True)
    colors = {"mean": "#1f77b4", "sum": "#1f77b4", "std": "#ff7f0e", "max": "#d62728", "min": "#2ca02c"}
    for ax, stat in zip(axes, stats):
        if stat in all_means and all_means[stat]:
            ax.plot(years, all_means[stat], marker=".", markersize=3, color=colors.get(stat, "#333333"))
            ax.set_ylabel(f"{stat.upper()} ({var_info['units']})")
            ax.set_title(f"{var_info['long_name']} — {stat.upper()}")
            ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Year")
    plt.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


@click.command()
@click.option("--variables", "-v", multiple=True)
@click.option("--all", "run_all", is_flag=True)
@click.option("--start-year", type=int)
@click.option("--end-year", type=int)
@click.option("--workers", "-w", type=int, default=4)
@click.option("--map-every", type=int, default=10, show_default=True, help="Generate map plots every N years.")
@click.option("--plot-only", is_flag=True, help="Generate plots from existing CAMS outputs only.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing raw downloads, outputs, and plots.")
def main(variables, run_all, start_year, end_year, workers, map_every, plot_only, overwrite):
    config = load_cams_config()
    all_vars = config["variables"]
    var_list = all_vars if run_all else {v: all_vars[v] for v in variables if v in all_vars}
    y0 = max(start_year or config["temporal_range"][0], YEAR_START)
    y1 = min(end_year or config["temporal_range"][1], YEAR_END)
    years = list(range(y0, y1 + 1))
    secrets = load_secrets() if not plot_only else None

    for short_name, info in var_list.items():
        click.echo(f"\nProcessing {short_name}")
        ts_years, ts_means = [], {s: [] for s in _get_stat_names(info["aggregation"])}

        if plot_only:
            existing_years = []
            primary_dir = FINAL_DIR / f"{short_name}_{info['aggregation']}"
            if primary_dir.exists():
                existing_years = sorted(
                    year for year in (int(p.stem) for p in primary_dir.glob("*.nc"))
                    if y0 <= year <= y1
                )
            for year in existing_years:
                means = read_means_from_outputs(short_name, info["aggregation"], year)
                if means:
                    ts_years.append(year)
                    for s, v in means.items():
                        ts_means[s].append(v)
            if existing_years:
                for year in existing_years:
                    if year % map_every == 0 or year == existing_years[-1]:
                        plot_spatial_map(short_name, info, year, overwrite=overwrite)
        else:
            for year in years:
                means = None if overwrite else read_means_from_outputs(short_name, info["aggregation"], year)
                if not means:
                    downloaded = download_years_parallel(
                        secrets,
                        info["ads_name"],
                        short_name,
                        [year],
                        config["dataset"],
                        config["product_type"],
                        workers,
                        overwrite=overwrite,
                    )
                    raw_file = RAW_DIR / short_name / f"{year}.nc"
                    if year in downloaded or raw_file.exists():
                        means = process_raw_file(short_name, info, year, overwrite=overwrite)
                        raw_file = RAW_DIR / short_name / f"{year}.nc"
                        if raw_file.exists():
                            raw_file.unlink()
                if means:
                    ts_years.append(year)
                    for s, v in means.items():
                        ts_means[s].append(v)
                if year % map_every == 0 or year == years[-1]:
                    plot_spatial_map(short_name, info, year, overwrite=overwrite)
        if ts_years:
            plot_time_series(short_name, info, ts_years, ts_means, overwrite=overwrite)

    click.echo("\nPipeline complete.")


if __name__ == "__main__":
    main()
