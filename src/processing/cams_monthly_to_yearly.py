"""Aggregate CAMS EAC4 monthly data to yearly, producing derived maps.

For each variable and year, produces 4 output files under data/final/air_quality/:
    {var}_{mean|sum}/{year}.nc  — primary aggregate
    {var}_std/{year}.nc         — standard deviation across months
    {var}_max/{year}.nc         — maximum monthly value
    {var}_min/{year}.nc         — minimum monthly value
"""

import gc
import zipfile
import tempfile
import shutil
from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from src.utils import enforce_periodic_edge_interp, get_logger
from src.grid import LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, N_LAT, N_LON
from src.year_policy import resolve_year_list

logger = get_logger("processing.cams")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "cams.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cams"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "air_quality"


def load_cams_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _normalize_coords(ds: xr.Dataset) -> xr.Dataset:
    """Ensure lat is ascending (-90→90) and lon is 0→360."""
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


def _regrid_to_master(data: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target_lat = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    target_lon = np.linspace(LON_MIN, LON_MAX, N_LON)
    if data.shape == (N_LAT, N_LON):
        return data, target_lat, target_lon

    src_lon_support = np.asarray(src_lon, dtype=np.float64).copy()

    if src_lat[0] > src_lat[-1]:
        src_lat = src_lat[::-1]
        data = data[::-1, :]
    if src_lon[0] > src_lon[-1]:
        order = np.argsort(src_lon)
        src_lon = src_lon[order]
        data = data[:, order]
    if src_lon_support[0] > src_lon_support[-1]:
        src_lon_support = np.sort(src_lon_support)
    if len(src_lon) >= 4 and float(src_lon.max() - src_lon.min()) > 300.0:
        pad_cells = min(2, len(src_lon) // 2)
        src_lon = np.concatenate([src_lon[-pad_cells:] - 360.0, src_lon, src_lon[:pad_cells] + 360.0])
        data = np.concatenate([data[:, -pad_cells:], data, data[:, :pad_cells]], axis=1)

    target_lon_grid, target_lat_grid = np.meshgrid(target_lon, target_lat)
    target_points = np.column_stack([target_lat_grid.ravel(), target_lon_grid.ravel()])

    from scipy.interpolate import RegularGridInterpolator

    interp_linear = RegularGridInterpolator(
        (src_lat, src_lon), data,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    regridded = interp_linear(target_points).reshape(N_LAT, N_LON)

    nan_mask = np.isnan(regridded)
    if nan_mask.any():
        data_filled = np.where(np.isnan(data), 0.0, data)
        interp_nearest = RegularGridInterpolator(
            (src_lat, src_lon), data_filled,
            method="nearest", bounds_error=False, fill_value=np.nan,
        )
        nearest_vals = interp_nearest(target_points).reshape(N_LAT, N_LON)
        src_has_data = RegularGridInterpolator(
            (src_lat, src_lon), (~np.isnan(data)).astype(np.float32),
            method="nearest", bounds_error=False, fill_value=0.0,
        )(target_points).reshape(N_LAT, N_LON) > 0.5
        regridded = np.where(nan_mask & src_has_data, nearest_vals, regridded)

    regridded = enforce_periodic_edge_interp(regridded, target_lon, src_lon_support)

    return regridded, target_lat, target_lon


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
    target_lat = target_lon = None
    for idx in range(monthly.shape[0]):
        arr, target_lat, target_lon = _regrid_to_master(monthly[idx], src_lat, src_lon)
        regridded.append(arr.astype(np.float32, copy=False))
    return np.stack(regridded, axis=0), target_lat, target_lon


def _save_derived(
    data: np.ndarray,
    lat: np.ndarray,
    lon: np.ndarray,
    var_name: str,
    stat_name: str,
    long_name: str,
    units: str,
    year: int,
    output_dir: Path,
    overwrite: bool = False,
) -> Path | None:
    folder_name = f"{var_name}_{stat_name}"
    out_path = output_dir / folder_name / f"{year}.nc"

    if out_path.exists() and not overwrite:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure data is 2D (lat, lon)
    if data.ndim > 2:
        data = np.squeeze(data)

    data, lat, lon = _regrid_to_master(data, lat, lon)

    time_coords = [np.datetime64(f"{year}-01-01")]
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
            "title": f"WorldTensor CAMS EAC4 {folder_name}",
            "source": "CAMS global reanalysis EAC4 (ECMWF)",
            "aggregation_method": stat_name,
            "year": year,
        },
    )
    ds = ds.expand_dims(time=time_coords)

    ds.to_netcdf(out_path, encoding={folder_name: {"zlib": True, "complevel": 4, "dtype": "float32"}})
    return out_path


def process_variable_year(short_name, var_info, year, raw_dir, output_dir, overwrite=False):
    raw_path = raw_dir / short_name / f"{year}.nc"
    if not raw_path.exists():
        return 0

    agg = var_info["aggregation"]
    stats = [agg, "std", "max", "min"]
    if not overwrite and all((output_dir/f"{short_name}_{s}"/f"{year}.nc").exists() for s in stats):
        return 0

    tmp_dir = None
    try:
        # Handle ZIP if necessary
        actual_path = raw_path
        if zipfile.is_zipfile(raw_path):
            tmp_dir = Path(tempfile.mkdtemp(prefix="cams_process_"))
            with zipfile.ZipFile(raw_path, 'r') as zip_ref:
                zip_ref.extractall(tmp_dir)
            
            # Find the .nc file inside the zip
            nc_files = list(tmp_dir.glob("*.nc"))
            if not nc_files:
                raise ValueError(f"No .nc files found in ZIP: {raw_path}")
            actual_path = nc_files[0]

        ds = xr.open_dataset(actual_path)
        ds = _normalize_coords(ds)
        data_var = _find_data_var(ds, short_name)
        da = ds[data_var].squeeze(drop=True)
        non_spatial_dims = [dim for dim in da.dims if dim not in ("lat", "lon")]
        if not non_spatial_dims:
            raise ValueError(f"No time-like dimension found for {short_name}/{year}")
        time_dim = "time" if "time" in non_spatial_dims else non_spatial_dims[0]
        src_lat, src_lon = ds.lat.values, ds.lon.values
        monthly = da.transpose(time_dim, "lat", "lon").values
        ds.close()

        monthly_regridded, lat, lon = _regrid_monthly_stack_to_master(monthly, src_lat, src_lon)
        derived = {
            agg: _reduce_monthly_stack(monthly_regridded, agg),
            "std": _reduce_monthly_stack(monthly_regridded, "std"),
            "max": _reduce_monthly_stack(monthly_regridded, "max"),
            "min": _reduce_monthly_stack(monthly_regridded, "min"),
        }

        written = 0
        for s, d in derived.items():
            if _save_derived(d, lat, lon, short_name, s, var_info["long_name"], var_info["units"], year, output_dir, overwrite):
                written += 1
        return written
    except Exception as e:
        logger.error("Failed %s/%d: %s", short_name, year, e)
        return 0
    finally:
        if tmp_dir and tmp_dir.exists():
            shutil.rmtree(tmp_dir)


@click.command()
@click.option("--variables", "-v", multiple=True)
@click.option("--all", "process_all", is_flag=True)
@click.option("--years", "-y", multiple=True, type=int)
@click.option("--start-year", type=int)
@click.option("--end-year", type=int)
@click.option("--overwrite", is_flag=True)
def main(variables, process_all, years, start_year, end_year, overwrite):
    config = load_cams_config()
    all_vars = config["variables"]
    t_range = config["temporal_range"]

    if variables:
        var_list = {v: all_vars[v] for v in variables if v in all_vars}
    elif process_all:
        var_list = all_vars
    else:
        return

    year_list = resolve_year_list(
        years,
        start_year=start_year,
        end_year=end_year,
        default_start=t_range[0],
        default_end=t_range[1],
        label="CAMS processing years",
    )
    tasks = [(s, i, y) for s, i in var_list.items() for y in year_list if (RAW_DIR/s/f"{y}.nc").exists()]

    with tqdm(total=len(tasks), desc="Processing CAMS") as pbar:
        for s, i, y in tasks:
            written = process_variable_year(s, i, y, RAW_DIR, FINAL_DIR, overwrite)
            pbar.update(1)
            pbar.set_postfix(var=s, written=written)


if __name__ == "__main__":
    main()
