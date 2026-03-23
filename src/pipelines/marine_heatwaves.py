"""Full marine heatwave pipeline: download -> process -> plot.

Uses NOAA PSL monthly OISST anomaly + monthly 90th percentile threshold files.
Annual outputs are derived as:
  - heatwave_marine_months: number of months where anomaly > threshold
  - heatwave_marine_mean_excess: mean positive anomaly excess over threshold
"""

from __future__ import annotations

import gc
from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml

from src.download.marine_heatwaves import DEFAULT_RAW_DIR, download_marine_heatwaves
from src.grid import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, N_LAT, N_LON
from src.utils import get_logger, plot_global_map, plot_time_series

logger = get_logger("pipeline.marine_heatwaves")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "marine_heatwaves.yml"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
PLOTS_DIR = PROJECT_ROOT / "plots" / "marine_heatwaves"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _normalize_coords(da: xr.DataArray) -> xr.DataArray:
    if "latitude" in da.dims:
        da = da.rename({"latitude": "lat"})
    if "longitude" in da.dims:
        da = da.rename({"longitude": "lon"})
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.isel(lat=slice(None, None, -1))
    if float(da.lon.min()) < 0:
        da = da.assign_coords(lon=(da.lon % 360.0)).sortby("lon")
    return da


def _pick_var(ds: xr.Dataset, preferred: str | None = None) -> str:
    if preferred and preferred in ds.data_vars:
        return preferred
    data_vars = [v for v in ds.data_vars if ds[v].ndim >= 2]
    if data_vars:
        return data_vars[0]
    raise ValueError(f"No data variables in dataset: {list(ds.data_vars)}")


def _pad_periodic_longitude(da: xr.DataArray, pad_cells: int = 2) -> xr.DataArray:
    da = da.sortby("lon")
    lon_vals = da.lon.values.astype(float)
    _, unique_idx = np.unique(lon_vals, return_index=True)
    da = da.isel(lon=np.sort(unique_idx))
    left = da.isel(lon=slice(-pad_cells, None)).assign_coords(lon=da.isel(lon=slice(-pad_cells, None)).lon - 360.0)
    right = da.isel(lon=slice(0, pad_cells)).assign_coords(lon=da.isel(lon=slice(0, pad_cells)).lon + 360.0)
    return xr.concat([left, da, right], dim="lon")


def _to_target_grid(da: xr.DataArray) -> xr.DataArray:
    same_shape = da.sizes.get("lat") == N_LAT and da.sizes.get("lon") == N_LON
    if same_shape:
        lat_ok = np.isclose(float(da.lat.min()), LAT_MIN) and np.isclose(float(da.lat.max()), LAT_MAX)
        lon_ok = np.isclose(float(da.lon.min()), LON_MIN) and np.isclose(float(da.lon.max()), LON_MAX)
        if lat_ok and lon_ok:
            return da
    return _pad_periodic_longitude(da).interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear")


def _save_year(var_key: str, long_name: str, units: str, year: int, arr2d: np.ndarray, domain: str, source_file: str, overwrite: bool):
    out_path = FINAL_DIR / domain / var_key / f"{year}.nc"
    if out_path.exists() and not overwrite:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out = xr.Dataset(
        {
            var_key: (
                ["lat", "lon"],
                arr2d.astype(np.float32),
                {"long_name": long_name, "units": units},
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor {long_name}",
            "source": "NOAA PSL marinehw",
            "source_file": source_file,
            "year": year,
        },
    )
    ds_out.to_netcdf(out_path, encoding={var_key: {"zlib": True, "complevel": 4, "dtype": "float32"}})
    ds_out.close()
    return out_path


def _plot_map(var_key: str, year: int, long_name: str, cmap: str, domain: str):
    nc = FINAL_DIR / domain / var_key / f"{year}.nc"
    if not nc.exists():
        return
    ds = xr.open_dataset(nc, decode_timedelta=False)
    out = PLOTS_DIR / "maps" / var_key / f"{year}.png"
    try:
        plot_global_map(ds[var_key], title=f"{long_name} ({year})", out_path=out, cmap=cmap)
    finally:
        ds.close()


def _plot_series(var_key: str, long_name: str, units: str, years: list[int], vals: list[float]):
    if not years:
        return
    out = PLOTS_DIR / "timeseries" / f"{var_key}.png"
    plot_time_series(
        years=years,
        values=vals,
        title=f"{long_name} (global mean)",
        ylabel=units,
        out_path=out,
        color="#d62728",
    )


@click.command()
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--plot-every", type=int, default=5, show_default=True, help="Map every N years.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
def main(
    start_year: int | None,
    end_year: int | None,
    plot_every: int,
    skip_download: bool,
    overwrite: bool,
):
    config = load_config()
    domain = config.get("domain", "extremes")
    variables = config["variables"]
    require_full_year = bool(config.get("require_full_year", True))

    if skip_download:
        anom_path = DEFAULT_RAW_DIR / config["files"]["anomaly"]
        p90_path = DEFAULT_RAW_DIR / config["files"]["threshold_p90"]
    else:
        downloaded = download_marine_heatwaves(raw_dir=DEFAULT_RAW_DIR, overwrite=overwrite)
        anom_path = downloaded.get("anomaly", DEFAULT_RAW_DIR / config["files"]["anomaly"])
        p90_path = downloaded.get("threshold_p90", DEFAULT_RAW_DIR / config["files"]["threshold_p90"])

    if not anom_path.exists() or not p90_path.exists():
        raise click.ClickException("Marine source files are missing. Run without --skip-download.")

    ds_anom = xr.open_dataset(anom_path)
    ds_p90 = xr.open_dataset(p90_path)
    anom_var = _pick_var(ds_anom, preferred="sst_anom")
    p90_var = _pick_var(ds_p90, preferred="sst_anom")

    da_anom = _normalize_coords(ds_anom[anom_var])
    da_p90 = _normalize_coords(ds_p90[p90_var])

    y0_cfg, y1_cfg = config["temporal_range"]
    y0 = int(start_year) if start_year is not None else int(y0_cfg)
    y1 = int(end_year) if end_year is not None else int(y1_cfg)
    if y1 < y0:
        ds_anom.close()
        ds_p90.close()
        raise click.ClickException("end-year must be >= start-year")

    all_years = sorted(set(int(y) for y in da_anom.time.dt.year.values if y0 <= int(y) <= y1))
    years: list[int] = []
    for y in all_years:
        n_month = int(da_anom.sel(time=str(y)).sizes.get("time", 0))
        if require_full_year and n_month < 12:
            continue
        years.append(y)
    if not years:
        ds_anom.close()
        ds_p90.close()
        raise click.ClickException("No valid years available for requested range.")

    click.echo(f"Marine heatwaves pipeline: years={years[0]}-{years[-1]}, plot_every={plot_every}")
    plot_years = set(range(years[0], years[-1] + 1, max(1, plot_every)))
    plot_years.add(years[-1])

    ts_years: dict[str, list[int]] = {k: [] for k in variables}
    ts_vals: dict[str, list[float]] = {k: [] for k in variables}

    for year in years:
        da_y = da_anom.sel(time=str(year))
        month_idx = xr.DataArray(da_y.time.dt.month.values, dims="time", coords={"time": da_y.time})
        thr_y = da_p90.sel(month=month_idx)

        exceed = da_y > thr_y
        mhw_months = exceed.sum(dim="time")
        excess = (da_y - thr_y).where(exceed)
        mhw_excess = excess.mean(dim="time", skipna=True).fillna(0.0)

        annual_data = {
            "heatwave_marine_months": mhw_months,
            "heatwave_marine_mean_excess": mhw_excess,
        }

        for var_key, da_year in annual_data.items():
            info = variables[var_key]
            da_025 = _to_target_grid(da_year)
            _save_year(
                var_key=var_key,
                long_name=info["long_name"],
                units=info["units"],
                year=year,
                arr2d=da_025.values,
                domain=domain,
                source_file=anom_path.name,
                overwrite=overwrite,
            )
            ts_years[var_key].append(year)
            ts_vals[var_key].append(float(da_025.mean(skipna=True).values))
            if year in plot_years:
                _plot_map(
                    var_key=var_key,
                    year=year,
                    long_name=info["long_name"],
                    cmap=info.get("cmap", "viridis"),
                    domain=domain,
                )
        gc.collect()

    for var_key, info in variables.items():
        _plot_series(
            var_key=var_key,
            long_name=info["long_name"],
            units=info["units"],
            years=ts_years[var_key],
            vals=ts_vals[var_key],
        )

    ds_anom.close()
    ds_p90.close()
    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR / domain} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
