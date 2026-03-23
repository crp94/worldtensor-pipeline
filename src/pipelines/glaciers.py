"""Full glacier pipeline: download -> process -> plot."""

from __future__ import annotations

import gc
from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml

from src.download.glaciers import DEFAULT_RAW_DIR, download_glaciers
from src.grid import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, N_LAT, N_LON
from src.utils import get_logger, plot_global_map, plot_time_series

logger = get_logger("pipeline.glaciers")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "glaciers.yml"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
PLOTS_DIR = PROJECT_ROOT / "plots" / "glaciers"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_variables(config: dict, selected: tuple[str, ...], run_all: bool) -> dict:
    all_vars = config.get("variables", {})
    if selected:
        resolved = {k: v for k, v in all_vars.items() if k in selected}
        missing = sorted(set(selected) - set(resolved))
        if missing:
            logger.warning("Unknown glacier variables skipped: %s", missing)
        return resolved
    if run_all:
        return all_vars
    return {}


def _normalize_source(ds: xr.Dataset) -> xr.Dataset:
    if "lat" in ds.coords and float(ds.lat.values[0]) > float(ds.lat.values[-1]):
        ds = ds.isel(lat=slice(None, None, -1))
    if "lon" in ds.coords and float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360.0))
        ds = ds.sortby("lon")
    return ds


def _pad_periodic_longitude(da: xr.DataArray, pad_cells: int = 2) -> xr.DataArray:
    da = da.sortby("lon")
    lon_vals = da["lon"].values.astype(float)
    _, unique_idx = np.unique(lon_vals, return_index=True)
    da = da.isel(lon=np.sort(unique_idx))

    left = da.isel(lon=slice(-pad_cells, None)).copy()
    right = da.isel(lon=slice(0, pad_cells)).copy()
    left = left.assign_coords(lon=left.lon - 360.0)
    right = right.assign_coords(lon=right.lon + 360.0)
    return xr.concat([left, da, right], dim="lon")


def _to_target_grid(da: xr.DataArray) -> xr.DataArray:
    same_shape = da.sizes.get("lat") == N_LAT and da.sizes.get("lon") == N_LON
    if same_shape:
        lat_ok = np.isclose(float(da.lat.min()), LAT_MIN) and np.isclose(float(da.lat.max()), LAT_MAX)
        lon_ok = np.isclose(float(da.lon.min()), LON_MIN) and np.isclose(float(da.lon.max()), LON_MAX)
        if lat_ok and lon_ok:
            return da
    return _pad_periodic_longitude(da).interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear")


def _save_year(var_key: str, long_name: str, units: str, year: int, data_2d: np.ndarray, source_file: str, domain: str, overwrite: bool):
    out_path = FINAL_DIR / domain / var_key / f"{year}.nc"
    if out_path.exists() and not overwrite:
        return out_path

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds_out = xr.Dataset(
        {
            var_key: (
                ["lat", "lon"],
                data_2d.astype(np.float32),
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
            "source": "WGMS AMCE",
            "source_file": source_file,
            "year": year,
        },
    )
    ds_out.to_netcdf(out_path, encoding={var_key: {"zlib": True, "complevel": 4, "dtype": "float32"}})
    ds_out.close()
    return out_path


def _plot_map(var_key: str, year: int, long_name: str, cmap: str, domain: str):
    nc_path = FINAL_DIR / domain / var_key / f"{year}.nc"
    if not nc_path.exists():
        return
    ds = xr.open_dataset(nc_path, decode_timedelta=False)
    out_path = PLOTS_DIR / "maps" / var_key / f"{year}.png"
    try:
        plot_global_map(ds[var_key], title=f"{long_name} ({year})", out_path=out_path, cmap=cmap)
    finally:
        ds.close()


def _plot_series(var_key: str, long_name: str, units: str, years: list[int], values: list[float]):
    if not years:
        return
    out_path = PLOTS_DIR / "timeseries" / f"{var_key}.png"
    plot_time_series(
        years=years,
        values=values,
        title=f"{long_name} (global mean)",
        ylabel=units,
        out_path=out_path,
        color="#1f77b4",
    )


@click.command()
@click.option("--variables", "-v", multiple=True, help="Variable keys.")
@click.option("--all", "run_all", is_flag=True, help="Process all configured variables.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--plot-every", type=int, default=10, show_default=True, help="Map every N years.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--cleanup-raw/--keep-raw", default=False, show_default=True, help="Delete raw files after processing.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
def main(
    variables: tuple[str, ...],
    run_all: bool,
    start_year: int | None,
    end_year: int | None,
    plot_every: int,
    skip_download: bool,
    cleanup_raw: bool,
    overwrite: bool,
):
    config = load_config()
    selected = _resolve_variables(config, variables, run_all)
    if not selected:
        click.echo("Specify --variables or --all")
        return

    if skip_download:
        nc_path = DEFAULT_RAW_DIR / config["netcdf_filename"]
    else:
        nc_path = download_glaciers(raw_dir=DEFAULT_RAW_DIR, overwrite=overwrite)

    if not nc_path.exists():
        raise click.ClickException(f"Missing glacier source file: {nc_path}")

    ds = xr.open_dataset(nc_path)
    ds = _normalize_source(ds)
    source_years = ds.time.dt.year.astype(int).values.tolist()
    y0 = start_year if start_year is not None else int(config["temporal_range"][0])
    y1 = end_year if end_year is not None else int(config["temporal_range"][1])
    years = [y for y in range(y0, y1 + 1) if y in source_years]
    if not years:
        ds.close()
        raise click.ClickException("No overlapping years between request and source file.")

    domain = config.get("domain", "Cryosphere")
    plot_years = set(range(years[0], years[-1] + 1, max(1, plot_every)))
    plot_years.add(years[-1])

    click.echo(f"Glacier pipeline: vars={list(selected)}, years={years[0]}-{years[-1]}, plot_every={plot_every}")

    for var_key, info in selected.items():
        source_var = info["source_var"]
        if source_var not in ds.data_vars:
            logger.warning("Source variable missing: %s", source_var)
            continue

        ts_years: list[int] = []
        ts_vals: list[float] = []

        for year in years:
            da_year = ds[source_var].sel(time=f"{year}-01-01")
            da_year = _to_target_grid(da_year)
            _save_year(
                var_key=var_key,
                long_name=info["long_name"],
                units=info["units"],
                year=year,
                data_2d=da_year.values,
                source_file=nc_path.name,
                domain=domain,
                overwrite=overwrite,
            )
            ts_years.append(year)
            ts_vals.append(float(da_year.mean(skipna=True).values))
            if year in plot_years:
                _plot_map(
                    var_key=var_key,
                    year=year,
                    long_name=info["long_name"],
                    cmap=info.get("cmap", "viridis"),
                    domain=domain,
                )
            gc.collect()

        _plot_series(
            var_key=var_key,
            long_name=info["long_name"],
            units=info["units"],
            years=ts_years,
            values=ts_vals,
        )

    ds.close()

    if cleanup_raw:
        for p in DEFAULT_RAW_DIR.glob("*"):
            if p.is_file():
                p.unlink(missing_ok=True)

    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR / domain} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
