"""Full precomputed land heatwave pipeline: download -> process -> plot.

Uses HadEX3 annual EHF heatwave indices (precomputed) and writes yearly
0.25° outputs:
  data/final/extremes/{variable}/{YYYY}.nc
"""

from __future__ import annotations

import gc
from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml

from src.download.land_heatwaves import DEFAULT_RAW_DIR, download_land_heatwaves
from src.grid import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, N_LAT, N_LON
from src.utils import get_logger, plot_global_map, plot_time_series

logger = get_logger("pipeline.land_heatwaves")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "land_heatwaves.yml"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "extremes"
PLOTS_DIR = PROJECT_ROOT / "plots" / "land_heatwaves"

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
            logger.warning("Unknown variables skipped: %s", missing)
        return resolved
    if run_all:
        return all_vars
    return {}


def _normalize_dataset(ds: xr.Dataset) -> xr.Dataset:
    rename = {}
    if "latitude" in ds.dims:
        rename["latitude"] = "lat"
    if "longitude" in ds.dims:
        rename["longitude"] = "lon"
    if rename:
        ds = ds.rename(rename)

    if "lat" in ds.coords and float(ds.lat.values[0]) > float(ds.lat.values[-1]):
        ds = ds.isel(lat=slice(None, None, -1))

    if "lon" in ds.coords and float(ds.lon.min()) < 0:
        ds = ds.assign_coords(lon=(ds.lon % 360.0))
        ds = ds.sortby("lon")

    return ds


def _pick_data_var(ds: xr.Dataset, preferred: str | None = None) -> str:
    if preferred and preferred in ds.data_vars:
        return preferred
    candidates = [v for v in ds.data_vars if ds[v].ndim >= 2 and "time" in ds[v].dims]
    if candidates:
        return candidates[0]
    raise ValueError(f"No time-varying data variable found. Available: {list(ds.data_vars)}")


def _prepare_yearly_series(
    da: xr.DataArray,
    start_year: int,
    end_year: int,
    interpolate_missing: bool,
) -> tuple[xr.DataArray, list[int]]:
    if "time" not in da.dims:
        raise ValueError("Expected a time dimension in source data.")

    years = da.time.dt.year.astype(int).values
    available = [int(y) for y in years if start_year <= int(y) <= end_year]
    if not available:
        return da.isel(time=slice(0, 0)), []

    s_year = max(start_year, min(available))
    e_year = min(end_year, max(available))
    target_years = list(range(s_year, e_year + 1))

    da = da.sel(time=slice(np.datetime64(f"{s_year}-01-01"), np.datetime64(f"{e_year}-12-31")))

    target_time = xr.DataArray(
        [np.datetime64(f"{y}-01-01") for y in target_years],
        dims="time",
        name="time",
    )
    da = da.reindex(time=target_time)

    if interpolate_missing:
        da = da.interpolate_na(dim="time", method="linear", use_coordinate=True)

    return da, target_years


def _to_target_grid(da: xr.DataArray) -> xr.DataArray:
    same_shape = da.sizes.get("lat") == N_LAT and da.sizes.get("lon") == N_LON
    if same_shape:
        lat_ok = np.isclose(float(da.lat.min()), LAT_MIN) and np.isclose(float(da.lat.max()), LAT_MAX)
        lon_ok = np.isclose(float(da.lon.min()), LON_MIN) and np.isclose(float(da.lon.max()), LON_MAX)
        if lat_ok and lon_ok:
            return da

    da_wrapped = _pad_periodic_longitude(da, lon_name="lon", pad_cells=2)
    return da_wrapped.interp(lat=TARGET_LAT, lon=TARGET_LON, method="linear")


def _pad_periodic_longitude(da: xr.DataArray, lon_name: str = "lon", pad_cells: int = 2) -> xr.DataArray:
    """Pad longitude periodically so interpolation near 0/360 has neighbors on both sides."""
    if lon_name not in da.coords:
        return da
    if da.sizes.get(lon_name, 0) < 4:
        return da

    da = da.sortby(lon_name)
    lon_vals = da[lon_name].values.astype(float)
    _, unique_idx = np.unique(lon_vals, return_index=True)
    da = da.isel({lon_name: np.sort(unique_idx)})

    left = da.isel({lon_name: slice(-pad_cells, None)}).copy()
    right = da.isel({lon_name: slice(0, pad_cells)}).copy()
    left = left.assign_coords({lon_name: left[lon_name] - 360.0})
    right = right.assign_coords({lon_name: right[lon_name] + 360.0})

    return xr.concat([left, da, right], dim=lon_name)


def _extract_year_slice(da: xr.DataArray, year: int) -> xr.DataArray | None:
    sub = da.where(da.time.dt.year == year, drop=True)
    if sub.sizes.get("time", 0) == 0:
        return None
    if sub.sizes["time"] > 1:
        sub = sub.mean(dim="time", skipna=True)
    else:
        sub = sub.isel(time=0)
    return sub


def _save_year(da_2d: xr.DataArray, var_key: str, long_name: str, units: str, year: int, source_file: str, overwrite: bool):
    out_dir = FINAL_DIR / var_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}.nc"
    if out_path.exists() and not overwrite:
        return out_path

    ds_out = xr.Dataset(
        {
            var_key: (
                ["lat", "lon"],
                da_2d.values.astype(np.float32),
                {
                    "units": units,
                    "long_name": long_name,
                },
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor Land Heatwaves ({var_key})",
            "source": "HadEX3 (Met Office Hadley Centre)",
            "source_file": source_file,
            "year": year,
        },
    )
    ds_out.to_netcdf(out_path, encoding={var_key: {"zlib": True, "complevel": 4, "dtype": "float32"}})
    ds_out.close()
    return out_path


def _plot_map(var_key: str, year: int, cmap: str, long_name: str):
    nc_path = FINAL_DIR / var_key / f"{year}.nc"
    if not nc_path.exists():
        return
    out_path = PLOTS_DIR / "maps" / var_key / f"{year}.png"
    ds = xr.open_dataset(nc_path, decode_timedelta=False)
    try:
        da = ds[var_key]
        plot_global_map(
            da=da,
            title=f"{long_name} ({year})",
            out_path=out_path,
            cmap=cmap,
        )
    finally:
        ds.close()


def _plot_series(var_key: str, long_name: str, units: str, years: list[int], values: list[float]):
    if not years:
        return
    out_path = PLOTS_DIR / "timeseries" / f"{var_key}.png"
    ylabel = units if units else var_key
    plot_time_series(
        years=years,
        values=values,
        title=f"{long_name} (global land mean)",
        ylabel=ylabel,
        out_path=out_path,
        color="#d62728",
    )


def process_variable(
    var_key: str,
    info: dict,
    raw_nc_path: Path,
    start_year: int,
    end_year: int,
    interpolate_missing: bool,
    plot_every: int,
    overwrite: bool,
) -> tuple[list[int], list[float]]:
    if not raw_nc_path.exists():
        logger.warning("Missing raw file for %s: %s", var_key, raw_nc_path)
        return [], []

    ds = xr.open_dataset(raw_nc_path, decode_timedelta=False)
    try:
        ds = _normalize_dataset(ds)
        data_var = _pick_data_var(ds, preferred=info.get("source_var"))
        da = ds[data_var]
        da, target_years = _prepare_yearly_series(
            da=da,
            start_year=start_year,
            end_year=end_year,
            interpolate_missing=interpolate_missing,
        )
        if not target_years:
            logger.warning("No overlapping years for %s in requested window.", var_key)
            return [], []

        da = _to_target_grid(da)
        years_out: list[int] = []
        means_out: list[float] = []

        plot_years = set(range(target_years[0], target_years[-1] + 1, max(1, plot_every)))
        plot_years.add(target_years[-1])

        for year in target_years:
            da_year = _extract_year_slice(da, year)
            if da_year is None:
                continue

            _save_year(
                da_2d=da_year,
                var_key=var_key,
                long_name=info["long_name"],
                units=info["units"],
                year=year,
                source_file=raw_nc_path.name,
                overwrite=overwrite,
            )

            years_out.append(year)
            means_out.append(float(da_year.mean(skipna=True).values))

            if year in plot_years:
                try:
                    _plot_map(
                        var_key=var_key,
                        year=year,
                        cmap=info.get("cmap", "YlOrRd"),
                        long_name=info["long_name"],
                    )
                except Exception as e:
                    logger.warning("Map plotting failed for %s/%d: %s", var_key, year, e)

        return years_out, means_out
    finally:
        ds.close()


@click.command()
@click.option("--variables", "-v", multiple=True, help="Variable keys to process.")
@click.option("--all", "run_all", is_flag=True, help="Process all configured variables.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--plot-every", type=int, default=10, show_default=True, help="Save map every N years.")
@click.option("--skip-download", is_flag=True, help="Skip download and use existing raw NetCDF files.")
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

    t_start_cfg, t_end_cfg = config["temporal_range"]
    t_start = start_year if start_year is not None else int(t_start_cfg)
    t_end = end_year if end_year is not None else int(t_end_cfg)
    if t_end < t_start:
        raise click.ClickException("end-year must be >= start-year")

    interpolate_missing = bool(config.get("interpolate_missing_years", True))

    click.echo(
        f"Land heatwaves pipeline: vars={list(selected)}, years={t_start}-{t_end}, "
        f"plot_every={plot_every}, interpolate_missing={interpolate_missing}"
    )

    downloaded: dict[str, Path] = {}
    if not skip_download:
        downloaded = download_land_heatwaves(
            selected_variables=tuple(selected.keys()),
            run_all=False,
            raw_dir=DEFAULT_RAW_DIR,
            overwrite=overwrite,
        )

    for var_key, info in selected.items():
        raw_name = info["source_filename"].replace(".nc.gz", ".nc")
        raw_path = downloaded.get(var_key, DEFAULT_RAW_DIR / raw_name)
        years, means = process_variable(
            var_key=var_key,
            info=info,
            raw_nc_path=raw_path,
            start_year=t_start,
            end_year=t_end,
            interpolate_missing=interpolate_missing,
            plot_every=plot_every,
            overwrite=overwrite,
        )
        _plot_series(
            var_key=var_key,
            long_name=info["long_name"],
            units=info["units"],
            years=years,
            values=means,
        )
        gc.collect()

    if cleanup_raw:
        for info in selected.values():
            gz_path = DEFAULT_RAW_DIR / info["source_filename"]
            nc_path = DEFAULT_RAW_DIR / info["source_filename"].replace(".nc.gz", ".nc")
            gz_path.unlink(missing_ok=True)
            nc_path.unlink(missing_ok=True)

    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
