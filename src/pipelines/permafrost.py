"""Full permafrost pipeline: download -> downscale -> harmonize -> plot."""

from __future__ import annotations

import gc
import subprocess
from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml

from src.download.permafrost import DEFAULT_RAW_DIR, download_permafrost
from src.processing.raster_to_grid import load_raster, regrid_raster
from src.utils import get_logger, plot_global_map, plot_time_series

logger = get_logger("pipeline.permafrost")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "permafrost.yml"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
PLOTS_DIR = PROJECT_ROOT / "plots" / "permafrost"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_datasets(config: dict, selected: tuple[str, ...], run_all: bool) -> dict:
    all_ds = config.get("datasets", {})
    if selected:
        resolved = {k: v for k, v in all_ds.items() if k in selected}
        missing = sorted(set(selected) - set(resolved))
        if missing:
            logger.warning("Unknown permafrost datasets skipped: %s", missing)
        return resolved
    if run_all:
        return all_ds
    return {}


def _resolve_years(config: dict, years: tuple[int, ...], start_year: int | None, end_year: int | None) -> list[int]:
    if years:
        return sorted(set(int(y) for y in years))
    y0, y1 = config["temporal_range"]
    s = int(start_year) if start_year is not None else int(y0)
    e = int(end_year) if end_year is not None else int(y1)
    if e < s:
        raise ValueError("end-year must be >= start-year")
    return list(range(s, e + 1))


def _warp_subdataset_to_025(
    src_nc: Path,
    source_var: str,
    dst_tif: Path,
    method: str = "average",
    src_nodata: float | int | None = None,
    dst_nodata: float = -9999.0,
) -> None:
    src_subdataset = f'NETCDF:"{src_nc}":{source_var}'
    dst_tif.parent.mkdir(parents=True, exist_ok=True)

    cmd = [
        "gdalwarp",
        "-q",
        "-overwrite",
        "-of",
        "GTiff",
        "-t_srs",
        "EPSG:4326",
        "-te",
        "-180",
        "-90",
        "180",
        "90",
        "-tr",
        "0.25",
        "0.25",
        "-tap",
        "-r",
        method,
        "-ot",
        "Float32",
        "-dstnodata",
        str(dst_nodata),
    ]
    if src_nodata is not None:
        cmd.extend(["-srcnodata", str(src_nodata)])
    cmd.extend([src_subdataset, str(dst_tif)])
    subprocess.run(cmd, check=True)


def _save_year(ds_025: xr.Dataset, var_key: str, info: dict, year: int, domain: str, source_file: str, overwrite: bool):
    out_path = FINAL_DIR / domain / var_key / f"{year}.nc"
    if out_path.exists() and not overwrite:
        return out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = ds_025[var_key].astype(np.float32)
    clip_min = info.get("clip_min")
    clip_max = info.get("clip_max")
    if clip_min is not None or clip_max is not None:
        data = data.clip(min=clip_min, max=clip_max)
    ds_out = xr.Dataset(
        {
            var_key: (
                ["lat", "lon"],
                data.values,
                {"long_name": info["long_name"], "units": info["units"]},
            )
        },
        coords=ds_025.coords,
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor {info['long_name']}",
            "source": "ESA CCI Permafrost v05",
            "source_file": source_file,
            "year": year,
        },
    )
    ds_out.to_netcdf(out_path, encoding={var_key: {"zlib": True, "complevel": 4, "dtype": "float32"}})
    ds_out.close()
    return out_path


def _plot_map(var_key: str, year: int, info: dict, domain: str):
    nc_path = FINAL_DIR / domain / var_key / f"{year}.nc"
    if not nc_path.exists():
        return
    ds = xr.open_dataset(nc_path, decode_timedelta=False)
    out_path = PLOTS_DIR / "maps" / var_key / f"{year}.png"
    try:
        plot_global_map(ds[var_key], title=f"{info['long_name']} ({year})", out_path=out_path, cmap=info.get("cmap", "viridis"))
    finally:
        ds.close()


def _plot_series(var_key: str, info: dict, years: list[int], vals: list[float]):
    if not years:
        return
    out_path = PLOTS_DIR / "timeseries" / f"{var_key}.png"
    plot_time_series(
        years=years,
        values=vals,
        title=f"{info['long_name']} (global mean)",
        ylabel=info["units"],
        out_path=out_path,
        color="#2ca02c",
    )


@click.command()
@click.option("--datasets", "-d", multiple=True, help="Dataset keys.")
@click.option("--all", "run_all", is_flag=True, help="Process all configured datasets.")
@click.option("--years", "-y", multiple=True, type=int, help="Specific years.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--plot-every", type=int, default=5, show_default=True, help="Map every N years.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
def main(
    datasets: tuple[str, ...],
    run_all: bool,
    years: tuple[int, ...],
    start_year: int | None,
    end_year: int | None,
    plot_every: int,
    skip_download: bool,
    overwrite: bool,
):
    config = load_config()
    selected = _resolve_datasets(config, datasets, run_all)
    if not selected:
        click.echo("Specify --datasets or --all")
        return

    year_list = _resolve_years(config, years, start_year, end_year)
    if not year_list:
        raise click.ClickException("No years selected.")
    domain = config.get("domain", "Cryosphere")

    click.echo(
        f"Permafrost pipeline: datasets={list(selected)}, years={year_list[0]}-{year_list[-1]}, plot_every={plot_every}"
    )

    if not skip_download:
        download_permafrost(
            dataset_keys=tuple(selected.keys()),
            run_all=False,
            years=tuple(year_list),
            raw_dir=DEFAULT_RAW_DIR,
            overwrite=overwrite,
        )

    for key, info in selected.items():
        ts_years: list[int] = []
        ts_vals: list[float] = []
        work_dir = DEFAULT_RAW_DIR / key / "work"
        work_dir.mkdir(parents=True, exist_ok=True)

        plot_years = set(range(year_list[0], year_list[-1] + 1, max(1, plot_every)))
        plot_years.add(year_list[-1])

        for year in year_list:
            raw_nc = DEFAULT_RAW_DIR / key / f"{year}.nc"
            if not raw_nc.exists():
                continue

            warped_tif = work_dir / f"{year}_025.tif"
            if overwrite or not warped_tif.exists():
                _warp_subdataset_to_025(
                    src_nc=raw_nc,
                    source_var=info["source_var"],
                    dst_tif=warped_tif,
                    method=info.get("downscale_method", "average"),
                    src_nodata=info.get("src_nodata"),
                    dst_nodata=-9999.0,
                )

            da = load_raster(warped_tif)
            squeeze_dims = [d for d in da.dims if d not in ("x", "y", "lon", "lat") and da.sizes.get(d, 0) == 1]
            if squeeze_dims:
                da = da.isel({d: 0 for d in squeeze_dims}, drop=True)
            ds_025 = regrid_raster(da, year=year, var_name=key, method="linear")
            data = ds_025[key].where(ds_025[key] > -9000)
            scale_factor = info.get("scale_factor")
            if scale_factor is not None:
                data = data * float(scale_factor)
            clip_min = info.get("clip_min")
            clip_max = info.get("clip_max")
            if clip_min is not None or clip_max is not None:
                data = data.clip(min=clip_min, max=clip_max)
            ds_025[key] = data

            _save_year(
                ds_025=ds_025,
                var_key=key,
                info=info,
                year=year,
                domain=domain,
                source_file=raw_nc.name,
                overwrite=overwrite,
            )

            ts_years.append(year)
            ts_vals.append(float(data.mean(skipna=True).values))

            if year in plot_years:
                _plot_map(key, year, info, domain)

            ds_025.close()
            gc.collect()

        _plot_series(key, info, ts_years, ts_vals)

    click.echo(f"\nPipeline complete. Outputs in {FINAL_DIR / domain} and {PLOTS_DIR}")


if __name__ == "__main__":
    main()
