"""Candidate Kummu socioeconomic pipeline: download -> process -> plot.

This pipeline is intended for dataset evaluation before promotion to the main
WorldTensor variable registry.

Usage:
    python -m src.pipelines.kummu_candidates --all
    python -m src.pipelines.kummu_candidates --all --overwrite
    python -m src.pipelines.kummu_candidates --process --plot --skip-download
    python -m src.pipelines.kummu_candidates -d gni_per_capita --all
"""

from __future__ import annotations

import gc
import re
import subprocess
from pathlib import Path

import click
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import requests
import rioxarray
import xarray as xr
import yaml
from tqdm import tqdm

from src.processing.raster_to_grid import regrid_raster
from src.utils import get_logger
try:
    from src.visualization.plot_raster import plot_variable
except ImportError:
    plot_variable = None

logger = get_logger("pipeline.kummu_candidates")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "kummu_candidates.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "kummu_candidates"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
PLOTS_DIR = PROJECT_ROOT / "plots" / "kummu_candidates"
CHUNK_SIZE = 8 * 1024 * 1024


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_datasets(config: dict, dataset_keys: tuple[str, ...]) -> dict:
    datasets = config.get("datasets", {})
    if dataset_keys:
        selected = {k: v for k, v in datasets.items() if k in dataset_keys}
        missing = sorted(set(dataset_keys) - set(selected))
        if missing:
            logger.warning("Unknown dataset keys ignored: %s", missing)
        return selected
    return {k: v for k, v in datasets.items() if v.get("enabled", True)}


def download_datasets(datasets: dict, overwrite: bool = False) -> list[Path]:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded = []
    for key, info in datasets.items():
        out_path = RAW_DIR / info["filename"]
        url = info["source_url"]
        if out_path.exists() and not overwrite:
            logger.info("Skipping download (exists): %s", out_path.name)
            downloaded.append(out_path)
            continue

        tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
        logger.info("Downloading %s from %s", key, url)
        try:
            resp = requests.get(url, stream=True, timeout=600)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with (
                open(tmp_path, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc=key) as pbar,
            ):
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
            tmp_path.replace(out_path)
            downloaded.append(out_path)
            logger.info("Downloaded -> %s", out_path)
        except Exception as e:
            logger.error("Download failed for %s: %s", key, e)
            tmp_path.unlink(missing_ok=True)
    return downloaded


def _parse_year(label: str, index0: int, pattern: str) -> int:
    m = re.search(pattern, label)
    if m:
        return int(m.group(0))
    # Fallback: assume first band/year is 1990
    return 1990 + index0


def _process_multiband_geotiff(
    dataset_key: str,
    info: dict,
    raw_path: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> int:
    cube = rioxarray.open_rasterio(raw_path, masked=True)
    descriptions = cube.attrs.get("long_name")
    band_count = int(cube.sizes["band"])
    processed = 0
    clip_min = info.get("clip_min")
    clip_max = info.get("clip_max")

    for i in range(band_count):
        label = str(descriptions[i]) if isinstance(descriptions, (list, tuple)) else f"band_{i + 1}"
        year = _parse_year(label, i, info.get("year_pattern", r"(19|20)\d{2}"))
        out_path = out_dir / f"{year}.nc"
        if out_path.exists() and not overwrite:
            continue

        da = cube.isel(band=i).squeeze(drop=True).rename({"x": "lon", "y": "lat"})
        nodata = info.get("nodata_value")
        if nodata is not None:
            da = da.where(da != nodata)

        da.name = info["var_name"]
        da.attrs["units"] = info["units"]
        da.attrs["long_name"] = info["long_name"]
        if clip_min is not None or clip_max is not None:
            da = da.clip(min=clip_min, max=clip_max)

        ds_out = regrid_raster(da, year=year, var_name=info["var_name"], method="linear")
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(
            out_path,
            encoding={info["var_name"]: {"zlib": True, "complevel": 4, "dtype": "float32"}},
        )
        processed += 1
        gc.collect()

    cube.close()
    logger.info("Processed %s (%d yearly files)", dataset_key, processed)
    return processed


def _year_from_time_value(val) -> int:
    if np.issubdtype(type(val), np.datetime64):
        return int(str(val)[:4])
    return int(float(val))


def _extract_archive_if_needed(dataset_key: str, raw_path: Path, overwrite: bool = False) -> Path:
    extract_dir = RAW_DIR / f"{dataset_key}_extracted"
    if extract_dir.exists() and not overwrite and any(extract_dir.rglob("*.tif")):
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["7z", "x", "-y", str(raw_path), f"-o{extract_dir}"]
    logger.info("Extracting archive for %s -> %s", dataset_key, extract_dir)
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as e:
        stderr = (e.stderr or "").strip()
        raise RuntimeError(f"7z extraction failed for {raw_path.name}: {stderr}") from e
    return extract_dir


def _process_archive_yearly_tiffs(
    dataset_key: str,
    info: dict,
    raw_path: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> int:
    extract_dir = _extract_archive_if_needed(dataset_key, raw_path, overwrite=overwrite)
    tiff_paths = sorted(extract_dir.glob(info.get("archive_glob", "**/*.tif")))
    if not tiff_paths:
        raise ValueError(f"No TIFF files found for {dataset_key} in {extract_dir}")

    exclude_regex = info.get("exclude_regex")
    year_pattern = info.get("year_pattern", r"(19|20)\d{2}")
    y0 = info.get("historical_start_year")
    y1 = info.get("historical_end_year")
    method = info.get("regrid_method", "linear")
    processed = 0

    for tif_path in tiff_paths:
        name = tif_path.name
        if exclude_regex and re.search(exclude_regex, name):
            continue

        m = re.search(year_pattern, name)
        if not m:
            continue
        year = int(m.group(0))
        if y0 is not None and year < int(y0):
            continue
        if y1 is not None and year > int(y1):
            continue

        out_path = out_dir / f"{year}.nc"
        if out_path.exists() and not overwrite:
            continue

        da = rioxarray.open_rasterio(tif_path, masked=True).squeeze(drop=True).rename({"x": "lon", "y": "lat"})
        nodata = info.get("nodata_value")
        if nodata is not None:
            da = da.where(da != nodata)

        da.name = info["var_name"]
        da.attrs["units"] = info["units"]
        da.attrs["long_name"] = info["long_name"]

        clip_min = info.get("clip_min")
        clip_max = info.get("clip_max")
        if clip_min is not None or clip_max is not None:
            da = da.clip(min=clip_min, max=clip_max)

        ds_out = regrid_raster(da, year=year, var_name=info["var_name"], method=method)
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(
            out_path,
            encoding={info["var_name"]: {"zlib": True, "complevel": 4, "dtype": "float32"}},
        )
        da.close()
        ds_out.close()
        processed += 1
        gc.collect()

    logger.info("Processed %s (%d yearly files)", dataset_key, processed)
    return processed


def _process_netcdf_timecube(
    dataset_key: str,
    info: dict,
    raw_path: Path,
    out_dir: Path,
    overwrite: bool = False,
) -> int:
    src_var = info["source_var"]
    lat_name = info.get("lat_name", "lat")
    lon_name = info.get("lon_name", "lon")
    time_name = info.get("time_name", "time")
    ds = xr.open_dataset(raw_path)
    if src_var not in ds.data_vars:
        ds.close()
        raise ValueError(f"Variable '{src_var}' not found in {raw_path.name}")

    cube = ds[src_var].rename({lat_name: "lat", lon_name: "lon"})
    clip_min = info.get("clip_min")
    clip_max = info.get("clip_max")
    processed = 0
    for t in cube[time_name].values:
        year = _year_from_time_value(t)
        out_path = out_dir / f"{year}.nc"
        if out_path.exists() and not overwrite:
            continue
        da = cube.sel({time_name: t})
        da.name = info["var_name"]
        da.attrs["units"] = info["units"]
        da.attrs["long_name"] = info["long_name"]
        if clip_min is not None or clip_max is not None:
            da = da.clip(min=clip_min, max=clip_max)

        ds_out = regrid_raster(da, year=year, var_name=info["var_name"], method="linear")
        out_dir.mkdir(parents=True, exist_ok=True)
        ds_out.to_netcdf(
            out_path,
            encoding={info["var_name"]: {"zlib": True, "complevel": 4, "dtype": "float32"}},
        )
        processed += 1
        gc.collect()

    ds.close()
    logger.info("Processed %s (%d yearly files)", dataset_key, processed)
    return processed


def process_datasets(datasets: dict, overwrite: bool = False) -> int:
    total = 0
    for key, info in datasets.items():
        raw_path = RAW_DIR / info["filename"]
        if not raw_path.exists():
            logger.warning("Raw file missing for %s: %s", key, raw_path)
            continue
        out_dir = FINAL_DIR / info["domain"] / key
        fmt = info["format"]
        if fmt == "multiband_geotiff":
            total += _process_multiband_geotiff(key, info, raw_path, out_dir, overwrite)
        elif fmt == "netcdf_timecube":
            total += _process_netcdf_timecube(key, info, raw_path, out_dir, overwrite)
        elif fmt == "archive_yearly_tiffs":
            total += _process_archive_yearly_tiffs(key, info, raw_path, out_dir, overwrite)
        else:
            logger.warning("Unsupported format '%s' for %s", fmt, key)
    return total


def _plot_timeseries(dataset_key: str, info: dict, data_dir: Path, output_path: Path) -> None:
    var_name = info["var_name"]
    nc_files = sorted(data_dir.glob("*.nc"))
    years, means = [], []
    for p in nc_files:
        year = int(p.stem)
        ds = xr.open_dataset(p)
        years.append(year)
        means.append(float(ds[var_name].mean(skipna=True)))
        ds.close()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(years, means, marker="o", markersize=2.5, linewidth=1.2)
    ax.set_title(dataset_key)
    ax.set_xlabel("Year")
    ax.set_ylabel("Spatial mean")
    ax.grid(alpha=0.3)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_datasets(datasets: dict, map_every: int = 10) -> None:
    for key, info in datasets.items():
        var_name = info["var_name"]
        data_dir = FINAL_DIR / info["domain"] / key
        nc_files = sorted(data_dir.glob("*.nc"))
        if not nc_files:
            logger.warning("No processed files to plot for %s", key)
            continue
        years = sorted(int(p.stem) for p in nc_files)
        selected_years = sorted({years[0], years[-1], *[y for y in years if y % map_every == 0]})

        if not bool(info.get("skip_map_plot", False)):
            map_dir = PLOTS_DIR / "maps" / key
            for year in selected_years:
                ds = xr.open_dataset(data_dir / f"{year}.nc")
                if plot_variable is not None:
                    plot_variable(
                    ds=ds,
                    var_name=var_name,
                    year=year,
                    output_path=map_dir / f"{year}.png",
                    cmap=info.get("cmap", "viridis"),
                    log_scale=bool(info.get("log_scale", False)),
                )
                ds.close()

        _plot_timeseries(
            dataset_key=key,
            info=info,
            data_dir=data_dir,
            output_path=PLOTS_DIR / "timeseries" / f"{key}.png",
        )


@click.command()
@click.option("--datasets", "-d", multiple=True, help="Dataset key(s) from config.")
@click.option("--all", "run_all", is_flag=True, help="Run download + process + plot.")
@click.option("--download", is_flag=True, help="Run download step.")
@click.option("--process", is_flag=True, help="Run processing step.")
@click.option("--plot", "plot_step", is_flag=True, help="Run plotting step.")
@click.option("--skip-download", is_flag=True, help="Skip download when using --all.")
@click.option("--map-every", type=int, default=10, show_default=True, help="Plot map every N years.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
def main(datasets, run_all, download, process, plot_step, skip_download, map_every, overwrite):
    """Run the Kummu candidate socioeconomic data pipeline."""
    config = load_config()
    selected = _resolve_datasets(config, datasets)
    if not selected:
        raise click.ClickException("No datasets selected.")

    if run_all:
        do_download = not skip_download
        do_process = True
        do_plot = True
    else:
        do_download = download
        do_process = process
        do_plot = plot_step

    if not any([do_download, do_process, do_plot]):
        raise click.ClickException("Specify --all or at least one of --download/--process/--plot.")

    if do_download:
        download_datasets(selected, overwrite=overwrite)
    if do_process:
        n = process_datasets(selected, overwrite=overwrite)
        logger.info("Total yearly files written: %d", n)
    if do_plot:
        plot_datasets(selected, map_every=map_every)
        logger.info("Plots written under: %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
