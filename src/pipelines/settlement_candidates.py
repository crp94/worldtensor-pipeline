"""Settlement/built-environment candidate pipeline: download -> process -> plot.

Includes:
1) WSF Evolution (1985-2015, first-detection year tiles -> yearly presence)
2) Annual urban extent from harmonized NTL (1992-2020)
3) GISA impervious dynamics (1972-2019, first-detection code tiles -> yearly presence)
5) GHSL built surface/volume (5-year epochs -> yearly linear interpolation),
   and GHSL building height snapshot (2018).
"""

from __future__ import annotations

import gc
import re
import subprocess
import zipfile
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

logger = get_logger("pipeline.settlement_candidates")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "settlement_candidates.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "settlement_candidates"
FINAL_DIR = PROJECT_ROOT / "data" / "final"
PLOTS_DIR = PROJECT_ROOT / "plots" / "settlement_candidates"
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


def _download_stream(url: str, out_path: Path, overwrite: bool = False) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return False

    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    resp = requests.get(url, stream=True, timeout=1200)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))
    with (
        open(tmp_path, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name, leave=False) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            f.write(chunk)
            pbar.update(len(chunk))
    tmp_path.replace(out_path)
    return True


def _download_figshare_article(dataset_key: str, info: dict, overwrite: bool = False) -> int:
    article_id = int(info["article_id"])
    regex = re.compile(info["filename_regex"])
    out_dir = RAW_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    url = f"https://api.figshare.com/v2/articles/{article_id}/files?page_size=1000"
    files = requests.get(url, timeout=120).json()
    n = 0
    for f in files:
        name = f.get("name", "")
        if not regex.match(name):
            continue
        out_path = out_dir / name
        if _download_stream(f["download_url"], out_path, overwrite=overwrite):
            n += 1
    logger.info("Downloaded %s: %d files", dataset_key, n)
    return n


def _download_zenodo_record_archive(dataset_key: str, info: dict, overwrite: bool = False) -> int:
    record_id = int(info["record_id"])
    out_dir = RAW_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    archive_path = out_dir / info.get("archive_filename", f"{record_id}.zip")
    archive_url = f"https://zenodo.org/api/records/{record_id}/files-archive"
    _download_stream(archive_url, archive_path, overwrite=overwrite)
    return 1


def _download_ghsl_epoch_zips(dataset_key: str, info: dict, overwrite: bool = False) -> int:
    years = info["epoch_years"]
    tpl = info["url_template"]
    out_dir = RAW_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for y in years:
        url = tpl.format(year=y)
        out_path = out_dir / f"{dataset_key}_{y}.zip"
        if _download_stream(url, out_path, overwrite=overwrite):
            n += 1
    logger.info("Downloaded %s: %d epoch archives", dataset_key, n)
    return n


def _download_ghsl_single_zip(dataset_key: str, info: dict, overwrite: bool = False) -> int:
    out_dir = RAW_DIR / dataset_key
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / info["filename"]
    _download_stream(info["source_url"], out_path, overwrite=overwrite)
    return 1


def _download_wsf_tiles(dataset_key: str, info: dict, overwrite: bool = False) -> int:
    root_url = info["root_url"]
    prefix = info.get("tile_filename_prefix", "WSFevolution_v1_")
    out_dir = RAW_DIR / dataset_key / "tiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    html = requests.get(root_url, timeout=120).text
    folders = sorted(set(re.findall(r'href="(' + re.escape(prefix) + r'[^/"]+/)"', html)))
    n = 0
    for folder in tqdm(folders, desc=f"Downloading {dataset_key} tiles"):
        folder_name = folder.strip("/")
        tif_name = f"{folder_name}.tif"
        url = f"{root_url}{folder}{tif_name}"
        out_path = out_dir / tif_name
        try:
            if _download_stream(url, out_path, overwrite=overwrite):
                n += 1
        except Exception as e:
            logger.warning("Tile download failed (%s): %s", tif_name, e)
    logger.info("Downloaded %s: %d tiles", dataset_key, n)
    return n


def download_datasets(datasets: dict, overwrite: bool = False) -> int:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    total = 0
    for key, info in datasets.items():
        st = info.get("source_type")
        if st == "figshare_article":
            total += _download_figshare_article(key, info, overwrite=overwrite)
        elif st == "zenodo_record_archive":
            total += _download_zenodo_record_archive(key, info, overwrite=overwrite)
        elif st == "ghsl_epoch_zips":
            total += _download_ghsl_epoch_zips(key, info, overwrite=overwrite)
        elif st == "ghsl_single_zip":
            total += _download_ghsl_single_zip(key, info, overwrite=overwrite)
        elif st == "wsf_tiles":
            total += _download_wsf_tiles(key, info, overwrite=overwrite)
        else:
            logger.warning("Unsupported source_type '%s' for %s", st, key)
    return total


def _save_regridded_da(da: xr.DataArray, info: dict, year: int, out_dir: Path, method: str = "linear") -> None:
    var_name = info["var_name"]
    da.name = var_name
    da.attrs["units"] = info["units"]
    da.attrs["long_name"] = info["long_name"]
    for key in ["_FillValue", "missing_value", "nodatavals", "nodata"]:
        da.attrs.pop(key, None)

    scale_factor = info.get("scale_factor")
    if scale_factor is not None:
        da = da * float(scale_factor)

    clip_min = info.get("clip_min")
    clip_max = info.get("clip_max")
    if clip_min is not None or clip_max is not None:
        da = da.clip(min=clip_min, max=clip_max)

    ds_out = regrid_raster(da, year=year, var_name=var_name, method=method)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{year}.nc"
    ds_out.to_netcdf(
        out_path,
        encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32", "_FillValue": None}},
    )
    ds_out.close()


def _process_annual_tiffs(dataset_key: str, info: dict, overwrite: bool = False) -> int:
    src_dir = RAW_DIR / dataset_key
    out_dir = FINAL_DIR / info["domain"] / dataset_key
    work_dir = src_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    year_pattern = re.compile(info.get("year_pattern", r"(19|20)\d{2}"))
    method = info.get("regrid_method", "nearest")
    downscale_method = info.get("downscale_method", "average")
    n = 0
    warp_dst_nodata = info.get("warp_dst_nodata", -9999.0)
    fill_nodata_with_zero = bool(info.get("fill_nodata_with_zero", False))
    for tif_path in sorted(src_dir.glob("*.tif")):
        m = year_pattern.search(tif_path.name)
        if not m:
            continue
        year = int(m.group(0))
        out_path = out_dir / f"{year}.nc"
        if out_path.exists() and not overwrite:
            continue
        warped = work_dir / f"{dataset_key}_{year}_025.tif"
        if overwrite or not warped.exists():
            _warp_to_global_025(
                src_path=tif_path,
                dst_path=warped,
                method=downscale_method,
                dst_nodata=warp_dst_nodata,
            )
        da = rioxarray.open_rasterio(warped, masked=True).squeeze(drop=True).rename({"x": "lon", "y": "lat"})
        nodata = info.get("nodata_value")
        if nodata is not None:
            da = da.where(da != nodata)
        if fill_nodata_with_zero:
            da = da.fillna(0.0)
        _save_regridded_da(da, info, year=year, out_dir=out_dir, method=method)
        da.close()
        n += 1
        gc.collect()
    logger.info("Processed %s (%d years)", dataset_key, n)
    return n


def _build_tiles_vrt(
    tile_paths: list[Path],
    vrt_path: Path,
    src_nodata: float | None = None,
    vrt_nodata: float | None = None,
) -> None:
    vrt_path.parent.mkdir(parents=True, exist_ok=True)
    list_path = vrt_path.with_suffix(".txt")
    with open(list_path, "w") as f:
        for p in tile_paths:
            f.write(str(p.resolve()) + "\n")
    cmd = ["gdalbuildvrt", "-overwrite"]
    if src_nodata is not None:
        cmd.extend(["-srcnodata", str(src_nodata)])
    if vrt_nodata is not None:
        cmd.extend(["-vrtnodata", str(vrt_nodata)])
    cmd.extend(["-input_file_list", str(list_path), str(vrt_path)])
    subprocess.run(cmd, check=True)


def _warp_to_global_025(
    src_path: Path,
    dst_path: Path,
    method: str,
    dst_nodata: float = 0.0,
    src_nodata: float | None = None,
) -> None:
    cmd = [
        "gdalwarp",
        "-q",
        "-overwrite",
        "-t_srs", "EPSG:4326",
        "-te", "-180", "-90", "180", "90",
        "-tr", "0.25", "0.25",
        "-tap",
        "-r", method,
    ]
    if src_nodata is not None:
        cmd.extend(["-srcnodata", str(src_nodata)])
    cmd.extend([
        "-dstnodata", str(dst_nodata),
        "-ot", "Float32",
        str(src_path),
        str(dst_path),
    ])
    subprocess.run(cmd, check=True)


def _process_firstyear_tiles(dataset_key: str, info: dict, overwrite: bool = False) -> int:
    src_dir = RAW_DIR / dataset_key
    if (src_dir / "tiles").exists():
        tile_dir = src_dir / "tiles"
    else:
        tile_dir = src_dir / "tiles_extracted"
        if not tile_dir.exists():
            tile_dir.mkdir(parents=True, exist_ok=True)
            archive_path = src_dir / info.get("archive_filename", "")
            if archive_path.exists():
                with zipfile.ZipFile(archive_path) as zf:
                    for m in zf.namelist():
                        if re.match(info["filename_regex"], Path(m).name):
                            zf.extract(m, tile_dir)
                # Normalize into flat tif list
                for p in tile_dir.rglob("*.tif"):
                    flat = tile_dir / p.name
                    if p != flat and not flat.exists():
                        p.rename(flat)

    tiles = sorted(tile_dir.glob("*.tif"))
    if not tiles:
        raise FileNotFoundError(f"No tiles found for {dataset_key} in {tile_dir}")

    work_dir = src_dir / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    vrt_path = work_dir / f"{dataset_key}.vrt"
    coarse_path = work_dir / f"{dataset_key}_firstyear_025.tif"
    src_nodata = info.get("src_nodata")
    vrt_nodata = info.get("vrt_nodata", src_nodata)
    rebuild_intermediate = bool(info.get("rebuild_intermediate", False))
    if rebuild_intermediate or not vrt_path.exists():
        _build_tiles_vrt(tiles, vrt_path, src_nodata=src_nodata, vrt_nodata=vrt_nodata)
    if rebuild_intermediate or not coarse_path.exists():
        _warp_to_global_025(
            vrt_path,
            coarse_path,
            method=info.get("downscale_method", "min"),
            dst_nodata=0.0,
        )

    da = rioxarray.open_rasterio(coarse_path, masked=False).squeeze(drop=True).rename({"x": "lon", "y": "lat"})
    out_dir = FINAL_DIR / info["domain"] / dataset_key
    start_year = int(info["start_year"])
    end_year = int(info["end_year"])
    regrid_method = info.get("regrid_method", "nearest")
    base_year = int(info.get("base_year", start_year))
    output_mode = info.get("output_mode", "presence")
    n = 0

    if output_mode == "fraction_linear":
        coarse_max_path = work_dir / f"{dataset_key}_firstyear_max_025.tif"
        if rebuild_intermediate or not coarse_max_path.exists():
            _warp_to_global_025(
                vrt_path,
                coarse_max_path,
                method="max",
                dst_nodata=0.0,
            )
        da_max = (
            rioxarray.open_rasterio(coarse_max_path, masked=False)
            .squeeze(drop=True)
            .rename({"x": "lon", "y": "lat"})
        )
        valid = (da > 0) & (da_max > 0) & (da_max >= da)
        span = xr.where(valid, da_max - da + 1.0, 1.0).astype(np.float32)

        for year in range(start_year, end_year + 1):
            out_path = out_dir / f"{year}.nc"
            if out_path.exists() and not overwrite:
                continue
            if info["format"] == "firstyear_tiles_offset":
                code = year - (base_year - 1)
            else:
                code = year
            frac = xr.where(valid, (float(code) - da + 1.0) / span, 0.0).clip(min=0.0, max=1.0)
            _save_regridded_da(frac.astype(np.float32), info, year=year, out_dir=out_dir, method=regrid_method)
            n += 1
            gc.collect()

        da_max.close()
        da.close()
        logger.info("Processed %s (%d yearly files)", dataset_key, n)
        return n

    for year in range(start_year, end_year + 1):
        out_path = out_dir / f"{year}.nc"
        if out_path.exists() and not overwrite:
            continue
        if info["format"] == "firstyear_tiles_offset":
            code = year - (base_year - 1)
            mask = xr.where((da > 0) & (da <= code), 1.0, 0.0)
        else:
            mask = xr.where((da > 0) & (da <= year), 1.0, 0.0)
        _save_regridded_da(mask, info, year=year, out_dir=out_dir, method=regrid_method)
        n += 1
        gc.collect()

    da.close()
    logger.info("Processed %s (%d yearly files)", dataset_key, n)
    return n


def _find_zip_member(zip_path: Path, member_regex: str) -> str:
    pat = re.compile(member_regex)
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.namelist():
            if pat.search(Path(m).name):
                return m
    raise FileNotFoundError(f"No member matching {member_regex} in {zip_path.name}")


def _ghsl_warp_member_to_025(zip_path: Path, member_name: str, out_tif: Path, method: str = "average") -> None:
    out_tif.parent.mkdir(parents=True, exist_ok=True)
    src = f"/vsizip/{zip_path.resolve()}/{member_name}"
    cmd = [
        "gdalwarp",
        "-q",
        "-overwrite",
        "-t_srs", "EPSG:4326",
        "-te", "-180", "-90", "180", "90",
        "-tr", "0.25", "0.25",
        "-tap",
        "-r", method,
        "-dstnodata", "nan",
        "-ot", "Float32",
        src,
        str(out_tif),
    ]
    subprocess.run(cmd, check=True)


def _process_ghsl_epoch_interp(dataset_key: str, info: dict, overwrite: bool = False) -> int:
    raw_dir = RAW_DIR / dataset_key
    out_dir = FINAL_DIR / info["domain"] / dataset_key
    years = sorted(int(y) for y in info["epoch_years"])
    member_regex = info["zip_member_regex"]
    method = info.get("downscale_method", "average")

    anchor = {}
    for y in years:
        zip_path = raw_dir / f"{dataset_key}_{y}.zip"
        if not zip_path.exists():
            logger.warning("Missing epoch archive: %s", zip_path)
            continue
        member = _find_zip_member(zip_path, member_regex)
        warped = raw_dir / "work" / f"{dataset_key}_{y}_025.tif"
        if overwrite or not warped.exists():
            _ghsl_warp_member_to_025(zip_path, member, warped, method=method)
        da = rioxarray.open_rasterio(warped, masked=True).squeeze(drop=True).rename({"x": "lon", "y": "lat"})
        _save_regridded_da(da, info, year=y, out_dir=out_dir, method="nearest")
        # Reopen saved anchor to ensure identical template coordinates
        ds_y = xr.open_dataset(out_dir / f"{y}.nc")
        anchor[y] = ds_y[info["var_name"]].values.astype(np.float32)
        lat = ds_y["lat"].values
        lon = ds_y["lon"].values
        ds_y.close()
        da.close()
        gc.collect()

    if not anchor:
        return 0

    y_min, y_max = years[0], years[-1]
    full_years = list(range(y_min, y_max + 1))
    var_name = info["var_name"]
    written = 0

    for y in full_years:
        out_path = out_dir / f"{y}.nc"
        if out_path.exists() and not overwrite:
            continue
        if y in anchor:
            continue
        y0 = max(k for k in years if k <= y)
        y1 = min(k for k in years if k >= y)
        if y0 == y1:
            grid = anchor[y0]
        else:
            t = (y - y0) / (y1 - y0)
            grid = (1.0 - t) * anchor[y0] + t * anchor[y1]
        da_i = xr.DataArray(grid.astype(np.float32), coords={"lat": lat, "lon": lon}, dims=("lat", "lon"))
        da_i.name = var_name
        da_i.attrs["units"] = info["units"]
        da_i.attrs["long_name"] = info["long_name"]
        ds_out = xr.Dataset(
            {var_name: da_i},
            coords={"lat": lat, "lon": lon, "time": [np.datetime64(f"{y}-01-01")]},
            attrs={"Conventions": "CF-1.8", "title": "WorldTensor Settlement Candidate Dataset"},
        )
        ds_out.to_netcdf(
            out_path,
            encoding={var_name: {"zlib": True, "complevel": 4, "dtype": "float32"}},
        )
        ds_out.close()
        written += 1

    logger.info("Processed %s (%d years incl. interpolated)", dataset_key, len(list(out_dir.glob('*.nc'))))
    return len(list(out_dir.glob("*.nc")))


def _process_ghsl_snapshot(dataset_key: str, info: dict, overwrite: bool = False) -> int:
    raw_dir = RAW_DIR / dataset_key
    out_dir = FINAL_DIR / info["domain"] / dataset_key
    zip_path = raw_dir / info["filename"]
    if not zip_path.exists():
        logger.warning("Missing snapshot archive: %s", zip_path)
        return 0
    year = int(info["snapshot_year"])
    out_path = out_dir / f"{year}.nc"
    if out_path.exists() and not overwrite:
        return 0
    member = _find_zip_member(zip_path, info["zip_member_regex"])
    warped = raw_dir / "work" / f"{dataset_key}_{year}_025.tif"
    if overwrite or not warped.exists():
        _ghsl_warp_member_to_025(zip_path, member, warped, method=info.get("downscale_method", "average"))
    da = rioxarray.open_rasterio(warped, masked=True).squeeze(drop=True).rename({"x": "lon", "y": "lat"})
    _save_regridded_da(da, info, year=year, out_dir=out_dir, method="nearest")
    da.close()
    logger.info("Processed %s snapshot (%d)", dataset_key, year)
    return 1


def process_datasets(datasets: dict, overwrite: bool = False) -> int:
    total = 0
    for key, info in datasets.items():
        fmt = info["format"]
        try:
            if fmt == "annual_tiffs":
                total += _process_annual_tiffs(key, info, overwrite=overwrite)
            elif fmt in {"firstyear_tiles_actualyear", "firstyear_tiles_offset"}:
                total += _process_firstyear_tiles(key, info, overwrite=overwrite)
            elif fmt == "ghsl_epoch_interp":
                total += _process_ghsl_epoch_interp(key, info, overwrite=overwrite)
            elif fmt == "ghsl_snapshot":
                total += _process_ghsl_snapshot(key, info, overwrite=overwrite)
            else:
                logger.warning("Unsupported format '%s' for %s", fmt, key)
        except Exception as e:
            logger.error("Processing failed for %s: %s", key, e)
    return total


def _plot_timeseries(dataset_key: str, info: dict, data_dir: Path, output_path: Path) -> None:
    var_name = info["var_name"]
    nc_files = sorted(data_dir.glob("*.nc"))
    if not nc_files:
        return
    years = []
    means = []
    for p in nc_files:
        year = int(p.stem)
        ds = xr.open_dataset(p)
        years.append(year)
        means.append(float(ds[var_name].mean(skipna=True)))
        ds.close()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.plot(years, means, marker="o", markersize=2.3, linewidth=1.2)
    ax.set_title(dataset_key)
    ax.set_xlabel("Year")
    ax.set_ylabel("Spatial mean")
    ax.grid(alpha=0.3)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_datasets(datasets: dict, map_every: int = 10) -> None:
    for key, info in datasets.items():
        data_dir = FINAL_DIR / info["domain"] / key
        if not data_dir.exists():
            continue
        nc_files = sorted(data_dir.glob("*.nc"))
        if not nc_files:
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
                    var_name=info["var_name"],
                    year=year,
                    output_path=map_dir / f"{year}.png",
                    cmap=info.get("cmap", "viridis"),
                    log_scale=bool(info.get("log_scale", False)),
                )
                ds.close()

        _plot_timeseries(key, info, data_dir, PLOTS_DIR / "timeseries" / f"{key}.png")


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
    """Run settlement candidate data pipeline."""
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
        n = download_datasets(selected, overwrite=overwrite)
        logger.info("Downloaded files total: %d", n)
    if do_process:
        n = process_datasets(selected, overwrite=overwrite)
        logger.info("Processed files total: %d", n)
    if do_plot:
        plot_datasets(selected, map_every=map_every)
        logger.info("Plots written under: %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
