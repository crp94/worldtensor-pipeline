"""Process MCD64A1 monthly burned area (tiled SIN) to yearly 0.25° NetCDF outputs.

Optimized Version (V9):
1. Zero subprocess calls — all GDAL ops via Python API (osgeo.gdal).
2. /vsimem/ VRT + MEM-driver warp: no intermediate disk I/O for mosaic step.
3. Direct HDF4 subdataset reads via GDAL Python API (no gdal_translate).
4. 12 parallel workers for tile extraction.
"""

from __future__ import annotations

import gc
import re
import warnings
import os
import uuid
import shutil
from datetime import datetime
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import click
import numpy as np
import xarray as xr
import yaml
from osgeo import gdal

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.utils import get_logger
from src.year_policy import resolve_year_bounds

logger = get_logger("processing.mcd64a1_v2")

# ── GDAL configuration ──────────────────────────────────────────────────────
gdal.UseExceptions()
gdal.SetConfigOption("CPL_LOG", "/dev/null")
gdal.SetConfigOption("GDAL_CACHEMAX", "1024")
gdal.SetConfigOption("GDAL_NUM_THREADS", "ALL_CPUS")
os.environ["CPL_LOG"] = "/dev/null"

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "mcd64a1.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "mcd64a1"
DEFAULT_FINAL_DIR = PROJECT_ROOT / "data" / "final" / "vegetation"

TMP_BASE = PROJECT_ROOT / "gdal_tmp"
TMP_BASE.mkdir(parents=True, exist_ok=True)

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(0, 360, N_LON, endpoint=False)

STATS_CONFIG = {
    "burned_area": ("sum",),
}

DATE_RE = re.compile(r"\.A(?P<year>\d{4})(?P<doy>\d{3})\.")

TILE_WORKERS = 12


def load_mcd64a1_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def parse_granule_date(path: Path) -> datetime | None:
    m = DATE_RE.search(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group('year')}{m.group('doy')}", "%Y%j")
    except ValueError:
        return None


def list_raw_granules(raw_dir: Path) -> list[Path]:
    return list(raw_dir.glob("*.hdf")) + list(raw_dir.glob("*.HDF"))


# ── Tile-level helpers (GDAL Python API — zero subprocesses) ─────────────────

def _get_burn_date_sds(granule_path: Path) -> str | None:
    """Return the 'Burn Date' subdataset path via GDAL Python API."""
    try:
        ds = gdal.Open(str(granule_path))
        if not ds:
            return None
        sds_list = ds.GetSubDatasets()
        ds = None
        for sds_path, sds_desc in sds_list:
            if "Burn Date" in sds_desc:
                return sds_path
    except Exception:
        pass
    return None


def _process_single_tile(args):
    """Read HDF4 SDS via GDAL Python API, write burned-area mask TIF."""
    sds_path, tmp_dir, tile_idx = args
    area_tif = str(tmp_dir / f"area_{tile_idx}.tif")

    try:
        src_ds = gdal.Open(sds_path)
        if not src_ds:
            return None
        raw = src_ds.ReadAsArray()
        geo = src_ds.GetGeoTransform()
        proj = src_ds.GetProjection()
        ny, nx = raw.shape
        src_ds = None

        valid = (raw >= 1) & (raw <= 366)
        drv = gdal.GetDriverByName("GTiff")

        area = np.where(valid, 1.0, 0.0).astype(np.float32)
        ds_out = drv.Create(area_tif, nx, ny, 1, gdal.GDT_Float32)
        ds_out.SetGeoTransform(geo)
        ds_out.SetProjection(proj)
        ds_out.GetRasterBand(1).WriteArray(area)
        ds_out.GetRasterBand(1).SetNoDataValue(float("nan"))
        ds_out = None

        return area_tif
    except Exception as e:
        logger.warning("Tile %d failed: %s", tile_idx, e)
        return None


# ── Mosaic + reproject (GDAL Python API — /vsimem/ VRT + MEM warp) ───────────

def _warp_to_025(tile_paths: list[str]) -> np.ndarray | None:
    """BuildVRT in /vsimem/ then Warp to MEM — no disk I/O, no subprocess."""
    vrt_path = f"/vsimem/vrt_{uuid.uuid4().hex}.vrt"
    try:
        vrt_ds = gdal.BuildVRT(vrt_path, tile_paths)
        if not vrt_ds:
            return None
        vrt_ds.FlushCache()

        warped = gdal.Warp(
            "",
            vrt_ds,
            format="MEM",
            dstSRS="EPSG:4326",
            outputBounds=(-0.125, -90.125, 359.875, 90.125),
            width=N_LON,
            height=N_LAT,
            resampleAlg="average",
            dstNodata=float("nan"),
            multithread=True,
            warpMemoryLimit=1024 * 1024 * 1024,
        )

        data = warped.ReadAsArray().astype(np.float32)
        geo = warped.GetGeoTransform()
        warped = None
        vrt_ds = None

        if geo[5] < 0:
            data = data[::-1, :]
        return data
    except Exception as e:
        logger.warning("Warp failed: %s", e)
        return None
    finally:
        try:
            gdal.Unlink(vrt_path)
        except Exception:
            pass


# ── Monthly aggregation ──────────────────────────────────────────────────────

def _process_month(month_granules: list[Path], variables: list[str], config: dict) -> dict[str, np.ndarray]:
    results = {}
    tmp_dir = TMP_BASE / f"month_{uuid.uuid4().hex}"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Gather SDS paths (GDAL Python API — no subprocess)
        tasks = []
        for i, p in enumerate(month_granules):
            sds = _get_burn_date_sds(p)
            if sds:
                tasks.append((sds, tmp_dir, i))
        if not tasks:
            return {}

        # Extract tiles in parallel (GDAL Python API — no subprocess)
        area_tiles = []
        with ThreadPoolExecutor(max_workers=TILE_WORKERS) as executor:
            for a in executor.map(_process_single_tile, tasks):
                if a:
                    area_tiles.append(a)

        # BuildVRT + Warp (GDAL Python API — no subprocess)
        if "burned_area" in variables and area_tiles:
            grid = _warp_to_025(area_tiles)
            if grid is not None:
                lats = np.deg2rad(TARGET_LAT)
                cell_area = 773.23 * np.cos(lats)
                results["burned_area"] = grid * cell_area[:, np.newaxis]
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return results


# ── Yearly save ──────────────────────────────────────────────────────────────

def _save_derived(data, var_name, stat_name, year, var_info, output_dir, overwrite):
    folder_name = f"{var_name}_{stat_name}"
    out_path = output_dir / folder_name / f"{year}.nc"
    if out_path.exists() and not overwrite:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds = xr.Dataset(
        {folder_name: (["lat", "lon"], data.astype(np.float32),
                       {"units": var_info["units"], "long_name": f"{var_info['long_name']} ({stat_name})"})},
        coords={"lat": ("lat", TARGET_LAT, {"units": "degrees_north"}),
                "lon": ("lon", TARGET_LON, {"units": "degrees_east"})},
        attrs={"year": year, "aggregation_method": stat_name, "source": "MCD64A1 v6.1"},
    )
    ds.to_netcdf(out_path, encoding={folder_name: {"zlib": True, "complevel": 4}})
    return out_path


# ── Main processing loop ─────────────────────────────────────────────────────

def process_mcd64a1(variables, start_year=None, end_year=None, raw_dir=DEFAULT_RAW_DIR,
                    output_dir=DEFAULT_FINAL_DIR, overwrite=False, qa_mode="moderate"):
    config = load_mcd64a1_config()
    granules = list_raw_granules(raw_dir)
    by_month = defaultdict(list)
    for p in granules:
        dt = parse_granule_date(p)
        if dt:
            by_month[(dt.year, dt.month)].append(p)
    years = sorted(set(y for y, m in by_month.keys()))
    y0, y1 = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=config["temporal_range"][0],
        default_end=config["temporal_range"][1],
        label="MCD64A1 processing years",
    )
    files_written, years_processed = 0, 0

    for year in years:
        if not (y0 <= year <= y1):
            continue
        logger.info("Processing Year %d (V9-ZeroSubprocess)", year)
        month_grids = {v: [] for v in variables}

        for month in range(1, 13):
            month_tiles = by_month.get((year, month), [])
            if not month_tiles:
                continue
            logger.info("  Month %d: %d tiles", month, len(month_tiles))
            grids = _process_month(month_tiles, variables, config)
            for v, g in grids.items():
                month_grids[v].append(g)

        wrote_any = False
        for var_name, grids in month_grids.items():
            if not grids:
                continue
            stack = np.stack(grids)
            stats_to_compute = STATS_CONFIG.get(var_name, ("mean", "std", "max", "min"))
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                derived = {}
                if "sum" in stats_to_compute:
                    derived["sum"] = np.nansum(stack, axis=0)
                if "mean" in stats_to_compute:
                    derived["mean"] = np.nanmean(stack, axis=0)
                if "std" in stats_to_compute:
                    derived["std"] = np.nanstd(stack, axis=0)
                if "max" in stats_to_compute:
                    derived["max"] = np.nanmax(stack, axis=0)
                if "min" in stats_to_compute:
                    derived["min"] = np.nanmin(stack, axis=0)
            for stat, data in derived.items():
                data_masked = np.where(data == 0, np.nan, data)
                if _save_derived(data_masked, var_name, stat, year,
                                 config["variables"][var_name], output_dir, overwrite):
                    files_written += 1
                    wrote_any = True
        if wrote_any:
            years_processed += 1
        gc.collect()

    return {"years_written": years_processed, "files_written": files_written}


@click.command()
@click.option("--all", "run_all", is_flag=True)
@click.option("--start-year", type=int)
@click.option("--end-year", type=int)
@click.option("--overwrite", is_flag=True)
def main(run_all, start_year, end_year, overwrite):
    config = load_mcd64a1_config()
    var_list = list(config["variables"].keys())
    summary = process_mcd64a1(variables=var_list, start_year=start_year, end_year=end_year, overwrite=overwrite)
    click.echo(f"Done. years={summary['years_written']}, files={summary['files_written']}")


if __name__ == "__main__":
    main()
