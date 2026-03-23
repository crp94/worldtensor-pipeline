"""Process MOD13C2 monthly HDF granules to yearly 0.25° NetCDF outputs.

Pipeline steps per monthly granule:
1. Read NDVI/EVI SDS from HDF subdatasets.
2. Apply scale/valid-range decoding and optional QA masking.
3. Regrid 0.05° -> 0.25° using area-weighted block averaging.
4. Aggregate monthly stacks to yearly mean/std/max/min.

Usage:
    python -m src.processing.mod13c2_monthly_to_yearly --all
    python -m src.processing.mod13c2_monthly_to_yearly -v ndvi -v evi --start-year 2010 --end-year 2020
"""

from __future__ import annotations

import re
import json
import warnings
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

import click
import numpy as np
import rasterio
import xarray as xr
import yaml

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.utils import get_logger
from src.year_policy import resolve_year_bounds

logger = get_logger("processing.mod13c2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "mod13c2.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "mod13c2"
DEFAULT_FINAL_DIR = PROJECT_ROOT / "data" / "final" / "vegetation"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)
STATS = ("mean", "std", "max", "min")

DATE_RE = re.compile(r"\.A(?P<year>\d{4})(?P<doy>\d{3})\.")


def load_mod13c2_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def parse_granule_date(path: Path) -> datetime | None:
    """Parse acquisition date from MODIS filename token .AYYYYDDD."""
    m = DATE_RE.search(path.name)
    if not m:
        return None
    try:
        return datetime.strptime(f"{m.group('year')}{m.group('doy')}", "%Y%j")
    except ValueError:
        return None


def list_raw_granules(raw_dir: Path) -> list[Path]:
    """List HDF/HDF-EOS granules sorted by acquisition date."""
    files = list(raw_dir.glob("*.hdf")) + list(raw_dir.glob("*.HDF")) + list(raw_dir.glob("*.h5"))
    dated = []
    for p in files:
        dt = parse_granule_date(p)
        if dt is not None:
            dated.append((dt, p))
        else:
            logger.warning("Skipping file with unrecognized date token: %s", p.name)
    dated.sort(key=lambda t: t[0])
    return [p for _, p in dated]


def _find_subdataset(subdatasets: list[str], include: list[str], exclude: list[str] | None = None) -> str | None:
    """Find first subdataset whose name matches include/exclude token rules."""
    exclude = exclude or []
    include_l = [s.lower() for s in include]
    exclude_l = [s.lower() for s in exclude]

    for sds in subdatasets:
        low = sds.lower()
        if all(tok in low for tok in include_l) and not any(tok in low for tok in exclude_l):
            return sds
    return None


def _list_subdatasets_with_gdalinfo(granule_path: Path) -> list[str]:
    """List HDF subdataset names using system gdalinfo JSON output."""
    cmd = ["gdalinfo", "-json", str(granule_path)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"gdalinfo failed for {granule_path.name}: {proc.stderr[:300]}")

    info = json.loads(proc.stdout)
    sub_meta = info.get("metadata", {}).get("SUBDATASETS", {})

    found: list[tuple[int, str]] = []
    for key, val in sub_meta.items():
        if not key.endswith("_NAME"):
            continue
        # Key format: SUBDATASET_{idx}_NAME
        parts = key.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            found.append((int(parts[1]), val))

    found.sort(key=lambda t: t[0])
    return [name for _, name in found]


def _read_subdataset_array(sds_path: str) -> xr.DataArray:
    """Read one HDF subdataset into a 2D DataArray with lat/lon coords.

    Uses system `gdal_translate` to a temporary GeoTIFF because rasterio in
    this environment may not include HDF4 drivers.
    """
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_tif = Path(tmp.name)

    cmd = ["gdal_translate", "-q", "-of", "GTiff", sds_path, str(tmp_tif)]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        tmp_tif.unlink(missing_ok=True)
        raise RuntimeError(f"gdal_translate failed for subdataset: {proc.stderr[:300]}")

    with rasterio.open(tmp_tif) as src:
        arr = src.read(1)
        transform = src.transform
        h, w = src.height, src.width

        # Pixel-center coordinates from affine transform
        lon = transform.c + transform.a * (np.arange(w) + 0.5)
        lat = transform.f + transform.e * (np.arange(h) + 0.5)

    da = xr.DataArray(
        arr,
        dims=("lat", "lon"),
        coords={"lat": lat.astype(np.float64), "lon": lon.astype(np.float64)},
    )
    tmp_tif.unlink(missing_ok=True)
    return da


def _decode_vi_raw(raw: xr.DataArray, var_info: dict) -> xr.DataArray:
    """Decode scaled VI integers to physical NDVI/EVI values."""
    scale = float(var_info.get("scale_factor", 10000.0))
    fill = float(var_info.get("fill_value", -3000))
    vmin, vmax = var_info.get("valid_range_raw", [-2000, 10000])

    data = raw.astype(np.float32)
    mask = (raw == fill) | (raw < vmin) | (raw > vmax)

    # MOD13 convention for NDVI/EVI: physical_value = raw / 10000.
    data = data / scale
    data = data.where(~mask)
    return data


def _apply_qa_mask(vi: xr.DataArray, qa_raw: xr.DataArray | None, qa_mode: str) -> xr.DataArray:
    """Apply QA filter based on MOD13 VI Quality summary bits.

    Bit interpretation used:
    - bits 0-1 == 00: good quality
    - bits 0-1 == 01: check other QA (accepted in moderate mode)
    """
    if qa_mode == "none" or qa_raw is None:
        return vi

    qa = qa_raw.astype(np.int32)
    summary = qa & 0b11

    if qa_mode == "strict":
        keep = summary == 0
    elif qa_mode == "moderate":
        keep = summary <= 1
    else:
        raise ValueError(f"Unsupported qa_mode: {qa_mode}")

    return vi.where(keep)


def _normalize_coords(da: xr.DataArray) -> xr.DataArray:
    """Normalize coords to ascending lat and lon in [0, 360)."""
    if da.lat.values[0] > da.lat.values[-1]:
        da = da.isel(lat=slice(None, None, -1))

    if float(da.lon.min()) < 0:
        da = da.assign_coords(lon=(da.lon.values % 360.0)).sortby("lon")

    return da


def area_weighted_regrid_to_025(da: xr.DataArray) -> xr.DataArray:
    """Regrid 0.05° MODIS to 0.25° using area-weighted averaging.

    Method
    ------
    1. Perform latitude-area-weighted block averaging on native grid:
       - weight = cos(latitude)
       - coarsen factor = round(0.25 / source_resolution)
    2. Interpolate from coarsened 0.25° centers to the project target grid
       (which includes pole rows at -90 and +90).

    The area weighting is applied in step (1), where it matters physically.
    """
    da = _normalize_coords(da)

    lat_step = float(np.median(np.abs(np.diff(da.lat.values))))
    lon_step = float(np.median(np.abs(np.diff(da.lon.values))))
    factor_lat = int(round(0.25 / lat_step))
    factor_lon = int(round(0.25 / lon_step))

    if factor_lat <= 0 or factor_lon <= 0:
        raise ValueError("Invalid coarsening factors derived from source resolution")

    trim_lat = (da.sizes["lat"] // factor_lat) * factor_lat
    trim_lon = (da.sizes["lon"] // factor_lon) * factor_lon
    da = da.isel(lat=slice(0, trim_lat), lon=slice(0, trim_lon))

    lat_weights = xr.DataArray(
        np.cos(np.deg2rad(da.lat.values)).astype(np.float32),
        dims=("lat",),
        coords={"lat": da.lat.values},
    )
    weight_2d = lat_weights.broadcast_like(da)

    valid = da.notnull()
    num = (da * weight_2d).where(valid).coarsen(lat=factor_lat, lon=factor_lon, boundary="trim").sum()
    den = weight_2d.where(valid).coarsen(lat=factor_lat, lon=factor_lon, boundary="trim").sum()
    coarse = num / den

    # Align to master project grid (721x1440, lon 0..359.75).
    return coarse.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )


def _save_derived(
    data: np.ndarray,
    var_name: str,
    stat_name: str,
    year: int,
    var_info: dict,
    output_dir: Path,
    overwrite: bool = False,
) -> Path | None:
    folder_name = f"{var_name}_{stat_name}"
    out_path = output_dir / folder_name / f"{year}.nc"

    if out_path.exists() and not overwrite:
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    ds = xr.Dataset(
        {
            folder_name: (
                ["lat", "lon"],
                data.astype(np.float32),
                {
                    "units": var_info["units"],
                    "long_name": f"{var_info['long_name']} ({stat_name})",
                    "source": "MOD13C2 v6.1 (MODIS/Terra) via LP DAAC",
                },
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor MOD13C2 {folder_name}",
            "source": "MOD13C2 v6.1 (MODIS/Terra)",
            "year": year,
            "aggregation_method": stat_name,
        },
    )

    ds.to_netcdf(
        out_path,
        encoding={folder_name: {"zlib": True, "complevel": 4, "dtype": "float32"}},
    )
    return out_path


def _read_monthly_grids(
    granule_path: Path,
    var_names: list[str],
    config: dict,
    qa_mode: str,
) -> dict[str, np.ndarray]:
    """Extract and regrid selected variables from one monthly granule."""
    monthly: dict[str, np.ndarray] = {}

    subdatasets = _list_subdatasets_with_gdalinfo(granule_path)

    qa_raw = None
    if qa_mode != "none":
        qa_cfg = config["qa"]
        qa_sds = _find_subdataset(
            subdatasets,
            include=qa_cfg["sds_include"],
            exclude=qa_cfg.get("sds_exclude", []),
        )
        if qa_sds is None:
            logger.warning("QA SDS not found in %s; proceeding without QA mask", granule_path.name)
        else:
            qa_raw = _read_subdataset_array(qa_sds)

    for var_name in var_names:
        info = config["variables"][var_name]
        sds_path = _find_subdataset(
            subdatasets,
            include=info["sds_include"],
            exclude=info.get("sds_exclude", []),
        )
        if sds_path is None:
            logger.warning("SDS for '%s' not found in %s", var_name, granule_path.name)
            continue

        raw = _read_subdataset_array(sds_path)
        vi = _decode_vi_raw(raw, info)
        vi = _apply_qa_mask(vi, qa_raw, qa_mode)
        vi = vi.clip(min=-1.0, max=1.0)

        gridded = area_weighted_regrid_to_025(vi)
        monthly[var_name] = gridded.values.astype(np.float32)

    return monthly


def _flush_year(
    year: int,
    buffers: dict[str, list[np.ndarray]],
    config: dict,
    output_dir: Path,
    overwrite: bool,
) -> tuple[int, int]:
    """Aggregate monthly buffers for one year and write yearly stat files.

    Returns
    -------
    tuple[int, int]
        (years_written, files_written)
    """
    files_written = 0
    wrote_any = False

    for var_name, monthly_arrays in buffers.items():
        if not monthly_arrays:
            continue

        stack = np.stack(monthly_arrays, axis=0).astype(np.float32)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            with np.errstate(invalid="ignore"):
                derived = {
                    "mean": np.nanmean(stack, axis=0),
                    "std": np.nanstd(stack, axis=0),
                    "max": np.nanmax(stack, axis=0),
                    "min": np.nanmin(stack, axis=0),
                }

        for stat_name, data in derived.items():
            out = _save_derived(
                data=data,
                var_name=var_name,
                stat_name=stat_name,
                year=year,
                var_info=config["variables"][var_name],
                output_dir=output_dir,
                overwrite=overwrite,
            )
            if out is not None:
                files_written += 1
                wrote_any = True

    return (1 if wrote_any else 0), files_written


def process_mod13c2(
    variables: list[str],
    start_year: int | None = None,
    end_year: int | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    output_dir: Path = DEFAULT_FINAL_DIR,
    overwrite: bool = False,
    qa_mode: str = "moderate",
) -> dict:
    """Process downloaded MOD13C2 monthly granules to yearly outputs."""
    config = load_mod13c2_config()
    cfg_start, cfg_end = config["temporal_range"]
    granules = list_raw_granules(raw_dir)
    if not granules:
        logger.warning("No MOD13C2 raw granules found in %s", raw_dir)
        return {"granules": 0, "years_written": 0, "files_written": 0}

    available_years = [parse_granule_date(p).year for p in granules if parse_granule_date(p) is not None]
    if not available_years:
        logger.warning("No parseable acquisition dates in raw directory: %s", raw_dir)
        return {"granules": 0, "years_written": 0, "files_written": 0}

    y0, y1 = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=cfg_start,
        default_end=cfg_end,
        label="MOD13C2 processing years",
    )

    selected = []
    for p in granules:
        dt = parse_granule_date(p)
        if dt is None:
            continue
        if y0 <= dt.year <= y1:
            selected.append((dt, p))

    if not selected:
        logger.warning("No granules in selected range %d-%d", y0, y1)
        return {"granules": 0, "years_written": 0, "files_written": 0}

    selected.sort(key=lambda t: t[0])

    years_written = 0
    files_written = 0

    current_year = None
    buffers = {v: [] for v in variables}

    for dt, granule in selected:
        year = dt.year

        if current_year is None:
            current_year = year
        elif year != current_year:
            y_written, f_written = _flush_year(current_year, buffers, config, output_dir, overwrite)
            years_written += y_written
            files_written += f_written
            buffers = {v: [] for v in variables}
            current_year = year

        monthly = _read_monthly_grids(granule, variables, config, qa_mode=qa_mode)
        for var_name in variables:
            if var_name in monthly:
                buffers[var_name].append(monthly[var_name])

    if current_year is not None:
        y_written, f_written = _flush_year(current_year, buffers, config, output_dir, overwrite)
        years_written += y_written
        files_written += f_written

    logger.info(
        "Processed MOD13C2 granules=%d, years_written=%d, files_written=%d",
        len(selected), years_written, files_written,
    )
    return {
        "granules": len(selected),
        "years_written": years_written,
        "files_written": files_written,
        "year_range": [y0, y1],
    }


@click.command()
@click.option("--variables", "variables", "-v", multiple=True, help="Variables (ndvi, evi).")
@click.option("--all", "run_all", is_flag=True, help="Process all configured variables.")
@click.option("--start-year", type=int, default=None, help="Start year filter.")
@click.option("--end-year", type=int, default=None, help="End year filter.")
@click.option("--raw-dir", type=click.Path(), default=None, help=f"Raw directory (default: {DEFAULT_RAW_DIR})")
@click.option(
    "--output-dir",
    type=click.Path(),
    default=None,
    help=f"Output directory (default: {DEFAULT_FINAL_DIR})",
)
@click.option("--overwrite", is_flag=True, help="Overwrite existing yearly files.")
@click.option(
    "--qa-mode",
    type=click.Choice(["strict", "moderate", "none"], case_sensitive=False),
    default="moderate",
    show_default=True,
    help="Quality mask mode.",
)
def main(variables, run_all, start_year, end_year, raw_dir, output_dir, overwrite, qa_mode):
    """Process MOD13C2 monthly granules to yearly 0.25° outputs."""
    config = load_mod13c2_config()

    if variables:
        var_list = [v for v in variables if v in config["variables"]]
        missing = [v for v in variables if v not in config["variables"]]
        if missing:
            logger.warning("Unknown variables skipped: %s", missing)
    elif run_all:
        var_list = list(config["variables"].keys())
    else:
        click.echo("Specify --variables or --all.")
        return

    in_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    out_dir = Path(output_dir) if output_dir else DEFAULT_FINAL_DIR

    summary = process_mod13c2(
        variables=var_list,
        start_year=start_year,
        end_year=end_year,
        raw_dir=in_dir,
        output_dir=out_dir,
        overwrite=overwrite,
        qa_mode=qa_mode.lower(),
    )
    click.echo(
        "Done. "
        f"granules={summary.get('granules', 0)}, "
        f"years_written={summary.get('years_written', 0)}, "
        f"files_written={summary.get('files_written', 0)}"
    )


if __name__ == "__main__":
    main()
