#!/usr/bin/env python3
"""
Harmonize NetCDF files in data/final/ to the canonical WorldTensor layout.

Fixes applied:
  1. Normalize spatial dimension names to lat/lon
  2. Regrid files that do not match the master 0.25 degree grid
  3. Add time dimension to year-named annual files
  4. Rename single data variables to their canonical registry ids
  5. Cast numeric data variables to float32
  6. Add missing CF metadata from the registry (units, long_name)
  7. Add missing global "Conventions" attribute
  8. Delete pre-1900 annual files

Usage:
    python src/harmonize_data.py                       # dry run (default)
    python src/harmonize_data.py --apply              # apply changes
    python src/harmonize_data.py --category static    # target one category
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator

from src.data_layout import canonical_id_for_path, get_variable_spec
from src.utils import enforce_periodic_edge_interp

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FINAL_DIR = PROJECT_ROOT / "data" / "final"

# Import from the canonical source
from src.grid import YEAR_START, YEAR_END, N_LAT, N_LON, LAT_MIN, LAT_MAX

REF_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
REF_LON = np.linspace(0, 359.75, N_LON)

YEAR_RE = re.compile(r"^(\d{4})\.nc$")

# Directories that are truly static (no time dimension)
STATIC_PREFIX = FINAL_DIR / "static"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FMT = "%(asctime)s [%(levelname)s] %(message)s"
logging.basicConfig(format=LOG_FMT, datefmt="%H:%M:%S", level=logging.INFO, stream=sys.stdout)
log = logging.getLogger("harmonize")


# ---------------------------------------------------------------------------
# Counters
# ---------------------------------------------------------------------------
@dataclass
class Stats:
    regridded: int = 0
    coord_names_fixed: int = 0
    time_dim_added: int = 0
    periodic_lon_fixed: int = 0
    var_renamed: int = 0
    dtype_fixed: int = 0
    cf_attrs_added: int = 0
    conventions_added: int = 0
    pre1900_deleted: int = 0
    errors: int = 0
    skipped: int = 0

    def summary(self) -> str:
        return (
            f"\n{'='*60}\n"
            f"  Harmonization Summary\n"
            f"{'='*60}\n"
            f"  Coord names fixed   : {self.coord_names_fixed:>8,}\n"
            f"  Regridded to 0.25°  : {self.regridded:>8,}\n"
            f"  Time dimension added : {self.time_dim_added:>8,}\n"
            f"  Periodic seam fixed : {self.periodic_lon_fixed:>8,}\n"
            f"  Variable renamed    : {self.var_renamed:>8,}\n"
            f"  Dtype cast to f32   : {self.dtype_fixed:>8,}\n"
            f"  CF attrs patched    : {self.cf_attrs_added:>8,}\n"
            f"  Conventions added   : {self.conventions_added:>8,}\n"
            f"  Pre-1900 deleted    : {self.pre1900_deleted:>8,}\n"
            f"  Errors              : {self.errors:>8,}\n"
            f"  Skipped             : {self.skipped:>8,}\n"
            f"{'='*60}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_static(nc_path: Path) -> bool:
    """Return True if the file lives under data/final/static/."""
    try:
        nc_path.resolve().relative_to(STATIC_PREFIX.resolve())
        return True
    except ValueError:
        return False


def _relative(p: Path) -> str:
    """Short display path relative to FINAL_DIR."""
    try:
        return str(p.relative_to(FINAL_DIR))
    except ValueError:
        return str(p)


def _year_from_filename(nc_path: Path) -> int | None:
    """Extract year from a filename like 2003.nc; return None otherwise."""
    m = YEAR_RE.match(nc_path.name)
    return int(m.group(1)) if m else None


def _dir_key(nc_path: Path) -> str | None:
    """Return the relative directory key for a NetCDF file."""
    try:
        rel = nc_path.parent.relative_to(FINAL_DIR)
        return str(rel)
    except ValueError:
        return None


def _registry_context(nc_path: Path):
    """Return (canonical_id, spec) for a file when it is in the registry."""
    try:
        rel = nc_path.resolve().relative_to(FINAL_DIR.resolve())
    except ValueError:
        return None, None

    key = rel.parent.as_posix() if _year_from_filename(nc_path) is not None else rel.as_posix()
    canonical_id = canonical_id_for_path(key)
    if canonical_id is None:
        return None, None
    return canonical_id, get_variable_spec(canonical_id)


def _expected_var_name(nc_path: Path) -> str:
    canonical_id, _ = _registry_context(nc_path)
    if canonical_id:
        return canonical_id
    if _year_from_filename(nc_path) is not None:
        return nc_path.parent.name
    return nc_path.stem


def _needs_dtype_fix(ds: xr.Dataset) -> bool:
    return any(np.issubdtype(ds[v].dtype, np.number) and ds[v].dtype != np.float32 for v in ds.data_vars)


def _needs_cf_patch(ds: xr.Dataset, spec) -> bool:
    if spec is None or len(ds.data_vars) != 1:
        return False
    vname = next(iter(ds.data_vars))
    attrs = ds[vname].attrs
    return bool(spec.units and not str(attrs.get("units", "")).strip()) or bool(
        spec.long_name and not str(attrs.get("long_name", "")).strip()
    )


def _regrid_to_master(data: np.ndarray, src_lat: np.ndarray, src_lon: np.ndarray) -> np.ndarray:
    """Regrid a 2-D array from a non-standard grid to the 0.25° master grid.

    Uses bilinear interpolation, with nearest-neighbour backfill for NaN holes.
    """
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

    # Pad periodic longitude so interpolation at 359.5/359.75 can wrap to 0/0.25.
    if len(src_lon) >= 4 and float(src_lon.max() - src_lon.min()) > 300.0:
        pad_cells = min(2, len(src_lon) // 2)
        src_lon = np.concatenate([src_lon[-pad_cells:] - 360.0, src_lon, src_lon[:pad_cells] + 360.0])
        data = np.concatenate([data[:, -pad_cells:], data, data[:, :pad_cells]], axis=1)

    tgt_lon_grid, tgt_lat_grid = np.meshgrid(REF_LON, REF_LAT)
    target_points = np.column_stack([tgt_lat_grid.ravel(), tgt_lon_grid.ravel()])

    interp = RegularGridInterpolator(
        (src_lat, src_lon), data,
        method="linear", bounds_error=False, fill_value=np.nan,
    )
    regridded = interp(target_points).reshape(N_LAT, N_LON)

    nan_mask = np.isnan(regridded)
    if nan_mask.any():
        data_filled = np.where(np.isnan(data), 0.0, data)
        interp_nn = RegularGridInterpolator(
            (src_lat, src_lon), data_filled,
            method="nearest", bounds_error=False, fill_value=np.nan,
        )
        nearest_vals = interp_nn(target_points).reshape(N_LAT, N_LON)
        src_valid = RegularGridInterpolator(
            (src_lat, src_lon), (~np.isnan(data)).astype(np.float32),
            method="nearest", bounds_error=False, fill_value=0.0,
        )(target_points).reshape(N_LAT, N_LON) > 0.5
        regridded = np.where(nan_mask & src_valid, nearest_vals, regridded)

    regridded = enforce_periodic_edge_interp(regridded, REF_LON, src_lon_support)

    return regridded


def _edge_nan_counts(mask: np.ndarray) -> tuple[int, int]:
    left = 0
    while left < mask.size and bool(mask[left]):
        left += 1
    right = 0
    while right < mask.size and bool(mask[-(right + 1)]):
        right += 1
    return left, right


def _needs_periodic_lon_fix(ds: xr.Dataset) -> bool:
    for vname in ds.data_vars:
        da = ds[vname]
        if "lon" not in da.dims:
            continue
        arr = da.transpose(*[d for d in da.dims if d != "lon"], "lon").values
        flat = arr.reshape((-1, arr.shape[-1]))
        nan_cols = np.all(np.isnan(flat), axis=0)
        left_missing, right_missing = _edge_nan_counts(nan_cols)
        if (left_missing or right_missing) and left_missing != right_missing:
            return True
    return False


def _repair_periodic_lon_gaps(ds: xr.Dataset) -> xr.Dataset:
    lon = ds["lon"].values.astype(np.float64)
    repaired = ds.copy()

    for vname in list(repaired.data_vars):
        da = repaired[vname]
        if "lon" not in da.dims:
            continue

        work = da.transpose(*[d for d in da.dims if d != "lon"], "lon")
        arr = np.array(work.values, copy=True)
        flat = arr.reshape((-1, arr.shape[-1]))
        nan_cols = np.all(np.isnan(flat), axis=0)
        left_missing, right_missing = _edge_nan_counts(nan_cols)
        n_lon = flat.shape[-1]

        if right_missing and not left_missing:
            last_valid = n_lon - right_missing - 1
            x0 = float(lon[last_valid])
            x1 = float(lon[0] + 360.0)
            left_vals = flat[:, last_valid]
            right_vals = flat[:, 0]
            for idx in range(n_lon - right_missing, n_lon):
                xt = float(lon[idx] + 360.0)
                t = (xt - x0) / max(x1 - x0, 1e-12)
                filled = (1.0 - t) * left_vals + t * right_vals
                filled = np.where(np.isnan(filled), np.where(np.isnan(left_vals), right_vals, left_vals), filled)
                flat[:, idx] = filled
        elif left_missing and not right_missing:
            first_valid = left_missing
            x0 = float(lon[-1] - 360.0)
            x1 = float(lon[first_valid])
            left_vals = flat[:, -1]
            right_vals = flat[:, first_valid]
            for idx in range(left_missing):
                xt = float(lon[idx] - 360.0)
                t = (xt - x0) / max(x1 - x0, 1e-12)
                filled = (1.0 - t) * left_vals + t * right_vals
                filled = np.where(np.isnan(filled), np.where(np.isnan(left_vals), right_vals, left_vals), filled)
                flat[:, idx] = filled
        else:
            continue

        repaired[vname] = work.copy(data=flat.reshape(arr.shape)).transpose(*da.dims)

    return repaired


def _atomic_write(ds: xr.Dataset, target: Path, encoding: dict) -> None:
    """Write dataset to a temp file, then atomically rename to target."""
    fd, tmp_path = tempfile.mkstemp(suffix=".nc", dir=target.parent)
    os.close(fd)
    try:
        ds.to_netcdf(tmp_path, encoding=encoding)
        os.replace(tmp_path, target)
    except Exception:
        # Clean up temp file on failure
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


# ---------------------------------------------------------------------------
# Per-file fix logic (runs in worker processes)
# ---------------------------------------------------------------------------

def fix_single_file(nc_path_str: str, dry_run: bool) -> dict:
    """
    Apply all applicable fixes to a single NetCDF file.
    Returns a dict of action counts for the stats aggregator.
    """
    nc_path = Path(nc_path_str)
    actions: dict[str, int] = {
        "time_dim_added": 0,
        "periodic_lon_fixed": 0,
        "coord_names_fixed": 0,
        "var_renamed": 0,
        "dtype_fixed": 0,
        "cf_attrs_added": 0,
        "conventions_added": 0,
        "pre1900_deleted": 0,
        "regridded": 0,
        "errors": 0,
        "skipped": 0,
    }

    year = _year_from_filename(nc_path)
    canonical_id, spec = _registry_context(nc_path)
    is_static = spec.is_static if spec is not None else _is_static(nc_path)
    expected_var_name = canonical_id or _expected_var_name(nc_path)

    # --- Issue #5: delete pre-1900 files (non-static only) ----------------
    if year is not None and year < YEAR_START and not is_static:
        if dry_run:
            log.info("[DRY] Would delete pre-1900 file: %s", _relative(nc_path))
        else:
            nc_path.unlink()
            log.info("Deleted pre-1900 file: %s", _relative(nc_path))
        actions["pre1900_deleted"] = 1
        return actions

    # --- Open the file and determine what needs fixing --------------------
    try:
        ds = xr.open_dataset(nc_path)
    except Exception as exc:
        log.warning("Cannot open %s: %s", _relative(nc_path), exc)
        actions["errors"] = 1
        return actions

    needs_rewrite = False
    sizes = dict(ds.sizes)

    # Issue #0: Regrid files with wrong grid dimensions
    regrid = False
    lat_dim = "lat" if "lat" in sizes else "latitude" if "latitude" in sizes else None
    lon_dim = "lon" if "lon" in sizes else "longitude" if "longitude" in sizes else None
    normalize_coord_names = bool(lat_dim and lon_dim and (lat_dim != "lat" or lon_dim != "lon"))
    if normalize_coord_names:
        needs_rewrite = True
    if lat_dim and lon_dim:
        if sizes[lat_dim] != N_LAT or sizes[lon_dim] != N_LON:
            regrid = True
            needs_rewrite = True

    fix_periodic_lon = False
    if not regrid and lat_dim == "lat" and lon_dim == "lon" and sizes.get("lat") == N_LAT and sizes.get("lon") == N_LON:
        fix_periodic_lon = _needs_periodic_lon_fix(ds)
        if fix_periodic_lon:
            needs_rewrite = True

    # Issue #1: Add time dimension
    add_time = False
    if year is not None and not is_static and "time" not in ds.dims:
        add_time = True
        needs_rewrite = True

    # Issue #2: dtype fix
    fix_dtype = _needs_dtype_fix(ds)
    if fix_dtype:
        needs_rewrite = True

    # Issue #3: CF attribute patch
    patch_cf_attrs = _needs_cf_patch(ds, spec)
    if patch_cf_attrs:
        needs_rewrite = True

    # Issue #4: Conventions attribute
    add_conventions = "Conventions" not in ds.attrs
    if add_conventions:
        needs_rewrite = True

    rename_var = None
    if len(ds.data_vars) == 1:
        current_name = next(iter(ds.data_vars))
        if current_name != expected_var_name:
            rename_var = (current_name, expected_var_name)
            needs_rewrite = True

    if not needs_rewrite:
        ds.close()
        actions["skipped"] = 1
        return actions

    # --- Apply fixes in memory --------------------------------------------

    if normalize_coord_names:
        rename_map = {}
        if lat_dim and lat_dim != "lat":
            rename_map[lat_dim] = "lat"
        if lon_dim and lon_dim != "lon":
            rename_map[lon_dim] = "lon"
        ds = ds.rename(rename_map)
        lat_dim = "lat"
        lon_dim = "lon"
        actions["coord_names_fixed"] = 1

    # Issue #0: Regrid to master grid
    if regrid:
        src_lat = ds[lat_dim].values
        src_lon = ds[lon_dim].values
        new_vars = {}
        new_coords = {
            "lat": ("lat", REF_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", REF_LON, {"units": "degrees_east", "long_name": "longitude"}),
        }
        if "time" in ds.coords:
            new_coords["time"] = ("time", ds["time"].values, dict(ds["time"].attrs))
        for vname in list(ds.data_vars):
            arr = ds[vname].values
            attrs = dict(ds[vname].attrs)
            if arr.ndim == 3:
                regridded = np.stack(
                    [_regrid_to_master(arr[i], src_lat, src_lon) for i in range(arr.shape[0])],
                    axis=0,
                ).astype(np.float32)
                new_vars[vname] = (["time", "lat", "lon"], regridded, attrs)
            else:
                regridded = _regrid_to_master(arr, src_lat, src_lon).astype(np.float32)
                new_vars[vname] = (["lat", "lon"], regridded, attrs)

        ds_new = xr.Dataset(new_vars, coords=new_coords, attrs=dict(ds.attrs))
        ds.close()
        ds = ds_new
        actions["regridded"] = 1

    if fix_periodic_lon:
        ds = _repair_periodic_lon_gaps(ds)
        actions["periodic_lon_fixed"] = 1

    if add_time:
        time_val = np.datetime64(f"{year}-01-01")
        ds = ds.expand_dims(
            dim={"time": [time_val]},
        )
        # Ensure time has long_name
        ds["time"].attrs.setdefault("long_name", "time")
        actions["time_dim_added"] = 1

    if fix_dtype:
        for vname in list(ds.data_vars):
            if np.issubdtype(ds[vname].dtype, np.number) and ds[vname].dtype != np.float32:
                ds[vname] = ds[vname].astype(np.float32)
        actions["dtype_fixed"] = 1

    if rename_var:
        ds = ds.rename({rename_var[0]: rename_var[1]})
        actions["var_renamed"] = 1

    if patch_cf_attrs and spec is not None and len(ds.data_vars) == 1:
        vname = next(iter(ds.data_vars))
        if spec.units and not str(ds[vname].attrs.get("units", "")).strip():
            ds[vname].attrs["units"] = spec.units
        if spec.long_name and not str(ds[vname].attrs.get("long_name", "")).strip():
            ds[vname].attrs["long_name"] = spec.long_name
        actions["cf_attrs_added"] = 1

    if add_conventions:
        ds.attrs["Conventions"] = "CF-1.8"
        actions["conventions_added"] = 1

    if "lat" in ds.coords:
        ds["lat"].attrs.setdefault("units", "degrees_north")
        ds["lat"].attrs.setdefault("long_name", "latitude")
    if "lon" in ds.coords:
        ds["lon"].attrs.setdefault("units", "degrees_east")
        ds["lon"].attrs.setdefault("long_name", "longitude")
    if "time" in ds.coords:
        ds["time"].attrs.setdefault("long_name", "time")

    # --- Build encoding ---------------------------------------------------
    encoding = {}
    for vname in ds.data_vars:
        enc = {
            "zlib": True,
            "complevel": 4,
        }
        if np.issubdtype(ds[vname].dtype, np.number):
            enc["dtype"] = str(ds[vname].dtype)
        encoding[vname] = enc
    # Encode time coordinate if present
    if "time" in ds.coords:
        encoding["time"] = {"units": "days since 1900-01-01", "calendar": "standard"}

    # --- Write ------------------------------------------------------------
    parts = []
    if actions["coord_names_fixed"]:
        parts.append("fix_coords")
    if actions.get("regridded"):
        parts.append("regrid")
    if actions["periodic_lon_fixed"]:
        parts.append("fix_periodic_lon")
    if actions["time_dim_added"]:
        parts.append("add_time")
    if actions["var_renamed"]:
        parts.append("rename_var")
    if actions["dtype_fixed"]:
        parts.append("fix_dtype")
    if actions["cf_attrs_added"]:
        parts.append("add_cf_attrs")
    if actions["conventions_added"]:
        parts.append("add_conventions")

    if dry_run:
        log.info("[DRY] Would rewrite %s (%s)", _relative(nc_path), ", ".join(parts))
    else:
        _atomic_write(ds, nc_path, encoding)
        log.info("Rewrote %s (%s)", _relative(nc_path), ", ".join(parts))

    ds.close()
    return actions


# ---------------------------------------------------------------------------
# Collect files
# ---------------------------------------------------------------------------

def collect_nc_files(category: str | None = None) -> list[Path]:
    """Gather all .nc files under FINAL_DIR, sorted for determinism."""
    search_dir = FINAL_DIR / category if category else FINAL_DIR
    files = sorted(search_dir.rglob("*.nc"))
    return files


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run_harmonization(*, apply: bool, workers: int = 8, category: str | None = None) -> Stats:
    dry_run = not apply

    mode_label = "DRY RUN" if dry_run else "APPLY"
    log.info("Starting harmonization [%s] in %s", mode_label, FINAL_DIR)

    if not FINAL_DIR.is_dir():
        log.error("FINAL_DIR does not exist: %s", FINAL_DIR)
        raise SystemExit(1)

    if category and not (FINAL_DIR / category).exists():
        log.error("Category does not exist under data/final: %s", category)
        raise SystemExit(1)

    stats = Stats()

    # ------------------------------------------------------------------
    # Phase 1: Collect files
    # ------------------------------------------------------------------
    all_files = collect_nc_files(category)
    log.info("Phase 1: Collected %d NetCDF files for per-file fixes", len(all_files))

    # ------------------------------------------------------------------
    # Phase 2: Parallel per-file fixes
    # ------------------------------------------------------------------
    log.info("Phase 2: Per-file fixes with %d workers", workers)
    done = 0
    total = len(all_files)

    file_paths_str = [str(f) for f in all_files]

    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {
            pool.submit(fix_single_file, fp, dry_run): fp
            for fp in file_paths_str
        }
        for future in as_completed(futures):
            done += 1
            if done % 5000 == 0 or done == total:
                log.info("Progress: %d / %d files processed (%.1f%%)", done, total, 100.0 * done / total)

            try:
                result = future.result()
            except Exception as exc:
                log.error("Worker error on %s: %s", futures[future], exc)
                stats.errors += 1
                continue

            stats.regridded += result.get("regridded", 0)
            stats.coord_names_fixed += result.get("coord_names_fixed", 0)
            stats.time_dim_added += result.get("time_dim_added", 0)
            stats.periodic_lon_fixed += result.get("periodic_lon_fixed", 0)
            stats.var_renamed += result.get("var_renamed", 0)
            stats.dtype_fixed += result.get("dtype_fixed", 0)
            stats.cf_attrs_added += result.get("cf_attrs_added", 0)
            stats.conventions_added += result.get("conventions_added", 0)
            stats.pre1900_deleted += result.get("pre1900_deleted", 0)
            stats.errors += result.get("errors", 0)
            stats.skipped += result.get("skipped", 0)

    # ------------------------------------------------------------------
    # Done
    # ------------------------------------------------------------------
    log.info(stats.summary())
    if dry_run:
        log.info("This was a DRY RUN. Re-run with --apply to make changes.")
    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Harmonize all NetCDF files in data/final/",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        default=False,
        help="Actually apply changes. Without this flag, runs in dry-run mode.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers (default: 8)",
    )
    parser.add_argument(
        "--category",
        type=str,
        default=None,
        help="Restrict harmonization to one top-level data/final category.",
    )
    args = parser.parse_args()
    run_harmonization(apply=args.apply, workers=args.workers, category=args.category)


if __name__ == "__main__":
    main()
