"""Minimal torch-compatible loaders for the WorldTensor canonical layout.

These examples read directly from ``data/final`` using the registry in
``src.data_layout``. They intentionally avoid any merged mega-cube.
"""

from __future__ import annotations

from collections import OrderedDict
from pathlib import Path
import sys
from typing import Any

import numpy as np
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_layout import DEFAULT_FINAL_ROOT, candidate_paths_for, get_variable_spec

try:
    import torch
    from torch.utils.data import Dataset as TorchDataset
except ImportError:
    torch = None

    class TorchDataset:
        """Fallback base class when torch is unavailable."""

        pass


def require_torch():
    """Return the torch module or raise a helpful error."""
    if torch is None:
        raise ImportError(
            "PyTorch is not installed. Install it first, for example:\n"
            "  pip install torch\n"
            "Then rerun the example."
        )
    return torch


def _resolve_as_torch(as_torch: bool | None) -> bool:
    if as_torch is None:
        return torch is not None
    if as_torch and torch is None:
        require_torch()
    return bool(as_torch)


def _resolve_path(canonical_id: str, year: int | None, base_dir: str | Path) -> Path:
    spec = get_variable_spec(canonical_id)
    base_dir = Path(base_dir)
    candidates = candidate_paths_for(canonical_id, base_dir=base_dir)

    if spec.is_static:
        for candidate in candidates:
            if candidate.exists():
                return candidate
        raise FileNotFoundError(f"Could not find static variable '{canonical_id}' under {base_dir}")

    if year is None:
        raise ValueError(f"year is required for annual variable '{canonical_id}'")

    for candidate in candidates:
        path = candidate if candidate.suffix == ".nc" else candidate / f"{int(year)}.nc"
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find annual variable '{canonical_id}' for year {year} under {base_dir}")


def _pick_data_array(ds: xr.Dataset, canonical_id: str) -> xr.DataArray:
    if canonical_id in ds.data_vars:
        da = ds[canonical_id]
    else:
        da = None
        for name, var in ds.data_vars.items():
            if "lat" in var.dims and "lon" in var.dims:
                da = ds[name]
                break
        if da is None:
            raise KeyError(f"No lat/lon variable found in {list(ds.data_vars)}")

    if "time" in da.dims:
        da = da.isel(time=0)
    return da


def _normalize(values: np.ndarray, mask: np.ndarray, mode: str | None) -> np.ndarray:
    out = values.astype(np.float32, copy=True)
    if mode is None:
        return out

    valid = out[mask]
    if valid.size == 0:
        return out

    if mode == "zscore":
        mean = float(valid.mean())
        std = float(valid.std())
        if std > 0:
            out[mask] = (out[mask] - mean) / std
        else:
            out[mask] = out[mask] - mean
        return out

    if mode == "minmax":
        lo = float(valid.min())
        hi = float(valid.max())
        if hi > lo:
            out[mask] = (out[mask] - lo) / (hi - lo)
        else:
            out[mask] = 0.0
        return out

    raise ValueError(f"Unknown normalize mode: {mode!r}")


def load_variable(
    canonical_id: str,
    *,
    year: int | None = None,
    base_dir: str | Path = DEFAULT_FINAL_ROOT,
) -> dict[str, Any]:
    """Load one WorldTensor variable from the canonical per-variable layout."""
    path = _resolve_path(canonical_id, year=year, base_dir=base_dir)
    with xr.open_dataset(path, decode_timedelta=False) as ds:
        da = _pick_data_array(ds, canonical_id)
        values = np.asarray(da.values, dtype=np.float32)
        lat = np.asarray(da["lat"].values, dtype=np.float32)
        lon = np.asarray(da["lon"].values, dtype=np.float32)

    if values.ndim != 2:
        raise ValueError(f"Expected a 2D lat/lon grid for '{canonical_id}', got shape {values.shape}")

    return {
        "values": values,
        "lat": lat,
        "lon": lon,
        "canonical_id": canonical_id,
        "year": None if year is None else int(year),
        "path": path,
    }


def load_channels(
    variables: list[str] | tuple[str, ...],
    *,
    year: int | None = None,
    base_dir: str | Path = DEFAULT_FINAL_ROOT,
    normalize: str | None = "zscore",
    fill_value: float = 0.0,
    as_torch: bool | None = None,
) -> dict[str, Any]:
    """Load multiple variables and stack them into ``[C, H, W]``."""
    if not variables:
        raise ValueError("variables must be a non-empty list of canonical ids")

    use_torch = _resolve_as_torch(as_torch)
    arrays: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    lat = None
    lon = None
    grid_shape = None

    for canonical_id in variables:
        loaded = load_variable(canonical_id, year=year, base_dir=base_dir)
        values = loaded["values"]
        mask = np.isfinite(values)
        values = _normalize(values, mask, normalize)
        values = np.where(mask, values, fill_value).astype(np.float32)

        if grid_shape is None:
            grid_shape = values.shape
            lat = loaded["lat"]
            lon = loaded["lon"]
        elif values.shape != grid_shape:
            raise ValueError(
                f"Grid mismatch for '{canonical_id}': expected {grid_shape}, got {values.shape}"
            )

        arrays.append(values)
        masks.append(mask.astype(bool))

    x = np.ascontiguousarray(np.stack(arrays, axis=0), dtype=np.float32)
    mask = np.ascontiguousarray(np.stack(masks, axis=0), dtype=bool)
    lat = np.ascontiguousarray(lat, dtype=np.float32)
    lon = np.ascontiguousarray(lon, dtype=np.float32)

    sample = {
        "x": torch.from_numpy(x) if use_torch else x,
        "mask": torch.from_numpy(mask) if use_torch else mask,
        "lat": torch.from_numpy(lat) if use_torch else lat,
        "lon": torch.from_numpy(lon) if use_torch else lon,
        "variables": tuple(variables),
        "year": None if year is None else int(year),
    }
    return sample


class WorldTensorYearDataset(TorchDataset):
    """A tiny dataset where each item is one global multi-channel grid."""

    def __init__(
        self,
        *,
        variables: list[str] | tuple[str, ...],
        years: list[int] | tuple[int, ...],
        base_dir: str | Path = DEFAULT_FINAL_ROOT,
        normalize: str | None = "zscore",
        fill_value: float = 0.0,
        as_torch: bool | None = None,
        return_mask: bool = True,
    ):
        self.variables = tuple(variables)
        self.years = tuple(int(year) for year in years)
        self.base_dir = Path(base_dir)
        self.normalize = normalize
        self.fill_value = float(fill_value)
        self.as_torch = as_torch
        self.return_mask = return_mask

    def __len__(self) -> int:
        return len(self.years)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        sample = load_channels(
            self.variables,
            year=self.years[idx],
            base_dir=self.base_dir,
            normalize=self.normalize,
            fill_value=self.fill_value,
            as_torch=self.as_torch,
        )
        if not self.return_mask:
            sample.pop("mask")
        return sample


class WorldTensorPatchDataset(TorchDataset):
    """Patch sampler built on top of the canonical per-variable files."""

    def __init__(
        self,
        *,
        variables: list[str] | tuple[str, ...],
        years: list[int] | tuple[int, ...],
        patch_size: int = 64,
        patches_per_year: int = 256,
        patch_stride: int | None = None,
        sampling: str = "random",
        output_format: str = "dense",
        min_valid_fraction: float = 0.5,
        max_tries: int = 24,
        seed: int = 42,
        base_dir: str | Path = DEFAULT_FINAL_ROOT,
        normalize: str | None = "zscore",
        fill_value: float = 0.0,
        as_torch: bool | None = None,
        return_mask: bool = True,
        cache_size: int = 2,
    ):
        self.variables = tuple(variables)
        self.years = tuple(int(year) for year in years)
        self.patch_size = int(patch_size)
        self.patches_per_year = int(patches_per_year)
        self.patch_stride = int(patch_stride) if patch_stride is not None else None
        self.sampling = sampling
        self.output_format = output_format
        self.min_valid_fraction = float(min_valid_fraction)
        self.max_tries = int(max_tries)
        self.seed = int(seed)
        self.base_dir = Path(base_dir)
        self.normalize = normalize
        self.fill_value = float(fill_value)
        self.as_torch = as_torch
        self.return_mask = return_mask
        self.cache_size = int(cache_size)
        self._cache: OrderedDict[int, dict[str, Any]] = OrderedDict()
        self._neighbor_grid: tuple[list[int], list[int]] | None = None

        if self.sampling not in {"random", "neighbor"}:
            raise ValueError("sampling must be one of {'random', 'neighbor'}")
        if self.output_format not in {"dense", "dict"}:
            raise ValueError("output_format must be one of {'dense', 'dict'}")

    def __len__(self) -> int:
        if self.sampling == "neighbor":
            rows, cols = self._get_neighbor_grid()
            return len(self.years) * len(rows) * len(cols)
        return len(self.years) * self.patches_per_year

    def _get_year_sample(self, year: int) -> dict[str, Any]:
        cached = self._cache.get(year)
        if cached is not None:
            self._cache.move_to_end(year)
            return cached

        sample = load_channels(
            self.variables,
            year=year,
            base_dir=self.base_dir,
            normalize=self.normalize,
            fill_value=self.fill_value,
            as_torch=False,
        )
        self._cache[year] = sample
        while len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)
        return sample

    def _get_neighbor_grid(self) -> tuple[list[int], list[int]]:
        if self._neighbor_grid is not None:
            return self._neighbor_grid

        first_year = self.years[0]
        sample = self._get_year_sample(first_year)
        _, height, width = sample["x"].shape
        stride = self.patch_stride or self.patch_size
        if stride <= 0:
            raise ValueError("patch_stride must be a positive integer")

        max_row = height - self.patch_size
        max_col = width - self.patch_size
        rows = list(range(0, max_row + 1, stride))
        cols = list(range(0, max_col + 1, stride))
        if rows[-1] != max_row:
            rows.append(max_row)
        if cols[-1] != max_col:
            cols.append(max_col)

        self._neighbor_grid = rows, cols
        return self._neighbor_grid

    def _choose_patch(self, mask: np.ndarray, idx: int, year: int) -> tuple[int, int, float]:
        _, height, width = mask.shape
        if self.patch_size > height or self.patch_size > width:
            raise ValueError(
                f"patch_size={self.patch_size} does not fit inside grid shape {(height, width)}"
            )

        if self.sampling == "neighbor":
            rows, cols = self._get_neighbor_grid()
            per_year = len(rows) * len(cols)
            local_idx = idx % per_year
            row = rows[local_idx // len(cols)]
            col = cols[local_idx % len(cols)]
            patch_mask = mask[:, row : row + self.patch_size, col : col + self.patch_size]
            return row, col, float(patch_mask.mean())

        rng = np.random.default_rng(self.seed + year * 100_000 + idx)
        best_row = 0
        best_col = 0
        best_valid = -1.0

        for _ in range(self.max_tries):
            row = int(rng.integers(0, height - self.patch_size + 1))
            col = int(rng.integers(0, width - self.patch_size + 1))
            patch_mask = mask[:, row : row + self.patch_size, col : col + self.patch_size]
            valid_fraction = float(patch_mask.mean())
            if valid_fraction > best_valid:
                best_row, best_col, best_valid = row, col, valid_fraction
            if valid_fraction >= self.min_valid_fraction:
                break

        return best_row, best_col, best_valid

    def _format_dense(
        self,
        x_patch: np.ndarray,
        mask_patch: np.ndarray,
        lat_patch: np.ndarray,
        lon_patch: np.ndarray,
        year: int,
        row: int,
        col: int,
        valid_fraction: float,
        *,
        use_torch: bool,
    ) -> dict[str, Any]:
        sample = {
            "x": torch.from_numpy(x_patch) if use_torch else x_patch,
            "mask": torch.from_numpy(mask_patch) if use_torch else mask_patch,
            "lat": torch.from_numpy(lat_patch) if use_torch else lat_patch,
            "lon": torch.from_numpy(lon_patch) if use_torch else lon_patch,
            "year": year,
            "row": row,
            "col": col,
            "valid_fraction": valid_fraction,
        }
        if not self.return_mask:
            sample.pop("mask")
        return sample

    def _format_sparse_dict(
        self,
        x_patch: np.ndarray,
        mask_patch: np.ndarray,
        lat_patch: np.ndarray,
        lon_patch: np.ndarray,
        year: int,
        row: int,
        col: int,
        valid_fraction: float,
        *,
        use_torch: bool,
    ) -> dict[str, Any]:
        valid_cells = np.any(mask_patch, axis=0)
        rel_rows, rel_cols = np.nonzero(valid_cells)
        coordinates = np.column_stack([lat_patch[rel_rows], lon_patch[rel_cols]]).astype(np.float32)
        values = x_patch[:, rel_rows, rel_cols].T.astype(np.float32)
        value_mask = mask_patch[:, rel_rows, rel_cols].T.astype(bool)
        grid_index = np.column_stack([rel_rows + row, rel_cols + col]).astype(np.int64)

        sample = {
            "coordinates": torch.from_numpy(coordinates) if use_torch else coordinates,
            "values": torch.from_numpy(values) if use_torch else values,
            "mask": torch.from_numpy(value_mask) if use_torch else value_mask,
            "grid_index": torch.from_numpy(grid_index) if use_torch else grid_index,
            "year": year,
            "row": row,
            "col": col,
            "valid_fraction": valid_fraction,
        }
        if not self.return_mask:
            sample.pop("mask")
        return sample

    def __getitem__(self, idx: int) -> dict[str, Any]:
        if self.sampling == "neighbor":
            rows, cols = self._get_neighbor_grid()
            per_year = len(rows) * len(cols)
            year = self.years[idx // per_year]
        else:
            year = self.years[idx // self.patches_per_year]
        year_sample = self._get_year_sample(year)
        row, col, valid_fraction = self._choose_patch(year_sample["mask"], idx, year)

        x_patch = np.ascontiguousarray(
            year_sample["x"][:, row : row + self.patch_size, col : col + self.patch_size],
            dtype=np.float32,
        )
        mask_patch = np.ascontiguousarray(
            year_sample["mask"][:, row : row + self.patch_size, col : col + self.patch_size],
            dtype=bool,
        )
        lat_patch = np.ascontiguousarray(year_sample["lat"][row : row + self.patch_size], dtype=np.float32)
        lon_patch = np.ascontiguousarray(year_sample["lon"][col : col + self.patch_size], dtype=np.float32)

        use_torch = _resolve_as_torch(self.as_torch)
        if self.output_format == "dict":
            return self._format_sparse_dict(
                x_patch,
                mask_patch,
                lat_patch,
                lon_patch,
                year,
                row,
                col,
                valid_fraction,
                use_torch=use_torch,
            )
        return self._format_dense(
            x_patch,
            mask_patch,
            lat_patch,
            lon_patch,
            year,
            row,
            col,
            valid_fraction,
            use_torch=use_torch,
        )


def sparse_dict_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate sparse patch dictionaries without forcing equal point counts."""
    first = batch[0]
    use_torch = torch is not None and isinstance(first["values"], torch.Tensor)

    collated = {
        "coordinates": [item["coordinates"] for item in batch],
        "values": [item["values"] for item in batch],
    }
    if use_torch:
        collated["year"] = torch.as_tensor([item["year"] for item in batch], dtype=torch.int64)
        collated["row"] = torch.as_tensor([item["row"] for item in batch], dtype=torch.int64)
        collated["col"] = torch.as_tensor([item["col"] for item in batch], dtype=torch.int64)
        collated["valid_fraction"] = torch.as_tensor(
            [item["valid_fraction"] for item in batch],
            dtype=torch.float32,
        )
    else:
        collated["year"] = np.asarray([item["year"] for item in batch], dtype=np.int64)
        collated["row"] = np.asarray([item["row"] for item in batch], dtype=np.int64)
        collated["col"] = np.asarray([item["col"] for item in batch], dtype=np.int64)
        collated["valid_fraction"] = np.asarray(
            [item["valid_fraction"] for item in batch],
            dtype=np.float32,
        )

    if "mask" in first:
        collated["mask"] = [item["mask"] for item in batch]
    if "grid_index" in first:
        collated["grid_index"] = [item["grid_index"] for item in batch]
    return collated


__all__ = [
    "WorldTensorPatchDataset",
    "WorldTensorYearDataset",
    "load_channels",
    "load_variable",
    "require_torch",
    "sparse_dict_collate",
]
