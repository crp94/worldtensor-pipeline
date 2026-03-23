"""Shared utilities: logging, I/O helpers, CF metadata enforcement."""

import logging
from pathlib import Path

import numpy as np
import xarray as xr
import yaml

from src.data_layout import get_variable_spec, output_path_for


CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "variables.yml"


def add_cyclic_point_xr(da: xr.DataArray, dim: str = "lon"):
    """Robustly add a cyclic point to a DataArray for Cartopy plotting.
    
    Returns (data_cyclic, lon_cyclic, lat_values).
    Automatically shifts longitude to -180..180 to move the seam to the edges.
    """
    from cartopy.util import add_cyclic_point
    
    # Squeeze out extra dims (like time) if size 1
    da = da.squeeze()

    # Standardize to -180..180 for seamless plotting
    # If lon is 0..360, we roll the data so it goes from -180..180
    if float(da[dim].max()) > 180:
        # Shift 0..360 -> -180..180
        da = da.assign_coords({dim: (((da[dim] + 180) % 360) - 180)})
        # Sort so coordinates are strictly increasing
        da = da.sortby(dim)
        
    # Ensure cyclic dim is last for add_cyclic_point
    if da.dims[-1] != dim:
        da = da.transpose(..., dim)
        
    # add_cyclic_point expects a numpy array and a 1D coordinate
    data_cyclic, lon_cyclic = add_cyclic_point(da.values, coord=da[dim].values)
    return data_cyclic, lon_cyclic, da.lat.values


def enforce_periodic_edge_interp(data: np.ndarray, target_lon: np.ndarray, src_lon: np.ndarray) -> np.ndarray:
    """Repair target columns that lie outside the native periodic longitude support."""
    arr = np.array(data, copy=True)
    tgt = np.asarray(target_lon, dtype=np.float64)
    src = np.asarray(src_lon, dtype=np.float64)

    if arr.ndim != 2 or tgt.ndim != 1 or src.ndim != 1 or tgt.size < 2 or src.size < 2:
        return arr

    if np.nanmin(src) < 0:
        src = np.mod(src, 360.0)
    src = np.sort(src[np.isfinite(src)])
    if src.size < 2 or float(src.max() - src.min()) <= 300.0:
        return arr

    right_edge = np.where(tgt > src.max() + 1e-9)[0]
    if right_edge.size:
        left_idx = int(np.searchsorted(tgt, src.max() + 1e-9, side="right") - 1)
        if 0 <= left_idx < arr.shape[1]:
            left_vals = arr[:, left_idx]
            right_vals = arr[:, 0]
            x0 = float(tgt[left_idx])
            x1 = float(tgt[0] + 360.0)
            denom = max(x1 - x0, 1e-12)
            for idx in right_edge:
                xt = float(tgt[idx])
                t = (xt - x0) / denom
                filled = (1.0 - t) * left_vals + t * right_vals
                fallback = np.where(np.isfinite(left_vals), left_vals, right_vals)
                arr[:, idx] = np.where(np.isfinite(filled), filled, fallback)

    left_edge = np.where(tgt < src.min() - 1e-9)[0]
    if left_edge.size:
        right_idx = int(np.searchsorted(tgt, src.min() - 1e-9, side="left"))
        if 0 <= right_idx < arr.shape[1]:
            left_vals = arr[:, -1]
            right_vals = arr[:, right_idx]
            x0 = float(tgt[-1] - 360.0)
            x1 = float(tgt[right_idx])
            denom = max(x1 - x0, 1e-12)
            for idx in left_edge:
                xt = float(tgt[idx] - 360.0)
                t = (xt - x0) / denom
                filled = (1.0 - t) * left_vals + t * right_vals
                fallback = np.where(np.isfinite(right_vals), right_vals, left_vals)
                arr[:, idx] = np.where(np.isfinite(filled), filled, fallback)

    return arr


def calculate_cell_area_ha(lat_grid, lon_grid, res=0.25):
    """Calculate the area of each cell in a regular lat/lon grid in hectares."""
    R = 6371000.0  # Earth radius in meters
    d_lat = np.radians(res)
    d_lon = np.radians(res)
    
    # Grid of latitudes in radians
    lat_rad = np.radians(lat_grid)
    
    # Area = (R^2) * cos(lat) * d_lat * d_lon
    cos_lat = np.cos(lat_rad)
    
    # If lat_grid is 1D, broadcast to 2D
    if lat_grid.ndim == 1:
        area_m2 = (R**2) * cos_lat[:, np.newaxis] * d_lat * d_lon
    else:
        area_m2 = (R**2) * cos_lat * d_lat * d_lon
        
    return area_m2 / 10000.0  # Convert m2 to hectares


def get_logger(name: str) -> logging.Logger:
    """Return a project-wide logger with consistent formatting."""
    logger = logging.getLogger(f"worldtensor.{name}")
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s", datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def plot_global_map(da: xr.DataArray, title: str, out_path: Path, cmap: str = "viridis", force_log: bool = False):
    """Standardized global map plotting for WorldTensor.
    
    Uses Robinson projection and adds cyclic points to avoid the meridian seam.
    """
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from matplotlib.colors import LogNorm, Normalize

    fig = plt.subplots(1, 1, figsize=(12, 7), subplot_kw={"projection": ccrs.Robinson()})
    fig, ax = fig
    
    data_cyclic, lon_cyclic, lat_values = add_cyclic_point_xr(da)
    
    # Robust scaling: Use valid data only
    data_valid = data_cyclic[np.isfinite(data_cyclic)]
    if data_valid.size > 0:
        vmin = float(np.nanpercentile(data_valid, 2))
        vmax = float(np.nanpercentile(data_valid, 98))
        if vmin == vmax:
            vmin, vmax = float(np.min(data_valid)), float(np.max(data_valid))
    else:
        vmin, vmax = 0, 1

    # Use LogNorm for highly skewed data or if forced
    if force_log or (vmin > 0 and (vmax / vmin) > 1000):
        norm = LogNorm(vmin=max(vmin, 1e-6), vmax=vmax)
    else:
        norm = Normalize(vmin=vmin, vmax=vmax)

    im = ax.pcolormesh(
        lon_cyclic, lat_values, data_cyclic,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        shading='auto'
    )
    
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, aspect=40, shrink=0.8,
                 label=da.attrs.get("units", ""))
    
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.set_title(title, fontsize=12, pad=10)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_time_series(years: list[int], values: list[float], title: str, ylabel: str, out_path: Path, color: str = "#1f77b4"):
    """Standardized time series plotting for WorldTensor."""
    import matplotlib.pyplot as plt
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    ax.plot(years, values, marker="o", markersize=4, linewidth=1.5, color=color)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("Year")
    ax.grid(True, alpha=0.3)
    
    # Ensure x-axis shows years correctly
    if len(years) > 1:
        ax.set_xticks(years[::max(1, len(years)//10)])
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()


def load_config(path: Path | None = None) -> dict:
    """Load the variable registry from variables.yml.

    Parameters
    ----------
    path : Path, optional
        Path to the YAML config. Defaults to config/variables.yml.

    Returns
    -------
    dict
        Parsed YAML contents.
    """
    path = path or CONFIG_PATH
    with open(path) as f:
        return yaml.safe_load(f) or {}


def save_to_netcdf(
    ds: xr.Dataset,
    var_name: str,
    year: int,
    output_dir: str | Path = "data/final",
) -> Path:
    """Save a dataset to a CF-compliant NetCDF file at {output_dir}/{var_name}/{year}.nc.

    Enforces that the variable has 'units' and 'long_name' attributes and applies
    consistent encoding (zlib compression, float32).

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the variable to save.
    var_name : str
        Name of the data variable in the dataset.
    year : int
        Year label for the output filename.
    output_dir : str or Path
        Base output directory. Defaults to data/final.

    Returns
    -------
    Path
        Path to the written file.
    """
    logger = get_logger("io")

    ds_out = _annual_dataset_from_variable(ds, canonical_id=var_name, year=year)
    var = ds_out[var_name]
    if "units" not in var.attrs:
        raise ValueError(f"Variable '{var_name}' is missing required 'units' attribute.")
    if "long_name" not in var.attrs:
        raise ValueError(f"Variable '{var_name}' is missing required 'long_name' attribute.")

    out_path = Path(output_dir) / var_name / f"{year}.nc"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    encoding = _encoding_for(var_name)
    encoding["time"] = {"units": "days since 1900-01-01", "calendar": "standard"}

    ds_out.to_netcdf(out_path, encoding=encoding)
    logger.info("Saved %s → %s", var_name, out_path)
    return out_path


def _coerce_to_dataset(data: xr.Dataset | xr.DataArray) -> xr.Dataset:
    if isinstance(data, xr.DataArray):
        name = data.name or "data"
        return data.to_dataset(name=name)
    return data.copy()


def _resolve_source_var(ds: xr.Dataset, canonical_id: str, source_var: str | None) -> str:
    if source_var:
        if source_var not in ds.data_vars:
            raise ValueError(f"Variable '{source_var}' not found in dataset. Available: {list(ds.data_vars)}")
        return source_var
    if canonical_id in ds.data_vars:
        return canonical_id
    data_vars = list(ds.data_vars)
    if len(data_vars) == 1:
        return data_vars[0]
    raise ValueError(
        f"Could not infer source variable for canonical id '{canonical_id}'. "
        f"Available variables: {data_vars}"
    )


def _encoding_for(var_name: str) -> dict[str, dict[str, object]]:
    return {
        var_name: {
            "zlib": True,
            "complevel": 4,
            "dtype": "float32",
        }
    }


def _standardize_metadata(da: xr.DataArray, canonical_id: str) -> xr.DataArray:
    try:
        spec = get_variable_spec(canonical_id)
    except KeyError:
        spec = None

    if np.issubdtype(da.dtype, np.number):
        da = da.astype(np.float32)

    if spec is not None:
        da.attrs.setdefault("units", spec.units)
        da.attrs.setdefault("long_name", spec.long_name)
    return da


def _annual_dataset_from_variable(
    data: xr.Dataset | xr.DataArray,
    canonical_id: str,
    year: int,
    source_var: str | None = None,
) -> xr.Dataset:
    ds = _coerce_to_dataset(data)
    src_name = _resolve_source_var(ds, canonical_id, source_var)
    da = _standardize_metadata(ds[src_name], canonical_id)

    if "time" in da.dims:
        if da.sizes["time"] != 1:
            raise ValueError(f"Annual variable '{canonical_id}' must have a singleton time dimension.")
        da = da.isel(time=0, drop=True)

    da = da.rename(canonical_id).expand_dims(time=[np.datetime64(f"{int(year)}-01-01")])

    out = da.to_dataset(name=canonical_id)
    out["time"].attrs.setdefault("long_name", "time")
    out.attrs.update(ds.attrs)
    out.attrs.setdefault("Conventions", "CF-1.8")
    return out


def _static_dataset_from_variable(
    data: xr.Dataset | xr.DataArray,
    canonical_id: str,
    source_var: str | None = None,
) -> xr.Dataset:
    ds = _coerce_to_dataset(data)
    src_name = _resolve_source_var(ds, canonical_id, source_var)
    da = _standardize_metadata(ds[src_name], canonical_id)

    if "time" in da.dims:
        if da.sizes["time"] != 1:
            raise ValueError(f"Static variable '{canonical_id}' must not have more than one time step.")
        da = da.isel(time=0, drop=True)

    da = da.rename(canonical_id)
    out = da.to_dataset(name=canonical_id)
    out.attrs.update(ds.attrs)
    out.attrs.setdefault("Conventions", "CF-1.8")
    return out


def save_annual_variable(
    data: xr.Dataset | xr.DataArray,
    canonical_id: str,
    year: int,
    source_var: str | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Save an annual variable using the canonical registry path."""

    logger = get_logger("io")
    spec = get_variable_spec(canonical_id)
    if spec.is_static:
        raise ValueError(f"'{canonical_id}' is registered as static. Use save_static_variable().")

    ds = _annual_dataset_from_variable(data, canonical_id=canonical_id, year=year, source_var=source_var)
    out_path = output_path_for(canonical_id, year=year, base_dir=base_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    encoding = _encoding_for(canonical_id)
    encoding["time"] = {"units": "days since 1900-01-01", "calendar": "standard"}
    ds.to_netcdf(out_path, encoding=encoding)
    logger.info("Saved %s → %s", canonical_id, out_path)
    return out_path


def save_static_variable(
    data: xr.Dataset | xr.DataArray,
    canonical_id: str,
    source_var: str | None = None,
    base_dir: Path | None = None,
) -> Path:
    """Save a static variable using the canonical registry path."""

    logger = get_logger("io")
    spec = get_variable_spec(canonical_id)
    if not spec.is_static:
        raise ValueError(f"'{canonical_id}' is not registered as static. Use save_annual_variable().")

    ds = _static_dataset_from_variable(data, canonical_id=canonical_id, source_var=source_var)
    out_path = output_path_for(canonical_id, base_dir=base_dir)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path, encoding=_encoding_for(canonical_id))
    logger.info("Saved %s → %s", canonical_id, out_path)
    return out_path
