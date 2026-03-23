"""Regrid any raster to the master 0.25° template via xarray interpolation."""

import numpy as np
import xarray as xr
import rioxarray
import rasterio
from pathlib import Path

from src.grid import make_template
from src.utils import get_logger

logger = get_logger("raster_to_grid")


def load_raster(path: str | Path) -> xr.DataArray:
    """Load a raster file (GeoTIFF, etc.) as an xarray DataArray."""
    da = rioxarray.open_rasterio(path)
    if "band" in da.dims:
        da = da.squeeze("band", drop=True)

    # Standardize to x/y for rioxarray operations
    rename = {}
    if "lon" in da.dims:
        rename["lon"] = "x"
    if "lat" in da.dims:
        rename["lat"] = "y"
    if rename:
        da = da.rename(rename)

    return da


def regrid_raster(
    source: xr.DataArray,
    year: int,
    var_name: str | None = None,
    method: str = "linear",
    source_lon_convention: str = "auto",
) -> xr.Dataset:
    """Regrid a source raster to the master 0.25° grid."""
    template = make_template(year)
    template_rio = template.rename({"lat": "y", "lon": "x"})
    template_rio.rio.write_crs("EPSG:4326", inplace=True)

    # 0. CRS-aware reprojection using rioxarray
    # This is the preferred method for rasters with valid CRS
    if source.rio.crs is not None:
        logger.info("Performing CRS-aware reprojection using rioxarray")
        rs = rasterio.enums.Resampling.bilinear if method == "linear" else rasterio.enums.Resampling.nearest
        
        # Ensure source has x/y dims for reproject
        if "lon" in source.dims:
            source = source.rename({"lon": "x", "lat": "y"})

        # Reproject to the template grid
        # rioxarray handles the CRS transform (including -180/180 to 0/360 if needed)
        regridded_da = source.rio.reproject_match(
            template_rio,
            resampling=rs
        )
        # Mask NoData values
        regridded_da = regridded_da.where(regridded_da > -9000)
        regridded = regridded_da.values
    else:
        # 1. Fallback to simple interpolation for data without CRS metadata
        logger.info("Performing simple xarray interpolation (no CRS found)")
        
        # Mask NoData first
        source = source.where(source > -9000)
        
        # Ensure coordinates are named lon/lat for normalization logic
        if "x" in source.dims:
            source = source.rename({"x": "lon", "y": "lat"})
            
        source = _normalize_longitude(source, source_lon_convention)
        source = _normalize_lat_direction(source)

        regridded_obj = source.interp(
            lat=template.lat,
            lon=template.lon,
            method=method,
            kwargs={"fill_value": "extrapolate"},
        )
        regridded = regridded_obj.values

    out_var = var_name or source.name or "data"
    ds = template.copy()
    ds[out_var] = (("lat", "lon"), regridded)
    ds[out_var].attrs = source.attrs

    logger.info("Regridded '%s' to master grid", out_var)
    return ds


def _normalize_longitude(da: xr.DataArray, convention: str) -> xr.DataArray:
    """Convert longitude to 0–360 convention and handle periodic boundaries."""
    # Ensure coordinates are sorted and consistent
    lon_name = "lon" if "lon" in da.coords else "x"
    
    # Drop duplicates if any (MODIS sometimes has small rounding overlaps)
    _, index = np.unique(da[lon_name], return_index=True)
    da = da.isel({lon_name: index})

    if convention == "auto":
        lon_vals = da[lon_name].values
        convention = "-180_180" if float(np.min(lon_vals)) < 0 else "0_360"

    if convention == "-180_180":
        da = da.assign_coords({lon_name: (da[lon_name].values % 360)})
        da = da.sortby(lon_name)
        # Check again for duplicates after modulo
        _, index = np.unique(da[lon_name], return_index=True)
        da = da.isel({lon_name: index})
        logger.info("Converted longitude from -180..180 to 0..360")
    else:
        da = da.sortby(lon_name)

    # To handle the 0/360 meridian properly during interpolation,
    # we pad the array by wrapping a small slice from each end to the other.
    # We add 2 points from each side (assuming global data).
    pad_width = 2
    
    # xarray's pad with mode='wrap' works on the data
    da_padded = da.pad({lon_name: pad_width}, mode="wrap")
    
    # But coordinate values become NaN in the padded regions.
    # We must manually reconstruct the coordinate axis to be monotonic.
    orig_lons = da[lon_name].values
    dx = np.median(np.diff(orig_lons))
    
    new_lons = np.zeros(len(orig_lons) + 2 * pad_width)
    new_lons[pad_width:-pad_width] = orig_lons
    
    # Fill the left padding (values < min_lon)
    for i in range(pad_width):
        new_lons[pad_width - 1 - i] = new_lons[pad_width - i] - dx
        
    # Fill the right padding (values > max_lon)
    for i in range(pad_width):
        new_lons[-pad_width + i] = new_lons[-pad_width - 1 + i] + dx
        
    da_padded = da_padded.assign_coords({lon_name: new_lons})
    
    if lon_name == "x":
        da_padded = da_padded.rename({"x": "lon"})
        
    return da_padded


def _normalize_lat_direction(da: xr.DataArray) -> xr.DataArray:
    """Ensure latitude is in ascending order (-90 to 90)."""
    lat_name = "lat" if "lat" in da.coords else "y"
    
    # Drop duplicates
    _, index = np.unique(da[lat_name], return_index=True)
    da = da.isel({lat_name: index})

    if da[lat_name].values[0] > da[lat_name].values[-1]:
        da = da.isel({lat_name: slice(None, None, -1)})
        logger.info("Flipped latitude to ascending order")
    
    if lat_name == "y":
        da = da.rename({"y": "lat"})
    return da
