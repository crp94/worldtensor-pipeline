"""Compute line-length intersections per grid cell using geocube."""

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from src.grid import RESOLUTION, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX, N_LAT, N_LON, make_template
from src.utils import get_logger

logger = get_logger("lines_to_grid")


def lines_to_grid(
    gdf: gpd.GeoDataFrame,
    year: int,
    var_name: str = "line_length",
    attrs: dict | None = None,
    crs_m: str = "EPSG:6933",
) -> xr.Dataset:
    """Compute total line length (meters) intersecting each 0.25° grid cell.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with LineString or MultiLineString geometries.
    year : int
        Year for the output template.
    var_name : str
        Name for the output variable.
    attrs : dict, optional
        CF attributes (units, long_name) for the output variable.
    crs_m : str
        Projected CRS for computing lengths in meters.

    Returns
    -------
    xr.Dataset
        Dataset on the master grid with line length per cell.
    """
    template = make_template(year)

    grid_cells = _build_grid_cells()

    gdf_m = gdf.to_crs(crs_m)

    data = np.zeros((N_LAT, N_LON), dtype=np.float32)

    spatial_index = gdf.sindex
    for idx, cell in grid_cells.iterrows():
        lat_i, lon_i = idx
        candidates = list(spatial_index.intersection(cell.geometry.bounds))
        if not candidates:
            continue

        clipped = gdf_m.iloc[candidates].clip(cell.geometry)
        total_length = clipped.geometry.length.sum()
        data[lat_i, lon_i] = total_length

    ds = template.copy()
    ds[var_name] = (["lat", "lon"], data)
    ds[var_name].attrs = attrs or {"units": "m", "long_name": f"Total {var_name} per grid cell"}

    logger.info("Computed line-length grid for '%s' from %d features", var_name, len(gdf))
    return ds


def _build_grid_cells() -> gpd.GeoDataFrame:
    """Build a GeoDataFrame of all grid cell polygons, indexed by (lat_idx, lon_idx)."""
    records = []
    lat_edges = np.linspace(LAT_MIN - RESOLUTION / 2, LAT_MAX + RESOLUTION / 2, N_LAT + 1)
    lon_edges = np.linspace(LON_MIN - RESOLUTION / 2, LON_MAX + RESOLUTION / 2, N_LON + 1)

    for i in range(N_LAT):
        for j in range(N_LON):
            cell = box(lon_edges[j], lat_edges[i], lon_edges[j + 1], lat_edges[i + 1])
            records.append({"geometry": cell, "lat_idx": i, "lon_idx": j})

    gdf = gpd.GeoDataFrame(records, crs="EPSG:4326")
    gdf = gdf.set_index(["lat_idx", "lon_idx"])
    return gdf
