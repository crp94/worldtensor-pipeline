"""Aggregate point data into the master 0.25° grid cells."""

import geopandas as gpd
import numpy as np
import xarray as xr
from shapely.geometry import box

from src.grid import RESOLUTION, make_template
from src.utils import get_logger

logger = get_logger("points_to_grid")


def points_to_grid(
    gdf: gpd.GeoDataFrame,
    value_column: str,
    year: int,
    aggregation: str = "sum",
    var_name: str | None = None,
    attrs: dict | None = None,
) -> xr.Dataset:
    """Aggregate point geometries into 0.25° grid cells.

    Parameters
    ----------
    gdf : gpd.GeoDataFrame
        GeoDataFrame with point geometries and a value column.
    value_column : str
        Column in gdf containing the values to aggregate.
    year : int
        Year for the output template.
    aggregation : str
        Aggregation method: 'sum', 'mean', or 'count'.
    var_name : str, optional
        Name for the output variable. Defaults to value_column.
    attrs : dict, optional
        CF attributes (units, long_name) for the output variable.

    Returns
    -------
    xr.Dataset
        Dataset on the master grid with aggregated values.
    """
    template = make_template(year)
    var_name = var_name or value_column

    gdf = gdf.copy()
    gdf["lat_idx"] = np.floor((gdf.geometry.y + 90) / RESOLUTION).astype(int)
    gdf["lon_idx"] = np.floor((gdf.geometry.x % 360) / RESOLUTION).astype(int)

    gdf["lat_idx"] = gdf["lat_idx"].clip(0, template.sizes["lat"] - 1)
    gdf["lon_idx"] = gdf["lon_idx"].clip(0, template.sizes["lon"] - 1)

    agg_funcs = {"sum": "sum", "mean": "mean", "count": "count"}
    if aggregation not in agg_funcs:
        raise ValueError(f"Unsupported aggregation '{aggregation}'. Use: {list(agg_funcs)}")

    grouped = gdf.groupby(["lat_idx", "lon_idx"])[value_column].agg(agg_funcs[aggregation])

    data = np.full((template.sizes["lat"], template.sizes["lon"]), np.nan, dtype=np.float32)
    for (lat_i, lon_i), val in grouped.items():
        data[lat_i, lon_i] = val

    ds = template.copy()
    ds[var_name] = (["lat", "lon"], data)
    ds[var_name].attrs = attrs or {}

    logger.info("Gridded %d points → '%s' (%s aggregation)", len(gdf), var_name, aggregation)
    return ds
