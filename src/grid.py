"""Master 0.25° grid template for WorldTensor."""

import numpy as np
import xarray as xr


RESOLUTION = 0.25
LAT_MIN, LAT_MAX = -90.0, 90.0
LON_MIN, LON_MAX = 0.0, 359.75
N_LAT = 721  # -90 to 90 inclusive at 0.25°
N_LON = 1440  # 0 to 359.75 inclusive at 0.25°

YEAR_START = 1900
YEAR_END = 2025


def make_template(year: int) -> xr.Dataset:
    """Create an empty xarray Dataset on the master 0.25° global grid.

    Parameters
    ----------
    year : int
        Year for the time coordinate (single annual timestamp).

    Returns
    -------
    xr.Dataset
        Empty dataset with lat (721), lon (1440), and time (1) coordinates.
    """
    lat = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
    lon = np.linspace(0, 360, N_LON, endpoint=False)
    time = [np.datetime64(f"{year}-01-01")]

    ds = xr.Dataset(
        coords={
            "lat": ("lat", lat, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", lon, {"units": "degrees_east", "long_name": "longitude"}),
            "time": ("time", time, {"long_name": "time"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": "WorldTensor Global Gridded Dataset",
            "resolution": "0.25 degree",
        },
    )
    return ds


if __name__ == "__main__":
    ds = make_template(2020)
    print(ds)
    print(f"\nlat: {ds.sizes['lat']} points ({float(ds.lat.min()):.2f} to {float(ds.lat.max()):.2f})")
    print(f"lon: {ds.sizes['lon']} points ({float(ds.lon.min()):.2f} to {float(ds.lon.max()):.2f})")
