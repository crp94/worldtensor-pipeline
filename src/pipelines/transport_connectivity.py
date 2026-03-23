"""Transport/connectivity candidate pipeline: download -> process -> plot.

Datasets implemented:
1) HMv2024 transport/accessibility (5-year anchors -> yearly linear interpolation)
2) Annual Human Footprint (figshare annual zips)
3) EDGAR transport emissions (annual sectors)
4) CEDS transport emissions (monthly -> yearly mean/std/min/max)
5) GMTDS shipping density (manual monthly files -> yearly mean/std/min/max)

Rules:
- only full years are kept
- monthly sources produce yearly mean/std/min/max
- HMv2024 missing years are filled by linear interpolation
"""

from __future__ import annotations

import gc
import re
import subprocess
import zipfile
from pathlib import Path
from typing import Any

import click
import matplotlib
matplotlib.use("Agg")
import numpy as np
import requests
import rioxarray
import xarray as xr
import yaml
from tqdm import tqdm

from src.download.edgar import (
    build_url as edgar_build_url,
    download_file as edgar_download_file,
    load_edgar_config,
)
from src.data_layout import output_dir_for, output_path_for
from src.grid import LAT_MAX, LAT_MIN, LON_MAX, LON_MIN, N_LAT, N_LON
from src.processing.edgar_to_yearly import process_file as edgar_process_file
from src.utils import add_cyclic_point_xr, get_logger, plot_time_series, save_annual_variable, save_to_netcdf

logger = get_logger("pipeline.transport_connectivity")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "transport_connectivity.yml"

TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)
MONTHLY_STATS = ("mean", "std", "min", "max")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_dataset_paths(config: dict) -> tuple[Path, Path, Path]:
    out = config["output"]
    final_root = PROJECT_ROOT / out["final_root"]
    plots_root = PROJECT_ROOT / out["plots_root"]
    raw_root = PROJECT_ROOT / out["raw_root"]
    final_root.mkdir(parents=True, exist_ok=True)
    plots_root.mkdir(parents=True, exist_ok=True)
    raw_root.mkdir(parents=True, exist_ok=True)
    return final_root, plots_root, raw_root


def _transport_output_dir(final_root: Path, var_name: str, default_domain: str | None = None) -> Path:
    try:
        return output_dir_for(var_name, base_dir=final_root)
    except KeyError:
        if default_domain:
            return final_root / default_domain / var_name
        return final_root / var_name


def _transport_output_path(
    final_root: Path,
    var_name: str,
    year: int,
    default_domain: str | None = None,
) -> Path:
    try:
        return output_path_for(var_name, year=year, base_dir=final_root)
    except KeyError:
        return _transport_output_dir(final_root, var_name, default_domain=default_domain) / f"{year}.nc"


def _resolve_datasets(config: dict, selected: tuple[str, ...]) -> dict:
    datasets = config.get("datasets", {})
    if selected:
        resolved = {k: v for k, v in datasets.items() if k in selected}
        missing = sorted(set(selected) - set(resolved))
        if missing:
            logger.warning("Unknown datasets skipped: %s", missing)
        return resolved
    return {k: v for k, v in datasets.items() if bool(v.get("enabled", True))}


def _normalize_coords(da: xr.DataArray) -> xr.DataArray:
    rename = {}
    if "latitude" in da.dims:
        rename["latitude"] = "lat"
    if "longitude" in da.dims:
        rename["longitude"] = "lon"
    if "x" in da.dims:
        rename["x"] = "lon"
    if "y" in da.dims:
        rename["y"] = "lat"
    if rename:
        da = da.rename(rename)

    if "lat" not in da.dims or "lon" not in da.dims:
        raise ValueError(f"Expected lat/lon dims, got {da.dims}")

    if float(da.lat.values[0]) > float(da.lat.values[-1]):
        da = da.isel(lat=slice(None, None, -1))

    if float(da.lon.min()) < 0:
        da = da.assign_coords(lon=(da.lon % 360.0))
        da = da.sortby("lon")
    else:
        da = da.sortby("lon")

    _, uniq_idx = np.unique(da.lon.values.astype(float), return_index=True)
    if len(uniq_idx) != da.sizes["lon"]:
        da = da.isel(lon=np.sort(uniq_idx))

    return da


def _pad_periodic_longitude(da: xr.DataArray, pad_cells: int = 2) -> xr.DataArray:
    da = da.sortby("lon")
    left = da.isel(lon=slice(-pad_cells, None)).assign_coords(
        lon=da.isel(lon=slice(-pad_cells, None)).lon - 360.0
    )
    right = da.isel(lon=slice(0, pad_cells)).assign_coords(
        lon=da.isel(lon=slice(0, pad_cells)).lon + 360.0
    )
    return xr.concat([left, da, right], dim="lon")


def _to_target_grid(da: xr.DataArray, method: str = "linear") -> xr.DataArray:
    da = _normalize_coords(da)
    same_shape = da.sizes.get("lat") == N_LAT and da.sizes.get("lon") == N_LON
    if same_shape:
        lat_ok = np.isclose(float(da.lat.min()), LAT_MIN) and np.isclose(float(da.lat.max()), LAT_MAX)
        lon_ok = np.isclose(float(da.lon.min()), LON_MIN) and np.isclose(float(da.lon.max()), LON_MAX)
        if lat_ok and lon_ok:
            return da

    da = _pad_periodic_longitude(da, pad_cells=2)
    return da.interp(lat=TARGET_LAT, lon=TARGET_LON, method=method)


def _save_year_grid(
    out_root: Path,
    var_name: str,
    long_name: str,
    units: str,
    year: int,
    arr2d: np.ndarray,
    source_name: str,
    overwrite: bool,
    default_domain: str | None = None,
) -> Path:
    out_path = _transport_output_path(out_root, var_name, year, default_domain=default_domain)
    if out_path.exists() and not overwrite:
        return out_path

    ds = xr.Dataset(
        {
            var_name: (
                ["lat", "lon"],
                arr2d.astype(np.float32),
                {"units": units, "long_name": long_name},
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor {long_name}",
            "source": source_name,
            "year": int(year),
        },
    )
    try:
        saved = save_annual_variable(ds, var_name, year, base_dir=out_root)
    except KeyError:
        target_dir = _transport_output_dir(out_root, var_name, default_domain=default_domain)
        saved = save_to_netcdf(ds, var_name, year, output_dir=target_dir.parent if target_dir.name == var_name else target_dir)
        if saved.parent != target_dir:
            saved = target_dir / saved.name
    ds.close()
    return saved


def _yearly_stats_from_monthly(da_monthly: xr.DataArray, require_full_year: bool = True) -> dict[int, dict[str, xr.DataArray]]:
    if "time" not in da_monthly.dims:
        raise ValueError("Monthly data must include a time dimension")

    out: dict[int, dict[str, xr.DataArray]] = {}
    years = sorted(set(int(y) for y in da_monthly.time.dt.year.values))
    for year in years:
        sub = da_monthly.sel(time=str(year))
        n_month = int(sub.sizes.get("time", 0))
        if require_full_year and n_month < 12:
            continue
        if n_month == 0:
            continue

        out[year] = {
            "mean": sub.mean(dim="time", skipna=True),
            "std": sub.std(dim="time", skipna=True),
            "min": sub.min(dim="time", skipna=True),
            "max": sub.max(dim="time", skipna=True),
        }
    return out


def _plot_maps_from_outputs(
    final_root: Path,
    plots_root: Path,
    var_name: str,
    long_name: str,
    cmap: str,
    map_every: int,
    default_domain: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
):
    data_dir = _transport_output_dir(final_root, var_name, default_domain=default_domain)
    if not data_dir.exists():
        return

    years = sorted(int(p.stem) for p in data_dir.glob("*.nc") if p.stem.isdigit())
    if not years:
        return

    if start_year is not None:
        years = [y for y in years if y >= start_year]
    if end_year is not None:
        years = [y for y in years if y <= end_year]
    if not years:
        return

    selected = sorted({years[0], years[-1], *[y for y in years if y % max(1, map_every) == 0]})
    for year in selected:
        in_nc = data_dir / f"{year}.nc"
        out_png = plots_root / "maps" / var_name / f"{year}.png"
        if out_png.exists():
            continue
        ds = xr.open_dataset(in_nc, decode_timedelta=False)
        try:
            _plot_transport_map(
                da=ds[var_name],
                title=f"{long_name} ({year})",
                out_path=out_png,
                cmap=cmap,
            )
        finally:
            ds.close()


def _plot_transport_map(da: xr.DataArray, title: str, out_path: Path, cmap: str = "viridis") -> None:
    """Plot with robust scaling for sparse/skewed transport variables."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    data_cyclic, lon_cyclic, lat_vals = add_cyclic_point_xr(da)
    arr = np.asarray(data_cyclic, dtype=np.float64)
    finite = arr[np.isfinite(arr)]

    norm = None
    vmin = None
    vmax = None

    if finite.size > 0:
        positive = finite[finite > 0]
        if positive.size > 100:
            p05 = float(np.nanpercentile(positive, 5))
            p99 = float(np.nanpercentile(positive, 99))
            if p99 > 0 and p05 > 0 and (p99 / p05) > 500:
                norm = LogNorm(vmin=max(p05, 1e-12), vmax=max(p99, p05 * 10))
            else:
                vmin = 0.0
                vmax = max(p99, 1e-12)
        else:
            p02 = float(np.nanpercentile(finite, 2))
            p98 = float(np.nanpercentile(finite, 98))
            vmin = p02
            vmax = p98 if p98 > p02 else p02 + 1e-9
    else:
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(1, 1, figsize=(12, 7), subplot_kw={"projection": ccrs.Robinson()})
    im = ax.pcolormesh(
        lon_cyclic,
        lat_vals,
        arr,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        norm=norm,
        vmin=None if norm is not None else vmin,
        vmax=None if norm is not None else vmax,
        shading="auto",
    )
    plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.05, aspect=40, shrink=0.8, label=da.attrs.get("units", ""))
    ax.coastlines(linewidth=0.5)
    ax.add_feature(cfeature.BORDERS, linewidth=0.3, linestyle=":")
    ax.set_global()
    ax.set_title(title, fontsize=12, pad=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_series_from_outputs(
    final_root: Path,
    plots_root: Path,
    var_name: str,
    long_name: str,
    units: str,
    default_domain: str | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
):
    data_dir = _transport_output_dir(final_root, var_name, default_domain=default_domain)
    if not data_dir.exists():
        return

    years = sorted(int(p.stem) for p in data_dir.glob("*.nc") if p.stem.isdigit())
    if start_year is not None:
        years = [y for y in years if y >= start_year]
    if end_year is not None:
        years = [y for y in years if y <= end_year]
    if not years:
        return

    vals: list[float] = []
    out_years: list[int] = []
    for year in years:
        ds = xr.open_dataset(data_dir / f"{year}.nc", decode_timedelta=False)
        try:
            vals.append(float(ds[var_name].mean(skipna=True).values))
            out_years.append(year)
        finally:
            ds.close()

    if not out_years:
        return

    out_path = plots_root / "timeseries" / f"{var_name}.png"
    plot_time_series(
        years=out_years,
        values=vals,
        title=f"{long_name} (global mean)",
        ylabel=units,
        out_path=out_path,
        color="#d62728",
    )


def _download_stream(url: str, out_path: Path, overwrite: bool = False, timeout: int = 1800) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    with requests.get(url, stream=True, timeout=timeout) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with open(tmp_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name, leave=False) as pbar:
            for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    tmp_path.replace(out_path)
    return out_path


def _warp_to_global_025(src_path: Path, dst_path: Path, method: str = "average", src_nodata: float | None = None):
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "gdalwarp",
        "-q",
        "-overwrite",
        "-t_srs",
        "EPSG:4326",
        "-te",
        "-180",
        "-90",
        "180",
        "90",
        "-tr",
        "0.25",
        "0.25",
        "-tap",
        "-r",
        method,
    ]
    if src_nodata is not None:
        cmd.extend(["-srcnodata", str(src_nodata)])
    cmd.extend(["-dstnodata", "nan", "-ot", "Float32", str(src_path), str(dst_path)])
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Dataset 1: HMv2024
# ---------------------------------------------------------------------------

def _zenodo_files_map(record_id: int) -> dict[str, str]:
    url = f"https://zenodo.org/api/records/{record_id}"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    payload = resp.json()
    out: dict[str, str] = {}
    for f in payload.get("files", []):
        key = str(f.get("key", ""))
        dl = f.get("links", {}).get("self") or f.get("download_url")
        if key and dl:
            out[key] = dl
    return out


def _run_hmv2024(
    ds_key: str,
    info: dict,
    final_root: Path,
    plots_root: Path,
    raw_root: Path,
    do_download: bool,
    do_process: bool,
    do_plot: bool,
    overwrite: bool,
    map_every: int,
    start_year: int | None,
    end_year: int | None,
):
    output_domain = str(info.get("output_domain", "")).strip("/") or None
    ds_raw = raw_root / ds_key
    work = ds_raw / "work"
    ds_raw.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)

    files_map: dict[str, str] = {}
    if do_download:
        files_map = _zenodo_files_map(int(info["zenodo_record_id"]))

    anchor_years = sorted(int(y) for y in info["anchor_years"])
    y0_out, y1_out = [int(v) for v in info["output_year_range"]]
    if start_year is not None:
        y0_out = max(y0_out, int(start_year))
    if end_year is not None:
        y1_out = min(y1_out, int(end_year))

    for var_name, var_cfg in info["variables"].items():
        anchors: dict[int, np.ndarray] = {}

        for year in anchor_years:
            src_name = var_cfg["source_filename_pattern"].format(year=year)
            raw_tif = ds_raw / src_name

            if do_download and (overwrite or not raw_tif.exists()):
                url = files_map.get(src_name)
                if not url:
                    # fallback fuzzy match
                    code = str(var_cfg.get("source_code", ""))
                    cand = [k for k in files_map if f"_{year}c_{code}.tif" in k]
                    if cand:
                        url = files_map[cand[0]]
                if url:
                    try:
                        _download_stream(url, raw_tif, overwrite=overwrite, timeout=3600)
                    except Exception as e:
                        logger.warning("HMv download failed %s: %s", src_name, e)

            if not raw_tif.exists():
                continue

            warped = work / f"{var_name}_{year}_025.tif"
            if overwrite or not warped.exists():
                _warp_to_global_025(
                    src_path=raw_tif,
                    dst_path=warped,
                    method=info.get("downscale_method", "average"),
                )

            da = rioxarray.open_rasterio(warped, masked=True).squeeze(drop=True)
            da = _to_target_grid(da.rename({"x": "lon", "y": "lat"}), method="nearest")
            anchors[year] = da.values.astype(np.float32)
            da.close()

        if do_process and anchors:
            years_anchor = sorted(anchors.keys())
            y0 = max(y0_out, years_anchor[0])
            y1 = min(y1_out, years_anchor[-1])
            for year in range(y0, y1 + 1):
                if year in anchors:
                    arr = anchors[year]
                else:
                    left = max(y for y in years_anchor if y <= year)
                    right = min(y for y in years_anchor if y >= year)
                    if left == right:
                        arr = anchors[left]
                    else:
                        t = (year - left) / (right - left)
                        arr = ((1.0 - t) * anchors[left] + t * anchors[right]).astype(np.float32)

                _save_year_grid(
                    out_root=final_root,
                    var_name=var_name,
                    long_name=str(var_cfg["long_name"]),
                    units=str(var_cfg["units"]),
                    year=year,
                    arr2d=arr,
                    source_name=info["source_name"],
                    overwrite=overwrite,
                    default_domain=output_domain,
                )

        if do_plot:
            _plot_maps_from_outputs(
                final_root=final_root,
                plots_root=plots_root / ds_key,
                var_name=var_name,
                long_name=str(var_cfg["long_name"]),
                cmap=var_cfg.get("cmap", "viridis"),
                map_every=map_every,
                default_domain=output_domain,
                start_year=start_year,
                end_year=end_year,
            )
            _plot_series_from_outputs(
                final_root=final_root,
                plots_root=plots_root / ds_key,
                var_name=var_name,
                long_name=str(var_cfg["long_name"]),
                units=str(var_cfg["units"]),
                default_domain=output_domain,
                start_year=start_year,
                end_year=end_year,
            )


# ---------------------------------------------------------------------------
# Dataset 2: Human Footprint annual
# ---------------------------------------------------------------------------

def _figshare_article(article_id: int) -> dict[str, Any]:
    url = f"https://api.figshare.com/v2/articles/{article_id}"
    resp = requests.get(url, timeout=120)
    resp.raise_for_status()
    return resp.json()


def _extract_first_tif(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        tif_members = [m for m in zf.namelist() if m.lower().endswith((".tif", ".tiff"))]
        if not tif_members:
            raise FileNotFoundError(f"No tif in {zip_path}")
        member = tif_members[0]
        out_name = Path(member).name
        out_path = out_dir / out_name
        if not out_path.exists():
            zf.extract(member, out_dir)
            extracted = out_dir / member
            if extracted != out_path:
                out_path.parent.mkdir(parents=True, exist_ok=True)
                extracted.replace(out_path)
        return out_path


def _run_human_footprint(
    ds_key: str,
    info: dict,
    final_root: Path,
    plots_root: Path,
    raw_root: Path,
    do_download: bool,
    do_process: bool,
    do_plot: bool,
    overwrite: bool,
    map_every: int,
    start_year: int | None,
    end_year: int | None,
):
    output_domain = str(info.get("output_domain", "")).strip("/") or None
    ds_raw = raw_root / ds_key
    work = ds_raw / "work"
    extracted = ds_raw / "extracted"
    ds_raw.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    extracted.mkdir(parents=True, exist_ok=True)

    y0, y1 = [int(v) for v in info["temporal_range"]]
    if start_year is not None:
        y0 = max(y0, int(start_year))
    if end_year is not None:
        y1 = min(y1, int(end_year))

    var_info = info["variable"]
    var_name = str(var_info["name"])
    zip_pat = re.compile(var_info["zip_name_pattern"])

    files_by_year: dict[int, dict[str, Any]] = {}
    if do_download:
        art = _figshare_article(int(info["figshare_article_id"]))
        for f in art.get("files", []):
            fname = str(f.get("name", ""))
            m = zip_pat.match(fname)
            if not m:
                continue
            year = int(m.group(1))
            files_by_year[year] = f

    for year in range(y0, y1 + 1):
        zip_path = ds_raw / f"hfp{year}.zip"
        if do_download and (overwrite or not zip_path.exists()):
            f = files_by_year.get(year)
            if f:
                _download_stream(f["download_url"], zip_path, overwrite=overwrite, timeout=2400)

        if do_process and zip_path.exists():
            tif_path = _extract_first_tif(zip_path, extracted / str(year))
            warped = work / f"hfp_{year}_025.tif"
            if overwrite or not warped.exists():
                _warp_to_global_025(tif_path, warped, method=info.get("downscale_method", "average"))

            da = rioxarray.open_rasterio(warped, masked=True).squeeze(drop=True)
            da = _to_target_grid(da.rename({"x": "lon", "y": "lat"}), method="nearest")
            _save_year_grid(
                out_root=final_root,
                var_name=var_name,
                long_name=str(var_info["long_name"]),
                units=str(var_info["units"]),
                year=year,
                arr2d=da.values,
                source_name=info["source_name"],
                overwrite=overwrite,
                default_domain=output_domain,
            )
            da.close()

    if do_plot:
        _plot_maps_from_outputs(
            final_root=final_root,
            plots_root=plots_root / ds_key,
            var_name=var_name,
            long_name=str(var_info["long_name"]),
            cmap=var_info.get("cmap", "viridis"),
            map_every=map_every,
            default_domain=output_domain,
            start_year=start_year,
            end_year=end_year,
        )
        _plot_series_from_outputs(
            final_root=final_root,
            plots_root=plots_root / ds_key,
            var_name=var_name,
            long_name=str(var_info["long_name"]),
            units=str(var_info["units"]),
            default_domain=output_domain,
            start_year=start_year,
            end_year=end_year,
        )


# ---------------------------------------------------------------------------
# Dataset 3: EDGAR transport annual
# ---------------------------------------------------------------------------

def _run_edgar_transport(
    ds_key: str,
    info: dict,
    final_root: Path,
    plots_root: Path,
    raw_root: Path,
    do_download: bool,
    do_process: bool,
    do_plot: bool,
    overwrite: bool,
    map_every: int,
    start_year: int | None,
    end_year: int | None,
):
    output_domain = str(info.get("output_domain", "")).strip("/") or None
    edgar_cfg = load_edgar_config()
    y0, y1 = [int(v) for v in info["temporal_range"]]
    if start_year is not None:
        y0 = max(y0, int(start_year))
    if end_year is not None:
        y1 = min(y1, int(end_year))

    years = list(range(y0, y1 + 1))
    sub = str(info["substance"])
    base_url = edgar_cfg["source_url_base"]

    edgar_raw = raw_root / ds_key
    edgar_raw.mkdir(parents=True, exist_ok=True)

    for var_name, sec_cfg in info["sectors"].items():
        sec_code = str(sec_cfg["code"])
        sec_long = sec_cfg.get("long_name", sec_code)
        raw_sec_dir = edgar_raw / sub / sec_code

        for year in tqdm(years, desc=f"EDGAR {sub}/{sec_code}", leave=False):
            raw_nc = raw_sec_dir / f"EDGAR_2025_GHG_{sub}_{year}_{sec_code}_flx_nc.nc"

            if do_download:
                url = edgar_build_url(base_url, sub, sec_code, year)
                edgar_download_file(url, raw_sec_dir, sub, sec_code, year, overwrite=overwrite)

            if do_process and raw_nc.exists():
                edgar_process_file(
                    raw_nc_path=raw_nc,
                    substance=sub,
                    sector=sec_code,
                    year=year,
                    units=str(sec_cfg["units"]),
                    long_name=str(sec_cfg["long_name"]),
                    sector_long_name=str(sec_long),
                    overwrite=overwrite,
                )

                # Read processed output and store with dataset-specific variable name.
                processed_nc = PROJECT_ROOT / "data" / "final" / "emissions" / f"{sub}_{sec_code}" / f"{year:04d}.nc"
                if processed_nc.exists():
                    ds = xr.open_dataset(processed_nc)
                    try:
                        source_var = f"{sub}_{sec_code}" if f"{sub}_{sec_code}" in ds.data_vars else next(iter(ds.data_vars))
                        arr = ds[source_var].values.astype(np.float32)
                    finally:
                        ds.close()
                    _save_year_grid(
                        out_root=final_root,
                        var_name=var_name,
                        long_name=str(sec_cfg["long_name"]),
                        units=str(sec_cfg["units"]),
                        year=year,
                        arr2d=arr,
                        source_name=info["source_name"],
                        overwrite=overwrite,
                        default_domain=output_domain,
                    )

        if do_plot:
            _plot_maps_from_outputs(
                final_root=final_root,
                plots_root=plots_root / ds_key,
                var_name=var_name,
                long_name=str(sec_cfg["long_name"]),
                cmap=sec_cfg.get("cmap", "viridis"),
                map_every=map_every,
                default_domain=output_domain,
                start_year=start_year,
                end_year=end_year,
            )
            _plot_series_from_outputs(
                final_root=final_root,
                plots_root=plots_root / ds_key,
                var_name=var_name,
                long_name=str(sec_cfg["long_name"]),
                units=str(sec_cfg["units"]),
                default_domain=output_domain,
                start_year=start_year,
                end_year=end_year,
            )


# ---------------------------------------------------------------------------
# Dataset 4: CEDS monthly input4CMIP
# ---------------------------------------------------------------------------

def _extract_member(zip_path: Path, member_pattern: str, out_dir: Path) -> Path:
    pat = re.compile(member_pattern)
    with zipfile.ZipFile(zip_path) as zf:
        members = [m for m in zf.namelist() if pat.search(Path(m).name)]
        if not members:
            raise FileNotFoundError(f"No member matching {member_pattern} in {zip_path.name}")
        member = members[0]
        out_path = out_dir / Path(member).name
        if out_path.exists():
            return out_path
        zf.extract(member, out_dir)
        extracted = out_dir / member
        if extracted != out_path:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            extracted.replace(out_path)
        return out_path


def _run_ceds_transport(
    ds_key: str,
    info: dict,
    final_root: Path,
    plots_root: Path,
    raw_root: Path,
    do_download: bool,
    do_process: bool,
    do_plot: bool,
    overwrite: bool,
    map_every: int,
    start_year: int | None,
    end_year: int | None,
):
    output_domain = str(info.get("output_domain", "")).strip("/") or None
    ds_raw = raw_root / ds_key
    ds_raw.mkdir(parents=True, exist_ok=True)

    source_zip = ds_raw / str(info["source_zip_key"])
    source_nc = ds_raw / str(info["source_nc_pattern"])

    if do_download and (overwrite or not source_zip.exists()):
        files_map = _zenodo_files_map(int(info["zenodo_record_id"]))
        key = str(info["source_zip_key"])
        url = files_map.get(key)
        if not url:
            raise click.ClickException(f"CEDS zip key not found in Zenodo record: {key}")
        _download_stream(url, source_zip, overwrite=overwrite, timeout=7200)

    if do_download and source_zip.exists() and (overwrite or not source_nc.exists()):
        source_nc = _extract_member(source_zip, str(info["source_nc_pattern"]), ds_raw)

    if not do_process or not source_nc.exists():
        return

    ds = xr.open_dataset(source_nc, decode_timedelta=False)
    data_vars = [v for v in ds.data_vars if {"time", "lat", "lon"}.issubset(set(ds[v].dims))]
    if not data_vars:
        ds.close()
        raise click.ClickException("No CEDS data variable with time/lat/lon found.")

    var = data_vars[0]
    da_all = ds[var]

    sector_dim = None
    for d in da_all.dims:
        if d not in {"time", "lat", "lon"}:
            sector_dim = d
            break

    y0, y1 = [int(v) for v in info["temporal_range"]]
    if start_year is not None:
        y0 = max(y0, int(start_year))
    if end_year is not None:
        y1 = min(y1, int(end_year))

    for out_var, var_cfg in info["variables"].items():
        sec_code = str(var_cfg["sector"])
        sec_idx = int(info["sector_index_map"][sec_code])

        if sector_dim is not None:
            da = da_all.isel({sector_dim: sec_idx})
        else:
            da = da_all

        da = _normalize_coords(da)

        for year in range(y0, y1 + 1):
            sub = da.sel(time=str(year))
            if int(sub.sizes.get("time", 0)) < 12 and bool(info.get("require_full_year", True)):
                continue
            if int(sub.sizes.get("time", 0)) == 0:
                continue

            da_025 = _to_target_grid(sub, method="linear")
            stats = _yearly_stats_from_monthly(da_025, require_full_year=bool(info.get("require_full_year", True)))
            if year not in stats:
                continue

            for stat in MONTHLY_STATS:
                var_stat = f"{out_var}_{stat}"
                _save_year_grid(
                    out_root=final_root,
                    var_name=var_stat,
                    long_name=f"{var_cfg['long_name']} ({stat})",
                    units=str(var_cfg["units"]),
                    year=year,
                    arr2d=stats[year][stat].values,
                    source_name=info["source_name"],
                    overwrite=overwrite,
                    default_domain=output_domain,
                )

            gc.collect()

        if do_plot:
            for stat in MONTHLY_STATS:
                var_stat = f"{out_var}_{stat}"
                _plot_maps_from_outputs(
                    final_root=final_root,
                    plots_root=plots_root / ds_key,
                    var_name=var_stat,
                    long_name=f"{var_cfg['long_name']} ({stat})",
                    cmap=var_cfg.get("cmap", "viridis"),
                    map_every=map_every,
                    default_domain=output_domain,
                    start_year=start_year,
                    end_year=end_year,
                )
                _plot_series_from_outputs(
                    final_root=final_root,
                    plots_root=plots_root / ds_key,
                    var_name=var_stat,
                    long_name=f"{var_cfg['long_name']} ({stat})",
                    units=str(var_cfg["units"]),
                    default_domain=output_domain,
                    start_year=start_year,
                    end_year=end_year,
                )

    ds.close()


# ---------------------------------------------------------------------------
# Dataset 5: GMTDS monthly manual
# ---------------------------------------------------------------------------

def _parse_yyyymm(name: str) -> tuple[int, int] | None:
    m = re.search(r"(19|20)\d{2}(0[1-9]|1[0-2])", name)
    if not m:
        return None
    token = m.group(0)
    return int(token[:4]), int(token[4:6])


def _load_monthly_file(path: Path) -> xr.DataArray:
    if path.suffix.lower() in {".tif", ".tiff"}:
        da = rioxarray.open_rasterio(path, masked=True).squeeze(drop=True)
        da = da.rename({"x": "lon", "y": "lat"})
        return da

    if path.suffix.lower() == ".nc":
        ds = xr.open_dataset(path, decode_timedelta=False)
        try:
            cand = [v for v in ds.data_vars if {"lat", "lon"}.issubset(set(ds[v].dims))]
            if not cand:
                # try renamed coords
                cand = [v for v in ds.data_vars if {"latitude", "longitude"}.issubset(set(ds[v].dims))]
            if not cand:
                raise ValueError(f"No lat/lon variable found in {path}")
            da = ds[cand[0]].load()
        finally:
            ds.close()
        return da

    raise ValueError(f"Unsupported GMTDS file type: {path}")


def _run_gmtds(
    ds_key: str,
    info: dict,
    final_root: Path,
    plots_root: Path,
    raw_root: Path,
    do_download: bool,
    do_process: bool,
    do_plot: bool,
    overwrite: bool,
    map_every: int,
    start_year: int | None,
    end_year: int | None,
):
    output_domain = str(info.get("output_domain", "")).strip("/") or None
    # GMTDS requires authorized portal export. We process local monthly exports.
    ds_raw = raw_root / ds_key / "monthly"
    ds_raw.mkdir(parents=True, exist_ok=True)

    var_cfg = info["variable"]
    base_var = str(var_cfg["name"])

    if do_download:
        logger.info(
            "GMTDS download is manual. Place monthly .tif/.nc files with YYYYMM in filename under: %s",
            ds_raw,
        )

    if do_process:
        files = sorted(
            [
                p
                for p in ds_raw.glob("*")
                if p.is_file() and p.suffix.lower() in {".tif", ".tiff", ".nc"} and _parse_yyyymm(p.name)
            ]
        )
        if not files:
            logger.warning("No GMTDS monthly files found in %s", ds_raw)
        grouped: dict[int, dict[int, Path]] = {}
        for p in files:
            ym = _parse_yyyymm(p.name)
            if ym is None:
                continue
            y, m = ym
            grouped.setdefault(y, {})[m] = p

        y0, y1 = [int(v) for v in info["temporal_range"]]
        if start_year is not None:
            y0 = max(y0, int(start_year))
        if end_year is not None:
            y1 = min(y1, int(end_year))

        for year in range(y0, y1 + 1):
            by_month = grouped.get(year, {})
            if bool(info.get("require_full_year", True)) and len(by_month) < 12:
                continue
            if not by_month:
                continue

            monthly_arrays = []
            times = []
            for month in sorted(by_month):
                p = by_month[month]
                da_m = _load_monthly_file(p)
                da_m = _to_target_grid(da_m, method="linear")
                monthly_arrays.append(da_m.values.astype(np.float32))
                times.append(np.datetime64(f"{year}-{month:02d}-01"))
                da_m.close() if hasattr(da_m, "close") else None

            stack = np.stack(monthly_arrays, axis=0)
            da_monthly = xr.DataArray(
                stack,
                dims=("time", "lat", "lon"),
                coords={"time": np.array(times), "lat": TARGET_LAT, "lon": TARGET_LON},
            )
            stats = _yearly_stats_from_monthly(da_monthly, require_full_year=bool(info.get("require_full_year", True)))
            if year not in stats:
                continue

            for stat in MONTHLY_STATS:
                out_var = f"{base_var}_{stat}"
                _save_year_grid(
                    out_root=final_root,
                    var_name=out_var,
                    long_name=f"{var_cfg['long_name']} ({stat})",
                    units=str(var_cfg["units"]),
                    year=year,
                    arr2d=stats[year][stat].values,
                    source_name=info["source_name"],
                    overwrite=overwrite,
                    default_domain=output_domain,
                )

            gc.collect()

    if do_plot:
        for stat in MONTHLY_STATS:
            out_var = f"{base_var}_{stat}"
            _plot_maps_from_outputs(
                final_root=final_root,
                plots_root=plots_root / ds_key,
                var_name=out_var,
                long_name=f"{var_cfg['long_name']} ({stat})",
                cmap=var_cfg.get("cmap", "viridis"),
                map_every=map_every,
                default_domain=output_domain,
                start_year=start_year,
                end_year=end_year,
            )
            _plot_series_from_outputs(
                final_root=final_root,
                plots_root=plots_root / ds_key,
                var_name=out_var,
                long_name=f"{var_cfg['long_name']} ({stat})",
                units=str(var_cfg["units"]),
                default_domain=output_domain,
                start_year=start_year,
                end_year=end_year,
            )


@click.command()
@click.option("--datasets", "dataset_keys", "-d", multiple=True, help="Dataset key(s) from config.")
@click.option("--all", "run_all", is_flag=True, help="Run download + process + plot for selected datasets.")
@click.option("--download", is_flag=True, help="Run download step.")
@click.option("--process", is_flag=True, help="Run processing step.")
@click.option("--plot", "plot_step", is_flag=True, help="Run plotting step.")
@click.option("--skip-download", is_flag=True, help="Skip download when using --all.")
@click.option("--map-every", type=int, default=None, help="Map every N years (default from config).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
@click.option("--start-year", type=int, default=None, help="Optional global lower bound on years.")
@click.option("--end-year", type=int, default=None, help="Optional global upper bound on years.")
def main(
    dataset_keys: tuple[str, ...],
    run_all: bool,
    download: bool,
    process: bool,
    plot_step: bool,
    skip_download: bool,
    map_every: int | None,
    overwrite: bool,
    start_year: int | None,
    end_year: int | None,
):
    config = load_config()
    selected = _resolve_datasets(config, dataset_keys)
    if not selected:
        raise click.ClickException("No datasets selected.")

    if run_all:
        do_download = not skip_download
        do_process = True
        do_plot = True
    else:
        do_download = bool(download)
        do_process = bool(process)
        do_plot = bool(plot_step)

    if not any([do_download, do_process, do_plot]):
        raise click.ClickException("Specify --all or at least one of --download/--process/--plot")

    final_root, plots_root, raw_root = _resolve_dataset_paths(config)
    map_every_effective = int(map_every) if map_every is not None else int(config["output"].get("map_every", 5))

    click.echo(
        f"Transport/connectivity pipeline: datasets={len(selected)}, "
        f"download={do_download}, process={do_process}, plot={do_plot}, map_every={map_every_effective}"
    )

    for ds_key, ds_info in selected.items():
        kind = str(ds_info.get("kind", ""))
        click.echo(f"\n[{ds_key}] kind={kind}")
        try:
            if kind == "hmv2024_5year_interp":
                _run_hmv2024(
                    ds_key=ds_key,
                    info=ds_info,
                    final_root=final_root,
                    plots_root=plots_root,
                    raw_root=raw_root,
                    do_download=do_download,
                    do_process=do_process,
                    do_plot=do_plot,
                    overwrite=overwrite,
                    map_every=map_every_effective,
                    start_year=start_year,
                    end_year=end_year,
                )
            elif kind == "human_footprint_figshare_annual":
                _run_human_footprint(
                    ds_key=ds_key,
                    info=ds_info,
                    final_root=final_root,
                    plots_root=plots_root,
                    raw_root=raw_root,
                    do_download=do_download,
                    do_process=do_process,
                    do_plot=do_plot,
                    overwrite=overwrite,
                    map_every=map_every_effective,
                    start_year=start_year,
                    end_year=end_year,
                )
            elif kind == "edgar_transport_annual":
                _run_edgar_transport(
                    ds_key=ds_key,
                    info=ds_info,
                    final_root=final_root,
                    plots_root=plots_root,
                    raw_root=raw_root,
                    do_download=do_download,
                    do_process=do_process,
                    do_plot=do_plot,
                    overwrite=overwrite,
                    map_every=map_every_effective,
                    start_year=start_year,
                    end_year=end_year,
                )
            elif kind == "ceds_monthly_input4cmip":
                _run_ceds_transport(
                    ds_key=ds_key,
                    info=ds_info,
                    final_root=final_root,
                    plots_root=plots_root,
                    raw_root=raw_root,
                    do_download=do_download,
                    do_process=do_process,
                    do_plot=do_plot,
                    overwrite=overwrite,
                    map_every=map_every_effective,
                    start_year=start_year,
                    end_year=end_year,
                )
            elif kind == "gmtds_monthly_manual":
                _run_gmtds(
                    ds_key=ds_key,
                    info=ds_info,
                    final_root=final_root,
                    plots_root=plots_root,
                    raw_root=raw_root,
                    do_download=do_download,
                    do_process=do_process,
                    do_plot=do_plot,
                    overwrite=overwrite,
                    map_every=map_every_effective,
                    start_year=start_year,
                    end_year=end_year,
                )
            else:
                logger.warning("Unsupported dataset kind for %s: %s", ds_key, kind)
        except Exception as e:
            logger.error("Dataset failed %s: %s", ds_key, e)

    click.echo(f"\nDone. Outputs in {final_root} and plots in {plots_root}")


if __name__ == "__main__":
    main()
