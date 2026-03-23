"""Download/stage point datasets used by `src.pipelines.point_datasets`.

Datasets targeted:
- UCDP GED (direct download)
- GDIS (attempted; often access-gated)
- USGS Earthquakes (FDSN API)
- IBTrACS cyclones (NOAA CSV)
- NOAA significant volcanic eruptions (HazEL API)
- NASA COOLR landslides (WPRDC mirror CSV)

Usage:
    python -m src.download.point_datasets --all
    python -m src.download.point_datasets --dataset ucdp_ged --dataset earthquakes_usgs
"""

from __future__ import annotations

import io
import time
import warnings
import zipfile
from pathlib import Path

import click
import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

from src.grid import YEAR_START
from src.utils import get_logger

logger = get_logger("download.point_datasets")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CYCLONE_PRESSURE_START_YEAR = 1951

REQUEST_HEADERS = {"User-Agent": "worldtensor/point-datasets (+https://github.com/worldtensor)"}
USGS_FDSN_CSV_URL = "https://earthquake.usgs.gov/fdsnws/event/1/query.csv"
IBTRACS_CSV_URL = (
    "https://www.ncei.noaa.gov/data/international-best-track-archive-for-climate-stewardship-ibtracs/v04r01/access/csv/ibtracs.ALL.list.v04r01.csv"
)
NOAA_VOLCANO_EVENTS_URL = "https://www.ngdc.noaa.gov/hazel/hazard-service/api/v1/volcanoes"
COOLR_LANDSLIDES_CSV_URL = (
    "https://data.wprdc.org/dataset/7db7daf4-1fcc-4ad6-ad5e-6ed21a45b154/resource/dde1f413-c849-413c-b791-0f861bf219ce/download/globallandslides.csv"
)


def _download_file(url: str, out_path: Path, verify_ssl: bool = True, timeout: int = 1800) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s -> %s", url, out_path)
    with requests.get(url, stream=True, timeout=timeout, verify=verify_ssl, headers=REQUEST_HEADERS) as r:
        r.raise_for_status()
        total = int(r.headers.get("content-length", 0))
        with open(out_path, "wb") as f, tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name, leave=False) as pbar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))
    return True


def download_ucdp_ged() -> Path:
    raw_dir = PROJECT_ROOT / "data" / "raw" / "ucdp_ged"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "ged251-csv.zip"
    csv_out = raw_dir / "ucdp_ged.csv"

    _download_file("https://ucdp.uu.se/downloads/ged/ged251-csv.zip", zip_path, verify_ssl=True)

    with zipfile.ZipFile(zip_path, "r") as zf:
        csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
        if not csv_members:
            raise RuntimeError("UCDP zip did not contain CSV")
        with zf.open(csv_members[0]) as src, open(csv_out, "wb") as dst:
            dst.write(src.read())

    df = pd.read_csv(csv_out, low_memory=False)
    df = _clip_to_min_year(df, ["year", "date_start", "date"], "UCDP GED")
    df.to_csv(csv_out, index=False)
    zip_path.unlink(missing_ok=True)
    logger.info("UCDP GED staged at %s", csv_out)
    return csv_out


def download_gdis() -> Path | None:
    raw_dir = PROJECT_ROOT / "data" / "raw" / "gdis"
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "gdis.zip"

    # Public link from PRIO/GDIS page (can be access-gated).
    url = "https://cdn.cloud.prio.org/files/cffb60dc-5978-4eec-a1d2-14551067d84d/gdis-1960-2018-disasterlocations-csv.zip?inline=true"

    try:
        _download_file(url, zip_path, verify_ssl=True)
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_members = [m for m in zf.namelist() if m.lower().endswith(".csv")]
            if not csv_members:
                raise RuntimeError("Downloaded GDIS archive has no CSV")
            out_csv = raw_dir / "gdis.csv"
            with zf.open(csv_members[0]) as src, open(out_csv, "wb") as dst:
                dst.write(src.read())
        df = pd.read_csv(out_csv, low_memory=False)
        df = _clip_to_min_year(df, ["year", "start_year", "event_year"], "GDIS")
        df.to_csv(out_csv, index=False)
        zip_path.unlink(missing_ok=True)
        logger.info("GDIS staged at %s", out_csv)
        return out_csv
    except Exception as e:
        logger.warning("GDIS auto-download failed (%s). Place gdis.csv manually in %s", e, raw_dir)
        return None


def _request_json(url: str, params: dict | None = None, timeout: int = 180) -> dict:
    resp = requests.get(url, params=params, timeout=timeout, headers=REQUEST_HEADERS)
    resp.raise_for_status()
    return resp.json()


def _coalesce_numeric(frame: pd.DataFrame, cols: list[str]) -> pd.Series:
    out = pd.Series(np.nan, index=frame.index, dtype=np.float64)
    for c in cols:
        if c in frame.columns:
            s = pd.to_numeric(frame[c], errors="coerce")
            out = out.where(np.isfinite(out), s)
    return out


def _extract_years(frame: pd.DataFrame, candidates: list[str]) -> pd.Series:
    for col in candidates:
        if col not in frame.columns:
            continue

        if any(token in col.lower() for token in ("date", "time")):
            years = pd.to_datetime(frame[col], errors="coerce", utc=True).dt.year
        else:
            years = pd.to_numeric(frame[col], errors="coerce")
            if not years.notna().any():
                years = pd.to_datetime(frame[col], errors="coerce", utc=True).dt.year

        years = pd.to_numeric(years, errors="coerce")
        if years.notna().any():
            return years.astype("Int64")

    return pd.Series(pd.NA, index=frame.index, dtype="Int64")


def _clip_to_min_year(frame: pd.DataFrame, candidates: list[str], label: str) -> pd.DataFrame:
    years = _extract_years(frame, candidates)
    if not years.notna().any():
        logger.warning("%s year filter skipped; no usable year column in %s", label, candidates)
        return frame
    clipped = frame.loc[years >= YEAR_START].copy()
    logger.info("%s staged rows=%d after year >= %d filter", label, len(clipped), YEAR_START)
    return clipped


def download_earthquakes_usgs(year_min: int = 1900, year_max: int = 2025, min_magnitude: float = 4.5) -> Path:
    """Download yearly USGS earthquake events and stage a compact CSV."""
    out_path = PROJECT_ROOT / "data" / "raw" / "earthquakes_usgs" / "earthquakes_usgs.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    limit = 20000
    frames: list[pd.DataFrame] = []
    for year in range(int(year_min), int(year_max) + 1):
        offset = 1
        year_rows = 0
        while True:
            params = {
                "format": "csv",
                "starttime": f"{year}-01-01",
                "endtime": f"{year}-12-31",
                "minmagnitude": f"{min_magnitude:.2f}",
                "orderby": "time-asc",
                "limit": limit,
                "offset": offset,
            }
            resp = requests.get(USGS_FDSN_CSV_URL, params=params, timeout=300, headers=REQUEST_HEADERS)
            resp.raise_for_status()
            text = resp.text.strip()
            if not text:
                break
            try:
                chunk = pd.read_csv(io.StringIO(text))
            except pd.errors.EmptyDataError:
                break
            if chunk.empty:
                break

            for col in ["time", "latitude", "longitude", "depth", "mag", "tsunami"]:
                if col not in chunk.columns:
                    chunk[col] = np.nan
            chunk = chunk[["time", "latitude", "longitude", "depth", "mag", "tsunami"]].copy()
            chunk["year"] = pd.to_datetime(chunk["time"], errors="coerce", utc=True).dt.year.astype("Int64")
            chunk = chunk.loc[chunk["year"].notna()].copy()
            frames.append(chunk)

            n = len(chunk)
            year_rows += n
            if n < limit:
                break
            offset += limit
            time.sleep(0.1)
        if year_rows > 0:
            logger.info("USGS earthquakes staged year=%d rows=%d", year, year_rows)

    if not frames:
        raise RuntimeError("USGS earthquake download returned no rows")

    df = pd.concat(frames, ignore_index=True)
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["depth"] = pd.to_numeric(df["depth"], errors="coerce")
    df["mag"] = pd.to_numeric(df["mag"], errors="coerce")
    df["tsunami"] = pd.to_numeric(df["tsunami"], errors="coerce").fillna(0.0)
    df = df.loc[
        (df["year"] >= YEAR_START)
        & np.isfinite(df["latitude"])
        & np.isfinite(df["longitude"])
    ].copy()
    df["source"] = "USGS FDSN"
    df.to_csv(out_path, index=False)
    logger.info("USGS earthquakes staged rows=%d at %s", len(df), out_path)
    return out_path


def download_ibtracs_cyclones() -> Path:
    """Download IBTrACS v04r01 and stage global track points as yearly events."""
    raw_dir = PROJECT_ROOT / "data" / "raw" / "ibtracs"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_csv = raw_dir / "ibtracs.ALL.list.v04r01.csv"
    out_path = raw_dir / "ibtracs_points.csv"

    _download_file(IBTRACS_CSV_URL, source_csv, verify_ssl=True, timeout=7200)

    usecols = ["SID", "SEASON", "ISO_TIME", "LAT", "LON", "WMO_WIND", "WMO_PRES", "USA_WIND", "USA_PRES", "BASIN", "NAME"]
    frames: list[pd.DataFrame] = []
    for chunk in pd.read_csv(source_csv, usecols=usecols, skiprows=[1], chunksize=500000, low_memory=False):
        chunk["latitude"] = pd.to_numeric(chunk["LAT"], errors="coerce")
        chunk["longitude"] = pd.to_numeric(chunk["LON"], errors="coerce")
        chunk["year"] = pd.to_datetime(chunk["ISO_TIME"], errors="coerce", utc=True).dt.year.astype("Int64")
        chunk["wind_kts"] = _coalesce_numeric(chunk, ["WMO_WIND", "USA_WIND"])
        chunk["pressure_mb"] = _coalesce_numeric(chunk, ["WMO_PRES", "USA_PRES"])
        keep = chunk[["SID", "SEASON", "BASIN", "NAME", "ISO_TIME", "year", "latitude", "longitude", "wind_kts", "pressure_mb"]].copy()
        keep = keep.loc[
            keep["year"].notna()
            & (keep["year"] >= YEAR_START)
            & np.isfinite(keep["latitude"])
            & np.isfinite(keep["longitude"])
        ]
        frames.append(keep)

    if not frames:
        raise RuntimeError("IBTrACS parse returned no rows")
    df = pd.concat(frames, ignore_index=True)
    df.loc[df["year"] < CYCLONE_PRESSURE_START_YEAR, "pressure_mb"] = np.nan
    df["source"] = "IBTrACS v04r01"
    df.to_csv(out_path, index=False)
    source_csv.unlink(missing_ok=True)
    logger.info("IBTrACS staged rows=%d at %s", len(df), out_path)
    return out_path


def _download_noaa_hazel_events(base_url: str, out_path: Path, year_min: int = 1900, year_max: int = 2025) -> pd.DataFrame:
    rows: list[dict] = []
    for year in range(int(year_min), int(year_max) + 1):
        page = 1
        yearly_rows = 0
        while True:
            params = {"minYear": year, "maxYear": year, "limit": 200, "page": page}
            js = _request_json(base_url, params=params, timeout=180)
            items = js.get("items", [])
            if not items:
                break
            rows.extend(items)
            yearly_rows += len(items)
            total_pages = int(js.get("totalPages", page) or page)
            if page >= total_pages:
                break
            page += 1
            time.sleep(0.05)
        if yearly_rows > 0:
            logger.info("NOAA HazEL staged year=%d rows=%d from %s", year, yearly_rows, base_url)

    if not rows:
        raise RuntimeError(f"No rows returned from {base_url}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    return df


def download_volcanoes_noaa(year_min: int = 1900, year_max: int = 2025) -> Path:
    out_path = PROJECT_ROOT / "data" / "raw" / "volcanoes_noaa" / "volcanoes_noaa.csv"
    df = _download_noaa_hazel_events(NOAA_VOLCANO_EVENTS_URL, out_path, year_min=year_min, year_max=year_max)
    df["year"] = pd.to_numeric(df.get("year"), errors="coerce").astype("Int64")
    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["vei"] = pd.to_numeric(df.get("vei"), errors="coerce")
    df["fatalities_total"] = pd.to_numeric(df.get("deathsTotal"), errors="coerce").fillna(0.0)
    df["damage_musd_total"] = pd.to_numeric(df.get("damageMillionsDollarsTotal"), errors="coerce").fillna(0.0)
    df = df.loc[
        df["year"].notna()
        & (df["year"] >= YEAR_START)
        & np.isfinite(df["latitude"])
        & np.isfinite(df["longitude"])
    ].copy()
    df["source"] = "NOAA HazEL Volcano Events"
    df.to_csv(out_path, index=False)
    logger.info("NOAA volcanoes staged rows=%d at %s", len(df), out_path)
    return out_path


def download_landslides_coolr() -> Path:
    raw_dir = PROJECT_ROOT / "data" / "raw" / "landslides_coolr"
    raw_dir.mkdir(parents=True, exist_ok=True)
    source_csv = raw_dir / "globallandslides.csv"
    out_path = raw_dir / "landslides_coolr.csv"

    _download_file(COOLR_LANDSLIDES_CSV_URL, source_csv, verify_ssl=True, timeout=1800)
    df = pd.read_csv(source_csv, low_memory=False)
    if df.empty:
        raise RuntimeError("COOLR landslide CSV is empty")

    df["latitude"] = pd.to_numeric(df.get("latitude"), errors="coerce")
    df["longitude"] = pd.to_numeric(df.get("longitude"), errors="coerce")
    df["fatalities"] = pd.to_numeric(df.get("fatalities"), errors="coerce").fillna(0.0)
    df["injuries"] = pd.to_numeric(df.get("injuries"), errors="coerce").fillna(0.0)
    ev_date = pd.to_datetime(df.get("ev_date"), errors="coerce")
    df["year"] = ev_date.dt.year.astype("Int64")
    df = df.loc[
        df["year"].notna()
        & (df["year"] >= YEAR_START)
        & np.isfinite(df["latitude"])
        & np.isfinite(df["longitude"])
    ].copy()
    df["source"] = "NASA COOLR (WPRDC mirror)"

    keep_cols = [
        "ev_id",
        "ev_date",
        "year",
        "latitude",
        "longitude",
        "ls_cat",
        "ls_trig",
        "ls_size",
        "fatalities",
        "injuries",
        "ctry_name",
        "ctry_code",
        "source",
    ]
    df = df[[c for c in keep_cols if c in df.columns]]
    df.to_csv(out_path, index=False)
    logger.info("COOLR landslides staged rows=%d at %s", len(df), out_path)
    return out_path


@click.command()
@click.option("--all", "run_all", is_flag=True, help="Download all configured datasets")
@click.option(
    "--dataset",
    "datasets",
    multiple=True,
    help=(
        "Subset: ucdp_ged|gdis|earthquakes_usgs|cyclones_ibtracs|"
        "volcanoes_noaa|landslides_coolr"
    ),
)
def main(run_all, datasets):
    if not run_all:
        click.echo("Specify --all")
        return

    selected = (
        list(datasets)
        if datasets
        else [
            "ucdp_ged",
            "gdis",
            "earthquakes_usgs",
            "cyclones_ibtracs",
            "volcanoes_noaa",
            "landslides_coolr",
        ]
    )
    allowed = {
        "ucdp_ged",
        "gdis",
        "earthquakes_usgs",
        "cyclones_ibtracs",
        "volcanoes_noaa",
        "landslides_coolr",
    }
    selected = [s for s in selected if s in allowed]

    for key in selected:
        try:
            if key == "ucdp_ged":
                download_ucdp_ged()
            elif key == "gdis":
                download_gdis()
            elif key == "earthquakes_usgs":
                download_earthquakes_usgs()
            elif key == "cyclones_ibtracs":
                download_ibtracs_cyclones()
            elif key == "volcanoes_noaa":
                download_volcanoes_noaa()
            elif key == "landslides_coolr":
                download_landslides_coolr()
        except Exception as e:
            logger.exception("Download failed for %s: %s", key, e)

    click.echo("\nPoint-dataset download complete.")


if __name__ == "__main__":
    main()
