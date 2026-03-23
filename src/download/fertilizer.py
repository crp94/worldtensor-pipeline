"""Download the Global Crop-Specific Fertilization Dataset (1961–2019).

Source
------
Decorte, T., Janssens, I., Coello Sanz, F., & Mortier, S. (2024).
Fertilizer application rate maps per crop and year.
Figshare. Dataset. https://doi.org/10.6084/m9.figshare.25435432.v3

Companion paper
---------------
Coello Sanz, F. et al. (2024). Global Crop-Specific Fertilization Dataset
from 1961–2019. Nature Scientific Data.
https://doi.org/10.1038/s41597-024-04030-4

License : CC0 (Creative Commons Public Domain Dedication)
Format  : ZIP archive containing per-crop, per-year GeoTIFF rasters at
          5-arcminute (~10 km) resolution.
Size    : ~5.7 GB (compressed)

Contents of the archive (Cropland_Maps.zip)
-------------------------------------------
The ZIP contains directories for each of the 13 crop groups.  Within each
directory are yearly GeoTIFF rasters with fertilizer application rates
(kg ha⁻¹) for three nutrients: N (nitrogen), P₂O₅ (phosphorus pentoxide),
and K₂O (potassium oxide).

Usage
-----
    python -m src.download.fertilizer
    python -m src.download.fertilizer --overwrite
"""

from __future__ import annotations

from pathlib import Path

import click
import requests
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.fertilizer")

FIGSHARE_URL = "https://ndownloader.figshare.com/files/50123172"
RAW_DIR = Path("data/raw/fertilizer")
ZIP_NAME = "Cropland_Maps.zip"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def download_fertilizer(overwrite: bool = False) -> Path:
    """Download the Coello et al. (2024) fertilizer dataset from Figshare.

    Returns the path to the downloaded ZIP file.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    zip_path = RAW_DIR / ZIP_NAME

    if zip_path.exists() and not overwrite:
        logger.info("Skipping download (exists): %s", zip_path)
        return zip_path

    tmp_path = zip_path.with_suffix(".partial")
    logger.info("Downloading fertilizer dataset (~5.7 GB) from Figshare...")
    resp = requests.get(FIGSHARE_URL, stream=True, timeout=600)
    resp.raise_for_status()
    total = int(resp.headers.get("content-length", 0))

    with (
        open(tmp_path, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True, desc="fertilizer") as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            if not chunk:
                continue
            f.write(chunk)
            pbar.update(len(chunk))

    tmp_path.replace(zip_path)
    logger.info("Downloaded -> %s", zip_path)
    return zip_path


def extract_fertilizer(overwrite: bool = False) -> Path:
    """Extract the ZIP archive into data/raw/fertilizer/.

    Returns the extraction directory.
    """
    import zipfile

    zip_path = RAW_DIR / ZIP_NAME
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP not found: {zip_path}. Run download first.")

    extract_dir = RAW_DIR / "extracted"
    if extract_dir.exists() and not overwrite and (any(extract_dir.rglob("*.tif")) or any(extract_dir.rglob("*.tiff"))):
        logger.info("Skipping extraction (already extracted): %s", extract_dir)
        return extract_dir

    extract_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Extracting %s -> %s", zip_path.name, extract_dir)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    n_tifs = len(list(extract_dir.rglob("*.tif")))
    logger.info("Extraction complete: %d TIF files", n_tifs)
    return extract_dir


@click.command()
@click.option("--overwrite", is_flag=True, help="Re-download and re-extract.")
def main(overwrite: bool) -> None:
    """Download and extract the Coello et al. fertilizer dataset."""
    download_fertilizer(overwrite=overwrite)
    extract_fertilizer(overwrite=overwrite)


if __name__ == "__main__":
    main()
