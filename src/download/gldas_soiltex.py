"""Download GLDAS/Noah soil texture classification from NASA GSFC.

The GLDAS soil texture map provides a single static global layer of
FAO 16-category soil texture types used by the Noah land surface model.
It is a hybrid of:
  - 30-second STATSGO over CONUS (continental US)
  - 5-minute FAO Soil Map of the World elsewhere

The file is a small (~150 KB) NetCDF4 at 0.25° resolution, covering
-59.875°S to 89.875°N (no Antarctica). Each cell contains an integer
class ID (1–16) corresponding to a FAO soil texture type:

    1  = Sand              9  = Clay loam
    2  = Loamy sand       10  = Sandy clay
    3  = Sandy loam       11  = Silty clay
    4  = Silt loam        12  = Clay
    5  = Silt (absent)    13  = Organic materials
    6  = Loam             14  = (unused)
    7  = Sandy clay loam  15  = Bedrock
    8  = Silty clay loam  16  = Other

Source: https://ldas.gsfc.nasa.gov/gldas/soils
No authentication required.

Usage:
    python -m src.download.gldas_soiltex
    python -m src.download.gldas_soiltex --overwrite
"""

from pathlib import Path

import click
import requests
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.gldas_soiltex")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "gldas_soils"
SOURCE_URL = "https://ldas.gsfc.nasa.gov/sites/default/files/ldas/gldas/SOILS/GLDASp4_soiltexture_025d.nc4"
FILENAME = "GLDASp4_soiltexture_025d.nc4"
CHUNK_SIZE = 1024 * 1024  # 1 MB (file is small)


def download_soiltex(output_dir: Path = DEFAULT_RAW_DIR,
                     overwrite: bool = False) -> Path:
    """Download the GLDAS soil texture NetCDF.

    Parameters
    ----------
    output_dir : Path
        Directory to save the file.
    overwrite : bool
        Re-download even if file already exists.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / FILENAME

    if dest.exists() and not overwrite:
        logger.info("Already exists: %s", dest)
        return dest

    logger.info("Downloading %s", SOURCE_URL)
    resp = requests.get(SOURCE_URL, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True,
             desc="GLDAS soiltex", leave=False) as pbar,
    ):
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            pbar.update(len(chunk))

    logger.info("Downloaded: %s (%d bytes)", dest, dest.stat().st_size)
    return dest


@click.command()
@click.option("--output-dir", type=click.Path(), default=None,
              help=f"Output directory (default: {DEFAULT_RAW_DIR})")
@click.option("--overwrite", is_flag=True, help="Re-download even if file exists.")
def main(output_dir, overwrite):
    """Download GLDAS/Noah soil texture classification."""
    out = Path(output_dir) if output_dir else DEFAULT_RAW_DIR
    path = download_soiltex(out, overwrite=overwrite)
    click.echo(f"File: {path}")


if __name__ == "__main__":
    main()
