"""Download FLDAS dominant vegetation type classification from NASA GSFC.

Data source
-----------
Website : https://ldas.gsfc.nasa.gov/fldas/vegetation-class
Citation: McNally et al. (2017), Scientific Data 4:170012.

The FLDAS vegetation class map provides a single static global layer of
IGBP-modified MODIS 20-category land cover types used by the Noah LSM.
Resolution is 0.1° (1500×3600), covering -59.95°S to 89.95°N.

No authentication required. File is ~5 MB.

Usage:
    python -m src.download.fldas_vegclass
    python -m src.download.fldas_vegclass --overwrite
"""

from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.fldas_vegclass")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "fldas_vegclass.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fldas_vegclass"
CHUNK_SIZE = 1024 * 1024


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_vegclass(output_dir: Path = DEFAULT_RAW_DIR,
                      overwrite: bool = False) -> Path:
    """Download the FLDAS vegetation class NetCDF.

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
    config = load_config()
    url = config["source_url"]
    filename = config["source_file"]

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / filename

    if dest.exists() and not overwrite:
        logger.info("Already exists: %s", dest)
        return dest

    logger.info("Downloading %s", url)
    resp = requests.get(url, stream=True, timeout=60)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True,
             desc="FLDAS vegclass", leave=False) as pbar,
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
    """Download FLDAS dominant vegetation type classification."""
    out = Path(output_dir) if output_dir else DEFAULT_RAW_DIR
    path = download_vegclass(out, overwrite=overwrite)
    click.echo(f"File: {path}")


if __name__ == "__main__":
    main()
