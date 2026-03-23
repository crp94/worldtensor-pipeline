"""Download distance-to-coastline from NOAA ERDDAP at 0.25° resolution.

Data source
-----------
Producer : NASA Goddard Space Flight Center (GSFC), Ocean Biology Processing
           Group (OBPG)
ERDDAP   : https://upwell.pfeg.noaa.gov/erddap/griddap/dist2coast_1deg.html
Docs     : https://oceancolor.gsfc.nasa.gov/resources/docs/distfromcoast/
Citation : NASA OBPG (2012). Distance to nearest coastline.

Dataset description
-------------------
A global grid of signed distances (km) to the nearest coastline, generated
using the Generic Mapping Tools (GMT) intermediate-resolution coastline.

Sign convention:
    positive values = over ocean (distance to nearest land)
    negative values = over land  (distance to nearest coast)

Landlocked water bodies (e.g. Caspian Sea) are treated as land (negative).
Uncertainty is up to ~1 km at any given point.

The native resolution is 0.01° (36000 x 18000 cells, ~2.6 GB). To avoid
downloading the full file, we use ERDDAP's server-side stride parameter
(stride=25) to request a pre-subsampled 0.25° grid (~2 MB). This is
equivalent to point-sampling every 25th cell — acceptable for a smooth
distance field.

No authentication required.

Usage:
    python -m src.download.dist2coast
    python -m src.download.dist2coast --overwrite
"""

from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.dist2coast")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "dist2coast.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "dist2coast"
CHUNK_SIZE = 1024 * 1024  # 1 MB


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_dist2coast(output_dir: Path = DEFAULT_RAW_DIR,
                        overwrite: bool = False) -> Path:
    """Download the distance-to-coast NetCDF via ERDDAP stride.

    ERDDAP's stride parameter (e.g. [start:stride:stop]) tells the server
    to return every Nth point, effectively subsampling the 0.01° grid to
    0.25° without downloading the full dataset.

    Parameters
    ----------
    output_dir : Path
        Directory to save the file.
    overwrite : bool
        Re-download even if file already exists.

    Returns
    -------
    Path
        Path to the downloaded NetCDF file.
    """
    config = load_config()
    url = config["source_url"]
    filename = config["source_file"]

    output_dir.mkdir(parents=True, exist_ok=True)
    dest = output_dir / filename

    if dest.exists() and not overwrite:
        logger.info("Already exists: %s", dest)
        return dest

    logger.info("Downloading from ERDDAP (0.25° stride) ...")
    resp = requests.get(url, stream=True, timeout=120, allow_redirects=True)
    resp.raise_for_status()

    total = int(resp.headers.get("content-length", 0))
    with (
        open(dest, "wb") as f,
        tqdm(total=total, unit="B", unit_scale=True,
             desc="dist2coast", leave=False) as pbar,
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
    """Download distance-to-coastline from NOAA ERDDAP."""
    out = Path(output_dir) if output_dir else DEFAULT_RAW_DIR
    path = download_dist2coast(out, overwrite=overwrite)
    click.echo(f"File: {path}")


if __name__ == "__main__":
    main()
