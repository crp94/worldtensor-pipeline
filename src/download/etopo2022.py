"""Download ETOPO 2022 global relief model GeoTIFF.

Data source
-----------
Website : https://www.ncei.noaa.gov/products/etopo-global-relief-model
Citation: NOAA NCEI (2022). ETOPO 2022 15 Arc-Second Global Relief Model.
License : Public domain (NOAA)

Downloads the 60 arc-second bedrock elevation GeoTIFF (~370 MB).
No authentication required.

Output: data/raw/etopo2022/ETOPO_2022_v1_60s_N90W180_bed.tif

Usage:
    python -m src.download.etopo2022 --all
    python -m src.download.etopo2022 --all --overwrite
"""

import time
from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.etopo2022")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "etopo2022.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "etopo2022"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
MAX_RETRIES = 5
INITIAL_TIMEOUT = 120


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_etopo(
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> Path | None:
    """Download the ETOPO 2022 GeoTIFF. Returns path to downloaded file."""
    config = load_config()
    url = config["url"]
    filename = config["source_file"]

    raw_dir.mkdir(parents=True, exist_ok=True)
    out_path = raw_dir / filename

    if out_path.exists() and not overwrite:
        logger.info("Already downloaded: %s", out_path.name)
        return out_path

    session = requests.Session()
    session.headers.update({"User-Agent": "WorldTensor/1.0"})

    for attempt in range(1, MAX_RETRIES + 1):
        timeout = INITIAL_TIMEOUT * attempt
        try:
            logger.info("Downloading ETOPO 2022 (attempt %d/%d, timeout %ds)",
                        attempt, MAX_RETRIES, timeout)
            resp = session.get(url, stream=True, timeout=timeout,
                               allow_redirects=True)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            with (
                open(out_path, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True,
                     desc="ETOPO2022", leave=False) as pbar,
            ):
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))

            logger.info("Downloaded → %s", out_path)
            return out_path

        except requests.RequestException as e:
            logger.warning("Attempt %d failed: %s", attempt, e)
            out_path.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt * 5
                logger.info("Retrying in %ds...", wait)
                time.sleep(wait)
            else:
                logger.error("All %d attempts failed", MAX_RETRIES)
                return None

    return None


@click.command()
@click.option("--all", "run_all", is_flag=True, help="Download all data.")
@click.option("--overwrite", is_flag=True, help="Re-download even if file exists.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Output directory (default: {DEFAULT_RAW_DIR})")
def main(run_all, overwrite, raw_dir):
    """Download ETOPO 2022 global relief GeoTIFF."""
    if not run_all:
        click.echo("Specify --all. Use --help for usage.")
        return

    out_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    result = download_etopo(raw_dir=out_dir, overwrite=overwrite)
    if result:
        click.echo(f"Downloaded: {result}")
    else:
        click.echo("Download failed.")


if __name__ == "__main__":
    main()
