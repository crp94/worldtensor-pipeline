"""Download HydroRIVERS v1.0 global river shapefile.

Data source
-----------
Website : https://www.hydrosheds.org/products/hydrorivers
Citation: Lehner & Grill (2013), Hydrol. Process. 27(15):2171-2186.
License : Free for non-commercial use

Downloads the global river centerline shapefile (~700 MB zipped).
No authentication required.

Output: data/raw/hydrorivers/

Usage:
    python -m src.download.hydrorivers --all
    python -m src.download.hydrorivers --all --overwrite
"""

import time
import zipfile
from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.hydrorivers")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "hydrorivers.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "hydrorivers"
CHUNK_SIZE = 8 * 1024 * 1024
MAX_RETRIES = 5
INITIAL_TIMEOUT = 180


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_hydrorivers(
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> Path | None:
    """Download and extract HydroRIVERS shapefile."""
    config = load_config()
    url = config["url"]
    zip_name = config["zip_name"]
    shapefile = config["shapefile"]

    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / zip_name
    shp_path = raw_dir / shapefile

    if shp_path.exists() and not overwrite:
        logger.info("Already extracted: %s", shp_path.name)
        return shp_path

    session = requests.Session()
    session.headers.update({"User-Agent": "WorldTensor/1.0"})

    for attempt in range(1, MAX_RETRIES + 1):
        timeout = INITIAL_TIMEOUT * attempt
        try:
            logger.info("Downloading HydroRIVERS (attempt %d/%d, timeout %ds)",
                        attempt, MAX_RETRIES, timeout)
            resp = session.get(url, stream=True, timeout=timeout,
                               allow_redirects=True)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            with (
                open(zip_path, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True,
                     desc="HydroRIVERS", leave=False) as pbar,
            ):
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))

            logger.info("Extracting %s ...", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(raw_dir)

            zip_path.unlink(missing_ok=True)

            if shp_path.exists():
                logger.info("Extracted: %s", shp_path)
                return shp_path
            else:
                # Search for any .shp file
                shps = list(raw_dir.rglob("*.shp"))
                if shps:
                    logger.info("Extracted: %s", shps[0])
                    return shps[0]
                logger.error("No shapefile found after extraction")
                return None

        except (requests.RequestException, zipfile.BadZipFile) as e:
            logger.warning("Attempt %d failed: %s", attempt, e)
            zip_path.unlink(missing_ok=True)
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
@click.option("--overwrite", is_flag=True, help="Re-download even if files exist.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Output directory (default: {DEFAULT_RAW_DIR})")
def main(run_all, overwrite, raw_dir):
    """Download HydroRIVERS v1.0 global river shapefile."""
    if not run_all:
        click.echo("Specify --all. Use --help for usage.")
        return

    out_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    result = download_hydrorivers(raw_dir=out_dir, overwrite=overwrite)
    if result:
        click.echo(f"Downloaded: {result}")
    else:
        click.echo("Download failed.")


if __name__ == "__main__":
    main()
