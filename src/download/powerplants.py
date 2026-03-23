"""Download WRI Global Power Plant Database.

Usage:
    python -m src.download.powerplants --all
"""

import zipfile
from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.powerplants")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "powerplants.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "powerplants"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_powerplants(raw_dir: Path = DEFAULT_RAW_DIR, overwrite: bool = False):
    config = load_config()
    url = config["source_url"]
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "gppd.zip"
    
    csv_file = raw_dir / "global_power_plant_database.csv"
    if csv_file.exists() and not overwrite:
        logger.info("Power plant database CSV already exists.")
        return

    logger.info("Downloading WRI Power Plant Database ZIP from %s", url)
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()
        
        total = int(resp.headers.get("content-length", 0))
        with (
            open(zip_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc="GPPD ZIP") as pbar
        ):
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
                pbar.update(len(chunk))
                
        logger.info("Extracting ZIP...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
            
        zip_path.unlink()
        logger.info("GPPD extraction complete.")
        
    except Exception as e:
        logger.error("Failed to download/extract GPPD: %s", e)
        if zip_path.exists(): zip_path.unlink()


@click.command()
@click.option("--all", "run_all", is_flag=True)
@click.option("--overwrite", is_flag=True)
def main(run_all, overwrite):
    if not run_all:
        click.echo("Specify --all")
        return
    download_powerplants(overwrite=overwrite)


if __name__ == "__main__":
    main()
