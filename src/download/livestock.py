"""Download AGLW Livestock yearly density ZIP from Zenodo.

Usage:
    python -m src.download.livestock --all
"""

import zipfile
from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.livestock")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "livestock.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "livestock"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_livestock(raw_dir: Path = DEFAULT_RAW_DIR, overwrite: bool = False):
    config = load_config()
    url = config["source_url"]
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "AGLW_1961_2021.zip"
    
    # Check for any .nc file as proxy for completion
    if any(raw_dir.glob("*.nc")) and not overwrite:
        logger.info("Livestock NetCDF files already exist.")
        return

    logger.info("Downloading AGLW Livestock ZIP (5.4 GB)...")
    try:
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()
        
        total = int(resp.headers.get("content-length", 0))
        with (
            open(zip_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc="AGLW ZIP") as pbar
        ):
            for chunk in resp.iter_content(chunk_size=1024*1024):
                f.write(chunk)
                pbar.update(len(chunk))
                
        logger.info("Extracting large Livestock ZIP...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
            
        zip_path.unlink()
        logger.info("Livestock extraction complete.")
        
    except Exception as e:
        logger.error("Failed to download/extract Livestock data: %s", e)
        if zip_path.exists(): zip_path.unlink()


@click.command()
@click.option("--all", "run_all", is_flag=True)
@click.option("--overwrite", is_flag=True)
def main(run_all, overwrite):
    if not run_all:
        click.echo("Specify --all")
        return
    download_livestock(overwrite=overwrite)


if __name__ == "__main__":
    main()
