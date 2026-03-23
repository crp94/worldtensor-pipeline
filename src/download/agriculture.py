"""Download GGCP10 Agriculture yearly production ZIP from Harvard Dataverse.

Usage:
    python -m src.download.agriculture --all
"""

import zipfile
import shutil
from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.agriculture")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "agriculture.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "agriculture"


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_agriculture(raw_dir: Path = DEFAULT_RAW_DIR, overwrite: bool = False):
    config = load_config()
    url = config["source_url"]
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    zip_path = raw_dir / "GGCP10_all.zip"
    
    # Check for any .tif file as proxy for completion
    if any(raw_dir.glob("*.tif")) and not overwrite:
        logger.info("Agriculture GeoTIFF files already exist.")
        return

    logger.info("Downloading GGCP10 Agriculture Dataset (ZIP) from %s", url)
    try:
        # We must use allow_redirects=True for Dataverse
        resp = requests.get(url, stream=True, timeout=600, allow_redirects=True)
        resp.raise_for_status()
        
        total = int(resp.headers.get("content-length", 0))
        with (
            open(zip_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc="GGCP10 ZIP") as pbar
        ):
            for chunk in resp.iter_content(chunk_size=1024*1024):
                f.write(chunk)
                pbar.update(len(chunk))
                
        logger.info("Extracting Agriculture ZIP...")
        # Note: This ZIP might contain other ZIPs (Maize.zip, etc.) 
        # or direct TIFs. We handle extraction recursively if needed.
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(raw_dir)
            
        # Check if it extracted nested ZIPs
        for nested_zip in raw_dir.glob("*.zip"):
            if nested_zip.name == "GGCP10_all.zip": continue
            logger.info("Extracting nested ZIP: %s", nested_zip.name)
            with zipfile.ZipFile(nested_zip, "r") as nzf:
                nzf.extractall(raw_dir)
            nested_zip.unlink()
            
        zip_path.unlink()
        logger.info("Agriculture extraction complete.")
        
    except Exception as e:
        logger.error("Failed to download/extract Agriculture data: %s", e)
        if zip_path.exists(): zip_path.unlink()


@click.command()
@click.option("--all", "run_all", is_flag=True)
@click.option("--overwrite", is_flag=True)
def main(run_all, overwrite):
    if not run_all:
        click.echo("Specify --all")
        return
    download_agriculture(overwrite=overwrite)


if __name__ == "__main__":
    main()
