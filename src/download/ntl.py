"""Download Harmonized NTL (1992-2018) from Figshare.

Usage:
    python -m src.download.ntl
"""

from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.ntl")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "ntl.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "ntl"
FIGSHARE_API_BASE = "https://api.figshare.com/v2/articles"


def load_ntl_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_ntl(
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> list[Path]:
    """Download NTL annual GeoTIFFs from Figshare API."""
    config = load_ntl_config()
    article_id = config["figshare_id"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    # 1. Get file list from Figshare (request more than default 10 files)
    url = f"{FIGSHARE_API_BASE}/{article_id}/files?page_size=100"
    resp = requests.get(url)
    resp.raise_for_status()
    files = resp.json()
    
    downloaded = []
    
    # 2. Filter and download each .tif file
    # Pattern: Harmonized_DN_NTL_{year}_*.tif
    for f_info in tqdm(files, desc="Downloading NTL"):
        fname = f_info["name"]
        if not fname.endswith(".tif") or "Harmonized_DN_NTL" not in fname:
            continue
            
        dest_path = raw_dir / fname
        if dest_path.exists() and not overwrite:
            downloaded.append(dest_path)
            continue
            
        # Download
        download_url = f_info["download_url"]
        try:
            f_resp = requests.get(download_url, stream=True)
            f_resp.raise_for_status()
            
            with open(dest_path, "wb") as f:
                for chunk in f_resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            downloaded.append(dest_path)
        except Exception as e:
            logger.error("Failed to download %s: %s", fname, e)

    return downloaded


@click.command()
@click.option("--overwrite", is_flag=True, help="Re-download existing files.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Output directory (default: {DEFAULT_RAW_DIR})")
def main(overwrite, raw_dir):
    """Download Harmonized NTL gridded estimates."""
    out_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    paths = download_ntl(raw_dir=out_dir, overwrite=overwrite)
    if paths:
        click.echo(f"Successfully downloaded {len(paths)} files to {out_dir}")
    else:
        click.echo("Failed to download NTL data.")


if __name__ == "__main__":
    main()
