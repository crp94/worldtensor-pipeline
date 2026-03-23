"""Download SectGDP30 gridded estimates (tar.gz archive of GeoTIFFs).

Downloads a single tar.gz, extracts GeoTIFFs, deletes archive.

Usage:
    python -m src.download.sectgdp
"""

import tarfile
from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.sectgdp")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "sectgdp.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "sectgdp"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def load_sectgdp_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_sectgdp(
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> list[Path]:
    """Download and extract SectGDP30 tar.gz.

    Returns list of extracted GeoTIFF paths for anchor years.
    """
    config = load_sectgdp_config()
    url = config["source_url"]
    anchor_years = config["anchor_years"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    # Check if all files for anchor years exist
    all_exist = True
    expected_files = []
    for var_info in config["variables"].values():
        for year in anchor_years:
            fname = var_info["filename_pattern"].format(year=year)
            p = raw_dir / fname
            expected_files.append(p)
            if not p.exists():
                all_exist = False

    if all_exist and not overwrite:
        logger.info("SectGDP30 raw files already exist in %s", raw_dir)
        return expected_files

    archive_path = raw_dir / "SectGDP30_v2.tar.gz"

    try:
        logger.info("Downloading SectGDP30 archive from %s", url)
        resp = requests.get(url, stream=True, timeout=600)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        with (
            open(archive_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True,
                 desc="SectGDP30 Archive") as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))

        # Extract files from tar.gz
        with tarfile.open(archive_path, "r:gz") as tar:
            logger.info("Extracting archive contents...")
            tar.extractall(raw_dir)

        archive_path.unlink(missing_ok=True)
        
        # Flatten structure if needed
        for p in raw_dir.rglob("*.tif"):
            if p.parent != raw_dir:
                target = raw_dir / p.name
                p.rename(target)

        return expected_files

    except Exception as e:
        logger.error("Download failed: %s", e)
        archive_path.unlink(missing_ok=True)
        return []


@click.command()
@click.option("--overwrite", is_flag=True, help="Re-download even if files exist.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Output directory (default: {DEFAULT_RAW_DIR})")
def main(overwrite, raw_dir):
    """Download SectGDP30 gridded estimates."""
    out_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    paths = download_sectgdp(raw_dir=out_dir, overwrite=overwrite)
    if paths:
        click.echo(f"Successfully downloaded/extracted files to {out_dir}")
    else:
        click.echo("Failed to download SectGDP30 data.")


if __name__ == "__main__":
    main()
