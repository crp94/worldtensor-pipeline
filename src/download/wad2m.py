"""Download WAD2M 0.25° monthly inundation fraction from Zenodo.
"""

import os
from pathlib import Path
import requests
from tqdm import tqdm
from src.utils import get_logger

logger = get_logger("download.wad2m")

RAW_DIR = Path("data/raw/wad2m")
URL = "https://zenodo.org/records/5553187/files/WAD2M_wetlands_2000-2020_025deg_Ver2.0.nc.zip"

def download_wad2m():
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RAW_DIR / "WAD2M_wetlands_2000-2020_025deg_Ver2.0.nc.zip"
    
    if out_path.exists():
        logger.info(f"File already exists: {out_path}")
        return
        
    logger.info("Downloading WAD2M (0.25° version)...")
    try:
        response = requests.get(URL, stream=True)
        response.raise_for_status()
        
        total_size = int(response.headers.get('content-length', 0))
        with open(out_path, 'wb') as f, tqdm(
            total=total_size, unit='iB', unit_scale=True, desc="WAD2M_025"
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
        logger.info(f"Successfully downloaded to {out_path}")
    except Exception as e:
        logger.error(f"Failed to download WAD2M: {e}")
        if out_path.exists():
            out_path.unlink()

if __name__ == "__main__":
    download_wad2m()
