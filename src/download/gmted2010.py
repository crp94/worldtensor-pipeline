"""Download GMTED2010 elevation grids and derive slope.

Data source
-----------
Website : https://www.usgs.gov/coastal-changes-and-impacts/gmted2010
Download: https://topotools.cr.usgs.gov/gmted_viewer/gmted2010_global_grids.php
Citation: Danielson & Gesch (2011), USGS Open-File Report 2011-1073
License : Public domain (USGS)

GMTED2010 provides 30 arc-second (~1 km) global terrain elevation grids
in ESRI ArcGrid format. This script downloads mean elevation and standard
deviation grids, then derives a slope grid using `gdaldem slope`.

No authentication required.

Output: data/raw/gmted2010/{mn30_grd, sd30_grd, slope_from_mn30.tif}

Usage:
    python -m src.download.gmted2010 --all
    python -m src.download.gmted2010 --all --overwrite
"""

import subprocess
import time
import zipfile
from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.gmted2010")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "gmted2010.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "gmted2010"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
MAX_RETRIES = 5
INITIAL_TIMEOUT = 120


def load_gmted_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _download_zip(url: str, dest_dir: Path, label: str,
                  overwrite: bool = False) -> Path | None:
    """Download and extract a GMTED2010 ArcGrid zip. Returns path to extracted grid dir."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{label}.zip"

    # The ArcGrid is a directory (e.g. mn30_grd/) inside the zip
    grid_dir = dest_dir / label
    if grid_dir.exists() and not overwrite:
        logger.info("Already extracted: %s", label)
        return grid_dir

    session = requests.Session()
    session.headers.update({"User-Agent": "WorldTensor/1.0"})

    for attempt in range(1, MAX_RETRIES + 1):
        timeout = INITIAL_TIMEOUT * attempt
        try:
            logger.info("Downloading %s (attempt %d/%d, timeout %ds)",
                        label, attempt, MAX_RETRIES, timeout)
            resp = session.get(url, stream=True, timeout=timeout,
                               allow_redirects=True)
            resp.raise_for_status()

            total = int(resp.headers.get("content-length", 0))
            with (
                open(zip_path, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True,
                     desc=label, leave=False) as pbar,
            ):
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    f.write(chunk)
                    pbar.update(len(chunk))

            logger.info("Extracting %s ...", zip_path.name)
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(dest_dir)

            zip_path.unlink(missing_ok=True)

            if grid_dir.exists():
                logger.info("Extracted ArcGrid: %s", grid_dir)
                return grid_dir
            else:
                logger.error("Expected grid directory not found: %s", grid_dir)
                return None

        except (requests.RequestException, zipfile.BadZipFile) as e:
            logger.warning("Attempt %d failed for %s: %s", attempt, label, e)
            zip_path.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                wait = 2 ** attempt * 5
                logger.info("Retrying in %ds...", wait)
                time.sleep(wait)
            else:
                logger.error("All %d attempts failed for %s", MAX_RETRIES, label)
                return None

    return None


def derive_slope(elevation_grid: Path, raw_dir: Path,
                 overwrite: bool = False) -> Path | None:
    """Derive slope (degrees) from an elevation ArcGrid using gdaldem.

    Runs at native 30" resolution for accuracy, then the processing step
    resamples to 0.25°.
    """
    out_path = raw_dir / "slope_from_mn30.tif"

    if out_path.exists() and not overwrite:
        logger.info("Slope already derived: %s", out_path.name)
        return out_path

    # gdaldem slope needs scale factor for geographic coords (meters per degree)
    # 111120 is the standard conversion for lat/lon in meters
    cmd = [
        "gdaldem", "slope",
        str(elevation_grid),
        str(out_path),
        "-compute_edges",
        "-s", "111120",
        "-of", "GTiff",
        "-co", "COMPRESS=LZW",
    ]

    logger.info("Deriving slope from %s ...", elevation_grid.name)
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        logger.error("gdaldem slope failed: %s", result.stderr.strip())
        out_path.unlink(missing_ok=True)
        return None

    logger.info("Slope derived → %s", out_path)
    return out_path


def download_gmted(
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download GMTED2010 grids and derive slope.

    Returns dict mapping variable code to path of raw data.
    """
    config = load_gmted_config()
    base_url = config["base_url"]
    variables = config["variables"]
    results = {}

    # Download elevation grids
    for var_code, var_info in variables.items():
        if "zip_name" not in var_info:
            continue  # derived variable (slope)

        zip_name = var_info["zip_name"]
        grid_name = var_info["grid_file"]
        url = f"{base_url}/{zip_name}"

        grid_path = _download_zip(url, raw_dir, grid_name, overwrite)
        if grid_path:
            results[var_code] = grid_path

    # Derive slope from mean elevation
    if "elevation_mean" in results:
        slope_path = derive_slope(results["elevation_mean"], raw_dir, overwrite)
        if slope_path:
            results["slope_mean"] = slope_path

    return results


@click.command()
@click.option("--all", "run_all", is_flag=True, help="Download all data.")
@click.option("--overwrite", is_flag=True, help="Re-download even if files exist.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Output directory (default: {DEFAULT_RAW_DIR})")
def main(run_all, overwrite, raw_dir):
    """Download GMTED2010 elevation grids and derive slope."""
    if not run_all:
        click.echo("Specify --all. Use --help for usage.")
        return

    out_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    results = download_gmted(raw_dir=out_dir, overwrite=overwrite)
    click.echo(f"Downloaded/derived {len(results)} layers to {out_dir}")
    for var, path in results.items():
        click.echo(f"  {var}: {path}")


if __name__ == "__main__":
    main()
