"""Download SoilGrids 2.0 soil properties via gdalwarp + /vsicurl/.

Data source
-----------
Website : https://www.isric.org/explore/soilgrids
Data    : https://files.isric.org/soilgrids/latest/data/
Citation: Poggio et al. (2021), Soil 7:217-240, doi:10.5194/soil-7-217-2021
License : CC-BY 4.0

250m-resolution predictions of soil properties at 6 standard depths.
Native CRS is Interrupted Goode Homolosine (EPSG:152160).

Uses GDAL to read remote VRT files (with overviews), reproject from
Interrupted Goode Homolosine to WGS84, and resample to 0.25° resolution
in a single step. No authentication required.

Output: data/raw/soilgrids/{property}_{depth}_mean.tif (1440×720 GeoTIFF)

Usage:
    python -m src.download.soilgrids --all
    python -m src.download.soilgrids --properties bdod clay
    python -m src.download.soilgrids --all --workers 8
"""

import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import click
import yaml

from src.utils import get_logger

logger = get_logger("download.soilgrids")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "soilgrids.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "soilgrids"


def load_soilgrids_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_vrt_url(base_url: str, prop: str, depth: str, statistic: str) -> str:
    """Build the VRT URL for a given property/depth/statistic."""
    return f"{base_url}/{prop}/{prop}_{depth}_{statistic}.vrt"


def download_one(
    base_url: str,
    prop: str,
    depth: str,
    statistic: str,
    raw_dir: Path,
    overwrite: bool = False,
) -> Path | None:
    """Download and reproject a single SoilGrids layer using gdalwarp.

    Returns path to the output TIF, or None on failure.
    """
    out_name = f"{prop}_{depth}_{statistic}.tif"
    out_path = raw_dir / out_name

    if out_path.exists() and not overwrite:
        logger.info("Already exists: %s", out_name)
        return out_path

    raw_dir.mkdir(parents=True, exist_ok=True)

    vrt_url = build_vrt_url(base_url, prop, depth, statistic)
    src = f"/vsicurl/{vrt_url}"

    cmd = [
        "gdalwarp",
        "-t_srs", "EPSG:4326",
        "-tr", "0.25", "0.25",
        "-te", "-180", "-90", "180", "90",
        "-r", "average",
        "-ovr", "AUTO",
        "-overwrite",
        src,
        str(out_path),
    ]

    label = f"{prop}_{depth}"
    logger.info("Downloading %s ...", label)

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if result.returncode != 0:
            logger.error("gdalwarp failed for %s: %s", label, result.stderr.strip())
            out_path.unlink(missing_ok=True)
            return None

        logger.info("Done: %s → %s", label, out_path)
        return out_path

    except subprocess.TimeoutExpired:
        logger.error("Timeout downloading %s", label)
        out_path.unlink(missing_ok=True)
        return None
    except FileNotFoundError:
        logger.error("gdalwarp not found. Install GDAL: apt install gdal-bin / conda install gdal")
        return None


def download_soilgrids(
    properties: list[str] | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
    workers: int = 4,
) -> list[Path]:
    """Download SoilGrids layers for given properties (all depths).

    Returns list of paths to downloaded TIFs.
    """
    config = load_soilgrids_config()
    base_url = config["base_url"]
    statistic = config["statistic"]
    depths = config["depths"]
    all_props = config["properties"]

    prop_list = properties or list(all_props.keys())

    # Build task list
    tasks = []
    for prop in prop_list:
        if prop not in all_props:
            logger.warning("Unknown property: %s", prop)
            continue
        for depth in depths:
            tasks.append((prop, depth))

    logger.info("Downloading %d SoilGrids layers with %d workers", len(tasks), workers)

    downloaded = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(
                download_one, base_url, prop, depth, statistic, raw_dir, overwrite
            ): (prop, depth)
            for prop, depth in tasks
        }
        for future in as_completed(futures):
            prop, depth = futures[future]
            try:
                path = future.result()
                if path:
                    downloaded.append(path)
            except Exception as e:
                logger.error("Failed %s_%s: %s", prop, depth, e)

    logger.info("Downloaded %d / %d layers", len(downloaded), len(tasks))
    return downloaded


@click.command()
@click.option("--properties", "-p", multiple=True,
              help="Property(ies) to download (e.g. bdod clay).")
@click.option("--all", "run_all", is_flag=True, help="Download all properties.")
@click.option("--overwrite", is_flag=True, help="Re-download even if file exists.")
@click.option("--workers", "-w", type=int, default=4,
              help="Number of parallel downloads (default: 4).")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Output directory (default: {DEFAULT_RAW_DIR})")
def main(properties, run_all, overwrite, workers, raw_dir):
    """Download SoilGrids 2.0 soil properties via gdalwarp."""
    if not properties and not run_all:
        click.echo("Specify --properties or --all. Use --help for usage.")
        return

    out_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    prop_list = list(properties) if properties else None

    paths = download_soilgrids(
        properties=prop_list,
        raw_dir=out_dir,
        overwrite=overwrite,
        workers=workers,
    )
    click.echo(f"Downloaded {len(paths)} files to {out_dir}")


if __name__ == "__main__":
    main()
