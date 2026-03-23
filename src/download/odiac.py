"""Download ODIAC2024 fossil fuel CO2 emissions data (1° NetCDF product).

Data source
-----------
Oda, T. and Maksyutov, S. (2015).
ODIAC Fossil Fuel CO2 Emissions Dataset (Version ODIAC2024).
NIES, DOI: 10.17595/20170411.001

License : CC-BY 4.0
Format  : Yearly NetCDF files, each containing 12 monthly layers at 1° × 1°.
Size    : ~6 MB per year, ~143 MB total (2000–2023).

Variables inside each file
--------------------------
- land        : fossil fuel CO2 from combustion, cement, gas flaring [gC/m²/d]
- intl_bunker : international aviation/marine bunker CO2              [gC/m²/d]

Usage
-----
    python -m src.download.odiac
    python -m src.download.odiac --overwrite
    python -m src.download.odiac --years 2020 2021 2022
"""

from __future__ import annotations

from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.odiac")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "odiac.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "odiac"
CHUNK_SIZE = 1024 * 1024  # 1 MB


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def download_odiac(
    years: list[int] | None = None,
    overwrite: bool = False,
) -> list[Path]:
    """Download ODIAC 1° NetCDF files from NIES.

    Parameters
    ----------
    years : list of int, optional
        Specific years to download.  Defaults to the full range in config.
    overwrite : bool
        Re-download even if the file already exists.

    Returns
    -------
    list of Path — downloaded file paths.
    """
    config = load_config()
    base_url = config["base_url"]
    pattern = config["file_pattern"]
    y0, y1 = config["temporal_range"]

    if years is None:
        years = list(range(y0, y1 + 1))

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []

    for year in years:
        fname = pattern.format(year=year)
        out_path = RAW_DIR / fname
        url = f"{base_url}/{fname}"

        if out_path.exists() and not overwrite:
            logger.info("Skipping (exists): %s", out_path.name)
            downloaded.append(out_path)
            continue

        tmp_path = out_path.with_suffix(".partial")
        logger.info("Downloading %s ...", fname)
        try:
            resp = requests.get(url, stream=True, timeout=120)
            resp.raise_for_status()
            total = int(resp.headers.get("content-length", 0))
            with (
                open(tmp_path, "wb") as f,
                tqdm(total=total, unit="B", unit_scale=True, desc=fname) as pbar,
            ):
                for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue
                    f.write(chunk)
                    pbar.update(len(chunk))
            tmp_path.replace(out_path)
            downloaded.append(out_path)
            logger.info("Downloaded -> %s", out_path)
        except Exception as e:
            logger.error("Download failed for %s: %s", fname, e)
            tmp_path.unlink(missing_ok=True)

    logger.info("Downloaded %d / %d files", len(downloaded), len(years))
    return downloaded


@click.command()
@click.option("--years", "-y", multiple=True, type=int, help="Specific year(s).")
@click.option("--overwrite", is_flag=True, help="Re-download existing files.")
def main(years: tuple[int, ...], overwrite: bool) -> None:
    """Download ODIAC2024 1° NetCDF files from NIES (CC-BY 4.0)."""
    year_list = list(years) if years else None
    download_odiac(years=year_list, overwrite=overwrite)


if __name__ == "__main__":
    main()
