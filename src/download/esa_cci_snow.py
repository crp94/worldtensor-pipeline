"""Download ESA Snow CCI daily SWE files from CEDA.

Source catalogue:
https://catalogue.ceda.ac.uk/uuid/edf8abd23f4a40aabd4d52e48dec06ea/

Archive root:
https://dap.ceda.ac.uk/neodc/esacci/snow/data/swe/MERGED/v4.0/

Output structure:
data/raw/esa_cci_snow/{year}/{month}/{daily_file}.nc
"""

from __future__ import annotations

import os
import re
import time
from pathlib import Path
from urllib.parse import urljoin

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger
from src.year_policy import resolve_year_list

logger = get_logger("download.esa_cci_snow")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "esa_cci_snow.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "esa_cci_snow"

TIMEOUT = 120
CHUNK_SIZE = 4 * 1024 * 1024
MAX_RETRIES = 3


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _get_auth() -> tuple[str, str] | None:
    # Dataset is currently public, but keep auth support for protected mirrors.
    user = os.environ.get("CEDA_USER")
    passwd = os.environ.get("CEDA_PASS")
    if user and passwd:
        return user, passwd
    return None


def _new_session() -> requests.Session:
    s = requests.Session()
    s.headers.update({"User-Agent": "worldtensor/esa-cci-snow"})
    auth = _get_auth()
    if auth:
        s.auth = auth
    return s


def _extract_links(html_text: str) -> list[str]:
    return re.findall(r'href="([^"]+)"', html_text)


def _list_months(session: requests.Session, year_url: str) -> list[str]:
    r = session.get(year_url, timeout=TIMEOUT)
    r.raise_for_status()
    out = []
    for href in _extract_links(r.text):
        m = href.strip().strip("/")
        if re.fullmatch(r"\d{2}", m):
            out.append(m)
    return sorted(set(out))


def _list_nc_files(session: requests.Session, month_url: str) -> list[str]:
    r = session.get(month_url, timeout=TIMEOUT)
    r.raise_for_status()
    out = []
    for href in _extract_links(r.text):
        name = href.strip()
        if name.endswith(".nc") and "/" not in name:
            out.append(name)
    return sorted(set(out))


def _download_file(session: requests.Session, url: str, out_path: Path, overwrite: bool) -> bool:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return False

    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with session.get(url, stream=True, timeout=TIMEOUT) as r:
                r.raise_for_status()
                with open(tmp_path, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        if chunk:
                            f.write(chunk)
            tmp_path.replace(out_path)
            return True
        except Exception:
            if tmp_path.exists():
                tmp_path.unlink(missing_ok=True)
            if attempt < MAX_RETRIES:
                time.sleep(2 ** attempt)
                continue
            raise


def download_year(year: int, raw_dir: Path, config: dict, overwrite: bool = False) -> int:
    base_url = str(config["base_url"]).rstrip("/") + "/"
    year_url = urljoin(base_url, f"{year}/")

    session = _new_session()
    downloaded = 0

    try:
        months = _list_months(session, year_url)
    except Exception as e:
        logger.warning("Year %d listing failed (%s); skipping.", year, e)
        return 0

    for month in months:
        month_url = urljoin(year_url, f"{month}/")
        try:
            nc_files = _list_nc_files(session, month_url)
        except Exception as e:
            logger.warning("Month listing failed year=%d month=%s (%s)", year, month, e)
            continue

        for fname in tqdm(nc_files, desc=f"ESA Snow {year}-{month}", leave=False):
            file_url = urljoin(month_url, fname)
            out_path = raw_dir / str(year) / month / fname
            try:
                if _download_file(session, file_url, out_path, overwrite=overwrite):
                    downloaded += 1
            except Exception as e:
                logger.warning("Download failed: %s (%s)", file_url, e)

    return downloaded


def download_esa_cci_snow(
    years: list[int] | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> int:
    config = load_config()
    y0, y1 = [int(v) for v in config["temporal_range"]]
    if years is None:
        years = resolve_year_list(
            start_year=y0,
            end_year=y1,
            default_start=y0,
            default_end=y1,
            label="ESA CCI Snow download years",
        )

    years = resolve_year_list(
        years,
        default_start=y0,
        default_end=y1,
        label="ESA CCI Snow download years",
    )
    raw_dir.mkdir(parents=True, exist_ok=True)

    total = 0
    for year in tqdm(years, desc="ESA CCI Snow years"):
        n = download_year(year, raw_dir=raw_dir, config=config, overwrite=overwrite)
        total += n
        if n > 0:
            logger.info("Year %d downloaded files: %d", year, n)
    return total


@click.command()
@click.option("--all", "run_all", is_flag=True, help="Download all years in config.")
@click.option("--years", "-y", multiple=True, type=int, help="Specific years.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--overwrite", is_flag=True, help="Re-download existing files.")
@click.option("--raw-dir", type=click.Path(), default=None, help=f"Output directory (default: {DEFAULT_RAW_DIR})")
def main(run_all, years, start_year, end_year, overwrite, raw_dir):
    """Download ESA Snow CCI daily SWE files from CEDA."""
    config = load_config()
    y0, y1 = [int(v) for v in config["temporal_range"]]

    if years or start_year is not None or end_year is not None or run_all:
        year_list = resolve_year_list(
            years,
            start_year=start_year,
            end_year=end_year,
            default_start=y0,
            default_end=y1,
            label="ESA CCI Snow download years",
        )
    else:
        click.echo("Specify --all, --years, or --start-year/--end-year")
        return

    out_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    n = download_esa_cci_snow(years=year_list, raw_dir=out_dir, overwrite=overwrite)
    click.echo(f"Downloaded {n} file(s) into {out_dir}")


if __name__ == "__main__":
    main()
