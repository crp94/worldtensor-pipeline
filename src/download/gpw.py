"""Download GPWv4.11 gridded population data from NASA Earthdata.

Data migrated from SEDAC to data.earthdata.nasa.gov.
Uses NASA Earthdata Bearer token authentication (from config/secrets.yml).
Each zip contains a NetCDF or GeoTIFF at 2.5 arc-minute resolution.
Downloads zip, extracts contents, deletes zip.

Usage:
    python -m src.download.gpw --all
    python -m src.download.gpw --variables population_count population_density
"""

import time
import zipfile
from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.gpw")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "gpw.yml"
SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "gpw"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
MAX_RETRIES = 5
INITIAL_TIMEOUT = 120  # seconds


def load_gpw_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_earthdata_token() -> str:
    with open(SECRETS_PATH) as f:
        secrets = yaml.safe_load(f)
    return secrets["earthdata"]["token"]


def build_url(base_url: str, slug: str, filename: str) -> str:
    """Build the download URL for a GPW file."""
    return f"{base_url}/{slug}/{filename}"


def _download_and_extract(url: str, dest_dir: Path, label: str,
                          token: str,
                          overwrite: bool = False) -> Path | None:
    """Download a zip and extract its contents.

    Uses Bearer token auth, follows Earthdata → CloudFront redirects.
    Retries with exponential backoff.

    Returns path to the first extracted file, or None on failure.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)

    zip_name = url.rsplit("/", 1)[-1]
    zip_path = dest_dir / zip_name

    session = requests.Session()
    session.headers.update({"Authorization": f"Bearer {token}"})

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

            # Extract from zip
            with zipfile.ZipFile(zip_path, "r") as zf:
                members = zf.namelist()
                if not members:
                    logger.error("Empty zip: %s", zip_path)
                    zip_path.unlink(missing_ok=True)
                    return None
                zf.extractall(dest_dir)
                logger.info("Extracted %d file(s) for %s", len(members), label)

            zip_path.unlink(missing_ok=True)

            # Return first data file
            data_files = sorted(dest_dir.glob("*.nc")) or sorted(dest_dir.glob("*.tif"))
            return data_files[0] if data_files else None

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


def download_standard_variable(base_url: str, var_name: str, var_info: dict,
                               raw_dir: Path, token: str,
                               overwrite: bool) -> Path | None:
    """Download a standard GPW variable (single zip with NC or TIF)."""
    dest_dir = raw_dir / var_name
    slug = var_info["slug"]
    filename = var_info["filename"]

    # Check if already downloaded
    fmt = var_info.get("format", "nc")
    existing = list(dest_dir.glob(f"*.{fmt}"))
    if existing and not overwrite:
        logger.info("Already downloaded %s (%d files)", var_name, len(existing))
        return existing[0]

    url = build_url(base_url, slug, filename)
    return _download_and_extract(url, dest_dir, var_name, token, overwrite)


def download_gpw(
    variables: list[str] | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> list[Path]:
    """Download GPW files for given variables.

    Returns list of paths to downloaded files.
    """
    config = load_gpw_config()
    token = load_earthdata_token()
    base_url = config["base_url"]
    all_vars = config["variables"]

    var_list = variables or list(all_vars.keys())

    downloaded = []
    for var_name in var_list:
        var_info = all_vars.get(var_name)
        if var_info is None:
            logger.warning("Unknown variable: %s", var_name)
            continue

        path = download_standard_variable(base_url, var_name, var_info,
                                          raw_dir, token, overwrite)
        if path:
            downloaded.append(path)

    return downloaded


@click.command()
@click.option("--variables", "-v", multiple=True,
              help="Variable(s) to download (e.g. population_count).")
@click.option("--all", "run_all", is_flag=True, help="Download all variables.")
@click.option("--overwrite", is_flag=True, help="Re-download even if file exists.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Output directory (default: {DEFAULT_RAW_DIR})")
def main(variables, run_all, overwrite, raw_dir):
    """Download GPWv4.11 gridded population data from NASA Earthdata."""
    if not variables and not run_all:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    out_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    var_list = list(variables) if variables else None

    paths = download_gpw(
        variables=var_list,
        raw_dir=out_dir,
        overwrite=overwrite,
    )
    click.echo(f"Downloaded {len(paths)} files to {out_dir}")


if __name__ == "__main__":
    main()
