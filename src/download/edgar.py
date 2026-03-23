"""Download EDGAR 2025 GHG gridded emissions (per-year/substance/sector zips).

Each zip contains a single NetCDF with 0.1° global fluxes (kg m-2 s-1).
Downloads zip, extracts NC, deletes zip.

Usage:
    python -m src.download.edgar --all
    python -m src.download.edgar --substances CO2 CH4 --sectors TOTALS ENE
    python -m src.download.edgar --substances CO2 --sectors TOTALS --start-year 2020 --end-year 2024
"""

import zipfile
from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger
from src.year_policy import resolve_year_bounds

logger = get_logger("download.edgar")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "edgar.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "edgar"
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB


def load_edgar_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def build_url(base: str, substance: str, sector: str, year: int) -> str:
    """Build the download URL for one substance/sector/year."""
    fname = f"EDGAR_2025_GHG_{substance}_{year}_{sector}_flx_nc.zip"
    return f"{base}/{substance}/{sector}/flx_nc/{fname}"


def expected_nc_name(substance: str, sector: str, year: int) -> str:
    """Expected NetCDF filename inside the zip."""
    return f"EDGAR_2025_GHG_{substance}_{year}_{sector}_flx_nc.nc"


def download_file(url: str, dest_dir: Path, substance: str, sector: str,
                  year: int, overwrite: bool = False) -> Path | None:
    """Download and extract a single EDGAR zip file.

    Returns path to the extracted NC, or None on failure.
    Deletes the zip after extraction.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    nc_name = expected_nc_name(substance, sector, year)
    nc_path = dest_dir / nc_name

    if nc_path.exists() and not overwrite:
        return nc_path

    zip_path = dest_dir / f"{nc_name.replace('.nc', '.zip')}"
    try:
        resp = requests.get(url, stream=True, timeout=120)
        resp.raise_for_status()

        total = int(resp.headers.get("content-length", 0))
        with (
            open(zip_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True,
                 desc=f"{substance}/{sector}/{year}", leave=False) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                f.write(chunk)
                pbar.update(len(chunk))

        # Extract NC from zip
        with zipfile.ZipFile(zip_path, "r") as zf:
            nc_members = [m for m in zf.namelist() if m.endswith(".nc")]
            if not nc_members:
                logger.error("No NC file in zip: %s", zip_path)
                zip_path.unlink(missing_ok=True)
                return None
            # Extract the first NC file
            member = nc_members[0]
            zf.extract(member, dest_dir)
            extracted = dest_dir / member
            if extracted != nc_path:
                extracted.rename(nc_path)

        zip_path.unlink(missing_ok=True)
        return nc_path

    except requests.HTTPError as e:
        logger.warning("HTTP error for %s/%s/%d: %s", substance, sector, year, e)
        zip_path.unlink(missing_ok=True)
        return None
    except Exception as e:
        logger.error("Download failed %s/%s/%d: %s", substance, sector, year, e)
        zip_path.unlink(missing_ok=True)
        return None


def download_edgar(
    substances: list[str] | None = None,
    sectors: list[str] | None = None,
    start_year: int | None = None,
    end_year: int | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> list[Path]:
    """Download EDGAR files for given substances/sectors/years.

    Returns list of paths to downloaded NC files.
    """
    config = load_edgar_config()
    base_url = config["source_url_base"]
    t_range = config["temporal_range"]

    y_start, y_end = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=t_range[0],
        default_end=t_range[1],
        label="EDGAR download years",
    )
    years = list(range(y_start, y_end + 1))

    sub_list = substances or list(config["substances"].keys())
    all_sectors = config["sectors"]

    downloaded = []
    for sub in sub_list:
        sub_info = config["substances"].get(sub)
        if sub_info is None:
            logger.warning("Unknown substance: %s", sub)
            continue

        available_sectors = sub_info["sectors"]
        if sectors:
            sec_list = [s for s in sectors if s in available_sectors]
            skipped = [s for s in sectors if s not in available_sectors]
            if skipped:
                logger.info("Sectors not available for %s: %s", sub, skipped)
        else:
            sec_list = available_sectors

        for sec in sec_list:
            dest_dir = raw_dir / sub / sec
            for year in tqdm(years, desc=f"{sub}/{sec}", unit="yr"):
                url = build_url(base_url, sub, sec, year)
                path = download_file(url, dest_dir, sub, sec, year, overwrite)
                if path:
                    downloaded.append(path)

    return downloaded


@click.command()
@click.option("--substances", "-s", multiple=True,
              help="Substance(s) to download (e.g. CO2 CH4).")
@click.option("--sectors", "-S", multiple=True,
              help="Sector(s) to download (e.g. TOTALS ENE).")
@click.option("--all", "run_all", is_flag=True, help="Download all substances/sectors.")
@click.option("--start-year", type=int, default=None, help="Start year (default: 1970).")
@click.option("--end-year", type=int, default=None, help="End year (default: 2024).")
@click.option("--overwrite", is_flag=True, help="Re-download even if NC exists.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Output directory (default: {DEFAULT_RAW_DIR})")
def main(substances, sectors, run_all, start_year, end_year, overwrite, raw_dir):
    """Download EDGAR 2025 GHG gridded emissions."""
    if not substances and not run_all:
        click.echo("Specify --substances or --all. Use --help for usage.")
        return

    out_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    sub_list = list(substances) if substances else None
    sec_list = list(sectors) if sectors else None

    paths = download_edgar(
        substances=sub_list,
        sectors=sec_list,
        start_year=start_year,
        end_year=end_year,
        raw_dir=out_dir,
        overwrite=overwrite,
    )
    click.echo(f"Downloaded {len(paths)} files to {out_dir}")


if __name__ == "__main__":
    main()
