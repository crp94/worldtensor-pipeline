"""Download MODIS-Aqua Monthly Chlorophyll-a via earthaccess.

Uses NASA EarthData authentication from config/secrets.yml.
Target: 4km Monthly L3m composites.

Usage:
    python -m src.download.chlorophyll --years 2020 2021
    python -m src.download.chlorophyll --all
"""

import os
import re
import datetime
from pathlib import Path

import click
import yaml
from src.utils import get_logger
from src.year_policy import resolve_year_list

logger = get_logger("download.chlorophyll")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "chlorophyll.yml"
SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "chlorophyll"


def _import_earthaccess():
    try:
        import earthaccess
        return earthaccess
    except ImportError:
        raise RuntimeError("earthaccess not installed. Run 'pip install earthaccess'")


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_secrets() -> dict:
    if not SECRETS_PATH.exists(): return {}
    with open(SECRETS_PATH) as f:
        return yaml.safe_load(f) or {}


def login_earthaccess():
    ea = _import_earthaccess()
    secrets = load_secrets()
    earth = secrets.get("earthdata", {})
    
    if "username" in earth and "password" in earth:
        os.environ.setdefault("EARTHDATA_USERNAME", str(earth["username"]))
        os.environ.setdefault("EARTHDATA_PASSWORD", str(earth["password"]))
    
    if "token" in earth:
        os.environ.setdefault("EARTHDATA_TOKEN", str(earth["token"]))

    auth = ea.login(strategy="environment")
    if not auth.authenticated:
        auth = ea.login(strategy="netrc")
    return ea


def get_fname(g) -> str:
    """Robustly extract filename from earthaccess granule."""
    try:
        # Try standard methods if available in future versions
        if hasattr(g, "get_filename"): return g.get_filename()
    except Exception:
        pass
    
    # Use data links as fallback
    links = g.data_links()
    if links:
        return links[0].split("/")[-1]
    
    return ""


def download_chlorophyll(year_list: list[int], raw_dir: Path = DEFAULT_RAW_DIR):
    ea = login_earthaccess()
    
    for year in year_list:
        logger.info("Searching Chlorophyll-a for year %d", year)
        temporal = (f"{year}-01-01", f"{year}-12-31")
        
        granules = ea.search_data(
            short_name="MODISA_L3m_CHL",
            temporal=temporal,
        )
        
        if not granules:
            logger.warning("No granules found for %d", year)
            continue

        # Filter for monthly composites (MO) and 4km resolution
        monthly_4km = [
            g for g in granules 
            if ".MO." in get_fname(g) and ".4km." in get_fname(g)
        ]
        
        if not monthly_4km:
            logger.warning("No monthly 4km granules found for %d among %d total", year, len(granules))
            continue

        logger.info("Found %d monthly 4km granules for %d", len(monthly_4km), year)
        
        year_dir = raw_dir / str(year)
        year_dir.mkdir(parents=True, exist_ok=True)
        
        ea.download(monthly_4km, str(year_dir))
        
        # Standardize filenames to 01.nc, 02.nc etc
        # Pattern: AQUA_MODIS.20100101_20100131.L3m.MO.CHL.chlor_a.4km.nc
        for p in year_dir.glob("*.nc"):
            match = re.search(r"\.(\d{4})(\d{2})(\d{2})_", p.name)
            if match:
                month = match.group(2)
                new_name = f"{month}.nc"
                if p.name != new_name:
                    dest = p.parent / new_name
                    if dest.exists(): dest.unlink()
                    p.rename(dest)


@click.command()
@click.option("--years", "-y", multiple=True, type=int)
@click.option("--all", "run_all", is_flag=True)
def main(years, run_all):
    config = load_config()
    if years:
        year_list = resolve_year_list(
            years,
            default_start=config["temporal_range"][0],
            default_end=config["temporal_range"][1],
            label="chlorophyll download years",
        )
    elif run_all:
        year_list = resolve_year_list(
            start_year=config["temporal_range"][0],
            end_year=config["temporal_range"][1],
            default_start=config["temporal_range"][0],
            default_end=config["temporal_range"][1],
            label="chlorophyll download years",
        )
    else:
        click.echo("Specify -y or --all")
        return

    download_chlorophyll(year_list)


if __name__ == "__main__":
    main()
