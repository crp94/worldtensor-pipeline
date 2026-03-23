"""Download GRACE-FO and GLDAS data via earthaccess.
"""

import os
from pathlib import Path
import yaml
import click
from src.utils import get_logger

logger = get_logger("download.hydrology")

def login_earthaccess():
    import earthaccess
    secrets_path = Path("config/secrets.yml")
    if not secrets_path.exists():
        raise FileNotFoundError("config/secrets.yml missing")
    with open(secrets_path) as f:
        secrets = yaml.safe_load(f)
    earth = secrets.get("earthdata", {})
    if "token" in earth: os.environ["EARTHDATA_TOKEN"] = str(earth["token"])
    if "username" in earth: os.environ["EARTHDATA_USERNAME"] = str(earth["username"])
    if "password" in earth: os.environ["EARTHDATA_PASSWORD"] = str(earth["password"])
    auth = earthaccess.login(strategy="environment")
    return earthaccess

def download_grace():
    ...
    if granules:
        ea.download(granules, str(raw_dir))

def download_grdi():
    ea = login_earthaccess()
    raw_dir = Path("data/raw/grdi")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # CIESIN SEDAC GRDI 2010-2020
    short_name = "CIESIN_SEDAC_PMP_GRDI_2010_2020"
    logger.info(f"Searching for {short_name}...")
    granules = ea.search_data(short_name=short_name, count=1)
    if granules:
        ea.download(granules, str(raw_dir))

def download_lgii():
    ...
    if granules:
        ea.download(granules, str(raw_dir))

def download_imr():
    ea = login_earthaccess()
    raw_dir = Path("data/raw/imr")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # CIESIN SEDAC IMR (Infant Mortality Rate)
    short_name = "CIESIN_SEDAC_PMP_IMR_V2.01"
    logger.info(f"Searching for {short_name}...")
    granules = ea.search_data(short_name=short_name, count=1)
    if granules:
        ea.download(granules, str(raw_dir))

def download_gldas(year):
    ea = login_earthaccess()
    raw_dir = Path(f"data/raw/gldas/{year}")
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # GLDAS Noah Land Surface Model L4 monthly 0.25 x 0.25 degree V2.1
    short_name = "GLDAS_NOAH025_M"
    logger.info(f"Searching for {short_name} for {year}...")
    granules = ea.search_data(
        short_name=short_name,
        version="2.1",
        temporal=(f"{year}-01-01", f"{year}-12-31")
    )
    if granules:
        logger.info(f"Found {len(granules)} monthly granules. Downloading...")
        ea.download(granules, str(raw_dir))

@click.command()
@click.option("--dataset", type=click.Choice(["grace", "gldas", "grdi", "lgii", "imr"]), required=True)
@click.option("--year", type=int, help="Year for GLDAS")
def main(dataset, year):
    if dataset == "grace":
        download_grace()
    elif dataset == "grdi":
        download_grdi()
    elif dataset == "lgii":
        download_lgii()
    elif dataset == "imr":
        download_imr()
    else:
        if not year:
            logger.error("Year required for GLDAS")
            return
        download_gldas(year)

if __name__ == "__main__":
    main()
