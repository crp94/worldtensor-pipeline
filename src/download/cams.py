"""Download CAMS Global Reanalysis (EAC4) monthly-mean variables from ADS API.

Usage:
    python -m src.download.cams --variables no2 pm2p5 --years 2020 2021
    python -m src.download.cams --all
    python -m src.download.cams --all --start-year 2003 --end-year 2024
"""

import time
from pathlib import Path

import click
import cdsapi
import yaml
from tqdm import tqdm

from src.utils import get_logger
from src.year_policy import resolve_year_list

logger = get_logger("download.cams")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "cams.yml"
SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "cams"


def load_cams_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_secrets() -> dict:
    """Load ADS/CDS API credentials from config/secrets.yml."""
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            f"Secrets file not found: {SECRETS_PATH}\n"
            f"Copy config/secrets.yml.example to config/secrets.yml and add your ADS API key."
        )
    with open(SECRETS_PATH) as f:
        secrets = yaml.safe_load(f)
    
    # Prefer 'ads' section, fallback to 'cds'
    if "ads" in secrets:
        return secrets["ads"]
    elif "cds" in secrets:
        logger.warning("Using 'cds' credentials for ADS request.")
        return {
            "url": "https://ads.atmosphere.copernicus.eu/api",
            "key": secrets["cds"]["key"]
        }
    else:
        raise KeyError("No 'ads' or 'cds' section found in secrets.yml")


def download_variable_year(
    client: cdsapi.Client,
    ads_name: str,
    short_name: str,
    year: int,
    dataset: str,
    product_type: str,
    output_dir: Path,
    overwrite: bool = False,
) -> Path | None:
    """Download all 12 months for one variable and one year."""
    out_path = output_dir / short_name / f"{year}.nc"

    if out_path.exists() and not overwrite:
        logger.debug("Skipping %s/%d (already exists)", short_name, year)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    request = {
        "variable": [ads_name],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

    # Some variables are 3D and require a pressure level.
    # We use 1000 hPa as a proxy for surface concentration for these.
    # Total column variables do NOT take pressure_level.
    if not short_name.startswith("tc") and short_name not in ["aod550"]:
        request["pressure_level"] = ["1000"]

    try:
        client.retrieve(dataset, request, str(out_path))
        logger.info("Downloaded %s/%d → %s", short_name, year, out_path)
        return out_path
    except Exception as e:
        logger.error("Failed %s/%d: %s", short_name, year, e)
        if out_path.exists():
            out_path.unlink()
        return None


@click.command()
@click.option("--variables", "-v", multiple=True, help="Short names (e.g. no2 pm2p5).")
@click.option("--all", "download_all", is_flag=True, help="Download all in config.")
@click.option("--years", "-y", multiple=True, type=int, help="Specific years.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--output-dir", type=click.Path(), default=None, help="Override output directory.")
@click.option("--overwrite", is_flag=True, help="Re-download existing files.")
@click.option("--delay", type=float, default=1.0, help="Seconds between requests.")
@click.option("--dry-run", is_flag=True, help="Print what would be downloaded.")
def main(variables, download_all, years, start_year, end_year, output_dir, overwrite, delay, dry_run):
    """Download CAMS EAC4 data from ADS."""
    config = load_cams_config()
    dataset = config["dataset"]
    all_vars = config["variables"]
    t_range = config["temporal_range"]

    if variables:
        var_list = {v: all_vars[v] for v in variables if v in all_vars}
    elif download_all:
        var_list = all_vars
    else:
        click.echo("Specify --variables or --all.")
        return

    year_list = resolve_year_list(
        years,
        start_year=start_year,
        end_year=end_year,
        default_start=t_range[0],
        default_end=t_range[1],
        label="CAMS download years",
    )

    out_dir = Path(output_dir) if output_dir else RAW_DIR

    if dry_run:
        for short_name, info in var_list.items():
            for year in year_list:
                click.echo(f"Would download: {short_name}/{year}.nc ({info['ads_name']})")
        return

    secrets = load_secrets()
    client = cdsapi.Client(url=secrets["url"], key=secrets["key"])

    with tqdm(total=len(var_list) * len(year_list), desc="Downloading CAMS") as pbar:
        for short_name, info in var_list.items():
            for year in year_list:
                result = download_variable_year(
                    client=client,
                    ads_name=info["ads_name"],
                    short_name=short_name,
                    year=year,
                    dataset=dataset,
                    product_type=config["product_type"],
                    output_dir=out_dir,
                    overwrite=overwrite,
                )
                if result:
                    time.sleep(delay)
                pbar.update(1)


if __name__ == "__main__":
    main()
