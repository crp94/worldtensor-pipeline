"""Download ERA5 monthly-mean single-level variables from CDS API.

Usage:
    python -m src.download.era5 --variables t2m tp --years 2020 2021
    python -m src.download.era5 --all
    python -m src.download.era5 --all --start-year 2000 --end-year 2020
    python -m src.download.era5 --variables t2m --years 2020 --dry-run
"""

import time
from pathlib import Path

import click
import cdsapi
import yaml
from tqdm import tqdm

from src.utils import get_logger
from src.year_policy import resolve_year_list

logger = get_logger("download.era5")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "era5.yml"
SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "era5"


def load_era5_config() -> dict:
    with open(CONFIG_PATH) as f:
        config = yaml.safe_load(f)
    excluded = {str(v) for v in config.get("excluded_variables", [])}
    if excluded:
        config["variables"] = {k: v for k, v in config.get("variables", {}).items() if k not in excluded}
    return config


def load_secrets() -> dict:
    """Load CDS API credentials from config/secrets.yml."""
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            f"Secrets file not found: {SECRETS_PATH}\n"
            f"Copy config/secrets.yml.example to config/secrets.yml and add your CDS API key.\n"
            f"Get your key at: https://cds.climate.copernicus.eu/profile"
        )
    with open(SECRETS_PATH) as f:
        return yaml.safe_load(f)


def download_variable_year(
    client: cdsapi.Client,
    cds_name: str,
    short_name: str,
    year: int,
    dataset: str,
    product_type: str,
    output_dir: Path,
    overwrite: bool = False,
) -> Path | None:
    """Download all 12 months for one variable and one year.

    Returns the output path, or None if skipped/failed.
    """
    out_path = output_dir / short_name / f"{year}.nc"

    if out_path.exists() and not overwrite:
        logger.debug("Skipping %s/%d (already exists)", short_name, year)
        return None

    out_path.parent.mkdir(parents=True, exist_ok=True)

    request = {
        "product_type": [product_type],
        "variable": [cds_name],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
    }

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
@click.option("--variables", "-v", multiple=True, help="Short names to download (e.g. t2m tp). Omit for --all.")
@click.option("--all", "download_all", is_flag=True, help="Download all variables in config.")
@click.option("--years", "-y", multiple=True, type=int, help="Specific years (e.g. 2020 2021).")
@click.option("--start-year", type=int, default=None, help="Start year (default: from config).")
@click.option("--end-year", type=int, default=None, help="End year (default: from config).")
@click.option("--output-dir", type=click.Path(), default=None, help="Override output directory.")
@click.option("--overwrite", is_flag=True, help="Re-download existing files.")
@click.option("--delay", type=float, default=1.0, help="Seconds between requests.")
@click.option("--dry-run", is_flag=True, help="Print what would be downloaded without downloading.")
def main(variables, download_all, years, start_year, end_year, output_dir, overwrite, delay, dry_run):
    """Download ERA5 monthly-mean data from CDS."""
    config = load_era5_config()
    dataset = config["dataset"]
    product_type = config["product_type"]
    all_vars = config["variables"]
    t_range = config["temporal_range"]

    # Resolve variables
    if variables:
        var_list = {v: all_vars[v] for v in variables if v in all_vars}
        missing = [v for v in variables if v not in all_vars]
        if missing:
            logger.warning("Unknown variables (skipped): %s", missing)
    elif download_all:
        var_list = all_vars
    else:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    # Resolve years
    year_list = resolve_year_list(
        years,
        start_year=start_year,
        end_year=end_year,
        default_start=t_range[0],
        default_end=t_range[1],
        label="ERA5 download years",
    )
    if not year_list:
        click.echo("No valid years selected.")
        return

    out_dir = Path(output_dir) if output_dir else RAW_DIR

    total = len(var_list) * len(year_list)
    click.echo(f"Variables: {len(var_list)}, Years: {year_list[0]}–{year_list[-1]}, Total requests: {total}")

    if dry_run:
        click.echo("\n[DRY RUN] Would download:")
        for short_name, info in var_list.items():
            for year in year_list:
                path = out_dir / short_name / f"{year}.nc"
                status = "EXISTS" if path.exists() else "NEW"
                click.echo(f"  {short_name}/{year}.nc  ({info['cds_name']})  [{status}]")
        return

    secrets = load_secrets()
    client = cdsapi.Client(url=secrets["cds"]["url"], key=secrets["cds"]["key"])
    downloaded = 0
    skipped = 0

    with tqdm(total=total, desc="Downloading ERA5") as pbar:
        for short_name, info in var_list.items():
            for year in year_list:
                result = download_variable_year(
                    client=client,
                    cds_name=info["cds_name"],
                    short_name=short_name,
                    year=year,
                    dataset=dataset,
                    product_type=product_type,
                    output_dir=out_dir,
                    overwrite=overwrite,
                )
                if result:
                    downloaded += 1
                    time.sleep(delay)
                else:
                    skipped += 1
                pbar.update(1)
                pbar.set_postfix(dl=downloaded, skip=skipped)

    click.echo(f"\nDone. Downloaded: {downloaded}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
