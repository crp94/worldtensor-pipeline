"""Download CDS derived drought historical monthly data.

Dataset:
  derived-drought-historical-monthly

Usage:
    python -m src.download.climate_extremes --all
    python -m src.download.climate_extremes --variables spi --accumulation-periods 12 --start-year 2000 --end-year 2020
"""

from __future__ import annotations

import base64
import json
import shutil
import tempfile
import time
import zipfile
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import cdsapi
import click
import yaml
from tqdm import tqdm

from src.utils import get_logger
from src.year_policy import resolve_year_list

logger = get_logger("download.climate_extremes")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "climate_extremes.yml"
SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "climate_extremes"


def load_climate_extremes_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_secrets() -> dict:
    if not SECRETS_PATH.exists():
        raise FileNotFoundError(
            f"Secrets file not found: {SECRETS_PATH}\n"
            "Copy config/secrets.yml.example -> config/secrets.yml and add your CDS API key."
        )
    with open(SECRETS_PATH) as f:
        return yaml.safe_load(f)


def make_client(secrets: dict) -> cdsapi.Client:
    return cdsapi.Client(url=secrets["cds"]["url"], key=secrets["cds"]["key"])


def _parse_cds_key(raw_key: str) -> tuple[str, str]:
    if ":" not in raw_key:
        raise ValueError("CDS key must be in '<uid>:<api_key>' format")
    uid, api_key = raw_key.split(":", 1)
    return uid.strip(), api_key.strip()


def fetch_available_years(secrets: dict, dataset: str, timeout: int = 30) -> list[int] | None:
    """Return available years from CDS process schema, or None if unavailable."""
    try:
        base_url = secrets["cds"]["url"].rstrip("/")
        endpoint = f"{base_url}/retrieve/v1/processes/{dataset}"

        raw_key = str(secrets.get("cds", {}).get("key", "")).strip()
        header_candidates = []

        if raw_key:
            if ":" in raw_key:
                uid, api_key = _parse_cds_key(raw_key)
                token = base64.b64encode(f"{uid}:{api_key}".encode("utf-8")).decode("ascii")
                header_candidates.append({"Accept": "application/json", "Authorization": f"Basic {token}"})
            else:
                header_candidates.append({"Accept": "application/json", "Authorization": f"Bearer {raw_key}"})

        # CDS process schema is often public, so keep a no-auth fallback.
        header_candidates.append({"Accept": "application/json"})

        payload = None
        last_error = None
        for headers in header_candidates:
            try:
                req = Request(endpoint, headers=headers)
                with urlopen(req, timeout=timeout) as resp:
                    payload = json.loads(resp.read().decode("utf-8"))
                break
            except (HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError) as e:
                last_error = e

        if payload is None:
            raise RuntimeError(str(last_error) if last_error else "unknown error")

        year_schema = payload.get("inputs", {}).get("year", {}).get("schema", {})

        enum_values = []
        if isinstance(year_schema.get("enum"), list):
            enum_values.extend(year_schema["enum"])

        items = year_schema.get("items", {})
        if isinstance(items, dict) and isinstance(items.get("enum"), list):
            enum_values.extend(items["enum"])

        years = sorted({int(str(v)) for v in enum_values if str(v).isdigit()})
        return years or None

    except (KeyError, ValueError, HTTPError, URLError, TimeoutError, OSError, json.JSONDecodeError, RuntimeError) as e:
        logger.warning("Could not fetch available years from CDS schema: %s", e)
        return None


def clamp_years_to_available(years: list[int], available_years: list[int] | None) -> list[int]:
    """Keep only years present in CDS; if availability is unknown, return input."""
    if not years or not available_years:
        return years

    available_set = set(available_years)
    kept = [y for y in years if y in available_set]
    dropped = [y for y in years if y not in available_set]

    if dropped:
        logger.warning(
            "Skipping unavailable years from CDS schema: %s",
            dropped[:12] if len(dropped) > 12 else dropped,
        )
    return kept


def build_combo_name(short_name: str, accumulation: int) -> str:
    return f"{short_name}_{accumulation:02d}"


def make_cds_name(var_info: dict, accumulation: int) -> str:
    template = var_info.get("cds_name_template")
    if template:
        return template.format(accumulation=accumulation)

    cds_name = var_info.get("cds_name")
    if cds_name:
        return cds_name

    raise ValueError("Missing 'cds_name' or 'cds_name_template' in variable config")


def raw_path_for(raw_dir: Path, short_name: str, accumulation: int, year: int) -> Path:
    combo = build_combo_name(short_name, accumulation)
    return raw_dir / combo / f"{year}.nc"


def _extract_first_netcdf(zip_path: Path, out_nc_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "r") as zf:
        members = [m for m in zf.namelist() if m.lower().endswith(".nc")]
        if not members:
            raise RuntimeError(f"No NetCDF file found inside archive: {zip_path.name}")

        with zf.open(members[0]) as src, open(out_nc_path, "wb") as dst:
            shutil.copyfileobj(src, dst)


def _retrieve_to_netcdf(
    client: cdsapi.Client,
    dataset: str,
    request: dict,
    out_nc_path: Path,
) -> None:
    out_nc_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(prefix="climext_", suffix=".download", dir=out_nc_path.parent, delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        client.retrieve(dataset, request, str(tmp_path))

        if zipfile.is_zipfile(tmp_path):
            _extract_first_netcdf(tmp_path, out_nc_path)
        else:
            tmp_path.rename(out_nc_path)
    finally:
        tmp_path.unlink(missing_ok=True)


def _build_modern_request(config: dict, cds_name: str, accumulation: int, year: int) -> dict:
    return {
        "dataset_type": config["dataset_type"],
        "product_type": [config["product_type"]],
        "variable": [cds_name],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "accumulation_period": [str(accumulation)],
        "version": config["version"],
        "data_format": "netcdf",
        "download_format": "zip",
    }


def _build_legacy_request(config: dict, cds_name: str, accumulation: int, year: int) -> dict:
    return {
        "dataset_type": config["dataset_type"],
        "product_type": [config["product_type"]],
        "variable": cds_name,
        "year": str(year),
        "month": [f"{m:02d}" for m in range(1, 13)],
        "accumulation_period": str(accumulation),
        "version": config["version"],
        "format": "zip",
    }


def download_one(
    secrets: dict,
    config: dict,
    cds_name: str,
    short_name: str,
    accumulation: int,
    year: int,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
    max_retries: int = 4,
) -> tuple[int, Path | None, bool]:
    """Download one variable/accumulation/year file.

    Returns
    -------
    tuple
        (year, output_path_or_none, was_downloaded)
    """
    out_path = raw_path_for(raw_dir, short_name, accumulation, year)
    if out_path.exists() and not overwrite:
        return year, out_path, False

    dataset = config["dataset"]

    for attempt in range(1, max_retries + 1):
        try:
            client = make_client(secrets)

            # First try CDS modern request fields.
            modern_req = _build_modern_request(config, cds_name, accumulation, year)
            try:
                _retrieve_to_netcdf(client, dataset, modern_req, out_path)
                return year, out_path, True
            except Exception as modern_err:
                # Fallback for datasets that still require legacy request keys.
                legacy_req = _build_legacy_request(config, cds_name, accumulation, year)
                _retrieve_to_netcdf(client, dataset, legacy_req, out_path)
                logger.debug(
                    "Modern request failed for %s/%02d/%d; succeeded with legacy fields: %s",
                    short_name,
                    accumulation,
                    year,
                    str(modern_err)[:160],
                )
                return year, out_path, True

        except Exception as e:
            out_path.unlink(missing_ok=True)
            err = str(e)
            is_transient = any(k in err.lower() for k in ("timeout", "temporarily", "queued", "503", "502", "500"))
            if is_transient and attempt < max_retries:
                wait = min(30 * attempt, 120)
                logger.warning(
                    "Retry %d/%d for %s/%02d/%d after %ds: %s",
                    attempt,
                    max_retries,
                    short_name,
                    accumulation,
                    year,
                    wait,
                    err[:160],
                )
                time.sleep(wait)
            else:
                logger.error("Failed %s/%02d/%d: %s", short_name, accumulation, year, err[:220])
                return year, None, False

    return year, None, False


def download_climate_extremes(
    variables: list[str],
    accumulation_periods: list[int],
    years: list[int],
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
    delay: float = 0.2,
) -> tuple[int, int]:
    """Download multiple variable/accumulation/year combinations.

    Returns
    -------
    tuple[int, int]
        (downloaded_count, skipped_or_failed_count)
    """
    config = load_climate_extremes_config()
    secrets = load_secrets()

    total = len(variables) * len(accumulation_periods) * len(years)
    downloaded = 0
    skipped = 0

    with tqdm(total=total, desc="Downloading Climate Extremes") as pbar:
        for short_name in variables:
            info = config["variables"][short_name]

            for accumulation in accumulation_periods:
                cds_name = make_cds_name(info, accumulation)
                for year in years:
                    _, path, was_new = download_one(
                        secrets=secrets,
                        config=config,
                        cds_name=cds_name,
                        short_name=short_name,
                        accumulation=accumulation,
                        year=year,
                        raw_dir=raw_dir,
                        overwrite=overwrite,
                    )
                    if path and was_new:
                        downloaded += 1
                        if delay > 0:
                            time.sleep(delay)
                    else:
                        skipped += 1

                    pbar.update(1)
                    pbar.set_postfix(dl=downloaded, skip=skipped)

    return downloaded, skipped


@click.command()
@click.option("--variables", "variables", "-v", multiple=True, help="Variables (spi, spei).")
@click.option(
    "--accumulation-periods",
    "accumulation_periods",
    "-a",
    multiple=True,
    type=int,
    help="Accumulation period(s) in months (e.g. 1 3 12).",
)
@click.option("--all", "run_all", is_flag=True, help="Download all configured variables.")
@click.option("--years", "years", "-y", multiple=True, type=int, help="Specific years.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
@click.option("--delay", type=float, default=0.2, show_default=True, help="Delay between requests.")
@click.option("--dry-run", is_flag=True, help="Print planned downloads without executing.")
def main(variables, accumulation_periods, run_all, years, start_year, end_year, overwrite, delay, dry_run):
    """Download CDS derived drought historical monthly files."""
    config = load_climate_extremes_config()

    if variables:
        var_list = [v for v in variables if v in config["variables"]]
        missing = [v for v in variables if v not in config["variables"]]
        if missing:
            logger.warning("Unknown variables skipped: %s", missing)
    elif run_all:
        var_list = list(config["variables"].keys())
    else:
        click.echo("Specify --variables or --all. Use --help for usage.")
        return

    allowed_acc = {int(v) for v in config["accumulation_periods"]}
    if accumulation_periods:
        acc_list = sorted(set(int(v) for v in accumulation_periods if int(v) in allowed_acc))
        missing_acc = sorted(set(int(v) for v in accumulation_periods) - allowed_acc)
        if missing_acc:
            logger.warning("Unsupported accumulation periods skipped: %s", missing_acc)
    else:
        acc_list = sorted(allowed_acc)
    if not acc_list:
        click.echo("No valid accumulation periods selected.")
        return

    year_list = resolve_year_list(
        years,
        start_year=start_year,
        end_year=end_year,
        default_start=config["temporal_range"][0],
        default_end=config["temporal_range"][1],
        label="climate extremes download years",
    )
    if not year_list:
        raise click.ClickException("No valid years selected.")

    available_years = None
    if SECRETS_PATH.exists():
        try:
            secrets = load_secrets()
            available_years = fetch_available_years(secrets, config["dataset"])
        except Exception as e:
            logger.warning("Could not load CDS year availability: %s", e)

    year_list = clamp_years_to_available(year_list, available_years)
    if not year_list:
        raise click.ClickException("No requested years are available in CDS for this dataset.")

    total = len(var_list) * len(acc_list) * len(year_list)
    click.echo(
        f"Variables: {var_list}; accumulations: {acc_list}; years: {year_list[0]}-{year_list[-1]}; total requests: {total}"
    )

    if dry_run:
        for short_name in var_list:
            info = config["variables"][short_name]
            for accumulation in acc_list:
                cds_name = make_cds_name(info, accumulation)
                for year in year_list:
                    p = raw_path_for(DEFAULT_RAW_DIR, short_name, accumulation, year)
                    status = "EXISTS" if p.exists() else "NEW"
                    click.echo(f"  {p}  ({cds_name}) [{status}]")
        return

    downloaded, skipped = download_climate_extremes(
        variables=var_list,
        accumulation_periods=acc_list,
        years=year_list,
        raw_dir=DEFAULT_RAW_DIR,
        overwrite=overwrite,
        delay=delay,
    )

    click.echo(f"Done. Downloaded: {downloaded}, Skipped/Failed: {skipped}")


if __name__ == "__main__":
    main()
