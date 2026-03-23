"""Download MOD13C2 monthly granules via earthaccess.

Usage:
    python -m src.download.mod13c2 --all
    python -m src.download.mod13c2 --start-year 2015 --end-year 2020
"""

from __future__ import annotations

import os
from pathlib import Path

import click
import yaml

from src.utils import get_logger
from src.year_policy import resolve_year_bounds

logger = get_logger("download.mod13c2")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "mod13c2.yml"
SECRETS_PATH = PROJECT_ROOT / "config" / "secrets.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "mod13c2"


def _import_earthaccess():
    try:
        import earthaccess  # type: ignore
    except Exception as e:  # pragma: no cover - import guard
        raise RuntimeError(
            "earthaccess is required for MOD13C2 download. "
            "Install it in the current environment (e.g. `pip install earthaccess`)."
        ) from e
    return earthaccess


def load_mod13c2_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def load_secrets() -> dict:
    if not SECRETS_PATH.exists():
        return {}
    with open(SECRETS_PATH) as f:
        return yaml.safe_load(f) or {}


def _prepare_auth_env(secrets: dict) -> None:
    """Set earthaccess-compatible auth environment variables from secrets.yml.

    Supports either:
    - earthdata.username + earthdata.password
    - earthdata.token
    """
    earth = secrets.get("earthdata", {})

    username = earth.get("username")
    password = earth.get("password")
    token = earth.get("token")

    if username and password:
        os.environ.setdefault("EARTHDATA_USERNAME", str(username))
        os.environ.setdefault("EARTHDATA_PASSWORD", str(password))

    if token:
        os.environ.setdefault("EARTHDATA_TOKEN", str(token))


def login_earthaccess() -> None:
    earthaccess = _import_earthaccess()

    _prepare_auth_env(load_secrets())

    # Non-interactive first: env vars, then netrc.
    for strategy in ("environment", "netrc"):
        try:
            auth = earthaccess.login(strategy=strategy)
            if getattr(auth, "authenticated", False):
                logger.info("Authenticated with earthaccess using '%s' strategy", strategy)
                return
        except Exception:
            continue

    raise RuntimeError(
        "Earthaccess authentication failed. Add Earthdata credentials to "
        "config/secrets.yml (earthdata.username/password or earthdata.token), "
        "or configure ~/.netrc."
    )


def search_granules(start_year: int | None = None, end_year: int | None = None) -> list:
    """Search MOD13C2 monthly granules in CMR."""
    earthaccess = _import_earthaccess()
    config = load_mod13c2_config()

    # Default to configured range so --all remains bounded and reproducible.
    y0, y1 = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=config["temporal_range"][0],
        default_end=config["temporal_range"][1],
        label="MOD13C2 download years",
    )
    temporal = (f"{y0}-01-01", f"{y1}-12-31")

    kwargs = {
        "short_name": config["short_name"],
        "version": config["version"],
    }
    kwargs["temporal"] = temporal

    granules = earthaccess.search_data(**kwargs)
    logger.info("Found %d MOD13C2 granules", len(granules))
    return granules


def download_mod13c2(
    start_year: int | None = None,
    end_year: int | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
) -> list[Path]:
    """Authenticate, search, and download MOD13C2 granules.

    Returns local file paths for downloaded (or already-present) files.
    """
    earthaccess = _import_earthaccess()

    login_earthaccess()
    granules = search_granules(start_year=start_year, end_year=end_year)
    if not granules:
        return []

    raw_dir.mkdir(parents=True, exist_ok=True)

    downloaded = earthaccess.download(granules, str(raw_dir))
    paths = []
    for item in downloaded:
        if item is None:
            continue
        paths.append(Path(item))

    # Deduplicate while preserving order
    unique = list(dict.fromkeys(paths))
    logger.info("Downloaded/available files: %d", len(unique))
    return unique


@click.command()
@click.option("--all", "run_all", is_flag=True, help="Search full archive.")
@click.option("--start-year", type=int, default=None, help="Start year filter.")
@click.option("--end-year", type=int, default=None, help="End year filter.")
@click.option(
    "--raw-dir",
    type=click.Path(),
    default=None,
    help=f"Output directory (default: {DEFAULT_RAW_DIR})",
)
def main(run_all, start_year, end_year, raw_dir):
    """Download MOD13C2 monthly granules via earthaccess."""
    if not run_all and start_year is None and end_year is None:
        click.echo("Specify --all or provide --start-year/--end-year.")
        return

    out_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR
    paths = download_mod13c2(start_year=start_year, end_year=end_year, raw_dir=out_dir)
    click.echo(f"Downloaded/available {len(paths)} files in {out_dir}")


if __name__ == "__main__":
    main()
