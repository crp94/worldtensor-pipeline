"""Download precomputed annual land heatwave indices from HadEX3.

Usage:
    python -m src.download.land_heatwaves --all
    python -m src.download.land_heatwaves --variables land_heatwave_frequency
"""

from __future__ import annotations

import gzip
import shutil
from pathlib import Path
from urllib.parse import urljoin

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.land_heatwaves")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "land_heatwaves.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "land_heatwaves"
CHUNK_SIZE = 8 * 1024 * 1024


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_variables(config: dict, selected: tuple[str, ...], run_all: bool) -> dict:
    all_vars = config.get("variables", {})
    if selected:
        resolved = {k: v for k, v in all_vars.items() if k in selected}
        missing = sorted(set(selected) - set(resolved))
        if missing:
            logger.warning("Unknown variables skipped: %s", missing)
        return resolved
    if run_all:
        return all_vars
    return {}


def _download_to_path(url: str, out_path: Path, overwrite: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    with requests.get(url, stream=True, timeout=600) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        with (
            open(tmp_path, "wb") as f,
            tqdm(total=total, unit="B", unit_scale=True, desc=out_path.name, leave=False) as pbar,
        ):
            for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                if not chunk:
                    continue
                f.write(chunk)
                pbar.update(len(chunk))

    tmp_path.replace(out_path)
    return out_path


def _gunzip_file(gz_path: Path, nc_path: Path, overwrite: bool = False) -> Path:
    if nc_path.exists() and not overwrite:
        return nc_path
    with gzip.open(gz_path, "rb") as src, open(nc_path, "wb") as dst:
        shutil.copyfileobj(src, dst)
    return nc_path


def download_land_heatwaves(
    selected_variables: tuple[str, ...] = (),
    run_all: bool = False,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> dict[str, Path]:
    """Download selected HadEX3 annual heatwave files and return extracted NetCDF paths."""
    config = load_config()
    selected = _resolve_variables(config, selected_variables, run_all)
    if not selected:
        return {}

    base_url = str(config["source_base_url"]).rstrip("/") + "/"
    raw_dir.mkdir(parents=True, exist_ok=True)

    out: dict[str, Path] = {}
    for var_key, info in selected.items():
        source_filename = info["source_filename"]
        url = urljoin(base_url, source_filename)
        gz_path = raw_dir / source_filename
        nc_path = raw_dir / source_filename.replace(".nc.gz", ".nc")

        try:
            _download_to_path(url, gz_path, overwrite=overwrite)
            _gunzip_file(gz_path, nc_path, overwrite=overwrite)
            out[var_key] = nc_path
            logger.info("Ready: %s -> %s", var_key, nc_path)
        except Exception as e:
            logger.error("Failed download for %s (%s): %s", var_key, url, e)

    return out


@click.command()
@click.option("--variables", "-v", multiple=True, help="Variable keys to download.")
@click.option("--all", "run_all", is_flag=True, help="Download all configured variables.")
@click.option("--overwrite", is_flag=True, help="Re-download and re-extract existing files.")
def main(variables: tuple[str, ...], run_all: bool, overwrite: bool):
    if not variables and not run_all:
        click.echo("Specify --variables or --all")
        return

    downloaded = download_land_heatwaves(selected_variables=variables, run_all=run_all, overwrite=overwrite)
    if downloaded:
        click.echo(f"Prepared {len(downloaded)} file(s) in {DEFAULT_RAW_DIR}")
    else:
        click.echo("No files downloaded.")


if __name__ == "__main__":
    main()
