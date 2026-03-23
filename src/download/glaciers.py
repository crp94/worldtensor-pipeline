"""Download WGMS annual glacier mass-change estimates (AMCE)."""

from __future__ import annotations

import zipfile
from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.glaciers")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "glaciers.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "glaciers"
CHUNK_SIZE = 8 * 1024 * 1024


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _download_file(url: str, out_path: Path, overwrite: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    with requests.get(url, stream=True, timeout=1200) as resp:
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


def download_glaciers(raw_dir: Path = DEFAULT_RAW_DIR, overwrite: bool = False) -> Path:
    config = load_config()
    url = config["source_url"]
    archive_name = config.get("archive_filename", Path(url).name)
    nc_name = config["netcdf_filename"]

    raw_dir.mkdir(parents=True, exist_ok=True)
    archive_path = raw_dir / archive_name
    nc_out = raw_dir / nc_name

    if nc_out.exists() and not overwrite:
        logger.info("Glacier NetCDF already present: %s", nc_out)
        return nc_out

    _download_file(url, archive_path, overwrite=overwrite)

    with zipfile.ZipFile(archive_path, "r") as zf:
        members = [m for m in zf.namelist() if Path(m).name == nc_name]
        if not members:
            raise FileNotFoundError(f"{nc_name} not found in archive {archive_path}")
        with zf.open(members[0]) as src, open(nc_out, "wb") as dst:
            dst.write(src.read())

    logger.info("Prepared glacier file: %s", nc_out)
    return nc_out


@click.command()
@click.option("--all", "run_all", is_flag=True, help="Download configured glacier source.")
@click.option("--overwrite", is_flag=True, help="Re-download and re-extract files.")
def main(run_all: bool, overwrite: bool):
    if not run_all:
        click.echo("Specify --all")
        return
    out = download_glaciers(overwrite=overwrite)
    click.echo(f"Ready: {out}")


if __name__ == "__main__":
    main()
