"""Download NOAA PSL marine heatwave monthly source files."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.marine_heatwaves")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "marine_heatwaves.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "marine_heatwaves"
CHUNK_SIZE = 8 * 1024 * 1024


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _download_file(url: str, out_path: Path, overwrite: bool = False) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if out_path.exists() and not overwrite:
        return out_path

    tmp_path = out_path.with_suffix(out_path.suffix + ".partial")
    with requests.get(url, stream=True, timeout=1800) as resp:
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


def download_marine_heatwaves(raw_dir: Path = DEFAULT_RAW_DIR, overwrite: bool = False) -> dict[str, Path]:
    config = load_config()
    base = str(config["source_base_url"]).rstrip("/") + "/"
    files = config["files"]

    out: dict[str, Path] = {}
    for key, fname in files.items():
        out_path = raw_dir / fname
        url = urljoin(base, fname)
        try:
            _download_file(url, out_path, overwrite=overwrite)
            out[key] = out_path
            logger.info("Ready: %s -> %s", key, out_path)
        except Exception as e:
            logger.error("Failed %s (%s): %s", key, url, e)
    return out


@click.command()
@click.option("--all", "run_all", is_flag=True, help="Download configured marine heatwave files.")
@click.option("--overwrite", is_flag=True, help="Re-download existing files.")
def main(run_all: bool, overwrite: bool):
    if not run_all:
        click.echo("Specify --all")
        return
    out = download_marine_heatwaves(overwrite=overwrite)
    click.echo(f"Prepared {len(out)} file(s) in {DEFAULT_RAW_DIR}")


if __name__ == "__main__":
    main()
