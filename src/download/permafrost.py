"""Download ESA CCI Permafrost yearly northern-hemisphere products."""

from __future__ import annotations

from pathlib import Path
from urllib.parse import urljoin

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger
from src.year_policy import resolve_year_list

logger = get_logger("download.permafrost")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "permafrost.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "permafrost"
CHUNK_SIZE = 8 * 1024 * 1024


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _resolve_datasets(config: dict, selected: tuple[str, ...], run_all: bool) -> dict:
    datasets = config.get("datasets", {})
    if selected:
        resolved = {k: v for k, v in datasets.items() if k in selected}
        missing = sorted(set(selected) - set(resolved))
        if missing:
            logger.warning("Unknown permafrost datasets skipped: %s", missing)
        return resolved
    if run_all:
        return datasets
    return {}


def _resolve_years(config: dict, years: tuple[int, ...], start_year: int | None, end_year: int | None) -> list[int]:
    y0, y1 = config["temporal_range"]
    return resolve_year_list(
        years,
        start_year=start_year,
        end_year=end_year,
        default_start=y0,
        default_end=y1,
        label="permafrost download years",
    )


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


def download_permafrost(
    dataset_keys: tuple[str, ...] = (),
    run_all: bool = False,
    years: tuple[int, ...] = (),
    start_year: int | None = None,
    end_year: int | None = None,
    raw_dir: Path = DEFAULT_RAW_DIR,
    overwrite: bool = False,
) -> dict[str, dict[int, Path]]:
    config = load_config()
    datasets = _resolve_datasets(config, dataset_keys, run_all)
    year_list = _resolve_years(config, years, start_year, end_year)
    if not datasets or not year_list:
        return {}

    result: dict[str, dict[int, Path]] = {}
    for key, info in datasets.items():
        base_url = str(info["base_url"]).rstrip("/") + "/"
        filename_template = info["filename_template"]
        result[key] = {}
        for year in year_list:
            fname = filename_template.format(year=year)
            url = urljoin(base_url, f"{year}/{fname}")
            out_path = raw_dir / key / f"{year}.nc"
            try:
                _download_file(url, out_path, overwrite=overwrite)
                result[key][year] = out_path
            except Exception as e:
                logger.warning("Download failed for %s %d: %s", key, year, e)
    return result


@click.command()
@click.option("--datasets", "-d", multiple=True, help="Dataset keys to download.")
@click.option("--all", "run_all", is_flag=True, help="Download all configured permafrost datasets.")
@click.option("--years", "-y", multiple=True, type=int, help="Specific years.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--overwrite", is_flag=True, help="Re-download existing files.")
def main(
    datasets: tuple[str, ...],
    run_all: bool,
    years: tuple[int, ...],
    start_year: int | None,
    end_year: int | None,
    overwrite: bool,
):
    if not datasets and not run_all:
        click.echo("Specify --datasets or --all")
        return

    out = download_permafrost(
        dataset_keys=datasets,
        run_all=run_all,
        years=years,
        start_year=start_year,
        end_year=end_year,
        overwrite=overwrite,
    )
    n = sum(len(v) for v in out.values())
    click.echo(f"Downloaded {n} file(s) into {DEFAULT_RAW_DIR}")


if __name__ == "__main__":
    main()
