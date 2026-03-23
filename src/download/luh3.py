"""Download and stage LUH3 v3.1.1 input4MIPs files into data/raw/luh3.

LUH3 historical data is distributed on input4MIPs as four files:
  - states
  - transitions
  - management
  - static

This script supports two paths:
  1) Parse ESGF-generated wget scripts and download files directly in Python.
  2) Stage already downloaded files from disk.

Canonical staged outputs:
  data/raw/luh3/{states,transitions,management,static}.nc

Usage:
    # Download directly from one or more ESGF wget scripts
    python -m src.download.luh3 --all --wget-script /path/to/wget_states.sh --wget-script /path/to/wget_transitions.sh

    # Or stage files you already downloaded manually
    python -m src.download.luh3 --all --source-dir ~/Downloads/luh3
"""

from __future__ import annotations

import hashlib
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import click
import requests
import yaml
from tqdm import tqdm

from src.utils import get_logger

logger = get_logger("download.luh3")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "luh3.yml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "luh3"
KIND_ORDER = ("states", "transitions", "management", "static")
CORE_KIND_ORDER = ("states", "transitions")
CHUNK_SIZE = 8 * 1024 * 1024  # 8 MB
MAX_RETRIES = 5
REQUEST_TIMEOUT = 180
DEFAULT_ESGF_SEARCH_URL = "https://esgf-node.llnl.gov/esg-search/search/"
DEFAULT_ESGF_SEARCH_URLS = (
    "https://esgf-node.llnl.gov/esg-search/search/",
    "https://aims2.llnl.gov/esg-search/search/",
)


def _normalize_kinds(kinds: list[str] | tuple[str, ...] | None) -> tuple[str, ...]:
    """Return a validated kind-order tuple."""
    if kinds is None:
        return KIND_ORDER
    out: list[str] = []
    for kind in kinds:
        if kind not in KIND_ORDER:
            raise click.ClickException(f"Unknown LUH3 kind: {kind}")
        if kind not in out:
            out.append(kind)
    if not out:
        raise click.ClickException("No LUH3 kinds selected.")
    return tuple(out)


def load_luh3_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def _discover_candidates(search_dirs: list[Path], pattern: str) -> list[Path]:
    matches: list[Path] = []
    seen = set()
    for directory in search_dirs:
        if not directory.exists():
            continue
        for path in directory.glob(pattern):
            if path.is_file():
                resolved = path.resolve()
                if resolved not in seen:
                    matches.append(resolved)
                    seen.add(resolved)
    return matches


def _pick_candidate(kind: str, candidates: list[Path]) -> Path | None:
    if not candidates:
        return None
    if len(candidates) == 1:
        return candidates[0]

    # Prefer newest file if multiple matches exist.
    chosen = max(candidates, key=lambda p: p.stat().st_mtime)
    logger.warning("Multiple %s candidates found; selecting newest: %s", kind, chosen)
    return chosen


def _default_search_dirs(source_dir: Path | None) -> list[Path]:
    if source_dir:
        return [source_dir]
    return [
        Path.cwd(),
        Path.home() / "Downloads",
        DEFAULT_OUTPUT_DIR,
    ]


def _kind_from_text(text: str) -> str | None:
    s = text.lower()
    if "multiple-transitions" in s or "landtransitions" in s:
        return "transitions"
    if "multiple-management" in s or "landmanagement" in s:
        return "management"
    if "multiple-static" in s or "landstatic" in s:
        return "static"
    if "multiple-states" in s or "landstate" in s:
        return "states"
    return None


def parse_wget_script(script_path: Path) -> list[dict[str, str]]:
    """Extract file/url/checksum tuples from ESGF wget script body."""
    text = script_path.read_text()
    pattern = re.compile(r"'([^']+)'\s+'([^']+)'\s+'([^']+)'\s+'([^']+)'")

    entries: list[dict[str, str]] = []
    for file_name, url, chksum_type, chksum in pattern.findall(text):
        kind = _kind_from_text(f"{file_name} {url}")
        if not kind:
            continue
        entries.append(
            {
                "kind": kind,
                "file": file_name,
                "url": url,
                "checksum_type": chksum_type.lower(),
                "checksum": chksum.lower(),
            }
        )
    return entries


def _extract_httpserver_url(url_entries: list[str] | None) -> str | None:
    if not url_entries:
        return None
    for entry in url_entries:
        parts = entry.split("|")
        if len(parts) >= 3 and parts[2].upper() == "HTTPSERVER":
            return parts[0]
    return None


def _normalize_search_urls(esgf_search_url: str | list[str] | tuple[str, ...]) -> list[str]:
    if isinstance(esgf_search_url, str):
        urls = [u.strip() for u in esgf_search_url.split(",") if u.strip()]
    else:
        urls = [str(u).strip() for u in esgf_search_url if str(u).strip()]
    return urls or list(DEFAULT_ESGF_SEARCH_URLS)


def discover_luh3_entries(
    esgf_search_url: str | list[str] | tuple[str, ...] = DEFAULT_ESGF_SEARCH_URLS,
    timeout: int = REQUEST_TIMEOUT,
) -> dict[str, dict[str, str]]:
    """Discover LUH3 file URLs and checksums directly from ESGF search API."""
    cfg = load_luh3_config()
    params = {
        "type": "File",
        "project": cfg["project"],
        "institution_id": cfg["institution_id"],
        "source_id": "UofMD-landState-3-1-1",
        "source_version": cfg["source_version"],
        "latest": "true",
        "format": "application/solr+json",
        "limit": "1000",
    }

    payload = None
    last_error: Exception | None = None
    for url in _normalize_search_urls(esgf_search_url):
        try:
            resp = requests.get(url, params=params, timeout=timeout)
            resp.raise_for_status()
            payload = resp.json()
            if payload.get("response", {}).get("docs"):
                break
        except Exception as e:
            last_error = e
            logger.warning("ESGF discovery failed via %s: %s", url, e)

    if payload is None or not payload.get("response", {}).get("docs"):
        if last_error:
            raise click.ClickException(f"Could not discover LUH3 files from ESGF: {last_error}")
        raise click.ClickException("Could not discover LUH3 files from ESGF.")

    selected: dict[str, dict[str, str]] = {}
    selected_ts: dict[str, int] = {}

    for doc in payload.get("response", {}).get("docs", []):
        kind = _kind_from_text(" ".join(str(doc.get(k, "")) for k in ("id", "title", "dataset_id")))
        if not kind:
            continue

        url = _extract_httpserver_url(doc.get("url"))
        if not url:
            continue

        checksum_values = doc.get("checksum", [])
        checksum_type_values = doc.get("checksum_type", [])
        checksum = str(checksum_values[0]).lower() if checksum_values else ""
        checksum_type = str(checksum_type_values[0]).lower() if checksum_type_values else ""
        raw_ts = doc.get("_timestamp", 0)
        if isinstance(raw_ts, (int, float)):
            timestamp = int(raw_ts)
        elif isinstance(raw_ts, str):
            try:
                timestamp = int(datetime.fromisoformat(raw_ts.replace("Z", "+00:00")).timestamp())
            except ValueError:
                timestamp = 0
        else:
            timestamp = 0

        # Keep newest document for each kind.
        if kind in selected and selected_ts.get(kind, 0) >= timestamp:
            continue

        selected[kind] = {
            "kind": kind,
            "file": str(doc.get("title", "")),
            "url": url,
            "checksum_type": checksum_type,
            "checksum": checksum,
        }
        selected_ts[kind] = timestamp

    return selected


def _checksum_file(path: Path, checksum_type: str) -> str:
    if checksum_type not in {"sha256", "md5"}:
        raise ValueError(f"Unsupported checksum type: {checksum_type}")
    h = hashlib.sha256() if checksum_type == "sha256" else hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(CHUNK_SIZE), b""):
            h.update(chunk)
    return h.hexdigest()


def download_one(
    url: str,
    out_path: Path,
    overwrite: bool = False,
    max_retries: int = MAX_RETRIES,
    timeout: int = REQUEST_TIMEOUT,
) -> Path:
    """Download one file with HTTP Range resume support."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = out_path.with_suffix(out_path.suffix + ".partial")

    if out_path.exists() and not overwrite:
        return out_path

    if overwrite:
        out_path.unlink(missing_ok=True)
        part_path.unlink(missing_ok=True)

    for attempt in range(1, max_retries + 1):
        resume_byte = part_path.stat().st_size if part_path.exists() else 0
        headers = {"Range": f"bytes={resume_byte}-"} if resume_byte > 0 else {}
        mode = "ab" if resume_byte > 0 else "wb"

        try:
            with requests.get(url, stream=True, headers=headers, timeout=timeout) as resp:
                resp.raise_for_status()

                if resume_byte > 0 and resp.status_code != 206:
                    # Server ignored Range; restart from scratch.
                    part_path.unlink(missing_ok=True)
                    resume_byte = 0
                    mode = "wb"

                total = int(resp.headers.get("content-length", 0))
                if resume_byte > 0 and resp.status_code == 206:
                    total += resume_byte

                with (
                    open(part_path, mode) as f,
                    tqdm(
                        total=total if total > 0 else None,
                        initial=resume_byte,
                        unit="B",
                        unit_scale=True,
                        desc=out_path.name,
                        leave=False,
                    ) as pbar,
                ):
                    for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
                        if not chunk:
                            continue
                        f.write(chunk)
                        pbar.update(len(chunk))

            part_path.rename(out_path)
            return out_path

        except Exception as e:
            logger.warning("Download attempt %d/%d failed for %s: %s", attempt, max_retries, out_path.name, e)
            if attempt == max_retries:
                raise
            time.sleep(min(5 * (2 ** (attempt - 1)), 60))

    return out_path


def download_from_wget_scripts(
    wget_scripts: list[Path],
    raw_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
    verify_checksum: bool = True,
    kinds: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Path], list[str]]:
    """Download LUH3 files described by ESGF wget scripts."""
    selected: dict[str, dict[str, str]] = {}
    for script in wget_scripts:
        entries = parse_wget_script(script)
        if not entries:
            logger.warning("No LUH3 entries found in script: %s", script)
            continue
        for entry in entries:
            kind = entry["kind"]
            if kind in selected:
                logger.warning("Duplicate %s entry encountered; keeping first from %s", kind, script)
                continue
            selected[kind] = entry

    return _download_selected(
        selected=selected,
        raw_dir=raw_dir,
        overwrite=overwrite,
        verify_checksum=verify_checksum,
        checksum_optional=False,
        kinds=kinds,
    )


def download_from_esgf_discovery(
    raw_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
    verify_checksum: bool = True,
    esgf_search_url: str | list[str] | tuple[str, ...] = DEFAULT_ESGF_SEARCH_URLS,
    kinds: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Path], list[str]]:
    """Discover and download LUH3 files directly from ESGF search API."""
    selected = discover_luh3_entries(esgf_search_url=esgf_search_url)
    return _download_selected(
        selected=selected,
        raw_dir=raw_dir,
        overwrite=overwrite,
        verify_checksum=verify_checksum,
        checksum_optional=True,
        kinds=kinds,
    )


def download_from_config_fallback_urls(
    raw_dir: Path = DEFAULT_OUTPUT_DIR,
    overwrite: bool = False,
    verify_checksum: bool = True,
    kinds: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Path], list[str]]:
    """Download LUH3 files from fallback direct URLs in config/luh3.yml."""
    cfg = load_luh3_config()
    direct_urls = cfg.get("direct_urls", {})
    selected: dict[str, dict[str, str]] = {}

    for kind in KIND_ORDER:
        url = str(direct_urls.get(kind, "")).strip()
        if not url:
            continue
        selected[kind] = {
            "kind": kind,
            "file": Path(url).name,
            "url": url,
            "checksum_type": "",
            "checksum": "",
        }

    if not selected:
        raise click.ClickException("No direct_urls configured for LUH3 fallback download.")

    return _download_selected(
        selected=selected,
        raw_dir=raw_dir,
        overwrite=overwrite,
        verify_checksum=verify_checksum,
        checksum_optional=True,
        kinds=kinds,
    )


def _download_selected(
    selected: dict[str, dict[str, str]],
    raw_dir: Path,
    overwrite: bool,
    verify_checksum: bool,
    checksum_optional: bool,
    kinds: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Path], list[str]]:
    cfg = load_luh3_config()
    raw_dir.mkdir(parents=True, exist_ok=True)
    canonical = cfg["raw_files"]
    wanted_kinds = _normalize_kinds(kinds)

    staged: dict[str, Path] = {}
    missing: list[str] = []

    for kind in wanted_kinds:
        out_path = raw_dir / canonical[kind]
        if out_path.exists() and not overwrite:
            if out_path.stat().st_size > 0:
                staged[kind] = out_path
                continue
            logger.warning("Removing zero-byte staged file and re-downloading: %s", out_path)
            out_path.unlink(missing_ok=True)

        entry = selected.get(kind)
        if not entry:
            missing.append(kind)
            continue

        click.echo(f"Downloading {kind}: {entry['url']}")
        result = download_one(entry["url"], out_path, overwrite=overwrite)

        if verify_checksum:
            checksum_type = entry.get("checksum_type", "").lower()
            checksum = entry.get("checksum", "").lower()
            if checksum and checksum_type in {"sha256", "md5"}:
                actual = _checksum_file(result, checksum_type)
                if actual != checksum:
                    result.unlink(missing_ok=True)
                    raise click.ClickException(
                        f"Checksum mismatch for {kind}: expected {checksum}, got {actual}"
                    )
                logger.info("Checksum verified for %s (%s)", kind, checksum_type)
            elif not checksum_optional:
                raise click.ClickException(f"Missing checksum metadata for required file: {kind}")
            else:
                logger.warning("Checksum metadata not available for %s; skipping verification.", kind)

        staged[kind] = result

    return staged, missing


def ingest_luh3_files(
    raw_dir: Path = DEFAULT_OUTPUT_DIR,
    source_dir: Path | None = None,
    explicit_paths: dict[str, Path | None] | None = None,
    overwrite: bool = False,
    kinds: list[str] | tuple[str, ...] | None = None,
) -> tuple[dict[str, Path], list[str]]:
    """Locate and copy the four LUH3 files to canonical raw paths.

    Returns
    -------
    tuple
        (staged_files_by_kind, missing_kinds)
    """
    cfg = load_luh3_config()
    raw_dir.mkdir(parents=True, exist_ok=True)

    explicit_paths = explicit_paths or {}
    search_dirs = _default_search_dirs(source_dir)
    patterns = cfg["search_patterns"]
    filenames = cfg["raw_files"]
    wanted_kinds = _normalize_kinds(kinds)

    staged: dict[str, Path] = {}
    missing: list[str] = []

    for kind in wanted_kinds:
        out_path = raw_dir / filenames[kind]
        if out_path.exists() and not overwrite:
            if out_path.stat().st_size > 0:
                staged[kind] = out_path
                continue
            logger.warning("Removing zero-byte staged file: %s", out_path)
            out_path.unlink(missing_ok=True)

        src = explicit_paths.get(kind)
        if src is None:
            candidates = _discover_candidates(search_dirs, patterns[kind])
            src = _pick_candidate(kind, candidates)

        if src is None:
            missing.append(kind)
            continue
        if src.stat().st_size == 0:
            logger.warning("Ignoring zero-byte source candidate for %s: %s", kind, src)
            missing.append(kind)
            continue

        out_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, out_path)
        staged[kind] = out_path
        logger.info("Staged %s -> %s", kind, out_path)

    return staged, missing


def _print_download_instructions(kinds: list[str] | tuple[str, ...] | None = None) -> None:
    wanted_kinds = _normalize_kinds(kinds)
    click.echo("\nDownload instructions (manual):")
    click.echo("  1) Open https://aims2.llnl.gov/search")
    click.echo("  2) Filter Project = input4MIPs")
    click.echo("  3) Filter Institution ID = UofMD")
    click.echo("  4) Filter Source Version = 3.1.1")
    click.echo(f"  5) Download file(s): {', '.join(wanted_kinds)}")
    click.echo("  6) Optionally generate ESGF wget scripts and pass them via --wget-script")


@click.command()
@click.option("--all", "run_all", is_flag=True, help="Stage all required LUH3 files.")
@click.option("--wget-script", "wget_scripts", multiple=True, type=click.Path(exists=True),
              help="Path(s) to ESGF-generated wget scripts.")
@click.option("--discover/--no-discover", default=True, show_default=True,
              help="Auto-discover LUH3 file URLs from ESGF search API when wget scripts are not provided.")
@click.option(
    "--esgf-search-url",
    default=",".join(DEFAULT_ESGF_SEARCH_URLS),
    show_default=True,
    help="Comma-separated ESGF search API endpoint(s) used for LUH3 auto-discovery.",
)
@click.option("--source-dir", type=click.Path(exists=True, file_okay=False, dir_okay=True), default=None,
              help="Directory containing downloaded LUH3 NetCDF files.")
@click.option("--states-path", type=click.Path(exists=True), default=None, help="Explicit path to LUH3 states file.")
@click.option("--transitions-path", type=click.Path(exists=True), default=None,
              help="Explicit path to LUH3 transitions file.")
@click.option("--management-path", type=click.Path(exists=True), default=None,
              help="Explicit path to LUH3 management file.")
@click.option("--static-path", type=click.Path(exists=True), default=None, help="Explicit path to LUH3 static file.")
@click.option("--raw-dir", type=click.Path(), default=None, help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
@click.option("--overwrite", is_flag=True, help="Overwrite files already staged in raw dir.")
@click.option("--no-verify-checksum", is_flag=True, help="Skip checksum verification for network downloads.")
def main(
    run_all,
    wget_scripts,
    discover,
    esgf_search_url,
    source_dir,
    states_path,
    transitions_path,
    management_path,
    static_path,
    raw_dir,
    overwrite,
    no_verify_checksum,
):
    """Locate and stage LUH3 input4MIPs files for processing."""
    if not run_all:
        click.echo("Specify --all. Use --help for usage.")
        return

    out_dir = Path(raw_dir) if raw_dir else DEFAULT_OUTPUT_DIR
    source = Path(source_dir).expanduser().resolve() if source_dir else None

    explicit_paths = {
        "states": Path(states_path).resolve() if states_path else None,
        "transitions": Path(transitions_path).resolve() if transitions_path else None,
        "management": Path(management_path).resolve() if management_path else None,
        "static": Path(static_path).resolve() if static_path else None,
    }
    wanted_kinds = CORE_KIND_ORDER

    staged: dict[str, Path] = {}
    missing: list[str] = []

    any_explicit = any(explicit_paths.values())

    if wget_scripts:
        script_paths = [Path(p).expanduser().resolve() for p in wget_scripts]
        dl_staged, dl_missing = download_from_wget_scripts(
            wget_scripts=script_paths,
            raw_dir=out_dir,
            overwrite=overwrite,
            verify_checksum=not no_verify_checksum,
            kinds=wanted_kinds,
        )
        staged.update(dl_staged)
        missing = dl_missing
    elif discover and source is None and not any_explicit:
        click.echo(f"Discovering LUH3 files via ESGF API: {esgf_search_url}")
        try:
            dl_staged, dl_missing = download_from_esgf_discovery(
                raw_dir=out_dir,
                overwrite=overwrite,
                verify_checksum=not no_verify_checksum,
                esgf_search_url=esgf_search_url,
                kinds=wanted_kinds,
            )
        except click.ClickException as e:
            logger.warning("ESGF discovery failed; trying direct URL fallback: %s", e)
            dl_staged, dl_missing = download_from_config_fallback_urls(
                raw_dir=out_dir,
                overwrite=overwrite,
                verify_checksum=not no_verify_checksum,
                kinds=wanted_kinds,
            )
        staged.update(dl_staged)
        missing = dl_missing

    staged2, missing2 = ingest_luh3_files(
        raw_dir=out_dir,
        source_dir=source,
        explicit_paths=explicit_paths,
        overwrite=overwrite,
        kinds=wanted_kinds,
    )
    staged.update(staged2)

    if missing:
        missing = [k for k in missing if k not in staged]
    missing = sorted(set(missing + [k for k in missing2 if k not in staged]))

    click.echo(f"Staged files: {len(staged)}/{len(wanted_kinds)}")
    for kind in wanted_kinds:
        if kind in staged:
            p = staged[kind]
            click.echo(f"  {kind:<12} {p} ({p.stat().st_size / 1e9:.2f} GB)")

    if missing:
        click.echo(f"Missing files: {missing}")
        _print_download_instructions(wanted_kinds)
        raise click.ClickException("Required LUH3 files are missing.")


if __name__ == "__main__":
    main()
