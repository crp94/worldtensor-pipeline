"""Canonical output-path registry helpers for the WorldTensor data layout."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_REGISTRY_PATH = PROJECT_ROOT / "config" / "variables.yml"
DEFAULT_FINAL_ROOT = PROJECT_ROOT / "data" / "final"


@dataclass(frozen=True)
class VariableSpec:
    """Registry entry for one published WorldTensor variable."""

    canonical_id: str
    relative_dir: str
    relative_file: str | None
    legacy_relative_paths: tuple[str, ...]
    domain: str
    paper_domain: str
    source_family: str
    long_name: str
    units: str
    is_static: bool
    expected_time_range: tuple[int, int] | None
    display_name: str | None = None
    source_url: str | None = None
    stage2_friendly_id: str | None = None
    status: str = "active"
    static_group: str | None = None
    notes: str = ""

    @property
    def relative_path(self) -> str:
        if self.relative_file:
            return self.relative_file
        return self.relative_dir


def _coerce_time_range(value) -> tuple[int, int] | None:
    if not value:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])
    raise ValueError(f"Invalid expected_time_range: {value!r}")


def _normalize_path_list(values) -> tuple[str, ...]:
    if not values:
        return ()
    return tuple(str(v).strip("/") for v in values if str(v).strip("/"))


def _load_yaml(path: Path) -> dict:
    with open(path) as f:
        return yaml.safe_load(f) or {}


@lru_cache(maxsize=4)
def _load_registry_cached(path_str: str) -> dict[str, VariableSpec]:
    data = _load_yaml(Path(path_str))
    variables = data.get("variables", {})
    registry: dict[str, VariableSpec] = {}

    for canonical_id, raw in variables.items():
        relative_dir = str(raw.get("relative_dir", "")).strip("/")
        relative_file = raw.get("relative_file")
        if relative_file:
            relative_file = str(relative_file).strip("/")

        spec = VariableSpec(
            canonical_id=str(raw.get("canonical_id", canonical_id)),
            relative_dir=relative_dir,
            relative_file=relative_file,
            legacy_relative_paths=_normalize_path_list(raw.get("legacy_relative_paths") or raw.get("legacy_relative_dirs")),
            domain=str(raw.get("domain", "")),
            paper_domain=str(raw.get("paper_domain", raw.get("domain", ""))),
            source_family=str(raw.get("source_family", "")),
            long_name=str(raw.get("long_name", "")),
            units=str(raw.get("units", "")),
            is_static=bool(raw.get("is_static", False)),
            expected_time_range=_coerce_time_range(raw.get("expected_time_range")),
            display_name=raw.get("display_name"),
            source_url=raw.get("source_url"),
            stage2_friendly_id=raw.get("stage2_friendly_id"),
            status=str(raw.get("status", "active")),
            static_group=raw.get("static_group"),
            notes=str(raw.get("notes", "")),
        )
        registry[spec.canonical_id] = spec

    return registry


def load_registry(path: Path | None = None) -> dict[str, VariableSpec]:
    """Load the canonical variable registry."""

    registry_path = Path(path or DEFAULT_REGISTRY_PATH).resolve()
    return _load_registry_cached(str(registry_path))


def list_variable_specs(path: Path | None = None, include_deprecated: bool = True) -> list[VariableSpec]:
    """Return all registered variables."""

    specs = list(load_registry(path).values())
    if include_deprecated:
        return sorted(specs, key=lambda s: s.canonical_id)
    return sorted((s for s in specs if s.status == "active"), key=lambda s: s.canonical_id)


def canonical_ids(path: Path | None = None, include_deprecated: bool = True) -> list[str]:
    """Return canonical ids from the registry."""

    return [spec.canonical_id for spec in list_variable_specs(path, include_deprecated=include_deprecated)]


def get_variable_spec(canonical_id: str, path: Path | None = None) -> VariableSpec:
    """Return the registry entry for a canonical variable id."""

    registry = load_registry(path)
    try:
        return registry[canonical_id]
    except KeyError as exc:
        raise KeyError(f"Unknown canonical variable id: {canonical_id}") from exc


def is_static_variable(canonical_id: str, path: Path | None = None) -> bool:
    """Return whether a canonical variable is static."""

    return get_variable_spec(canonical_id, path).is_static


def output_dir_for(canonical_id: str, base_dir: Path | None = None, path: Path | None = None) -> Path:
    """Return the canonical output directory for a variable."""

    spec = get_variable_spec(canonical_id, path)
    root = Path(base_dir or DEFAULT_FINAL_ROOT)
    return root / spec.relative_dir


def output_path_for(
    canonical_id: str,
    year: int | None = None,
    base_dir: Path | None = None,
    path: Path | None = None,
) -> Path:
    """Return the canonical output file path for a variable."""

    spec = get_variable_spec(canonical_id, path)
    root = Path(base_dir or DEFAULT_FINAL_ROOT)
    if spec.is_static:
        filename = spec.relative_file or str(Path(spec.relative_dir) / f"{spec.canonical_id}.nc")
        return root / filename
    if year is None:
        raise ValueError(f"year is required for annual variable '{canonical_id}'")
    return root / spec.relative_dir / f"{int(year)}.nc"


def legacy_paths_for(canonical_id: str, base_dir: Path | None = None, path: Path | None = None) -> list[Path]:
    """Return legacy on-disk paths for a variable."""

    spec = get_variable_spec(canonical_id, path)
    root = Path(base_dir or DEFAULT_FINAL_ROOT)
    return [root / rel for rel in spec.legacy_relative_paths]


def candidate_paths_for(canonical_id: str, base_dir: Path | None = None, path: Path | None = None) -> list[Path]:
    """Return canonical and legacy candidate locations for a variable."""

    spec = get_variable_spec(canonical_id, path)
    candidates = [output_path_for(canonical_id, base_dir=base_dir, path=path) if spec.is_static else output_dir_for(canonical_id, base_dir=base_dir, path=path)]
    for legacy in legacy_paths_for(canonical_id, base_dir=base_dir, path=path):
        if legacy not in candidates:
            candidates.append(legacy)
    return candidates


@lru_cache(maxsize=4)
def _legacy_index_cached(path_str: str) -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical_id, spec in load_registry(Path(path_str)).items():
        index[spec.relative_path.strip("/")] = canonical_id
        for legacy in spec.legacy_relative_paths:
            index[legacy.strip("/")] = canonical_id
    return index


def canonical_id_for_path(relative_path: str | Path, path: Path | None = None) -> str | None:
    """Resolve a canonical id from a canonical or legacy relative path."""

    registry_path = Path(path or DEFAULT_REGISTRY_PATH).resolve()
    rel = str(relative_path).strip("/")
    return _legacy_index_cached(str(registry_path)).get(rel)


def validate_registry(path: Path | None = None) -> list[str]:
    """Return registry validation errors."""

    errors: list[str] = []
    seen_current: dict[str, str] = {}
    seen_legacy: dict[str, str] = {}

    for canonical_id, spec in load_registry(path).items():
        if not spec.long_name:
            errors.append(f"{canonical_id}: missing long_name")
        if not spec.units:
            errors.append(f"{canonical_id}: missing units")
        if not spec.relative_dir:
            errors.append(f"{canonical_id}: missing relative_dir")

        rel = spec.relative_path
        owner = seen_current.get(rel)
        if owner and owner != canonical_id:
            errors.append(f"duplicate target path '{rel}' for {owner} and {canonical_id}")
        seen_current[rel] = canonical_id

        for legacy in spec.legacy_relative_paths:
            owner = seen_legacy.get(legacy)
            if owner and owner != canonical_id:
                errors.append(f"duplicate legacy path '{legacy}' for {owner} and {canonical_id}")
            seen_legacy[legacy] = canonical_id

    return errors


def active_specs(path: Path | None = None) -> Iterable[VariableSpec]:
    """Iterate active registry entries."""

    for spec in list_variable_specs(path, include_deprecated=False):
        yield spec
