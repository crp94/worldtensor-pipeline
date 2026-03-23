"""Shared year-range policy helpers for WorldTensor."""

from __future__ import annotations

from collections.abc import Iterable

from src.grid import YEAR_END, YEAR_START


def resolve_year_bounds(
    start_year: int | None = None,
    end_year: int | None = None,
    *,
    default_start: int | None = None,
    default_end: int | None = None,
    floor: int = YEAR_START,
    ceiling: int = YEAR_END,
    label: str = "year range",
) -> tuple[int, int]:
    """Resolve and clamp a year range to the global WorldTensor policy."""

    if start_year is None:
        if default_start is None:
            raise ValueError(f"{label}: missing start year")
        resolved_start = int(default_start)
    else:
        resolved_start = int(start_year)

    if end_year is None:
        if default_end is None:
            raise ValueError(f"{label}: missing end year")
        resolved_end = int(default_end)
    else:
        resolved_end = int(end_year)

    resolved_start = max(resolved_start, int(floor))
    resolved_end = min(resolved_end, int(ceiling))

    if resolved_end < resolved_start:
        raise ValueError(
            f"{label}: no valid years remain after clamping to {floor}-{ceiling}"
        )

    return resolved_start, resolved_end


def filter_years(
    years: Iterable[int],
    *,
    floor: int = YEAR_START,
    ceiling: int = YEAR_END,
    source_start: int | None = None,
    source_end: int | None = None,
) -> list[int]:
    """Filter explicit years to the allowed WorldTensor policy window."""

    lower = max(int(floor), int(source_start)) if source_start is not None else int(floor)
    upper = min(int(ceiling), int(source_end)) if source_end is not None else int(ceiling)
    if upper < lower:
        return []

    return sorted({int(y) for y in years if lower <= int(y) <= upper})


def resolve_year_list(
    years: Iterable[int] | None = None,
    *,
    start_year: int | None = None,
    end_year: int | None = None,
    default_start: int | None = None,
    default_end: int | None = None,
    floor: int = YEAR_START,
    ceiling: int = YEAR_END,
    label: str = "year range",
) -> list[int]:
    """Resolve either an explicit year list or a bounded inclusive year span."""

    if years:
        return filter_years(
            years,
            floor=floor,
            ceiling=ceiling,
            source_start=default_start,
            source_end=default_end,
        )

    y0, y1 = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=default_start,
        default_end=default_end,
        floor=floor,
        ceiling=ceiling,
        label=label,
    )
    return list(range(y0, y1 + 1))
