"""Deprecated: country_mask is no longer part of the curated final dataset."""

from src.utils import get_logger

logger = get_logger("processing.country_mask")

def create_country_mask():
    raise SystemExit(
        "country_mask has been removed from the curated data/final dataset. "
        "Use vector boundaries or build an in-memory regionmask mask where needed."
    )

if __name__ == "__main__":
    create_country_mask()
