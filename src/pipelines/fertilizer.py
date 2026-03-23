"""Fertilizer Pipeline — Coello et al. (2024) Global Crop-Specific Fertilization.

Processes 5-arcminute yearly fertilizer application-rate rasters from Decorte
et al. (2024, Figshare) into WorldTensor's 0.25° master grid.

Data source
-----------
Decorte, T., Janssens, I., Coello Sanz, F., & Mortier, S. (2024).
Fertilizer application rate maps per crop and year.
Figshare. Dataset. https://doi.org/10.6084/m9.figshare.25435432.v3
License: CC0

Companion paper:
Coello Sanz, F. et al. (2024). Global Crop-Specific Fertilization Dataset
from 1961–2019. Nature Scientific Data.
https://doi.org/10.1038/s41597-024-04215-x

Processing steps
----------------
1. For each year (1961–2019), discover all per-crop GeoTIFF rasters for each
   nutrient (N, P2O5, K2O).
2. Sum across all 13 crop groups to produce a single total application-rate
   layer per nutrient per year.
3. Regrid from 5-arcminute to the 0.25° master grid via bilinear interpolation.
4. Save as CF-compliant annual NetCDF files under data/final/agriculture/.
5. Generate diagnostic map and time-series plots.

Output variables
----------------
- fertilizer_nitrogen   : Total N application rate   [kg ha⁻¹ yr⁻¹]
- fertilizer_phosphorus : Total P₂O₅ application rate [kg ha⁻¹ yr⁻¹]
- fertilizer_potassium  : Total K₂O application rate  [kg ha⁻¹ yr⁻¹]

Usage
-----
    python -m src.pipelines.fertilizer --all
    python -m src.pipelines.fertilizer --all --overwrite
    python -m src.pipelines.fertilizer --process --plot
    python -m src.pipelines.fertilizer --nutrient nitrogen --process
"""

from __future__ import annotations

import gc
import re
from pathlib import Path

import click
import numpy as np
import rioxarray
import xarray as xr

from src.processing.raster_to_grid import regrid_raster
from src.utils import get_logger, plot_global_map, plot_time_series, save_annual_variable

logger = get_logger("pipeline.fertilizer")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "fertilizer"
EXTRACT_DIR = RAW_DIR / "extracted"
PLOTS_DIR = PROJECT_ROOT / "plots" / "fertilizer"

# ---------------------------------------------------------------------------
# Nutrient definitions.
#
# Each nutrient maps to:
#   canonical_id  – the variable name registered in config/variables.yml
#   file_token    – substring used to identify the correct TIFFs in the archive
#   long_name     – CF-compliant long_name attribute
#   units         – CF-compliant units string
#   cmap          – matplotlib colormap for diagnostic plots
#   color         – line colour for the time-series plot
# ---------------------------------------------------------------------------
NUTRIENTS: dict[str, dict] = {
    "nitrogen": {
        "canonical_id": "fertilizer_nitrogen",
        "file_token": "_N_",
        "long_name": "Total nitrogen fertilizer application rate",
        "units": "kg ha-1 yr-1",
        "cmap": "YlOrBr",
        "color": "#d62728",
    },
    "phosphorus": {
        "canonical_id": "fertilizer_phosphorus",
        "file_token": "_P2O5_",
        "long_name": "Total phosphorus (P2O5) fertilizer application rate",
        "units": "kg ha-1 yr-1",
        "cmap": "YlGnBu",
        "color": "#2ca02c",
    },
    "potassium": {
        "canonical_id": "fertilizer_potassium",
        "file_token": "_K2O_",
        "long_name": "Total potassium (K2O) fertilizer application rate",
        "units": "kg ha-1 yr-1",
        "cmap": "PuRd",
        "color": "#9467bd",
    },
}

YEAR_RE = re.compile(r"(19|20)\d{2}")
YEAR_RANGE = range(1961, 2020)  # 1961–2019 inclusive


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _discover_tiffs(nutrient_key: str) -> dict[int, list[Path]]:
    """Find all per-crop TIFFs for a nutrient, grouped by year.

    Scans EXTRACT_DIR recursively for GeoTIFFs whose filename contains the
    nutrient's file_token (e.g. ``_N_``) and a valid four-digit year.

    Returns
    -------
    dict mapping year (int) -> list of Path objects (one per crop group).
    """
    token = NUTRIENTS[nutrient_key]["file_token"]
    year_to_paths: dict[int, list[Path]] = {}

    for tif in sorted([*EXTRACT_DIR.rglob("*.tif"), *EXTRACT_DIR.rglob("*.tiff")]):
        name = tif.name
        if token not in name:
            continue
        m = YEAR_RE.search(name)
        if not m:
            continue
        year = int(m.group(0))
        if year not in YEAR_RANGE:
            continue
        year_to_paths.setdefault(year, []).append(tif)

    return year_to_paths


def _sum_crop_tiffs(paths: list[Path]) -> xr.DataArray:
    """Load and sum multiple per-crop GeoTIFFs into a single DataArray.

    Each GeoTIFF represents the fertilizer application rate for one crop group
    in a given year.  We sum across all crop groups to get the total nutrient
    application rate per grid cell.
    """
    total = None
    for p in paths:
        da = rioxarray.open_rasterio(p, masked=True).squeeze("band", drop=True)
        if total is None:
            total = da.astype("float64")
        else:
            # Treat NaN as zero when summing crops (a cell may grow some but
            # not all crops), but keep the cell as NaN only if ALL crops are NaN.
            total = total.fillna(0) + da.fillna(0)
            # Restore NaN where both were NaN
            all_nan = total.isnull() & da.isnull()
            total = total.where(~all_nan)
        da.close()
    return total


# ---------------------------------------------------------------------------
# Core processing
# ---------------------------------------------------------------------------

def process_nutrient(
    nutrient_key: str,
    overwrite: bool = False,
) -> int:
    """Process a single nutrient across all available years.

    Returns the number of yearly files written.
    """
    info = NUTRIENTS[nutrient_key]
    canonical_id = info["canonical_id"]

    logger.info("Discovering TIFFs for %s (token=%s)...", nutrient_key, info["file_token"])
    year_to_paths = _discover_tiffs(nutrient_key)

    if not year_to_paths:
        logger.warning("No TIFFs found for %s in %s", nutrient_key, EXTRACT_DIR)
        return 0

    logger.info(
        "Found %d years of %s data (%d–%d, %d crop TIFFs total)",
        len(year_to_paths),
        nutrient_key,
        min(year_to_paths),
        max(year_to_paths),
        sum(len(v) for v in year_to_paths.values()),
    )

    written = 0
    for year in sorted(year_to_paths):
        crop_paths = year_to_paths[year]
        logger.info("  %s %d: summing %d crop rasters...", nutrient_key, year, len(crop_paths))

        # Sum across crop groups
        da_total = _sum_crop_tiffs(crop_paths)
        da_total.name = canonical_id
        da_total.attrs["units"] = info["units"]
        da_total.attrs["long_name"] = info["long_name"]

        # Regrid from 5-arcmin to 0.25° master grid
        ds_out = regrid_raster(da_total, year=year, var_name=canonical_id, method="linear")

        # Save
        save_annual_variable(ds_out, canonical_id, year, source_var=canonical_id)
        written += 1

        da_total.close()
        ds_out.close()
        gc.collect()

    logger.info("Processed %s: %d yearly files written", nutrient_key, written)
    return written


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_nutrient(nutrient_key: str, map_every: int = 10) -> None:
    """Generate diagnostic map and time-series plots for one nutrient."""
    from src.data_layout import output_path_for

    info = NUTRIENTS[nutrient_key]
    canonical_id = info["canonical_id"]

    # Locate processed files
    # Infer the output directory from the first year's path
    sample_path = output_path_for(canonical_id, year=2000)
    data_dir = sample_path.parent

    nc_files = sorted(data_dir.glob("*.nc"))
    if not nc_files:
        logger.warning("No processed files to plot for %s", nutrient_key)
        return

    years = sorted(int(p.stem) for p in nc_files)

    # Map plots for selected years
    selected_years = sorted({years[0], years[-1], *[y for y in years if y % map_every == 0]})
    for year in selected_years:
        ds = xr.open_dataset(data_dir / f"{year}.nc")
        plot_global_map(
            ds[canonical_id],
            f"{info['long_name']} — {year}",
            PLOTS_DIR / f"{nutrient_key}_{year}.png",
            cmap=info["cmap"],
        )
        ds.close()

    # Time-series plot
    ts_years, ts_means = [], []
    for p in nc_files:
        ds = xr.open_dataset(p)
        ts_years.append(int(p.stem))
        ts_means.append(float(ds[canonical_id].mean(skipna=True)))
        ds.close()

    plot_time_series(
        ts_years,
        ts_means,
        f"Global Mean {info['long_name']}",
        info["units"],
        PLOTS_DIR / f"timeseries_{nutrient_key}.png",
        color=info["color"],
    )
    logger.info("Plots written for %s", nutrient_key)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--nutrient", "-n",
    multiple=True,
    type=click.Choice(list(NUTRIENTS.keys()), case_sensitive=False),
    help="Nutrient(s) to process. Default: all three.",
)
@click.option("--all", "run_all", is_flag=True, help="Run download + process + plot.")
@click.option("--download", "do_download", is_flag=True, help="Run download step.")
@click.option("--process", "do_process", is_flag=True, help="Run processing step.")
@click.option("--plot", "do_plot", is_flag=True, help="Run plotting step.")
@click.option("--map-every", type=int, default=10, show_default=True, help="Plot map every N years.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
def main(
    nutrient: tuple[str, ...],
    run_all: bool,
    do_download: bool,
    do_process: bool,
    do_plot: bool,
    map_every: int,
    overwrite: bool,
) -> None:
    """Run the fertilizer data pipeline (Coello et al. 2024, Figshare CC0)."""
    from src.download.fertilizer import download_fertilizer, extract_fertilizer

    selected = list(nutrient) if nutrient else list(NUTRIENTS.keys())

    if run_all:
        do_download = do_process = do_plot = True

    if not any([do_download, do_process, do_plot]):
        raise click.ClickException("Specify --all or at least one of --download/--process/--plot.")

    if do_download:
        download_fertilizer(overwrite=overwrite)
        extract_fertilizer(overwrite=overwrite)

    if do_process:
        total = 0
        for nk in selected:
            total += process_nutrient(nk, overwrite=overwrite)
        logger.info("Total yearly files written: %d", total)

    if do_plot:
        for nk in selected:
            plot_nutrient(nk, map_every=map_every)
        logger.info("Plots written under: %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
