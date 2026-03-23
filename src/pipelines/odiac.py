"""ODIAC Pipeline — Fossil Fuel CO2 Emissions (ODIAC2024, NIES).

Downloads, processes, and plots the ODIAC 1° monthly fossil-fuel CO2 emission
grids, aggregating them to yearly statistics on the WorldTensor 0.25° grid.

Data source
-----------
Oda, T. and Maksyutov, S. (2015).
ODIAC Fossil Fuel CO2 Emissions Dataset (Version ODIAC2024).
Center for Global Environmental Research, NIES.
DOI: 10.17595/20170411.001
License: CC-BY 4.0

References
----------
Oda, T. and Maksyutov, S. (2011). A very high-resolution (1 km × 1 km)
global fossil fuel CO2 emission inventory derived using a point source
database and satellite observations of nighttime light.
Atmos. Chem. Phys., 11, 543–556. https://doi.org/10.5194/acp-11-543-2011

Oda, T., Maksyutov, S. and Andres, R. J. (2018). The Open-source Data
Inventory for Anthropogenic CO2, version 2016 (ODIAC2016).
Earth Syst. Sci. Data, 10, 87–107. https://doi.org/10.5194/essd-10-87-2018

Raw file structure
------------------
Each yearly NetCDF (180 lat × 360 lon, 1° resolution) contains:
    - ``land``        : CO2 from fossil fuel combustion, cement, gas flaring
    - ``intl_bunker`` : CO2 from international aviation/marine bunker
Dimension ``month`` runs 1–12.  Units are gC/m²/d.

Processing steps
----------------
1. Download 24 yearly NetCDF files from NIES (2000–2023, ~143 MB total).
2. For each year, compute ``total = land + intl_bunker``.
3. For each of the three source fields (land, intl_bunker, total), compute
   five yearly statistics over the 12 monthly layers:
       mean, sum, max, min, std
4. Regrid each statistic from 1° to 0.25° via bilinear interpolation.
5. Save as CF-compliant annual NetCDF under ``data/final/emissions/``.
6. Generate diagnostic map plots (first, last, and every 5 years) and
   time-series plots for every output variable.

Output variables (15 total)
---------------------------
    odiac_land_mean        odiac_intl_bunker_mean        odiac_total_mean
    odiac_land_sum         odiac_intl_bunker_sum         odiac_total_sum
    odiac_land_max         odiac_intl_bunker_max         odiac_total_max
    odiac_land_min         odiac_intl_bunker_min         odiac_total_min
    odiac_land_std         odiac_intl_bunker_std         odiac_total_std

Usage
-----
    python -m src.pipelines.odiac --all
    python -m src.pipelines.odiac --all --overwrite
    python -m src.pipelines.odiac --process --plot
    python -m src.pipelines.odiac --source land --stat mean sum --process
"""

from __future__ import annotations

import gc
from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml

from src.data_layout import output_path_for
from src.processing.raster_to_grid import regrid_raster
from src.utils import get_logger, plot_global_map, plot_time_series, save_annual_variable

logger = get_logger("pipeline.odiac")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "odiac.yml"
RAW_DIR = PROJECT_ROOT / "data" / "raw" / "odiac"
PLOTS_DIR = PROJECT_ROOT / "plots" / "odiac"

# Source fields to process and the statistics to compute for each.
SOURCE_FIELDS = ("land", "intl_bunker", "total")
STAT_NAMES = ("mean", "sum", "max", "min", "std")

# Plot settings per source field.
PLOT_SETTINGS: dict[str, dict] = {
    "land": {"cmap": "inferno", "color": "#d62728"},
    "intl_bunker": {"cmap": "cividis", "color": "#ff7f0e"},
    "total": {"cmap": "hot", "color": "#1f77b4"},
}


def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def _canonical_id(source: str, stat: str) -> str:
    """Build the canonical variable ID, e.g. ``odiac_land_mean``."""
    return f"odiac_{source}_{stat}"


def _compute_stat(monthly: xr.DataArray, stat: str) -> xr.DataArray:
    """Reduce 12 monthly layers to a single yearly statistic."""
    if stat == "mean":
        return monthly.mean(dim="month")
    if stat == "sum":
        return monthly.sum(dim="month")
    if stat == "max":
        return monthly.max(dim="month")
    if stat == "min":
        return monthly.min(dim="month")
    if stat == "std":
        return monthly.std(dim="month")
    raise ValueError(f"Unknown stat: {stat}")


def _regrid_monthly_stack(monthly: xr.DataArray, year: int) -> xr.DataArray:
    """Regrid each monthly layer to the master grid before yearly reduction.

    Computing min/mean/max after interpolation preserves the ordering
    relationship that can be broken when each reduced field is interpolated
    independently.
    """
    monthly = monthly.transpose("month", "lat", "lon")
    tmp_name = monthly.name or "data"
    frames: list[xr.DataArray] = []
    for idx in range(monthly.sizes["month"]):
        ds_out = regrid_raster(monthly.isel(month=idx), year=year, var_name=tmp_name, method="linear")
        frames.append(ds_out[tmp_name])
    stacked = xr.concat(frames, dim=monthly["month"])
    stacked.name = tmp_name
    stacked.attrs = monthly.attrs.copy()
    return stacked


def process_year(
    year: int,
    sources: tuple[str, ...] = SOURCE_FIELDS,
    stats: tuple[str, ...] = STAT_NAMES,
    overwrite: bool = False,
) -> int:
    """Process one ODIAC yearly file into regridded annual statistics.

    Returns the number of output files written.
    """
    config = load_config()
    out_vars = config["output_variables"]
    fname = config["file_pattern"].format(year=year)
    nc_path = RAW_DIR / fname

    if not nc_path.exists():
        logger.warning("Raw file missing: %s", nc_path)
        return 0

    ds = xr.open_dataset(nc_path)
    written = 0

    # Pre-compute total if needed.
    if "total" in sources:
        total_da = ds["land"] + ds["intl_bunker"]
        total_da.name = "total"
        total_da.attrs = {
            "units": "gC/m2/d",
            "long_name": "Total fossil fuel CO2 emissions (land + intl bunker)",
        }

    for source in sources:
        if source == "total":
            monthly = total_da
        else:
            if source not in ds.data_vars:
                logger.warning("Variable '%s' not found in %s", source, fname)
                continue
            monthly = ds[source]

        monthly_regridded = _regrid_monthly_stack(monthly, year)

        for stat in stats:
            cid = _canonical_id(source, stat)
            if cid not in out_vars:
                continue
            out_path = output_path_for(cid, year=year)
            if out_path.exists() and not overwrite:
                continue

            var_info = out_vars[cid]
            da_stat = _compute_stat(monthly_regridded, stat).astype(np.float32)

            # Attach metadata after regridding/reduction on the target grid.
            da_stat.name = cid
            da_stat.attrs["units"] = var_info["units"]
            da_stat.attrs["long_name"] = var_info["long_name"]

            save_annual_variable(da_stat.to_dataset(name=cid), cid, year, source_var=cid)
            written += 1
            gc.collect()

    ds.close()
    return written


def process_all(
    sources: tuple[str, ...] = SOURCE_FIELDS,
    stats: tuple[str, ...] = STAT_NAMES,
    overwrite: bool = False,
) -> int:
    """Process all available ODIAC years."""
    config = load_config()
    y0, y1 = config["temporal_range"]
    total = 0
    for year in range(y0, y1 + 1):
        n = process_year(year, sources=sources, stats=stats, overwrite=overwrite)
        total += n
        logger.info("Year %d: %d files written", year, n)
    return total


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_all(
    sources: tuple[str, ...] = SOURCE_FIELDS,
    stats: tuple[str, ...] = STAT_NAMES,
    map_every: int = 5,
) -> None:
    """Generate diagnostic maps and time-series plots for all output vars."""
    from src.data_layout import output_path_for

    config = load_config()
    out_vars = config["output_variables"]

    for source in sources:
        for stat in stats:
            cid = _canonical_id(source, stat)
            if cid not in out_vars:
                continue

            var_info = out_vars[cid]
            ps = PLOT_SETTINGS.get(source, PLOT_SETTINGS["total"])

            # Discover processed files.
            sample = output_path_for(cid, year=2000)
            data_dir = sample.parent
            nc_files = sorted(data_dir.glob("*.nc"))
            if not nc_files:
                logger.warning("No files to plot for %s", cid)
                continue

            years = sorted(int(p.stem) for p in nc_files)

            # Map plots for selected years.
            selected = sorted({
                years[0], years[-1],
                *[y for y in years if y % map_every == 0],
            })
            for yr in selected:
                ds = xr.open_dataset(data_dir / f"{yr}.nc")
                plot_global_map(
                    ds[cid],
                    f"{var_info['long_name']} — {yr}",
                    PLOTS_DIR / "maps" / cid / f"{yr}.png",
                    cmap=ps["cmap"],
                    force_log=(stat in ("mean", "sum", "max")),
                )
                ds.close()

            # Time-series plot.
            ts_years, ts_means = [], []
            for p in nc_files:
                ds = xr.open_dataset(p)
                ts_years.append(int(p.stem))
                ts_means.append(float(ds[cid].mean(skipna=True)))
                ds.close()

            plot_time_series(
                ts_years,
                ts_means,
                f"Global Mean: {var_info['long_name']}",
                var_info["units"],
                PLOTS_DIR / "timeseries" / f"{cid}.png",
                color=ps["color"],
            )
            logger.info("Plotted %s (%d maps, 1 timeseries)", cid, len(selected))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.command()
@click.option(
    "--source", "-s", "sources", multiple=True,
    type=click.Choice(list(SOURCE_FIELDS), case_sensitive=False),
    help="Source field(s) to process. Default: all three.",
)
@click.option(
    "--stat", "stats", multiple=True,
    type=click.Choice(list(STAT_NAMES), case_sensitive=False),
    help="Statistic(s) to compute. Default: all five.",
)
@click.option("--all", "run_all", is_flag=True, help="Download + process + plot.")
@click.option("--download", "do_download", is_flag=True, help="Download step.")
@click.option("--process", "do_process", is_flag=True, help="Processing step.")
@click.option("--plot", "do_plot", is_flag=True, help="Plotting step.")
@click.option("--map-every", type=int, default=5, show_default=True, help="Map plot interval (years).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing files.")
def main(
    sources: tuple[str, ...],
    stats: tuple[str, ...],
    run_all: bool,
    do_download: bool,
    do_process: bool,
    do_plot: bool,
    map_every: int,
    overwrite: bool,
) -> None:
    """Run the ODIAC fossil fuel CO2 pipeline (ODIAC2024, CC-BY 4.0)."""
    from src.download.odiac import download_odiac

    sel_sources = tuple(sources) if sources else SOURCE_FIELDS
    sel_stats = tuple(stats) if stats else STAT_NAMES

    if run_all:
        do_download = do_process = do_plot = True

    if not any([do_download, do_process, do_plot]):
        raise click.ClickException(
            "Specify --all or at least one of --download / --process / --plot."
        )

    if do_download:
        download_odiac(overwrite=overwrite)

    if do_process:
        n = process_all(sources=sel_sources, stats=sel_stats, overwrite=overwrite)
        logger.info("Total files written: %d", n)

    if do_plot:
        plot_all(sources=sel_sources, stats=sel_stats, map_every=map_every)
        logger.info("Plots written under: %s", PLOTS_DIR)


if __name__ == "__main__":
    main()
