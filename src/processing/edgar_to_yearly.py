"""Regrid EDGAR 2025 GHG 0.1° emissions to the master 0.25° grid.

Reads per-year/substance/sector NetCDF files (1800×3600, -180..180 lon),
interpolates to 721×1440 (0..360 lon), and writes CF-1.8 compliant output.

Usage:
    python -m src.processing.edgar_to_yearly --all
    python -m src.processing.edgar_to_yearly --substances CO2 --sectors TOTALS --start-year 2020 --end-year 2024
"""

from pathlib import Path

import click
import numpy as np
import xarray as xr
import yaml
from tqdm import tqdm

from src.grid import N_LAT, N_LON, LAT_MIN, LAT_MAX, LON_MIN, LON_MAX
from src.utils import get_logger, save_to_netcdf
from src.year_policy import resolve_year_bounds

logger = get_logger("processing.edgar")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "edgar.yml"
DEFAULT_RAW_DIR = PROJECT_ROOT / "data" / "raw" / "edgar"
FINAL_DIR = PROJECT_ROOT / "data" / "final" / "emissions"

# Master grid coordinates
TARGET_LAT = np.linspace(LAT_MIN, LAT_MAX, N_LAT)
TARGET_LON = np.linspace(LON_MIN, LON_MAX, N_LON)


def load_edgar_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


def process_file(
    raw_nc_path: Path,
    substance: str,
    sector: str,
    year: int,
    units: str,
    long_name: str,
    sector_long_name: str,
    overwrite: bool = False,
) -> Path | None:
    """Regrid a single EDGAR NC file and save to final directory.

    Parameters
    ----------
    raw_nc_path : Path
        Path to the raw 0.1° NetCDF file.
    substance, sector, year : str, str, int
        Identifiers for the output path.
    units, long_name, sector_long_name : str
        Metadata for CF attributes.
    overwrite : bool
        Overwrite existing output files.

    Returns
    -------
    Path or None
        Output path, or None if skipped.
    """
    canonical_id = f"{substance}_{sector}".lower()
    out_path = FINAL_DIR / canonical_id / f"{year:04d}.nc"
    if out_path.exists() and not overwrite:
        return None

    ds = xr.open_dataset(raw_nc_path)

    # Find the flux variable (usually "fluxes")
    data_vars = list(ds.data_vars)
    if "fluxes" in data_vars:
        da = ds["fluxes"].load()
    elif len(data_vars) == 1:
        da = ds[data_vars[0]].load()
    else:
        logger.warning("Cannot identify flux variable in %s: %s", raw_nc_path, data_vars)
        ds.close()
        return None

    # Shift longitude from -180..180 to 0..360
    lon_shifted = da.lon.values % 360
    da = da.assign_coords(lon=lon_shifted).sortby("lon")

    # Interpolate to master grid
    da = da.interp(
        lat=TARGET_LAT,
        lon=TARGET_LON,
        method="linear",
        kwargs={"fill_value": "extrapolate"},
    )

    # Emissions cannot be negative
    da = da.clip(min=0)

    # Build CF-compliant output dataset
    out_ds = xr.Dataset(
        {
            canonical_id: (
                ["lat", "lon"],
                da.values.astype(np.float32),
                {
                    "units": units,
                    "long_name": f"{long_name} — {sector_long_name}",
                    "source": "EDGAR 2025 GHG (Crippa et al. 2025)",
                },
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor EDGAR {substance}_{sector}",
            "source": "EDGAR 2025 GHG (Crippa et al. 2025)",
            "year": year,
            "substance": substance,
            "sector": sector,
        },
    )
    save_to_netcdf(out_ds, canonical_id, year, output_dir=FINAL_DIR)
    ds.close()
    return out_path


def aggregate_sectors(
    substance: str,
    agg_name: str,
    agg_info: dict,
    sub_info: dict,
    year: int,
    overwrite: bool = False,
) -> Path | None:
    """Sum multiple sector files into an aggregate variable.

    Parameters
    ----------
    substance : str
        Substance name (e.g. "CO2").
    agg_name : str
        Aggregate sector name (e.g. "Aviation").
    agg_info : dict
        Aggregate config with 'long_name' and 'components' list.
    sub_info : dict
        Substance config with 'units' and 'long_name'.
    year : int
        Year to aggregate.
    overwrite : bool
        Overwrite existing output.

    Returns
    -------
    Path or None
        Output path, or None if skipped or missing components.
    """
    canonical_id = f"{substance}_{agg_name}".lower()
    out_path = FINAL_DIR / canonical_id / f"{year:04d}.nc"
    if out_path.exists() and not overwrite:
        return None

    components = agg_info["components"]
    arrays = []
    for comp in components:
        comp_id = f"{substance}_{comp}".lower()
        comp_path = FINAL_DIR / comp_id / f"{year:04d}.nc"
        if not comp_path.exists():
            return None
        ds = xr.open_dataset(comp_path)
        if comp_id in ds.data_vars:
            arrays.append(ds[comp_id].load())
        else:
            arrays.append(ds[next(iter(ds.data_vars))].load())
        ds.close()

    # Sum all components
    total = arrays[0]
    for a in arrays[1:]:
        total = total + a
    total = total.clip(min=0)

    out_ds = xr.Dataset(
        {
            canonical_id: (
                ["lat", "lon"],
                total.values.astype(np.float32),
                {
                    "units": sub_info["units"],
                    "long_name": f"{sub_info['long_name']} — {agg_info['long_name']}",
                    "source": "EDGAR 2025 GHG (Crippa et al. 2025)",
                    "components": ", ".join(components),
                },
            )
        },
        coords={
            "lat": ("lat", TARGET_LAT, {"units": "degrees_north", "long_name": "latitude"}),
            "lon": ("lon", TARGET_LON, {"units": "degrees_east", "long_name": "longitude"}),
        },
        attrs={
            "Conventions": "CF-1.8",
            "title": f"WorldTensor EDGAR {substance}_{agg_name}",
            "source": "EDGAR 2025 GHG (Crippa et al. 2025)",
            "year": year,
            "substance": substance,
            "sector": agg_name,
        },
    )
    save_to_netcdf(out_ds, canonical_id, year, output_dir=FINAL_DIR)
    logger.info("Aggregated %s_%s/%04d from %d components", substance, agg_name, year, len(components))
    return out_path


def expected_nc_name(substance: str, sector: str, year: int) -> str:
    """Expected raw NetCDF filename."""
    return f"EDGAR_2025_GHG_{substance}_{year}_{sector}_flx_nc.nc"


@click.command()
@click.option("--substances", "-s", multiple=True, help="Substance(s) to process.")
@click.option("--sectors", "-S", multiple=True, help="Sector(s) to process.")
@click.option("--all", "run_all", is_flag=True, help="Process all substances/sectors.")
@click.option("--start-year", type=int, default=None, help="Start year (default: 1970).")
@click.option("--end-year", type=int, default=None, help="End year (default: 2024).")
@click.option("--overwrite", is_flag=True, help="Overwrite existing output files.")
@click.option("--raw-dir", type=click.Path(), default=None,
              help=f"Raw data directory (default: {DEFAULT_RAW_DIR})")
@click.option("--cleanup-raw", is_flag=True,
              help="Delete raw NC files after successful processing.")
def main(substances, sectors, run_all, start_year, end_year, overwrite, raw_dir, cleanup_raw):
    """Regrid EDGAR 0.1° emissions to 0.25° master grid."""
    config = load_edgar_config()
    t_range = config["temporal_range"]

    y_start, y_end = resolve_year_bounds(
        start_year=start_year,
        end_year=end_year,
        default_start=t_range[0],
        default_end=t_range[1],
        label="EDGAR processing years",
    )
    years = list(range(y_start, y_end + 1))

    if not substances and not run_all:
        click.echo("Specify --substances or --all. Use --help for usage.")
        return

    sub_list = list(substances) if substances else list(config["substances"].keys())
    all_sectors = config["sectors"]
    src_dir = Path(raw_dir) if raw_dir else DEFAULT_RAW_DIR

    total_processed = 0
    total_skipped = 0

    for sub in sub_list:
        sub_info = config["substances"].get(sub)
        if sub_info is None:
            logger.warning("Unknown substance: %s", sub)
            continue

        available_sectors = sub_info["sectors"]
        if sectors:
            sec_list = [s for s in sectors if s in available_sectors]
        else:
            sec_list = available_sectors

        for sec in sec_list:
            sec_info = all_sectors.get(sec, {})
            sec_long_name = sec_info.get("long_name", sec)
            processed = 0
            skipped = 0

            for year in tqdm(years, desc=f"{sub}/{sec}", unit="yr"):
                nc_name = expected_nc_name(sub, sec, year)
                raw_path = src_dir / sub / sec / nc_name

                if not raw_path.exists():
                    continue

                result = process_file(
                    raw_path, sub, sec, year,
                    units=sub_info["units"],
                    long_name=sub_info["long_name"],
                    sector_long_name=sec_long_name,
                    overwrite=overwrite,
                )
                if result:
                    processed += 1
                    if cleanup_raw:
                        raw_path.unlink(missing_ok=True)
                else:
                    skipped += 1

            total_processed += processed
            total_skipped += skipped
            if processed or skipped:
                click.echo(f"  {sub}/{sec}: {processed} new, {skipped} skipped")

    click.echo(f"Done. {total_processed} processed, {total_skipped} skipped. Output in {FINAL_DIR}")


if __name__ == "__main__":
    main()
