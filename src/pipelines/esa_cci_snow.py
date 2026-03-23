"""ESA Snow CCI full pipeline: download -> yearly processing -> plots."""

from __future__ import annotations

import gc
from pathlib import Path

import click
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import xarray as xr

from src.download.esa_cci_snow import download_esa_cci_snow, load_config
from src.processing.esa_cci_snow_to_yearly import process_esa_cci_snow
from src.utils import get_logger, plot_global_map, plot_time_series

logger = get_logger("pipeline.esa_cci_snow")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_variables(cfg: dict, selected: tuple[str, ...], run_all: bool) -> list[str]:
    all_vars = list(cfg["variables"].keys())
    if selected:
        return [v for v in selected if v in cfg["variables"]]
    if run_all:
        return all_vars
    return []


def _resolve_years(cfg: dict, years: tuple[int, ...], start_year: int | None, end_year: int | None) -> list[int]:
    y0, y1 = [int(v) for v in cfg["temporal_range"]]
    if years:
        return sorted(set(int(y) for y in years if y0 <= int(y) <= y1))
    ys = int(start_year) if start_year is not None else y0
    ye = int(end_year) if end_year is not None else y1
    if ye < ys:
        raise ValueError("end-year must be >= start-year")
    return [y for y in range(ys, ye + 1) if y0 <= y <= y1]


def _output_specs(cfg: dict, variable_keys: list[str]) -> list[dict]:
    stats = list(cfg.get("stats", ["mean", "std", "min", "max"]))
    specs: list[dict] = []
    for key in variable_keys:
        info = cfg["variables"][key]
        for stat in stats:
            specs.append(
                {
                    "name": f"{info['output_prefix']}_{stat}",
                    "long_name": f"{info['long_name']} ({stat})",
                    "units": str(info["units"]),
                    "cmap": str(info.get("cmap", "viridis")),
                }
            )
    return specs


def _plot_map(
    final_root: Path,
    plots_root: Path,
    domain: str,
    var_name: str,
    long_name: str,
    cmap: str,
    year: int,
) -> None:
    nc_path = final_root / domain / var_name / f"{year}.nc"
    if not nc_path.exists():
        return
    out_path = plots_root / "maps" / var_name / f"{year}.png"

    ds = xr.open_dataset(nc_path)
    try:
        da = ds[var_name]
        if "time" in da.dims:
            da = da.isel(time=0, drop=True)
        plot_global_map(da, title=f"{long_name} - {year}", out_path=out_path, cmap=cmap)
    finally:
        ds.close()


def _plot_series(
    final_root: Path,
    plots_root: Path,
    domain: str,
    var_name: str,
    long_name: str,
    units: str,
    years: list[int],
) -> None:
    vals: list[float] = []
    ys: list[int] = []
    for year in years:
        nc_path = final_root / domain / var_name / f"{year}.nc"
        if not nc_path.exists():
            continue
        ds = xr.open_dataset(nc_path)
        try:
            da = ds[var_name]
            if "time" in da.dims:
                da = da.isel(time=0, drop=True)
            vals.append(float(da.mean(skipna=True).values))
            ys.append(year)
        finally:
            ds.close()
    if not ys:
        return
    out_path = plots_root / "timeseries" / f"{var_name}.png"
    plot_time_series(
        years=ys,
        values=vals,
        title=f"{long_name} (global mean)",
        ylabel=units,
        out_path=out_path,
    )


@click.command()
@click.option("--variables", "-v", multiple=True, help="Variable keys from config (swe, swe_std).")
@click.option("--all", "run_all", is_flag=True, help="Run all configured variables.")
@click.option("--years", "-y", multiple=True, type=int, help="Specific years.")
@click.option("--start-year", type=int, default=None, help="Start year.")
@click.option("--end-year", type=int, default=None, help="End year.")
@click.option("--plot-every", type=int, default=5, show_default=True, help="Map every N years.")
@click.option("--skip-download", is_flag=True, help="Skip download step.")
@click.option("--overwrite", is_flag=True, help="Overwrite existing outputs.")
@click.option("--interpolate-missing/--no-interpolate-missing", default=True, show_default=True)
def main(
    variables: tuple[str, ...],
    run_all: bool,
    years: tuple[int, ...],
    start_year: int | None,
    end_year: int | None,
    plot_every: int,
    skip_download: bool,
    overwrite: bool,
    interpolate_missing: bool,
):
    cfg = load_config()
    variable_keys = _resolve_variables(cfg, variables, run_all)
    if not variable_keys:
        click.echo("Specify --variables or --all")
        return

    year_list = _resolve_years(cfg, years, start_year, end_year)
    if not year_list:
        click.echo("No valid years selected.")
        return

    output_cfg = cfg["output"]
    final_root = PROJECT_ROOT / str(output_cfg.get("final_root", "data/final"))
    plots_root = PROJECT_ROOT / str(output_cfg.get("plots_root", "plots/esa_cci_snow"))
    domain = str(output_cfg.get("domain", "Cryosphere"))

    click.echo(
        f"ESA Snow pipeline: variables={variable_keys}, years={year_list[0]}-{year_list[-1]}, plot_every={plot_every}"
    )

    if not skip_download:
        click.echo("\n-- Download --")
        n = download_esa_cci_snow(years=year_list, overwrite=overwrite)
        click.echo(f"Downloaded files: {n}")

    click.echo("\n-- Processing --")
    summary = process_esa_cci_snow(
        years=year_list,
        variable_keys=variable_keys,
        overwrite=overwrite,
        interpolate_missing=interpolate_missing,
    )
    click.echo(
        f"Processing done: files_written={summary['files_written']}, "
        f"interpolated_files={summary['interpolated_files_written']}"
    )

    click.echo("\n-- Visualization --")
    plot_years = set(range(year_list[0], year_list[-1] + 1, max(1, int(plot_every))))
    plot_years.add(year_list[-1])
    specs = _output_specs(cfg, variable_keys)

    for spec in specs:
        var_name = spec["name"]
        for year in sorted(plot_years):
            try:
                _plot_map(
                    final_root=final_root,
                    plots_root=plots_root,
                    domain=domain,
                    var_name=var_name,
                    long_name=spec["long_name"],
                    cmap=spec["cmap"],
                    year=year,
                )
            except Exception as e:
                logger.warning("Map plot failed var=%s year=%d (%s)", var_name, year, e)
            gc.collect()
            plt.close("all")

        try:
            _plot_series(
                final_root=final_root,
                plots_root=plots_root,
                domain=domain,
                var_name=var_name,
                long_name=spec["long_name"],
                units=spec["units"],
                years=year_list,
            )
        except Exception as e:
            logger.warning("Series plot failed var=%s (%s)", var_name, e)
        gc.collect()
        plt.close("all")

    click.echo(f"\nPipeline complete. Outputs in {final_root / domain} and plots in {plots_root}")


if __name__ == "__main__":
    main()
