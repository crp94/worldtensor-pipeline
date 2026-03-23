# WorldTensor

**A Harmonized Dataset for Earth System Foundation Models**

WorldTensor aligns hundreds of environmental and socioeconomic variables onto a
shared 0.25° latitude-longitude grid and annual temporal framework. The dataset
spans 14 thematic domains — climate, extremes, air quality, emissions, land use,
vegetation, hydrology, cryosphere, ocean, agriculture, energy, human systems,
hazards and conflict, and static geographic context — comprising 757 released
variable families (658 temporal + 99 static), 52,823 individual NetCDF files, and
approximately 46 GB on disk.

This repository contains the complete pipeline code used to acquire, harmonize,
and quality-check the dataset, together with PyTorch examples for machine learning
ingestion. The released data files are distributed separately via
[Zenodo](https://zenodo.org/records/19047618).

> Rodriguez-Pardo, C. and Tavoni, M. (2026). WorldTensor: A Harmonized Dataset
> for Earth System Foundation Models. *Nature Scientific Data* (submitted).

---

## Repository structure

```
worldtensor/
├── config/                      # YAML configuration for every source dataset
│   ├── variables.yml            # Master registry: maps canonical variable IDs
│   │                            #   to file paths and metadata
│   ├── era5.yml                 # ERA5 variable selection and download config
│   ├── edgar.yml                # EDGAR emissions sector/gas definitions
│   ├── ...                      # One config per source (~50 files)
│   └── secrets.yml.example      # Credential template (see Setup below)
│
├── src/
│   ├── download/                # Source-specific data acquisition modules (42)
│   │   ├── era5.py              # CDS API downloads for ERA5
│   │   ├── edgar.py             # EDGAR emissions raster downloads
│   │   ├── mod13c2.py           # MODIS vegetation index downloads
│   │   └── ...
│   │
│   ├── processing/              # Spatial and temporal harmonization (29)
│   │   ├── era5_monthly_to_yearly.py
│   │   ├── luh3_states_to_yearly.py
│   │   ├── luh3_transitions_to_yearly.py
│   │   ├── raster_to_grid.py    # Generic raster regridding to 0.25°
│   │   ├── points_to_grid.py    # Point/event data rasterization
│   │   ├── lines_to_grid.py     # Line geometry rasterization
│   │   └── ...
│   │
│   ├── pipelines/               # End-to-end workflows: download → process → output (51)
│   │   ├── era5.py              # Full ERA5 pipeline
│   │   ├── edgar.py             # Full EDGAR pipeline
│   │   ├── powerplants.py       # Power plant rasterization pipeline
│   │   ├── point_datasets.py    # Hazard/conflict event rasterization
│   │   └── ...
│   │
│   ├── grid.py                  # Canonical grid definition (721 × 1440)
│   ├── data_layout.py           # Variable registry and path resolution
│   ├── harmonize_data.py        # Post-hoc structural audit and repair
│   ├── utils.py                 # Shared helpers (save_annual_variable, etc.)
│   └── year_policy.py           # Temporal range resolution per dataset
│
├── examples/torch/              # PyTorch dataset interfaces
│   ├── worldtensor_torch.py     # WorldTensorYearDataset, WorldTensorPatchDataset
│   ├── 01_global_tensor.py      # Full-resolution global tensor demo
│   ├── 02_patch_dataloader.py   # Spatial patch sampling demo
│   └── USAGE.txt                # Detailed usage instructions
│
├── pyproject.toml               # Package metadata and dependencies
├── LICENSE                      # MIT License
└── .gitignore
```

---

## Setup

### 1. Clone and install

```bash
git clone <REPOSITORY_URL>
cd worldtensor

python -m venv .venv
source .venv/bin/activate

pip install -e .
```

### 2. Configure credentials

Several data sources require API credentials. Copy the template and fill in
your keys:

```bash
cp config/secrets.yml.example config/secrets.yml
```

Not all pipelines require credentials. Most datasets (EDGAR, LUH3, SoilGrids,
ETOPO, VODCA, etc.) download directly from public archives without
authentication. Only the pipelines listed below need API keys.

#### Copernicus Climate Data Store (CDS) API

**Required by:** `era5`, `cams`, `climate_extremes`

1. Create a free account at https://cds.climate.copernicus.eu/
2. After login, go to your profile page (click your name in the top right)
3. Scroll to the **Personal Access Token** section and copy your API key
4. Add to `config/secrets.yml`:
   ```yaml
   cds:
     url: https://cds.climate.copernicus.eu/api
     key: <your-uid>:<your-api-key>
   ```
5. You must also accept the license terms for each dataset on the CDS website
   before the API will serve data. Visit the dataset pages for
   [ERA5](https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means),
   [CAMS](https://ads.atmosphere.copernicus.eu/datasets/cams-global-reanalysis-eac4-monthly),
   and the drought indices, then click "Accept Terms".

#### NASA Earthdata

**Required by:** `mod13c2` (MODIS NDVI/EVI), `mcd64a1` (MODIS burned area),
`chlorophyll` (MODIS-Aqua), `gpw` (population), `hydrology` (GRACE/GLDAS),
`esa_cci_snow`

1. Create a free account at https://urs.earthdata.nasa.gov/
2. Log in, then go to **My Profile > Generate Token** to create a Bearer token
3. Add to `config/secrets.yml` (either token or username/password):
   ```yaml
   earthdata:
     token: <your-bearer-token>
     # OR use username/password instead:
     # username: <your-username>
     # password: <your-password>
   ```
4. Some datasets require you to approve specific "Applications" in your
   Earthdata profile. If a download returns a 403 error, visit
   https://urs.earthdata.nasa.gov/profile and authorize the relevant
   application (e.g., "LP DAAC Data Pool" for MODIS products, "GES DISC"
   for GLDAS/GRACE).

### 3. Manual and semi-manual downloads

Most pipelines download data automatically. A few datasets require manual
steps:

#### LUH3 (Land Use Harmonization)

The LUH3 v3.1.1 files are distributed via the ESGF portal. The download
module attempts automatic discovery from ESGF search endpoints, but if that
fails you can obtain the files manually:

1. Go to https://aims2.llnl.gov/search
2. Filter: Project = `input4MIPs`, Institution ID = `UofMD`,
   Source Version = `3.1.1`
3. Download the `states`, `transitions`, `management`, and `static` NetCDF
   files
4. Pass them to the pipeline:
   ```bash
   python -m src.pipelines.luh3 --source-dir /path/to/downloaded/files
   python -m src.pipelines.luh3_transitions --source-dir /path/to/downloaded/files
   ```

Alternatively, if ESGF provides wget scripts, you can use:
```bash
python -m src.download.luh3 --all --wget-script /path/to/wget_states.sh \
                                   --wget-script /path/to/wget_transitions.sh
```

#### GMTDS shipping density (transport_connectivity pipeline)

GMTDS data requires authorized portal export. Download monthly `.tif` or
`.nc` files with `YYYYMM` in the filename and place them under:
```
data/raw/gmtds_monthly_manual/monthly/
```

Then run:
```bash
python -m src.pipelines.transport_connectivity
```

#### Global Integrated Power Tracker (powerplants pipeline)

The GEM Global Integrated Power Tracker spreadsheet must be obtained
manually:

1. Download `Global-Integrated-Power-March-2026.xlsx` from
   [Global Energy Monitor](https://globalenergymonitor.org/)
2. Place the file in `data/raw/`:
   ```
   data/raw/Global-Integrated-Power-March-2026.xlsx
   ```
3. Run the pipeline:
   ```bash
   python -m src.pipelines.powerplants --all
   ```

Alternatively, pass a custom path with `--input-xlsx /path/to/file.xlsx`.

#### GDIS disaster events (point_datasets pipeline)

The GDIS dataset is normally auto-downloaded. If the automatic download
fails, place `gdis.csv` manually in `data/raw/gdis/`.

### 4. Install optional dependencies

For the PyTorch examples:

```bash
pip install -e ".[ml]"
```

This adds `torch`, `torchgeo`, and `scikit-learn`.

---

## Obtaining the data

### Option A: Download the pre-built dataset

The released WorldTensor files are available on Zenodo. Download and extract
them into `data/final/`:

```bash
mkdir -p data/final
# Download from Zenodo and extract into data/final/
```

The expected layout:

```
data/final/
├── climate/
│   ├── t2m_mean/
│   │   ├── 1940.nc
│   │   ├── 1941.nc
│   │   └── ...
│   └── ...
├── emissions/
├── land_use/
│   ├── states/
│   └── transitions/
├── static/
│   ├── topography/
│   ├── soil_properties/
│   └── ...
└── ...
```

### Option B: Rebuild from source

Each pipeline can be run independently. Pipelines download raw data, process
it, and write harmonized NetCDF files to `data/final/`.

```bash
# Example: build all ERA5 climate variables
python -m src.pipelines.era5 --all

# Example: build EDGAR emissions
python -m src.pipelines.edgar

# Example: build power plant infrastructure layers
python -m src.pipelines.powerplants

# Example: build LUH3 land-use states and transitions
python -m src.pipelines.luh3
python -m src.pipelines.luh3_transitions
```

Most pipelines accept `--help` for available options:

```bash
python -m src.pipelines.era5 --help
```

Common options include `--start-year`, `--end-year`, `--workers`, and
`--variables` (or `--all`).

After running pipelines, use the harmonization tool to audit and repair
structural compliance across all generated files:

```bash
# Check all files in data/final/ for convention compliance
python -m src.harmonize_data --check

# Apply fixes (add missing CF attributes, fix dtypes, regrid nonconforming files)
python -m src.harmonize_data --apply
```

This verifies grid dimensions, coordinate names, longitude convention,
CF attributes, compression, data types, and temporal continuity.

### Available pipelines

Each pipeline maps to a source dataset. The **Auth** column indicates which
credentials are needed: **CDS** = Copernicus CDS API key, **ED** = NASA
Earthdata token, **--** = no authentication required (public download),
**Manual** = requires manual file placement (see above).

| Pipeline | Domain | Source | Auth |
|----------|--------|--------|------|
| `era5` | Climate | ERA5 monthly reanalysis | CDS |
| `climate_extremes` | Extremes | SPI/SPEI drought indices | CDS |
| `land_heatwaves` | Extremes | HadEX3 land heatwave indices | -- |
| `marine_heatwaves` | Extremes | NOAA marine heatwave metrics | -- |
| `cams` | Air quality | CAMS EAC4 reanalysis | CDS |
| `edgar` | Emissions | EDGAR v8.0 non-CO2 and biogenic CO2 | -- |
| `odiac` | Emissions | ODIAC fossil CO2 | -- |
| `luh3` | Land use | LUH3 states | -- (semi-manual) |
| `luh3_transitions` | Land use | LUH3 transitions | -- (semi-manual) |
| `mod13c2` | Vegetation | MODIS NDVI/EVI | ED |
| `mcd64a1` | Vegetation | MODIS burned area | ED |
| `vodca` | Vegetation | VODCA vegetation optical depth | -- |
| `agriculture` | Agriculture | GGCP10 crop production | -- |
| `fertilizer` | Agriculture | Crop-specific fertilizer maps | -- |
| `livestock` | Agriculture | AGLW livestock density | -- |
| `grace_fo` | Hydrology | GRACE/GRACE-FO water storage | ED |
| `groundwater` | Hydrology | GLDAS soil moisture | ED |
| `wetlands_wad2m` | Hydrology | WAD2M wetland inundation | -- |
| `esa_cci_snow` | Cryosphere | ESA Snow CCI | ED |
| `permafrost` | Cryosphere | ESA Permafrost CCI | -- |
| `glaciers` | Cryosphere | WGMS glacier fields | -- |
| `chlorophyll` | Ocean | MODIS-Aqua chlorophyll-a | ED |
| `powerplants` | Energy | Global Integrated Power Tracker | Manual |
| `point_datasets` | Hazards & conflict | UCDP, ComCat, IBTrACS, volcanoes, GDIS | -- |
| `gpw` | Human systems | GPW population | ED |
| `kummu_gdp` | Human systems | GDP, GNI, HDI, Gini | -- |
| `kummu_candidates` | Human systems | GDP per capita, inequality | -- |
| `sectgdp` | Human systems | SectGDP30 sectoral GDP | -- |
| `ntl` | Human systems | Harmonized nighttime lights | -- |
| `settlement_candidates` | Human systems | GHSL, GISA, WSF, urban extents | -- |
| `accessibility` | Static | Travel time to cities | -- |
| `gmted2010` | Static | GMTED2010 topography | -- |
| `etopo2022` | Static | ETOPO bathymetry | -- |
| `soilgrids` | Static | SoilGrids soil properties | -- |
| `gldas_soiltex` | Static | GLDAS soil texture classes | -- |
| `fldas_vegclass` | Static | FLDAS vegetation classes | -- |
| `dist2coast` | Static | Distance to coastline | -- |
| `hydrorivers` | Static | Distance to river | -- |
| `transport_connectivity` | Human systems | Transport corridors (HMv2024, GMTDS) | -- / Manual |
| `esv` | Human systems | Ecosystem service value (derived) | -- |

---

## Data conventions

All output files follow these conventions:

- **Grid**: 721 x 1440 (lat x lon), 0.25° resolution
- **Latitude**: -90° to 90° (south to north)
- **Longitude**: 0° to 359.75° (eastward from prime meridian)
- **Coordinate names**: `lat`, `lon`, `time`
- **Data type**: `float32`, zlib-compressed (complevel 4)
- **Metadata**: CF-1.8 conventions; every variable carries `units` and `long_name`
- **File layout**: `<domain>/<variable>/<YYYY>.nc` for temporal variables;
  `static/<group>/<variable>.nc` for static layers
- **Exception**: `land_use/states/<variable>/<YYYY>.nc` and
  `land_use/transitions/<variable>/<YYYY>.nc`

### Variable registry

The master registry at `config/variables.yml` maps every canonical variable ID
to its file path, domain, temporal/static classification, and metadata. It is
the single source of truth for discovery and path resolution. The helper
functions `output_path_for()`, `save_annual_variable()`, and
`save_static_variable()` in `src/utils.py` and `src/data_layout.py` enforce
this layout programmatically.

---

## Configuration

Each pipeline is driven by a YAML configuration file in `config/`. These files
define:

- Source URLs and download parameters
- Variable selections and naming mappings
- Temporal ranges and year policies
- Processing options (statistics, interpolation, clipping)

The configurations are self-documenting. For example, `config/era5.yml` lists
all ERA5 variables with their CDS names, short names, and aggregation
statistics. `config/edgar.yml` defines all substance/sector combinations.

### Credentials

API credentials are stored in `config/secrets.yml` (git-ignored). See the
[Setup](#2-configure-credentials) section above for detailed registration
instructions. Pipelines that require credentials will raise a clear error
message if the secrets file is missing or incomplete.

---

## PyTorch examples

Two reference PyTorch interfaces are provided under `examples/torch/`. These
read directly from the per-variable file layout — no pre-merged cube is needed.

### Global tensor

```bash
python examples/torch/01_global_tensor.py
```

Loads selected variables for a given year into a `[C, H, W]` tensor:

```python
from examples.torch.worldtensor_torch import WorldTensorYearDataset

dataset = WorldTensorYearDataset(
    variables=["t2m_mean", "tp_sum", "ndvi_mean", "elevation_mean"],
    years=[2015, 2016, 2017],
)

sample = dataset[0]
# sample["x"]    -> (C, 721, 1440) tensor
# sample["mask"] -> (C, 721, 1440) finite-value mask
# sample["year"] -> 2015
```

### Patch dataloader

```bash
python examples/torch/02_patch_dataloader.py
```

Samples spatial crops for minibatch training:

```python
from examples.torch.worldtensor_torch import WorldTensorPatchDataset

patches = WorldTensorPatchDataset(
    variables=["t2m_mean", "tp_sum", "tcc_mean", "elevation_mean"],
    years=[2015, 2016],
    patch_size=64,
    patches_per_year=128,
)
```

Supports dense patches and sparse dictionaries with coordinates. See
`examples/torch/USAGE.txt` for full options.

---

## Requirements

**Core** (installed via `pip install -e .`):

- Python >= 3.10
- xarray, netCDF4, rioxarray, rasterio, numpy, scipy
- geopandas, shapely, geocube
- dask, pyyaml, tqdm, click
- cartopy, matplotlib
- cdsapi, earthaccess, requests

**ML extras** (installed via `pip install -e ".[ml]"`):

- torch >= 2.0
- torchgeo >= 0.5
- scikit-learn >= 1.3

Some processing pipelines additionally require GDAL command-line tools
(`gdalwarp`, `gdal_translate`) for raster operations on MODIS and similar
tiled products.

---

## License

This code is released under the [MIT License](LICENSE).

Source datasets used to construct WorldTensor are publicly available from the
repositories cited in the accompanying paper. Users should consult the original
licenses for each source when using the data beyond the scope of this release.

---

## Citation

If you use WorldTensor in your research, please cite:

```bibtex
@article{rodriguez-pardo2026worldtensor,
  title   = {WorldTensor: A Harmonized Dataset for Earth System Foundation Models},
  author  = {Rodriguez-Pardo, Carlos and Tavoni, Massimo},
  journal = {Nature Scientific Data},
  year    = {2026},
  note    = {submitted}
}
```

## Acknowledgements

This work was supported by the European Research Council, ERC grant agreement
number 101044703 (EUNICE) CUP D87G22000340006.
