# N₂O Emission Framework - Canada (500 m Grid)

Satellite-enhanced N₂O estimation framework for Canadian agricultural soils at 500 m resolution using a per-pixel Tier 2 emission-factor approach.

## Overview

| Item | Detail |
|---|---|
| CRS | EPSG:3347 (Stats Canada Lambert, equal-area) |
| Resolution | 500 m · 25 ha per pixel (constant, equal-area) |
| Analysis year | 2025 |
| Climate covariates | ERA5-Land, GPM IMERG, MODIS (year-matched) |
| Crop fractions | AAFC Annual Crop Inventory 2023 |
| Livestock parameters | 2021 Census of Agriculture scaled via FAO FAOSTAT |

## Repository Structure

```
├── config.py           # paths, constants, year settings, livestock parameters
├── grid.py             # 500 m EPSG:3347 grid definition and Canada land mask
├── gee_export.py       # GEE authentication, Drive API, covariate export/download
├── covariates.py       # load and reproject covariate GeoTIFFs to grid
├── ag_mask.py          # agricultural mask (AAFC ACI) and SLC soil properties
├── emission_model.py   # per-pixel N₂O emission model
└── uncertainty.py      # parameter sensitivity (tornado) and Monte Carlo UQ
├── requirements.txt
├── .gitignore
└── README.md
```

## Data Requirements

Place the following files in `data/` before running:

| File | Source |
|---|---|
| `lpr_000b21a_e.shp` (+ `.dbf`, `.prj`, `.shx`) | Statistics Canada — province boundaries |
| `ca_all_slc_v3r2.shp` (+ sidecar files) | AAFC — Soil Landscapes of Canada v3r2 |
| `SLC_v3r2_soil_properties_by_polygon.csv` | AAFC — SLC soil property table |
| `census_agriculture_livestock_2018_2025.csv` | Statistics Canada — Census of Agriculture |

GEE covariate GeoTIFFs are exported automatically by `gee_export.py` on first run and cached in `gee_cache/`.

## Setup

```bash
pip install -r requirements.txt
```

Set your Google Earth Engine project ID in `src/config.py` or via environment variable:

```bash
export GEE_PROJECT=your-gee-project-id
export N2O_DATA_DIR=data        # optional, defaults to ./data
export N2O_OUT_DIR=outputs      # optional, defaults to ./outputs
export N2O_CACHE_DIR=gee_cache  # optional, defaults to ./gee_cache
```

## Module Reference

### `src/config.py`
Central settings file. Edit this first.

- Analysis year (`YEAR`, `CLIM_YEAR`, `ACI_YEAR`)
- All file paths (overridable via environment variables)
- Grid constants: `PIXEL_M = 500`, `PIXEL_HA = 25`
- Province code lookup (`PROVINCES`)
- Livestock N-excretion rates (`N_EXCRETION`) and manure management fractions (`MMS_FRAC`)
- Soil drainage multiplication factors (`DRAINAGE_FACTOR`)

### `src/grid.py`
Derives the 500 m EPSG:3347 grid bounding box from the province boundaries shapefile and builds the Canada land mask.

Exports: `NROWS`, `NCOLS`, `TRANSFORM_3347`, `CRS_3347`, `CRS_4326`, `PIX_LON`, `PIX_LAT`, `CANADA_MASK`, `prov_gdf`

### `src/gee_export.py`
Authenticates with Google Earth Engine and exports 7 covariate layers to Google Drive, then downloads them via the Drive REST API to `LOCAL_CACHE`.

**Layers exported at 500 m / EPSG:3347:**

| Variable | Source | Period |
|---|---|---|
| `NDVI_{year}` | MODIS MOD13A2 | Growing season (Jun–Aug) |
| `TEMP_{year}` | ERA5-Land 2 m temperature | Annual mean |
| `PRECIP_{year}` | GPM IMERG monthly (mm/hr → annual mm) | Annual sum |
| `SNOW_{year}` | MODIS MOD10A1 NDSI > 10 | Oct–May |
| `SM_{year}` | SMAP SPL4SMGP surface | Growing season mean |
| `CROP_ANN_{aci_year}` | AAFC ACI classes 130–199 | Single year |
| `CROP_PAS_{aci_year}` | AAFC ACI classes 110, 122 | Single year |

Key functions: `authenticate()`, `export_and_download()`, `read_tif_to_grid()`

### `src/covariates.py`
Loads covariate GeoTIFFs from `LOCAL_CACHE`, auto-reprojecting to the notebook grid if the shape does not match. Falls back through prior years if the requested year is missing.

Key functions:
- `load_covariate(varname, year)` → float32 `(NROWS, NCOLS)`
- `load_all(clim_year, aci_year)` → dict of all standard covariates including derived `FREEZE_THAW`

### `src/ag_mask.py`
Builds the agricultural mask from AAFC ACI GeoTIFFs and rasterises SLC v3r2 soil properties onto the grid in RAM-safe chunked strips.

Exports (module-level arrays, built on import):

| Array | Shape | Description |
|---|---|---|
| `AG_MASK` | `(NROWS, NCOLS)` bool | True for agricultural pixels |
| `PROV_GRID` | `(NROWS, NCOLS)` str | 2-letter province code per pixel |
| `CROP_ANN` | `(NROWS, NCOLS)` float32 | Annual crop fraction |
| `CROP_PAS` | `(NROWS, NCOLS)` float32 | Pasture fraction |
| `CLAY_GRID` | `(NROWS, NCOLS)` float32 | Clay % |
| `OC_GRID` | `(NROWS, NCOLS)` float32 | Organic carbon % |
| `PH_GRID` | `(NROWS, NCOLS)` float32 | Soil pH |
| `DRAIN_F_GRID` | `(NROWS, NCOLS)` float32 | Drainage multiplication factor |
| `DRAIN_POOR` | `(NROWS, NCOLS)` bool | True for Poor / Very Poor drainage |

Falls back to coordinate-based approximations when ACI GeoTIFFs or SLC files are absent.

### `src/emission_model.py`
Core per-pixel N₂O emission model.

**Emission factor:**
```
EF = clip(0.01 × drainage_f × texture_f × sm_f × temp_f × freeze_thaw_f × oc_f, 0.001, 0.06)
```

**Pathways:**
```
N2O_direct   = eff_N × EF × (44/28)  +  N_pas × 0.02 × (44/28)
N2O_vol      = eff_N × 0.20 × 0.01 × (44/28)
N2O_leach    = eff_N × f_leach × 0.0075 × drain_m × (44/28)
N2O_total    = N2O_direct + N2O_vol + N2O_leach
```

Where `eff_N = N_app × n_up` and `n_up` is an NDVI-based plant uptake scalar. Leaching fraction `f_leach` is precipitation-dependent (0.08–0.30).

**Entry point:**
```python
from src.emission_model import run
out = run(covs, livestock_grids, ag_mask, drain_f, drain_poor, clay_grid, oc_grid, crop_ann)
```

Returns a dict with keys: `EF`, `N2O_direct`, `N2O_indirect`, `N2O_total`, `N2O_ipcc`, `N2O_density`, `EF_departure_pct`, `national_Gg`, `tier1_Gg`.

### `src/uncertainty.py`
Parameter sensitivity analysis and Monte Carlo uncertainty quantification.

**`sensitivity(flat_arrays, baseline_Gg)`**
Runs each of 12 parameters across its low/high range while holding others at default, producing a tornado chart DataFrame saved to `outputs/parameter_sensitivity.csv`.

**`monte_carlo(flat_arrays, baseline_Gg, n_mc=1000, sample_n=500_000)`**
Stratified pixel sample with lognormal/normal parameter draws. Returns the MC distribution array and a summary dict (`mean_Gg`, `std_Gg`, `cv_pct`, `p5_Gg`, `p95_Gg`). Saves `outputs/mc_summary.csv` and `outputs/mc_distribution.npy`.

Parameters varied (12 total): base EF, drainage factor, clay slope, SM wet/dry factors, temperature warm factor, freeze-thaw coefficient, OC slope, NDVI uptake, leaching fraction, MMS fraction (dairy), N excretion rate (dairy).

## Typical Workflow

```python
# 1. Export/cache covariates (once per year, requires GEE auth)
from src.gee_export import authenticate, export_all, _build_drive_service
from src.grid import GRID_XMIN, GRID_XMAX, GRID_YMIN, GRID_YMAX
import ee

gee_ok = authenticate()
svc    = _build_drive_service()
region = ee.Geometry.Rectangle(
    [GRID_XMIN, GRID_YMIN, GRID_XMAX, GRID_YMAX],
    proj=ee.Projection("EPSG:3347"), geodesic=False)
layers = export_all(gee_ok, svc, region)

# 2. Load covariates
from src.covariates import load_all
covs = load_all()

# 3. Build ag mask and soil grids (runs on import)
import src.ag_mask as am

# 4. Run emission model (pass pre-allocated livestock_grids)
from src.emission_model import run
out = run(covs, livestock_grids, am.AG_MASK, am.DRAIN_F_GRID,
          am.DRAIN_POOR, am.CLAY_GRID, am.OC_GRID, am.CROP_ANN)

# 5. Uncertainty quantification
from src.uncertainty import sensitivity, monte_carlo
# build flat_arrays dict by ravelling AG_MASK-masked arrays first
sens_df = sensitivity(flat_arrays, out["national_Gg"])
mc_dist, mc_summary = monte_carlo(flat_arrays, out["national_Gg"])
```

## Citation

If you use this framework please cite the associated manuscript (in preparation).
