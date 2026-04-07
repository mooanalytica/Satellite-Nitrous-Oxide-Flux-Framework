import os

# ── Analysis years ────────────────────────────────────────────────────────────
YEAR       = "2025"
CLIM_YEAR  = "2025"
ACI_YEAR   = "2023"
STUDY_YEARS = list(range(2018, 2026))

# ── Paths — override with environment variables or edit here ──────────────────
DATA_DIR    = os.environ.get("N2O_DATA_DIR",    "data")
OUT_DIR     = os.environ.get("N2O_OUT_DIR",     "outputs")
LOCAL_CACHE = os.environ.get("N2O_CACHE_DIR",   "gee_cache")

PROV_BOUNDARIES_PATH = os.path.join(DATA_DIR, "lpr_000b21a_e.shp")
SLC_SHAPEFILE_PATH   = os.path.join(DATA_DIR, "ca_all_slc_v3r2.shp")
SLC_SOIL_CSV_PATH    = os.path.join(DATA_DIR, "SLC_v3r2_soil_properties_by_polygon.csv")
LIVESTOCK_CSV_PATH   = os.path.join(DATA_DIR, "census_agriculture_livestock_2018_2025.csv")

GEE_PROJECT       = os.environ.get("GEE_PROJECT", "your-gee-project-id")
GEE_DRIVE_FOLDER  = "N2O_covariates"

# ── Grid constants ────────────────────────────────────────────────────────────
PIXEL_M  = 500
PIXEL_HA = (PIXEL_M ** 2) / 10_000   # 25 ha

# ── Province codes ────────────────────────────────────────────────────────────
PROVINCES = {
    "BC": "British Columbia", "AB": "Alberta",     "SK": "Saskatchewan",
    "MB": "Manitoba",         "ON": "Ontario",     "QC": "Quebec",
    "NB": "New Brunswick",    "NS": "Nova Scotia", "PE": "Prince Edward Island",
    "NL": "Newfoundland and Labrador",
}

# ── Livestock parameters ──────────────────────────────────────────────────────
ANIMAL_COLS  = ["dairy", "beef", "hogs", "poultry", "sheep"]

N_EXCRETION  = {
    "dairy": 120.0, "beef": 70.0, "hogs": 20.0,
    "poultry": 0.6, "sheep": 12.0,
}

MMS_FRAC = {
    "dairy": 0.65, "beef": 0.35, "hogs": 0.90,
    "poultry": 0.95, "sheep": 0.20,
}

# ── Soil drainage ─────────────────────────────────────────────────────────────
DRAINAGE_FACTOR = {
    "Very Poor": 2.0, "Poor": 1.6, "Imperfect": 1.3, "Moderate": 1.0,
    "Moderately Well": 1.0, "Well": 0.9, "Rapid": 0.7, "Very Rapid": 0.7,
    "Unknown": 1.0,
}

for _d in [OUT_DIR, LOCAL_CACHE]:
    os.makedirs(_d, exist_ok=True)
