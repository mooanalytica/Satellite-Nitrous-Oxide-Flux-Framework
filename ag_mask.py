"""
04_ag_mask.py — Agricultural mask (AAFC ACI) and soil properties (SLC v3r2).

Exports:
    AG_MASK        — bool (NROWS, NCOLS): True for agricultural pixels
    PROV_GRID      — str  (NROWS, NCOLS): 2-letter province code per pixel
    CLAY_GRID      — float32 (NROWS, NCOLS): clay %
    OC_GRID        — float32 (NROWS, NCOLS): organic carbon %
    PH_GRID        — float32 (NROWS, NCOLS): soil pH
    DRAIN_INT_GRID — uint8   (NROWS, NCOLS): drainage class index
    DRAIN_F_GRID   — float32 (NROWS, NCOLS): drainage multiplication factor
    DRAIN_POOR     — bool    (NROWS, NCOLS): True for Poor / Very Poor drainage
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
from rasterio.features import rasterize
from rasterio.warp import reproject, Resampling
import rasterio
from shapely.geometry import box as _bbox_geom

from src.config import (
    SLC_SHAPEFILE_PATH, SLC_SOIL_CSV_PATH, ACI_YEAR, LOCAL_CACHE,
    PROVINCES, DRAINAGE_FACTOR,
)
from src.grid import (
    NROWS, NCOLS, PIXEL_M, PIXEL_HA, PIXEL_M,
    GRID_XMIN, GRID_XMAX, GRID_YMIN, GRID_YMAX,
    TRANSFORM_3347, CRS_3347, CANADA_MASK, prov_gdf, PIX_LON, PIX_LAT,
)

DRAINAGE_INT = {k: i for i, k in enumerate(DRAINAGE_FACTOR.keys())}
INT_DRAINAGE  = {v: k for k, v in DRAINAGE_INT.items()}


# ── Agricultural mask from AAFC ACI ──────────────────────────────────────────

def _load_aci(path):
    with rasterio.open(path) as src:
        if src.height == NROWS and src.width == NCOLS:
            arr = src.read(1).astype("float32")
            if src.nodata is not None:
                arr[arr == src.nodata] = np.nan
            return arr
        arr_out = np.full((NROWS, NCOLS), np.nan, dtype="float32")
        reproject(
            source=rasterio.band(src, 1), destination=arr_out,
            src_transform=src.transform, src_crs=src.crs,
            dst_transform=TRANSFORM_3347, dst_crs=CRS_3347,
            resampling=Resampling.bilinear,
            src_nodata=src.nodata, dst_nodata=np.nan,
        )
        if src.nodata is not None:
            arr_out[arr_out == src.nodata] = np.nan
        return arr_out.astype("float32")


_ann_path = os.path.join(LOCAL_CACHE, f"CROP_ANN_{ACI_YEAR}.tif")
_pas_path = os.path.join(LOCAL_CACHE, f"CROP_PAS_{ACI_YEAR}.tif")

if os.path.exists(_ann_path):
    CROP_ANN = _load_aci(_ann_path)
    CROP_PAS = _load_aci(_pas_path) if os.path.exists(_pas_path) else np.zeros((NROWS, NCOLS), "float32")
    AG_MASK  = (
        ((np.nan_to_num(CROP_ANN, nan=0.0) > 0) |
         (np.nan_to_num(CROP_PAS, nan=0.0) > 0))
        & CANADA_MASK
    )
else:
    print("WARNING: ACI GeoTIFFs not found — using coordinate-based fallback.")
    AG_MASK  = (
        (PIX_LON >= -141) & (PIX_LON <= -52) &
        (PIX_LAT >=   42) & (PIX_LAT <=  55)
    ).astype(bool) & CANADA_MASK
    CROP_ANN = np.where(AG_MASK, 0.5, 0.0).astype("float32")
    CROP_PAS = np.where(AG_MASK, 0.2, 0.0).astype("float32")

_ag_mha = AG_MASK.sum() * PIXEL_HA / 1e6
if _ag_mha > 200:
    raise RuntimeError(f"AG_MASK = {_ag_mha:.1f} Mha — too large. Re-run 02_gee_export.py.")
print(f"Ag mask: {AG_MASK.sum():,} pixels = {_ag_mha:.1f} Mha")


# ── Soil properties from SLC v3r2 ─────────────────────────────────────────────

CLAY_GRID      = np.full((NROWS, NCOLS), 25.0,                dtype="float32")
OC_GRID        = np.full((NROWS, NCOLS),  2.0,                dtype="float32")
PH_GRID        = np.full((NROWS, NCOLS),  6.5,                dtype="float32")
DRAIN_INT_GRID = np.full((NROWS, NCOLS), DRAINAGE_INT["Well"], dtype="uint8")

NROW_CHUNK = 200

def _rasterize_chunked(shapes, grid, fill, dtype):
    from rasterio.transform import from_origin as _fo
    for r0 in range(0, NROWS, NROW_CHUNK):
        r1    = min(r0 + NROW_CHUNK, NROWS)
        y_top = GRID_YMAX - r0 * PIXEL_M
        tr_c  = _fo(GRID_XMIN, y_top, PIXEL_M, PIXEL_M)
        strip = rasterize(shapes, out_shape=(r1 - r0, NCOLS),
                          transform=tr_c, fill=fill, dtype=dtype)
        grid[r0:r1, :] = strip
    return grid


if os.path.exists(SLC_SHAPEFILE_PATH):
    slc_raw = gpd.read_file(SLC_SHAPEFILE_PATH)
    _rb     = slc_raw.total_bounds
    _deg    = -200 < _rb[0] < 200
    if _deg:
        slc_raw = slc_raw.set_crs("EPSG:4269", allow_override=True).to_crs("EPSG:3347")
    elif slc_raw.crs is None or slc_raw.crs.to_epsg() != 3347:
        if slc_raw.crs is None:
            slc_raw = slc_raw.set_crs("EPSG:4269", allow_override=True)
        slc_raw = slc_raw.to_crs("EPSG:3347")

    _box = _bbox_geom(GRID_XMIN, GRID_YMIN, GRID_XMAX, GRID_YMAX)
    slc_raw = slc_raw[slc_raw.geometry.intersects(_box)].copy().reset_index(drop=True)
    if "POLY_ID" in slc_raw.columns and "SL_ID" not in slc_raw.columns:
        slc_raw = slc_raw.rename(columns={"POLY_ID": "SL_ID"})
    if "SL_ID" not in slc_raw.columns:
        slc_raw["SL_ID"] = range(len(slc_raw))

    if os.path.exists(SLC_SOIL_CSV_PATH):
        csv    = pd.read_csv(SLC_SOIL_CSV_PATH)
        jcol   = "POLY_ID" if "POLY_ID" in csv.columns else "SL_ID"
        csv    = csv.set_index(jcol)
        for attr, default, col in [
            ("clay_pct", 25.0, "clay_pct"),
            ("organic_carbon", 2.0, "organic_carbon"),
            ("ph_water", 6.5, "soil_ph"),
        ]:
            slc_raw[col] = (
                slc_raw["SL_ID"].map(csv[attr]).fillna(default)
                if attr in csv.columns else default)
        slc_raw["drainage_class"] = (
            slc_raw["SL_ID"].map(csv.get("drainage_class", pd.Series())).fillna("Well"))
    else:
        rng = np.random.default_rng(7)
        cx  = slc_raw.geometry.centroid.x.values / 1e6
        slc_raw["clay_pct"]       = np.clip(25 + 8*cx + rng.normal(0, 8, len(slc_raw)), 5, 65)
        slc_raw["organic_carbon"] = np.clip(rng.lognormal(0.7, 0.5, len(slc_raw)), 0.5, 12)
        slc_raw["soil_ph"]        = np.clip(rng.normal(6.5, 0.8, len(slc_raw)), 4.5, 8.5)
        slc_raw["drainage_class"] = rng.choice(
            list(DRAINAGE_FACTOR.keys()), len(slc_raw),
            p=[0.05, 0.15, 0.20, 0.25, 0.15, 0.15, 0.03, 0.02, 0.00])

    for col, grid, fill in [
        ("clay_pct",       CLAY_GRID, 25.0),
        ("organic_carbon", OC_GRID,    2.0),
        ("soil_ph",        PH_GRID,    6.5),
    ]:
        shapes = [(g, float(v)) for g, v in zip(slc_raw.geometry, slc_raw[col])
                  if g is not None and not g.is_empty and np.isfinite(float(v))]
        _rasterize_chunked(shapes, grid, fill, "float32")

    drain_shapes = [
        (g, int(DRAINAGE_INT.get(d, DRAINAGE_INT["Well"])))
        for g, d in zip(slc_raw.geometry, slc_raw["drainage_class"])
        if g is not None and not g.is_empty
    ]
    _rasterize_chunked(drain_shapes, DRAIN_INT_GRID, DRAINAGE_INT["Well"], "uint8")

DRAIN_F_GRID = np.vectorize(
    lambda i: DRAINAGE_FACTOR[INT_DRAINAGE[i]])(DRAIN_INT_GRID).astype("float32")
DRAIN_POOR   = np.isin(DRAIN_INT_GRID,
                       [DRAINAGE_INT["Very Poor"], DRAINAGE_INT["Poor"]])


# ── Province grid ─────────────────────────────────────────────────────────────

def _coord_province(lons, lats):
    prov = np.full(lons.shape, "NL", dtype="U2")
    prov = np.where((lons < -120) & (lats < 54),              "BC", prov)
    prov = np.where((lons >= -120) & (lons < -110),           "AB", prov)
    prov = np.where((lons >= -110) & (lons < -102),           "SK", prov)
    prov = np.where((lons >= -102) & (lons < -95),            "MB", prov)
    prov = np.where((lons >= -95)  & (lons < -80),            "ON", prov)
    prov = np.where((lons >= -80)  & (lons < -66),            "QC", prov)
    prov = np.where((lons >= -66)  & (lons < -64) & (lats > 45.5), "NB", prov)
    prov = np.where((lons >= -64)  & (lons < -61) & (lats < 47),   "NS", prov)
    prov = np.where((lons >= -64)  & (lons < -62) & (lats > 46),   "PE", prov)
    return prov

try:
    prov_name_col = next(
        (c for c in prov_gdf.columns if c.upper() in ["PRENAME", "PRNAME", "PROVNAME", "NAME"]),
        None)
    code2int = {k: i + 1 for i, k in enumerate(PROVINCES.keys())}
    int2code  = {v: k for k, v in code2int.items()}
    PROV_INT  = np.zeros((NROWS, NCOLS), dtype="uint8")
    for pcode, pint in code2int.items():
        pname = PROVINCES[pcode]
        pg    = prov_gdf[prov_gdf[prov_name_col] == pname] if prov_name_col else gpd.GeoDataFrame()
        if pg.empty:
            continue
        PROV_INT = np.maximum(
            PROV_INT,
            rasterize([(g, pint) for g in pg.geometry if g is not None],
                      out_shape=(NROWS, NCOLS), transform=TRANSFORM_3347,
                      fill=0, dtype="uint8"))
    PROV_GRID = np.vectorize(lambda i: int2code.get(i, "NL"))(PROV_INT)
except Exception:
    from src.grid import PIX_LON, PIX_LAT
    PROV_GRID = _coord_province(PIX_LON, PIX_LAT)

if __name__ == "__main__":
    print(f"AG_MASK: {AG_MASK.sum():,} pixels = {AG_MASK.sum() * PIXEL_HA / 1e6:.1f} Mha")
    print(f"DRAIN_F mean (ag): {DRAIN_F_GRID[AG_MASK].mean():.3f}")
