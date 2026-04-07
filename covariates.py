"""
03_covariates.py — Load and reproject covariate GeoTIFFs to the notebook grid.

Provides load_covariate(varname, year) which returns a float32 (NROWS, NCOLS)
array aligned to TRANSFORM_3347 / CRS_3347.

Also exposes load_all(year) returning a dict of all standard covariates.
"""

import os
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling

from src.config import LOCAL_CACHE, CLIM_YEAR, ACI_YEAR
from src.grid import NROWS, NCOLS, TRANSFORM_3347, CRS_3347


def load_covariate(varname: str, year: str, fallback_years=(2024, 2023, 2022)):
    """Load a GeoTIFF from LOCAL_CACHE; auto-reprojects if shape mismatches."""
    for yr in [year] + list(fallback_years):
        fpath = os.path.join(LOCAL_CACHE, f"{varname}_{yr}.tif")
        if not os.path.exists(fpath):
            continue
        with rasterio.open(fpath) as src:
            if src.height == NROWS and src.width == NCOLS:
                arr = src.read(1).astype("float32")
                if src.nodata is not None:
                    arr[arr == src.nodata] = np.nan
                if str(yr) != str(year):
                    print(f"  [{varname}] {year} missing — proxy {yr}")
                return arr

            arr_out = np.full((NROWS, NCOLS), np.nan, dtype="float32")
            reproject(
                source=rasterio.band(src, 1),
                destination=arr_out,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=TRANSFORM_3347,
                dst_crs=CRS_3347,
                resampling=Resampling.bilinear,
                src_nodata=src.nodata,
                dst_nodata=np.nan,
            )
            if src.nodata is not None:
                arr_out[arr_out == src.nodata] = np.nan
            if str(yr) != str(year):
                print(f"  [{varname}] {year} missing — proxy {yr}")
            return arr_out.astype("float32")

    raise FileNotFoundError(
        f"No GeoTIFF found for {varname}_{year} in {LOCAL_CACHE}. "
        "Run 02_gee_export.py first."
    )


def load_all(clim_year: str = CLIM_YEAR, aci_year: str = ACI_YEAR) -> dict:
    covs = {}
    for name in ["NDVI", "TEMP", "PRECIP", "SNOW", "SM"]:
        covs[name] = load_covariate(name, clim_year)
    for name in ["CROP_ANN", "CROP_PAS"]:
        covs[name] = np.nan_to_num(load_covariate(name, aci_year), nan=0.0)
    covs["FREEZE_THAW"] = np.clip(covs["SNOW"] * 0.6, 0, 180).astype("float32")
    return covs


if __name__ == "__main__":
    covs = load_all()
    for k, v in covs.items():
        print(f"  {k}: mean={np.nanmean(v):.4f}")
