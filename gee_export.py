"""
02_gee_export.py — Google Earth Engine authentication & covariate export.

Exports 7 covariate layers for CLIM_YEAR at 500 m / EPSG:3347 to Google Drive
then downloads them to LOCAL_CACHE via the Drive REST API.

Layers exported:
    NDVI_{CLIM_YEAR}     — MODIS MOD13A2, growing-season mean
    TEMP_{CLIM_YEAR}     — ERA5-Land 2 m temp, annual mean (°C)
    PRECIP_{CLIM_YEAR}   — GPM IMERG monthly, annual sum (mm) [unit-corrected]
    SNOW_{CLIM_YEAR}     — MODIS MOD10A1, snow-cover days (Oct–May)
    SM_{CLIM_YEAR}       — SMAP SPL4SMGP, surface SM, growing-season mean
    CROP_ANN_{ACI_YEAR}  — AAFC ACI annual crop fraction
    CROP_PAS_{ACI_YEAR}  — AAFC ACI pasture fraction
"""

import io
import os
import pathlib
import time

import ee
import numpy as np
import rasterio
import rasterio.warp
from google.auth import default as _gauth
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from rasterio.enums import Resampling

from src.config import (
    CLIM_YEAR, ACI_YEAR, LOCAL_CACHE, GEE_PROJECT, GEE_DRIVE_FOLDER,
)
from src.grid import (
    GRID_XMIN, GRID_XMAX, GRID_YMIN, GRID_YMAX,
    NROWS, NCOLS, TRANSFORM_3347, CRS_3347,
)


def authenticate(project: str = GEE_PROJECT) -> bool:
    try:
        ee.Authenticate()
        ee.Initialize(project=project)
        ee.Number(1).getInfo()
        print("GEE connected.")
        return True
    except Exception as e:
        print(f"GEE unavailable ({e}). Physics-based fallbacks will be used.")
        return False


def _build_drive_service():
    creds, _ = _gauth()
    return build("drive", "v3", credentials=creds, cache_discovery=False)


def _gdrive_find_file(svc, prefix, folder=GEE_DRIVE_FOLDER):
    q = f"mimeType='application/vnd.google-apps.folder' and name='{folder}' and trashed=false"
    folders = svc.files().list(q=q, fields="files(id)").execute().get("files", [])
    for fid in [f["id"] for f in folders]:
        q2 = (f"'{fid}' in parents and name contains '{prefix}' and trashed=false "
              f"and mimeType!='application/vnd.google-apps.folder'")
        files = svc.files().list(q=q2, fields="files(id,name,size)").execute().get("files", [])
        if files:
            files.sort(key=lambda x: int(x.get("size", 0)), reverse=True)
            return files[0]["id"], files[0]["name"]
    return None, None


def _gdrive_download(svc, file_id, dest):
    pathlib.Path(dest).parent.mkdir(parents=True, exist_ok=True)
    req = svc.files().get_media(fileId=file_id)
    with io.FileIO(dest, "wb") as fh:
        dl = MediaIoBaseDownload(fh, req, chunksize=50 * 1024 * 1024)
        done = False
        while not done:
            _, done = dl.next_chunk()


def export_and_download(
    image, desc, filename, gee_ok, svc, gee_region,
    scale=500, max_wait=1200,
):
    local = os.path.join(LOCAL_CACHE, f"{filename}.tif")
    if os.path.exists(local):
        return local

    file_id, found = _gdrive_find_file(svc, filename)
    if file_id:
        _gdrive_download(svc, file_id, local)
        return local

    if not gee_ok:
        return None

    task = ee.batch.Export.image.toDrive(
        image=image.toFloat(),
        description=desc,
        folder=GEE_DRIVE_FOLDER,
        fileNamePrefix=filename,
        region=gee_region,
        crs="EPSG:3347",
        scale=scale,
        maxPixels=1e10,
        fileFormat="GeoTIFF",
    )
    task.start()
    t0 = time.time()
    while True:
        state = task.status()["state"]
        if state in ("COMPLETED", "FAILED", "CANCELLED"):
            break
        if time.time() - t0 > max_wait:
            return None
        time.sleep(15)
    if state != "COMPLETED":
        return None

    for _ in range(36):
        file_id, _ = _gdrive_find_file(svc, filename)
        if file_id:
            _gdrive_download(svc, file_id, local)
            return local
        time.sleep(5)
    return None


def read_tif_to_grid(fpath, nodata_fill=np.nan):
    with rasterio.open(fpath) as src:
        _nd = src.nodata
        data, _ = rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=np.empty((NROWS, NCOLS), dtype="float32"),
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=TRANSFORM_3347,
            dst_crs=CRS_3347,
            resampling=Resampling.bilinear,
        )
    data = data.astype("float32")
    if _nd is not None:
        data[data == _nd] = nodata_fill
    data[~np.isfinite(data)] = nodata_fill
    return data


def export_all(gee_ok: bool, svc, gee_region):
    year, aci = CLIM_YEAR, ACI_YEAR
    GS = f"{year}-06-01";  GE = f"{year}-08-31"
    YS = f"{year}-01-01";  YE = f"{year}-12-31"
    WE = f"{year}-05-31"

    layers = {}

    def _load(image, desc, name, fill=np.nan):
        path = export_and_download(image, desc, name, gee_ok, svc, gee_region)
        if path is None:
            raise RuntimeError(f"{desc}: export failed.")
        return read_tif_to_grid(path, nodata_fill=fill)

    img = (ee.ImageCollection("MODIS/061/MOD13A2")
           .filterDate(GS, GE).select("NDVI").mean().multiply(0.0001))
    layers["NDVI"] = _load(img, "NDVI_growing", f"NDVI_{year}")

    img = (ee.ImageCollection("ECMWF/ERA5_LAND/MONTHLY_AGGR")
           .filterDate(YS, YE).select("temperature_2m").mean().subtract(273.15))
    layers["TEMP"] = _load(img, "ERA5_temp_annual", f"TEMP_{year}")

    col = ee.ImageCollection("NASA/GPM_L3/IMERG_MONTHLY_V07").filterDate(YS, YE)
    def _gpm_to_mm(i):
        hrs = ee.Number(i.date().difference(i.date().advance(1, "month"), "hour")).abs()
        return i.select("precipitation").multiply(hrs).rename("precip_mm")
    img = col.map(_gpm_to_mm).sum()
    layers["PRECIP"] = _load(img, "GPM_precip_annual", f"PRECIP_{year}")

    img = (ee.ImageCollection("MODIS/061/MOD10A1")
           .filterDate(f"{int(year)-1}-10-01", WE)
           .select("NDSI_Snow_Cover")
           .map(lambda i: i.gt(10).unmask(0).rename("snow_days"))
           .sum())
    layers["SNOW"] = _load(img, "MODIS_snow_days", f"SNOW_{year}", fill=0.0)

    img = (ee.ImageCollection("NASA/SMAP/SPL4SMGP/008")
           .filterDate(GS, GE).select("sm_surface").mean())
    layers["SM"] = _load(img, "SMAP_SM_growing", f"SM_{year}")

    aci_img = (ee.ImageCollection("AAFC/ACI")
               .filterDate(f"{aci}-01-01", f"{aci}-12-31")
               .first().select("landcover"))
    layers["CROP_ANN"] = _load(
        aci_img.gte(130).And(aci_img.lte(199)).toFloat(),
        "ACI_crop_annual", f"CROP_ANN_{aci}")
    layers["CROP_PAS"] = _load(
        aci_img.eq(110).Or(aci_img.eq(122)).toFloat(),
        "ACI_crop_pasture", f"CROP_PAS_{aci}")

    return layers


if __name__ == "__main__":
    gee_ok = authenticate()
    svc    = _build_drive_service()
    region = (ee.Geometry.Rectangle(
        [GRID_XMIN, GRID_YMIN, GRID_XMAX, GRID_YMAX],
        proj=ee.Projection("EPSG:3347"), geodesic=False) if gee_ok else None)
    layers = export_all(gee_ok, svc, region)
    for k, v in layers.items():
        print(f"  {k}: mean={np.nanmean(v):.4f}")
