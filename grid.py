"""
01_grid.py — 500 m EPSG:3347 grid definition.

Derives bounding box from province boundaries shapefile, snaps outward to
the nearest 500 m boundary, and rasterises the Canada land mask.

Exports (module-level):
    GRID_XMIN, GRID_XMAX, GRID_YMIN, GRID_YMAX
    NROWS, NCOLS, TOTAL_PIX, PIXEL_M, PIXEL_HA
    TRANSFORM_3347, CRS_3347, CRS_4326
    PIX_X, PIX_Y         — pixel centre coords in EPSG:3347 (NROWS, NCOLS)
    PIX_LON, PIX_LAT     — pixel centre coords in WGS84  (NROWS, NCOLS)
    CANADA_MASK          — bool (NROWS, NCOLS)
    prov_gdf             — GeoDataFrame of province boundaries (EPSG:3347)
"""

import numpy as np
import geopandas as gpd
from pyproj import Transformer
from rasterio.transform import from_origin
from rasterio.features import rasterize
from rasterio.crs import CRS

from src.config import PROV_BOUNDARIES_PATH, PIXEL_M, PIXEL_HA  # noqa

PIXEL_M  = 500
PIXEL_HA = PIXEL_M ** 2 / 10_000

prov_gdf = gpd.read_file(PROV_BOUNDARIES_PATH).to_crs("EPSG:3347")
_b       = prov_gdf.total_bounds

GRID_XMIN = int(np.floor(_b[0] / PIXEL_M) * PIXEL_M)
GRID_XMAX = int(np.ceil (_b[2] / PIXEL_M) * PIXEL_M)
GRID_YMIN = int(np.floor(_b[1] / PIXEL_M) * PIXEL_M)
GRID_YMAX = int(np.ceil (_b[3] / PIXEL_M) * PIXEL_M)

NCOLS     = int((GRID_XMAX - GRID_XMIN) / PIXEL_M)
NROWS     = int((GRID_YMAX - GRID_YMIN) / PIXEL_M)
TOTAL_PIX = NCOLS * NROWS

TRANSFORM_3347 = from_origin(GRID_XMIN, GRID_YMAX, PIXEL_M, PIXEL_M)
CRS_3347       = CRS.from_epsg(3347)
CRS_4326       = CRS.from_epsg(4326)

xs_1d = np.arange(NCOLS) * PIXEL_M + GRID_XMIN + PIXEL_M / 2
ys_1d = GRID_YMAX - (np.arange(NROWS) * PIXEL_M + PIXEL_M / 2)
PIX_X, PIX_Y = np.meshgrid(xs_1d, ys_1d)

_tf_to_4326         = Transformer.from_crs(3347, 4326, always_xy=True)
_lon_flat, _lat_flat = _tf_to_4326.transform(PIX_X.ravel(), PIX_Y.ravel())
PIX_LON = _lon_flat.reshape(NROWS, NCOLS).astype("float32")
PIX_LAT = _lat_flat.reshape(NROWS, NCOLS).astype("float32")

CANADA_MASK = rasterize(
    [(geom, 1) for geom in prov_gdf.geometry if geom is not None],
    out_shape=(NROWS, NCOLS),
    transform=TRANSFORM_3347,
    fill=0,
    dtype="uint8",
) > 0

if __name__ == "__main__":
    print(f"Grid: {NCOLS} cols × {NROWS} rows = {TOTAL_PIX / 1e6:.1f}M pixels")
    print(f"Canada land mask: {CANADA_MASK.sum() / 1e6:.2f}M pixels "
          f"= {CANADA_MASK.sum() * PIXEL_HA / 1e6:.1f} Mha")
