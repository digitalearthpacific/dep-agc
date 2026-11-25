"""Native tile masks and carbon products."""

import numpy as np
import rasterio
from rasterio.crs import CRS

from .io import sha256, write_raster
from .policy import LAND_CLASSES, RULE
from .science import product_masks


def pixel_products(task, arrays, raw_height, cover, processed):
    d = task["grid"]["context"]
    h, w = task["grid"]["retained_shape"]
    sl = np.s_[d:d+h, d:d+w]
    height = raw_height[sl]
    hv = arrays["rgb_valid"][sl] & np.isfinite(height) & (height >= 0)
    source = arrays["rgb_valid"][sl] & arrays["red_valid"][sl] & arrays["nir_valid"][sl]
    land = processed & np.isin(cover, LAND_CLASSES)
    unknown = processed & (cover == RULE["unknown_class"])
    assessed, status, carbon = product_masks(
        land, land & arrays["rgb_valid"][sl], source, height, hv,
        arrays["ndvi"][sl], arrays["ndvi_valid"][sl], unknown, policy=task["policy"],
    )
    status[~processed] = 6
    layers = [
        ("RGB", arrays["rgb"][:, sl[0], sl[1]], processed & arrays["rgb_valid"][sl], None),
        ("height", height, land & hv, -9999),
        ("carbon", carbon, assessed, -9999),
        ("status", status, processed, None),
        ("NDVI", arrays["ndvi"][sl], land & arrays["ndvi_valid"][sl], -9999),
        ("count", arrays["count"][sl], processed & arrays["count_valid"][sl], None),
        ("WorldCover", cover, processed, None),
    ]
    return layers


def write_products(output, layers, grid, crs):
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, value, valid, nodata in layers:
        path = output / (name + ".tif")
        write_raster(path, value, valid, grid, crs, nodata)
        files[name] = dict(path=path.name, sha256=sha256(path))
    return dict(files=files, grid=list(grid)[:6], crs=crs)


def read_worldcover(path, task, grid):
    with rasterio.open(path) as src:
        if (src.driver != "GTiff" or src.count != 1 or src.dtypes != ("uint8",) or src.nodata is not None
                or src.transform != grid or src.shape != tuple(task["grid"]["retained_shape"])
                or src.crs != CRS.from_user_input(task["grid"]["crs"])
                or src.files != [str(path)]):
            raise ValueError("Expected a self-contained WorldCover raster on the retained grid")
        cover, processed = src.read(1), src.read_masks(1) > 0
    classes = [*LAND_CLASSES, RULE["water_class"], RULE["unknown_class"]]
    if not processed.any() or not np.isin(cover[processed], classes).all():
        raise ValueError("WorldCover has no processed pixels or has invalid classes")
    return cover, processed


def tile_products(task, arrays, raw_height, grid, worldcover, output):
    """Write tile products from aligned WorldCover, without country geometry."""
    layers = pixel_products(task, arrays, raw_height, *worldcover)
    return write_products(output, layers, grid, task["grid"]["crs"])
