"""Read one canonical window and apply the accepted fixed RGB/support contract."""

import numpy as np
import rasterio
from affine import Affine
from rasterio.crs import CRS
from rasterio.windows import Window

from .grid import receiving_transform
from .io import local_input
from .science import calculate_ndvi, encode, support


def prepare(package, task, output):
    grid = task["grid"]
    window = Window(*grid["receiving_window"])
    with rasterio.Env(GDAL_PAM_ENABLED="NO", GDAL_CACHEMAX=64*2**20, PROJ_NETWORK="OFF"):
        with rasterio.open(local_input(package, task["inputs"]["canonical"]["path"])) as src:
            if (src.transform != Affine(*grid["affine"])
                    or src.shape != tuple(grid["shape"])
                    or src.crs != CRS.from_user_input(grid["crs"])):
                raise ValueError("Canonical grid differs")
            if list(src.descriptions) != task["inputs"]["bands"]:
                raise ValueError("Canonical band order differs")
            values = src.read(window=window)
            arrays = {
                band.replace("-valid", "_valid"): (
                    values[i].astype(bool) if band.endswith("-valid") else values[i].copy()
                ) for i, band in enumerate(src.descriptions)
            }
    raw = np.stack([arrays[k] for k in ["red", "green", "blue"]])
    valid = support(raw, np.stack([arrays[k+"_valid"] for k in ["red", "green", "blue"]]),
                    arrays["count"], arrays["count_valid"])
    arrays["rgb"] = encode(raw, valid, .0001, 0)
    arrays["rgb_valid"] = valid
    arrays["ndvi"], masks = calculate_ndvi(
        arrays["red"], arrays["nir"], arrays["red_valid"], arrays["nir_valid"],
        (.0001, 0, 0), (.0001, 0, 0),
    )
    arrays["ndvi_valid"] = masks["ndvi_valid"]
    transform = receiving_transform(grid)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "tci-input.tif"
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
        path, "w", driver="GTiff", height=int(window.height), width=int(window.width),
        count=3, dtype="uint8", crs=grid["crs"], transform=transform,
        compress="deflate", nodata=None,
    ) as dst:
        dst.write(arrays["rgb"])
        dst.write_mask(valid.astype(np.uint8) * 255)
        dst.colorinterp = (rasterio.enums.ColorInterp.red,
                          rasterio.enums.ColorInterp.green,
                          rasterio.enums.ColorInterp.blue)
    return arrays, transform, path
