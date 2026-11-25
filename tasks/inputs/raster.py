"""Check GeoMAD encodings and prepare bounded native inputs from local assets."""

import math

from affine import Affine
import numpy as np
import rasterio
from rasterio.windows import Window, from_bounds
from shapely.geometry import box
from shapely.ops import unary_union

from tasks.loading import native_mosaic
from .canonical import BANDS, pin


def check_native_geomad(src, item, band):
    spec = item['assets'][band]['raster:bands'][0]
    if (src.crs is None or src.crs.to_epsg() != 3832 or src.shape != (9600, 9600)
            or src.transform != Affine(*item['properties']['proj:transform'][:6])
            or src.dtypes != ('uint16',) or src.nodata != spec['nodata']):
        raise ValueError('Actual GeoMAD TIFF grid/type/nodata differs')


def native_blocks(window, items, folder, route):
    """Read bounded native blocks and validity masks from pinned local GeoMAD."""
    bounds = window["input_bounds"]
    records = []
    for item in items:
        intersection = box(*item["properties"]["proj:bbox"]).intersection(box(*bounds))
        if intersection.area <= 0:
            continue
        affine = Affine(*item["properties"]["proj:transform"][:6])
        region = from_bounds(*intersection.bounds, affine)
        coordinates = [(xx, yy)
                       for yy in range(max(0, math.floor(region.row_off/1024)), min(10, math.ceil((region.row_off+region.height)/1024)))
                       for xx in range(max(0, math.floor(region.col_off/1024)), min(10, math.ceil((region.col_off+region.width)/1024)))]
        indexed = {}
        for band in BANDS:
            spec = item["assets"][band]["raster:bands"][0]
            with rasterio.open(route[item["assets"][band]["href"]]) as src:
                check_native_geomad(src, item, band)
                for xx, yy in coordinates:
                    block = Window(xx*1024, yy*1024, min(1024, 9600-xx*1024), min(1024, 9600-yy*1024))
                    data = src.read(1, window=block)
                    valid = src.read_masks(1, window=block) > 0
                    transform = src.window_transform(block)
                    target = folder / item["id"] / f"{xx}-{yy}"
                    target.mkdir(parents=True, exist_ok=True)
                    profile = dict(driver="GTiff", crs="EPSG:3832", transform=transform,
                                   height=int(block.height), width=int(block.width), count=1,
                                   compress="deflate", tiled=True, blockxsize=256, blockysize=256)
                    native, mask = target/(band+".tif"), target/(band+"-valid.tif")
                    with rasterio.open(native, "w", **profile, dtype="uint16", nodata=spec["nodata"]) as dst:
                        dst.write(data, 1)
                    with rasterio.open(mask, "w", **profile, dtype="uint8", nodata=0) as dst:
                        dst.write(valid.astype(np.uint8), 1)
                    with rasterio.open(native) as saved:
                        if (not np.array_equal(saved.read(1), data)
                                or not np.array_equal(saved.read_masks(1) > 0, valid)):
                            raise ValueError("Decoded native bytes/masks changed on write")
                    record = indexed.setdefault((xx, yy), dict(bands={}))
                    record["bands"][band] = dict(native=pin(native), valid=pin(mask),
                                                grid=list(transform)[:6], shape=list(data.shape),
                                                crs=3832, nodata=spec["nodata"])
            print("NATIVE", item["id"], band, len(coordinates), flush=True)
        records.extend(indexed.values())
    if not records:
        raise ValueError("No annual sources intersect the full context")
    coverage = box(*bounds).difference(unary_union([
        box(*rasterio.transform.array_bounds(*r["bands"]["red"]["shape"], Affine(*r["bands"]["red"]["grid"])))
        for r in records]))
    # This checks native roster footprint, not availability of observations.
    if coverage.area > 1:
        raise ValueError("DEP tile window is missing real GeoMAD context")
    vrts = {}
    for band in BANDS:
        for valid in [False, True]:
            key = band + ("-valid" if valid else "")
            path = folder / (key+".vrt")
            native_mosaic(path, records, band, valid)  # Rejects any overlap.
            vrts[key] = path
    return vrts
