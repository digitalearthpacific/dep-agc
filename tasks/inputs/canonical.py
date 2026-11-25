"""Accepted canonical warp, extracted without changing numerical operations.

Origin: native raster preparation at 54cfce2. Source equivalence is checked in tests.
"""

from contextlib import ExitStack
import math
import os
import xml.etree.ElementTree as ET

from affine import Affine
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.windows import Window
from carbon.io import sha256

BANDS = ["red", "green", "blue", "nir", "count"]
CANONICAL_BANDS = [b + suffix for b in BANDS for suffix in ["", "-valid"]]


def pin(path):
    return dict(path=str(path), sha256=sha256(path))


def canonical_vrt(path, blocks, grid, dimensions, bands, crs):
    """Accepted sparse VRT layout, with task CRS and package-relative children."""
    h, w = dimensions
    root = ET.Element("VRTDataset", rasterXSize=str(w), rasterYSize=str(h))
    ET.SubElement(root, "SRS").text = crs
    ET.SubElement(root, "GeoTransform").text = ",".join(map(str, grid.to_gdal()))
    for index, name in enumerate(bands, 1):
        band = ET.SubElement(root, "VRTRasterBand", dataType="UInt16", band=str(index))
        ET.SubElement(band, "Description").text = name
        ET.SubElement(band, "NoDataValue").text = "0"
        for block in blocks:
            source = ET.SubElement(band, "SimpleSource")
            ET.SubElement(source, "SourceFilename", relativeToVRT="1").text = os.path.relpath(
                block["file"]["path"], path.parent).replace("\\", "/")
            ET.SubElement(source, "SourceBand").text = str(index)
            ET.SubElement(source, "SrcRect", xOff="0", yOff="0", xSize="512", ySize="512")
            ET.SubElement(source, "DstRect", xOff=str(block["x"]), yOff=str(block["y"]), xSize="512", ySize="512")
    ET.ElementTree(root).write(path, encoding="utf8", xml_declaration=True)


def canonical(plan, vrts, package, guard):
    """Freeze only selected blocks on the full declared virtual target grid."""
    grid = Affine(*plan["grid"]["affine"])
    height, width = plan["grid"]["shape"]
    x, y, w, h = plan["selected"]["global_input_window"]
    blocks = []
    with ExitStack() as stack:
        readers = []
        for name in CANONICAL_BANDS:
            src = stack.enter_context(rasterio.open(vrts[name]))
            readers.append(stack.enter_context(WarpedVRT(
                src, crs=plan["grid"]["crs"], transform=grid, width=width, height=height,
                resampling=Resampling.nearest, tolerance=1e-12)))
        for xx in range(x//512*512, math.ceil((x+w)/512)*512, 512):
            for yy in range(y//512*512, math.ceil((y+h)/512)*512, 512):
                guard()
                window = Window(xx, yy, 512, 512)
                values = np.stack([src.read(1, window=window) for src in readers])
                path = package / "inputs/canonical" / f"{xx}-{yy}.tif"
                path.parent.mkdir(parents=True, exist_ok=True)
                with rasterio.open(path, "w", driver="GTiff", height=512, width=512, count=10,
                                   dtype="uint16", crs=plan["grid"]["crs"],
                                   transform=grid*Affine.translation(xx, yy), compress="deflate",
                                   tiled=True, blockxsize=256, blockysize=256) as ds:
                    ds.write(values)
                    for i, name in enumerate(CANONICAL_BANDS, 1):
                        ds.set_band_description(i, name)
                blocks.append(dict(x=xx, y=yy, file=pin(path)))
    path = package / "inputs/canonical-selected.vrt"
    canonical_vrt(path, blocks, grid, [height, width], CANONICAL_BANDS, plan["grid"]["crs"])
    return blocks, path


def canonical_receiving(grid, window, values, package):
    """Pin an already aligned receiving window on the worker's canonical grid."""
    x, y, width, height = window['global_input_window']
    affine = Affine(*grid['affine'])
    if values.shape != (len(CANONICAL_BANDS), height, width) or values.dtype != np.uint16:
        raise ValueError('Canonical receiving values differ from the declared window')
    source = package/'inputs/native-receiving.tif'
    source.parent.mkdir(parents=True)
    with rasterio.open(source, 'w', driver='GTiff', height=height, width=width,
                       count=len(CANONICAL_BANDS), dtype='uint16', crs=grid['crs'],
                       transform=affine * Affine.translation(x, y), compress='deflate',
                       tiled=True, blockxsize=256, blockysize=256) as dst:
        dst.write(values)
        dst.descriptions = tuple(CANONICAL_BANDS)

    h, w = grid['shape']
    vrt = package/'inputs/canonical-selected.vrt'
    root = ET.Element('VRTDataset', rasterXSize=str(w), rasterYSize=str(h))
    ET.SubElement(root, 'SRS').text = grid['crs']
    ET.SubElement(root, 'GeoTransform').text = ','.join(map(str, affine.to_gdal()))
    for index, name in enumerate(CANONICAL_BANDS, 1):
        band = ET.SubElement(root, 'VRTRasterBand', dataType='UInt16', band=str(index))
        ET.SubElement(band, 'Description').text = name
        entry = ET.SubElement(band, 'SimpleSource')
        ET.SubElement(entry, 'SourceFilename', relativeToVRT='1').text = source.name
        ET.SubElement(entry, 'SourceBand').text = str(index)
        ET.SubElement(entry, 'SrcRect', xOff='0', yOff='0', xSize=str(width), ySize=str(height))
        ET.SubElement(entry, 'DstRect', xOff=str(x), yOff=str(y), xSize=str(width), ySize=str(height))
    ET.ElementTree(root).write(vrt, encoding='utf8', xml_declaration=True)
    return [dict(file=pin(source))], vrt


