"""Accepted canonical warp, extracted without changing numerical operations.

Origin: spc_agb_inputs.raster at54cfce2; source equivalence checked in tests.
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
from spc_agb_worker.io import sha256

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


