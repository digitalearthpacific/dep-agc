"""Pinned saved-native loading, then the accepted canonical preparation."""

from copy import deepcopy
from pathlib import Path
import shutil
import xml.etree.ElementTree as ET

from affine import Affine
from dep_tools.loaders import Loader
import numpy as np
import rasterio
from rasterio.windows import Window
from shapely.geometry import box

from tasks.inputs.canonical import BANDS, canonical
from carbon.grid import receiving_transform
from carbon.io import check_file, local_input, read_json, sha256, write_json


def native_mosaic(path, records, band, valid=False):
    """geomad_sources.mosaic_vrt at4277015: same disjoint native layout.

    Kept local to avoid importing country transport/storage controllers.
    No warp, validity inference, overlap precedence or source substitution here.
    """
    values = [r['bands'][band] for r in records]
    left = min(v['grid'][2] for v in values)
    top = max(v['grid'][5] for v in values)
    right = max(v['grid'][2] + v['shape'][1]*10 for v in values)
    bottom = min(v['grid'][5] - v['shape'][0]*10 for v in values)
    root = ET.Element('VRTDataset', rasterXSize=str(round((right-left)/10)),
                      rasterYSize=str(round((top-bottom)/10)))
    ET.SubElement(root, 'SRS').text = 'EPSG:'+str(values[0]['crs'])
    ET.SubElement(root, 'GeoTransform').text = ','.join(map(str, [left,10,0,top,0,-10]))
    b = ET.SubElement(root, 'VRTRasterBand', dataType='Byte' if valid else 'UInt16', band='1')
    ET.SubElement(b, 'NoDataValue').text = str(0 if valid else values[0]['nodata'])
    rectangles = []
    for v in values:
        x, y = round((v['grid'][2]-left)/10), round((top-v['grid'][5])/10)
        h, w = v['shape']
        rect = box(x, y, x+w, y+h)
        if any(rect.intersection(other).area > 0 for other in rectangles):
            raise ValueError('Overlapping GeoMAD tile ownership')
        rectangles.append(rect)
        src = ET.SubElement(b, 'SimpleSource')
        ET.SubElement(src, 'SourceFilename', relativeToVRT='0').text = v['valid' if valid else 'native']['path']
        ET.SubElement(src, 'SourceBand').text = '1'
        ET.SubElement(src, 'SrcRect', xOff='0', yOff='0', xSize=str(w), ySize=str(h))
        ET.SubElement(src, 'DstRect', xOff=str(x), yOff=str(y), xSize=str(w), ySize=str(h))
    ET.ElementTree(root).write(path, encoding='utf8', xml_declaration=True)


def local_pin(root, spec):
    name = Path(spec['path'])
    p = (root/name).resolve()
    if (name.is_absolute() or '..' in name.parts or '\\' in spec['path'] or ':' in spec['path']
            or not p.is_relative_to(root.resolve()) or p.is_symlink()):
        raise ValueError('Native cache reference escapes root')
    if not p.is_file():
        raise FileNotFoundError(p)
    check_file(p, spec['sha256'])
    return p


def compare_receiving(fresh, accepted, task):
    """Full ten-band values and masks, including source-invalid stored pixels."""
    window = Window(*task['grid']['receiving_window'])
    with rasterio.open(fresh) as a, rasterio.open(accepted) as b:
        if (a.crs, a.transform, a.shape, a.dtypes, a.descriptions, a.nodatavals) != (
                b.crs, b.transform, b.shape, b.dtypes, b.descriptions, b.nodatavals):
            raise ValueError('Canonical spatial/band metadata differs')
        if (not np.array_equal(a.read(window=window), b.read(window=window))
                or not np.array_equal(a.read_masks(window=window), b.read_masks(window=window))
                or not np.array_equal(a.dataset_mask(window=window), b.dataset_mask(window=window))):
            raise ValueError('Canonical receiving values/validity differ')


class NativeGeoMADLoader(Loader):
    def __init__(self, index, index_sha256, static_inputs, task_file, output):
        self.index, self.index_sha256 = index, index_sha256
        self.static_inputs, self.task_file, self.output = static_inputs, task_file, output

    def load(self, area):
        check_file(self.index, self.index_sha256)
        index, task = read_json(self.index), read_json(self.task_file)
        if (index['year'], index['version']) != (task['year'], task['source_version']):
            raise ValueError('GeoMAD source identity differs')
        if (area.transform != receiving_transform(task['grid'])
                or area.shape != tuple(reversed(task['grid']['receiving_window'][2:]))
                or str(area.crs) != str(rasterio.crs.CRS.from_user_input(task['grid']['crs']))):
            raise ValueError('DEP area differs from accepted receiving grid')
        self.output.mkdir(parents=True, exist_ok=False)
        records = deepcopy(index['records'])
        for rec in records:
            for band in BANDS:
                v = rec['bands'][band]
                for kind in ['native', 'valid']:
                    v[kind]['path'] = str(local_pin(self.index.parent, v[kind]))
                with rasterio.open(v['native']['path']) as data, rasterio.open(v['valid']['path']) as mask:
                    expected = (rasterio.crs.CRS.from_epsg(v['crs']), Affine(*v['grid']), tuple(v['shape']))
                    if ((data.crs, data.transform, data.shape) != expected
                            or (mask.crs, mask.transform, mask.shape) != expected
                            or data.dtypes != ('uint16',) or mask.dtypes != ('uint8',)
                            or data.nodata != v['nodata'] or data.transform.a != 10 or data.transform.e != -10
                            or data.transform.b != 0 or data.transform.d != 0):
                        raise ValueError('Saved native spatial/type/NoData contract differs')
                    if not np.array_equal(data.read_masks(1)>0, mask.read(1)>0):
                        raise ValueError('Native source validity differs from explicit mask')
        vrts = {}
        for band in BANDS:
            for valid in [False, True]:
                key = band+('-valid' if valid else '')
                path = self.output/(key+'.vrt')
                native_mosaic(path, records, band, valid)
                vrts[key] = path
        target = self.output/'package'
        target.mkdir()
        plan = dict(grid=task['grid'], selected=dict(global_input_window=task['grid']['global_input_window']))
        with rasterio.Env(GDAL_CACHEMAX=64*2**20, GDAL_PAM_ENABLED='NO', PROJ_NETWORK='OFF'):
            blocks, vrt = canonical(plan, vrts, target, lambda: None)
        # Retain the pinned task verbatim: any new file identity mismatch stops here.
        check_file(vrt, task['inputs']['canonical']['sha256'])
        for row in task['inputs']['blocks']:
            check_file(target/row['path'], row['sha256'])
        for row in [task['inputs']['owner_order'], *[o['file'] for o in task['inputs']['owners']]]:
            src = local_input(self.static_inputs, row['path'])
            check_file(src, row['sha256'])
            dst = target/row['path']; dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst); check_file(dst, row['sha256'])
        # Canonical identities are input pins, not prior output rasters. Runtime
        # never opens a comparison package; full receiving comparisons are a
        # separate qualification check.
        shutil.copyfile(self.task_file, target/'task.json')
        write_json(self.output/'load.json', dict(native_index_sha256=sha256(self.index),
            source_items=index['source_items'], native_records=len(records), canonical_blocks=len(blocks),
            canonical_input_pins_exact=True, task_sha256=sha256(self.task_file),
            warp='accepted nearest / tolerance1e-12 / full global grid', new_provider_bytes=0))
        return dict(task=task, package=target, index=index, native_index_sha256=sha256(self.index),
                    loader_receipt=self.output/'load.json')
