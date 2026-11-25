"""Pinned saved-native loading, then the accepted canonical preparation."""

from pathlib import Path
import xml.etree.ElementTree as ET

from shapely.geometry import box

from carbon.io import check_file, sha256


def native_mosaic(path, records, band, valid=False):
    """Mosaic disjoint native blocks without changing values or validity."""
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


def pin(path, root):
    return dict(path=path.relative_to(root).as_posix(), sha256=sha256(path))
