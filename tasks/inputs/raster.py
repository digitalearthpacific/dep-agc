"""Bounded native reads and the accepted frozen nearest-neighbour canonical warp.

Origins: sparse_annual_sources.native_sources/canonical/canonical_vrt and
geomad_sources.mosaic_vrt at 4277015. Country execution/reuse is not imported.
"""

from contextlib import ExitStack
import math
import os
from pathlib import Path
import xml.etree.ElementTree as ET

from affine import Affine
import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds, reproject
from rasterio.windows import Window, from_bounds
from shapely.geometry import box
from shapely.ops import unary_union

from carbon.grid import MODEL_PATCH
from carbon.io import sha256
from carbon.io import check_file, read_json, write_json

from .canonical import BANDS, CANONICAL_BANDS, pin, canonical_vrt, canonical


def receiving_source_bounds(plan, window):
    """Preserve legacy padding; DEP tiles require only the real model context."""
    grid = Affine(*plan["grid"]["affine"])
    x, y, w, h = window["global_input_window"]
    if 'tile_id' not in plan['grid']:
        w = math.ceil(w / MODEL_PATCH) * MODEL_PATCH
        h = math.ceil(h / MODEL_PATCH) * MODEL_PATCH
    return rasterio.windows.bounds(Window(x, y, w, h), grid)


def check_native_geomad(src, item, band):
    spec = item['assets'][band]['raster:bands'][0]
    if (src.crs is None or src.crs.to_epsg() != 3832 or src.shape != (9600, 9600)
            or src.transform != Affine(*item['properties']['proj:transform'][:6])
            or src.dtypes != ('uint16',) or src.nodata != spec['nodata']):
        raise ValueError('Actual GeoMAD TIFF grid/type/nodata differs')


def native_blocks(plan, folder, route, guard, *, resume=False):
    """Same disjoint <=1024px native blocks/validity as accepted sparse producer."""
    from tasks.loading import native_mosaic as mosaic_vrt

    bounds = transform_bounds(plan["grid"]["crs"], 3832,
                              *receiving_source_bounds(plan, plan["selected"]), densify_pts=101)
    records = []
    for item in plan["source_items"]:
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
                    guard()
                    window = Window(xx*1024, yy*1024, min(1024, 9600-xx*1024), min(1024, 9600-yy*1024))
                    target = folder / item["id"] / f"{xx}-{yy}"
                    receipt = target/(band+'-receipt.json')
                    expected = dict(source_href=item['assets'][band]['href'],
                                    source_window=list(window.flatten()),
                                    grid=list(src.window_transform(window))[:6],
                                    crs=3832, shape=[int(window.height), int(window.width)],
                                    nodata=spec['nodata'])
                    if resume and receipt.exists():
                        value = read_json(receipt)
                        if any(value[k] != v for k, v in expected.items()):
                            raise ValueError('Changed native cache contract; explicit new cache required')
                        for kind in ['native', 'valid']:
                            value[kind]['path'] = str(target/value[kind]['path'])
                            check_file(Path(value[kind]['path']), value[kind]['sha256'])
                        record = indexed.setdefault((xx, yy), dict(item=item['id'], bands={}, files=[]))
                        record['bands'][band] = value
                        record['files'].extend([value['native'], value['valid']])
                        continue
                    if resume and any((target/name).exists() for name in [band+'.tif', band+'-valid.tif']):
                        raise FileExistsError('Uncommitted native files preserved: '+str(target))
                    data = src.read(1, window=window)
                    valid = src.read_masks(1, window=window) > 0
                    transform = src.window_transform(window)
                    target = folder / item["id"] / f"{xx}-{yy}"
                    target.mkdir(parents=True, exist_ok=True)
                    profile = dict(driver="GTiff", crs="EPSG:3832", transform=transform,
                                   height=int(window.height), width=int(window.width), count=1,
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
                    record = indexed.setdefault((xx, yy), dict(item=item["id"], bands={}, files=[]))
                    record["bands"][band] = dict(native=pin(native), valid=pin(mask),
                                                grid=list(transform)[:6], shape=list(data.shape),
                                                crs=3832, nodata=spec["nodata"],
                                                source_window=list(window.flatten()),
                                                source_href=item["assets"][band]["href"])
                    record["files"].extend([pin(native), pin(mask)])
                    if resume:
                        from copy import deepcopy
                        saved = deepcopy(record['bands'][band])
                        for kind in ['native', 'valid']:
                            saved[kind]['path'] = Path(saved[kind]['path']).name
                        write_json(receipt, saved)
            print("NATIVE", item["id"], band, len(coordinates), flush=True)
        records.extend(indexed.values())
    if not records:
        raise ValueError("No annual sources intersect the full context")
    coverage = box(*bounds).difference(unary_union([
        box(*rasterio.transform.array_bounds(*r["bands"]["red"]["shape"], Affine(*r["bands"]["red"]["grid"])))
        for r in records]))
    # This checks native roster footprint, not availability of observations.
    missing = []
    for row in plan.get('missing_source_items', []):
        if row.get('status') != 404 or not row.get('id', '').startswith('dep_s2_geomad_'):
            continue
        col, r = map(int, row['id'].split('_')[3:5])
        missing.append(box(-3000000+col*96000, -4000000+r*96000,
                           -3000000+(col+1)*96000, -4000000+(r+1)*96000))
    if 'tile_id' in plan['grid'] and coverage.area > 1:
        raise ValueError('DEP tile window is missing real GeoMAD context')
    if coverage.area > 1 and (not missing or coverage.difference(unary_union(missing)).area > 1):
        raise ValueError("Native roster misses padded context without an explicit missing-source record")
    vrts = {}
    for band in BANDS:
        for valid in [False, True]:
            key = band + ("-valid" if valid else "")
            path = folder / (key+".vrt")
            mosaic_vrt(path, records, band, valid)  # Rejects any overlap.
            vrts[key] = path
    return records, vrts


def static_owner(geometry, bounds, crs, sources, route, native_folder):
    """Accepted bounded WorldCover nearest preparation and exact physical weights."""
    from tasks.inputs.geometry import worldcover_tiles
    from carbon.weights import area_weights
    from pyproj import Transformer
    from shapely.ops import transform

    grid = Affine(10, 0, bounds[0], 0, -10, bounds[3])
    weights = area_weights(geometry, grid, crs=crs)
    cover = np.zeros((500, 500), np.uint8)
    geographic = transform(Transformer.from_crs(crs, 4326, always_xy=True).transform,
                          box(*bounds).segmentize(240))
    references = []
    for tile in worldcover_tiles(geographic):
        source = next((s for s in sources if s["tile"] == tile), None)
        if source is None or source["status"] == 404:
            references.append(dict(tile=tile, status="unknown_missing_static_object"))
            continue
        with rasterio.open(route[source["url"]]) as src:
            if src.crs.to_epsg() != 4326 or src.dtypes != ("uint8",) or src.nodata != 0:
                raise ValueError("WorldCover native metadata differs")
            west = int(tile[4:]) * (-1 if tile[3] == "W" else 1)
            south = int(tile[1:3]) * (-1 if tile[0] == "S" else 1)
            intersection = geographic.intersection(box(west, south, west+3, south+3))
            if intersection.area <= 0:
                continue
            win = from_bounds(*intersection.bounds, src.transform)
            left, top = max(0, math.floor(win.col_off)-1), max(0, math.floor(win.row_off)-1)
            win = Window(left, top, min(src.width, math.ceil(win.col_off+win.width)+1)-left,
                         min(src.height, math.ceil(win.row_off+win.height)+1)-top)
            if win.width*win.height > 1024**2:
                raise ValueError("Bounded WorldCover source window exceeded")
            data = src.read(1, window=win)
            valid = src.read_masks(1, window=win)
            affine = src.window_transform(win)
            dst = np.zeros_like(cover)
            reproject(data, dst, src_transform=affine, src_crs=src.crs,
                      dst_transform=grid, dst_crs=crs, src_nodata=0, dst_nodata=0,
                      resampling=Resampling.nearest)
        target = native_folder / (tile+".tif")
        target.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(target, "w", driver="GTiff", width=data.shape[1], height=data.shape[0],
                           count=1, dtype="uint8", crs="EPSG:4326", transform=affine,
                           nodata=0, compress="deflate") as ds:
            ds.write(data, 1)
        with rasterio.open(target) as ds:
            if not np.array_equal(ds.read(1), data) or not np.array_equal(ds.read_masks(1), valid):
                raise ValueError("Saved WorldCover native data/masks differ")
        take = (weights > 0) & (dst != 0)
        cover[take] = dst[take]
        references.append(dict(tile=tile, native=pin(target), window=list(win.flatten())))
    cover[weights == 0] = 0
    if not np.isin(cover, [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]).all():
        raise ValueError("Unexpected WorldCover class")
    return dict(projected=cover, physical_area_ha=weights, support=weights > 0), references
