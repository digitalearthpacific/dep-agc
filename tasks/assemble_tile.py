"""Assemble checked native window outputs on one official Pacific tile."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from affine import Affine
from dep_tools.namers import GenericItemPath
from dep_tools.stac_utils import StacCreator, set_derived_from
import numpy as np
import pystac
from pystac.validation.local_validator import LocalValidator
from pyproj import Transformer
import rasterio
from rasterio.windows import Window
from shapely.geometry import box, mapping
from shapely.ops import transform, unary_union
import xarray as xr

from tasks.inputs.geometry import pacific_tile_plan
from tasks.inputs.metadata import GEOMAD_ITEMS_URL
from carbon.grid import retained_transform
from carbon.io import check_file, read_json, sha256, write_json
from carbon.observation import annual_interval
from .outputs import STATUS, UNITS, lossless_cog


def assemble_tile(tile_id, year, workers, validated_source_items, output):
    """Write a tile Item using the GeoMAD Items validated during input loading."""
    grid = pacific_tile_plan(tile_id)
    affine = Affine(*grid['tile_affine'])
    tile_shape = tuple(grid['tile_shape'])
    selected = {w['id']: w for w in grid['windows'] if w['id'] in workers}
    if not workers or set(workers) != set(selected):
        raise ValueError('Expected finished cores of the official tile')
    sizes = {}
    identities = set()
    for key, window in selected.items():
        worker = workers[key]
        record = read_json(worker/'completion.json')
        declared = record['declared_grid']
        h, w = window['retained_shape']
        x, y = window['global_input_window'][:2]
        if (record['status'], record['window'], record['year'], record['source_version'],
                declared['id'], declared['crs'], declared['retained_shape'], retained_transform(declared)) != (
                'complete', key, year, '0.4.0', grid['id'], grid['crs'], [h, w], affine * Affine.translation(x, y)):
            raise ValueError(f'Completion does not match official tile core {key}')
        if set(record['files']) != set(UNITS):
            raise ValueError(f'Unexpected native assets for {key}')
        for name, row in record['files'].items():
            check_file(worker/'native'/f'{name}.tif', row['sha256'])
        sizes[key] = Window(x, y, w, h)
        identities.add(json.dumps({k: record['identities'][k]
                                   for k in ('model', 'source_commit', 'source_sha256')}, sort_keys=True))
    if len(identities) != 1:
        raise ValueError('Mixed model or inference source identities within tile')
    originals = {item['id']: item for item in validated_source_items}
    if not originals or len(originals) != len(validated_source_items):
        raise ValueError('Expected distinct validated GeoMAD source Items')
    sources = [pystac.Item.from_dict(originals[key], href=f'{GEOMAD_ITEMS_URL}/{key}')
               for key in sorted(originals)]
    release = read_json(Path(__file__).resolve().parents[1]/'release.json')
    names = GenericItemPath(sensor='s2', dataset_id=release['stac']['dataset_id'], version=release['version'],
                            time=str(year), prefix=release['stac']['prefix'], full_path_prefix=output)
    tile_index = tuple(map(int, tile_id.split('_')))
    output.mkdir(parents=True, exist_ok=False)
    checks = {}
    with TemporaryDirectory(dir=output.parent) as scratch, rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        for name in UNITS:
            target = Path(scratch)/f'{name}.tif'
            with rasterio.open(next(iter(workers.values()))/'native'/f'{name}.tif') as first:
                profile, colors = first.profile, first.colorinterp
            profile.update(driver='GTiff', height=tile_shape[0], width=tile_shape[1],
                           transform=affine, crs=grid['crs'], tiled=True, blockxsize=512, blockysize=512,
                           compress='deflate', sparse_ok=True)
            fill = 0 if profile['nodata'] is None else profile['nodata']
            valid_pixels = 0
            with rasterio.open(target, 'w', **profile) as dst:
                dst.write_mask(False)
                dst.colorinterp = colors
                for key, placement in sizes.items():
                    w, h = int(placement.width), int(placement.height)
                    native = workers[key]/'native'
                    with rasterio.open(native/f'{name}.tif') as src:
                        if (src.shape != (h, w) or src.transform != dst.window_transform(placement)
                                or src.crs != dst.crs or src.dtypes != dst.dtypes
                                or src.nodatavals != dst.nodatavals or src.colorinterp != colors):
                            raise ValueError(f'Invalid native {name} grid or encoding in {key}')
                        values, valid = src.read(), src.dataset_mask()
                    if name != 'status':
                        with rasterio.open(native/'status.tif') as status:
                            if ((valid > 0) & (status.dataset_mask() == 0)).any():
                                raise ValueError(f'Valid {name} pixels outside processed support in {key}')
                    values[:, valid == 0] = fill
                    dst.write(values, window=placement)
                    dst.write_mask(valid, window=placement)
                    valid_pixels += int(np.count_nonzero(valid))
            cog = Path(names.path(tile_index, name, absolute=True))
            checks[name] = dict(lossless_cog(target, cog), path=cog.relative_to(output).as_posix(),
                                valid_pixels=valid_pixels)
    footprint = unary_union([box(*w['bounds']) for w in selected.values()])
    geographic = transform(Transformer.from_crs(grid['crs'], 4326, always_xy=True).transform,
                           footprint.segmentize(10))
    start, end = annual_interval(year)
    data = xr.Dataset({name: xr.DataArray(0) for name in UNITS})
    data.attrs['stac_properties'] = {
        'start_datetime': start, 'end_datetime': end, 'agc:tile_id': tile_id,
        'agc:scope': 'Processed tile pixels; unprocessed pixels are NoData',
        'agc:window_count': len(workers), 'agc:source_version': '0.4.0',
        'agc:statuses': STATUS,
    }
    data.attrs['stac_geometry'] = mapping(geographic)
    set_derived_from(sources, data)
    item = StacCreator(names, with_raster=True, set_geometry_from_input=True).process(data, tile_index)
    item.id, item.bbox, item.datetime = tile_id, list(geographic.bounds), None
    item.properties.update(datetime=None, start_datetime=start, end_datetime=end)
    item.remove_links('collection')
    item.collection_id = None
    filename = output/f'{tile_id}.stac-item.json'
    item.set_self_href(str(filename))
    item.make_asset_hrefs_relative()
    for name, asset in item.assets.items():
        asset.roles = ['visual'] if name == 'RGB' else ['data']
        asset.extra_fields['agc:sha256'] = checks[name]['sha256']
        for band in asset.extra_fields.get('raster:bands', []):
            band['unit'] = UNITS[name]
    payload = json.loads(json.dumps(item.to_dict(include_self_link=False)))
    LocalValidator().item_validator().validate(payload)
    write_json(filename, payload)
    result = dict(status='verified', tile=tile_id, year=year, item=filename.name, item_sha256=sha256(filename),
                  planned_windows=len(grid['windows']), window_count=len(workers),
                  native_shape=list(tile_shape), cog_checks=checks, source_ids=sorted(originals))
    write_json(output/'tile-verification.json', result)
    return result
