"""Process annual Pacific tiles and deliver them to S3."""

import json
from math import isnan

import antimeridian
import rasterio
from shapely import orient_polygons
from shapely.geometry import box, mapping
from shapely.ops import unary_union

from src.utils import (
    GEOMAD_COLLECTION, GEOMAD_ITEMS_URL, STATUS_ASSUMED_ZERO, STATUS_NODATA, SUPPORTED_VERSION, check_geomad_sources,
)


UNITS = {'carbon': 't C/ha', 'status': '1'}
STATUS = {'0':'excluded water', '1':'unavailable inputs', '4':'modelled (including genuine zero)',
          '5':'unknown classification', str(STATUS_ASSUMED_ZERO):'NDVI-assumed zero'}


def run_tile(plan, year, *, searcher, loader, worldcover, processor, writer, stac_creator, stac_writer,
             model_uri, model_sha256):
    """Load imagery, run the processor, and deliver one tile for a validated year."""
    tile = plan['tile']
    tile_index = tuple(map(int, plan['tile_id'].split('_')))
    target = plan['buffered_tile']
    # Find this year's GeoMAD imagery for the tile and the area around it.
    items = searcher.search(target)
    sources = tile_sources(plan, year, items)
    # Read the remote rasters without looking for extra files beside them.
    with rasterio.Env(GDAL_PAM_ENABLED='NO', PROJ_NETWORK='OFF', GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR',
                      GDAL_HTTP_CONNECTTIMEOUT='10', GDAL_HTTP_TIMEOUT='60', GDAL_HTTP_MULTIRANGE='NO'):
        check_geomad_sources(sources, year)
        cover = worldcover()
        geomad = loader.load(sources, target)
        # Use GeoMAD and WorldCover to estimate height and produce carbon and status.
        data = processor(geomad, cover, tile, target)
        if set(data.data_vars) != set(UNITS):
            raise ValueError('Expected carbon and status tile layers')
        for name, dtype in (('carbon', 'float32'), ('status', 'uint8')):
            layer = data[name]
            nodata = layer.odc.nodata
            valid_nodata = (nodata is not None and isnan(nodata)) if name == 'carbon' else nodata == STATUS_NODATA
            if (layer.dims != tile.dimensions or layer.odc.geobox != tile
                    or layer.dtype != dtype or not valid_nodata):
                raise ValueError(f'Invalid tile grid or encoding for {name}')
        # Upload both rasters before publishing the STAC Item that links to them.
        if len(writer.write(data, tile_index)) != len(UNITS):
            raise ValueError('Expected one COG for each tile raster')
        item = create_tile_item(data, plan, year, sources, tile_index, stac_creator, model_uri, model_sha256)
        return stac_writer.write(item, tile_index)


def tile_sources(plan, year, items):
    """Keep overlapping GeoMAD Items in search order and require full buffered coverage."""
    if not items:
        raise ValueError(f'No annual GeoMAD Items for tile {plan["tile_id"]} in {year}')
    seen = set()
    context = box(*plan['buffered_tile'].boundingbox.bbox)
    selected = []
    for item in items:
        item_id = item.id
        if item.collection_id != GEOMAD_COLLECTION:
            raise ValueError(f'Unexpected GeoMAD collection for {item_id}')
        if item_id in seen:
            raise ValueError(f'Duplicate annual GeoMAD Item: {item_id}')
        seen.add(item_id)
        bounds = box(*item.properties['proj:bbox'])
        if context.intersection(bounds).area > 0:
            selected.append((item, bounds))
    # The loader can fill a gap with NoData, so check that the imagery covers the full buffer.
    if context.difference(unary_union([bounds for _, bounds in selected])).area > 1:
        raise ValueError('Annual GeoMAD Items do not cover the full buffered tile')
    return [item for item, _ in selected]


def create_tile_item(data, plan, year, source_items, tile_index, stac_creator, model_uri, model_sha256):
    """Create and validate the annual STAC Item for the written COGs."""
    source_ids = sorted(item.id for item in source_items)
    footprint = orient_polygons(plan['tile'].extent.to_crs('EPSG:4326', wrapdateline=True).geom)
    start = f'{year:04d}-01-01T00:00:00Z'
    end = f'{year:04d}-12-31T23:59:59Z'
    data.attrs['stac_properties'] = {
        'start_datetime': start, 'end_datetime': end, 'agc:tile_id': plan['tile_id'],
        'agc:scope': 'Processed tile pixels; unprocessed pixels are NoData',
        'agc:source_version': SUPPORTED_VERSION,
        'agc:model_uri': model_uri,
        'agc:model_sha256': model_sha256,
        'agc:statuses': STATUS,
    }
    data.attrs['stac_geometry'] = mapping(footprint)
    # Link to the official catalogue entry for each GeoMAD source.
    data.attrs['stac_derived_from'] = [f'{GEOMAD_ITEMS_URL}/{item_id}' for item_id in source_ids]
    item = stac_creator.process(data, tile_index)
    # Keep the bounding box narrow when the tile crosses 180° longitude.
    item.bbox = antimeridian.bbox(footprint)
    item.datetime = None
    item.properties.update(datetime=None, start_datetime=start, end_datetime=end)
    item.make_asset_hrefs_relative()
    for name, asset in item.assets.items():
        for band in asset.extra_fields.get('raster:bands', []):
            band['unit'] = UNITS[name]
    json.dumps(item.to_dict(), allow_nan=False)
    item.validate()
    return item
