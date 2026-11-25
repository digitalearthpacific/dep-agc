"""Discover annual DEP sources and read the imagery needed by each tile window."""

from contextlib import contextmanager
import math

from affine import Affine
from dep_tools.exceptions import EmptyCollectionError
from dep_tools.loaders import OdcLoader
from dep_tools.searchers import PystacSearcher
import numpy as np
from odc.geo.geobox import GeoBox
from pystac_client import Client
from pystac_client.stac_api_io import StacApiIO
import rasterio
from rasterio.enums import MaskFlags
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window, from_bounds
from shapely.geometry import box

from carbon.accounting import read_worldcover
from carbon.io import write_json, write_raster
from carbon.observation import SUPPORTED_VERSION, annual_interval
from .canonical import BANDS, canonical_receiving
from .geometry import worldcover_tiles
from .metadata import DEP_STAC_URL, canonical_geomad_item, validate_item
from .raster import check_native_geomad


WORLDCOVER = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_{}_Map.tif'


def discover_geomad(grid, year):
    """Find GeoMAD over the full tile and its outer model context."""
    shape = tuple(size + 2 * grid['context'] for size in grid['tile_shape'])
    area = GeoBox(shape, Affine(*grid['affine']), grid['crs'])
    client = Client.open(DEP_STAC_URL, stac_io=StacApiIO(timeout=(10, 30), max_retries=0))
    searcher = PystacSearcher(client=client, collections=['dep_s2_geomad'],
                              datetime='/'.join(annual_interval(year)))
    try:
        items = searcher.search(area)
    except EmptyCollectionError as exc:
        raise ValueError(f'No annual GeoMAD Items for tile {grid["tile_id"]} in {year}') from exc
    return [validate_item(item.to_dict(), year, SUPPORTED_VERSION) for item in items]


@contextmanager
def live_tile_sources(grid, items, work):
    """Check source headers, align WorldCover and enable direct COG range reads."""
    tile = GeoBox(tuple(grid['tile_shape']), Affine(*grid['tile_affine']), grid['crs'])
    geographic = tile.extent.to_crs('EPSG:4326', resolution=240, wrapdateline=True).geom
    cover_urls = [WORLDCOVER.format(name) for name in worldcover_tiles(geographic)]
    write_json(work/'geomad-items.json', items)
    with rasterio.Env(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_HTTP_CONNECTTIMEOUT='10',
                      GDAL_HTTP_TIMEOUT='60', GDAL_HTTP_MULTIRANGE='NO'):
        for item in items:
            for band in BANDS:
                with rasterio.open(item['assets'][band]['href']) as src:
                    check_native_geomad(src, item, band)
                    if src.mask_flag_enums[0] != [MaskFlags.nodata]:
                        raise ValueError(f'Unsupported explicit source mask for {item["id"]}/{band}')
        path = work/'worldcover.tif'
        align_worldcover(tile, geographic, cover_urls, path)
        cover = read_worldcover(path, {'grid': {'crs': grid['crs'], 'retained_shape': grid['tile_shape']}},
                                tile.transform)
        yield cover


def align_worldcover(tile, geographic, sources, path):
    """Align class values and source availability separately on the retained grid."""
    cover = np.zeros(tile.shape, np.uint8)
    available = np.zeros(tile.shape, bool)
    for url in sources:
        with rasterio.open(url) as src:
            if src.crs is None or src.crs.to_epsg() != 4326 or src.dtypes != ('uint8',) or src.nodata != 0:
                raise ValueError('WorldCover native grid/type/nodata differs')
            region = geographic.intersection(box(*src.bounds))
            if region.area <= 0:
                continue
            read = from_bounds(*region.bounds, src.transform)
            left, top = max(0, math.floor(read.col_off)-1), max(0, math.floor(read.row_off)-1)
            right = min(src.width, math.ceil(read.col_off+read.width)+1)
            bottom = min(src.height, math.ceil(read.row_off+read.height)+1)
            window = Window(left, top, right-left, bottom-top)
            raw = src.read(1, window=window)
            # Source NoData is class 0 (unknown). Preserve a separate explicit mask when present.
            presence = (np.ones(raw.shape, np.uint8) if src.mask_flag_enums[0] == [MaskFlags.nodata]
                        else (src.read_masks(1, window=window) > 0).astype(np.uint8))
            values, valid = np.zeros_like(cover), np.zeros_like(cover)
            for data, destination in ((raw, values), (presence, valid)):
                reproject(data, destination, src_transform=src.window_transform(window), src_crs=src.crs,
                          dst_transform=tile.transform, dst_crs=str(tile.crs), resampling=Resampling.nearest)
        use = valid > 0
        if np.any(available & use & (cover != values)):
            raise ValueError('Overlapping WorldCover sources disagree on the tile grid')
        cover[use] = values[use]
        available |= use
    write_raster(path, cover, available, tile.transform, str(tile.crs))


def load_geomad_window(grid, window, items, package):
    x, y, w, h = window['global_input_window']
    target = GeoBox((h, w), Affine(*grid['affine']) * Affine.translation(x, y), grid['crs'])
    sources = [canonical_geomad_item(item) for item in items]
    loader = OdcLoader(bands=BANDS, groupby='solar_day', resampling='nearest',
                       fail_on_error=True, clip_to_areas=False)
    data = loader.load(sources, target)
    if data.odc.geobox != target or data.sizes.get('time') != 1:
        raise ValueError('Loaded GeoMAD grid or annual observation differs')
    values = np.empty((2*len(BANDS), h, w), np.uint16)
    for i, band in enumerate(BANDS):
        layer = data[band].isel(time=0)
        nodata = 9999 if band == 'count' else 0
        if layer.dtype != np.uint16 or layer.rio.nodata != nodata:
            raise ValueError(f'Loaded GeoMAD dtype or NoData differs for {band}')
        values[2*i] = layer.values
        values[2*i+1] = values[2*i] != nodata
    return canonical_receiving(grid, window, values, package)
