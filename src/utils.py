"""Plan DEP tiles, load imagery and validate tile inputs."""

import configparser
from datetime import datetime, timezone
import gc
from itertools import product
import logging
import math
from pathlib import Path
import re
from urllib.parse import urlsplit

from affine import Affine
import numpy as np
from odc.geo.xr import wrap_xr
import rasterio
from rasterio.enums import MaskFlags
from rasterio.errors import RasterioIOError
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window, from_bounds
import requests
import rioxarray  # Register the .rio accessor for GeoMAD NoData checks.
from shapely.geometry import box
import xarray as xr


# Read GeoMAD beyond the tile boundary so the height model has context at the edges.
MODEL_BUFFER = 224
STATUS_ASSUMED_ZERO = 7
STATUS_NODATA = 255
SUPPORTED_VERSION = '0.4.0'
GEOMAD_REFLECTANCE_SCALE = 0.0001
GEOMAD_NODATA = {'red': 0, 'green': 0, 'blue': 0, 'nir': 0, 'count': 9999}
DEP_STAC_URL = 'https://stac.digitalearthpacific.org'
GEOMAD_COLLECTION = 'dep_s2_geomad'
GEOMAD_ITEMS_URL = f'{DEP_STAC_URL}/collections/{GEOMAD_COLLECTION}/items'
GEOMAD_ASSET_PREFIX = (
    f"https://dep-public-data.s3.us-west-2.amazonaws.com/{GEOMAD_COLLECTION}/{SUPPORTED_VERSION.replace('.', '-')}/"
)
WORLDCOVER_ROOT = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021'
WORLDCOVER_INDEX = f'{WORLDCOVER_ROOT}/esa_worldcover_grid.geojson'
WORLDCOVER = f'{WORLDCOVER_ROOT}/map/ESA_WorldCover_10m_2021_v200_{{}}_Map.tif'

WORLDCOVER_LAND_CLASSES = (10, 20, 30, 40, 50, 60, 70, 90, 95, 100)
WORLDCOVER_WATER_CLASS = 80
WORLDCOVER_UNKNOWN_CLASS = 0

NDVI_CUTOFF = np.float32(0.3)
# Asner et al. (2016), https://doi.org/10.1186/s13021-015-0043-4
# The equation was calibrated on Hawaiʻi Island using top-of-canopy height in metres.
# AGC uses modelled height as a proxy and reports tonnes of carbon per hectare.
HEIGHT_TO_CARBON_COEFFICIENT = 3.744
HEIGHT_TO_CARBON_EXPONENT = 1.391

MODEL_WEIGHTS = 'Delta1_model_traced.pth'


def parse_years(value: str) -> range:
    """Return the years to process from a single year or an inclusive range."""
    match = re.fullmatch(r'([0-9]+)(?:-([0-9]+))?', value)
    if match is None:
        raise ValueError('Expected a year or an inclusive range such as 2024-2025')
    start = int(match[1])
    end = int(match[2] or match[1])
    if not (1 <= start <= 9999 and 1 <= end <= 9999):
        raise ValueError('Observation year must be an integer calendar year (1–9999)')
    if end < start:
        raise ValueError('Year range must start with the earlier year')
    return range(start, end + 1)


def check_geomad_sources(items, year):
    """Check the source metadata and TIFFs before ODC reads the buffered tile."""
    annual_start = datetime(year, 1, 1, tzinfo=timezone.utc)
    annual_end = datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
    for item in items:
        properties = item.properties
        start = datetime.fromisoformat(properties['start_datetime'])
        end = datetime.fromisoformat(properties['end_datetime'])
        if start != annual_start or end.replace(microsecond=0) != annual_end:
            raise ValueError(f'Expected the full {year} annual interval for GeoMAD Item {item.id}')
        epsg, code = properties.get('proj:epsg'), properties.get('proj:code')
        shape = tuple(properties['proj:shape'])
        transform = Affine(*properties['proj:transform'][:6])
        if ((epsg, code) not in ((3832, None), (None, 'EPSG:3832'), (3832, 'EPSG:3832'))
                or shape != (9600, 9600) or transform[:2] != (10, 0) or transform[3:5] != (0, -10)):
            raise ValueError(f'Unexpected GeoMAD grid metadata for {item.id}')

        for band, nodata in GEOMAD_NODATA.items():
            asset = item.assets.get(band)
            if asset is None:
                raise ValueError(f'Missing required {band} asset for GeoMAD Item {item.id}')
            asset_years = re.findall(r'(?<![0-9])([0-9]{4})(?![0-9])', urlsplit(asset.href).path)
            if not asset.href.startswith(GEOMAD_ASSET_PREFIX) or set(asset_years) != {f'{year:04d}'}:
                raise ValueError(f'Unexpected GeoMAD asset year or version for {item.id}/{band}')
            # AGC uses fixed band scaling, so check the scale and offset before loading.
            spec = asset.extra_fields['raster:bands'][0]
            expected = (1 if band == 'count' else GEOMAD_REFLECTANCE_SCALE, 0, nodata)
            if (spec['scale'], spec['offset'], spec['nodata']) != expected:
                raise ValueError(f'Unexpected GeoMAD scale, offset or NoData for {item.id}/{band}')
            # The loader can read masked pixels as valid. Check the TIFF mask before loading.
            with rasterio.open(asset.href) as src:
                if (src.crs is None or src.crs.to_epsg() != 3832
                        or src.shape != shape or src.transform != transform
                        or src.dtypes != ('uint16',) or src.nodata != nodata
                        or src.mask_flag_enums[0] != [MaskFlags.nodata]):
                    raise ValueError(f'Unexpected GeoMAD TIFF grid, encoding or mask for {item.id}/{band}')


def prepare_worldcover(tile):
    """Return WorldCover classes and coverage on the output tile grid."""
    geographic = tile.extent.to_crs('EPSG:4326', resolution=240, wrapdateline=True).geom
    with requests.get(WORLDCOVER_INDEX, timeout=(10, 60)) as response:
        response.raise_for_status()
        indexed_tiles = {feature['properties']['ll_tile'] for feature in response.json()['features']}
    sources = [WORLDCOVER.format(name) for name in worldcover_tiles(geographic) if name in indexed_tiles]
    classes, available = align_worldcover(tile, geographic, sources)
    allowed_classes = (*WORLDCOVER_LAND_CLASSES, WORLDCOVER_WATER_CLASS, WORLDCOVER_UNKNOWN_CLASS)
    if not np.isin(classes[available], allowed_classes).all():
        raise ValueError('WorldCover has invalid classes')
    return classes, available


def worldcover_tiles(geographic):
    west, south, east, north = geographic.bounds
    return [f'{"S" if lat < 0 else "N"}{abs(lat):02d}{"W" if lon < 0 else "E"}{abs(lon):03d}'
            for lon in range(math.floor(west/3)*3, math.ceil(east/3)*3, 3)
            for lat in range(math.floor(south/3)*3, math.ceil(north/3)*3, 3)
            if geographic.intersection(box(lon, lat, lon+3, lat+3)).area > 0]


def align_worldcover(tile, geographic, sources):
    """Align WorldCover classes and coverage on the retained grid."""
    # Use 256 for missing pixels because the source stores values from 0 to 255.
    missing = 256
    cover = np.zeros(tile.shape, np.uint8)
    available = np.zeros(tile.shape, bool)
    if not sources:
        return cover, available
    aligned = np.empty(tile.shape, np.uint16)
    for url in sources:
        try:
            with rasterio.open(url) as src:
                if src.crs is None or src.crs.to_epsg() != 4326 or src.dtypes != ('uint8',) or src.nodata != 0:
                    raise ValueError(f'WorldCover native grid/type/nodata differs for {url}')
                region = geographic.intersection(box(*src.bounds))
                if region.area <= 0:
                    continue
                # Read one extra pixel at each edge so alignment has the neighbours it needs.
                read = from_bounds(*region.bounds, src.transform)
                left, top = max(0, math.floor(read.col_off)-1), max(0, math.floor(read.row_off)-1)
                right = min(src.width, math.ceil(read.col_off+read.width)+1)
                bottom = min(src.height, math.ceil(read.row_off+read.height)+1)
                window = Window(left, top, right-left, bottom-top)
                raw = src.read(1, window=window, out_dtype='uint16')
                # Keep class 0 where the source covers the pixel.
                if src.mask_flag_enums[0] != [MaskFlags.nodata]:
                    raw[src.read_masks(1, window=window) == 0] = missing
                reproject(raw, aligned, src_transform=src.window_transform(window), src_crs=src.crs,
                          dst_transform=tile.transform, dst_crs=str(tile.crs), dst_nodata=missing,
                          resampling=Resampling.nearest)
        except RasterioIOError as exc:
            raise ValueError(f'Could not read indexed WorldCover TIFF {url}: {exc}') from exc
        del raw
        use = aligned != missing
        if np.any(available & use & (cover != aligned)):
            raise ValueError('Overlapping WorldCover sources disagree on the tile grid')
        np.copyto(cover, aligned, where=use, casting='unsafe')
        available |= use
    return cover, available


def process_tile(geomad, worldcover, tile, buffered_tile, *, model_dir):
    """Run height inference and return the carbon and status layers."""
    try:
        arrays, rgb = prepare(geomad, buffered_tile)
    finally:
        geomad.close()
    # The caller holds the dataset through this call. Release its bands before inference.
    for band in GEOMAD_NODATA:
        del geomad[band]
    # The loader uses rioxarray accessors, which can retain arrays in reference cycles.
    gc.collect()
    height = infer_height(model_dir, rgb)
    del rgb
    validate_height(height, buffered_tile, arrays['rgb_valid'])
    return tile_products(tile, buffered_tile, arrays, height, worldcover)


def prepare(geomad, buffered_tile):
    if geomad.sizes.get('time') != 1:
        raise ValueError('Expected one annual GeoMAD observation')
    # ODC can prefer newer STAC band metadata over the raster:bands values checked above.
    for band, nodata in GEOMAD_NODATA.items():
        if geomad[band].rio.nodata != nodata:
            raise ValueError(f'Loaded GeoMAD NoData differs for {band}')

    height, width = buffered_tile.shape
    dims = ('time', *buffered_tile.dimensions)
    if geomad.odc.geobox != buffered_tile or any(
        geomad[name].dims != dims or geomad[name].dtype != np.uint16 for name in GEOMAD_NODATA
    ):
        raise ValueError("Expected uint16 GeoMAD bands on the buffered grid")
    bands = [geomad[name].isel(time=0).data for name in GEOMAD_NODATA]
    rgb = np.empty((3, height, width), np.uint8)
    rgb_valid = np.empty((height, width), bool)
    ndvi = np.empty((height, width), np.float32)
    ndvi_valid = np.empty((height, width), bool)
    # Prepare RGB and NDVI in small blocks to limit temporary arrays.
    block_size = 256
    for row, col in product(range(0, height, block_size), range(0, width, block_size)):
        placement = (slice(row, row + block_size), slice(col, col + block_size))
        raw = np.stack([np.asarray(band[placement]) for band in bands])
        valid = np.stack([band != nodata for band, nodata in zip(raw, GEOMAD_NODATA.values())])
        supported = valid[:3].all(axis=0) & valid[4] & (raw[4] > 0)
        rgb[:, *placement] = encode(raw[:3], supported)
        rgb_valid[placement] = supported
        ndvi[placement], ndvi_valid[placement] = calculate_ndvi(
            raw[0], raw[3], valid[0], valid[3],
        )
    return dict(rgb_valid=rgb_valid, ndvi=ndvi, ndvi_valid=ndvi_valid), rgb


def encode(raw, valid):
    physical = raw.astype(np.float64) * GEOMAD_REFLECTANCE_SCALE
    rgb = np.clip(np.floor(physical * (255 / .25)), 1, 255).astype(np.uint8)
    rgb[:, ~valid] = 0
    return rgb


def calculate_ndvi(red, nir, red_valid, nir_valid):
    source = red_valid & nir_valid
    r = red.astype(np.float64) * GEOMAD_REFLECTANCE_SCALE
    n = nir.astype(np.float64) * GEOMAD_REFLECTANCE_SCALE
    finite = np.isfinite(r) & np.isfinite(n)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        denominator = n + r
        # Numerical cancellation only, not an empirical dark-pixel/noise threshold.
        epsilon = 8 * np.finfo(np.float64).eps * np.maximum(1, np.abs(n) + np.abs(r))
        bad_denominator = ~np.isfinite(denominator) | (np.abs(denominator) <= epsilon)
        supported = source & finite & ~bad_denominator
        ratio = np.full(red.shape, np.nan, np.float64)
        np.divide(n-r, denominator, out=ratio, where=supported)
        stored = ratio.astype(np.float32)
    valid = supported & np.isfinite(stored)
    stored[~valid] = -9999
    return stored, valid


def infer_height(model_dir, rgb):
    """Predict float32 heights on the buffered RGB grid."""
    # Patch stitching follows DEP Veg. The original inference reference is Geoneon's
    # segmentation-inference at revision d0875749fd42a6811c74271d88ced4e5dbf28af7.
    from lightning import Fabric
    import torch
    from torch.utils.data import DataLoader, Dataset
    from torchvision.transforms import Normalize, ToTensor
    from tqdm import tqdm

    class PatchDataset(Dataset):
        def __init__(self, rgb, size, stride):
            self.rgb = rgb
            self.size = size
            self.stride = stride
            height, width = rgb.shape[-2:]
            if not 0 < stride < size <= min(height, width) or (size - stride) % 2:
                raise ValueError('RGB dimensions, patch size or stride do not support centred stitching')
            rows = [*range(0, height - size, stride), height - size]
            cols = [*range(0, width - size, stride), width - size]
            self.to_tensor = ToTensor()
            self.normalize = Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            # An empty patch has no valid RGB in any part of its output.
            self.origins = [
                (row, col) for row, col in product(rows, cols)
                if rgb[:, row:row + size, col:col + size].any()
            ]

        def __len__(self):
            return len(self.origins)

        def __getitem__(self, index):
            row, col = self.origins[index]
            patch = self.rgb[:, row:row + self.size, col:col + self.size].transpose(1, 2, 0)
            return self.normalize(self.to_tensor(patch)), (row, col)

    training = configparser.ConfigParser(inline_comment_prefixes='#')
    training.read(model_dir/'training_config.cfg')
    channels = training.getint('general', 'n_classes')
    if channels not in (1, 2):
        raise ValueError(f'Expected one height channel or tree and height channels, got {channels}')
    patch_size = training.getint('data', 's')

    fabric = Fabric(accelerator='gpu', devices=[0], strategy='auto', precision='32-true')
    fabric.launch()
    logging.getLogger(__name__).info('Running height inference on %s', fabric.device)
    model = fabric.setup(torch.jit.load(model_dir/MODEL_WEIGHTS))

    patches = PatchDataset(rgb, patch_size, stride=64)
    batches = DataLoader(patches, batch_size=1, num_workers=0)
    batches = fabric.setup_dataloaders(batches)
    height, width = rgb.shape[-2:]
    result = torch.zeros((height, width), dtype=torch.float32)
    margin = (patch_size - patches.stride) // 2
    centre_end = margin + patches.stride
    model.eval()
    with torch.no_grad():
        for inputs, (rows, cols) in tqdm(batches, desc='Height patches'):
            values = (model(inputs.to(fabric.device)).cpu()[:, channels - 1] * 60).clamp(min=1) - 1
            for patch, row, col in zip(values, rows.tolist(), cols.tolist(), strict=True):
                # Stitch patch centres to favour predictions with surrounding context.
                # Keep the outer margins to fill the buffered grid.
                top = 0 if row == 0 else margin
                left = 0 if col == 0 else margin
                bottom = patch_size if row + patch_size == height else centre_end
                right = patch_size if col + patch_size == width else centre_end
                result[row + top:row + bottom, col + left:col + right] = patch[top:bottom, left:right]

    # Prepared RGB encodes unavailable pixels as zero in all three channels.
    height_values = result.numpy()
    height_values[~rgb.any(axis=0)] = 0
    return height_values


def validate_height(height, buffered_tile, rgb_valid):
    if height.shape != tuple(buffered_tile.shape) or height.dtype != np.float32:
        raise ValueError("Inference output grid/type differs")
    if rgb_valid.shape != height.shape or rgb_valid.dtype != bool:
        raise ValueError("RGB validity mask differs from the inference grid")
    for start in range(0, height.shape[0], 256):
        rows = slice(start, start + 256)
        block = height[rows]
        if np.any(rgb_valid[rows] & (~np.isfinite(block) | (block < 0))):
            raise ValueError("Nonfinite or negative runtime height at source-valid pixels")


def tile_products(tile, buffered_tile, arrays, raw_height, worldcover):
    """Build the two output layers using heights validated against the RGB mask."""
    cover, available = worldcover
    core = buffered_tile.overlap_roi(tile)
    height = raw_height[core]
    height_valid = arrays["rgb_valid"][core]
    ndvi_valid = arrays["ndvi_valid"][core]
    land = available & np.isin(cover, WORLDCOVER_LAND_CLASSES)
    unknown = available & (cover == WORLDCOVER_UNKNOWN_CLASS)
    _, status, carbon = carbon_and_status(
        land, height, height_valid, arrays["ndvi"][core], ndvi_valid, unknown,
    )
    status[~available] = STATUS_NODATA
    return xr.Dataset({
        'carbon': wrap_xr(carbon, tile, nodata=np.nan),
        'status': wrap_xr(status, tile, nodata=STATUS_NODATA),
    })


def carbon_and_status(land, height, height_valid, ndvi, ndvi_valid, unknown):
    """Calculate carbon and status from heights validated against the RGB mask."""
    masks = (land, height_valid, ndvi_valid, unknown)
    if any(mask.shape != height.shape or mask.dtype != bool for mask in masks) or ndvi.shape != height.shape:
        raise ValueError('Product support grids/types differ')
    eligible = land & height_valid & ndvi_valid
    modelled = eligible & (ndvi >= NDVI_CUTOFF)
    # Assume zero carbon below the cutoff where the inputs support an estimate.
    # Missing inputs leave carbon masked as NoData.
    assumed_zero = eligible & (ndvi < NDVI_CUTOFF)

    status = np.zeros(land.shape, np.uint8)  # Excluded water within the status mask.
    status[land] = 1  # Unavailable imagery, NDVI or height.
    status[modelled] = 4
    status[unknown] = 5
    status[assumed_zero] = STATUS_ASSUMED_ZERO

    density = np.full(land.shape, np.nan, np.float32)
    with np.errstate(over='ignore'):
        modelled_density = (
            HEIGHT_TO_CARBON_COEFFICIENT * height[modelled].astype(np.float64)**HEIGHT_TO_CARBON_EXPONENT
        ).astype(np.float32)
    if not np.isfinite(modelled_density).all():
        raise ValueError('Modelled carbon density exceeds the float32 range')
    density[modelled] = modelled_density
    density[assumed_zero] = 0
    return modelled | assumed_zero, status, density
