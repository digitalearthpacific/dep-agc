"""Official Pacific tile windows and ESA WorldCover tile names."""

import math
import re

from affine import Affine
from rasterio.transform import array_bounds
from shapely.geometry import box

from carbon.grid import DEP_CORE, MODEL_CONTEXT, MODEL_PATCH, MODEL_STRIDE


def pacific_tile_plan(tile_id):
    """Plan all 25 equal cores on the official Pacific tile grid."""
    from dep_tools.grids import PACIFIC_GRID_10

    if not isinstance(tile_id, str) or not re.fullmatch(r'\d{3}_\d{3}', tile_id):
        raise ValueError('Expected an official DEP tile ID such as 075_025')
    tile = PACIFIC_GRID_10[tuple(map(int, tile_id.split('_')))]
    width = 9600
    per_axis, remainder = divmod(width, DEP_CORE)
    if tile.crs.epsg != 3832 or tuple(tile.shape) != (width, width) or remainder:
        raise ValueError('Unexpected PACIFIC_GRID_10 tile geometry')
    receiving = tile.transform * Affine.translation(-MODEL_CONTEXT, -MODEL_CONTEXT)
    size = DEP_CORE + 2 * MODEL_CONTEXT
    windows = []
    for row in range(per_axis):
        for col in range(per_axis):
            x, y = col * DEP_CORE, row * DEP_CORE
            windows.append(dict(id=f'r{row:02d}c{col:02d}',
                bounds=list(array_bounds(DEP_CORE, DEP_CORE, tile.transform * Affine.translation(x, y))),
                input_bounds=list(array_bounds(size, size, receiving * Affine.translation(x, y))),
                global_input_window=[x, y, size, size], retained_shape=[DEP_CORE, DEP_CORE]))
    # Only canonical storage is block padded. The worker reads the exact receiving window.
    padded = math.ceil((width + 2 * MODEL_CONTEXT) / MODEL_PATCH) * MODEL_PATCH
    return dict(id=f'PACIFIC_GRID_10/{tile_id}', tile_id=tile_id, crs='EPSG:3832',
        tile_affine=list(tile.transform)[:6], tile_shape=[width, width],
        context=MODEL_CONTEXT, patch=MODEL_PATCH, stride=MODEL_STRIDE,
        affine=list(receiving)[:6], shape=[padded, padded], windows=windows)


def worldcover_tiles(geographic):
    x0, y0, x1, y1 = geographic.bounds
    return [f'{"S" if lat < 0 else "N"}{abs(lat):02d}{"W" if lon < 0 else "E"}{abs(lon):03d}'
            for lon in range(math.floor(x0/3)*3, math.ceil(x1/3)*3, 3)
            for lat in range(math.floor(y0/3)*3, math.ceil(y1/3)*3, 3)
            if geographic.intersection(box(lon, lat, lon+3, lat+3)).area > 0]
