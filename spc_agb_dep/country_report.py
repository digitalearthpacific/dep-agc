"""Read tile density and report stock over exact country intersections."""

from itertools import combinations
import math

import numpy as np
import rasterio
from rasterio.windows import bounds
from shapely.geometry import box

from spc_agb_inputs.weights import area_weights
from spc_agb_worker.accounting import weighted_stock


def report_countries(carbon_path, status_path, countries, *, pilot_footprint=None):
    """Report polygons in the tile CRS. Coverage uses their full in-scope ground area."""
    with rasterio.open(carbon_path) as carbon, rasterio.open(status_path) as status:
        grid = carbon.transform
        crs = carbon.crs
        if (crs is None or crs.to_epsg() != 3832
                or (grid.a, grid.b, grid.d, grid.e) != (10, 0, 0, -10)
                or carbon.count != 1 or carbon.dtypes != ('float32',)
                or status.count != 1 or status.dtypes != ('uint8',)
                or (crs, grid, carbon.shape) != (status.crs, status.transform, status.shape)):
            raise ValueError('Carbon and status must share a native Pacific 10 m grid')
        tile = box(*carbon.bounds)
        geometries = list(countries.values())
        if pilot_footprint is not None:
            geometries.append(pilot_footprint)
        if any(not g.is_valid or g.geom_type not in {'Polygon', 'MultiPolygon'}
               or g.area <= 0 for g in geometries):
            raise ValueError('Expected valid polygonal country and pilot footprints in the tile CRS')
        scope = tile if pilot_footprint is None else tile.intersection(pilot_footprint)
        clipped = {key: scope.intersection(geometry) for key, geometry in countries.items()}
        if any(a.intersection(b).area > 0 for a, b in combinations(clipped.values(), 2)):
            raise ValueError('Country footprints overlap with positive area')
        return {
            'scope': 'pilot-footprint totals' if pilot_footprint is not None else 'country totals within tile',
            'coverage_denominator': 'Full WGS84 ground area of the country within the reporting footprint',
            'countries': {key: _report(carbon, status, geometry) for key, geometry in clipped.items()},
        }


def _report(carbon, status, geometry):
    totals = {key: [] for key in ('footprint_ha', 'assessed_ha', 'stock_tC', 'water_ha',
                                  'unknown_ha', 'unavailable_ha', 'unprocessed_ha')}
    pixels = 0
    for _, window in carbon.block_windows(1):
        piece = geometry.intersection(box(*bounds(window, carbon.transform)))
        if piece.area <= 0:
            continue
        weights = area_weights(piece, carbon.window_transform(window),
                               (int(window.height), int(window.width)), crs=str(carbon.crs))
        density = carbon.read(1, window=window)
        assessed = carbon.read_masks(1, window=window) > 0
        classes = status.read(1, window=window)
        processed = status.read_masks(1, window=window) > 0
        if (not np.isin(classes[processed], [0, 1, 2, 4, 5, 7]).all()
                or not np.array_equal(assessed, processed & np.isin(classes, [4, 7]))
                or not np.isfinite(density[assessed]).all() or (density[assessed] < 0).any()
                or (density[processed & (classes == 7)] != 0).any()):
            raise ValueError('Tile density and status disagree')
        totals['footprint_ha'].append(weights.sum())
        totals['assessed_ha'].append(weights[assessed].sum())
        totals['stock_tC'].append(weighted_stock(weights, density, assessed))
        totals['water_ha'].append(weights[processed & (classes == 0)].sum())
        totals['unknown_ha'].append(weights[processed & (classes == 5)].sum())
        totals['unavailable_ha'].append(weights[processed & np.isin(classes, [1, 2])].sum())
        totals['unprocessed_ha'].append(weights[~processed].sum())
        pixels += int(np.count_nonzero(assessed & (weights > 0)))
    result = {key: math.fsum(parts) for key, parts in totals.items()}
    denominator = result['footprint_ha']
    return dict(result, assessed_pixels=pixels,
                coverage_percent=(result['assessed_ha'] / denominator) * 100 if denominator else None)
