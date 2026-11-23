"""Accepted physical-area arithmetic; extracted unchanged except explicit CRS.

Origin: scripts/run_viti_island.py at 4277015, intersecting_cells/area_weights.
The existing normal ellipsoid helpers remain shared, not copied or compiled.
"""
import numpy as np
import rasterio
from shapely.geometry import box
from .physical_area import GEOD, coordinate_transformer, ground_area_ha

def intersecting_cells(geometry, grid, dimensions):
    """Conservative exact candidates, including shallow coastal slivers.

    GDAL ALL_TOUCHED can omit positive subpixel intersections along nearly
    horizontal edges. Candidate discovery must not lose geometry before the
    unchanged ellipsoid intersection calculation. Zero-area touches are safe.
    """
    import shapely
    rows, cols = np.indices(dimensions)
    x = grid.c + cols * grid.a
    y = grid.f + rows * grid.e
    cells = shapely.box(x, y + grid.e, x + grid.a, y)
    shapely.prepare(geometry)
    take = shapely.intersects(geometry, cells)
    return (rows[take], cols[take], cells[take])

def area_weights(geometry, grid, dimensions=(500, 500), *, crs):
    """Same <=5m densified WGS84 horizontal intersection area as accountant.

    Batch only the projection of full cells; retain exact geometry intersections
    at coast/holes. No island-sized array and no constant/projected pixel area.
    """
    import shapely
    h, w = dimensions
    if grid.b or grid.d or grid.a != 10 or (grid.e != -10):
        raise ValueError('Native10m north-up grid required')
    rows, cols, cells = intersecting_cells(geometry, grid, dimensions)
    result = np.zeros((h, w), np.float64)
    x = grid.c + cols * 10
    y = grid.f - rows * 10
    full = shapely.covered_by(cells, geometry)
    project = coordinate_transformer(crs).transform
    offsets = np.array([[10, -10], [10, -5], [10, 0], [5, 0], [0, 0], [0, -5], [0, -10], [5, -10]])
    for start in range(0, len(rows), 4096):
        indices = np.arange(start, min(start + 4096, len(rows)))
        indices = indices[full[indices]]
        if not len(indices):
            continue
        lon, lat = project(x[indices, None] + offsets[:, 0], y[indices, None] + offsets[:, 1])
        result[rows[indices], cols[indices]] = [abs(GEOD.polygon_area_perimeter(a, b)[0]) / 10000 for a, b in zip(lon, lat)]
    for i in np.flatnonzero(~full):
        result[rows[i], cols[i]] = ground_area_ha(geometry.intersection(cells[i]), crs)
    total = ground_area_ha(geometry.intersection(box(*rasterio.transform.array_bounds(h, w, grid))), crs)
    if not np.isclose(result.sum(), total, rtol=1e-08, atol=1e-07):
        raise ValueError('Physical pixel/owner area partition differs')
    return result
