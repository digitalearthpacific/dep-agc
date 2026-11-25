"""Country vectors and a grid derived from an existing project anchor."""

import math
import re
from pathlib import Path
import sqlite3

from affine import Affine
from pyproj import CRS, Transformer
from rasterio.transform import array_bounds
from shapely import from_wkb, get_coordinates
from shapely.geometry import box, mapping, shape
from shapely.ops import transform, unary_union

from carbon.io import read_json
from carbon.grid import DEP_CORE, MODEL_CONTEXT, MODEL_PATCH, MODEL_STRIDE
from .dependencies import dependency_file


def country_boundary(path, code):
    """Read the declared GADM 4.1 ADM0 boundary without implicit repair."""
    with sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True) as db:
        if db.execute("SELECT srs_id FROM gpkg_geometry_columns WHERE table_name='ADM_ADM_0'").fetchone() != (4326,):
            raise ValueError("Expected GADM geographic CRS")
        rows = db.execute("SELECT geom FROM ADM_ADM_0 WHERE GID_0 = ?", (code,)).fetchall()
        if len(rows) != 1:
            raise ValueError("Exactly one declared country ADM0 required")
        blob = rows[0][0]
        if blob[:2] != b"GP":
            raise ValueError("GeoPackage geometry required")
        skip = 8 + {0: 0, 1: 32, 2: 48, 3: 48, 4: 64}[(blob[3] >> 1) & 7]
        geometry = from_wkb(blob[skip:])
    if geometry.geom_type != "MultiPolygon" or not geometry.is_valid:
        raise ValueError("Valid multipart country footprint required; no implicit repair")
    return geometry


def projected_boundary(geometry, crs):
    target = CRS.from_user_input(crs)
    a = target.area_of_use
    west, south, east, north = geometry.bounds
    if a.west <= a.east:
        longitude_ok = a.west <= west <= east <= a.east
    else:
        # EPSG:3832's Pacific area of use crosses180; [-180,180] bounds alone
        # cannot describe it. Check every unchanged source vertex on its circle.
        longitude = (get_coordinates(geometry)[:, 0] - a.west) % 360 + a.west
        longitude_ok = bool((longitude <= a.east + 360).all())
    if not (longitude_ok and a.south <= south <= north <= a.north):
        raise ValueError("Country is outside proposed CRS area of use")
    result = transform(Transformer.from_crs(4326, crs, always_xy=True).transform, geometry)
    if not result.is_valid:
        raise ValueError("Invalid projected country; no repair")
    return result


def owners(country, anchor):
    """Same 5km disjoint owner lattice as sparse_annual.prepare; no raster allocation."""
    ax, ay = anchor
    x0, y0, x1, y1 = country.bounds
    rows = []
    for row in range(math.floor((ay-y1)/5000), math.ceil((ay-y0)/5000)):
        for col in range(math.floor((x0-ax)/5000), math.ceil((x1-ax)/5000)):
            bounds = [ax+col*5000, ay-(row+1)*5000, ax+(col+1)*5000, ay-row*5000]
            clipped = country.intersection(box(*bounds))
            if clipped.area > 0:
                rows.append(dict(id=f"c{col:+05d}_r{row:+05d}", bounds=bounds,
                                 geometry=mapping(clipped)))
    order = sorted(r["id"] for r in rows)
    if len(order) > 65535:
        raise ValueError("Owner IDs exceed uint16")
    for row in rows:
        row["numeric_id"] = order.index(row["id"]) + 1
    return rows, order


def grid_plan(country, owner_rows, anchor, component_bounds=None, guard=None):
    """annual_geomad.geometry_plan lattice, extended to full 2048 retained windows.

    Retains existing anchor minus224px and64px alignment. This is vector/grid
    planning only, not preparation of country-wide rasters.
    """
    bounds = unary_union([box(*r["bounds"]) for r in owner_rows]).bounds
    ax, ay = anchor
    left = ax-2240+math.floor((bounds[0]-ax)/640)*640
    top = ay+2240-math.floor((ay-bounds[3])/640)*640
    reference = Affine(10, 0, left, 0, -10, top)
    windows = []
    nrows = math.ceil((top-2240-bounds[1])/20480)
    ncols = math.ceil((bounds[2]-left-2240)/20480)
    if component_bounds is None:
        candidates = ((row, col) for row in range(nrows) for col in range(ncols))
    else:
        # Same lattice/order; omit only cells outside every component envelope.
        cells = set()
        for x0, y0, x1, y1 in component_bounds:
            for row in range(max(0, math.floor((top-2240-y1)/20480)),
                             min(nrows, math.ceil((top-2240-y0)/20480))):
                for col in range(max(0, math.floor((x0-left-2240)/20480)),
                                 min(ncols, math.ceil((x1-left-2240)/20480))):
                    cells.add((row, col))
        candidates = sorted(cells)
    for number, (row, col) in enumerate(candidates):
        if guard is not None and number % 64 == 0:
            guard()
        x, y = left+2240+col*20480, top-2240-row*20480
        b = [x, y-20480, x+20480, y]
        intersection = country.intersection(box(*b))
        if intersection.area > 0:
            windows.append(dict(id=f"r{row:+04d}-c{col:+04d}", row=row, col=col,
                                bounds=b, input_bounds=[x-2240, y-22720, x+22720, y+2240],
                                global_input_window=[col*2048, row*2048, 2496, 2496]))
    width = math.ceil(max(w["global_input_window"][0]+2496 for w in windows)/512)*512
    height = math.ceil(max(w["global_input_window"][1]+2496 for w in windows)/512)*512
    return dict(affine=list(reference)[:6], shape=[height, width], windows=windows)


def pacific_tile_plan(tile_id):
    """Plan all 25 equal cores before filtering by any accounting footprint."""
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
            windows.append(dict(id=f'r{row:02d}c{col:02d}', row=row, col=col,
                bounds=list(array_bounds(DEP_CORE, DEP_CORE, tile.transform * Affine.translation(x, y))),
                input_bounds=list(array_bounds(size, size, receiving * Affine.translation(x, y))),
                global_input_window=[x, y, size, size], retained_shape=[DEP_CORE, DEP_CORE]))
    # Only canonical storage is block padded. The worker reads the exact receiving window.
    padded = math.ceil((width + 2 * MODEL_CONTEXT) / MODEL_PATCH) * MODEL_PATCH
    return dict(id=f'PACIFIC_GRID_10/{tile_id}', tile_id=tile_id, crs='EPSG:3832',
        tile_affine=list(tile.transform)[:6], tile_shape=[width, width], tile_bounds=list(tile.boundingbox.bbox),
        context=MODEL_CONTEXT, patch=MODEL_PATCH, stride=MODEL_STRIDE,
        affine=list(receiving)[:6], shape=[padded, padded], windows=windows)


def provider_tiles(geographic, projected=None):
    geometry = projected if projected is not None else transform(
        Transformer.from_crs(4326, 3832, always_xy=True).transform, geographic.segmentize(.002))
    x0, y0, x1, y1 = geometry.bounds
    return sorted((col, row)
                  for col in range(math.floor((x0+3000000)/96000), math.ceil((x1+3000000)/96000))
                  for row in range(math.floor((y0+4000000)/96000), math.ceil((y1+4000000)/96000))
                  if geometry.intersection(box(-3000000+col*96000, -4000000+row*96000,
                                               -3000000+(col+1)*96000, -4000000+(row+1)*96000)).area > 0)


def worldcover_tiles(geographic):
    x0, y0, x1, y1 = geographic.bounds
    return [f'{"S" if lat < 0 else "N"}{abs(lat):02d}{"W" if lon < 0 else "E"}{abs(lon):03d}'
            for lon in range(math.floor(x0/3)*3, math.ceil(x1/3)*3, 3)
            for lat in range(math.floor(y0/3)*3, math.ceil(y1/3)*3, 3)
            if geographic.intersection(box(lon, lat, lon+3, lat+3)).area > 0]


def accepted_geometry(root, pin, crs):
    """Read the approved projected geometry verbatim; no union/repair/reprojection."""
    path = dependency_file(root, pin)
    obj = read_json(path)
    if obj['crs']['properties']['name'] != crs or len(obj['features']) != 1:
        raise ValueError('Accepted projected boundary CRS/feature contract differs')
    geometry = shape(obj['features'][0]['geometry'])
    if not geometry.is_valid or geometry.geom_type not in ['Polygon', 'MultiPolygon']:
        raise ValueError('Accepted projected boundary must be valid polygon geometry')
    return geometry


def preparation_geometry(geographic, crs, root, approved=None):
    """An explicit approved binding takes precedence; never fall back on failure."""
    return accepted_geometry(root, approved, crs) if approved else projected_boundary(geographic, crs)
