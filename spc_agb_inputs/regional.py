"""Full-boundary spatial plans only: no imagery, owner arrays, weights or models."""

import argparse
from copy import deepcopy
import math
from pathlib import Path
import time

import numpy as np
from pyproj import Transformer
from shapely import box as boxes, intersects, prepare
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union
from shapely.strtree import STRtree

from spc_agb_dep.catalog import country_plan
from spc_agb_dep.job import plan_windows
from spc_agb_worker.grid import validate_grid
from spc_agb_worker.io import check_file, read_json, sha256, write_json
from .geometry import country_boundary, grid_plan, projected_boundary, provider_tiles, accepted_geometry

TOLERANCE_M2 = 1e-3  # Existing vector/grid closure tolerance; not a physical-area change.


def owner_extent(country, anchor):
    ax, ay = anchor
    x0, y0, x1, y1 = country.bounds
    return [ax+math.floor((x0-ax)/5000)*5000, ay+math.floor((y0-ay)/5000)*5000,
            ax+math.ceil((x1-ax)/5000)*5000, ay+math.ceil((y1-ay)/5000)*5000]


def owner_capacity(country, anchor, guard):
    """Conservative intersecting 5km-cell count, without IDs, arrays or weights."""
    ax, ay = anchor
    candidates = set()
    for component in country.geoms:
        x0, y0, x1, y1 = component.bounds
        for row in range(math.floor((ay-y1)/5000), math.ceil((ay-y0)/5000)):
            for col in range(math.floor((x0-ax)/5000), math.ceil((x1-ax)/5000)):
                candidates.add((row, col))
        guard()
        if len(candidates) > 1_000_000:
            raise ValueError('Owner candidate representation limit exceeded')
    indices = np.asarray(sorted(candidates), dtype=np.int64)
    rectangles = boxes(ax+indices[:, 1]*5000, ay-(indices[:, 0]+1)*5000,
                       ax+(indices[:, 1]+1)*5000, ay-indices[:, 0]*5000)
    prepare(country)
    upper = int(intersects(country, rectangles).sum())  # Includes boundary-only touches.
    if upper > 65535:
        raise ValueError('Conservative owner count exceeds uint16; preparation not admitted')
    return upper


def new_plan(country, geographic, rule, guard):
    projected = projected_boundary(geographic, rule['crs'])
    # Same origin as owners()->grid_plan(), using only the snapped owner extent.
    # Do not allocate owner geometries/IDs or country rasters for spatial planning.
    extent = owner_extent(projected, rule['anchor'])
    lattice = grid_plan(projected, [dict(bounds=extent)], rule['anchor'],
                        [g.bounds for g in projected.geoms], guard)
    if len(lattice['windows']) > 50_000:
        raise ValueError('Processing-window representation limit exceeded')
    grid = dict(rule, **lattice, id=country['code'].lower()+'-10m-pacific3832-v2',
                canonical_global_offset=[0, 0])
    return dict(schema='agc-spatial-plan-v1', country=country,
                source_product='GeoMAD', source_version='0.4.0', grid=grid,
                owner_cell_upper_bound=owner_capacity(projected, rule['anchor'], guard),
                ownership='Full country boundary clipped by disjoint retained cells; '
                          'owner IDs/order/arrays and physical weights remain unprepared')


def check_plan(plan, geographic, guard, accepted_projected=None):
    """Check the declared retained partition and every original boundary component."""
    grid = plan['grid']
    windows = plan_windows(plan, plan['country']['code'])
    if max(grid['shape']) > 2**31-1:
        raise ValueError('GDAL per-axis dimension limit exceeded')
    source_components = transform(
        Transformer.from_crs(4326, grid['crs'], always_xy=True).transform, geographic)
    projected = accepted_projected if accepted_projected is not None else source_components
    if not projected.is_valid:
        raise ValueError('Projected boundary invalid; no implicit repair')
    rectangles = []
    for w in windows:
        guard()
        offset = grid.get('canonical_global_offset', [0, 0])
        global_window = w['global_input_window']
        receiving = w.get('receiving_window', [global_window[i]+offset[i] for i in range(2)]+global_window[2:])
        contract = dict(grid, receiving_window=receiving,
                        global_input_window=w['global_input_window'], bounds=w['bounds'],
                        canonical_global_offset=offset,
                        retained_shape=[side - 2 * grid['context'] for side in reversed(receiving[2:])])
        validate_grid(dict(schema='spc-agb-window-v3', country=plan['country'],
                           window=w['id'], grid=contract))
        rectangles.append(box(*w['bounds']))
    union = unary_union(rectangles)
    missing = float(projected.difference(union).area)
    tree = STRtree(rectangles)
    overlap = 0.0
    for i, rectangle in enumerate(rectangles):
        for j in tree.query(rectangle, predicate='intersects'):
            if j > i:
                overlap = max(overlap, float(rectangle.intersection(rectangles[j]).area))
    components = []
    for i, (source, component) in enumerate(zip(geographic.geoms, source_components.geoms)):
        guard()
        hits = [int(j) for j in tree.query(component, predicate='intersects')
                if component.intersection(rectangles[j]).area > 0]
        gap = float(component.difference(union).area)
        components.append(dict(id=i, geographic_bounds=list(source.bounds),
            projected_bounds=list(component.bounds), projected_m2=float(component.area),
            retained_windows=[windows[j]['id'] for j in sorted(hits)], uncovered_m2=gap))
        if not hits or gap > TOLERANCE_M2:
            raise ValueError(f'Boundary component{i} not covered: {gap}m2')
    if missing > TOLERANCE_M2 or overlap > TOLERANCE_M2:
        raise ValueError(f'Retained partition gap/overlap: {missing}/{overlap}m2')
    return dict(passed=True, components=components, window_count=len(windows),
                uncovered_m2=missing, maximum_retained_overlap_m2=overlap,
                tolerance_m2=TOLERANCE_M2, context_overlap_permitted=True,
                raster_inputs_prepared=False, source_pixel_coverage='uninspected')


def run(config, bindings, seed_catalog, root, prefix, seconds=5400, config_pin=None):
    if (config['source_product'], config['source_version'],
        config['boundary_source']['name'], config['boundary_source']['version']) != (
        'GeoMAD', '0.4.0', 'GADM', '4.1'
    ):
        raise ValueError('Pinned GeoMAD0.4.0/GADM4.1 spatial contract required')
    rule = config['new_grid']
    if (rule['crs'], rule['anchor'], rule['resolution'], rule['patch'], rule['stride'],
        rule['context'], rule['retained_chunk']) != ('EPSG:3832', [0, 0], 10, 512, 64, 224, 2048):
        raise ValueError('Declared native-Pacific annual planning grid required')
    codes = [r['code'] for r in config['countries']]
    if len(codes) != 22 or len(set(codes)) != 22:
        raise ValueError('Exactly22 unique engineering-roster codes required')
    relative = Path(prefix)
    if relative.is_absolute() or '..' in relative.parts or '\\' in prefix or ':' in prefix:
        raise ValueError('Relative typed run prefix required')
    # Boundaries are explicitly bound read-only working copies, with originals
    # retained separately. A planning-only SSD root need not contain INPUT/OUTPUT.
    if (not (root/'GEODATA').is_dir()
            or root.resolve().is_relative_to(Path(__file__).resolve().parents[1])):
        raise ValueError('Existing external GEODATA working root required')
    vector = root/'GEODATA/VECTOR'/relative
    report = root/'GEODATA/REPORT'/relative
    vector.mkdir(parents=True, exist_ok=False)
    report.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()

    def guard():
        if time.monotonic()-started > seconds:
            raise TimeoutError('Spatial-planning runtime budget exhausted')

    records, entries, common = [], [], []
    for row in config['countries']:
        guard()
        code = row['code']; country = dict(code=code, name=row['name'])
        print('Planning', code, flush=True)
        pin = bindings['boundaries'][code]
        source = root/pin['path']
        check_file(source, pin['sha256'])
        base = dict(country=country)
        try:
            geographic = country_boundary(source, code)
            if row['reuse_accepted_plan']:
                seed = country_plan(seed_catalog, code, None)
                if seed['root'] != (root/'GEODATA').resolve():
                    raise ValueError('Seed index must share the configured artifact root')
                plan = seed['plan']; entry = deepcopy(seed['entry'])
                plan_file = seed['plan_file']
            else:
                plan = new_plan(country, geographic, config['new_grid'], guard)
                plan['boundary'] = dict(pin, source='GADM', version='4.1',
                                         permission=config['boundary_source']['permission'])
                plan_file = vector/(code+'-spatial.json')
                write_json(plan_file, plan)
                entry = dict(base, plan=dict(path=plan_file.relative_to(root/'GEODATA').as_posix(),
                                             sha256=sha256(plan_file)), jobs=[])
            accepted = accepted_geometry(root, pin['accepted_projected'], plan['grid']['crs']) \
                if row['reuse_accepted_plan'] and pin.get('accepted_projected') else None
            checked = check_plan(plan, geographic, guard, accepted)
            if accepted is not None:
                checked['accepted_projected'] = pin['accepted_projected']
                entry['accepted_projected'] = dict(pin['accepted_projected'], path=(root/pin['accepted_projected']['path']).relative_to(root/'GEODATA').as_posix())
            # Geometry-only source tile candidates; no catalogue or pixel requests.
            projected = transform(Transformer.from_crs(plan['grid']['crs'], 3832,
                always_xy=True).transform, accepted.segmentize(240)) \
                if accepted is not None else projected_boundary(geographic, 'EPSG:3832')
            tiles = provider_tiles(geographic, projected if accepted is not None else None)
            common.append((code, projected))
            checked.update(country=country, plan_sha256=sha256(plan_file),
                           boundary_sha256=pin['sha256'], reused_plan=row['reuse_accepted_plan'],
                           provider_tile_candidates=[list(t) for t in tiles])
            entry['plan_status'] = 'checked'
        except (ValueError, KeyError) as exc:
            checked = dict(passed=False, country=country, error=str(exc), boundary_sha256=pin['sha256'])
            plan_file = vector/(code+'-blocked.json')
            write_json(plan_file, dict(base, planning_status='blocked', grid=None,
                                       reason=str(exc), boundary=pin))
            entry = dict(base, plan_status='blocked', jobs=[],
                         plan=dict(path=plan_file.relative_to(root/'GEODATA').as_posix(),
                                   sha256=sha256(plan_file)))
        check_file_path = report/(code+'-check.json')
        write_json(check_file_path, checked)
        entry['planning_check'] = dict(path=check_file_path.relative_to(root/'GEODATA').as_posix(),
                                       sha256=sha256(check_file_path))
        records.append(checked); entries.append(entry)
        print(code, entry['plan_status'], checked.get('window_count'), flush=True)
    cross = []
    for i, (a, ga) in enumerate(common):
        guard()
        for b, gb in common[i+1:]:
            if ga.intersects(gb):
                area = float(ga.intersection(gb).area)
                cross.append(dict(countries=[a, b], intersection_m2=area))
                if area > TOLERANCE_M2:
                    raise ValueError('Unresolved cross-country boundary overlap: '+str(cross[-1]))
    index = dict(schema='agc-spatial-countries-v1', artifact_root='../../../..', countries=entries)
    write_json(report/'countries.json', index)
    result = dict(records=records, regional_boundary_intersections=cross,
                  seconds=time.monotonic()-started, model_calls=0, imagery_acquisitions=0,
                  config_sha256=config_pin,
                  planner_sha256=sha256(Path(__file__)), source_pixel_availability='uninspected')
    write_json(report/'planning-result.json', result)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['config', 'bindings', 'seed-catalog', 'data-root']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--prefix', required=True)
    p.add_argument('--seconds', type=int, default=5400)
    args = p.parse_args()
    run(read_json(args.config), read_json(args.bindings), args.seed_catalog,
        args.data_root.resolve(), args.prefix, args.seconds, sha256(args.config))
