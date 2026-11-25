"""Prepare annual inputs, process official tile windows and assemble the tile."""

from contextlib import nullcontext
from functools import partial
from pathlib import Path
import sys

from affine import Affine
import numpy as np
import rasterio
from shapely.geometry import box
from shapely.ops import unary_union

from carbon.accounting import read_worldcover, tile_products
from carbon.grid import retained_transform
from carbon.io import check_file, local_input, read_json, sha256, validate_inputs, write_json, write_raster
from carbon.observation import SUPPORTED_VERSION, validate_observation
from carbon.prepare import prepare, read_raw_height
from carbon.science import crop_height
from .assemble_tile import assemble_tile
from .inputs.canonical import BANDS, canonical
from .inputs.geometry import pacific_tile_plan
from .inputs.live_tile import discover_geomad, live_tile_sources, load_geomad_window
from .inputs.metadata import validate_item
from .inputs.prepare_country import make_tile_task, pin
from .inputs.raster import native_blocks
from .loading import local_pin
from .processing import CarbonProcessor


ROOT = Path(__file__).resolve().parents[1]


def run_tile(tile_id, year, manifest, output, work, *, model=None, source=None, worker_python=sys.executable):
    """Build an annual tile from discovered or pinned inputs, or replay saved heights."""
    validate_observation(year, SUPPORTED_VERSION)
    grid = pacific_tile_plan(tile_id)
    if manifest is not None:
        inputs = read_json(manifest)
        if (inputs['tile'], inputs['year']) != (tile_id, year):
            raise ValueError('Input bundle tile and year differ from the requested tile and year')
        if inputs['schema'] == 'agc-tile-replay-v1':
            if model is not None or source is not None:
                raise ValueError('A replay uses saved heights and takes no model or inference source')
            return replay_tile(manifest.parent, inputs, grid, output, work)
        if inputs['schema'] != 'agc-tile-inputs-v1':
            raise ValueError('Unsupported tile input bundle schema')
    if model is None or source is None:
        raise ValueError('Tile production requires --model and --source')

    windows = {window['id']: window for window in grid['windows']}
    if manifest is None:
        items = discover_geomad(grid, year)
        prepare_window = prepare_discovered_core
    else:
        items, route = local_geomad(manifest.parent, inputs['geomad'], year)
        worldcover = local_pin(manifest.parent, inputs['worldcover'])
        prepare_window = partial(prepare_core, route=route)
    selected, lineage = window_sources(windows, items)
    profile = scientific_profile()
    workers = prepare_workspace(output, work, windows)
    replay_rows = {}
    with rasterio.Env(GDAL_PAM_ENABLED='NO', PROJ_NETWORK='OFF'):
        if manifest is None:
            sources = live_tile_sources(grid, lineage, work)
        else:
            cover = read_worldcover(worldcover, {'grid': {'crs': grid['crs'],
                                     'retained_shape': grid['tile_shape']}}, Affine(*grid['tile_affine']))
            sources = nullcontext(cover)
        with sources as cover:
            for key, window in windows.items():
                worker = workers[key]
                package = worker.parent/'package'
                task = prepare_window(grid, window, selected[key], worldcover=cover,
                                      profile=profile, year=year, package=package)
                completed = CarbonProcessor(worker_python, ROOT, model, source, worker).process(
                    {'task': task, 'package': package})['completion']
                replay_rows[key] = dict(task=pin(package/'task.json', work),
                                        prepared_rgb=pin(worker/'prepared/tci-input.tif', work),
                                        raw_height=pin(worker/completed['height']['raw'], work),
                                        source_ids=sorted(item['id'] for item in selected[key]))
        write_json(work/'replay.json', dict(schema='agc-tile-replay-v1', tile=tile_id, year=year,
                                           source_items=lineage, windows=replay_rows))
        return assemble_tile(tile_id, year, workers, lineage, output)


def replay_tile(root, inputs, grid, output, work):
    saved = inputs['windows']
    windows = {window['id']: window for window in grid['windows'] if window['id'] in saved}
    if not windows or windows.keys() != saved.keys():
        raise ValueError('Saved inputs do not identify finished cores of the official tile')
    year = inputs['year']
    items = [validate_item(item, year, SUPPORTED_VERSION) for item in inputs['source_items']]
    selected, lineage = window_sources(windows, items)
    for key, sources in selected.items():
        if set(saved[key]['source_ids']) != {item['id'] for item in sources}:
            raise ValueError('Saved input lineage differs from the window context')

    profile = scientific_profile()
    workers = prepare_workspace(output, work, windows)
    with rasterio.Env(GDAL_PAM_ENABLED='NO', PROJ_NETWORK='OFF'):
        for key, window in windows.items():
            replay_core(root, saved[key], grid, window, profile, year, workers[key])
        return assemble_tile(inputs['tile'], year, workers, lineage, output)


def scientific_profile():
    path = ROOT/'configs/scientific-profile.json'
    check_file(path, read_json(ROOT/'release.json')['scientific_profile_sha256'])
    return read_json(path)


def prepare_workspace(output, work, windows):
    if output.exists() or output == work or output.is_relative_to(work) or work.is_relative_to(output):
        raise ValueError('Use a fresh tile destination and a separate work directory')
    work.mkdir(parents=True, exist_ok=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    return {key: work/'windows'/key/'worker' for key in windows}


def window_sources(windows, items):
    if not items or len({item['id'] for item in items}) != len(items):
        raise ValueError('Expected distinct annual GeoMAD Items')
    if not windows:
        raise ValueError('No windows in the official tile')
    selected = {key: context_sources(window, items) for key, window in windows.items()}
    used = {item['id']: item for sources in selected.values() for item in sources}
    return selected, [used[key] for key in sorted(used)]


def context_sources(window, items):
    context = box(*window['input_bounds'])
    selected = []
    for item in items:
        bounds = box(*item['properties']['proj:bbox'])
        if context.intersection(bounds).area > 0:
            selected.append((item, bounds))
    if context.difference(unary_union([bounds for _, bounds in selected])).area > 1:
        raise ValueError('Annual GeoMAD Items do not cover the full imagery context')
    return [item for item, _ in selected]


def local_geomad(root, rows, year):
    items, route = [], {}
    for row in rows:
        item = validate_item(read_json(local_pin(root, row['item'])), year, SUPPORTED_VERSION)
        if set(row['assets']) != set(BANDS):
            raise ValueError('Expected pinned red, green, blue, nir and count assets')
        for band in BANDS:
            path = local_pin(root, row['assets'][band])
            with rasterio.open(path) as src:
                if src.driver != 'GTiff' or src.files != [str(path)]:
                    raise ValueError('GeoMAD assets must be self-contained local GeoTIFFs')
            href = item['assets'][band]['href']
            if href in route:
                raise ValueError('Duplicate GeoMAD asset href')
            route[href] = str(path)
        items.append(item)
    return items, route


def prepare_core(grid, window, items, route, worldcover, profile, year, package):
    plan = dict(grid=grid, selected=window, source_items=items)
    _, vrts = native_blocks(plan, package.parent/'native-inputs', route, lambda: None)
    blocks, vrt = canonical(plan, vrts, package, lambda: None)
    return write_core_task(grid, window, blocks, vrt, worldcover, profile, year, package)


def prepare_discovered_core(grid, window, items, worldcover, profile, year, package):
    blocks, vrt = load_geomad_window(grid, window, items, package)
    return write_core_task(grid, window, blocks, vrt, worldcover, profile, year, package)


def write_core_task(grid, window, blocks, vrt, worldcover, profile, year, package):
    x, y = window['global_input_window'][:2]
    h, w = window['retained_shape']
    core = np.s_[y:y+h, x:x+w]
    cover, processed = worldcover
    wc = package/'inputs/worldcover.tif'
    write_raster(wc, cover[core], processed[core], Affine(*grid['tile_affine']) * Affine.translation(x, y),
                 grid['crs'])
    task = make_tile_task(profile, grid, window, blocks, vrt, package, wc, year)
    write_json(package/'task.json', task)
    return task


def replay_core(root, row, grid, window, profile, year, worker):
    task_file = local_pin(root, row['task'])
    saved_rgb = local_pin(root, row['prepared_rgb'])
    raw_height = local_pin(root, row['raw_height'])
    task, package = read_json(task_file), task_file.parent
    validate_inputs(task, package)
    declared = task['grid']
    if ((task['year'], task['window'], declared['id'], declared['affine'], declared['shape'],
         declared['global_input_window'], declared['receiving_window']) !=
        (year, window['id'], grid['id'], grid['affine'], grid['shape'],
         window['global_input_window'], window['global_input_window'])
        or any(task[k] != profile[k] for k in ('policy', 'inference'))):
        raise ValueError('Saved task differs from the official window or the accepted scientific profile')
    wc = local_input(package, task['inputs']['worldcover']['path'])
    core = retained_transform(declared)
    worldcover = read_worldcover(wc, task, core)
    arrays, affine, _ = prepare(package, task, worker/'prepared')
    with rasterio.open(saved_rgb) as rgb:
        if (rgb.transform != affine or rgb.crs.to_string() != grid['crs']
                or rgb.dtypes != ('uint8',)*3 or not np.array_equal(rgb.read(), arrays['rgb'])
                or not np.array_equal(rgb.dataset_mask() > 0, arrays['rgb_valid'])):
            raise ValueError('Prepared imagery values or mask differ from the saved RGB input')
    height = read_raw_height(raw_height, declared, affine)
    d, (h, w) = declared['context'], declared['retained_shape']
    _, _, crop_grid = crop_height(height, arrays['rgb_valid'], affine, [d, d, w, h])
    products = tile_products(task, arrays, height, crop_grid, worldcover, worker/'native')
    identities = {k: task['inference'][k] for k in ('model', 'source_commit', 'source_sha256')}
    write_json(worker/'completion.json', dict(status='complete', window=window['id'], year=year,
               source_version=SUPPORTED_VERSION, declared_grid=declared, identities=identities,
               replay=dict(task_sha256=sha256(task_file), rgb_sha256=row['prepared_rgb']['sha256'],
                           raw_height_sha256=row['raw_height']['sha256']), **products))
