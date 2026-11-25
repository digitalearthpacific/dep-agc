"""Prepare every pinned country window; CPU only, resumable explicit inputs.

No scheduler or inference. Native blocks are shared within the country bundle;
the common processor regenerates canonical inputs from those pinned blocks.
"""
import argparse
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import numpy as np
import rasterio
from shapely.geometry import box, mapping, shape
from shapely.ops import transform
from pyproj import Transformer

from spc_agb_dep.catalog import country_plan
from spc_agb_dep.inputs import BANDS, local_pin
from spc_agb_dep.job import resolve_job, verify_inputs
from spc_agb_worker.accounting import domain
from spc_agb_worker.grid import MODEL_CONTEXT, MODEL_PATCH, MODEL_STRIDE, validate_grid
from spc_agb_worker.io import check_file, read_json, sha256, write_json
from spc_agb_worker.prepare import prepare
from .canonical import CANONICAL_BANDS, canonical
from .dependencies import ownership_metadata, owner_fingerprints, dependency_file
from .geometry import country_boundary, preparation_geometry, owners, provider_tiles, worldcover_tiles
from spc_agb_worker.observation import validate_observation
from .metadata import resolve_items
from .raster import native_blocks, receiving_source_bounds, static_owner
from .state import atomic, writer_lock
from .transport import RangeArchive, gateway, reader_options


def pin(path, root):
    return dict(path=path.relative_to(root).as_posix(), sha256=sha256(path))


def external(config_file, spec):
    path = (config_file.parent/spec['path']).resolve()
    check_file(path, spec['sha256'])
    return path


def signature(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def copy_pin(src, dst, digest):
    check_file(src, digest)
    if dst.exists():
        check_file(dst, digest)
    else:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, dst)
        check_file(dst, digest)


def make_task(template, plan, window, blocks, vrt, package, owner_rows, order_pin, year):
    task = deepcopy(template)
    task.update(schema='spc-agb-window-v3', year=year, country=plan['country'], window=window['id'])
    grid = plan['grid']
    task['grid'] = dict(id=grid['id'], crs=grid['crs'], affine=grid['affine'], shape=grid['shape'],
        receiving_window=window.get('receiving_window', window['global_input_window']),
        global_input_window=window['global_input_window'], canonical_global_offset=grid.get('canonical_global_offset', [0, 0]),
        context=MODEL_CONTEXT, patch=MODEL_PATCH, stride=MODEL_STRIDE,
        retained_shape=[side - 2 * MODEL_CONTEXT for side in reversed(window['global_input_window'][2:])],
        bounds=window['bounds'])
    task['inputs'] = dict(canonical=pin(vrt, package), owner_order=order_pin,
        bands=CANONICAL_BANDS, blocks=[pin(Path(b['file']['path']), package) for b in blocks], owners=owner_rows)
    validate_grid(task)
    return task


def verified_preparation(output, selection, config_file):
    """Completed inputs are verified read-only, without reacquiring original seeds.

    This runs no scientific preparation with changed code. The original producer
    identity/clock/expenditure remain in the untouched state; the verifier reports
    its own identity separately. Partial preparation still requires matching code.
    """
    path = output/'preparation-state.json'
    if not path.exists():
        return None
    state = read_json(path)
    if state.get('status') != 'prepared':
        return None
    contract = state['contract']
    if (contract['config_sha256'] != sha256(config_file)
            or contract['plan_sha256'] != sha256(selection['plan_file'])
            or contract['country'] != selection['plan']['country']
            or contract['year'] != selection['year']
            or set(state['completed']) != {w['id'] for w in selection['windows']}):
        raise ValueError('Completed preparation config/plan/roster differs; no silent reuse')
    for record in state['completed'].values():
        binding = resolve_job(local_pin(output, record['job']), contract['year'], '0.4.0')
        if sha256(binding['plan']) != contract['plan_sha256']:
            raise ValueError('Prepared job/plan identity differs')
        verify_inputs(binding)
        for row in record['canonical_files']:
            local_pin(output, row)
    prepared = country_plan(output/'countries.json',contract['country']['code'],contract['year'])
    if set(prepared['jobs']) != set(state['completed']):
        raise ValueError('Prepared catalog/job roster differs')
    return dict(state,read_only_verification=dict(passed=True,state_sha256=sha256(path),
        verifier_sha256=sha256(Path(__file__)),model_calls=0,new_provider_bytes=0,
        original_producer_identity_preserved=True,scientific_preparation_executed=False))


def prepare_country(catalog, code, year, config_file, output, archive):
    config = read_json(config_file)
    if config.get('schema') != 'agc-country-preparation-v4':
        raise ValueError('Generated country preparation v4 configuration required')
    if set(config) - {'schema', 'year', 'source_version', 'dependency_root', 'boundary', 'task_template', 'host_free_path', 'limits', 'ownership', 'accepted_projected'}:
        raise ValueError('Unsupported generated preparation setting; external seeds are not accepted')
    validate_observation(year, config['source_version'])
    if config['year'] != year:
        raise ValueError('Preparation CLI/configuration year differs')
    selection = country_plan(catalog, code, year)
    if selection.get('blocked_reason'):
        raise ValueError('Spatial plan is blocked')
    plan = selection['plan']
    complete = verified_preparation(output, selection, config_file)
    if complete is not None:
        return complete
    boundary_file = external(config_file, config['boundary'])
    if plan.get('boundary', {}).get('sha256') not in [None, config['boundary']['sha256']]:
        raise ValueError('Pinned plan/boundary identity differs')
    template = read_json(external(config_file, config['task_template']))
    geographic = country_boundary(boundary_file, code)
    approved = config.get('accepted_projected')
    expected = selection['entry'].get('accepted_projected')
    if bool(approved) != bool(expected) or (approved and (approved['sha256'], approved['crs']) != (expected['sha256'], expected['crs'])):
        raise ValueError('Required approved projected geometry differs or is missing')
    dependency_root = Path(config['dependency_root'])
    if not dependency_root.is_absolute():
        raise ValueError('Explicit absolute dependency mount root required')
    approved_file = dependency_file(dependency_root, approved) if approved else None
    country = preparation_geometry(geographic, plan['grid']['crs'], dependency_root, approved)
    if config.get('ownership') != selection['entry'].get('ownership'):
        raise ValueError('Approved ownership dependency/configuration differs')
    ownership = ownership_metadata(plan, config.get('ownership'))
    anchor = ownership['anchor']
    calculated, order = owners(country, anchor)
    if 'owner_count' in ownership:
        actual = owner_fingerprints(calculated, order)
        if any(actual[k] != ownership[k] for k in actual):
            raise ValueError('Prepared owner IDs/order/bounds differ from accepted metadata')
    if 'owners' in plan:
        if order != plan['owner_order'] or len(calculated) != len(plan['owners']):
            raise ValueError('Accepted owner identity/order differs')
        for a, b in zip(calculated, plan['owners']):
            if (a['id'], a['numeric_id'], a['bounds']) != (b['id'], b['numeric_id'], b['bounds']) or not shape(a['geometry']).equals_exact(shape(b['geometry']), 0):
                raise ValueError('Accepted owner geometry differs')
        calculated = plan['owners']
    contract = dict(approved_projected_sha256=approved['sha256'] if approved else None, config_sha256=sha256(config_file), plan_sha256=sha256(selection['plan_file']),
                    code_sha256={p.name:sha256(p) for p in Path(__file__).parent.glob('*.py')},
                    country=plan['country'], year=year, owner_order=order)
    output.mkdir(parents=True, exist_ok=True)
    state_file = output/'preparation-state.json'
    with writer_lock(output/'preparation.lock'):
        if state_file.exists():
            state = read_json(state_file)
            if state['contract'] != contract:
                raise ValueError('Preparation inputs/configuration/code changed; old work retained, use a new output root')
        else:
            if any(output.iterdir()) and set(p.name for p in output.iterdir()) != {'preparation.lock'}:
                raise FileExistsError('Unrecognised preparation output retained')
            state = dict(contract=contract, started_epoch=time.time(), completed={}, status='preparing')
            atomic(state_file, state)
        copy_pin(selection['plan_file'], output/'plan.json', contract['plan_sha256'])
        copy_pin(boundary_file, output/'boundary.gpkg', config['boundary']['sha256'])
        if approved:
            copy_pin(approved_file, output/'approved-projected.geojson', approved['sha256'])
            copy_pin(local_pin(selection['root'], selection['entry']['planning_check']), output/'planning-check.json', selection['entry']['planning_check']['sha256'])
        limits = config['limits']
        deadline = state['started_epoch']+limits['seconds']

        def guard():
            if time.time() > deadline:
                raise TimeoutError('Country preparation clock exhausted; completed work retained')
            if shutil.disk_usage(config['host_free_path']).free < limits['minimum_free_bytes']:
                raise RuntimeError('Actual host free-space reserve exhausted')

        guard()
        provider_geometry = transform(Transformer.from_crs(plan['grid']['crs'], 3832, always_xy=True).transform,
            country.segmentize(240)) if approved else None
        tiles = set(provider_tiles(geographic, provider_geometry))
        project = Transformer.from_crs(plan['grid']['crs'],4326,always_xy=True).transform
        for w in selection['windows']:
            bounds = box(*receiving_source_bounds(plan, w))
            if 'tile_id' in plan['grid']:
                tiles.update(provider_tiles(None, bounds))
            else:
                tiles.update(provider_tiles(transform(project,bounds.segmentize(240))))
        items, missing = resolve_items(plan, sorted(tiles), archive/'metadata', output/'metadata', guard, year, config['source_version'])
        if not items:
            raise ValueError(f'GeoMAD {year} v{config["source_version"]} unavailable: no resolved Items; no invented observation')
        source_plan = deepcopy(plan)
        source_plan['source_items'] = items
        source_plan['missing_source_items'] = missing
        sources = plan.get('worldcover')
        if sources is None:
            import requests
            sources = []
            for tile in worldcover_tiles(geographic):
                guard()
                url = 'https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map/ESA_WorldCover_10m_2021_v200_'+tile+'_Map.tif'
                path = output/(tile+'-availability.json')
                if not path.exists():
                    with requests.head(url, timeout=(10, 30), allow_redirects=False) as response:
                        if response.status_code not in [200, 404]:
                            response.raise_for_status()
                        write_json(path, dict(tile=tile, url=url, status=response.status_code, headers=dict(response.headers)))
                sources.append(read_json(path))
        for s in sources:
            if '/v200/2021/map/ESA_WorldCover_10m_2021_v200_' not in s['url'] or s['status'] not in [200, 404]:
                raise ValueError('WorldCover source/version contract differs')
        transport = RangeArchive(archive/'ranges', output/'transfer.json', limits['transfer_bytes'],
            deadline=time.monotonic()+max(0, deadline-time.time()), source_identity=None)
        all_urls = [i['assets'][b]['href'] for i in items for b in BANDS]+[s['url'] for s in sources if s['status']==200]
        options = dict(GDAL_DISABLE_READDIR_ON_OPEN='EMPTY_DIR', GDAL_PAM_ENABLED='NO',
            GDAL_GEOREF_SOURCES='INTERNAL', GDAL_CACHEMAX=64*2**20, PROJ_NETWORK='OFF', **reader_options())
        with gateway(transport, all_urls) as route, rasterio.Env(**options):
            for window in selection['windows']:
                guard()
                key = window['id']
                if key in state['completed']:
                    record = state['completed'][key]
                    binding = resolve_job(local_pin(output, record['job']), year, '0.4.0')
                    verify_inputs(binding)
                    for row in record['canonical_files']:
                        local_pin(output, row)
                    continue
                started = time.monotonic()
                working = output/'windows'/key
                working.mkdir(parents=True, exist_ok=True)
                specific = dict(source_plan, selected=window)
                # Shared native files are committed per band, never repeatedly copied.
                records, vrts = native_blocks(specific, output/'native', route, guard, resume=True)
                package = working/'package'
                package.mkdir(exist_ok=True)
                blocks, vrt = canonical(specific, vrts, package, guard)
                static = output/'static'/key
                ids = static/'inputs/owner-ids.json'
                if not ids.exists():
                    ids.parent.mkdir(parents=True, exist_ok=True)
                    write_json(ids, order)
                if read_json(ids) != order:
                    raise ValueError('Owner order changed')
                selected = box(*window['bounds'])
                owner_rows, support_rows = [], []
                for row in calculated:
                    geometry = shape(row['geometry']).intersection(selected)
                    if geometry.area <= 0:
                        continue
                    guard()
                    target = static/'inputs/owners'/(row['id']+'.npz')
                    receipt = target.with_suffix('.json')
                    identity = dict(geometry=mapping(geometry), crs=plan['grid']['crs'], bounds=row['bounds'],
                                    worldcover=sources, method='accepted physical ellipsoid <=5m / nearest WorldCover2021v200')
                    if receipt.exists():
                        saved = read_json(receipt)
                        if saved['identity'] != json.loads(json.dumps(identity)):
                            raise ValueError('Static cache scientific/source identity changed')
                        check_file(target, saved['sha256'])
                    else:
                        if target.exists():
                            raise FileExistsError('Uncommitted static input retained')
                        target.parent.mkdir(parents=True, exist_ok=True)
                        arrays, references = static_owner(geometry, row['bounds'], plan['grid']['crs'],
                            sources, route, working/'static-native'/row['id'])
                        np.savez_compressed(target, **arrays)
                        write_json(receipt, dict(identity=identity, sha256=sha256(target), source_references=references))
                    owner_rows.append(dict(id=row['id'], numeric_id=row['numeric_id'], bounds=row['bounds'], file=pin(target, static)))
                    support_rows.append(dict(id=row['id'], numeric_id=row['numeric_id'], bounds=row['bounds'], geometry=mapping(geometry)))
                task = make_task(template, plan, window, blocks, vrt, package, owner_rows, pin(ids, static), year)
                task['ownership_grid'] = ownership
                task_file = output/('task-'+key+'.json')
                if task_file.exists():
                    if read_json(task_file) != task:
                        raise ValueError('Partial task differs; preserve and diagnose')
                else:
                    write_json(task_file, task)
                support = output/('support-'+key+'.json')
                if not support.exists():
                    write_json(support, dict(schema='agc-native-accounting-support-v1', crs=plan['grid']['crs'],
                        country=plan['country'], grid_id=plan['grid']['id'], window=key, owners=support_rows))
                for record in records:
                    for band in record['bands'].values():
                        for kind in ['native','valid']:
                            band[kind] = pin(Path(band[kind]['path']), output)
                index = output/('native-index-'+key+'.json')
                if not index.exists():
                    write_json(index, dict(schema='agc-native-cache-v2', year=year, version='0.4.0',
                        records=records, source_items=items, missing_source_items=missing,
                        support=pin(support, output), plan_sha256=contract['plan_sha256']))
                job_file = output/('job-'+key+'.json')
                if not job_file.exists():
                    write_json(job_file, dict(schema='agc-geomad-job-v2', task=pin(task_file, output),
                        native_index=pin(index, output), plan=pin(output/'plan.json', output), static_root=static.relative_to(output).as_posix()))
                binding = resolve_job(job_file, year, '0.4.0')
                verify_inputs(binding)
                state['completed'][key] = dict(job=pin(job_file, output), seconds=time.monotonic()-started,
                    owners=len(owner_rows), canonical_files=[pin(vrt, output), *[pin(Path(b['file']['path']), output) for b in blocks]])
                atomic(state_file, state)
                print('PREPARED', code, key, flush=True)
        guard()
        if set(state['completed']) != {w['id'] for w in selection['windows']}:
            raise ValueError('Country preparation incomplete')
        entry = dict(country=plan['country'], ownership=ownership, year=year, source_version='0.4.0', plan=pin(output/'plan.json', output),
            jobs=[dict(window=w['id'], job=state['completed'][w['id']]['job'], output='results/'+w['id']) for w in selection['windows']])
        if approved:
            entry['accepted_projected'] = dict(path='approved-projected.geojson', sha256=approved['sha256'], crs=approved['crs'])
            entry['planning_check'] = pin(output/'planning-check.json', output)
        atomic(output/'countries.json', dict(schema='agc-prepared-countries-v2', artifact_root='.', countries=[entry]))
        state.update(status='prepared', provider_payload_bytes=transport.state['actual'], model_calls=0)
        atomic(state_file, state)
        return state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ['catalog', 'config', 'output', 'archive']:
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--country', required=True)
    p.add_argument('--year', type=int, required=True)
    p.add_argument('--verification-report',type=Path,help='Optional fresh report outside read-only inputs')
    a = p.parse_args()
    result=prepare_country(a.catalog.absolute(), a.country, a.year, a.config.absolute(), a.output.absolute(), a.archive.absolute())
    if a.verification_report:
        write_json(a.verification_report,result)
    print('PREPARATION',result['status'],len(result['completed']),
          'verified read-only' if result.get('read_only_verification') else 'generated inputs')


if __name__ == '__main__':
    main()
