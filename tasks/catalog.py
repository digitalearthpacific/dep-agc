"""Small pinned index of separately prepared country plans and operational jobs."""

from pathlib import Path
import os

from carbon.io import read_json, sha256, write_json
from .completion import output_status
from .loading import local_pin
from carbon.observation import validate_observation
from tasks.inputs.dependencies import ownership_metadata
from .job import plan_windows, resolve_job, verify_inputs


def relative_path(root, name):
    part = Path(name)
    path = (root/part).resolve()
    if (part.is_absolute() or '..' in part.parts or '\\' in name or ':' in name
            or not path.is_relative_to(root)):
        raise ValueError('Catalog artifact reference escapes root')
    return path


def country_plan(catalog, country, year, source_version='0.4.0'):
    index = read_json(catalog)
    schema = index.get('schema')
    if schema not in ['agc-spatial-countries-v1', 'agc-prepared-countries-v2']:
        raise ValueError('Expected spatial v1 or year-bound prepared v2 catalogue; no legacy migration')
    if year is None:
        if schema != 'agc-spatial-countries-v1':
            raise ValueError('Spatial planning requires a spatial-only catalogue')
    else:
        validate_observation(year, source_version)
    name = index.get('artifact_root', '.')
    if Path(name).is_absolute() or '\\' in name or ':' in name:
        raise ValueError('Catalog artifact_root must be relative')
    root = (catalog.parent/name).resolve()
    if not root.is_dir() or root == Path(root.anchor):
        raise ValueError('Existing non-filesystem-root artifact directory required')
    entries = index['countries']
    spatial = schema == 'agc-spatial-countries-v1'
    keys = [(r['country']['code'],) if spatial else (r['country']['code'], r['year'], r['source_version']) for r in entries]
    key = (country,) if spatial else (country, year, source_version)
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicate spatial or country/year/source bindings')
    if key not in keys:
        raise ValueError('No registered country/year/source combination')
    entry = dict(entries[keys.index(key)])
    if spatial:
        if entry.get('jobs'):
            raise ValueError('Reusable spatial catalogue must not bind annual jobs')
        if year is not None:
            entry.update(year=year, source_version=source_version)
    plan_file = local_pin(root, entry['plan'])
    plan = read_json(plan_file)
    if plan['country'] != entry['country']:
        raise ValueError('Registered country and spatial plan identity differ')
    if entry.get('plan_status') == 'blocked' or plan.get('planning_status') == 'blocked':
        if entry.get('jobs'):
            raise ValueError('Blocked spatial plan cannot bind operational jobs')
        return dict(root=root, entry=entry, plan=plan, plan_file=plan_file, jobs={},
                    windows=[], blocked_reason=plan.get('reason', 'Regional planning unresolved'))
    if entry.get('plan_status') == 'checked' and not entry.get('planning_check'):
        raise ValueError('Checked plan requires a pinned planning check')
    if entry.get('planning_check'):
        checked = read_json(local_pin(root, entry['planning_check']))
        if (not checked['passed'] or checked['country'] != entry['country']
                or checked['plan_sha256'] != entry['plan']['sha256']):
            raise ValueError('Spatial planning check/plan identity differs')
    ownership_metadata(plan, entry.get('ownership'))
    approved = entry.get('accepted_projected')
    required = checked.get('accepted_projected') if entry.get('planning_check') else None
    if bool(approved) != bool(required):
        raise ValueError('Approved projected geometry required by spatial check')
    if approved:
        if (approved['sha256'], approved['crs']) != (required['sha256'], required['crs']):
            raise ValueError('Approved geometry/check identity differs')
        local_pin(root, approved)
    windows = plan_windows(plan, country, year, source_version)
    if plan['country'] != entry['country']:
        raise ValueError('Registered country and plan identity differ')
    jobs = entry.get('jobs', [])
    ids = [r['window'] for r in jobs]
    outputs = [relative_path(root, r['output']) for r in jobs if r.get('output')]
    paths = [relative_path(root, r['job']['path']) for r in jobs]
    if (len(ids) != len(set(ids)) or len(outputs) != len(set(outputs))
            or len(paths) != len(set(paths))):
        raise ValueError('Duplicate window/job/output bindings')
    if set(ids) - {r['id'] for r in windows}:
        raise ValueError('Job binding is not a planned window')
    return dict(root=root, entry=entry, plan=plan, plan_file=plan_file,
                jobs={r['window']: r for r in jobs}, windows=windows, year=year, source_version=source_version)


def resolve_window(selection, window):
    if selection.get('blocked_reason'):
        raise ValueError('Country plan blocked: '+selection['blocked_reason'])
    if window not in {r['id'] for r in selection['windows']}:
        raise ValueError('No matching configured window')
    if window not in selection['jobs']:
        raise ValueError('Window is unprepared: no operational job binding')
    row = selection['jobs'][window]
    path = local_pin(selection['root'], row['job'])
    entry = selection['entry']
    binding = resolve_job(path, entry['year'], entry['source_version'])
    task = read_json(binding['task'])
    if (task['country'], task['grid']['id'], task['window']) != (
        entry['country'], selection['plan']['grid']['id'], window
    ) or sha256(binding['plan']) != entry['plan']['sha256']:
        raise ValueError('Registered plan/job country/grid/window identity differs')
    verify_inputs(binding)
    output = relative_path(selection['root'], row['output']) if row.get('output') else None
    return dict(job=path, binding=binding, output=output,
                completion_sha256=row.get('completion_sha256'))


def listed_tasks(catalog, country, year, window=None, pending_only=False,
                 source_version='0.4.0'):
    selection = country_plan(catalog, country, year, source_version)
    if selection.get('blocked_reason'):
        return [dict(country=country, year=year, source_version=source_version,
                     window=None, plan_status='blocked', status='plan-blocked',
                     reason=selection['blocked_reason'])]
    rows = []
    if window is not None and window not in {w['id'] for w in selection['windows']}:
        raise ValueError('No matching configured window')
    for w in selection['windows']:
        if window is not None and w['id'] != window:
            continue
        row = dict(country=country, year=year, source_version=source_version,
                   grid=selection['plan']['grid']['id'], window=w['id'],
                   plan_status=selection['entry'].get('plan_status', 'registered'),
                   plan_sha256=selection['entry']['plan']['sha256'], status='unprepared')
        if w['id'] in selection['jobs']:
            try:
                resolved = resolve_window(selection, w['id'])
                row['status'] = output_status(resolved['output'], resolved['job'],
                    resolved['binding'], resolved['completion_sha256'])
                row['job_sha256'] = sha256(resolved['job'])
                row['input_status'] = 'native/static files verified'
            except FileNotFoundError as exc:
                row.update(status='missing-inputs', error=str(exc))
        if not pending_only or row['status'] != 'completed':
            rows.append(row)
    return rows


def bind_results(catalog, country, year, results, output):
    """New execution index with explicit output locators; no input/receipt edits.

    The country bundle and results remain separate siblings. Moving their common
    root preserves all relative references and existing completion filtering.
    """
    selected = country_plan(catalog,country,year)
    root = selected['root']
    results, output = results.resolve(), output.absolute()
    common = Path(os.path.commonpath([root,results,output.parent.resolve()]))
    if common == Path(common.anchor) or root.is_relative_to(results):
        raise ValueError('Results need a non-root common parent, never a writable ancestor of inputs')
    from copy import deepcopy
    entry = deepcopy(selected['entry'])
    for field in ['plan','planning_check','accepted_projected']:
        if field in entry:
            entry[field]['path'] = (root/entry[field]['path']).relative_to(common).as_posix()
    for row in entry.get('jobs',[]):
        row['job']['path'] = (root/row['job']['path']).relative_to(common).as_posix()
        row['output'] = (results/row['window']).relative_to(common).as_posix()
        row.pop('completion_sha256',None)  # Fresh output locators never inherit an old receipt pin.
    value=dict(schema='agc-prepared-countries-v2',
        artifact_root=os.path.relpath(common,output.parent).replace('\\','/'),countries=[entry])
    if output.exists():
        if read_json(output)!=value:
            raise FileExistsError('Existing execution index differs; preserved, no silent rebinding')
    else:
        write_json(output,value)
    return output


if __name__ == '__main__':
    import argparse
    p=argparse.ArgumentParser(description='Bind one prepared country to an explicit separate results root; no dispatch')
    for name in ['catalog','results','output']:
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--country',required=True)
    p.add_argument('--year',type=int,required=True)
    a=p.parse_args()
    bind_results(a.catalog.absolute(),a.country,a.year,a.results.absolute(),a.output.absolute())
