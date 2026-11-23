"""Explicit, pinned operational inputs; no catalogue or workstation discovery."""

from pathlib import Path

from spc_agb_worker.grid import validate_grid
from spc_agb_worker.io import check_file, local_input, read_json
from .inputs import local_pin
from spc_agb_worker.observation import validate_observation
from spc_agb_inputs.metadata import validate_item


def plan_windows(plan, country, year=None, source_version='0.4.0'):
    if year is not None:
        validate_observation(year, source_version)
    elif source_version != '0.4.0':
        raise ValueError('Only GeoMAD v0.4.0 is supported')
    if plan['country']['code'] != country or plan.get('source_version', source_version) != source_version:
        raise ValueError('Configured country/spatial plan/GeoMAD version differs')
    windows = plan['grid']['windows']
    ids = [w['id'] for w in windows]
    footprints = [tuple(w['global_input_window']) for w in windows]
    if (not windows or len(ids) != len(set(ids))
            or len(footprints) != len(set(footprints))):
        raise ValueError('Empty or duplicate plan windows')
    return windows


def verify_inputs(binding, model=None, source=None):
    """Hash existing native/static inputs only; no canonical/raster generation."""
    task = read_json(binding['task'])
    index = read_json(binding['native_index'])
    validate_observation(task['year'], task['source_version'])
    if (index['year'], index['version']) != (task['year'], task['source_version']):
        raise ValueError('Native index annual source differs')
    for item in index['source_items']:
        validate_item(item, task['year'], task['source_version'])
    files = 0
    for record in index['records']:
        for band in ['red', 'green', 'blue', 'nir', 'count']:
            for kind in ['native', 'valid']:
                local_pin(binding['native_index'].parent, record['bands'][band][kind])
                files += 1
    local_pin(binding['native_index'].parent, index['support'])
    if (model is None) != (source is None):
        raise ValueError('Supply both model and source, or neither, for file verification')
    if model is not None:
        spec = task['inference']
        for name, key in [(spec['model']['weights'], 'weights_sha256'),
                          (spec['model']['training'], 'training_sha256')]:
            check_file(model/name, spec['model'][key])
        for name, digest in spec['source_sha256'].items():
            check_file(source/name, digest)
    return dict(native_files=files, static_files=len(task['inputs']['owners'])+1,
                model_source_verified=model is not None, raster_generation=False)


def resolve_job(path, year, source_version):
    job = read_json(path)
    if job.get('schema') != 'agc-geomad-job-v2':
        raise ValueError('Expected agc-geomad-job-v2')
    root = path.parent.resolve()
    task_file = local_pin(root, job['task'])
    task = read_json(task_file)
    validate_grid(task)
    validate_observation(year, source_version)
    if (task['year'], task['source_product'], task['source_version']) != (year, 'GeoMAD', source_version):
        raise ValueError('Requested observation year/version and task differ')
    index = local_pin(root, job['native_index'])
    plan_file = local_pin(root, job['plan'])
    plan = read_json(plan_file)
    windows = plan_windows(plan, task['country']['code'], year, source_version)
    if (plan['country'], plan['grid']['id']) != (task['country'], task['grid']['id']):
        raise ValueError('Country plan/task identity differs')
    if any(plan['grid'][key] != task['grid'][key] for key in ['crs','affine','shape']):
        raise ValueError('Country plan/task grid differs')
    selected = next((w for w in windows if w['id'] == task['window']), None)
    if selected is None or selected['global_input_window'] != task['grid']['global_input_window']:
        raise ValueError('Task window is not the configured plan window')
    static = Path(job['static_root'])
    if static.is_absolute() or '..' in static.parts or '\\' in str(static) or ':' in str(static):
        raise ValueError('Static root must be package relative')
    static = root / static
    if not static.resolve().is_relative_to(root) or static.is_symlink():
        raise ValueError('Static root escapes job')
    # Only these static files are supplied; canonical inputs are generated.
    for row in [task['inputs']['owner_order'], *(o['file'] for o in task['inputs']['owners'])]:
        p = local_input(static, row['path'])
        if not p.is_file():
            raise FileNotFoundError(p)
        check_file(p, row['sha256'])
    return dict(task=task_file, native_index=index,
                native_index_sha256=job['native_index']['sha256'], package=static, plan=plan_file)
