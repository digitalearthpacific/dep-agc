"""Read-only identity/output checks for existing common-processor completions.

This is completion filtering, not a new numerical audit or a retry mechanism.
"""

from pathlib import Path

from spc_agb_worker.grid import identity
from spc_agb_worker.observation import annual_interval
from spc_agb_worker.io import check_file, read_json, sha256

LAYERS = {'RGB', 'height', 'carbon', 'status', 'NDVI', 'count', 'WorldCover',
          'physical-area-ha', 'owner'}


def output_file(root, name):
    p = (root/name).resolve()
    if (Path(name).is_absolute() or '\\' in name or ':' in name
            or not p.is_relative_to(root.resolve()) or not p.is_file()):
        raise ValueError('Missing or escaping required output: '+name)
    return p


def verify_completion(output, job, binding, completion_sha256=None):
    record = output_file(output, 'integration-completion.json')
    if completion_sha256:
        check_file(record, completion_sha256)
    result = read_json(record)
    task = read_json(binding['task'])
    selected = [task['country']['code'], task['grid']['id'], task['window'], str(task['year'])]
    if (result['status'] != 'complete' or result['task_id'] != selected
            or result['job_sha256'] != sha256(job)):
        raise ValueError('Completion job/task identity differs')
    worker_file = output_file(output, 'worker/completion.json')
    worker = read_json(worker_file)
    ids = worker['identities']
    expected_inputs = {r['path']: r['sha256'] for r in [task['inputs']['canonical'],
        task['inputs']['owner_order'], *task['inputs']['blocks'],
        *(o['file'] for o in task['inputs']['owners'])]}
    expected_code = {p.name: sha256(p) for p in
                     (Path(__file__).resolve().parents[1]/'spc_agb_worker').glob('*.py')}
    if (worker['status'] != 'complete' or worker['task_identity'] != identity(task)
            or worker['declared_grid'] != task['grid']
            or (worker['year'], worker['source_product'], worker['source_version']) !=
               (task['year'], 'GeoMAD', task['source_version'])
            or ids['task_sha256'] != sha256(binding['task']) or ids['inputs'] != expected_inputs
            or ids['model'] != task['inference']['model']
            or ids['source_commit'] != task['inference']['source_commit']
            or ids['source_sha256'] != task['inference']['source_sha256']
            or ids['code'] != expected_code or worker['summary'] != result['summary']):
        raise ValueError('Completed scientific inputs/model/code/grid identity differs')
    if set(worker['files']) != LAYERS or set(result['cog_checks']) != LAYERS:
        raise ValueError('Required nine native/COG outputs differ')
    for name, pin in worker['files'].items():
        check_file(output_file(output, 'worker/native/'+name+'.tif'), pin['sha256'])
    for key in ['raw', 'retained']:
        check_file(output_file(output, 'worker/'+worker['height'][key]),
                   worker['height'][key+'_sha256'])
    check_file(output_file(output, 'worker/prepared/tci-input.tif'), worker['prepared_rgb_sha256'])
    item_file = output_file(output, result['item'])
    item = read_json(item_file)
    collection = read_json(output_file(output, result['collection']))
    if set(item['assets']) != LAYERS | {'completion', 'lineage', 'load', 'accounting-support'}:
        raise ValueError('Required STAC assets differ')
    start, end = annual_interval(task['year'])
    if (item['properties'].get('start_datetime') != start
            or item['properties'].get('end_datetime') != end
            or item['properties'].get('agc:source_version') != task['source_version']):
        raise ValueError('Completed STAC observation year/version differs')
    if (item['properties']['agc:country'] != task['country']
            or item['properties']['agc:grid_id'] != task['grid']['id']
            or item['properties']['agc:window'] != task['window']
            or item['properties']['agc:stock_tC'] != worker['summary']['stock_tC']):
        raise ValueError('STAC task/summary identity differs')
    for name, asset in item['assets'].items():
        path = output_file(output, (item_file.parent/asset['href']).relative_to(output).as_posix())
        check_file(path, asset['agc:sha256'])
        if name in LAYERS and asset['agc:sha256'] != result['cog_checks'][name]['sha256']:
            raise ValueError('COG completion/STAC identity differs')
    if not any(link['rel'] == 'item' and
               (output_file(output, result['collection']).parent/link['href']).resolve() == item_file
               for link in collection['links']):
        raise ValueError('Collection does not reference the completed Item')
    if read_json(output_file(output, 'product/evidence/completion.json')) != worker:
        raise ValueError('Product evidence/worker completion differs')
    lineage = read_json(output_file(output, 'product/evidence/lineage.json'))
    if (lineage['identities'] != ids or
            lineage['native_index_sha256'] != binding['native_index_sha256']):
        raise ValueError('Product lineage/input identities differ')
    return result


def output_status(output, job, binding, completion_sha256=None):
    if output is None or not output.exists():
        return 'configured'
    if not output.is_dir():
        raise ValueError('Output path is not a directory')
    if (output/'integration-failure.json').exists():
        return 'failed'
    if not (output/'integration-completion.json').exists():
        return 'incomplete'
    verify_completion(output, job, binding, completion_sha256)
    return 'completed'
