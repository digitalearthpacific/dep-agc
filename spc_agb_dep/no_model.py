"""Explicit no-model completion when no known land has eligible native inputs.

Uses the same preparation/accountant. Low-NDVI eligible land still runs the
accepted worker; this shortcut never invents model validity or assumed zeros.
"""
from pathlib import Path
import time

import numpy as np

from spc_agb_worker.accounting import account, domain
from spc_agb_worker.grid import identity
from spc_agb_worker.io import sha256, validate, write_json, write_raster
from spc_agb_worker.prepare import prepare
from spc_agb_worker import __version__


def eligible_land(arrays, task, dom):
    d = task['grid']['context']
    h, w = task['grid']['retained_shape']
    sl = np.s_[d:d+h, d:d+w]
    return (dom['land'] & arrays['rgb_valid'][sl] & arrays['red_valid'][sl]
            & arrays['nir_valid'][sl] & arrays['ndvi_valid'][sl])


def no_model_completion(loaded, model, source, output):
    """Return None for a normal GPU task; otherwise write a genuine CPU result."""
    started = time.monotonic()
    task, package = loaded['task'], loaded['package']
    probe = output.with_name(output.name+'-eligibility')
    arrays, grid, rgb = prepare(package, task, probe)
    dom = domain(package, task)
    eligible = eligible_land(arrays, task, dom)
    write_json(probe/'eligibility.json', dict(eligible_known_land_pixels=int(eligible.sum()),
        eligible_known_land_ha=float(dom['weights'][eligible].sum()),
        model_required=bool(eligible.any()), task_sha256=sha256(package/'task.json')))
    if eligible.any():
        return None
    output.mkdir(parents=True, exist_ok=False)
    inputs = validate(task, package, model, source)  # File hashes only, no model load.
    code = {p.name:sha256(p) for p in (Path(__file__).resolve().parents[1]/'spc_agb_worker').glob('*.py')}
    identities = dict(task_identity=identity(task), task_sha256=sha256(package/'task.json'), inputs=inputs,
        code=code, model=task['inference']['model'], source_commit=task['inference']['source_commit'],
        source_sha256=task['inference']['source_sha256'])
    # No inferred height exists. The accepted NoData sentinel makes model_valid false.
    height = np.full(arrays['rgb_valid'].shape, -9999, np.float32)
    h, w = task['grid']['retained_shape']
    d = task['grid']['context']
    from affine import Affine
    raw, cropped = output/'height/full-context.tif', output/'height/height-aoi.tif'
    write_raster(raw, np.full(height.shape, -9999, np.float32), np.zeros(height.shape, bool), grid, task['grid']['crs'], -9999)
    write_raster(cropped, np.full((h,w), -9999, np.float32), np.zeros((h,w), bool), grid*Affine.translation(d,d), task['grid']['crs'], -9999)
    account_started = time.monotonic()
    result = account(package, task, arrays, height, grid, output/'native')
    if result['summary']['assessed_ha'] or result['summary']['stock_tC']:
        raise ValueError('No-model branch produced classified carbon')
    # Preserve the completion file contract; raw/retained height explicitly unavailable.
    prepared = output/'prepared'; prepared.mkdir()
    import shutil
    shutil.copyfile(rgb, prepared/'tci-input.tif')
    result = dict(schema='spc-agb-window-completion-v3', task_identity=identity(task),
        declared_grid=task['grid'], status='complete', worker_version=__version__,
        window=task['window'], year=task['year'], source_product='GeoMAD', source_version='0.4.0',
        identities=identities, **result, prepared_rgb_sha256=sha256(rgb),
        height=dict(raw=raw.relative_to(output).as_posix(), raw_sha256=sha256(raw),
            retained=cropped.relative_to(output).as_posix(), retained_sha256=sha256(cropped)),
        completion_kind='no_model', reason='No known land has RGB/count/NIR/NDVI eligibility; exclusions retained',
        seconds=dict(prepare=account_started-started, inference=0., account=time.monotonic()-account_started),
        total_seconds=time.monotonic()-started, inference_metrics=dict(model_calls=0, model_loaded=False))
    write_json(output/'inputs.json', identities)
    write_json(output/'completion.json', result)
    return result
