"""Dependency paths and approved ownership metadata; standard library only."""
import hashlib
import json
import math
from pathlib import Path


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def dependency_file(root, pin):
    """Resolve a supplied dependency, distinct from the worker inputs/ contract."""
    root = Path(root).absolute()
    name = pin['path']
    if not isinstance(name, str) or not name:
        raise ValueError('Dependency file reference must be a nonempty string')
    relative = Path(name)
    if (relative.is_absolute()
            or '\\' in name or ':' in name or '..' in relative.parts):
        raise ValueError('Dependency must be a contained relative file reference')
    path = root / relative
    if any(p.is_symlink() for p in [path, *path.parents]):
        raise ValueError('Symlink dependency paths are not supported')
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file():
        raise ValueError('Missing or escaping dependency file: ' + name)
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    if digest.hexdigest() != pin['sha256']:
        raise ValueError('Dependency checksum differs: ' + name)
    return path


def ownership_metadata(plan, supplied=None):
    """Require an approved origin; never infer one from affine/bounds or defaults."""
    grid = plan.get('grid')
    if not isinstance(grid, dict):
        raise ValueError('Approved ownership requires a spatial grid definition')
    declared = grid.get('anchor')
    if supplied is None:
        if declared is None:
            raise ValueError('Blocked ownership: approved ownership-grid origin is missing')
        supplied = dict(schema='agc-ownership-grid-v1', anchor=declared,
                        cell_size_m=5000, crs=grid['crs'], source='pinned spatial plan grid.anchor')
    if not isinstance(supplied, dict):
        raise ValueError('Approved ownership metadata must be an object')
    if supplied.get('schema') != 'agc-ownership-grid-v1':
        raise ValueError('Explicit agc-ownership-grid-v1 metadata required')
    anchor = supplied.get('anchor')
    if (not isinstance(anchor, list) or len(anchor) != 2
            or any(type(v) not in [int, float] or not math.isfinite(v) for v in anchor)):
        raise ValueError('Approved ownership origin must contain two finite coordinates')
    if declared is not None and anchor != declared:
        raise ValueError('Ownership origin differs from the approved spatial plan')
    if supplied.get('cell_size_m') != 5000 or supplied.get('crs') != grid['crs']:
        raise ValueError('Ownership cell size/CRS differs from the accepted grid')
    affine = grid['affine']
    if len(affine) != 6 or [affine[i] for i in [0, 1, 3, 4]] != [10, 0, 0, -10]:
        raise ValueError('Ownership requires the declared north-up 10 m grid')
    if any(not math.isclose((origin - anchor[i]) / 10, round((origin - anchor[i]) / 10),
                            rel_tol=0, abs_tol=1e-8)
           for i, origin in enumerate([affine[2], affine[5]])):
        raise ValueError('Ownership origin is not aligned to the approved native pixels')
    expected = ['owner_count', 'owner_order_sha256', 'owner_bounds_sha256']
    if any(k in supplied for k in expected):
        if not all(k in supplied for k in expected):
            raise ValueError('Incomplete accepted owner count/order/bounds fingerprints')
        if type(supplied['owner_count']) is not int or not 0 < supplied['owner_count'] <= 65535:
            raise ValueError('Accepted owner count must fit the output dtype')
        for k in expected[1:]:
            if (not isinstance(supplied[k], str) or len(supplied[k]) != 64
                    or any(c not in '0123456789abcdef' for c in supplied[k])):
                raise ValueError('Invalid accepted ownership fingerprint: ' + k)
    return supplied


def owner_fingerprints(rows, order):
    """Fingerprint existing owner identities/bounds; no geometry or weight creation."""
    by_id = {r['id']: r for r in rows}
    if len(by_id) != len(rows) or len(set(order)) != len(order) or set(by_id) != set(order):
        raise ValueError('Duplicate/missing owner identity in supplied order')
    values = [dict(id=key, numeric_id=by_id[key]['numeric_id'],
                   bounds=[float(v) for v in by_id[key]['bounds']]) for key in order]
    return dict(owner_count=len(order), owner_order_sha256=fingerprint(order),
                owner_bounds_sha256=fingerprint(values))
