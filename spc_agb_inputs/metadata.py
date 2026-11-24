"""Resolve pinned annual Items using DEP's published item naming convention."""
import gzip
import io
import json
import re
from urllib.parse import urlsplit
from datetime import datetime, timezone
from pathlib import Path

import requests

from spc_agb_worker.observation import validate_observation
from spc_agb_worker.io import check_file, read_json, sha256, write_json


GEOMAD_ITEMS_URL = 'https://stac.digitalearthpacific.org/collections/dep_s2_geomad/items'


def validate_item(item, year, version):
    validate_observation(year, version)
    p = item['properties']
    start = datetime.fromisoformat(p['start_datetime'].replace('Z', '+00:00'))
    end = datetime.fromisoformat(p['end_datetime'].replace('Z', '+00:00'))
    if (item['collection'] != 'dep_s2_geomad'
            or not item['id'].startswith('dep_s2_geomad_') or not re.fullmatch(r'dep_s2_geomad_\d{3}_\d{3}_' + f'{year:04d}', item['id'])
            or start != datetime(year, 1, 1, tzinfo=timezone.utc)
            or end.replace(microsecond=0) != datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            or p['proj:epsg'] != 3832 or p['proj:shape'] != [9600, 9600]
            or p['proj:transform'][:2] != [10, 0] or p['proj:transform'][3:5] != [0, -10]):
        raise ValueError(f'Wrong requested annual GeoMAD item/grid for {year} v{version}')
    for band in ['red', 'green', 'blue', 'nir', 'count']:
        if band not in item['assets']:
            raise ValueError(f'Missing required {band} asset for GeoMAD {year} v{version}')
        a = item['assets'][band]
        spec = a['raster:bands'][0]
        path = urlsplit(a['href']).path
        asset_years = re.findall(r'(?<![0-9])([0-9]{4})(?![0-9])', path)
        expected = (1, 0, 9999) if band == 'count' else (.0001, 0, 0)
        if (not a['href'].startswith('https://dep-public-data.s3.us-west-2.amazonaws.com/dep_s2_geomad/0-4-0/')
                or f'{year:04d}' not in asset_years or any(int(y) != year for y in asset_years)
                or (spec['scale'], spec['offset'], spec['nodata']) != expected
                or spec['data_type'] != 'uint16'):
            raise ValueError(f'Incompatible {band} asset year/version/encoding for GeoMAD {year} v{version}')
    return item


def resolve_items(plan, tiles, incoming, report, guard, year, version):
    """Prefer the byte-pinned plan's saved Items; otherwise discover bounded IDs.

    Original provider response bytes remain separate from receipts. A saved404
    is explicit missing coverage, never a fabricated observation or replacement.
    """
    validate_observation(year, version)
    names = [f'dep_s2_geomad_{x:03d}_{y:03d}_{year:04d}' for x, y in tiles]
    # Spatial plans may retain historical source provenance; never adopt it for another year.
    saved = {i['id']: validate_item(i, year, version) for i in plan.get('source_items', [])
             if i['id'] in names}
    absent = {r.get('receipt', {}).get('url', '').split('/')[-1]
              for r in plan.get('missing_source_items', [])}
    items, missing = [], []
    for name in names:
        guard()
        if name in saved:
            items.append(saved[name])
            continue
        if name in absent:
            missing.append(dict(id=name, status=404, evidence='pinned plan missing_source_items'))
            continue
        incoming.mkdir(parents=True, exist_ok=True)
        report.mkdir(parents=True, exist_ok=True)
        original = incoming/(name+'.json')
        receipt = report/(name+'.json')
        if not receipt.exists():
            if original.exists():
                raise FileExistsError('Unreceipted metadata original retained')
            url = f'{GEOMAD_ITEMS_URL}/{name}'
            with requests.get(url, stream=True, timeout=(10, 30)) as response:
                if response.status_code not in [200, 404]:
                    response.raise_for_status()
                raw = response.raw.read(1048577)
                if len(raw) > 1048576:
                    raise ValueError('Metadata1MiB bound exceeded')
                original.write_bytes(raw)
                write_json(receipt, dict(url=url, status=response.status_code,
                    bytes=len(raw), sha256=sha256(original), headers=dict(response.headers)))
        row = read_json(receipt)
        check_file(original, row['sha256'])
        if row['status'] == 404:
            missing.append(dict(id=name, status=404, receipt_sha256=sha256(receipt)))
            continue
        raw = original.read_bytes()
        if raw[:2] == b'\x1f\x8b':
            raw = gzip.GzipFile(fileobj=io.BytesIO(raw)).read(1048577)
        if len(raw) > 1048576:
            raise ValueError('Decoded metadata1MiB bound exceeded')
        item = validate_item(json.loads(raw), year, version)
        if item['id'] != name:
            raise ValueError('Requested/resolved Item identity differs')
        items.append(item)
    return items, missing
