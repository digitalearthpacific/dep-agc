"""Validate annual GeoMAD Items and name their canonical DEP catalogue URLs."""
import re
from urllib.parse import urlsplit
from datetime import datetime, timezone

from carbon.observation import validate_observation


GEOMAD_ITEMS_URL = 'https://stac.digitalearthpacific.org/collections/dep_s2_geomad/items'


def validate_item(item, year, version):
    validate_observation(year, version)
    p = item['properties']
    epsg, code = p.get('proj:epsg'), p.get('proj:code')
    start = datetime.fromisoformat(p['start_datetime'].replace('Z', '+00:00'))
    end = datetime.fromisoformat(p['end_datetime'].replace('Z', '+00:00'))
    if (item['collection'] != 'dep_s2_geomad'
            or not item['id'].startswith('dep_s2_geomad_') or not re.fullmatch(r'dep_s2_geomad_\d{3}_\d{3}_' + f'{year:04d}', item['id'])
            or start != datetime(year, 1, 1, tzinfo=timezone.utc)
            or end.replace(microsecond=0) != datetime(year, 12, 31, 23, 59, 59, tzinfo=timezone.utc)
            or (epsg, code) not in ((3832, None), (None, 'EPSG:3832'), (3832, 'EPSG:3832'))
            or p['proj:shape'] != [9600, 9600]
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
