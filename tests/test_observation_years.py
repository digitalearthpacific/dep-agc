"""Prepared annual/spatial integration fixtures. Not executed in source review."""
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from shapely.geometry import MultiPolygon, box, mapping

from spc_agb_worker.observation import validate_observation, annual_interval
from spc_agb_worker.io import sha256, write_json
from spc_agb_inputs.metadata import validate_item, resolve_items
from spc_agb_inputs.geometry import preparation_geometry
from spc_agb_inputs.prepare_country import make_task
from spc_agb_dep.catalog import country_plan


def item(year):
    start, end = annual_interval(year)
    return dict(id=f'dep_s2_geomad_075_025_{year:04d}', collection='dep_s2_geomad',
        properties={'start_datetime': start, 'end_datetime': end, 'proj:epsg': 3832,
                    'proj:shape': [9600, 9600], 'proj:transform': [10, 0, 0, 0, -10, 0]},
        assets={band: {'href': f'https://dep-public-data.s3.us-west-2.amazonaws.com/dep_s2_geomad/0-4-0/{year:04d}/{band}.tif',
                       'raster:bands': [dict(scale=1 if band == 'count' else .0001, offset=0,
                                           nodata=9999 if band == 'count' else 0, data_type='uint16')]}
                for band in ['red', 'green', 'blue', 'nir', 'count']})


class ObservationTests(unittest.TestCase):
    def test_multiple_years_and_wrong_item_interval_asset_or_version(self):
        for year in [2022, 2024, 2025]:
            self.assertEqual(validate_observation(year, '0.4.0'), year)
            valid = item(year)
            self.assertEqual(validate_item(valid, year, '0.4.0'), valid)
            for changed in [item(year - 1), deepcopy(valid)]:
                if changed['id'] == valid['id']:
                    changed['assets']['red']['href'] = item(year - 1)['assets']['red']['href']
                with self.assertRaises(ValueError):
                    validate_item(changed, year, '0.4.0')
            changed = deepcopy(valid)
            changed['properties']['end_datetime'] = f'{year}-11-30T23:59:59Z'
            with self.assertRaises(ValueError):
                validate_item(changed, year, '0.4.0')
        for year, version in [(True, '0.4.0'), (0, '0.4.0'), (2024, '0.5.0')]:
            with self.assertRaises(ValueError):
                validate_observation(year, version)

    def test_discovery_requests_selected_year_and_caches_unavailable_without_substitution(self):
        class Response:
            headers = {}
            status_code = 404
            raw = io.BytesIO(b'not found')
            def __enter__(self): return self
            def __exit__(self, *args): pass
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('spc_agb_inputs.metadata.requests.get', return_value=Response()) as request:
                for _ in range(2):
                    items, missing = resolve_items({'source_items': [item(2024)]}, [(75, 25)],
                        root/'incoming', root/'receipts', lambda: None, 2025, '0.4.0')
                    self.assertEqual(items, [])
                    self.assertEqual(missing[0]['id'], 'dep_s2_geomad_075_025_2025')
                request.assert_called_once()
                self.assertTrue(request.call_args.args[0].endswith('_2025'))

    def test_discovery_validates_response_not_just_requested_url(self):
        class Response:
            headers = {}
            status_code = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
        response = Response()
        response.raw = io.BytesIO(json.dumps(item(2024)).encode())
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with patch('spc_agb_inputs.metadata.requests.get', return_value=response):
                with self.assertRaises(ValueError):
                    resolve_items({}, [(75, 25)], root/'incoming', root/'receipts',
                                  lambda: None, 2025, '0.4.0')

    def test_spatial_plan_reused_without_annual_copies_and_tasks_get_own_year(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            window = dict(id='r+000-c+000', global_input_window=[0, 0, 2496, 2496],
                          bounds=[402240, 8477280, 422720, 8497760])
            plan = dict(country=dict(code='AAA', name='Synthetic'), year=2024, source_version='0.4.0',
                grid=dict(anchor=[400000,8500000],id='preserved-grid', crs='EPSG:32702', affine=[10, 0, 400000, 0, -10, 8500000],
                          shape=[2496, 2496], windows=[window]))
            write_json(root/'plan.json', plan)
            digest = sha256(root/'plan.json')
            write_json(root/'countries.json', dict(schema='agc-spatial-countries-v1', countries=[
                dict(country=plan['country'], plan=dict(path='plan.json', sha256=digest))]))
            profile = json.loads((Path(__file__).parents[1]/'configs/scientific-profile.json').read_text())
            vrt = root/'input.vrt'; vrt.write_bytes(b'fixture')
            tasks = []
            for year in [2023, 2025]:
                selected = country_plan(root/'countries.json', 'AAA', year)
                self.assertEqual(selected['year'], year)
                task = make_task(profile, selected['plan'], window, [], vrt, root, [], {}, year)
                self.assertEqual(task['grid']['id'], 'preserved-grid')
                self.assertEqual(task['year'], year)
                tasks.append(task)
            self.assertEqual(sha256(root/'plan.json'), digest)
            self.assertEqual(tasks[0]['grid'], tasks[1]['grid'])
            self.assertEqual(tasks[0]['policy'], tasks[1]['policy'])

    def test_approved_geometry_is_read_verbatim_and_never_falls_back(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)/'dependencies/AAA'; root.mkdir(parents=True)
            geometry = MultiPolygon([box(0, 0, 10, 10)])
            write_json(root/'approved-projected.geojson', dict(type='FeatureCollection',
                crs=dict(properties=dict(name='EPSG:3832')),
                features=[dict(type='Feature', geometry=mapping(geometry))]))
            pin = dict(path='approved-projected.geojson', sha256=sha256(root/'approved-projected.geojson'))
            with patch('spc_agb_inputs.geometry.projected_boundary', side_effect=AssertionError('raw reprojection')):
                actual = preparation_geometry(None, 'EPSG:3832', root, pin)
                self.assertEqual(actual.wkb, geometry.wkb)
                from spc_agb_worker.io import local_input
                with self.assertRaisesRegex(ValueError, 'inputs/'):
                    local_input(root, 'approved-projected.geojson')
                with self.assertRaises(ValueError):
                    preparation_geometry(None, 'EPSG:3832', root, dict(pin, sha256='changed'))
                with self.assertRaises(ValueError):
                    preparation_geometry(None, 'EPSG:32702', root, pin)
                for bad in ['../approved-projected.geojson', str(root/'approved-projected.geojson'), 'nested/../../approved-projected.geojson', 'nested\\escape.geojson', 'missing.geojson']:
                    with self.subTest(path=bad), self.assertRaises(ValueError):
                        preparation_geometry(None, 'EPSG:3832', root, dict(pin, path=bad))
                link = root/'linked.geojson'
                try:
                    link.symlink_to(root/'approved-projected.geojson')
                except OSError:
                    self.skipTest('Symlink creation unavailable in this fixture environment')
                with self.assertRaisesRegex(ValueError, 'Symlink'):
                    preparation_geometry(None, 'EPSG:3832', root, dict(pin,path=link.name))
                directory = root/'linked-directory'
                directory.symlink_to(root, target_is_directory=True)
                with self.assertRaisesRegex(ValueError, 'Symlink'):
                    preparation_geometry(None, 'EPSG:3832', root, dict(pin,path='linked-directory/approved-projected.geojson'))
                write_json(root/'invalid.geojson', dict(type='FeatureCollection',
                    crs=dict(properties=dict(name='EPSG:3832')),
                    features=[dict(type='Feature',geometry=dict(type='Point',coordinates=[0,0]))]))
                with self.assertRaisesRegex(ValueError, 'polygon'):
                    preparation_geometry(None, 'EPSG:3832', root, dict(path='invalid.geojson',sha256=sha256(root/'invalid.geojson')))



if __name__ == '__main__':
    unittest.main()
