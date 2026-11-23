"""Synthetic host-delivery fixtures: no provider, raster processing or GPU."""
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
HOST = ROOT
sys.path.insert(0, str(HOST))
import configuration as op
import durable_product as product

spec = importlib.util.spec_from_file_location('delivery_launcher', HOST / 'launch-country.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('SPC_TEST_TMP'))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        fixture_codes = ['AAA', 'BBB'] + ['X' + chr(65 + n // 26) + chr(65 + n % 26) for n in range(20)]
        fixture_roster = [dict(country=dict(code=c, name='Example ' + str(i)), plan_status='checked',
                               execution_qualification='Synthetic validated fixture') for i, c in enumerate(fixture_codes)]
        mock = patch.object(op, 'roster', return_value=fixture_roster)
        mock.start(); self.addCleanup(mock.stop)
        self.config = self.root / 'run.json'
        self.write(self.config, dict(countries=['AAA'], year=2024, geomad_version='0.4.0',
                                   dependency_directory='dependencies', working_directory='work', output_directory='products'))

    def write(self, p, value):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(value))

    def test_selection_single_multiple_all_and_invalid(self):
        codes = [r['country']['code'] for r in op.roster(HOST)]
        self.assertEqual(len(codes), 22)
        self.assertEqual(op.selection('BBB', HOST), ['BBB'])
        self.assertEqual(op.selection(['AAA', 'BBB'], HOST), ['AAA', 'BBB'])
        self.assertEqual(op.selection(['ALL'], HOST), codes)
        for value in [['ALL', 'AAA'], ['XX'], [], ['AAA', 'AAA'], None, [2]]:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'Valid:'):
                op.selection(value, HOST)

    def test_listing_no_work_and_same_readme_catalog(self):
        rows = op.country_rows(HOST, self.config)
        self.assertEqual(len(rows), 22)
        self.assertTrue(all(r['dependencies'] == 'not supplied' for r in rows))
        self.assertIn('validated', next(r for r in rows if r['code'] == 'AAA')['execution'])
        self.assertFalse((self.root / 'work').exists())
        catalogue = op.read(HOST / 'configs/countries.json')['countries']
        self.assertEqual(len(catalogue), 22)
        self.assertEqual(len({r['country']['code'] for r in catalogue}), 22)

    def test_defaults_optional_overrides_and_version_errors(self):
        _, _, _, _, _, s = op.batch_settings(self.config, HOST)
        self.assertNotIn('model_attempts', s)
        self.assertEqual(s['image_reference'], 'dep-agc:0.1.0-review.5')
        c = op.read(self.config)
        c['advanced'] = dict(model_attempts=0, allow_acquisition=False)
        self.write(self.config, c)
        self.assertEqual(op.batch_settings(self.config, HOST)[-1]['model_attempts'], 0)
        c['geomad_version'] = '0.5.0'; self.write(self.config, c)
        with self.assertRaisesRegex(ValueError, 'Only GeoMAD'):
            op.batch_settings(self.config, HOST)

    def test_obsolete_public_field_is_rejected_without_alias(self):
        c = op.read(self.config)
        c['artifacts'] = c.pop('dependency_directory')
        self.write(self.config, c)
        with self.assertRaisesRegex(ValueError, 'Obsolete run field artifacts; use dependency_directory'):
            op.batch_settings(self.config, HOST)

    def test_missing_selected_dependencies_prevents_all_processing(self):
        c = op.read(self.config); c['countries'] = ['AAA', 'BBB']; self.write(self.config, c)
        with self.assertRaisesRegex(ValueError, '(?s)AAA:.*BBB:'):
            op.resolve_batch(self.config, HOST)
        self.assertFalse((self.root / 'work').exists())

    def test_overlapping_storage_rejected(self):
        c = op.read(self.config); c['output_directory'] = 'work/products'; self.write(self.config, c)
        with self.assertRaisesRegex(ValueError, 'separate'):
            op.batch_settings(self.config, HOST)

    def batch(self, runner):
        resolved = [(self.config, dict(country=c, year=2024, image='release', expected_image='sha256:qualified'))
                    for c in ['AAA', 'BBB']]
        with patch.object(op, 'resolve_batch', return_value=resolved), patch.object(op, 'batch_settings',
                return_value=({}, ['AAA', 'BBB'], None, self.root / 'work', None, {})), patch.object(
                launcher.subprocess, 'check_output', return_value='sha256:qualified\n'):
            return launcher.run_batch(self.config, runner)

    def test_serial_country_isolation_and_relaunch(self):
        seen = []
        def runner(config, c):
            seen.append(c['country'])
            return dict(status='complete', product=str(self.root / 'products' / c['country']))
        first = self.batch(runner)
        second = self.batch(runner)
        self.assertEqual(seen, ['AAA', 'BBB', 'AAA', 'BBB'])
        self.assertEqual(first['created_epoch'], second['created_epoch'])
        self.assertNotEqual(first['countries']['AAA']['product'], first['countries']['BBB']['product'])

    def test_batch_failure_is_not_complete_or_skipped(self):
        seen = []
        def runner(config, c):
            seen.append(c['country'])
            if c['country'] == 'BBB':
                raise RuntimeError('controlled second-country failure')
            return dict(status='complete')
        with self.assertRaisesRegex(RuntimeError, 'second-country'):
            self.batch(runner)
        state = op.read(self.root / 'work/batches/2024/batch-state.json')
        self.assertEqual(state['status'], 'failed')
        self.assertEqual(list(state['countries']), ['AAA'])

    def test_incomplete_country_stops_batch(self):
        with self.assertRaisesRegex(RuntimeError, 'Country incomplete'):
            self.batch(lambda *a: dict(status='incomplete'))

    def test_unqualified_source_image_prevents_any_dispatch(self):
        with patch.object(op, 'resolve_batch', return_value=[(self.config,
                dict(image='release', expected_image=None))]), patch.object(
                launcher.subprocess, 'check_output') as image, self.assertRaisesRegex(
                RuntimeError, 'no image is qualified'):
            launcher.run_batch(self.config, lambda *a: self.fail('runner executed'))
        image.assert_not_called()

    def test_wrong_image_prevents_runner(self):
        with patch.object(launcher.subprocess, 'check_output', return_value='wrong'), self.assertRaises(ValueError):
            # Outer patch must override the ordinary batch helper's image stub.
            with patch.object(op, 'resolve_batch', return_value=[(self.config,
                    dict(image='release', expected_image='expected'))]):
                launcher.run_batch(self.config, lambda *a: self.fail('runner executed'))

    def test_all_selected_ownership_failures_are_reported_before_dispatch(self):
        from spc_agb_inputs.dependencies import ownership_metadata
        entries = []
        for code in ['AAA', 'BBB']:
            bundle = self.root / 'dependencies' / code
            plan = dict(country=dict(code=code), grid=dict(crs='EPSG:32702',
                affine=[10,0,400000,0,-10,8500000], windows=[dict(id='window')]))
            self.write(bundle / 'plan.json', plan)
            entry = dict(country=plan['country'], plan=dict(path='plan.json',sha256=op.sha(bundle/'plan.json')))
            entries.append(entry)
            self.write(bundle / 'countries.json',dict(countries=[entry]))
            self.write(bundle / 'dependency-manifest.json',dict(catalog=dict(path='countries.json',sha256=op.sha(bundle/'countries.json'))))
        c = op.read(self.config); c['countries'] = ['AAA', 'BBB']; self.write(self.config,c)
        def missing_origin(path, root):
            ownership_metadata(dict(grid=dict(crs='EPSG:32702',affine=[10,0,400000,0,-10,8500000])))
        with patch.object(op,'roster',return_value=entries), patch.object(op,'resolve',side_effect=missing_origin) as check, patch.object(launcher,'Launcher') as dispatch:
            with self.assertRaisesRegex(ValueError,'(?s)AAA:.*ownership.*BBB:.*ownership'):
                launcher.run_batch(self.config)
            self.assertEqual(check.call_count,2)
            dispatch.assert_not_called()
        # Unselected incomplete bundles are not examined or a global blocker.
        c['countries']=['AAA']; self.write(self.config,c)
        with patch.object(op,'roster',return_value=entries), patch.object(op,'resolve',return_value=dict(limits={})) as check:
            selected=op.resolve_batch(self.config,HOST)
            self.assertEqual(len(selected),1)
            self.assertEqual(check.call_count,1)

    def test_country_settings_are_derived_and_pending_cap_is_not_fixed(self):
        entries = []
        for code in ['AAA', 'BBB']:
            bundle = self.root / 'dependencies' / code
            plan = dict(country=dict(code=code), year=2024, source_version='0.4.0',
                        grid=dict(windows=[dict(id='window')]))
            self.write(bundle / 'plan.json', plan)
            entry = dict(country=plan['country'], plan=dict(path='plan.json', sha256=op.sha(bundle / 'plan.json')))
            entries.append(entry)
            self.write(bundle / 'countries.json', dict(schema='agc-spatial-countries-v1', countries=[entry]))
            self.write(bundle / 'dependency-manifest.json', dict(catalog=dict(path='countries.json', sha256=op.sha(bundle / 'countries.json'))))
        config_data = op.read(self.config); config_data['countries'] = ['AAA', 'BBB']; self.write(self.config, config_data)
        with patch.object(op, 'roster', return_value=entries), patch.object(op, 'resolve', side_effect=lambda *a: dict(limits={})):
            resolved = op.resolve_batch(self.config, HOST)
        config, c = resolved[0]
        internal = op.read(config)
        self.assertEqual(c['limits']['model_attempts'], None)
        self.assertEqual(internal['log_directory'], str(self.root / 'work/AAA/2024/logs'))
        self.assertEqual(c['durable_output'], self.root / 'products/AAA/2024')
        self.assertEqual(internal['limits']['minutes'], 65)
        other = op.read(resolved[1][0])
        self.assertNotEqual(other['working_directory'], internal['working_directory'])
        self.assertNotEqual(other['log_directory'], internal['log_directory'])
        self.assertNotEqual(resolved[1][1]['durable_output'], c['durable_output'])


class ProductTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('SPC_TEST_TMP'))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.source = self.root / 'temporary-assembly'; self.source.mkdir()
        self.final = self.root / 'product'
        self.raster = self.root / 'old-work/carbon.tif'; self.raster.parent.mkdir()
        self.raster.write_bytes(b'synthetic unchanged raster bytes')
        self.write(self.source / 'country-summary.json', dict(complete=True, country=dict(code='AAA'), year=2024, summary=dict(stock_tC=123.)))
        self.write(self.source / 'collection.json', dict(type='Collection', assets=dict(
            carbon=dict(href='../old-work/carbon.tif', **{'agc:sha256':op.sha(self.raster)}),
            accounting=dict(href='country-summary.json')), links=[]))

    def write(self, p, obj):
        p.write_text(json.dumps(obj))

    def test_copy_relocate_and_verify_without_originals(self):
        before = self.raster.read_bytes()
        first = product.publish(self.source, self.final)
        self.assertEqual(first['status'], 'published-verified')
        second = product.publish(self.source, self.final)
        self.assertEqual(second['status'], 'reused-verified')
        moved = self.root / 'relocated'; self.final.rename(moved)
        self.source.rename(self.root / 'hidden-assembly'); self.raster.parent.rename(self.root / 'hidden-work')
        receipt = product.verify(moved)
        raster = next(r for r in receipt['files'] if r['path'].endswith('.tif'))
        self.assertEqual((moved / raster['path']).read_bytes(), before)

    def test_tamper_and_incomplete_are_rejected(self):
        product.publish(self.source, self.final)
        (self.final / 'country-summary.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'identity differs'):
            product.verify(self.final)
        self.write(self.source / 'country-summary.json', dict(complete=False))
        with self.assertRaisesRegex(ValueError, 'incomplete'):
            product.publish(self.source, self.root / 'incomplete')

    def test_missing_asset_preserves_failure_evidence(self):
        self.raster.unlink()
        with self.assertRaises(FileNotFoundError):
            product.publish(self.source, self.final)
        self.assertTrue(self.source.is_dir())

    def test_existing_unreceipted_output_is_not_overwritten(self):
        self.final.mkdir(); (self.final / 'partial').write_text('keep')
        with self.assertRaises(FileNotFoundError):
            product.publish(self.source, self.final)
        self.assertEqual((self.final / 'partial').read_text(), 'keep')


class ReleaseTests(unittest.TestCase):
    def test_generic_profile_has_explicit_annual_policy_and_no_geographic_bindings(self):
        profile = op.read(HOST / 'configs/scientific-profile.json')
        self.assertEqual(profile['policy']['version'], 'annual-geomad-carbon-v2')
        self.assertEqual(profile['policy']['low_ndvi_treatment'], 'assumed_zero')
        self.assertIsNone(profile['window'])
        for key in ['country', 'grid', 'inputs']:
            self.assertEqual(profile[key], {})

    def test_release_identity_and_profile_integrity(self):
        release = op.read(HOST / 'release.json')
        self.assertEqual(release['name'], 'dep-agc')
        self.assertEqual(release['version'], '0.1.0-review.5')
        self.assertEqual(
            release['image_id'],
            'sha256:d0dbe6ec7fc6fd63f298cea6b7cb5c40155e14b4a515071de9a53b0f08c13887')
        self.assertEqual(release['qualification'], 'local-matching-container-passed')
        inventory, _, profile, path = op.package_metadata(HOST)
        self.assertEqual(inventory['processor_image'], release['image_id'])
        self.assertEqual(op.sha(path), release['scientific_profile_sha256'])
        self.assertTrue(profile['template_in_image'].endswith('/scientific-profile.json'))

    def test_changed_profile_is_refused_before_country_resolution(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'configs').mkdir()
            (root / 'release.json').write_bytes((HOST / 'release.json').read_bytes())
            (root / 'configs/scientific-profile.json').write_text('{}')
            with self.assertRaisesRegex(ValueError, 'profile differs'):
                op.package_metadata(root)

    def test_single_public_configuration_and_no_superseded_entrypoints(self):
        self.assertEqual(set(op.read(HOST / 'run.json')), {
            'countries', 'year', 'geomad_version', 'dependency_directory',
            'working_directory', 'output_directory'})
        manifest = HOST / 'manifest.json'
        if manifest.exists():
            paths = {r['path'] for r in op.read(manifest)['files']}
            self.assertNotIn('DEP-PR.md', paths)
            self.assertNotIn('spc_agb_dep/stage_inputs.py', paths)
            self.assertFalse(any('execution_' in p for p in paths if p.endswith('.json')))



if __name__ == '__main__':
    unittest.main()
