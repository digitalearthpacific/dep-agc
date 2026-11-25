"""CPU-only run routing/completion fixtures; never dispatch a model."""

from argparse import Namespace
from copy import deepcopy
import json
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import test_processor_execution as fixtures
from tasks.catalog import country_plan, listed_tasks, resolve_window, bind_results
from tasks.completion import LAYERS
from tasks.job import resolve_job
from tasks.run_task import run, select_arguments
from carbon.grid import identity
from carbon.io import read_json, sha256


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2)+'\n')


class CatalogTests(unittest.TestCase):
    inputs = fixtures.ExecutionTests.inputs
    staged = fixtures.ExecutionTests.staged

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('SPC_TEST_TMP'))
        self.root = Path(self.tmp.name)
        self.job_root = self.staged()
        self.job = self.job_root/'job.json'
        self.plan = self.job_root/'plan.json'
        self.index = self.root/'catalog.json'
        self.row = dict(window='r+000-c+000', job=dict(path='job/job.json', sha256=sha256(self.job)),
                        output='results/area')
        self.entry = dict(country={'code': 'AAA', 'name': 'Example A'}, year=2024, source_version='0.4.0',
                          plan=dict(path='job/plan.json', sha256=sha256(self.plan)), jobs=[self.row])
        self.catalog = dict(schema='agc-prepared-countries-v2', artifact_root='.', countries=[self.entry])
        self.persist()

    def tearDown(self):
        self.tmp.cleanup()

    def persist(self):
        save(self.index, self.catalog)

    def test_bind_results_keeps_inputs_and_fresh_output_receipts_separate(self):
        before=sha256(self.index)
        target=self.root/'execution-countries.json'
        bind_results(self.index,'AAA',2024,self.root/'fresh-results',target)
        output_pin=sha256(target)
        bind_results(self.index,'AAA',2024,self.root/'fresh-results',target)
        self.assertEqual(sha256(target),output_pin)
        selection=country_plan(target,'AAA',2024)
        row=selection['jobs']['r+000-c+000']
        self.assertEqual(row['output'],'fresh-results/r+000-c+000')
        self.assertEqual(row['job']['sha256'],self.row['job']['sha256'])
        self.assertEqual(selection['entry']['plan']['sha256'],self.entry['plan']['sha256'])
        self.assertEqual(sha256(self.index),before)
        with self.assertRaisesRegex(ValueError,'ancestor'):
            bind_results(self.index,'AAA',2024,self.root,self.root/'invalid.json')

    def test_execution_job_ids_require_no_unpublished_json_helper(self):
        p=subprocess.run([sys.executable,'-B','-m','tasks.print_tasks','--catalog',str(self.index),
            '--country','AAA','--year','2024','--ids'],cwd=Path(__file__).parents[1],text=True,capture_output=True)
        self.assertEqual(p.returncode,0,p.stderr);self.assertEqual(p.stdout.strip(),'r+000-c+000')

    def args(self, **kw):
        values = dict(catalog=self.index, country='AAA', year=2024, source_version='0.4.0',
                      window='r+000-c+000', job=None, output=None, model=None, source=None,
                      verify_inputs=True, prepare_only=False, skip_completed=False)
        values.update(kw)
        return Namespace(**values)

    def completed(self):
        """Tiny receipt/hash fixtures, not raster/scientific validation."""
        out = self.root/'results/area'
        t = read_json(self.job_root/'task.json')
        inputs = {r['path']: r['sha256'] for r in [t['inputs']['canonical'],
                  t['inputs']['owner_order'], *t['inputs']['blocks'],
                  *(o['file'] for o in t['inputs']['owners'])]}
        code_root = Path(__file__).resolve().parents[1]/'carbon'
        ids = dict(task_sha256=sha256(self.job_root/'task.json'), inputs=inputs,
                   code={p.name: sha256(p) for p in code_root.glob('*.py')},
                   model=t['inference']['model'], source_commit=t['inference']['source_commit'],
                   source_sha256=t['inference']['source_sha256'])
        def file(name):
            p = out/name; p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(name.encode())
            return dict(sha256=sha256(p))
        worker = dict(status='complete', task_identity=identity(t), declared_grid=t['grid'],
                      year=2024, source_product='GeoMAD', source_version='0.4.0', identities=ids,
                      summary={'stock_tC': 123.0}, files={n: file('worker/native/'+n+'.tif') for n in LAYERS})
        worker['height'] = dict(raw='height/raw.tif', retained='height/retained.tif',
            raw_sha256=file('worker/height/raw.tif')['sha256'],
            retained_sha256=file('worker/height/retained.tif')['sha256'])
        worker['prepared_rgb_sha256'] = file('worker/prepared/tci-input.tif')['sha256']
        save(out/'worker/completion.json', worker)
        save(out/'product/evidence/completion.json', worker)
        save(out/'product/evidence/lineage.json', dict(identities=ids,
             native_index_sha256=read_json(self.job)['native_index']['sha256']))
        for name in ['load', 'accounting-support']:
            save(out/('product/evidence/'+name+'.json'), {})
        assets = {}; cogs = {}
        for name in LAYERS:
            pin = file('product/'+name+'.tif'); cogs[name] = pin
            assets[name] = dict(href=name+'.tif', **{'agc:sha256': pin['sha256']})
        for name in ['completion', 'lineage', 'load', 'accounting-support']:
            assets[name] = dict(href='evidence/'+name+'.json',
                **{'agc:sha256': sha256(out/('product/evidence/'+name+'.json'))})
        save(out/'product/item.json', dict(assets=assets, properties={
            'agc:country': t['country'], 'start_datetime': '2024-01-01T00:00:00Z',
             'end_datetime': '2024-12-31T23:59:59Z', 'agc:source_version': '0.4.0', 'agc:grid_id': t['grid']['id'],
            'agc:window': t['window'], 'agc:stock_tC': 123.0}))
        save(out/'product/collection.json', dict(links=[dict(rel='item', href='item.json')]))
        save(out/'integration-completion.json', dict(status='complete',
            task_id=['AAA', t['grid']['id'], t['window'], str(t['year'])], job_sha256=sha256(self.job),
            summary=worker['summary'], cog_checks=cogs, item='product/item.json',
            collection='product/collection.json'))
        return out

    def test_selection_and_explicit_job_use_same_binding_without_execution(self):
        args = select_arguments(self.args())
        explicit = resolve_job(self.job, 2024, '0.4.0')
        selected = resolve_job(args.job, 2024, '0.4.0')
        self.assertEqual(selected, explicit)
        with patch('tasks.run_task.NativeGeoMADLoader') as loader:
            result = run(args)
            self.assertEqual(result['status'], 'inputs-verified'); loader.assert_not_called()
        self.assertFalse((self.root/'results').exists())
        explicit_args = select_arguments(self.args(catalog=None, country=None, window=None, job=self.job))
        self.assertEqual(run(explicit_args)['job_sha256'], result['job_sha256'])

    def test_incorrect_selection_and_unsupported_product(self):
        for country, year, version in [('CCC', 2024, '0.4.0'), ('AAA', 2025, '0.4.0'),
                                       ('AAA', 2024, '0.5.0')]:
            with self.assertRaises(ValueError): country_plan(self.index, country, year, version)
        for window in ['absent']:
            with self.assertRaises(ValueError): listed_tasks(self.index, 'AAA', 2024, window)
        with self.assertRaises(ValueError): select_arguments(self.args(window=None))

    def test_all_windows_remain_visible_and_unprepared_is_not_executable(self):
        plan = read_json(self.plan); row = deepcopy(plan['grid']['windows'][0])
        row['id'] = 'r+003-c+005'; row['global_input_window'][1] += 2048
        plan['grid']['windows'].append(row); save(self.plan, plan)
        job = read_json(self.job); job['plan']['sha256'] = sha256(self.plan); save(self.job, job)
        self.entry['plan']['sha256'] = sha256(self.plan); self.row['job']['sha256'] = sha256(self.job)
        self.persist()
        self.assertEqual([r['status'] for r in listed_tasks(self.index, 'AAA', 2024)], ['configured', 'unprepared'])
        with self.assertRaisesRegex(ValueError, 'unprepared'):
            select_arguments(self.args(window=row['id']))

    def test_duplicate_country_window_job_and_output_bindings_rejected(self):
        self.catalog['countries'].append(deepcopy(self.entry)); self.persist()
        with self.assertRaisesRegex(ValueError, 'Duplicate spatial or country/year/source bindings'): country_plan(self.index, 'AAA', 2024)
        self.catalog['countries'].pop(); self.entry['jobs'].append(deepcopy(self.row)); self.persist()
        with self.assertRaisesRegex(ValueError, 'Duplicate window'): country_plan(self.index, 'AAA', 2024)
        self.entry['jobs'].pop(); self.persist()
        plan = read_json(self.plan); plan['grid']['windows'].append(deepcopy(plan['grid']['windows'][0]))
        save(self.plan, plan); self.entry['plan']['sha256'] = sha256(self.plan); self.persist()
        with self.assertRaisesRegex(ValueError, 'duplicate plan'): country_plan(self.index, 'AAA', 2024)

    def test_changed_pins_and_plan_job_mismatch_fail_explicitly(self):
        self.row['job']['sha256'] = 'wrong'; self.persist()
        with self.assertRaises(ValueError): listed_tasks(self.index, 'AAA', 2024)
        self.row['job']['sha256'] = sha256(self.job)
        plan = read_json(self.plan); plan['grid']['id'] = 'other-grid'
        other = self.root/'other-plan.json'; save(other, plan)
        self.entry['plan'] = dict(path=other.name, sha256=sha256(other)); self.persist()
        with self.assertRaisesRegex(ValueError, 'identity differs'): listed_tasks(self.index, 'AAA', 2024)

    def test_missing_native_input_is_visible_but_cannot_be_selected(self):
        (self.job_root/'geomad/band').unlink()
        self.assertEqual(listed_tasks(self.index, 'AAA', 2024)[0]['status'], 'missing-inputs')
        with self.assertRaises(FileNotFoundError): select_arguments(self.args())

    def test_distinct_windows_cannot_share_a_job_or_output_binding(self):
        plan = read_json(self.plan); other = deepcopy(plan['grid']['windows'][0])
        other['id'] = 'r+003-c+005'; other['global_input_window'][1] += 2048
        plan['grid']['windows'].append(other); save(self.plan, plan)
        self.entry['plan']['sha256'] = sha256(self.plan)
        second = deepcopy(self.row); second['window'] = other['id']; second['output'] = 'elsewhere'
        self.entry['jobs'].append(second); self.persist()
        with self.assertRaisesRegex(ValueError, 'Duplicate window/job/output'):
            country_plan(self.index, 'AAA', 2024)
        second['job'] = dict(path='another-job.json', sha256='unused')
        second['output'] = './results/area'; self.persist()
        with self.assertRaisesRegex(ValueError, 'Duplicate window/job/output'):
            country_plan(self.index, 'AAA', 2024)

    def test_relocation_preserves_relative_selection(self):
        moved = self.root/'relocated'; moved.mkdir()
        self.job_root.rename(moved/'job'); self.index.rename(moved/'catalog.json')
        selected = resolve_window(country_plan(moved/'catalog.json', 'AAA', 2024), self.row['window'])
        self.assertEqual(selected['job'], moved/'job/job.json')

    def test_complete_filter_checks_outputs_and_no_model_is_dispatched(self):
        out = self.completed()
        self.row['completion_sha256'] = sha256(out/'integration-completion.json'); self.persist()
        self.assertEqual(listed_tasks(self.index, 'AAA', 2024)[0]['status'], 'completed')
        self.assertEqual(listed_tasks(self.index, 'AAA', 2024, pending_only=True), [])
        # File-only validation has no model mount for this fixture; omit model/source as a pair.
        args = select_arguments(self.args(output=out)); args.verify_inputs=False; args.skip_completed=True
        before = (out/'integration-completion.json').read_bytes()
        with patch('tasks.run_task.CarbonProcessor') as processor:
            self.assertEqual(run(args)['status'], 'skipped-verified-complete'); processor.assert_not_called()
        self.assertEqual(before, (out/'integration-completion.json').read_bytes())
        (out/'product/carbon.tif').write_bytes(b'corrupt')
        with self.assertRaises(ValueError): listed_tasks(self.index, 'AAA', 2024, pending_only=True)
        with self.assertRaises(ValueError): run(args)

    def test_stac_presence_and_changed_completion_pin_do_not_mean_complete(self):
        out = self.root/'results/area'
        save(out/'product/item.json', {'type': 'Feature'})
        self.assertEqual(listed_tasks(self.index, 'AAA', 2024)[0]['status'], 'incomplete')
        self.assertEqual(len(listed_tasks(self.index, 'AAA', 2024, pending_only=True)), 1)
        self.completed(); self.row['completion_sha256'] = 'changed'; self.persist()
        with self.assertRaises(ValueError): listed_tasks(self.index, 'AAA', 2024)

    def test_failed_and_incomplete_outputs_preserved_without_retry(self):
        out = self.root/'results/area'; out.mkdir(parents=True); marker=out/'partial'; marker.write_bytes(b'partial')
        for status in ['incomplete', 'failed']:
            if status == 'failed': save(out/'integration-failure.json', dict(error='retained'))
            self.assertEqual(listed_tasks(self.index, 'AAA', 2024)[0]['status'], status)
            args = select_arguments(self.args(output=out)); args.verify_inputs=False; args.skip_completed=True
            with patch('tasks.run_task.CarbonProcessor') as processor:
                with self.assertRaises(FileExistsError): run(args)
                processor.assert_not_called()
            self.assertEqual(marker.read_bytes(), b'partial')

    def test_completion_identity_missing_native_and_empty_stac_not_enough(self):
        out = self.completed(); p = out/'integration-completion.json'; record=read_json(p)
        record['job_sha256']='changed'; save(p,record)
        with self.assertRaisesRegex(ValueError, 'Completion job'): listed_tasks(self.index, 'AAA', 2024)
        record['job_sha256']=sha256(self.job); save(p,record)
        (out/'worker/native/carbon.tif').unlink()
        with self.assertRaisesRegex(ValueError, 'required output'): listed_tasks(self.index, 'AAA', 2024)



if __name__ == '__main__':
    unittest.main()
