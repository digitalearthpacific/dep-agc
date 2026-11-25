"""Operational contract tests, independent of model execution and saved rasters."""

import ast
from argparse import Namespace
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest

from tasks.job import resolve_job
from tasks.print_tasks import tasks
from tasks.run_task import run
from carbon.io import read_json, sha256, write_json
import tasks.inputs.canonical as canonical


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('SPC_TEST_TMP'))
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def inputs(self):
        task = read_json(Path(__file__).resolve().parents[1]/'configs/scientific-profile.json')
        task.update(schema='spc-agb-window-v3', year=2024, country=dict(code='AAA', name='Example A'), window='r+000-c+000',
            grid=dict(anchor=[400000,8500000],id='synthetic-grid', crs='EPSG:32702', affine=[10,0,400000,0,-10,8500000],
                shape=[2496,2496], receiving_window=[0,0,2496,2496], global_input_window=[0,0,2496,2496],
                canonical_global_offset=[0,0], context=224, patch=512, stride=64,
                retained_shape=[2048,2048], bounds=[402240,8477280,422720,8497760]),
            inputs=dict(canonical=dict(path='inputs/canonical-selected.vrt',sha256='fixture'),
                        blocks=[],owner_order=dict(path='inputs/owner-ids.json'), owners=[]))
        static = self.root/'provided-static'
        # Minimal static payloads are sufficient for pin/closure tests. Scientific
        # array contents are exercised by the separate real-input preflight.
        for row in [task['inputs']['owner_order'], *(o['file'] for o in task['inputs']['owners'])]:
            p = static/row['path']; p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(row['path'].encode()); row['sha256'] = sha256(p)
        task_file = self.root/'task.json'; write_json(task_file, task)
        cache = self.root/'cache'; cache.mkdir()
        payload = cache/'band'; payload.write_bytes(b'pinned native bytes')
        support = cache/'support.json'; support.write_text('{}')
        row = dict(path='band', sha256=sha256(payload))
        index = dict(year=2024,version='0.4.0',records=[dict(bands={
            b:dict(native=row,valid=row) for b in ['red','green','blue','nir','count']})],
            source_items=[],support=dict(path='support.json',sha256=sha256(support)))
        index_file = cache/'index.json'; write_json(index_file,index)
        plan = dict(country=task['country'],year=2024,source_version='0.4.0',
                    grid=dict(task['grid'],windows=[dict(id=task['window'],
                              global_input_window=task['grid']['global_input_window'])]))
        plan_file = self.root/'plan.json'; write_json(plan_file,plan)
        return task_file,index_file,static,plan_file

    def staged(self):
        import shutil
        task,index,static,plan = self.inputs()
        root = self.root/'job'; root.mkdir()
        shutil.copytree(static,root/'static');shutil.copytree(index.parent,root/'geomad')
        shutil.copyfile(task,root/'task.json');shutil.copyfile(plan,root/'plan.json')
        write_json(root/'job.json',dict(schema='agc-geomad-job-v2',
            task=dict(path='task.json',sha256=sha256(root/'task.json')),
            native_index=dict(path='geomad/index.json',sha256=sha256(root/'geomad/index.json')),
            plan=dict(path='plan.json',sha256=sha256(root/'plan.json')),static_root='static'))
        return root

    def test_corrupt_static_or_source_pin_stops_before_processing(self):
        root = self.staged()
        t = read_json(root/'task.json')
        (root/'static'/t['inputs']['owner_order']['path']).write_bytes(b'corrupt')
        with self.assertRaisesRegex(ValueError,'changed input'):
            resolve_job(root/'job.json',2024,'0.4.0')

    def test_plan_grid_and_explicit_source_must_match(self):
        root = self.staged()
        for year,version in [(2023,'0.4.0'),(2024,'0.5.0')]:
            with self.assertRaises(ValueError):resolve_job(root/'job.json',year,version)
        plan = read_json(root/'plan.json'); plan['grid']['windows'][0]['global_input_window'][0] += 64
        p = root/'changed-plan.json'; write_json(p,plan)
        job = read_json(root/'job.json'); job['plan'] = dict(path=p.name,sha256=sha256(p))
        j = root/'changed-job.json'; write_json(j,job)
        with self.assertRaisesRegex(ValueError,'configured plan window'):
            resolve_job(j,2024,'0.4.0')

    def test_escaping_binding_is_rejected(self):
        root = self.staged(); job = read_json(root/'job.json')
        job['native_index']['path'] = '../cache/index.json'
        p = root/'escaping.json'; write_json(p,job)
        with self.assertRaisesRegex(ValueError,'escapes root'):
            resolve_job(p,2024,'0.4.0')

    def test_failure_record_and_existing_output_preservation(self):
        job = self.root/'bad-job.json'; job.write_text('{}')
        output = self.root/'failed'
        args = Namespace(output=output,job=job,year=2024,source_version='0.4.0')
        with self.assertRaises(ValueError):run(args)
        failure = output/'integration-failure.json'
        self.assertIn('agc-geomad-job-v2',read_json(failure)['error'])
        before = failure.read_bytes()
        with self.assertRaises(FileExistsError):run(args)
        self.assertEqual(before,failure.read_bytes())

    def test_listing_does_not_claim_prepared_or_available_coverage(self):
        _,_,_,p = self.inputs(); plan = read_json(p)
        result = tasks(plan,'AAA',2024,'r+000-c+000')
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['input_status'],'requires pinned job inputs')
        with self.assertRaises(ValueError):tasks(plan,'CCC',2024)
        with self.assertRaises(ValueError):tasks(plan,'AAA',2024,'missing')

    def test_extracted_canonical_operations_equal_accepted_54cfce2(self):
        # Frozen LF function-source fingerprints, independent of Python's AST
        # schema version. Processing imports ordinary Python modules.
        expected = {'pin':'1e4a72c748921b82704dcf1b81d4916e94e503f50183c2edb75e781b103a9a76',
            'canonical_vrt':'9083869b38eacdec06d2c660261e99387c1ad24211f6bc3540ce7f684a77cfa9',
            'canonical':'3352360a5661232b0303ba78193e62abee68e9021d82bc2a7bfe341689426bb5'}
        source = Path(canonical.__file__).read_text()
        tree = ast.parse(source)
        for node in tree.body:
            if isinstance(node,ast.FunctionDef) and node.name in expected:
                text = '\n'.join(source.splitlines()[node.lineno-1:node.end_lineno])
                self.assertEqual(hashlib.sha256(text.encode()).hexdigest(),
                                 expected.pop(node.name))
        self.assertEqual(expected,{})


if __name__ == '__main__':
    unittest.main()
