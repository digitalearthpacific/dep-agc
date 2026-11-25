"""Vector/grid fixtures only; no source pixels, owner arrays, weights or inference."""

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest

from shapely.geometry import MultiPolygon, box
from shapely.geometry import mapping
from shapely.ops import transform, unary_union
from pyproj import Transformer

from tasks.inputs.geometry import grid_plan, owners, pacific_tile_plan, projected_boundary, provider_tiles
from tasks.inputs.raster import receiving_source_bounds
from tasks.inputs.regional import new_plan, check_plan, owner_extent, owner_capacity, accepted_geometry, run
from tasks.catalog import country_plan, listed_tasks, resolve_window
from carbon.io import sha256
from carbon.grid import validate_grid

RULE = dict(crs='EPSG:3832', anchor=[0, 0], resolution=10, patch=512,
            stride=64, context=224, retained_chunk=2048)
COUNTRY = dict(code='TST', name='Synthetic portability fixture')


class PlanningTests(unittest.TestCase):
    def test_official_tile_partition_and_edge_context(self):
        grid = pacific_tile_plan('075_025')
        tile = box(4200000, -1600000, 4296000, -1504000)
        cores = [box(*w['bounds']) for w in grid['windows']]
        self.assertEqual(len(cores), 25)
        self.assertTrue(unary_union(cores).equals(tile))
        self.assertEqual(sum(c.area for c in cores), tile.area)
        for w, core in zip(grid['windows'], cores):
            self.assertEqual(w['retained_shape'], [1920, 1920])
            self.assertEqual(core.area, 19200**2)
            self.assertEqual(w['global_input_window'][2:], [2368, 2368])
            self.assertEqual(tuple(w['input_bounds']), receiving_source_bounds({'grid': grid}, w))
            self.assertTrue(box(*w['input_bounds']).equals(core.buffer(2240, join_style='mitre')))
        self.assertEqual(set(provider_tiles(None, box(*grid['windows'][0]['input_bounds']))),
                         {(74, 25), (74, 26), (75, 25), (75, 26)})
        self.assertEqual(set(provider_tiles(None, box(*grid['windows'][-1]['input_bounds']))),
                         {(75, 24), (75, 25), (76, 24), (76, 25)})

    def test_worker_accepts_equal_edge_core_and_rejects_wrong_context(self):
        grid = pacific_tile_plan('075_025')
        w = grid['windows'][-1]
        g = dict(grid, receiving_window=w['global_input_window'][:],
                 global_input_window=w['global_input_window'][:], canonical_global_offset=[0, 0],
                 retained_shape=w['retained_shape'], bounds=w['bounds'])
        task = dict(schema='spc-agb-window-v3', country=COUNTRY, window=w['id'], grid=g)
        self.assertEqual(validate_grid(task).to_epsg(), 3832)
        task['grid']['receiving_window'][2] -= 1
        with self.assertRaisesRegex(ValueError, 'Full receiving'):
            validate_grid(task)

    def test_sparse_candidates_and_extent_proxy_match_existing_planner(self):
        country = MultiPolygon([box(11, 17, 11000, 12000), box(51001, 75001, 51003, 75003)])
        rows, _ = owners(country, [0, 0])
        expected = grid_plan(country, rows, [0, 0])
        actual = grid_plan(country, [dict(bounds=owner_extent(country, [0, 0]))],
                           [0, 0], [g.bounds for g in country.geoms])
        self.assertEqual(actual, expected)

    def test_dateline_continuity_and_components(self):
        geographic = MultiPolygon([box(179.8, -2.1, 179.99, -2), box(-179.99, -2.1, -179.8, -2)])
        projected = projected_boundary(geographic, 'EPSG:3832')
        self.assertLess(projected.bounds[2]-projected.bounds[0], 50000)
        plan = new_plan(COUNTRY, geographic, RULE, lambda: None)
        result = check_plan(plan, geographic, lambda: None)
        self.assertEqual(len(result['components']), 2)
        self.assertEqual(result['uncovered_m2'], 0)
        self.assertEqual(result['maximum_retained_overlap_m2'], 0)

    def test_small_component_kept_and_repeat_plan_stable(self):
        geographic = MultiPolygon([box(170, -10, 170.01, -9.99),
                                   box(170.03, -10, 170.030001, -9.999999)])
        a = new_plan(COUNTRY, geographic, RULE, lambda: None)
        b = new_plan(COUNTRY, geographic, RULE, lambda: None)
        self.assertEqual(a, b)
        checked = check_plan(a, geographic, lambda: None)
        self.assertTrue(all(c['retained_windows'] for c in checked['components']))
        self.assertEqual(len(checked['components']), 2)

    def test_projection_bounds_phase_and_window_contract_rejections(self):
        with self.assertRaisesRegex(ValueError, 'outside'):
            projected_boundary(MultiPolygon([box(0, 0, .1, .1)]), 'EPSG:3832')
        geographic = MultiPolygon([box(170, -10, 170.01, -9.99)])
        plan = new_plan(COUNTRY, geographic, RULE, lambda: None)
        broken = deepcopy(plan); broken['grid']['windows'][0]['global_input_window'][0] += 1
        with self.assertRaises(ValueError): check_plan(broken, geographic, lambda: None)
        broken = deepcopy(plan); broken['grid']['windows'][0]['bounds'][0] += 10
        with self.assertRaisesRegex(ValueError, 'bounds/grid'):
            check_plan(broken, geographic, lambda: None)
        broken = deepcopy(plan); broken['grid']['shape'][0] = 2**31
        with self.assertRaisesRegex(ValueError, 'dimension'):
            check_plan(broken, geographic, lambda: None)

    def test_uint16_owner_capacity_guard_without_owner_generation(self):
        geographic = MultiPolygon([box(1, 1, 1_299_999, 1_299_999)])
        with self.assertRaisesRegex(ValueError, 'uint16'):
            owner_capacity(geographic, [0, 0], lambda: None)

    def test_blocked_plan_is_visible_and_cannot_execute(self):
        with tempfile.TemporaryDirectory(dir=os.environ.get('SPC_TEST_TMP')) as temporary:
            root = Path(temporary)
            plan = root/'blocked.json'
            plan.write_text(json.dumps(dict(country=COUNTRY, year=2024, source_version='0.4.0',
                planning_status='blocked', grid=None, reason='Unsupported spatial representation')))
            catalog = root/'index.json'
            catalog.write_text(json.dumps(dict(schema='agc-prepared-countries-v2', countries=[
                dict(country=COUNTRY, year=2024, source_version='0.4.0', jobs=[],
                     plan_status='blocked', plan=dict(path=plan.name, sha256=sha256(plan)))])))
            rows = listed_tasks(catalog, 'TST', 2024)
            self.assertEqual(rows[0]['status'], 'plan-blocked')
            self.assertIsNone(rows[0]['window'])
            with self.assertRaisesRegex(ValueError, 'plan blocked'):
                resolve_window(country_plan(catalog, 'TST', 2024), 'not-invented')

    def test_roster_is_exact_engineering_scope_with_pinned_sources(self):
        path = Path(__file__).resolve().parents[1]/'configs/regional-planning.json'
        obj = json.loads(path.read_text())
        expected = {r['country']['code'] for r in json.loads((path.parent/'countries.json').read_text())['countries']}
        self.assertEqual({c['code'] for c in obj['countries']}, expected)
        self.assertEqual(len(obj['countries']), 22)
        self.assertEqual(obj['roster_sources'][0]['commit'], '0305f104d1fe9d408d02602f4b6529d17602805a')
        self.assertEqual(obj['roster_sources'][1]['commit'], 'b4c6a6f1ef0cc2c6ede0a2af8aa3fad81324a63e')

    def test_source_contract_and_escaping_output_fail_before_planning(self):
        path=Path(__file__).resolve().parents[1]/'configs/regional-planning.json'
        config=json.loads(path.read_text())
        with tempfile.TemporaryDirectory(dir=os.environ.get('SPC_TEST_TMP')) as temporary:
            root=Path(temporary);(root/'GEODATA').mkdir()
            for key,value in [('source_version','0.5.0'),('source_product','Other')]:
                changed=deepcopy(config);changed[key]=value
                with self.assertRaisesRegex(ValueError,'Pinned'):
                    run(changed,{},root/'none',root,'not-created')
            for prefix in ['../escape','C:\\escape']:
                with self.assertRaisesRegex(ValueError,'Relative typed'):
                    run(config,{},root/'none',root,prefix)
            self.assertEqual(list((root/'GEODATA').iterdir()),[])

    def test_reused_projected_seam_boundary_is_read_verbatim(self):
        geographic = MultiPolygon([box(179.9, -17, 180, -16.9),
                                   box(-180, -17, -179.9, -16.9)])
        raw = transform(Transformer.from_crs(4326, 32760, always_xy=True).transform, geographic)
        accepted = unary_union(raw)  # Synthetic pre-existing seam-joined fixture only.
        with tempfile.TemporaryDirectory(dir=os.environ.get('SPC_TEST_TMP')) as temporary:
            root = Path(temporary); path=root/'accepted.json'
            path.write_text(json.dumps(dict(type='FeatureCollection',
                crs=dict(properties=dict(name='EPSG:32760')),
                features=[dict(type='Feature', geometry=mapping(accepted))])))
            pin=dict(path=path.name, sha256=sha256(path));before=path.read_bytes()
            loaded = accepted_geometry(root, pin, 'EPSG:32760')
            self.assertEqual(loaded.wkb, accepted.wkb)
            lattice=grid_plan(loaded,[dict(bounds=owner_extent(loaded,[0,0]))],[0,0])
            plan=dict(country=COUNTRY,year=2024,source_version='0.4.0',grid=dict(RULE,**lattice,
                crs='EPSG:32760',id='accepted-test-grid',canonical_global_offset=[0,0]))
            self.assertTrue(check_plan(plan,geographic,lambda:None,loaded)['passed'])
            self.assertEqual(path.read_bytes(),before)
            with self.assertRaises(ValueError):accepted_geometry(root,pin,'EPSG:3832')
            with self.assertRaises(ValueError):accepted_geometry(root,dict(pin,sha256='wrong'),'EPSG:32760')

    def test_checked_plan_requires_matching_pinned_certificate(self):
        plan = new_plan(COUNTRY, MultiPolygon([box(170, -10, 170.01, -9.99)]), RULE, lambda: None)
        with tempfile.TemporaryDirectory(dir=os.environ.get('SPC_TEST_TMP')) as temporary:
            root=Path(temporary);p=root/'plan.json';p.write_text(json.dumps(plan))
            row=dict(country=COUNTRY,year=2024,source_version='0.4.0',jobs=[],plan_status='checked',
                     plan=dict(path=p.name,sha256=sha256(p)))
            index=root/'index.json'
            def save():index.write_text(json.dumps(dict(schema='agc-prepared-countries-v2',countries=[row])))
            save()
            with self.assertRaisesRegex(ValueError,'pinned planning check'):country_plan(index,'TST',2024)
            certificate=root/'checked.json';certificate.write_text(json.dumps(dict(passed=True,
                country=COUNTRY,plan_sha256=sha256(p))))
            row['planning_check']=dict(path=certificate.name,sha256=sha256(certificate));save()
            self.assertTrue(all(r['plan_status']=='checked' for r in listed_tasks(index,'TST',2024)))
            certificate.write_text(json.dumps(dict(passed=True,country=COUNTRY,plan_sha256='wrong')))
            row['planning_check']['sha256']=sha256(certificate);save()
            with self.assertRaisesRegex(ValueError,'check/plan identity'):country_plan(index,'TST',2024)


if __name__ == '__main__':
    unittest.main()
