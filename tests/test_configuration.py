"""Synthetic run configuration/identity tests; no Docker or scientific run."""
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

spec = importlib.util.spec_from_file_location("launcher_v3", HOST / "launch-country.py")
engine = importlib.util.module_from_spec(spec)
spec.loader.exec_module(engine)


class ConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get("SPC_TEST_TMP"))
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.bundle = self.root / "dependencies"
        self.bundle.mkdir()
        for name in ["model", "source"]:
            (self.bundle / name).mkdir()
        for name in ["model/weights", "model/training", "source/inference.py", "boundary.gpkg", "template.json"]:
            (self.bundle / name).write_text(name)
        def pin(name):
            return dict(path=name, sha256=op.sha(self.bundle / name))
        plan = {"grid": dict(anchor=[400000,8500000],crs='EPSG:32702',affine=[10,0,400000,0,-10,8500000]), "boundary": {"sha256": pin("boundary.gpkg")["sha256"]}}
        ownership = dict(schema='agc-ownership-grid-v1',anchor=[400000,8500000],crs='EPSG:32702',cell_size_m=5000,source='synthetic approved fixture')
        self.write(self.bundle / "plan.json", plan)
        self.write(self.bundle / "countries.json", dict(schema="agc-spatial-countries-v1", countries=[dict(
            country=dict(code="AAA"), ownership=ownership, plan=pin("plan.json"))]))
        self.manifest = dict(schema="agc-dependencies-v4", country="AAA", ownership=ownership,
                             catalog=pin("countries.json"), boundary=pin("boundary.gpkg"),
                             model_directory="model", inference_source_directory="source")
        self.write(self.bundle / "dependency-manifest.json", self.manifest)
        self.config = dict(schema="agc-country-run-v3", country="AAA", year=2024, geomad_version="0.4.0",
            image="processor:review", dependency_directory="dependencies", working_directory="work", output_directory="products/country-v1",
            log_directory="logs", archive_directory="incoming", allow_acquisition=True,
            limits=dict(minutes=30, model_attempts=19, source_transfer_gib=3, minimum_free_gib=1, free_space_directory=str(self.root)))
        self.file = self.root / "example-2024.json"
        self.write(self.file, self.config)
        external = dict(model=dict(files={n:dict(sha256=op.sha(self.bundle / "model" / n)) for n in ["weights", "training"]}),
                        inference_source=dict(files={"inference.py":op.sha(self.bundle / "source/inference.py")}))
        self.metadata = patch.object(op, "package_metadata", return_value=(dict(processor_image="sha256:" + "b"*64),
            external, dict(geomad_version="0.4.0", template_in_image="/opt/processor/examples/internal.json"),
            self.bundle / "template.json"))
        self.metadata.start()
        self.addCleanup(self.metadata.stop)

    def write(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def resolve(self):
        return op.resolve(self.file, HOST)

    def test_fresh_run_resolves_without_any_processing_outputs(self):
        c = self.resolve()
        self.assertEqual(c["country_root"], self.root / "work")
        self.assertFalse((c["country_root"] / "results").exists())
        prep = op.read(c["preparation"]["file"])
        self.assertEqual(prep["boundary"]["path"], "/plans/boundary.gpkg")
        self.assertNotIn("native_seed", prep)
        self.assertNotIn("static_seed_job", prep)
        self.assertNotIn("checkpoint", c["bindings"])
        self.assertEqual(c["limits"]["seconds"], 1800)
        self.assertEqual(prep["limits"]["transfer_bytes"], 3*2**30)

    def test_same_dependencies_resolve_multiple_years_without_annual_bundles(self):
        manifest_pin = op.sha(self.bundle / 'dependency-manifest.json')
        for year in [2022, 2025]:
            c = dict(self.config, year=year, working_directory=f'work/{year}',
                     log_directory=f'logs/{year}', output_directory=f'products/{year}')
            self.write(self.file, c)
            resolved = self.resolve()
            self.assertEqual(resolved['year'], year)
            self.assertEqual(op.read(resolved['preparation']['file'])['year'], year)
            self.assertEqual(op.sha(self.bundle / 'dependency-manifest.json'), manifest_pin)

    def test_approved_geometry_metadata_is_required_and_pinned(self):
        projected = self.bundle / 'approved-projected.geojson'
        projected.write_text('synthetic hash-only dependency fixture')
        pin = dict(path=projected.name, sha256=op.sha(projected), crs='EPSG:3832')
        check = dict(accepted_projected=pin)
        self.write(self.bundle / 'check.json', check)
        index = op.read(self.bundle / 'countries.json')
        index['countries'][0].update(accepted_projected=pin,
            planning_check=dict(path='check.json', sha256=op.sha(self.bundle / 'check.json')))
        self.write(self.bundle / 'countries.json', index)
        self.manifest['catalog']['sha256'] = op.sha(self.bundle / 'countries.json')
        self.manifest['accepted_projected'] = pin
        self.write(self.bundle / 'dependency-manifest.json', self.manifest)
        c = self.resolve()
        self.assertEqual(op.read(c['preparation']['file'])['accepted_projected']['path'], 'approved-projected.geojson')
        self.assertEqual(op.read(c['preparation']['file'])['dependency_root'], '/plans')
        del self.manifest['accepted_projected']
        self.write(self.bundle / 'dependency-manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'geometry binding'):
            self.resolve()

    def test_country_and_year_only_from_config(self):
        c = engine.Launcher(self.file, resolved=self.resolve())
        self.assertEqual((c.country, c.year), ("AAA", 2024))
        self.assertEqual(c.c["image"], "processor:review")

    def test_rejects_unsupported_version_year_and_unknown_fields(self):
        for key, value in [("year", True), ("geomad_version", "0.5.0"), ("extra", "ignored")]:
            with self.subTest(key=key):
                changed = dict(self.config, **{key:value})
                self.write(self.file, changed)
                with self.assertRaises(ValueError):
                    self.resolve()

    def test_bad_units_and_offline_transfer_fail(self):
        for change in [dict(minutes=0), dict(source_transfer_gib=-1), dict(minimum_free_gib=True)]:
            self.write(self.file, dict(self.config, limits=dict(self.config["limits"], **change)))
            with self.assertRaises(ValueError):
                self.resolve()
        self.write(self.file, dict(self.config, allow_acquisition=False))
        with self.assertRaisesRegex(ValueError, "zero source transfer"):
            self.resolve()

    def test_changed_or_missing_dependencies_do_not_fallback(self):
        (self.bundle / "boundary.gpkg").write_text("changed")
        with self.assertRaisesRegex(ValueError, "Missing or changed"):
            self.resolve()

    def test_model_source_checks_are_automatic(self):
        (self.bundle / "source/inference.py").unlink()
        with self.assertRaisesRegex(ValueError, "Missing or changed"):
            self.resolve()

    def test_working_storage_cannot_contain_a_supplied_model(self):
        self.config['working_directory'] = str(self.bundle / 'model')
        self.write(self.file, self.config)
        with self.assertRaisesRegex(ValueError, 'overlaps supplied dependencies'):
            self.resolve()

    def test_manifest_country_and_boundary_must_match(self):
        self.write(self.bundle / "dependency-manifest.json", dict(self.manifest, country="BBB"))
        with self.assertRaisesRegex(ValueError, "does not match"):
            self.resolve()

    def test_readable_image_resolves_and_is_journaled_or_rejected(self):
        obj = engine.Launcher(self.file, resolved=self.resolve())
        obj.state = {}
        def command(stage, argv, env=None):
            self.assertEqual(argv[-1], "processor:review")
            return "sha256:" + "c"*64
        obj.command = command
        with self.assertRaisesRegex(ValueError, "image identity differs"):
            obj.stages()
        obj.command = lambda *a, **k: "sha256:" + "b"*64
        obj.docker = lambda *a, **k: (_ for _ in ()).throw(RuntimeError("stop after resolution"))
        with self.assertRaisesRegex(RuntimeError, "stop after resolution"):
            obj.stages()
        self.assertEqual(obj.state["resolved_image_id"], "sha256:" + "b"*64)
        self.assertEqual(obj.state["image_reference"], "processor:review")

    def binding_module(self):
        spec = importlib.util.spec_from_file_location('bind_dependencies', HOST/'bind-dependencies.py')
        binding = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(binding)
        return binding

    def supplied_geometry(self):
        # Exact retained layout: the certificate path is relative to a separate
        # metadata root; neither a CODE.geojson alias nor a catalogue edit exists.
        metadata_root = self.root / 'supplied-metadata'
        geometry = metadata_root / 'GEODATA/VECTOR/approved/AAA-accepted-projected-country.geojson'
        self.write(geometry, dict(type='FeatureCollection',crs=dict(properties=dict(name='EPSG:32702')),
            features=[dict(type='Feature',geometry=dict(type='MultiPolygon',coordinates=[[[[0,0],[1,0],[1,1],[0,1],[0,0]]]]))]))
        pin = dict(path=geometry.relative_to(metadata_root).as_posix(),
                   sha256=op.sha(geometry), crs='EPSG:32702')
        return metadata_root, geometry, pin

    def test_dependency_setup_emits_the_preparation_geometry_layout(self):
        binding = self.binding_module()
        metadata_root, geometry, pin = self.supplied_geometry()
        certificate=self.bundle/'check.json'
        self.write(certificate,dict(passed=True,country=dict(code='AAA'),
            plan_sha256=op.sha(self.bundle/'plan.json'),accepted_projected=pin))
        catalog=op.read(self.bundle/'countries.json')
        entry=catalog['countries'][0]
        entry.update(plan_status='checked',planning_check=dict(path=certificate.name,sha256=op.sha(certificate)))
        self.write(self.bundle/'countries.json',catalog)
        destination=self.root/'dependencies-issued/AAA'
        with patch.object(binding,'package_metadata',side_effect=op.package_metadata), patch.object(binding,'roster',return_value=[entry]):
            binding.bind(destination,self.bundle/'countries.json','AAA',self.bundle/'boundary.gpkg',
                         self.bundle/'model',self.bundle/'source',metadata_root)
        manifest=op.read(destination/'dependency-manifest.json')
        self.assertEqual(manifest['accepted_projected']['path'],'approved-projected.geojson')
        self.assertEqual((destination/'approved-projected.geojson').read_bytes(),geometry.read_bytes())
        self.assertEqual(manifest['ownership'],entry['ownership'])
        from tasks.inputs.dependencies import dependency_file
        self.assertEqual(dependency_file(destination,manifest['accepted_projected']),destination/'approved-projected.geojson')

    def test_certificate_locator_requires_root_file_checksum_and_matching_crs(self):
        binding = self.binding_module()
        metadata_root, geometry, pin = self.supplied_geometry()
        self.assertEqual(binding.approved_projected({}, pin, self.bundle, metadata_root,
                                                   'EPSG:32702'), geometry)
        with self.assertRaisesRegex(ValueError, 'explicit --metadata-root'):
            binding.approved_projected({}, pin, self.bundle, None, 'EPSG:32702')
        cases = [dict(pin, path='missing.geojson'), dict(pin, sha256='0'*64),
                 dict(pin, path='../outside.geojson'), dict(pin, path=str(geometry.absolute())),
                 dict(pin, path='C:\\escape.geojson'), dict(pin, crs='EPSG:3832'),
                 {k: v for k, v in pin.items() if k != 'path'}]
        for invalid in cases:
            with self.subTest(pin=invalid), self.assertRaises(ValueError):
                binding.approved_projected({}, invalid, self.bundle, metadata_root, 'EPSG:32702')
        obj = op.read(geometry)
        obj['crs']['properties']['name'] = 'EPSG:3832'
        self.write(geometry, obj)
        with self.assertRaisesRegex(ValueError, 'geometry CRS differs'):
            binding.approved_projected({}, dict(pin, sha256=op.sha(geometry)),
                                       self.bundle, metadata_root, 'EPSG:32702')

    def test_certificate_and_catalogue_locators_must_identify_the_same_file(self):
        binding = self.binding_module()
        metadata_root, geometry, pin = self.supplied_geometry()
        self.assertEqual(binding.approved_projected(dict(accepted_projected=pin), pin,
            metadata_root, metadata_root, 'EPSG:32702'), geometry)
        for change in [dict(sha256='0'*64), dict(crs='EPSG:3832')]:
            with self.subTest(change=change), self.assertRaisesRegex(ValueError, 'identities differ'):
                binding.approved_projected(dict(accepted_projected=dict(pin, **change)),
                    pin, metadata_root, metadata_root, 'EPSG:32702')
        other = metadata_root / 'other.geojson'
        other.write_bytes(geometry.read_bytes())
        with self.assertRaisesRegex(ValueError, 'Ambiguous'):
            binding.approved_projected(dict(accepted_projected=dict(pin, path=other.name)),
                pin, metadata_root, metadata_root, 'EPSG:32702')

    def test_certificate_locator_rejects_symlink_file_directory_and_root(self):
        binding = self.binding_module()
        metadata_root, geometry, pin = self.supplied_geometry()
        leaf = metadata_root / 'linked.geojson'
        parent = metadata_root / 'linked-directory'
        root_link = self.root / 'linked-root'
        try:
            leaf.symlink_to(geometry)
            parent.symlink_to(geometry.parent, target_is_directory=True)
            root_link.symlink_to(metadata_root, target_is_directory=True)
        except OSError as exc:
            self.skipTest('Symlink fixture requires filesystem support: ' + str(exc))
        for root, reference in [(metadata_root, dict(pin, path=leaf.name)),
                (metadata_root, dict(pin, path=parent.name + '/' + geometry.name)), (root_link, pin)]:
            with self.subTest(root=root, reference=reference), self.assertRaisesRegex(ValueError, 'Symlink'):
                binding.approved_projected({}, reference, metadata_root, root, 'EPSG:32702')

    def test_missing_ownership_stops_before_working_roots_are_created(self):
        self.manifest.pop('ownership')
        self.write(self.bundle / 'dependency-manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'ownership metadata'):
            self.resolve()
        self.assertFalse((self.root / 'work').exists())

    def test_ownership_cannot_override_approved_plan_origin(self):
        self.manifest['ownership']['anchor'] = [400010, 8500000]
        catalog = op.read(self.bundle / 'countries.json')
        catalog['countries'][0]['ownership'] = self.manifest['ownership']
        self.write(self.bundle / 'countries.json', catalog)
        self.manifest['catalog']['sha256'] = op.sha(self.bundle / 'countries.json')
        self.write(self.bundle / 'dependency-manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'origin differs'):
            self.resolve()
        self.assertFalse((self.root / 'work').exists())

    def test_historical_checkpoint_bindings_are_not_runtime_inputs(self):
        self.manifest['checkpoint'] = {'directory': 'old-experiment'}
        self.write(self.bundle / 'dependency-manifest.json', self.manifest)
        with self.assertRaisesRegex(ValueError, 'not supported'):
            self.resolve()


if __name__ == "__main__":
    unittest.main()
