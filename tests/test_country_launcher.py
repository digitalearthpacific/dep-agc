"""Host launcher contract tests, no Docker, GIS libraries, acquisition or model."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch


FILE = Path(__file__).resolve().parents[1] / 'launch-country.py'
spec = importlib.util.spec_from_file_location("country_launcher", FILE)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class Fake(launcher.Launcher):
    pending = False
    fail = None
    incomplete = False
    bad_assembly = False

    def command(self, stage, argv, env=None):
        self.guard()
        self.seen.append((stage, argv, env))
        if stage == self.fail:
            raise RuntimeError("controlled failure: " + stage)
        if stage == "image":
            return self.c["image"] + "\n"
        if stage in ["list", "verify-jobs"]:
            status = "configured" if (self.pending and stage == "list") or self.incomplete else "completed"
            return json.dumps([dict(country=self.country, year=self.year, window="r+000-c+000", status=status)])
        if stage == "assemble":
            root = self.c["assembly"]
            root.mkdir()
            launcher.save(root / "country-summary.json", dict(complete=not self.bad_assembly,
                summary=dict(stock_tC=123.), country=dict(code=self.country), year=self.year,
                windows=[dict(window="r+000-c+000", status="complete")]))
            launcher.save(root / "collection.json", {"agc:complete": not self.bad_assembly})
        return ""


class LauncherTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=os.environ.get("SPC_TEST_TMP"))
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        for n in ["plans", "configuration", "data", "archive", "model", "source"]:
            (self.root / n).mkdir()
        launcher.save(self.root / "plans/countries.json", {})
        launcher.save(self.root / "configuration/preparation.json", {"task_template": {"sha256": "a" * 64}})
        self.config = dict(schema="resolved-test-fixture", image="sha256:" + "b" * 64,
            catalog=dict(root=str(self.root / "plans"), path="countries.json"),
            preparation=dict(root=str(self.root / "configuration"), path="preparation.json"),
            image_task_template="/opt/processor/examples/scientific-profile.json",
            preparation_network="none", country_root=str(self.root / "data"),
            archive_root=str(self.root / "archive"), model=str(self.root / "model"), source=str(self.root / "source"),
            assembly="assembly/product-v1", run_root=str(self.root / "logs"),
            limits=dict(seconds=1800, model_attempts=1, free_space_path=str(self.root), minimum_free_bytes=1))
        self.path = self.root / "launch.json"
        launcher.save(self.path, self.config)
        self.result = self.root / "data/results/r+000-c+000/integration-completion.json"
        self.result.parent.mkdir(parents=True)
        launcher.save(self.result, {"pinned": "completion"})

    def instance(self, **attrs):
        import copy
        resolved = copy.deepcopy(self.config)
        resolved.update(schema='agc-country-launch-resolved-v3', country='AAA', year=2024)
        for key in ['country_root','archive_root','run_root','model','source']:
            resolved[key] = Path(resolved[key])
        for key in ['catalog','preparation']:
            resolved[key]['root']=Path(resolved[key]['root'])
            resolved[key]['file']=resolved[key]['root']/resolved[key]['path']
        relative = Path(resolved['assembly'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe assembly path')
        resolved['assembly']=resolved['country_root']/relative
        resolved['limits']['free_space_path']=Path(resolved['limits']['free_space_path'])
        resolved['scientific_profile']=self.root/'configuration/profile.json'
        obj = Fake(self.path, resolved=resolved)
        obj.seen = []
        for k, v in attrs.items():
            setattr(obj, k, v)
        return obj

    def test_order_paths_cpu_gpu_boundary_and_result(self):
        obj = self.instance(pending=True)
        result = obj.run()
        self.assertEqual([s for s, _, _ in obj.seen],
            ["image", "preflight", "prepare", "bind", "list", "process", "verify-jobs", "assemble"])
        for stage, argv, env in obj.seen:
            if stage != "process":
                self.assertNotIn("--gpus", argv)
            if stage == "process":
                self.assertEqual(argv[2:5], [self.config["image"], "AAA", "2024"])
                self.assertEqual(env, {"AGC_MAX_JOBS": "1"})
            if stage == "prepare":
                self.assertEqual(argv[argv.index("--network") + 1], "none")
                self.assertIn("/plans/countries.json", argv)
                self.assertIn("/config/preparation.json", argv)
            if stage in ["preflight", "prepare"]:
                self.assertIn(
                    "type=bind,src=" + str(obj.c['scientific_profile'].parent) +
                    ",dst=/opt/processor/examples,readonly", argv)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["new_model_requests"], 0)

    def test_every_stage_failure_stops_later_commands(self):
        for stage in ["image", "preflight", "prepare", "bind", "list", "process", "verify-jobs", "assemble"]:
            with self.subTest(stage=stage):
                obj = self.instance(fail=stage)
                with self.assertRaisesRegex(RuntimeError, "controlled failure"):
                    obj.run()
                self.assertEqual(obj.seen[-1][0], stage)
                self.assertEqual(launcher.read(obj.state_file)["status"], "failed")

    def test_incomplete_jobs_never_reach_assembly(self):
        obj = self.instance(incomplete=True)
        with self.assertRaisesRegex(RuntimeError, "Incomplete country"):
            obj.run()
        self.assertNotIn("assemble", [s for s, _, _ in obj.seen])

    def test_failed_or_uncommitted_assembly_is_not_overwritten(self):
        obj = self.instance(bad_assembly=True)
        with self.assertRaisesRegex(RuntimeError, "Incomplete country output"):
            obj.run()
        files = {p: p.read_bytes() for p in obj.c["assembly"].glob("*")}
        with self.assertRaises(FileExistsError):
            self.instance().run()
        self.assertTrue(all(p.read_bytes() == v for p, v in files.items()))

    def test_verified_relaunch_preserves_deadline_and_product(self):
        obj = self.instance()
        first = obj.run()
        state = launcher.read(obj.state_file)
        second = self.instance()
        result = second.run()
        self.assertEqual(result["product"], first["product"])
        self.assertEqual(result["assembly_status"], "reused-verified")
        self.assertNotIn("assemble", [s for s, _, _ in second.seen])
        self.assertEqual(launcher.read(obj.state_file)["deadline_epoch"], state["deadline_epoch"])

    def test_changed_assembly_or_completion_refuses_reuse(self):
        obj = self.instance()
        obj.run()
        self.result.write_text("changed")
        with self.assertRaises(FileExistsError):
            self.instance().run()

    def test_expired_deadline_cannot_be_reset(self):
        obj = self.instance(fail="image")
        with self.assertRaises(RuntimeError):
            obj.run()
        s = launcher.read(obj.state_file)
        s["deadline_epoch"] = time.time() - 1
        launcher.save(obj.state_file, s)
        with self.assertRaisesRegex(RuntimeError, "deadline exhausted"):
            self.instance().run()

    def test_zero_model_allowance_stops_pending_jobs(self):
        self.config["limits"]["model_attempts"] = 0
        launcher.save(self.path, self.config)
        obj = self.instance(pending=True)
        with self.assertRaisesRegex(RuntimeError, "remaining explicit model allowance"):
            obj.run()
        self.assertNotIn("process", [s for s, _, _ in obj.seen])

    def test_pending_count_is_derived_once_and_retained(self):
        obj = self.instance(pending=True)
        obj.c['limits']['model_attempts'] = None
        obj.run()
        self.assertEqual(launcher.read(obj.state_file)['model_attempt_limit'], 1)
        second = self.instance(pending=False)
        second.c['limits']['model_attempts'] = None
        second.run()
        self.assertEqual(launcher.read(obj.state_file)['model_attempt_limit'], 1)

    def test_new_request_spend_survives_relaunch_and_cannot_disappear(self):
        obj = self.instance(fail="image")
        with self.assertRaises(RuntimeError):
            obj.run()
        p = self.root / "data/results/r+000-c+000/worker/height/request.json"
        p.parent.mkdir(parents=True)
        p.write_text("one attempt")
        with self.assertRaises(RuntimeError):
            self.instance(fail="image").run()
        self.assertEqual(len(launcher.read(obj.state_file)["new_requests"]), 1)
        p.unlink()
        with self.assertRaisesRegex(RuntimeError, "request evidence removed"):
            self.instance().run()

    def test_config_changes_and_unsafe_paths_fail(self):
        obj = self.instance()
        obj.run()
        self.config["limits"]["seconds"] += 1
        launcher.save(self.path, self.config)
        with self.assertRaisesRegex(ValueError, "identity changed"):
            self.instance().run()
        self.config["assembly"] = "../overwrite"
        launcher.save(self.path, self.config)
        with self.assertRaises(ValueError):
            self.instance()


if __name__ == "__main__":
    unittest.main()
