#!/usr/bin/env python3
"""Thin Linux/WSL run launcher. Python stdlib, Bash and local Docker only.

Scientific preparation, completion filtering, processing and independent assembly
remain in the pinned image. This file manages explicit commands and their journal.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import subprocess
import sys
import time
import uuid

from spc_agb_worker.observation import validate_observation


def read(path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def save(path, value):
    temporary = path.with_name(path.name + ".tmp-" + uuid.uuid4().hex)
    with temporary.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


class Launcher:
    def __init__(self, config, resolved):
        self.config_file = config.resolve()
        self.c = resolved
        country, year = resolved['country'], resolved['year']
        validate_observation(year, "0.4.0")
        if not re.fullmatch(r"[A-Z]{3}", country):
            raise ValueError("Explicit country code required")
        self.country, self.year = country, year
        self.root = Path(__file__).resolve().parent
        self.state_file = self.c["run_root"] / "launcher-state.json"
        self.state = None

    def requests(self):
        root = self.c["country_root"]
        return {p.relative_to(root).as_posix(): sha(p)
                for p in (root / "results").glob("*/worker/height/request.json")}

    def guard(self):
        if time.time() >= self.state["deadline_epoch"]:
            raise RuntimeError("Original launcher deadline exhausted; not reset on relaunch")
        if shutil.disk_usage(self.c["limits"]["free_space_path"]).free < self.c["limits"]["minimum_free_bytes"]:
            raise RuntimeError("Configured physical free-space reserve exhausted")
        now = self.requests()
        previous = dict(self.state["baseline_requests"], **self.state["new_requests"])
        if any(now.get(k) != v for k, v in previous.items()):
            raise RuntimeError("Existing inference request evidence removed/changed")
        self.state["new_requests"].update({k: v for k, v in now.items() if k not in previous})
        save(self.state_file, self.state)
        limit = self.c["limits"]["model_attempts"]
        if limit is None:
            limit = self.state.get('model_attempt_limit')
        if limit is not None and len(self.state["new_requests"]) > limit:
            raise RuntimeError("Persistent model-attempt allowance exhausted")

    def stop_owned(self):
        p = subprocess.run(["docker", "ps", "--filter", "name=" + self.state["prefix"],
                            "--format", "{{.Names}}"], capture_output=True, text=True, timeout=15)
        for name in p.stdout.splitlines():
            if name.startswith(self.state["prefix"] + "-"):
                subprocess.run(["docker", "stop", "--time", "0", name], capture_output=True, timeout=20)

    def command(self, stage, argv, env=None):
        self.guard()
        print("STAGE", stage, flush=True)
        number = len(self.state["commands"]) + 1
        stem = self.c["run_root"] / f"{number:03d}-{stage}"
        record = dict(stage=stage, argv=argv, started_epoch=time.time(),
                      stdout=stem.name + ".stdout.log", stderr=stem.name + ".stderr.log")
        self.state["stage"] = stage
        self.state["commands"].append(record)
        save(self.state_file, self.state)
        proc = None
        try:
            with Path(str(stem) + ".stdout.log").open("x") as out, Path(str(stem) + ".stderr.log").open("x") as err:
                proc = subprocess.Popen(argv, stdout=out, stderr=err,
                                        env=dict(os.environ, **(env or {})), start_new_session=True)
                while proc.poll() is None:
                    self.guard()
                    time.sleep(.5)
                record["returncode"] = proc.returncode
                if proc.returncode:
                    raise RuntimeError(f"{stage} failed ({proc.returncode}); see {stem}.stderr.log")
        except BaseException:
            if proc is not None and proc.poll() is None:
                os.killpg(proc.pid, signal.SIGTERM)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(proc.pid, signal.SIGKILL)
                    proc.wait()
            self.stop_owned()
            raise
        finally:
            record["finished_epoch"] = time.time()
            save(self.state_file, self.state)
        return Path(str(stem) + ".stdout.log").read_text()

    def docker(self, stage, module=None, args=(), mounts=(), network="none", code=None):
        name = self.state["prefix"] + f"-{stage}-{len(self.state['commands'])}"
        cmd = ["docker", "run", "--name", name, "--pull", "never", "--network", network,
               "--read-only", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--user", "65534:65534", "--memory", "6g", "--memory-swap", "6g", "--cpus", "8",
               "--tmpfs", "/tmp:rw,noexec,nosuid,size=268435456"]
        for source, target, readonly in mounts:
            cmd += ["--mount", f"type=bind,src={source},dst={target}" + (",readonly" if readonly else "")]
        cmd += ["--entrypoint", "/opt/adapter-venv/bin/python", self.c["image"], "-B"]
        cmd += ["-c", code] if code else ["-m", module]
        return self.command(stage, cmd + list(args))

    def jobs(self, stage):
        rows = json.loads(self.docker(stage, "spc_agb_dep.print_tasks",
            ["--catalog", "/data/countries.json", "--country", self.country, "--year", str(self.year)],
            [(self.c["country_root"], "/data", True)]))
        if (not rows or len({r["window"] for r in rows}) != len(rows)
                or any(r["country"] != self.country or r["year"] != self.year
                       or r["status"] not in ["configured", "completed"] for r in rows)):
            raise RuntimeError("Missing, failed, incomplete or incompatible jobs; no automatic retry")
        return rows

    def completed_pins(self, rows):
        return {r["window"]: sha(self.c["country_root"] / "results" / r["window"] / "integration-completion.json")
                for r in rows}

    def assembly_files(self):
        root = self.c["assembly"]
        files = list(root.rglob("*"))
        if any(p.is_symlink() for p in files):
            raise ValueError("Symlink in assembly evidence")
        return {p.relative_to(root).as_posix(): sha(p) for p in files if p.is_file()}

    def assembly(self, rows):
        root = self.c["assembly"]
        pins = self.completed_pins(rows)
        receipt = self.state.get("assembly_receipt")
        if root.exists():
            if not receipt or receipt["completions"] != pins or receipt["files"] != self.assembly_files():
                raise FileExistsError("Assembly preserved: no matching launcher verification receipt; supply a fresh destination in a separately admitted run")
            print("STAGE assembly: reused verified product", root, flush=True)
            return "reused-verified"
        if receipt:
            raise FileNotFoundError("Previously verified assembly disappeared; no silent replacement")
        if not root.parent.exists():
            root.parent.mkdir()
            root.parent.chmod(0o1777)
        # Writable parent is explicit; source/results remain read-only.
        data, target = str(self.c["country_root"]), str(root)
        mounts = [(self.c["country_root"], data, True), (root.parent, str(root.parent), False)]
        self.docker("assemble", "spc_agb_dep.assemble_country",
            ["--catalog", data + "/countries.json", "--country", self.country, "--year", str(self.year),
             "--results", data + "/results", "--output", target], mounts)
        country = read(root / "country-summary.json")
        collection = read(root / "collection.json")
        if (country.get("complete") is not True or country.get("summary") is None
                or country["country"]["code"] != self.country or country["year"] != self.year
                or {r["window"] for r in country["windows"]} != set(pins)
                or any(r["status"] != "complete" for r in country["windows"])
                or collection.get("agc:complete") is not True):
            raise RuntimeError("Incomplete country output; not a verified complete product")
        self.state["assembly_receipt"] = dict(completions=pins, files=self.assembly_files())
        save(self.state_file, self.state)
        return "assembled-verified"

    def stages(self):
        c = self.c
        ident = self.command("image", ["docker", "image", "inspect", "--format", "{{.Id}}", c["image"]]).strip()
        if ident != c.get("expected_image", c["image"]):
            raise ValueError("Actual local image identity differs")
        if self.state.get("resolved_image_id", ident) != ident:
            raise ValueError("Readable image reference changed identity since the first launch")
        self.state["resolved_image_id"] = ident
        self.state["image_reference"] = c["image"]
        c["image"] = ident
        if c.get("bindings"):
            print("RESOLVED DEPENDENCIES", json.dumps(c["bindings"], indent=2), flush=True)
        prep = read(c["preparation"]["file"])
        prepared = c["country_root"] / "prepared"
        complete = prepared.exists() and read(prepared / "preparation-state.json").get("status") == "prepared"
        # Existing file-identity functions only; never import/load the model.
        check = ("import sys,os; from pathlib import Path; from spc_agb_worker.io import read_json,check_file; "
                 "t=Path(sys.argv[1]); check_file(t,sys.argv[2]); s=read_json(t)['inference']; "
                 "[check_file(Path('/model')/s['model'][n],s['model'][h]) for n,h in "
                 "[('weights','weights_sha256'),('training','training_sha256')]]; "
                 "[check_file(Path('/source')/n,h) for n,h in s['source_sha256'].items()]; "
                 "check_file(Path(sys.argv[3]),sys.argv[4]); check_file(Path(sys.argv[5]),sys.argv[6]); "
                 "assert all(Path(p).is_dir() for p in ['/data','/incoming']); "
                 "assert sys.argv[7]=='reuse' or all(os.access(p,os.W_OK) for p in ['/data','/incoming']); "
                 "print('mount/config/model/source pins verified; no model loaded')")
        self.docker("preflight", args=[c["image_task_template"], prep["task_template"]["sha256"],
            "/plans/" + c["catalog"]["path"], sha(c["catalog"]["file"]),
            "/config/" + c["preparation"]["path"], sha(c["preparation"]["file"]), "reuse" if complete else "fresh"],
            mounts=[(c["model"], "/model", True), (c["source"], "/source", True),
                    (c["catalog"]["root"], "/plans", True), (c["preparation"]["root"], "/config", True),
                    (c["country_root"], "/data", complete), (c["archive_root"], "/incoming", complete),
                    (c["scientific_profile"].parent, str(Path(c["image_task_template"]).parent), True)], code=check)
        prep_mounts = [(c["catalog"]["root"], "/plans", True), (c["preparation"]["root"], "/config", True),
                       (c["country_root"], "/data", complete), (c["archive_root"], "/incoming", complete),
                       (c["scientific_profile"].parent, str(Path(c["image_task_template"]).parent), True)]
        self.docker("prepare", "spc_agb_inputs.prepare_country",
            ["--catalog", "/plans/" + c["catalog"]["path"], "--country", self.country, "--year", str(self.year),
             "--config", "/config/" + c["preparation"]["path"], "--output", "/data/prepared", "--archive", "/incoming"],
            prep_mounts, c["preparation_network"])
        self.docker("bind", "spc_agb_dep.catalog",
            ["--catalog", "/data/prepared/countries.json", "--country", self.country, "--year", str(self.year),
             "--results", "/data/results", "--output", "/data/countries.json"],
            [(c["country_root"], "/data", False), (prepared, "/data/prepared", True)])
        rows = self.jobs("list")
        pending = sum(r["status"] != "completed" for r in rows)
        limit = c['limits']['model_attempts']
        if limit is None:
            if 'model_attempt_limit' not in self.state:
                self.state['model_attempt_limit'] = pending + len(self.state['new_requests'])
                save(self.state_file, self.state)
            limit = self.state['model_attempt_limit']
        if pending > limit - len(self.state["new_requests"]):
            raise RuntimeError("Pending jobs exceed remaining explicit model allowance; no dispatch")
        # Existing shell performs its own verified completion filtering and
        # requests NVIDIA only if a pending processing job actually exists.
        self.command("process", ["bash", str(self.root / "run-country.sh"), c["image"], self.country,
            str(self.year), str(c["country_root"]), str(c["model"]), str(c["source"]),
            str(c["country_root"] / "results"), self.state["prefix"]],
            {"AGC_MAX_JOBS": str(limit - len(self.state["new_requests"]))})
        rows = self.jobs("verify-jobs")
        if any(r["status"] != "completed" for r in rows):
            raise RuntimeError("Incomplete country: assembly not dispatched")
        return self.assembly(rows)

    def run(self):
        c = self.c
        c["run_root"].mkdir(parents=True, exist_ok=True)
        with (c["run_root"] / "launcher.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            contract = dict(config_sha256=sha(self.config_file), country=self.country, year=self.year,
                            launcher_sha256=sha(Path(__file__)), runner_sha256=sha(self.root / "run-country.sh"))
            if c.get("bindings"):
                contract.update(dependency_manifest_sha256=c["bindings"]["dependency_manifest_sha256"],
                                resolver_sha256=sha(self.root / "configuration.py"),
                                preparation_sha256=c["bindings"]["preparation_sha256"])
            if c.get('durable_output'):
                contract.update(configuration_sha256=c['source_config_sha256'],
                                product_helper_sha256=sha(self.root / 'durable_product.py'))
            if self.state_file.exists():
                self.state = read(self.state_file)
                if self.state["contract"] != contract:
                    raise ValueError("Launcher/config identity changed; cross-version migration not supported")
            else:
                self.state = dict(schema="agc-country-launch-state-v1", contract=contract,
                    created_epoch=time.time(), deadline_epoch=time.time() + c["limits"]["seconds"],
                    prefix="agc-launch-" + uuid.uuid4().hex[:16], baseline_requests=self.requests(),
                    new_requests={}, commands=[], status="started")
                save(self.state_file, self.state)
            try:
                result = self.stages()
                durable = None
                if c.get('durable_output'):
                    from durable_product import publish
                    print('STAGE durable-product', c['durable_output'], flush=True)
                    durable = publish(c['assembly'], c['durable_output'], self.guard)
                self.guard()
                self.state.update(status="complete", stage="complete", result=dict(country=self.country,
                    year=self.year, status="complete", assembly_status=result, prepared=str(c["country_root"] / "prepared"),
                    results=str(c["country_root"] / "results"), product=str(c["assembly"]),
                    collection=str(c["assembly"] / "collection.json"), image=c["image"],
                    new_model_requests=len(self.state["new_requests"])))
                if durable:
                    self.state['result'].update(durable=durable, product=durable['product'],
                        collection=str(c['durable_output'] / 'collection.json'))
                save(self.state_file, self.state)
                print(json.dumps(self.state["result"], indent=2), flush=True)
                return self.state["result"]
            except BaseException as exc:
                self.state.update(status="failed", error=str(exc))
                save(self.state_file, self.state)
                raise


def run_batch(path, runner=None):
    """One preflight, then an explicit serial loop over the common country runner."""
    from configuration import resolve_batch, batch_settings
    root = Path(__file__).resolve().parent
    resolved = resolve_batch(path.resolve(), root)
    # Every selected country's pins were checked above; no first-country dispatch
    # can hide a missing prerequisite for a later country.
    for _, c in resolved:
        if not c['expected_image']:
            raise RuntimeError('Source review candidate — verification pending: no image is qualified for this source')
        ident = subprocess.check_output(['docker', 'image', 'inspect', '--format', '{{.Id}}', c['image']], text=True).strip()
        if ident != c['expected_image']:
            raise ValueError('Supplied release image differs from qualified identity: ' + c['image'])
    _, selected, _, work, _, _ = batch_settings(path.resolve(), root)
    batch_root = work / 'batches' / str(resolved[0][1]['year'])
    state_file = batch_root / 'batch-state.json'
    contract = dict(config_sha256=sha(path), launcher_sha256=sha(Path(__file__)), countries=selected)
    batch_root.mkdir(parents=True, exist_ok=True)
    with (batch_root / 'batch.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if state_file.exists():
            state = read(state_file)
            if state['contract'] != contract:
                raise ValueError('Batch identity changed; old state preserved, no migration')
        else:
            state = dict(contract=contract, created_epoch=time.time(), status='started', countries={})
        try:
            for config, c in resolved:
                code = c['country']
                state.update(status='running', current_country=code)
                save(state_file, state)
                print('COUNTRY', code, c['year'], flush=True)
                result = runner(config, c) if runner else Launcher(config, resolved=c).run()
                if result.get('status') != 'complete':
                    raise RuntimeError('Country incomplete: ' + code)
                state['countries'][code] = result
                save(state_file, state)
            state.update(status='complete', current_country=None)
            save(state_file, state)
            print(json.dumps(dict(status='complete', countries=state['countries']), indent=2), flush=True)
            return state
        except BaseException as exc:
            state.update(status='failed', error=str(exc))
            save(state_file, state)
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument('--list-countries', action='store_true', help='List plans, supplied dependencies and execution qualification; no work')
    parser.add_argument('--markdown', action='store_true', help='Render the catalog table used in README')
    a = parser.parse_args()
    try:
        if a.list_countries:
            from configuration import country_rows
            rows = country_rows(Path(__file__).resolve().parent, a.config)
            if a.markdown:
                print('| Code | Country/territory | Plan | Dependencies | Execution |\n|---|---|---|---|---|')
                for row in rows:
                    print('| ' + ' | '.join(row[k] for k in ['code', 'name', 'plan', 'dependencies', 'execution']) + ' |')
            else:
                print(json.dumps(rows, indent=2))
        elif not a.config:
            raise ValueError('--config run.json is required to launch')
        else:
            run_batch(a.config)
    except (Exception, KeyboardInterrupt) as exc:
        print(f"LAUNCH FAILED: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
