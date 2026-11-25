"""Resolve one run configuration using supplied, verified dependency metadata.

No GIS, discovery, model loading or processing. Local bindings are explicit
dependency locations, never fallback searches. Relative bindings support a staged
bundle; absolute bindings describe retained/mounted dependencies and are recorded.
"""
import hashlib
import json
import math
from pathlib import Path

from carbon.observation import validate_observation
from tasks.inputs.dependencies import ownership_metadata, dependency_file


def read(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))


def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def location(base, value):
    if not isinstance(value, str) or not value or any(c in value for c in ",\r\n"):
        raise ValueError("Explicit dependency/storage path required")
    return (base / value).resolve()


def pinned(base, value):
    p = location(base, value["path"])
    if not p.is_file() or sha(p) != value["sha256"]:
        raise ValueError("Missing or changed supplied dependency: " + str(p))
    return p


def package_metadata(root):
    """Read maintained release/profile metadata, never a historical export index."""
    release = read(root / 'release.json')
    template = root / 'configs/scientific-profile.json'
    if sha(template) != release['scientific_profile_sha256']:
        raise ValueError('Scientific profile differs from the release')
    task = read(template)
    policy_digest = hashlib.sha256(json.dumps(task['policy'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    if policy_digest != release['annual_policy_sha256']:
        raise ValueError('Annual policy differs from the release')
    infer = task['inference']
    model = infer['model']
    external = dict(model=dict(id=model['id'], files={
        model['weights']: dict(sha256=model['weights_sha256']),
        model['training']: dict(sha256=model['training_sha256'])}),
        inference_source=dict(commit=infer['source_commit'], files=infer['source_sha256']))
    profile = dict(geomad_version=task['source_version'],
                   template_in_image='/opt/processor/examples/scientific-profile.json')
    return dict(processor_image=release['image_id']), external, profile, template


def roster(root):
    catalog = read(root / 'configs/countries.json')
    if catalog['schema'] != 'agc-spatial-countries-v1':
        raise ValueError('Year-independent spatial catalogue required')
    return catalog['countries']


def selection(value, root):
    codes = [r['country']['code'] for r in roster(root)]
    values = [value] if isinstance(value, str) else value
    if values == ['ALL']:
        return codes
    if (not isinstance(values, list) or not values or any(not isinstance(x, str) for x in values)
            or len(set(values)) != len(values) or any(x not in codes for x in values)):
        raise ValueError('Select one or more country codes, or ["ALL"] alone. Valid: ' + ', '.join(codes))
    return values


def batch_settings(path, root):
    c = read(path)
    if 'artifacts' in c:
        raise ValueError('Obsolete run field artifacts; use dependency_directory')
    required = {'countries', 'year', 'geomad_version', 'dependency_directory', 'working_directory', 'output_directory'}
    if not required <= set(c) or set(c) - required - {'advanced'}:
        raise ValueError('Expected countries/year/geomad_version/dependency_directory/working_directory/output_directory; optional advanced')
    validate_observation(c['year'], c['geomad_version'])
    selected = selection(c['countries'], root)
    release = read(root / 'release.json')
    dependencies = location(path.parent, c['dependency_directory'])
    environment_file = dependencies / 'environment.json'
    environment = read(environment_file) if environment_file.exists() else {}
    if not isinstance(environment, dict):
        raise ValueError('Environment settings must be a JSON object')
    allowed = {'image_reference', 'minimum_free_gib', 'free_space_directory', 'allow_acquisition',
               'minutes', 'model_attempts', 'source_transfer_gib'}
    advanced = c.get('advanced', {})
    if not isinstance(advanced, dict) or set(advanced) - allowed or set(environment) - allowed:
        raise ValueError('Unknown environment/advanced override')
    settings = dict(release['defaults'], **environment)
    settings.update(advanced)
    settings.setdefault('image_reference', release['image_reference'])
    work = location(path.parent, c['working_directory'])
    output = location(path.parent, c['output_directory'])
    if work.is_relative_to(output) or output.is_relative_to(work):
        raise ValueError('Working and durable output roots must be separate')
    for target in [work, output]:
        if target.is_relative_to(dependencies) or dependencies.is_relative_to(target):
            raise ValueError('Working/output roots overlap supplied dependencies')
    return c, selected, dependencies, work, output, settings


def country_rows(root, config=None):
    release = read(root / 'release.json')
    dependencies = location(config.parent, read(config)['dependency_directory']) if config else None
    rows = []
    for entry in roster(root):
        code = entry['country']['code']
        supplied = dependencies / code / 'dependency-manifest.json' if dependencies else None
        status = 'not supplied'
        if supplied and supplied.is_file():
            try:
                m = read(supplied)
                pinned(supplied.parent, m['catalog'])
                pinned(supplied.parent, m['boundary'])
                for key in ['model_directory', 'inference_source_directory']:
                    if not location(supplied.parent, m[key]).is_dir():
                        raise ValueError('Missing ' + key)
                if m['schema'] != 'agc-dependencies-v4' or m['country'] != code:
                    raise ValueError('Spatial dependency schema/country mismatch')
                status = 'supplied; full preflight at launch'
            except (OSError, ValueError, KeyError) as exc:
                status = 'invalid: ' + str(exc)
        rows.append(dict(code=code, name=entry['country']['name'], plan=entry.get('plan_status', 'configured'),
                         dependencies=status, execution=entry.get('execution_qualification', 'Not executed')))
    return rows


def resolve_batch(path, root):
    """Preflight every selected dependency before any country processing is launched."""
    c, selected, dependencies, work, output, settings = batch_settings(path, root)
    resolved = []
    errors = []
    for code in selected:
        try:
            bundle = dependencies / code
            manifest = read(bundle / 'dependency-manifest.json')
            catalog = pinned(bundle, manifest['catalog'])
            index = read(catalog)
            entry = next((x for x in index['countries'] if x['country']['code'] == code), None)
            if entry is None:
                raise ValueError('Supplied dependency catalogue has no matching country')
            plan = read(pinned((catalog.parent / index.get('artifact_root', '.')).resolve(), entry['plan']))
            if entry.get('plan_status') == 'blocked' or plan.get('planning_status') == 'blocked':
                raise ValueError(code + ': approved plan is blocked')
            maintained = next(x for x in roster(root) if x['country']['code'] == code)
            if entry['plan']['sha256'] != maintained['plan']['sha256']:
                raise ValueError(code + ': plan differs from maintained approved catalog')
            if (plan['country'] != entry['country'] or plan.get('source_version', c['geomad_version']) != c['geomad_version']):
                raise ValueError(code + ': approved spatial plan country/source differs')
            if maintained.get('planning_check') and not entry.get('planning_check'):
                raise ValueError(code + ': maintained planning certificate is required')
            if entry.get('planning_check'):
                if entry['planning_check']['sha256'] != maintained.get('planning_check', {}).get('sha256'):
                    raise ValueError(code + ': planning certificate differs from maintained approval')
                checked = read(pinned(catalog.parent, entry['planning_check']))
                if not checked['passed'] or checked['plan_sha256'] != entry['plan']['sha256']:
                    raise ValueError(code + ': planning verification differs')
            elif entry.get('plan_status') == 'checked':
                raise ValueError(code + ': checked plan requires its approval record')
            count = len(plan['grid']['windows'])
            if not count:
                raise ValueError(code + ': approved plan contains no windows')
            base = work / code / str(c['year'])
            limits = dict(minutes=settings.get('minutes', settings['base_minutes'] + count * settings['minutes_per_window']),
                          model_attempts=settings.get('model_attempts', count),
                          source_transfer_gib=settings.get('source_transfer_gib', count * settings['source_transfer_gib_per_window']),
                          minimum_free_gib=settings['minimum_free_gib'],
                          free_space_directory=str(location(path.parent, settings.get('free_space_directory', str(base)))))
            if not settings['allow_acquisition']:
                limits['source_transfer_gib'] = 0
            internal = dict(schema='agc-country-run-v3', country=code, year=c['year'], geomad_version=c['geomad_version'],
                            image=settings['image_reference'], dependency_directory=str(bundle), working_directory=str(base / 'intermediate'),
                            output_directory=str(base / 'assembly' / 'country-product'), log_directory=str(base / 'logs'),
                            archive_directory=str(base / 'incoming'), allow_acquisition=settings['allow_acquisition'], limits=limits)
            config = base / 'configuration.json'
            immutable(config, (json.dumps(internal, indent=2) + '\n').encode())
            try:
                country = resolve(config, root)
            except (OSError, ValueError, KeyError, StopIteration) as exc:
                raise ValueError(code + ': prerequisite verification failed: ' + str(exc)) from exc
            # None means derive once from verified pending jobs, then persist the cap.
            if 'model_attempts' not in settings:
                country['limits']['model_attempts'] = None
            country['durable_output'] = output / code / str(c['year'])
            country['source_config_sha256'] = sha(path)
            resolved.append((config, country))
        except (OSError, ValueError, KeyError, TypeError, StopIteration) as exc:
            errors.append(code + ': ' + str(exc))
    if errors:
        raise ValueError('Selected-country preflight failed; no country dispatched:\n' + '\n'.join(errors))
    return resolved


def immutable(path, raw):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("Resolved configuration changed; old state preserved: " + str(path))
    else:
        with path.open("xb") as stream:
            stream.write(raw)


def resolve(path, package_root):
    c = read(path)
    required = {"schema", "country", "year", "geomad_version", "image", "dependency_directory",
                "working_directory", "output_directory", "log_directory", "archive_directory",
                "allow_acquisition", "limits"}
    if set(c) != required or c["schema"] != "agc-country-run-v3":
        raise ValueError("Complete agc-country-run-v3 configuration required; unknown settings refused")
    inventory, external, profile, template = package_metadata(package_root)
    validate_observation(c["year"], c["geomad_version"])
    if type(c["allow_acquisition"]) is not bool:
        raise ValueError("allow_acquisition must be true or false")
    if not isinstance(c["image"], str) or not c["image"] or any(ch.isspace() for ch in c["image"]):
        raise ValueError("Explicit locally supplied image reference required")
    limits = c["limits"]
    if set(limits) != {"minutes", "model_attempts", "source_transfer_gib", "minimum_free_gib", "free_space_directory"}:
        raise ValueError("Explicit limits in minutes/GiB and a physical free-space directory required")
    for key in ["minutes", "source_transfer_gib", "minimum_free_gib"]:
        if type(limits[key]) not in [int, float] or not math.isfinite(limits[key]) or limits[key] < 0:
            raise ValueError("Invalid resource limit: " + key)
    if limits["minutes"] <= 0 or limits["minimum_free_gib"] <= 0 or type(limits["model_attempts"]) is not int or limits["model_attempts"] < 0:
        raise ValueError("Positive time/reserve and nonnegative model allowance required")
    if not c["allow_acquisition"] and limits["source_transfer_gib"] != 0:
        raise ValueError("Offline runs must declare zero source transfer")
    bundle = location(path.parent, c["dependency_directory"])
    manifest = bundle / "dependency-manifest.json"
    m = read(manifest)
    if m["schema"] != "agc-dependencies-v4" or m["country"] != c["country"]:
        raise ValueError("Supplied spatial dependency schema/country does not match run")
    catalog = pinned(bundle, m["catalog"])
    boundary = pinned(bundle, m["boundary"])
    index = read(catalog)
    catalog_root = (catalog.parent / index.get("artifact_root", ".")).resolve()
    if index["schema"] != "agc-spatial-countries-v1":
        raise ValueError("Year-independent spatial dependency catalogue required")
    entry = next((e for e in index["countries"] if e["country"]["code"] == c["country"]), None)
    if entry is None:
        raise ValueError("Approved spatial catalogue has no matching country")
    plan = read(pinned(catalog_root, entry["plan"]))
    checked_boundary = read(pinned(catalog_root, entry['planning_check'])) if entry.get('planning_check') else {}
    expected_boundary = plan.get('boundary', {}).get('sha256') or checked_boundary.get('boundary_sha256')
    if expected_boundary != m["boundary"]["sha256"]:
        raise ValueError("Supplied boundary differs from approved plan")
    maintained = next((e for e in roster(package_root) if e['country']['code'] == c['country']), {})
    approved_ownership = maintained.get('ownership')
    supplied_ownership = m.get('ownership')
    if supplied_ownership is None or supplied_ownership != entry.get('ownership'):
        raise ValueError('Required ownership metadata is missing or differs from dependency catalogue')
    if approved_ownership is not None and supplied_ownership != approved_ownership:
        raise ValueError('Ownership metadata differs from maintained accepted evidence')
    if plan['grid'].get('anchor') is None and approved_ownership is None:
        raise ValueError('Blocked ownership: no maintained approved origin for this plan')
    ownership = ownership_metadata(plan, supplied_ownership)
    projected = m.get('accepted_projected')
    checked = read(pinned(catalog_root, entry['planning_check'])) if entry.get('planning_check') else {}
    required_projected = checked.get('accepted_projected')
    if bool(projected) != bool(required_projected) or projected != entry.get('accepted_projected'):
        raise ValueError('Required approved projected geometry binding differs or is missing')
    if projected:
        if (projected['sha256'], projected['crs']) != (required_projected['sha256'], required_projected['crs']):
            raise ValueError('Projected geometry differs from approved planning certificate')
        dependency_file(bundle, projected)
    model, source = (location(bundle, m[k]) for k in ["model_directory", "inference_source_directory"])
    for name, pin in external["model"]["files"].items():
        pinned(model, dict(path=name, sha256=pin["sha256"]))
    for name, digest in external["inference_source"]["files"].items():
        pinned(source, dict(path=name, sha256=digest))
    work = location(path.parent, c["working_directory"])
    logs = location(path.parent, c["log_directory"])
    output = location(path.parent, c["output_directory"])
    archive = location(path.parent, c["archive_directory"])
    if output == work or output == logs or work.is_relative_to(output) or logs.is_relative_to(output):
        raise ValueError("Product destination must not contain working/log storage")
    for protected in [bundle, model, source]:
        for writable in [work, logs, archive, output]:
            if writable.is_relative_to(protected) or protected.is_relative_to(writable):
                raise ValueError("Working/output storage overlaps supplied dependencies")
    for p in [work, logs, archive]:
        new = not p.exists()
        p.mkdir(parents=True, exist_ok=True)
        if new and p in [work, archive]:
            p.chmod(0o1777)  # Writable by the isolated non-root container user.
    resolved = logs / "resolved"
    free = location(path.parent, limits["free_space_directory"])
    if 'checkpoint' in m:
        raise ValueError('Historical checkpoint bindings are not supported by this release; use a fresh workspace')
    data = work
    prep = dict(schema="agc-country-preparation-v4", dependency_root="/plans", ownership=ownership, year=c["year"], source_version=c["geomad_version"],
                boundary=dict(path="/plans/" + boundary.relative_to(bundle).as_posix(), sha256=m["boundary"]["sha256"]),
                task_template=dict(path=profile["template_in_image"], sha256=sha(template)),
                host_free_path="/data", limits=dict(seconds=int(limits["minutes"] * 60),
                minimum_free_bytes=int(limits["minimum_free_gib"] * 2**30), transfer_bytes=int(limits["source_transfer_gib"] * 2**30)))
    if projected:
        prep['accepted_projected'] = dict(projected)
    prep_raw = (json.dumps(prep, indent=2) + "\n").encode()
    immutable(resolved / "preparation.json", prep_raw)
    bindings = dict(dependency_manifest=str(manifest), dependency_manifest_sha256=sha(manifest),
                    catalog=str(catalog), boundary=str(boundary), model=str(model), source=str(source),
                    image_reference=c["image"], expected_image=inventory["processor_image"],
                    template_sha256=sha(template), preparation_sha256=hashlib.sha256(prep_raw).hexdigest())
    immutable(resolved / "dependencies.json", (json.dumps(bindings, indent=2) + "\n").encode())
    return dict(schema="agc-country-launch-resolved-v3", country=c["country"], year=c["year"],
                image=c["image"], expected_image=inventory["processor_image"], bindings=bindings,
                catalog=dict(root=bundle, path=catalog.relative_to(bundle).as_posix(), file=catalog),
                preparation=dict(root=resolved, path="preparation.json", file=resolved / "preparation.json"),
                scientific_profile=template, image_task_template=profile["template_in_image"], preparation_network="default" if c["allow_acquisition"] else "none",
                country_root=data, archive_root=archive, run_root=logs, model=model, source=source,
                assembly=output, limits=dict(seconds=int(limits["minutes"] * 60), model_attempts=limits["model_attempts"],
                free_space_path=free, minimum_free_bytes=int(limits["minimum_free_gib"] * 2**30)))
