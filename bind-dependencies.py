#!/usr/bin/env python3
"""Issue supporting data: create a verified country manifest, not a processing stage.

Copies approved plan/boundary metadata and binds the model/source folders.
This supporting-data set requires no previous processing outputs.
"""
import argparse
import json
import os
from pathlib import Path
import shutil

from configuration import package_metadata, pinned, read, sha, selection, roster
from spc_agb_inputs.dependencies import dependency_file, ownership_metadata


def approved_projected(entry, required, catalog_root, metadata_root, crs):
    """Resolve the certificate locator, never a guessed filename or repaired shape."""
    if (not isinstance(required, dict) or not required.get('path')
            or not required.get('sha256') or required.get('crs') != crs):
        raise ValueError('Approved projected geometry requires a certificate path/checksum and matching CRS')
    if metadata_root is None:
        raise ValueError('Approved projected geometry requires an explicit --metadata-root')
    original = dependency_file(metadata_root, required)
    declared = entry.get('accepted_projected')
    if declared is not None:
        if (not isinstance(declared, dict) or declared.get('sha256') != required['sha256']
                or declared.get('crs') != required['crs']):
            raise ValueError('Catalogue and certificate projected geometry identities differ')
        located = dependency_file(catalog_root, declared)
        if not located.samefile(original):
            raise ValueError('Ambiguous catalogue and certificate projected geometry locations')
    obj = read(original)
    if obj.get('crs', {}).get('properties', {}).get('name') != required['crs']:
        raise ValueError('Approved projected geometry CRS differs')
    return original


def bind(destination, catalog, country, boundary, model, source, metadata_root=None):
    _, external, profile, _ = package_metadata(Path(__file__).resolve().parent)
    index = read(catalog)
    root = (catalog.parent / index.get("artifact_root", ".")).resolve()
    candidates = [e for e in index['countries'] if e['country']['code'] == country]
    if len(candidates) != 1:
        raise ValueError('Supply one approved spatial definition per country')
    entry = candidates[0]
    plan = pinned(root, entry["plan"])
    plan_metadata = read(plan)
    check_metadata = read(pinned(root, entry['planning_check'])) if entry.get('planning_check') else {}
    maintained = next(e for e in roster(Path(__file__).resolve().parent) if e['country']['code'] == country)
    if entry['plan']['sha256'] != maintained['plan']['sha256']:
        raise ValueError('Supplied spatial plan differs from maintained approved identity')
    supplied_ownership = maintained.get('ownership') or entry.get('ownership')
    if plan_metadata['grid'].get('anchor') is None and not maintained.get('ownership'):
        raise ValueError('Blocked ownership: approved origin has not been established for this plan')
    ownership = ownership_metadata(plan_metadata, supplied_ownership)
    expected = plan_metadata.get('boundary', {}).get('sha256') or check_metadata.get('boundary_sha256')
    if not expected:
        raise ValueError('Approved plan/check must pin the supplied raw boundary')
    pinned(boundary.parent, dict(path=boundary.name, sha256=expected))
    for name, pin in external["model"]["files"].items():
        pinned(model, dict(path=name, sha256=pin["sha256"]))
    for name, digest in external["inference_source"]["files"].items():
        pinned(source, dict(path=name, sha256=digest))
    if (destination / "dependency-manifest.json").exists() or (destination / "countries.json").exists():
        raise FileExistsError("Existing dependency binding preserved; use a fresh destination")
    destination.mkdir(parents=True, exist_ok=True)
    def copy(p, name):
        target = destination / name
        if target.exists():
            if sha(p) != sha(target):
                raise FileExistsError("Existing dependency differs: " + str(target))
        else:
            shutil.copyfile(p, target)
        assert sha(p) == sha(target)
    for p, name in [(plan, "plan.json"), (boundary, "boundary.gpkg")]:
        copy(p, name)
    new_entry = dict(country=entry["country"], ownership=ownership, plan=dict(path="plan.json", sha256=sha(plan)))
    if entry.get("plan_status") == "blocked":
        raise ValueError("Blocked plan cannot be issued as executable input")
    projected = None
    if entry.get("planning_check"):
        checked = pinned(root, entry["planning_check"])
        record = read(checked)
        if not record["passed"] or record["country"] != entry["country"] or record["plan_sha256"] != sha(plan):
            raise ValueError("Approved planning check differs")
        if record.get('boundary_sha256', expected) != expected:
            raise ValueError('Approved boundary/check identity differs')
        required_geometry = record.get('accepted_projected')
        if required_geometry:
            original = approved_projected(entry, required_geometry, root, metadata_root,
                                          plan_metadata['grid']['crs'])
            copy(original, 'approved-projected.geojson')
            projected = dict(path='approved-projected.geojson', sha256=sha(original), crs=required_geometry['crs'])
            new_entry['accepted_projected'] = projected
        elif entry.get('accepted_projected'):
            raise ValueError('Projected geometry lacks an approval certificate')
        copy(checked, "planning-check.json")
        new_entry.update(plan_status=entry.get("plan_status", "checked"),
                         planning_check=dict(path="planning-check.json", sha256=sha(checked)))
    elif entry.get("plan_status") == "checked":
        raise ValueError("Checked plan requires its approval record")
    # A new plan-only binding, not mutation of any old catalog/job/receipt.
    (destination / "countries.json").write_text(json.dumps(dict(schema="agc-spatial-countries-v1", artifact_root=".", countries=[new_entry]), indent=2) + "\n")
    def located(p):
        return os.path.relpath(p, destination) if p.is_relative_to(destination) else str(p)
    manifest = dict(schema="agc-dependencies-v4", ownership=ownership, country=country, binding_kind="explicit-local-or-mounted-assets",
                    catalog=dict(path="countries.json", sha256=sha(destination / "countries.json")),
                    boundary=dict(path="boundary.gpkg", sha256=expected),
                    model_directory=located(model), inference_source_directory=located(source),
                    provenance=dict(catalog_sha256=sha(catalog), plan_sha256=sha(plan),
                                    copied_dependencies="Plan/boundary bytes unchanged; new plan-only catalog", payloads_copied=False))
    if projected:
        manifest['accepted_projected'] = projected
    (destination / "dependency-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    for name in ["destination", "catalog", "model", "source", "boundary-directory"]:
        p.add_argument("--" + name, required=True, type=Path)
    p.add_argument('--metadata-root', type=Path,
                   help='Explicit bounded root for certificate-pinned projected geometry paths')
    p.add_argument('--countries', required=True, nargs='+', help='Country codes or ALL')
    a = p.parse_args()
    try:
        for code in selection(a.countries, Path(__file__).resolve().parent):
            bind(a.destination.resolve() / code, a.catalog.resolve(), code,
                 (a.boundary_directory / ('gadm41_' + code + '.gpkg')).resolve(),
                 a.model.resolve(), a.source.resolve(),
                 a.metadata_root.absolute() if a.metadata_root else None)
    except (OSError, ValueError, KeyError, StopIteration) as exc:
        p.exit(1, 'Dependency setup failed: ' + str(exc) + '\n')
