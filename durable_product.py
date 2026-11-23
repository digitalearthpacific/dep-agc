"""Copy verified product assets and local STAC lineage, never processing inputs.

No raster conversion or accounting: binary assets and non-STAC evidence retain
their bytes. Only local STAC hrefs are relocated; historical products are read-only.
"""
import json
import os
from pathlib import Path
import shutil
from urllib.parse import unquote, urlsplit

from configuration import read, sha


def local_target(parent, href):
    if not isinstance(href, str):
        raise ValueError('STAC href must be a string')
    url = urlsplit(href)
    if url.scheme == 'file':
        raise ValueError('Use explicit local relative STAC paths, not file URIs')
    if url.scheme or url.netloc:
        return None
    if url.query or url.fragment:
        raise ValueError('Unsupported local STAC query/fragment: ' + href)
    return (parent / unquote(url.path)).resolve()


def references(doc):
    """Only STAC references; embedded provenance paths remain historical records."""
    if doc.get('type') not in ['Feature', 'Collection', 'Catalog']:
        return []
    return list(doc.get('assets', {}).values()) + doc.get('links', [])


def verify(root, guard=lambda: None):
    root = root.resolve()
    receipt = read(root / 'product-manifest.json')
    expected = {r['path'] for r in receipt['files']} | {'product-manifest.json'}
    actual = {p.relative_to(root).as_posix() for p in root.rglob('*') if p.is_file()}
    if actual != expected or any(p.is_symlink() for p in root.rglob('*')):
        raise ValueError('Durable product inventory changed')
    for row in receipt['files']:
        guard()
        p = root / row['path']
        if not p.resolve().is_relative_to(root) or p.stat().st_size != row['bytes'] or sha(p) != row['sha256']:
            raise ValueError('Durable product identity differs: ' + row['path'])
        if p.suffix == '.json':
            for ref in references(read(p)):
                target = local_target(p.parent, ref['href'])
                if target is not None and (not target.is_relative_to(root) or not target.is_file()):
                    raise ValueError('Non-contained product reference: ' + ref['href'])
                if target is not None and ref.get('agc:sha256') and sha(target) != ref['agc:sha256']:
                    raise ValueError('Published asset checksum differs')
    return receipt


def publish(source, destination, guard=lambda: None):
    source, destination = source.resolve(), destination.resolve()
    summary = read(source / 'country-summary.json')
    if summary.get('complete') is not True:
        raise ValueError('Cannot publish incomplete country accounting')
    identity = dict(country=summary['country'], year=summary['year'],
                    summary_sha256=sha(source / 'country-summary.json'), collection_sha256=sha(source / 'collection.json'))
    if destination.exists():
        receipt = verify(destination, guard)
        if receipt['source_identity'] != identity:
            raise FileExistsError('Existing durable product belongs to different accounting; preserved')
        return dict(status='reused-verified', product=str(destination), files=len(receipt['files']))
    if destination.is_relative_to(source) or source.is_relative_to(destination):
        raise ValueError('Durable product and temporary assembly must be separate')
    # Discover the finite product graph first; no copying input cache/native workers.
    mapping, documents, queue = {}, {}, []
    def register(p, preferred=None):
        p = p.resolve()
        if p in mapping:
            return
        if not p.is_file() or p.is_symlink():
            raise FileNotFoundError('Missing regular product asset: ' + str(p))
        digest = sha(p)
        target = Path(preferred) if preferred else Path('assets' if p.suffix != '.json' else 'records') / digest / p.name
        mapping[p] = target
        queue.append(p)
    for p in sorted(source.rglob('*')):
        if p.is_file():
            register(p, p.relative_to(source))
    while queue:
        guard()
        p = queue.pop(0)
        if p.suffix == '.json':
            doc = read(p)
            refs = references(doc)
            if refs:
                documents[p] = doc
            for ref in refs:
                target = local_target(p.parent, ref['href'])
                if target is not None:
                    register(target)
    destination.mkdir(parents=True, exist_ok=False)
    inventory = []
    for p, relative in mapping.items():
        guard()
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        existed = target.exists()
        if p in documents:
            doc = documents[p]
            changed = False
            for ref in references(doc):
                resolved = local_target(p.parent, ref['href'])
                if resolved is not None:
                    ref['href'] = os.path.relpath(destination / mapping[resolved], target.parent)
                    changed = True
            raw = (json.dumps(doc, indent=2, allow_nan=False) + '\n').encode() if changed else p.read_bytes()
            if existed:
                if target.read_bytes() != raw:
                    raise ValueError('Conflicting relocated metadata identity')
            else:
                with target.open('xb') as stream:
                    stream.write(raw)
        else:
            if not existed:
                with p.open('rb') as inp, target.open('xb') as out:
                    shutil.copyfileobj(inp, out, 8 * 1024 * 1024)
            if sha(p) != sha(target):
                raise ValueError('Asset copy changed bytes: ' + str(p))
        if existed:
            continue
        inventory.append(dict(path=relative.as_posix(), bytes=target.stat().st_size, sha256=sha(target),
                              original_sha256=sha(p), copied_bytes_unchanged=p not in documents,
                              provenance_source=str(p)))
    receipt = dict(schema='agc-durable-product-v1', source_identity=identity, files=inventory,
                   raster_values_unchanged=True, local_STAC_references_self_contained=True,
                   original_products_modified=False)
    with (destination / 'product-manifest.json').open('x') as stream:
        json.dump(receipt, stream, indent=2); stream.write('\n')
    verify(destination, guard)
    return dict(status='published-verified', product=str(destination), files=len(inventory))
