"""Explicit local files and native raster I/O. No storage/service discovery."""

import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
import rasterio

from .grid import validate_grid
from .policy import validate_policy
from .observation import validate_observation


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")


def local_input(package, name):
    relative = Path(name)
    if (not name.startswith("inputs/") or "\\" in name or ":" in name
            or ".." in relative.parts):
        raise ValueError("Input must be a package-relative inputs/ reference: " + name)
    path = package / relative
    if not path.resolve().is_relative_to(package.resolve()):
        raise ValueError("Input escapes package")
    if any(p.is_symlink() for p in [path, *path.parents] if p.is_relative_to(package)):
        raise ValueError("Symlink inputs are not supported")
    return path


def check_file(path, digest):
    if not path.is_file() or sha256(path) != digest:
        raise ValueError("Missing or changed input: " + str(path))


def validate_inputs(task, package):
    validate_grid(task)
    validate_observation(task["year"], task["source_version"])
    if task["source_product"] != "GeoMAD":
        raise ValueError("Unsupported annual product contract")
    inputs = task["inputs"]
    if "worldcover" not in inputs or {"owners", "owner_order"} & inputs.keys():
        raise ValueError("Pacific tile tasks require pinned WorldCover and no country ownership inputs")
    rows = [inputs["canonical"], *inputs["blocks"], inputs["worldcover"]]
    identities = {}
    for row in rows:
        path = local_input(package, row["path"])
        check_file(path, row["sha256"])
        identities[row["path"]] = row["sha256"]
    vrt = local_input(package, inputs["canonical"]["path"])
    children = set()
    for node in ET.parse(vrt).findall(".//SourceFilename"):
        if node.get("relativeToVRT") != "1":
            raise ValueError("VRT source must be relative")
        path = (vrt.parent / node.text).resolve()
        if not path.is_relative_to(package.resolve()):
            raise ValueError("External VRT dependency")
        children.add(path.relative_to(package.resolve()).as_posix())
    if children != {r["path"] for r in inputs["blocks"]}:
        raise ValueError("VRT block closure differs")
    infer = task["inference"]
    m = infer["model"]
    settings = infer["settings"]
    if (m["id"] != "24123113h21" or settings["general"]["model_name"] != m["weights"]
            or settings["data"] != {
                "s1": "64", "num_workers": "0", "prefetch_factor": "8", "batch_size": "1"}
            or settings["resources"] != {
                "accelerator": "gpu", "devices": "0", "strategy": "auto", "precision": "32-true"}
            or settings["transform"] != {"output_transform": "multiply60"}
            or infer["timeout_seconds"] != 900):
        raise ValueError("Accepted inference settings required")
    validate_policy(task["policy"])
    return identities


def validate(task, package, model, source):
    identities = validate_inputs(task, package)
    infer = task["inference"]
    m = infer["model"]
    for name, key in [(m["weights"], "weights_sha256"), (m["training"], "training_sha256")]:
        check_file(model / name, m[key])
    for name, digest in infer["source_sha256"].items():
        check_file(source / name, digest)
    return identities


def write_raster(path, values, valid, grid, crs, nodata=None):
    """Accepted native writer with explicit task CRS; encoding/masks unchanged."""
    values = values[None] if values.ndim == 2 else values
    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
        path, "w", driver="GTiff", height=values.shape[1], width=values.shape[2],
        count=values.shape[0], dtype=values.dtype, crs=crs, transform=grid,
        nodata=nodata, compress="deflate", tiled=True, blockxsize=256, blockysize=256,
    ) as ds:
        ds.write(values)
        ds.write_mask(valid.astype(np.uint8) * 255)
        if values.shape[0] == 3:
            ds.colorinterp = (rasterio.enums.ColorInterp.red,
                              rasterio.enums.ColorInterp.green,
                              rasterio.enums.ColorInterp.blue)
