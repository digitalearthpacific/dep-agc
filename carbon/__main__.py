"""Prepare imagery, run height inference and write native products."""

import argparse
from pathlib import Path
import sys
import time

import numpy as np
import rasterio

from . import __version__
from .accounting import read_worldcover, tile_products
from .grid import identity, retained_transform
from .inference import run as infer
from .io import local_input, read_json, sha256, validate, write_json
from .prepare import prepare, read_raw_height
from .science import crop_height


def work(task_file, package, model, source, output):
    output.mkdir(parents=True, exist_ok=False)  # Never overwrite/replay an earlier run.
    stage = "validate"
    started = time.monotonic()
    times = {}
    task = {}
    try:
        task = read_json(task_file)
        inputs = validate(task, package, model, source)
        path = local_input(package, task["inputs"]["worldcover"]["path"])
        worldcover = read_worldcover(path, task, retained_transform(task["grid"]))
        code = {p.name: sha256(p) for p in Path(__file__).parent.glob("*.py")}
        identities = dict(
            task_identity=identity(task), task_sha256=sha256(task_file), inputs=inputs,
            code=code, model=task["inference"]["model"],
            source_commit=task["inference"]["source_commit"],
            source_sha256=task["inference"]["source_sha256"],
        )
        write_json(output / "inputs.json", identities)
        stage = "prepare"
        tick = time.monotonic()
        arrays, grid, rgb = prepare(package, task, output / "prepared")
        times[stage] = time.monotonic() - tick

        stage = "inference"
        tick = time.monotonic()
        raw_path, metrics = infer(task, model, source, rgb, output / "height")
        times[stage] = time.monotonic() - tick
        height = read_raw_height(raw_path, task["grid"], grid)
        d = task["grid"]["context"]
        h, w = task["grid"]["retained_shape"]
        final, valid, crop_grid = crop_height(height, arrays["rgb_valid"], grid, [d, d, w, h])
        cropped = output / "height/height-aoi.tif"
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
            cropped, "w", driver="GTiff", count=1, dtype="float32", height=h, width=w,
            crs=task["grid"]["crs"], transform=crop_grid, nodata=-9999, compress="deflate",
        ) as ds:
            ds.write(final, 1)
            ds.write_mask(valid.astype(np.uint8) * 255)
        stage = "products"
        tick = time.monotonic()
        result = tile_products(task, arrays, height, crop_grid, worldcover, output / "native")
        times[stage] = time.monotonic() - tick
        completion = dict(
            schema="spc-agb-window-completion-v3", task_identity=identity(task),
            declared_grid=task["grid"], status="complete", worker_version=__version__,
            window=task["window"], year=task["year"], source_product=task["source_product"],
            source_version=task["source_version"], identities=identities, **result,
            prepared_rgb_sha256=sha256(rgb),
            height=dict(raw=str(raw_path.relative_to(output)), raw_sha256=sha256(raw_path),
                        retained=str(cropped.relative_to(output)), retained_sha256=sha256(cropped)),
            seconds=times, total_seconds=time.monotonic()-started, inference_metrics=metrics,
        )
        write_json(output / "completion.json", completion)
        return completion
    except Exception as exc:
        write_json(output / "failure.json", dict(
            schema="spc-agb-window-failure-v3", task_identity=identity(task), stage=stage,
            type=type(exc).__name__, error=str(exc), seconds=time.monotonic()-started,
        ))
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    names = ["task", "package", "model", "source", "output"]
    for name in names:
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    try:
        result = work(*(getattr(args, n).absolute() for n in names))
        selected = result["task_identity"]
        print(f"{selected['grid_id']}/{result['window']}: tile products complete")
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
