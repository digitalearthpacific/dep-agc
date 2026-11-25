"""Declared spatial contract; no CRS selection, reprojection or weight creation."""

import math

from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import array_bounds


MODEL_PATCH = 512
MODEL_STRIDE = 64
MODEL_CONTEXT = 224
DEP_CORE = 1920


def identity(task):
    """Available task identity, also retained when validation fails."""
    if not isinstance(task, dict):
        return {"task_schema": None, "grid_id": None, "window": None, "year": None, "source_version": None}
    grid = task.get("grid")
    return {
        "task_schema": task.get("schema"),
        "year": task.get("year"),
        "source_version": task.get("source_version"),
        "grid_id": grid.get("id") if isinstance(grid, dict) else None,
        "window": task.get("window"),
    }


def receiving_transform(grid):
    return Affine(*grid["affine"]) * Affine.translation(
        *grid["receiving_window"][:2]
    )


def retained_transform(grid):
    d = grid["context"]
    return receiving_transform(grid) * Affine.translation(d, d)


def integer_vector(value, length, name):
    if (not isinstance(value, list) or len(value) != length
            or any(type(v) is not int for v in value)):
        raise ValueError(f"{name} must contain {length} integers")
    return value


def validate_grid(task):
    if task.get("schema") != "spc-agb-window-v3":
        raise ValueError("Task schema v3 required; older tasks are historical, not implicitly migrated")
    grid = task["grid"]
    if not isinstance(grid.get("id"), str) or not grid["id"].startswith("PACIFIC_GRID_10/"):
        raise ValueError("Official PACIFIC_GRID_10 grid id required")
    if not isinstance(task.get("window"), str) or not task["window"].strip():
        raise ValueError("Window identity required")
    crs = CRS.from_user_input(grid["crs"])
    if crs.to_epsg() != 3832:
        raise ValueError("PACIFIC_GRID_10 requires EPSG:3832")
    affine = grid["affine"]
    if (len(affine) != 6 or not all(math.isfinite(v) for v in affine)
            or [affine[i] for i in (0, 1, 3, 4)] != [10, 0, 0, -10]):
        raise ValueError("North-up unrotated 10 m grid required")
    shape = integer_vector(grid["shape"], 2, "Canonical shape")
    receiving = integer_vector(grid["receiving_window"], 4, "Receiving window")
    global_window = integer_vector(grid["global_input_window"], 4, "Global window")
    offset = integer_vector(grid["canonical_global_offset"], 2, "Canonical/global offset")
    integer_vector(grid["retained_shape"], 2, "Retained shape")
    if (grid["patch"], grid["stride"], grid["context"]) != (MODEL_PATCH, MODEL_STRIDE, MODEL_CONTEXT):
        raise ValueError("Accepted patch/context contract required")
    core = grid["retained_shape"]
    if core != [DEP_CORE, DEP_CORE]:
        raise ValueError("Unsupported retained window shape")
    input_size = [side + 2 * MODEL_CONTEXT for side in reversed(core)]
    if receiving[2:] != input_size or global_window[2:] != input_size:
        raise ValueError("Full receiving window required")
    if (min(shape) <= 0 or min(receiving[:2]) < 0
            or receiving[0] + receiving[2] > shape[1]
            or receiving[1] + receiving[3] > shape[0]):
        raise ValueError("Receiving window exceeds canonical grid")
    if any(v % MODEL_STRIDE for v in global_window[:2]):
        raise ValueError("Global patch phase differs")
    if [receiving[i] - offset[i] for i in range(2)] != global_window[:2]:
        raise ValueError("Canonical/global offsets disagree")
    bounds = grid["bounds"]
    expected = list(array_bounds(*grid["retained_shape"], retained_transform(grid)))
    if bounds != expected:
        raise ValueError("Retained bounds/grid disagree")
    return crs
