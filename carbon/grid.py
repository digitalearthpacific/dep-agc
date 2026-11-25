"""Declared spatial contract; no CRS selection, reprojection or weight creation."""

import math

from affine import Affine
from rasterio.crs import CRS
from rasterio.transform import array_bounds


MODEL_PATCH = 512
MODEL_STRIDE = 64
MODEL_CONTEXT = 224
DEP_CORE = 1920
LEGACY_CORE = 2048


def is_tile_task(task):
    grid = task["grid"]
    return grid.get("retained_shape") == [DEP_CORE, DEP_CORE] and grid.get("id", "").startswith("PACIFIC_GRID_10/")


def identity(task):
    """Available task identity, also retained when validation fails."""
    if not isinstance(task, dict):
        return {"task_schema": None, "country": None, "grid_id": None, "window": None, "year": None, "source_version": None}
    grid = task.get("grid")
    return {
        "task_schema": task.get("schema"),
        "year": task.get("year"),
        "source_version": task.get("source_version"),
        "country": task.get("country"),
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
    country = task.get("country", {})
    grid = task["grid"]
    if not isinstance(grid.get("id"), str) or not grid["id"].strip():
        raise ValueError("Stable grid id required")
    if not is_tile_task(task) and any(not isinstance(country.get(k), str) or not country[k].strip()
                                     for k in ("code", "name")):
        raise ValueError("Explicit country code/name required")
    if not isinstance(task.get("window"), str) or not task["window"].strip():
        raise ValueError("Window identity required")
    crs = CRS.from_user_input(grid["crs"])
    if not crs.is_projected or crs.linear_units_factor[1] != 1.0:
        raise ValueError("Supported projected CRS with metre units required")
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
    if core != [LEGACY_CORE, LEGACY_CORE] and (crs.to_epsg() != 3832 or core != [DEP_CORE, DEP_CORE]):
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


def validate_owner_ids(rows, order=None):
    """Owner IDs are supplied, never renumbered; output storage is uint16."""
    if order is not None and (len(order) > 65535 or len(order) != len(set(order))):
        raise ValueError("Owner order duplicates or exceeds uint16 capacity")
    if len({row["id"] for row in rows}) != len(rows):
        raise ValueError("Duplicated owner ID")
    numeric = [row["numeric_id"] for row in rows]
    if any(type(v) is not int or not 1 <= v <= 65535 for v in numeric):
        raise ValueError("Owner numeric ID outside uint16 range 1..65535")
    if len(set(numeric)) != len(numeric):
        raise ValueError("Duplicated numeric owner ID")
    if order is not None:
        mapping = {name: i + 1 for i, name in enumerate(order)}
        if any(mapping.get(row["id"]) != row["numeric_id"] for row in rows):
            raise ValueError("Owner IDs/order differ")


def owner_placement(grid, bounds, shape):
    """Integer placement from supplied affine and owner bounds, without rounding drift."""
    left, bottom, right, top = bounds
    inverse = ~retained_transform(grid)
    x, y = inverse * (left, top)
    end_x, end_y = inverse * (right, bottom)
    coords = (x, y, end_x, end_y)
    # Coordinate alignment tolerance only; physical-area values are never changed.
    if (not all(math.isfinite(v) and abs(v - round(v)) <= 1e-8 for v in coords)
            or (round(end_y - y), round(end_x - x)) != tuple(shape)):
        raise ValueError("Owner bounds/shape do not align with task grid")
    return round(x), round(y)
