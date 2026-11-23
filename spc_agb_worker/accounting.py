"""Native ownership, masks and weighted accounting; no inference or execution state."""

import numpy as np
from affine import Affine

from .grid import owner_placement, validate_owner_ids
from .io import local_input, sha256, write_raster
from .policy import LAND_CLASSES
from .science import product_masks


def domain(package, task):
    """Accepted rectangle assignment with task-derived placement and duplicate guard."""
    validate_owner_ids(task["inputs"]["owners"])
    h, w = task["grid"]["retained_shape"]
    cover = np.zeros((h, w), np.uint8)
    weights = np.zeros((h, w), np.float64)
    owner = np.zeros((h, w), np.uint16)
    for row in task["inputs"]["owners"]:
        with np.load(local_input(package, row["file"]["path"])) as z:
            values = {k: z[k].copy() for k in z.files}
        shape = values["projected"].shape
        if (len(shape) != 2 or any(values[k].shape != shape
                                   for k in ["physical_area_ha", "support"])):
            raise ValueError("Owner array shapes differ")
        x, y = owner_placement(task["grid"], row["bounds"], shape)
        x0 = max(0, x)
        y0 = max(0, y)
        x1 = min(w, x + shape[1])
        y1 = min(h, y + shape[0])
        if x1 <= x0 or y1 <= y0:
            continue
        dst = np.s_[y0:y1, x0:x1]
        src = np.s_[y0-y:y1-y, x0-x:x1-x]
        if owner[dst].any():
            raise ValueError("Duplicated rectangular ownership")
        cover[dst] = values["projected"][src]
        weights[dst] = values["physical_area_ha"][src]
        owner[dst] = row["numeric_id"]
    land = (weights > 0) & np.isin(cover, LAND_CLASSES)
    return dict(worldcover=cover, weights=weights, owner=owner, land=land)


def totals(mask, domain, status, assessed, carbon, height_valid, unknown):
    """Same arithmetic and reductions as account_chunk's nested totals()."""
    w = domain["weights"]
    land = domain["land"]
    return dict(
        owned_ha=float(w[mask].sum()), known_land_ha=float(w[mask & land].sum()),
        water_ha=float(w[mask & (domain["worldcover"] == 80)].sum()),
        unknown_ha=float(w[mask & unknown].sum()),
        modelled_ha=float(w[mask & (status == 4)].sum()),
        assumed_zero_ha=float(w[mask & (status == 7)].sum()),
        unavailable_ha=float(w[mask & land & ~assessed].sum()),
        assessed_ha=float(w[mask & assessed].sum()),
        stock_tC=float(np.dot(w[mask & assessed], carbon[mask & assessed].astype(float))),
        height_valid_ha=float(w[mask & land & height_valid].sum()),
    )


def account(package, task, arrays, raw_height, receiving_grid, output):
    d = task["grid"]["context"]
    h, w = task["grid"]["retained_shape"]
    sl = np.s_[d:d+h, d:d+w]
    dom = domain(package, task)
    height = raw_height[sl]
    hv = arrays["rgb_valid"][sl] & np.isfinite(height) & (height >= 0)
    source = arrays["rgb_valid"][sl] & arrays["red_valid"][sl] & arrays["nir_valid"][sl]
    land = dom["land"]
    unknown = (dom["weights"] > 0) & (dom["worldcover"] == 0)
    assessed, status, carbon = product_masks(
        land, land & arrays["rgb_valid"][sl], source, height, hv,
        arrays["ndvi"][sl], arrays["ndvi_valid"][sl], unknown, policy=task["policy"],
    )
    status[dom["weights"] == 0] = 6
    grid = receiving_grid * Affine.translation(d, d)
    owned = dom["weights"] > 0
    layers = [
        ("RGB", arrays["rgb"][:, sl[0], sl[1]], owned & arrays["rgb_valid"][sl], None),
        ("height", height, land & hv, -9999),
        ("carbon", carbon, assessed, -9999),
        ("status", status, owned, None),
        ("NDVI", arrays["ndvi"][sl], land & arrays["ndvi_valid"][sl], -9999),
        ("count", arrays["count"][sl], owned & arrays["count_valid"][sl], None),
        ("WorldCover", dom["worldcover"], owned, None),
        ("physical-area-ha", dom["weights"], owned, None),
        ("owner", dom["owner"], owned, None),
    ]
    output.mkdir(parents=True, exist_ok=False)
    files = {}
    for name, value, valid, nodata in layers:
        path = output / (name + ".tif")
        write_raster(path, value, valid, grid, task["grid"]["crs"], nodata)
        files[name] = dict(path=path.name, sha256=sha256(path))
    summary = totals(owned, dom, status, assessed, carbon, hv, unknown)
    owners = {
        row["id"]: totals(dom["owner"] == row["numeric_id"], dom, status,
                          assessed, carbon, hv, unknown)
        for row in task["inputs"]["owners"]
    }
    return dict(summary=summary, owners=owners, files=files,
                grid=list(grid)[:6], crs=task["grid"]["crs"])
