"""Retained numeric definitions with explicit annual validation; see origins.json."""
import numpy as np
from affine import Affine
from .policy import validate_policy

def encode(raw,valid,scale,offset):
    raw=np.asarray(raw)
    if raw.shape[0]!=3 or valid.shape!=raw.shape[1:] or valid.dtype!=bool or not np.isfinite([scale,offset]).all() or scale<=0:raise ValueError('RGB radiometry/support differs')
    physical=raw.astype(np.float64)*scale+offset
    rgb=np.clip(np.floor(physical*(255/.25)),1,255).astype(np.uint8)
    rgb[:,~valid]=0
    return rgb

def support(rgb,rgb_valid,count,count_valid):
    if rgb.shape[0]!=3 or rgb_valid.shape!=rgb.shape or count.shape!=rgb.shape[1:] or count_valid.shape!=count.shape:raise ValueError('Composite support grids differ')
    return np.all(rgb_valid&(rgb!=0),axis=0)&count_valid&(count>0)

def calculate_ndvi(red, nir, red_valid, nir_valid, red_parameters, nir_parameters):
    if red.shape != nir.shape or any(a.shape != red.shape or a.dtype != bool for a in [red_valid, nir_valid]):
        raise ValueError('Band/validity grids differ')
    rs, ro, rn = red_parameters; ns, no, nn = nir_parameters
    source = red_valid & nir_valid & (red != rn) & (nir != nn)
    r = red.astype(np.float64) * rs + ro
    n = nir.astype(np.float64) * ns + no
    finite = np.isfinite(r) & np.isfinite(n)
    with np.errstate(invalid='ignore', over='ignore', divide='ignore'):
        denominator = n + r
        # Numerical cancellation only, not an empirical dark-pixel/noise threshold.
        epsilon = 8 * np.finfo(np.float64).eps * np.maximum(1, np.abs(n) + np.abs(r))
        bad_denominator = ~np.isfinite(denominator) | (np.abs(denominator) <= epsilon)
        supported = source & finite & ~bad_denominator
        ratio = np.full(red.shape, np.nan, np.float64)
        np.divide(n-r, denominator, out=ratio, where=supported)
        stored = ratio.astype(np.float32)
    valid = supported & np.isfinite(stored)
    ndvi = stored.copy(); ndvi[~valid] = -9999
    return ndvi, valid

def product_masks(land,usable,source,height_values,model_valid,vi,vi_valid,unknown=None,policy=None):
    """No forest/class/height gate. Separate priority reasons plus independent bits."""
    validate_policy(policy)
    masks=[land,usable,source,model_valid,vi_valid]
    if any(x.shape!=height_values.shape or x.dtype!=bool for x in masks) or vi.shape!=height_values.shape:
        raise ValueError('Product support grids/types differ')
    if np.any(usable & ~land) or np.any(model_valid & (~np.isfinite(height_values)|(height_values<0))):
        raise ValueError('Invalid usable-land/model support')
    assessed=land & usable & source & model_valid & vi_valid & (vi>=np.float32(.3))
    status=np.zeros(land.shape,np.uint8)  # 0 non-land
    status[land]=1  # unavailable model/source/NDVI
    status[land & source & model_valid & vi_valid & ~usable]=2  # unusable support (reserved status; no scene mask)
    status[land & source & model_valid & vi_valid & usable & (vi<np.float32(.3))]=3
    status[assessed]=4
    if unknown is not None:
        if unknown.shape!=land.shape or np.any(unknown & land):raise ValueError('Unknown land support conflicts')
        status[unknown]=5
    density=np.full(land.shape,-9999,np.float32)
    density[assessed]=(3.744*height_values[assessed].astype(np.float64)**1.391).astype(np.float32)
    assumed=status==3;status[assumed]=7;density[assumed]=0
    assessed=assessed | assumed
    return assessed,status,density

def crop_height(height, source_valid, transform, crop):
    """Crop already-converted runtime output; source validity alone sets nodata."""
    if height.ndim != 2 or height.shape != source_valid.shape or min(height.shape) < 512:
        raise ValueError("Runtime dimensions differ from input")
    if not np.isfinite(height[source_valid]).all() or (height[source_valid] < 0).any():
        raise ValueError("Nonfinite or negative runtime height at source-valid pixels")
    col, row, width, length = crop
    if min(col, row) < 0 or width <= 0 or length <= 0 or col + width > height.shape[1] or row + length > height.shape[0]:
        raise ValueError("Crop exceeds diagnostic grid")
    valid = source_valid[row:row + length, col:col + width].copy()
    out = height[row:row + length, col:col + width].astype("float32", copy=True)
    out[~valid] = -9999
    return out, valid, transform * Affine.translation(col, row)

