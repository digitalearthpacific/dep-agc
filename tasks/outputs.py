"""DEP Writer contract: lossless native COGs and supported STAC metadata flow."""

from pathlib import Path

import numpy as np
import rasterio
from rasterio.shutil import copy as rio_copy
from rio_cogeo.cogeo import cog_validate

from carbon.io import sha256

UNITS = {'carbon':'t C/ha', 'height':'m', 'NDVI':'1',
         'count':'observations', 'WorldCover':'1', 'status':'1', 'RGB':'1'}
STATUS = {'0':'excluded water', '1':'unavailable inputs', '4':'modelled (including genuine zero)',
          '5':'unknown classification', '7':'NDVI-assumed zero'}


def lossless_cog(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(target)
    with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True):
        rio_copy(source, target, driver='COG', compress='DEFLATE', blocksize=512,
                 overview_count=3, overview_resampling='NEAREST', num_threads=1)
    valid, errors, warnings = cog_validate(target, strict=True, quiet=True)
    if not valid:
        raise ValueError(f'Invalid COG: {errors}; {warnings}')
    with rasterio.open(source) as a, rasterio.open(target) as b:
        if (a.crs,a.transform,a.shape,a.dtypes,a.nodatavals,a.colorinterp) != (
                b.crs,b.transform,b.shape,b.dtypes,b.nodatavals,b.colorinterp):
            raise ValueError('COG native metadata differs')
        for _, window in a.block_windows(1):
            if (not np.array_equal(a.read(window=window),b.read(window=window))
                    or not np.array_equal(a.read_masks(window=window),b.read_masks(window=window))):
                raise ValueError('COG native values/masks differ')
        if not np.array_equal(a.dataset_mask(),b.dataset_mask()):
            raise ValueError('COG dataset mask differs')
        overviews = b.overviews(1)
    if Path(str(target)+'.msk').exists():
        raise ValueError('COG mask must be internal')
    return dict(sha256=sha256(target),bytes=target.stat().st_size,
                native_values_masks_metadata_exact=True,strict_cog_valid=True,
                overviews=overviews,overview_resampling='nearest; display only')
