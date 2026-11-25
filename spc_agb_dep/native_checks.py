"""Independent saved-layer checks, for the declared processing task.

No model, accountant or worker preparation import. Expected values are rebuilt
from source bands and supplied arrays, not from the generated summary.
"""
from pathlib import Path
import json

from affine import Affine
import numpy as np
import rasterio
from rasterio.windows import Window

LAND = [10,20,30,40,50,60,70,90,95,100]


def cog_native(source, target):
    """Saved-only lossless checks from verify_dep_window; no output generation."""
    from rio_cogeo.cogeo import cog_validate
    valid, errors, warnings = cog_validate(target,strict=True,quiet=True)
    if not valid:
        raise ValueError('Invalid saved COG: '+str((errors,warnings)))
    with rasterio.open(source) as a,rasterio.open(target) as b:
        if (a.crs,a.transform,a.shape,a.dtypes,a.nodatavals,a.colorinterp)!=(b.crs,b.transform,b.shape,b.dtypes,b.nodatavals,b.colorinterp):
            raise ValueError('Saved COG/native metadata differs')
        for _,window in a.block_windows(1):
            if not np.array_equal(a.read(window=window),b.read(window=window)) or not np.array_equal(a.read_masks(window=window),b.read_masks(window=window)):
                raise ValueError('Saved COG/native values/masks differ')
        if not np.array_equal(a.dataset_mask(),b.dataset_mask()) or b.overviews(1)!=[2,4,8]:
            raise ValueError('Saved COG mask/overview contract differs')
    return dict(strict_COG=True,native_values_masks_metadata_exact=True,overviews=[2,4,8],overview_accounting=False)


def supplied_domain(package, task):
    """Independent placement; same saved-verification arithmetic as aa67de1."""
    height, width = task['grid']['retained_shape']
    grid = Affine(*task['grid']['affine'])*Affine.translation(
        task['grid']['receiving_window'][0]+task['grid']['context'],
        task['grid']['receiving_window'][1]+task['grid']['context'])
    result = dict(owner=np.zeros((height,width),np.uint16),
        worldcover=np.zeros((height,width),np.uint8), weights=np.zeros((height,width),np.float64))
    from spc_agb_worker.io import read_json, sha256
    order = read_json(package/task['inputs']['owner_order']['path']) if 'owner_order' in task['inputs'] else None
    occupied = np.zeros((height,width),bool)
    for row in task['inputs']['owners']:
        if order is not None and row['numeric_id'] != order.index(row['id'])+1:
            raise ValueError('Independent owner order differs')
        path = package/row['file']['path']
        if row['file'].get('sha256') and sha256(path)!=row['file']['sha256']:
            raise ValueError('Independent supplied owner hash differs')
        with np.load(path,allow_pickle=False) as source:
            x,y = (~grid)*(row['bounds'][0],row['bounds'][3])
            if abs(x-round(x))>1e-8 or abs(y-round(y))>1e-8:
                raise ValueError('Independent owner alignment differs')
            x,y = int(round(x)),int(round(y))
            h,w = source['projected'].shape
            x0,y0,x1,y1 = max(x,0),max(y,0),min(x+w,width),min(y+h,height)
            if x1<=x0 or y1<=y0:
                raise ValueError('Independent nonintersecting owner')
            dst=np.s_[y0:y1,x0:x1];src=np.s_[y0-y:y1-y,x0-x:x1-x]
            if occupied[dst].any():
                raise ValueError('Independent duplicate rectangular ownership')
            occupied[dst]=True
            for key,array in [('owner',None),('worldcover','projected'),('weights','physical_area_ha')]:
                result[key][dst] = row['numeric_id'] if array is None else source[array][src]
            if not np.array_equal(source['support'],source['physical_area_ha']>0):
                raise ValueError('Independent support/weights differ')
    result['land']=(result['weights']>0)&np.isin(result['worldcover'],LAND)
    return result


def generated_inputs(package, task, worker, dom, no_model):
    grid=task['grid'];receiving=Affine(*grid['affine'])*Affine.translation(*grid['receiving_window'][:2])
    d=grid['context'];h,w=grid['retained_shape'];sl=np.s_[d:d+h,d:d+w]
    receiving_shape = (h + 2*d, w + 2*d)
    with rasterio.open(package/task['inputs']['canonical']['path']) as ds:
        values=ds.read(window=Window(*grid['receiving_window']))
    if values.shape!=(10, *receiving_shape):
        raise ValueError('Complete receiving dimensions/bands differ')
    raw=values[[0,2,4]]
    rgb_valid=np.all((values[[1,3,5]]!=0)&(raw!=0),axis=0)&(values[9]!=0)&(values[8]>0)
    rgb=np.clip(np.floor((raw.astype(np.float64)*.0001)*1020),1,255).astype(np.uint8);rgb[:,~rgb_valid]=0
    red,nir=values[0].astype(np.float64)*.0001,values[6].astype(np.float64)*.0001
    denominator=nir+red;epsilon=8*np.finfo(np.float64).eps*np.maximum(1,np.abs(nir)+np.abs(red))
    vi_valid=(values[1]!=0)&(values[7]!=0)&(values[0]!=0)&(values[6]!=0)&(np.abs(denominator)>epsilon)
    vi=np.full(red.shape,-9999,np.float32);vi[vi_valid]=((nir[vi_valid]-red[vi_valid])/denominator[vi_valid]).astype(np.float32)
    with rasterio.open(worker/'prepared/tci-input.tif') as ds:
        if (ds.shape!=receiving_shape or ds.transform!=receiving or ds.crs!=rasterio.crs.CRS.from_user_input(grid['crs'])
                or not np.array_equal(ds.read(),rgb) or not np.array_equal(ds.dataset_mask()>0,rgb_valid)):
            raise ValueError('Independent receiving RGB encoding/validity differs')
    completion=json.loads((worker/'completion.json').read_text())
    with rasterio.open(worker/completion['height']['raw']) as ds:
        if ds.shape!=receiving_shape or ds.transform!=receiving or ds.crs!=rasterio.crs.CRS.from_user_input(grid['crs']) or ds.dtypes!=('float32',):
            raise ValueError('Independent full-context height grid differs')
        height=ds.read(1)
        if no_model:
            if ds.dataset_mask().any() or np.any(height!=-9999):
                raise ValueError('No-model height must remain unavailable')
        elif not (np.isfinite(height).all() and (height[rgb_valid]>=0).all()
                  and (height[~rgb_valid]==0).all() and (ds.read_masks()==255).all() and ds.nodata is None):
            raise ValueError('Independent full-context height/source validity differs')
    crop_height=height[sl];valid_crop=rgb_valid[sl] if not no_model else np.zeros((h,w),bool)
    with rasterio.open(worker/completion['height']['retained']) as ds:
        a=ds.read(1)
        if (ds.transform!=receiving*Affine.translation(d,d) or ds.shape!=(h,w) or ds.nodata!=-9999
                or not np.array_equal(ds.dataset_mask()>0,valid_crop)
                or not np.array_equal(a[valid_crop],crop_height[valid_crop]) or np.any(a[~valid_crop]!=-9999)):
            raise ValueError('Independent retained crop differs')
    owned=dom['weights']>0;land=dom['land'];hv=rgb_valid[sl]&np.isfinite(crop_height)&(crop_height>=0)
    usable=land&rgb_valid[sl]&(values[1][sl]!=0)&(values[7][sl]!=0)&hv&vi_valid[sl]
    status=np.zeros((h,w),np.uint8);status[land]=1
    status[usable&(vi[sl]>=np.float32(.3))]=4;status[usable&(vi[sl]<np.float32(.3))]=7
    status[owned&(dom['worldcover']==0)]=5;status[~owned]=6
    expected_masks=dict(RGB=owned&rgb_valid[sl],height=land&hv,carbon=np.isin(status,[4,7]),status=owned,
        NDVI=land&vi_valid[sl],count=owned&(values[9][sl]!=0),WorldCover=owned,owner=owned,**{'physical-area-ha':owned})
    expected_values=dict(RGB=rgb[:,sl[0],sl[1]],height=crop_height,status=status,NDVI=vi[sl],count=values[8][sl],
        WorldCover=dom['worldcover'],owner=dom['owner'],**{'physical-area-ha':dom['weights']})
    for name,mask in expected_masks.items():
        with rasterio.open(worker/'native'/(name+'.tif')) as ds:
            if (ds.transform!=receiving*Affine.translation(d,d) or ds.shape!=(h,w)
                    or not np.array_equal(ds.dataset_mask()>0,mask)
                    or any(not np.array_equal(m>0,mask) for m in ds.read_masks())):
                raise ValueError('Independent native grid/mask differs: '+name)
            actual=ds.read() if name=='RGB' else ds.read(1)
            if name in expected_values and not np.array_equal(actual,expected_values[name]):
                raise ValueError('Independent native values differ: '+name)
            if name=='carbon' and np.any(actual[~mask]!=-9999):
                raise ValueError('Unavailable carbon is not NoData')
    return dict(complete_receiving_RGB_exact=True,native_masks_status_values_exact=True,
        raw_height_grid_valid=True,retained_crop_exact=True,no_model=no_model)
