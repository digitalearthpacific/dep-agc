"""DEP Writer contract: lossless native COGs and supported STAC metadata flow."""

from datetime import datetime, timezone
import importlib.metadata as metadata
from pathlib import Path
import sys

from dep_tools.namers import GenericItemPath
from dep_tools.stac_utils import StacCreator, set_derived_from
from dep_tools.writers import LocalStacWriter, Writer
import numpy as np
import pystac
import rasterio
from rasterio.shutil import copy as rio_copy
from rio_cogeo.cogeo import cog_validate
import xarray as xr

from spc_agb_inputs.metadata import GEOMAD_ITEMS_URL, validate_item
from spc_agb_worker.io import read_json, sha256, write_json
from .inputs import local_pin

UNITS = {'carbon':'t C/ha', 'height':'m', 'physical-area-ha':'ha', 'NDVI':'1',
         'count':'observations', 'owner':'1', 'WorldCover':'1', 'status':'1', 'RGB':'1'}
STATUS = {'0':'excluded water', '1':'unavailable inputs', '4':'modelled (including genuine zero)',
          '5':'unknown classification', '6':'outside owner', '7':'NDVI-assumed zero'}


class LocalReviewPath(GenericItemPath):
    def stac_path(self, item_id, absolute=True, **kwargs):
        return super().stac_path(item_id, absolute=absolute, **kwargs)


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


class NativeCogStacWriter(Writer):
    def __init__(self, root, index_root):
        self.root, self.index_root = root, index_root

    def write(self, result, item_id):
        task, completion = result['task'], result['completion']
        index = result['index']
        # The index carries the country source inventory; native records identify this window's inputs.
        source_ids = sorted({record['item'] for record in index['records']})
        source_items = {original['id']: original for original in index['source_items']}
        sources = []
        for source_id in source_ids:
            original = validate_item(source_items[source_id], task['year'], task['source_version'])
            sources.append(pystac.Item.from_dict(original, href=f'{GEOMAD_ITEMS_URL}/{source_id}'))
        self.root.mkdir(parents=True, exist_ok=False)
        # GenericItemPath only names local files; no bucket, AWS client or publication.
        release = read_json(Path(__file__).resolve().parents[1] / 'release.json')
        names = LocalReviewPath(sensor='s2', dataset_id=release['stac']['dataset_id'], version=release['version'],
                                time=str(task['year']), prefix=release['stac']['prefix'],
                                zero_pad_numbers=False, full_path_prefix=self.root)
        checks, variables = {}, {}
        for name in completion['files']:
            target = Path(names.path(item_id,name,absolute=True))
            checks[name] = lossless_cog(result['native']/(name+'.tif'),target)
            with rasterio.open(target) as src:
                values = src.read()
                # Metadata carrier for DEP StacCreator; these values are never rewritten.
                variables[name] = xr.DataArray(values if src.count>1 else values[0],
                                               dims=('band','y','x') if src.count>1 else ('y','x'))
        data = xr.Dataset(variables)
        now = datetime.now(timezone.utc).isoformat().replace('+00:00','Z')
        data.attrs['stac_properties'] = dict(
            start_datetime=f"{task['year']}-01-01T00:00:00Z",
            end_datetime=f"{task['year']}-12-31T23:59:59Z", created=now,
            **{'agc:scope':'one retained processing window; country completeness requires separate assembly',
               'agc:software_release':release['version'],
               'agc:country':task['country'], 'agc:grid_id':task['grid']['id'],
               'agc:window':task['window'], 'agc:source_product':task['source_product'],
               'agc:source_version':task['source_version'], 'agc:statuses':STATUS,
               'agc:stock_tC':completion['summary']['stock_tC'],
               'agc:classified_ha':completion['summary']['assessed_ha'],
               'agc:accounting':'Native densities and supplied physical-area support; no extrapolation',
               'agc:permission':'GADM permission unverified; private local engineering',
               'agc:completion_kind':completion.get('completion_kind', 'model')})
        # DEP StacCreator propagates stac_properties and reads raster/projection metadata.
        set_derived_from(sources, data)
        item = StacCreator(names, with_raster=True).process(data,item_id)
        item.properties['datetime'] = None  # Annual interval, not a daily acquisition.
        item.properties['start_datetime'] = data.attrs['stac_properties']['start_datetime']
        item.properties['end_datetime'] = data.attrs['stac_properties']['end_datetime']
        item.remove_links('collection')  # Replace helper's default remote staging URL below.
        for name, asset in item.assets.items():
            asset.roles = ['visual'] if name=='RGB' else ['data']
            asset.extra_fields['agc:sha256'] = checks[name]['sha256']
            asset.extra_fields['agc:validity'] = 'Internal native mask; zero is not NoData'
            for band in asset.extra_fields.get('raster:bands',[]):
                band['unit'] = UNITS[name]
        # Records are output evidence, not expected data supplied to processing.
        evidence = self.root/'evidence'; evidence.mkdir()
        write_json(evidence/'completion.json',completion)
        write_json(evidence/'lineage.json',dict(
            task_identity=completion['task_identity'], identities=completion['identities'],
            ownership_grid=task.get('ownership_grid'),
            native_source_items=source_ids,
            native_index_sha256=result['native_index_sha256'],
            processor_code={str(p.relative_to(Path(__file__).resolve().parents[1])):sha256(p)
                for folder in ['spc_agb_dep','spc_agb_inputs']
                for p in sorted((Path(__file__).resolve().parents[1]/folder).glob('*.py'))},
            runtime=dict(python=sys.version,packages={name:metadata.version(name) for name in
                ['dep-tools','numpy','rasterio','pyproj','shapely','rio-cogeo','pystac','odc-geo']}),
            loader_receipt_sha256=sha256(result['loader_receipt']),
            source_receipt='load.json', scientific_core=dict(
                worker_version=completion['worker_version'], executed_code_sha256=completion['identities']['code'],
                software_release=release['version'], comparison_reference='4277015',
                numerical_equivalence='pending; not established by source inspection'),
            metadata_flow='dep-tools StacCreator / stac_properties / set_derived_from',
            provenance_API_question='DEP to confirm any additional supported lineage helper/extension'))
        import shutil
        shutil.copyfile(result['loader_receipt'],evidence/'load.json')
        support = local_pin(self.index_root,index['support'])
        shutil.copyfile(support,evidence/'accounting-support.json')
        for name in ['completion','lineage','load','accounting-support']:
            p = evidence/(name+'.json')
            item.add_asset(name,pystac.Asset(str(p),media_type='application/json',roles=['metadata'],
                           extra_fields={'agc:sha256':sha256(p)}))
        source_dir = self.root/'sources'; source_dir.mkdir()
        for original in index['source_items']:
            p = source_dir/(original['id']+'.json'); write_json(p,original)
        collection = pystac.Collection(names.item_prefix,
            'Private local annual carbon retained window; country completeness requires separate assembly. '
            'Indicative; roots/soil/dead organic matter excluded; unavailable/unknown/water excluded; '
            'independent accuracy and sharing permission not established.',
            pystac.Extent(pystac.SpatialExtent([item.bbox]),pystac.TemporalExtent([[
                datetime(task['year'],1,1,tzinfo=timezone.utc),datetime(task['year'],12,31,23,59,59,tzinfo=timezone.utc)]])),
            license='proprietary')
        collection.set_self_href(str(self.root/'collection.json'))
        collection.add_item(item)
        # Collection.add_item may apply its layout; keep DEP's explicit namer path.
        item.set_self_href(names.stac_path(item_id))
        item.make_asset_hrefs_relative()
        LocalStacWriter(names).write(item,item_id)
        # Standard PySTAC serialization keeps navigation portable outside the
        # container's output mount; native assets are not rewritten here.
        collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
        write_json(self.root/'cog-checks.json',checks)
        return dict(item=str(Path(item.get_self_href()).relative_to(self.root.parent)),
                    collection=str(Path(collection.get_self_href()).relative_to(self.root.parent)),
                    cog_checks=checks,summary=completion['summary'],worker_seconds=result['worker_seconds'])
