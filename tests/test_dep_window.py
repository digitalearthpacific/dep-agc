"""Small portability/mask/metadata fixtures; no model import or dispatch."""
from unittest.mock import patch
import os
from pathlib import Path
import tempfile
import unittest

from affine import Affine
import numpy as np
import pystac
from pystac.validation.local_validator import LocalValidator
import rasterio

from tasks.loading import local_pin, native_mosaic
from tasks.assemble_tile import assemble_tile
from tasks.inputs.metadata import validate_item
from tasks.outputs import UNITS, lossless_cog
from tasks.processing import CarbonProcessor
from carbon.io import sha256, write_json


class DepWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('SPC_TEST_TMP'))
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cache_escape_and_hash(self):
        with self.assertRaises(ValueError):
            local_pin(self.root,dict(path='../outside',sha256='bad'))
        p=self.root/'input';p.write_bytes(b'original')
        with self.assertRaises(ValueError):
            local_pin(self.root,dict(path='input',sha256='bad'))

    def test_overlap_rejected_and_timeout_cannot_expand(self):
        rec={'bands':{'red':dict(grid=[10,0,0,0,-10,20],shape=[2,2],crs=32702,
                                nodata=0,native={'path':'a'},valid={'path':'b'})}}
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            native_mosaic(self.root/'bad.vrt',[rec,rec],'red')
        with self.assertRaises(ValueError):
            CarbonProcessor(Path('python'),self.root,self.root,self.root,self.root,0)

    def make_raster(self, path, dtype='float32', rgb=False, crs='EPSG:32702', grid=Affine(10,0,400000,0,-10,8470000)):
        values=np.arange(64*64,dtype=dtype).reshape(64,64)
        values[0,:4]=[0,1,2,3]
        mask=np.ones((64,64),bool);mask[1,0]=False
        count=3 if rgb else 1
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),rasterio.open(path,'w',driver='GTiff',
                width=64,height=64,count=count,dtype=dtype,crs=crs,
                transform=grid,nodata=None) as ds:
            ds.write(np.stack([values]*count));ds.write_mask(mask.astype('uint8')*255)
        return values,mask

    def test_lossless_zero_invalid_dtype_and_overviews(self):
        source=self.root/'source.tif';values,mask=self.make_raster(source,'float64')
        target=self.root/'output.tif';record=lossless_cog(source,target)
        self.assertTrue(record['native_values_masks_metadata_exact'])
        with rasterio.open(target) as ds:
            self.assertEqual(ds.dtypes,('float64',))
            self.assertEqual(ds.read(1)[0,0],0)
            self.assertEqual(ds.dataset_mask()[0,0],255)
            self.assertEqual(ds.dataset_mask()[1,0],0)
            self.assertEqual(ds.overviews(1),[2,4,8])
        with self.assertRaises(FileExistsError):lossless_cog(source,target)

    def test_stac_local_annual_interval_assets_and_lineage(self):
        native = self.root/'worker/native'
        native.mkdir(parents=True)
        crs = 'EPSG:3832'
        affine = Affine(10, 0, 4200000, 0, -10, -1504000)
        key, tile_id = 'r00c00', '075_025'
        grid_id = f'PACIFIC_GRID_10/{tile_id}'
        bounds = [4200000, -1504640, 4200640, -1504000]
        grid = dict(id=grid_id, crs=crs, tile_affine=list(affine)[:6], tile_shape=[64, 64],
                    windows=[dict(id=key, retained_shape=[64, 64], global_input_window=[0, 0, 512, 512],
                                  bounds=bounds)])
        files = {}
        for name in UNITS:
            dtype = {'RGB': 'uint8', 'status': 'uint8', 'WorldCover': 'uint8', 'count': 'uint16'}.get(name, 'float32')
            path = native/f'{name}.tif'
            self.make_raster(path, dtype, name == 'RGB', crs, affine)
            files[name] = {'sha256': sha256(path)}
        declared = dict(id=grid_id, crs=crs, retained_shape=[64, 64], context=224,
                        affine=list(affine * Affine.translation(-224, -224))[:6],
                        receiving_window=[0, 0, 512, 512])
        write_json(native.parent/'completion.json', dict(status='complete', window=key, year=2024,
                   source_version='0.4.0', declared_grid=declared, files=files,
                   identities=dict(model='fixture', source_commit='fixture', source_sha256='fixture')))
        first, second, unused = ('dep_s2_geomad_075_025_2024', 'dep_s2_geomad_075_026_2024',
                                 'dep_s2_geomad_076_025_2024')
        stac_root = 'https://stac.digitalearthpacific.org/collections/dep_s2_geomad/items'
        asset_root = 'https://dep-public-data.s3.us-west-2.amazonaws.com/dep_s2_geomad/0-4-0'
        sources = []
        for source_id in (first, second, unused):
            source = pystac.Item(source_id, None, None, None, collection='dep_s2_geomad',
                properties={'start_datetime': '2024-01-01T00:00:00Z', 'end_datetime': '2024-12-31T23:59:59Z',
                            'proj:epsg': 3832, 'proj:shape': [9600, 9600],
                            'proj:transform': [10, 0, 0, 0, -10, 0]})
            # One used Item has no self link; the other points to the unused Item.
            if source_id != first:
                source.set_self_href(f'{stac_root}/{unused}')
            for band in ['red', 'green', 'blue', 'nir', 'count']:
                count = band == 'count'
                source.add_asset(band, pystac.Asset(f'{asset_root}/{source_id}/{band}.tif',
                    extra_fields={'raster:bands': [dict(scale=1 if count else .0001, offset=0,
                                                       nodata=9999 if count else 0, data_type='uint16')]}))
            sources.append(source.to_dict())
        expected_links = [f'{stac_root}/{first}', f'{stac_root}/{second}']
        validated = [validate_item(source, 2024, '0.4.0') for source in sources]
        output = self.root/'product'
        with patch('tasks.assemble_tile.pacific_tile_plan', return_value=grid):
            with self.assertRaisesRegex(ValueError, 'distinct'):
                assemble_tile(tile_id, 2024, {key: native.parent}, [validated[0]] * 2, output)
            record = assemble_tile(tile_id, 2024, {key: native.parent}, validated[:2], output)

        item = pystac.Item.from_file(output/record['item'])
        LocalValidator().item_validator().validate(item.to_dict())
        self.assertEqual(item.id, tile_id)
        self.assertIsNone(item.properties['datetime'])
        self.assertEqual(item.properties['start_datetime'], '2024-01-01T00:00:00Z')
        self.assertEqual(item.properties['end_datetime'], '2024-12-31T23:59:59Z')
        self.assertEqual(set(item.assets), set(UNITS))
        self.assertEqual(item.assets['carbon'].extra_fields['raster:bands'][0]['unit'], 't C/ha')
        moved = self.root/'relocated-product'
        output.rename(moved)
        relocated = pystac.Item.from_file(moved/record['item'])
        for loaded in (item, relocated):
            self.assertEqual([link.get_href() for link in loaded.get_links('derived_from')], expected_links)
        for asset in relocated.assets.values():
            self.assertTrue(Path(asset.get_absolute_href()).is_file())

    def test_status_categories_survive_cog(self):
        source=self.root/'status.tif';self.make_raster(source,'uint8')
        with rasterio.open(source,'r+') as ds:
            data=ds.read(1);data[0,:5]=[0,1,4,5,7];ds.write(data,1)
        target=self.root/'status-cog.tif';lossless_cog(source,target)
        with rasterio.open(target) as ds:
            self.assertEqual(ds.read(1)[0,:5].tolist(),[0,1,4,5,7])
            self.assertTrue((ds.read_masks(1)[0,:5]==255).all())


if __name__=='__main__':unittest.main()
