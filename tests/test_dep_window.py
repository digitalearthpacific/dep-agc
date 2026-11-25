"""Small portability/mask/metadata fixtures; no model import or dispatch."""
import json
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
from tasks.outputs import lossless_cog, NativeCogStacWriter
from tasks.processing import CarbonProcessor
from tasks.run_task import task_id
from carbon.io import sha256, write_json


class DepWindowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('SPC_TEST_TMP'))
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_cache_escape_hash_and_task_identity(self):
        with self.assertRaises(ValueError):
            local_pin(self.root,dict(path='../outside',sha256='bad'))
        p=self.root/'input';p.write_bytes(b'original')
        with self.assertRaises(ValueError):
            local_pin(self.root,dict(path='input',sha256='bad'))
        self.assertEqual(task_id(dict(country={'code':'XYZ'},grid={'id':'other-grid'},window='r+000-c+000',year=2024,source_version='0.4.0')),
                         ('XYZ','other-grid','r+000-c+000','2024'))
        with self.assertRaises(ValueError):
            task_id(dict(country={'code':'../XYZ'},grid={'id':'grid'},window='window',year=2024,source_version='0.4.0'))

    def test_overlap_rejected_and_timeout_cannot_expand(self):
        rec={'bands':{'red':dict(grid=[10,0,0,0,-10,20],shape=[2,2],crs=32702,
                                nodata=0,native={'path':'a'},valid={'path':'b'})}}
        with self.assertRaisesRegex(ValueError,'Overlapping'):
            native_mosaic(self.root/'bad.vrt',[rec,rec],'red')
        with self.assertRaises(ValueError):
            CarbonProcessor(Path('python'),self.root,self.root,self.root,self.root,0)

    def make_raster(self, path, dtype='float32', rgb=False):
        values=np.arange(64*64,dtype=dtype).reshape(64,64)
        values[0,:4]=[0,1,2,3]
        mask=np.ones((64,64),bool);mask[1,0]=False
        count=3 if rgb else 1
        with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True),rasterio.open(path,'w',driver='GTiff',
                width=64,height=64,count=count,dtype=dtype,crs='EPSG:32702',
                transform=Affine(10,0,400000,0,-10,8470000),nodata=None) as ds:
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

    def test_stac_local_annual_interval_support_and_lineage(self):
        native=self.root/'native';native.mkdir()
        for name in ['carbon','height','physical-area-ha','status','owner','WorldCover','count','NDVI','RGB']:
            dtype={'RGB':'uint8','status':'uint8','WorldCover':'uint8','owner':'uint16',
                   'count':'uint16','physical-area-ha':'float64'}.get(name,'float32')
            self.make_raster(native/(name+'.tif'),dtype,name=='RGB')
        support=self.root/'support.json';write_json(support,{'crs':'EPSG:32702','owners':[]})
        receipt=self.root/'load.json';write_json(receipt,{'checked':True})
        completion=dict(files={p.stem:{} for p in native.iterdir()},summary={'stock_tC':1.,'assessed_ha':2.},
                        task_identity={'country':{'code':'XYZ'},'grid_id':'other'},
                        worker_version='synthetic',identities={'code':{'fixture.py':'test-hash'}})
        task=dict(year=2024,country={'code':'XYZ','name':'Synthetic'},window='r+000-c+000',
                  grid={'id':'other-grid'},source_product='GeoMAD',source_version='0.4.0')
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
        records = [{'item': first}, {'item': first}, {'item': second}]
        expected_links = [f'{stac_root}/{first}', f'{stac_root}/{second}']
        result=dict(task=task,completion=completion,native=native,
                    index={'source_items':sources,'records':records,
                           'support':{'path':'support.json','sha256':sha256(support)}},
                    native_index_sha256='pinned-test-index',loader_receipt=receipt,worker_seconds=0)
        record=NativeCogStacWriter(self.root/'product',self.root).write(result,task_id(task))
        item_path=self.root/record['item']
        item=pystac.Item.from_file(item_path)
        LocalValidator().item_validator().validate(item.to_dict())
        collection=pystac.Collection.from_file(self.root/record['collection'])
        LocalValidator().collection_validator().validate(collection.to_dict())
        self.assertIsNone(item.properties['datetime'])
        self.assertEqual(item.properties['start_datetime'],'2024-01-01T00:00:00Z')
        self.assertEqual(item.properties['end_datetime'],'2024-12-31T23:59:59Z')
        self.assertEqual(len(item.assets),13)
        self.assertIn('country completeness requires separate assembly',item.properties['agc:scope'])
        self.assertEqual(item.assets['physical-area-ha'].extra_fields['raster:bands'][0]['unit'],'ha')
        for asset in item.assets.values():
            self.assertTrue(Path(asset.get_absolute_href()).is_file())
        self.assertCountEqual([link.get_href() for link in item.get_links('derived_from')], expected_links)
        moved=self.root/'relocated-product'
        (self.root/'product').rename(moved)
        relocated=pystac.Collection.from_file(moved/'collection.json')
        child=next(relocated.get_items())
        self.assertCountEqual([link.get_href() for link in child.get_links('derived_from')], expected_links)
        for asset in child.assets.values():
            self.assertTrue(Path(asset.get_absolute_href()).is_file())

    def test_status_categories_survive_cog(self):
        source=self.root/'status.tif';self.make_raster(source,'uint8')
        with rasterio.open(source,'r+') as ds:
            data=ds.read(1);data[0,:6]=[0,1,4,5,6,7];ds.write(data,1)
        target=self.root/'status-cog.tif';lossless_cog(source,target)
        with rasterio.open(target) as ds:
            self.assertEqual(ds.read(1)[0,:6].tolist(),[0,1,4,5,6,7])
            self.assertTrue((ds.read_masks(1)[0,:6]==255).all())


if __name__=='__main__':unittest.main()
