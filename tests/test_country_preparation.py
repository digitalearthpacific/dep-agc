"""CPU country input/cache and saved-accounting failures; no provider/model calls."""
from copy import deepcopy
import ast
import hashlib
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from affine import Affine
import numpy as np

from spc_agb_inputs.physical_area import ground_area_ha
from spc_agb_inputs.prepare_country import make_task, signature
from spc_agb_inputs.prepare_country import verified_preparation
from spc_agb_inputs.metadata import validate_item
from spc_agb_inputs.raster import static_owner
from spc_agb_dep.no_model import eligible_land
from spc_agb_dep.no_model import no_model_completion
from spc_agb_dep.assemble_country import compare_summary, independent_native
from spc_agb_dep.assemble_country import assemble
from spc_agb_worker.io import sha256, write_json, write_raster
from shapely.geometry import box
import test_processor_execution as execution_fixtures


class CountryPreparationTests(unittest.TestCase):
    inputs=execution_fixtures.ExecutionTests.inputs
    staged=execution_fixtures.ExecutionTests.staged
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir=os.environ.get('SPC_TEST_TMP'))
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_contract_signature_cannot_silently_reuse_changed_config(self):
        a = dict(grid='one', year=2024, source='0.4.0')
        self.assertEqual(signature(a), signature(dict(reversed(list(a.items())))))
        self.assertNotEqual(signature(a), signature(dict(a, grid='another')))

    def test_task_follows_pinned_country_grid_window(self):
        template = json.loads((Path(__file__).parents[1]/'configs/scientific-profile.json').read_text())
        plan = dict(country=dict(code='XYZ',name='Synthetic'),grid=dict(id='synthetic',crs='EPSG:32702',affine=[10,0,400000,0,-10,8500000],shape=[2496,2496]))
        plan['grid']['id']='synthetic-grid'
        window = dict(id='different-label',global_input_window=[0,0,2496,2496],bounds=[402240,8477280,422720,8497760])
        package=self.root/'package';package.mkdir();vrt=package/'vrt';vrt.write_bytes(b'input-only')
        task=make_task(template,plan,window,[],vrt,package,[],dict(path='inputs/ids.json',sha256='pin'), 2024)
        self.assertEqual(task['country'],plan['country']);self.assertEqual(task['grid']['id'],'synthetic-grid')
        self.assertEqual(task['window'],'different-label');self.assertEqual(task['inference'],template['inference'])
        bad=deepcopy(window);bad['global_input_window'][0]+=1
        with self.assertRaises(ValueError):make_task(template,plan,bad,[],vrt,package,[],{},2024)

    def test_no_model_only_when_no_eligible_known_land(self):
        arrays={k:np.ones((2,2),bool) for k in ['rgb_valid','red_valid','nir_valid','ndvi_valid']}
        arrays['ndvi']=np.zeros((2,2),np.float32)  # Low NDVI alone must not skip accepted worker.
        task=dict(grid=dict(context=0,retained_shape=[2,2]))
        dom=dict(land=np.ones((2,2),bool))
        self.assertTrue(eligible_land(arrays,task,dom).all())
        arrays['nir_valid'][:]=False
        self.assertFalse(eligible_land(arrays,task,dom).any())
        arrays['nir_valid'][:]=True;dom['land'][:]=False
        self.assertFalse(eligible_land(arrays,task,dom).any())

    def fixture(self):
        # Small synthetic northern grid; full real receiving checks use saved inputs.
        grid=Affine(10,0,400000,0,-10,8450000)
        cover=np.array([[10,10],[80,0]],np.uint8);weights=np.ones((2,2),np.float64)
        package=self.root/'static';target=package/'inputs/owner.npz';target.parent.mkdir(parents=True)
        np.savez_compressed(target,projected=cover,physical_area_ha=weights,support=weights>0)
        task=dict(grid=dict(crs='EPSG:32602',affine=list(grid)[:6],receiving_window=[0,0,2,2],context=0,retained_shape=[2,2]),
            inputs=dict(owners=[dict(id='owner',numeric_id=1,bounds=[400000,8449980,400020,8450000],file=dict(path='inputs/owner.npz'))]))
        status=np.array([[4,7],[0,5]],np.uint8);height=np.array([[2,3],[0,0]],np.float32)
        carbon=np.full((2,2),-9999,np.float32);carbon[0,0]=np.float32(3.744*float(height[0,0])**1.391);carbon[0,1]=0
        native=self.root/'native';native.mkdir()
        for name,value,mask,nodata in [('carbon',carbon,np.isin(status,[4,7]),-9999),('height',height,cover==10,-9999),
                ('status',status,weights>0,None),('owner',np.ones((2,2),np.uint16),weights>0,None),
                ('physical-area-ha',weights,weights>0,None),('WorldCover',cover,weights>0,None)]:
            write_raster(native/(name+'.tif'),value,mask,grid,'EPSG:32602',nodata)
        return native,task,package

    def test_native_reduction_keeps_water_unknown_and_valid_zero(self):
        native,task,package=self.fixture();result=independent_native(native,task,package)
        self.assertEqual(result['summary']['owned_ha'],4)
        self.assertEqual(result['summary']['assessed_ha'],2)
        self.assertEqual(result['summary']['water_ha'],1)
        self.assertEqual(result['summary']['unknown_ha'],1)
        self.assertEqual(result['summary']['assumed_zero_ha'],1)

    def test_native_formula_failure_is_not_suppressed(self):
        native,task,package=self.fixture()
        import rasterio
        with rasterio.open(native/'carbon.tif','r+') as ds:
            a=ds.read(1);a[0,0]+=2;ds.write(a,1)
        with self.assertRaisesRegex(ValueError,'formula'):independent_native(native,task,package)

    def test_summary_frozen_tolerances_and_nonfinite(self):
        compare_summary(dict(stock_tC=2.5,owned_ha=1),dict(stock_tC=2,owned_ha=1))
        for fresh in [dict(stock_tC=3.1,owned_ha=1),dict(stock_tC=2,owned_ha=1.001),dict(stock_tC=float('nan'),owned_ha=1)]:
            with self.assertRaises(ValueError):compare_summary(fresh,dict(stock_tC=2,owned_ha=1))

    def test_physical_area_remains_ellipsoid_and_additive(self):
        whole=ground_area_ha(box(400000,8449900,400100,8450000),'EPSG:32702')
        parts=sum(ground_area_ha(box(x,8449900,x+50,8450000),'EPSG:32702') for x in [400000,400050])
        self.assertGreater(whole,0);self.assertLess(abs(whole-parts),1e-7)

    def test_missing_static_is_unknown_not_water_or_assumed_zero(self):
        polygon=box(400000,8474990,400010,8475000)
        # No source object supplied: this is a deliberately unknown synthetic fixture.
        arrays, references=static_owner(polygon,[400000,8470000,405000,8475000],
            'EPSG:32702',[],{},self.root/'static-native')
        self.assertTrue(arrays['support'].any())
        self.assertTrue((arrays['projected'][arrays['support']]==0).all())
        self.assertTrue(all(r['status']=='unknown_missing_static_object' for r in references))

    def test_source_item_encoding_version_and_requested_calendar(self):
        item=dict(id='dep_s2_geomad_075_025_2024',collection='dep_s2_geomad',
            properties={'start_datetime':'2024-01-01T00:00:00Z','end_datetime':'2024-12-31T23:59:59Z',
            'proj:epsg':3832,'proj:shape':[9600,9600],'proj:transform':[10,0,0,0,-10,0]},assets={})
        for band in ['red','green','blue','nir','count']:
            item['assets'][band]={'href':'https://dep-public-data.s3.us-west-2.amazonaws.com/dep_s2_geomad/0-4-0/2024/'+band+'.tif',
                'raster:bands':[dict(scale=1 if band=='count' else .0001,offset=0,nodata=9999 if band=='count' else 0,data_type='uint16')]}
        validate_item(item, 2024, '0.4.0')
        for changed in [dict(item,collection='wrong'),dict(item,properties=dict(item['properties'],start_datetime='2025-01-01'))]:
            with self.assertRaises(ValueError):validate_item(changed, 2024, '0.4.0')

    def test_no_model_completion_preserves_exclusions_and_missing_height(self):
        _,task,package=self.fixture()
        template=json.loads((Path(__file__).parents[1]/'configs/scientific-profile.json').read_text())
        task.update(schema='spc-agb-window-v3',year=2024,country=dict(code='XYZ',name='Synthetic'),window='fixture',
            inference=template['inference'],policy=template['policy']);task['grid']['id']='fixture'
        write_json(package/'task.json',task)
        rgb=self.root/'rgb.tif';rgb.write_bytes(b'unit fixture RGB')
        arrays=dict(rgb=np.ones((3,2,2),np.uint8),rgb_valid=np.ones((2,2),bool),
            red_valid=np.ones((2,2),bool),nir_valid=np.zeros((2,2),bool),
            ndvi_valid=np.zeros((2,2),bool),ndvi=np.full((2,2),-9999,np.float32),
            count=np.ones((2,2),np.uint16),count_valid=np.ones((2,2),bool))
        output=self.root/'worker'
        (self.root/'worker-eligibility').mkdir()  # Real prepare() creates this directory.
        with patch('spc_agb_dep.no_model.prepare',return_value=(arrays,Affine(*task['grid']['affine']),rgb)),patch('spc_agb_dep.no_model.validate',return_value={}):
            result=no_model_completion(dict(task=task,package=package),self.root,self.root,output)
        self.assertEqual(result['completion_kind'],'no_model');self.assertEqual(len(result['files']),9)
        self.assertEqual(result['summary']['stock_tC'],0);self.assertEqual(result['summary']['assessed_ha'],0)
        self.assertEqual(result['summary']['unavailable_ha'],2)
        self.assertEqual(result['summary']['water_ha'],1);self.assertEqual(result['summary']['unknown_ha'],1)
        self.assertFalse(result['inference_metrics']['model_loaded'])

    def test_incomplete_country_cannot_advertise_complete_total(self):
        plan=self.root/'plan.json';write_json(plan,dict(country=dict(code='XYZ',name='Synthetic')))
        selection=dict(windows=[dict(id='a',bounds=[0,0,10,10]),dict(id='b',bounds=[10,0,20,10])],
            jobs={},entry=dict(country=dict(code='XYZ',name='Synthetic')),plan_file=plan,plan={})
        with patch('spc_agb_dep.assemble_country.country_plan',return_value=selection):
            result=assemble(self.root/'catalog.json','XYZ',2024,self.root/'results',self.root/'country')
        self.assertFalse(result['complete']);self.assertIsNone(result['summary'])
        self.assertEqual([r['status'] for r in result['windows']],['unprepared','unprepared'])

    def test_country_duplicate_retained_ownership_is_rejected(self):
        plan=self.root/'plan.json';write_json(plan,{})
        selection=dict(windows=[dict(id='a',bounds=[0,0,10,10]),dict(id='b',bounds=[5,0,15,10])],
            jobs={},entry=dict(country=dict(code='XYZ',name='Synthetic')),plan_file=plan,plan={})
        with patch('spc_agb_dep.assemble_country.country_plan',return_value=selection),self.assertRaisesRegex(ValueError,'Duplicate'):
            assemble(self.root/'catalog.json','XYZ',2024,self.root/'results',self.root/'country')

    def test_physical_functions_match_accepted_operation_order(self):
        # Test-time AST fingerprints only; runtime uses ordinary imports/functions.
        # Origins: forest_carbon_accounting and inputs.weights at687ee46.
        expected={
            'coordinate_transformer':'424e90f8650644e0e77a1a6f710a16acd78b1e4843cbaaf65453dae6dd6a2f48',
            'ground_area_ha':'8979ffff938768da4f31f8d604aab0e952dd89e62eb00d89108296e1f4510344',
            'intersecting_cells':'f41b604e793b669aa9914e70fb19f78bd44f4fb3a597c1407ddf5707719281f4',
            'area_weights':'a3c977b37740e25118ab16dcffc0018b480d86c663b7fd1aecb26745cd53bc22'}
        found={}
        for name in ['physical_area.py','weights.py']:
            for f in ast.parse((Path(__file__).parents[1]/'spc_agb_inputs'/name).read_text()).body:
                if isinstance(f,ast.FunctionDef) and f.name in expected:
                    if isinstance(f.body[0],ast.Expr) and isinstance(f.body[0].value,ast.Constant) and isinstance(f.body[0].value.value,str):
                        f.body=f.body[1:]
                    # Python3.12 adds an empty type_params field to FunctionDef;
                    # the frozen fingerprints were recorded with Python3.11.
                    self.assertFalse(getattr(f,'type_params',[]))
                    value=ast.dump(f,include_attributes=False).replace(', type_params=[]','')
                    found[f.name]=hashlib.sha256(value.encode()).hexdigest()
        self.assertEqual(found,expected)

    def test_country_stac_metadata_and_lineage_survive_relocation(self):
        # Metadata-only synthetic fixture: scientific checks have separate tests.
        import pystac
        import shutil
        from datetime import datetime,timezone
        results=self.root/'results';integration=results/'a';product=integration/'product';product.mkdir(parents=True)
        (integration/'worker').mkdir();payload=product/'carbon.tif';payload.write_bytes(b'synthetic asset')
        source=product/'source.json';source.write_text('{}')
        item=pystac.Item('synthetic-window',{'type':'Polygon','coordinates':[[[0,0],[1,0],[1,1],[0,1],[0,0]]]},[0,0,1,1],None,
            properties=dict(start_datetime='2024-01-01T00:00:00Z',end_datetime='2024-12-31T23:59:59Z'))
        item.add_asset('carbon',pystac.Asset(str(payload),media_type='image/tiff'))
        item.add_link(pystac.Link('derived_from','source.json'))
        collection=pystac.Collection('window','Synthetic metadata fixture',pystac.Extent(pystac.SpatialExtent([[0,0,1,1]]),
            pystac.TemporalExtent([[datetime(2024,1,1,tzinfo=timezone.utc),datetime(2024,12,31,tzinfo=timezone.utc)]])))
        collection.set_self_href(str(product/'collection.json'));collection.add_item(item);item.set_self_href(str(product/'item.json'))
        item.make_asset_hrefs_relative();collection.save(catalog_type=pystac.CatalogType.SELF_CONTAINED)
        plan=self.root/'plan.json';write_json(plan,{})
        task=self.root/'task.json';write_json(task,dict(inputs=dict(blocks=[])))
        summary=dict(owned_ha=1.,assessed_ha=1.,stock_tC=2.)
        write_json(integration/'worker/completion.json',dict(summary=summary,owners={'one':summary},files={'carbon':{}}))
        write_json(integration/'integration-completion.json',{})
        selected=dict(windows=[dict(id='a',bounds=[0,0,10,10])],jobs={'a':{}},entry=dict(country=dict(code='XYZ',name='Synthetic')),plan_file=plan,plan={})
        with patch('spc_agb_dep.assemble_country.country_plan',return_value=selected),\
             patch('spc_agb_dep.assemble_country.resolve_window',return_value=dict(job=plan,binding=dict(task=task,package=self.root))),\
             patch('spc_agb_dep.assemble_country.verify_completion',return_value=dict(item='product/item.json')),\
             patch('spc_agb_dep.assemble_country.independent_native',return_value=dict(summary=summary,owners={'one':summary})),\
             patch('spc_agb_dep.assemble_country.supplied_domain',return_value={}),\
             patch('spc_agb_dep.assemble_country.generated_inputs',return_value={}),\
             patch('spc_agb_dep.assemble_country.cog_native',return_value=dict(metadata_fixture=True)):
            assembled=assemble(self.root/'catalog.json','XYZ',2024,results,self.root/'country')
        self.assertTrue(assembled['complete'])
        moved=self.root/'relocated';moved.mkdir();shutil.copytree(results,moved/'results');shutil.copytree(self.root/'country',moved/'country')
        relocated=pystac.Collection.from_file(str(moved/'country/collection.json'))
        for item in relocated.get_items():
            self.assertTrue(Path(item.assets['carbon'].get_absolute_href()).is_relative_to(moved))
            self.assertTrue(Path(item.assets['carbon'].get_absolute_href()).is_file())
            self.assertTrue(all((Path(item.get_self_href()).parent/link.get_href()).resolve().is_file() for link in item.get_links('derived_from')))

    def test_completed_preparation_verification_needs_no_original_cache_or_boundary(self):
        root=self.staged();plan=json.loads((root/'plan.json').read_text());key=plan['grid']['windows'][0]['id']
        config=self.root/'config.json';write_json(config,dict(schema='agc-country-preparation-v1',boundary=dict(path='unmounted-original')))
        state=dict(status='prepared',started_epoch=123,provider_payload_bytes=456,
            contract=dict(config_sha256=sha256(config),plan_sha256=sha256(root/'plan.json'),country=plan['country'],year=2024,code_sha256={'historical':'preserved'}),
            completed={key:dict(job=dict(path='job.json',sha256=sha256(root/'job.json')),canonical_files=[])})
        write_json(root/'preparation-state.json',state)
        write_json(root/'countries.json',dict(schema='agc-prepared-countries-v2',artifact_root='.',countries=[dict(country=plan['country'],year=2024,source_version='0.4.0',
            plan=dict(path='plan.json',sha256=sha256(root/'plan.json')),jobs=[dict(window=key,job=dict(path='job.json',sha256=sha256(root/'job.json')))])]))
        selected=dict(year=2024,plan=plan,plan_file=root/'plan.json',windows=plan['grid']['windows'])
        before=sha256(root/'preparation-state.json');result=verified_preparation(root,selected,config)
        self.assertTrue(result['read_only_verification']['passed'])
        self.assertEqual(result['started_epoch'],123);self.assertEqual(result['provider_payload_bytes'],456)
        self.assertEqual(sha256(root/'preparation-state.json'),before)
        config.write_text(config.read_text()+'\n')
        with self.assertRaisesRegex(ValueError,'differs'):verified_preparation(root,selected,config)

    def test_missing_required_native_asset_is_explicit_and_never_zero_or_retried(self):
        from spc_agb_inputs.transport import RangeArchive,SourceAccessFailure
        class Response:
            status_code=404
            headers={}
            def __enter__(self):return self
            def __exit__(self,*args):pass
        class Session:
            headers={}
            calls=0
            def request(self,*args,**kwargs):self.calls+=1;return Response()
        session=Session();archive=RangeArchive(self.root/'incoming',self.root/'transfer.json',1024,session=session)
        for _ in range(2):
            with self.assertRaises(SourceAccessFailure):archive.fetch('https://provider.example/absent.tif','HEAD')
        self.assertEqual(session.calls,1)
        self.assertEqual(archive.state['failure_kind'],'source_access')
        self.assertTrue(list((self.root/'incoming').glob('*.partial.body')))
