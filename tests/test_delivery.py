"""Annual CLI delivery with real raster writers and local stand-ins for providers."""

from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from hashlib import sha256
from io import BytesIO
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
import unittest
from unittest.mock import Mock, patch
from zipfile import ZipFile

from botocore.exceptions import ClientError
from dep_tools.grids import PACIFIC_GRID_10
from dep_tools.namers import S3ItemPath
from filelock import FileLock
import numpy as np
from odc.geo.xr import wrap_xr
import pystac
from pyproj import Transformer
import rasterio
from rasterio.enums import Resampling
from requests.exceptions import HTTPError
from rio_cogeo.cogeo import cog_validate
from shapely.geometry import box, shape
from shapely.ops import transform, unary_union
from typer import Typer
from typer.testing import CliRunner
import xarray as xr

from src.run_task import main, prepare_model
from src.utils import GEOMAD_ITEMS_URL, MODEL_BUFFER, STATUS_NODATA, parse_years


TILE = '075_025'
BUCKET = 'agc-fixture-output'
PREFIX = f's3://{BUCKET}/'
COLLECTION_ROOT = 'https://catalog.example.org/v1/collections'
MODEL_ZIP_NAME = 'dep-agc-model-24123113h21.zip'
MODEL_ZIP_URI = f'https://example.org/models/{MODEL_ZIP_NAME}'
SIGNED_MODEL_URIS = (
    f'{MODEL_ZIP_URI}?X-Amz-Credential=first&X-Amz-Security-Token=token-1&X-Amz-Signature=signature-1#download',
    f'{MODEL_ZIP_URI}?AWSAccessKeyId=second&Expires=60&Signature=signature-2',
)
MODEL_DATA = {'Delta1_model_traced.pth': b'model weights', 'training_config.cfg': b'training config'}
VERSION = '0.1.0-review.5'
COMMAND = Typer(add_completion=False)
COMMAND.command()(main)


def keys(tile_id, year):
    tile_index = tuple(map(int, tile_id.split('_')))
    namer = S3ItemPath(bucket=BUCKET, sensor='s2', dataset_id='aboveground-carbon',
                       version=VERSION, time=str(year), prefix='local-agc',
                       make_hrefs_https=False)
    return {name: namer.path(tile_index, name) for name in ('carbon', 'status')} | {'item': namer.stac_path(tile_index)}


def model_zip_response(entries):
    archive = BytesIO()
    with ZipFile(archive, 'w') as files:
        for name, data in entries.items():
            files.writestr(f'release/{name}', data)
    payload = archive.getvalue()
    return Mock(iter_content=Mock(return_value=[payload[:64], payload[64:]]))


def geomad_results(*, bbox, collections, datetime):
    year = datetime[:4]
    query = box(*bbox)
    items = []
    # Reverse the search order and use a non-canonical self link to exercise STAC lineage.
    for col in (76, 75, 74, 67, 66, 65):
        for row in (26, 25, 24):
            tile = PACIFIC_GRID_10[col, row]
            footprint = tile.extent.to_crs('EPSG:4326', wrapdateline=True).geom
            if not query.intersects(footprint):
                continue
            item_id = f'dep_s2_geomad_{col:03d}_{row:03d}_{year}'
            item = pystac.Item(
                item_id, None, None, None, collection=collections[0],
                properties={'start_datetime': f'{year}-01-01T00:00:00Z', 'end_datetime': f'{year}-12-31T23:59:59Z',
                            'proj:bbox': list(tile.boundingbox.bbox)},
            )
            item.set_self_href(f'https://example.org/noncanonical/{item_id}')
            items.append(item)
    return Mock(items=Mock(return_value=iter(items)), item_collection=Mock(return_value=pystac.ItemCollection(items)))


def fixture_values():
    values = np.arange(64*64).reshape(64, 64)
    valid = np.ones((64, 64), bool)
    valid[1, 0] = False
    carbon = np.where(valid, values, np.nan).astype(np.float32)
    status = np.array([0, 1, 4, 5, 7], np.uint8)[values % 5]
    status[~valid] = STATUS_NODATA
    return {'carbon': carbon, 'status': status}, valid


class DeliveryTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(TemporaryDirectory()))
        self.store = self.root/'store'
        self.writes = []
        self.failed_asset = None
        self.failed_year = None
        self.model_cache = self.root/'models'
        self.model_entry = self.model_cache/MODEL_ZIP_NAME/sha256(MODEL_ZIP_URI.encode()).hexdigest()
        self.enterContext(patch('src.run_task.MODEL_CACHE', self.model_cache))
        self.client = Mock()
        self.client.exceptions.ClientError = ClientError
        self.client.head_object.side_effect = self.head
        self.client.put_object.side_effect = self.put
        self.session = self.enterContext(patch('src.run_task.boto3.Session'))
        self.session.return_value.client.return_value = self.client
        self.download = self.enterContext(patch('src.run_task.requests.get'))
        response = model_zip_response(MODEL_DATA | {'README.txt': b'unused'})
        self.model_sha256 = sha256(b''.join(response.iter_content.return_value)).hexdigest()
        self.download.return_value = nullcontext(response)
        self.catalogue = self.enterContext(patch('src.run_task.Client.open'))
        self.catalogue.return_value.id = 'dep'
        self.catalogue.return_value.search.side_effect = geomad_results
        self.cover = self.enterContext(patch('src.run_task.prepare_worldcover'))
        self.process = self.enterContext(patch('src.run_task.process_tile', side_effect=self.tile_dataset))
        self.enterContext(patch('src.run_task.logging.basicConfig'))
        self.enterContext(patch('src.run_task.configure_s3_access'))
        self.enterContext(patch('src.run_task.MODEL_BUFFER', 2))
        # Small rasters retain the official tile footprint, including at the antimeridian.
        grid = Mock(tile_geobox=lambda index: PACIFIC_GRID_10[index].zoom_out(150))
        self.enterContext(patch('src.run_task.PACIFIC_GRID_10', grid))
        self.enterContext(patch('src.task.check_geomad_sources'))
        self.load = self.enterContext(patch('src.run_task.OdcLoader.load',
                                           side_effect=lambda items, area: int(items[0].id[-4:])))
        self.real_open = rasterio.open
        self.enterContext(patch('rasterio.open', side_effect=self.open_raster))

    def head(self, *, Bucket, Key):
        self.assertEqual(Bucket, BUCKET)
        if not (self.store/Key).is_file():
            raise ClientError({'Error': {'Code': '404'}, 'ResponseMetadata': {'HTTPStatusCode': 404}}, 'HeadObject')
        return {'ResponseMetadata': {'HTTPStatusCode': 200}}

    def put(self, *, Bucket, Key, Body, IfNoneMatch=None, **kwargs):
        self.assertEqual(Bucket, BUCKET)
        self.writes.append((Key, IfNoneMatch))
        path = self.store/Key
        if Key == self.failed_asset:
            raise ClientError({'Error': {'Code': 'InternalError'}}, 'PutObject')
        if IfNoneMatch == '*' and path.exists():
            raise ClientError({'Error': {'Code': 'PreconditionFailed'}}, 'PutObject')
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(Body.encode() if isinstance(Body, str) else Body)
        return {'ResponseMetadata': {'HTTPStatusCode': 200}}

    def open_raster(self, path, *args, **kwargs):
        if isinstance(path, str) and path.startswith(PREFIX):
            path = self.store/path.removeprefix(PREFIX)
        return self.real_open(path, *args, **kwargs)

    def tile_dataset(self, year, cover, tile, buffered, *, model_dir):
        self.assertEqual({name: (model_dir/name).read_bytes() for name in MODEL_DATA}, MODEL_DATA)
        if year == self.failed_year:
            raise RuntimeError('Fixture inference failure')
        values, _ = fixture_values()
        return xr.Dataset({
            'carbon': wrap_xr(values['carbon'], tile, nodata=np.nan),
            'status': wrap_xr(values['status'], tile, nodata=STATUS_NODATA),
        })

    def invoke(self, year='2024', tile_id=TILE, *, overwrite=False, **options):
        args = {'tile-id': tile_id, 'year': year, 'model-zip-uri': MODEL_ZIP_URI, 'output-bucket': BUCKET,
                'version': VERSION, 'collection-url-root': COLLECTION_ROOT + '/'} | options
        return CliRunner().invoke(COMMAND, [
            word for flag, value in args.items() if value is not None for word in (f'--{flag}', value)
        ] + (['--overwrite'] if overwrite else []))

    def seed(self, key, value):
        path = self.store/key
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(value)

    def expected_writes(self, tile_id, year, item_condition='*'):
        paths = keys(tile_id, year)
        return [(paths['carbon'], None), (paths['status'], None), (paths['item'], item_condition)]

    def test_annual_cogs_and_stac_item(self):
        tag, commit = 'dep-agc:provenance-test', 'a' * 40
        repo = f'https://github.com/digitalearthpacific/dep-agc/tree/{commit}/'
        with (patch.dict(os.environ, {'IMAGE_TAG': tag, 'COMMIT': commit, 'IMAGE_REPO': repo}),
              patch('src.run_task.ZipFile', wraps=ZipFile) as archive):
            for tile_id, year_arg, years, model_uri in (
                (TILE, '2024-2025', (2024, 2025), SIGNED_MODEL_URIS[0]),
                ('066_025', '2024', (2024,), SIGNED_MODEL_URIS[1]),
            ):
                with self.subTest(tile=tile_id):
                    start = len(self.writes)
                    before = self.cover.call_count
                    first_process = self.process.call_count
                    result = self.invoke(year_arg, tile_id, **{'model-zip-uri': model_uri})
                    self.assertEqual(result.exit_code, 0, result.output)
                    self.assertEqual(self.cover.call_count, before + 1)
                    self.assertEqual([call.kwargs['model_dir'] for call in self.process.call_args_list[first_process:]],
                                     [self.model_entry] * len(years))
                    self.assertEqual(result.stdout.splitlines(), [PREFIX + keys(tile_id, year)['item'] for year in years])
                    self.assertEqual(self.writes[start:], [
                        upload for year in years for upload in self.expected_writes(tile_id, year)
                    ])
                    for year in years:
                        with self.subTest(year=year):
                            item = self.assert_annual_products(tile_id, year)
                            self.assertEqual(item.properties['processing:software'], {'docker-image': tag, 'commit': commit})
                            self.assertEqual([link.get_href() for link in item.get_links('via')], [repo])
        self.download.assert_called_once_with(SIGNED_MODEL_URIS[0], stream=True, timeout=(10, 60))
        archive.assert_called_once()
        self.assertEqual(sha256((self.model_entry/MODEL_ZIP_NAME).read_bytes()).hexdigest(), self.model_sha256)

    def test_competing_model_setup_waits_for_the_published_entry(self):
        downloading, competing = Event(), Event()

        def download(*args, **kwargs):
            self.assertFalse(self.model_entry.exists())
            downloading.set()
            self.assertTrue(competing.wait(10))
            return self.download.return_value

        def lock(path):
            if downloading.is_set():
                competing.set()
            return FileLock(path)

        self.download.side_effect = download
        with patch('src.run_task.FileLock', side_effect=lock), ThreadPoolExecutor(max_workers=2) as jobs:
            first = jobs.submit(prepare_model, MODEL_ZIP_URI, MODEL_ZIP_URI, MODEL_ZIP_NAME)
            self.assertTrue(downloading.wait(10))
            second = jobs.submit(prepare_model, MODEL_ZIP_URI, MODEL_ZIP_URI, MODEL_ZIP_NAME)
            self.assertEqual(first.result(timeout=10), (self.model_entry, self.model_sha256))
            self.assertEqual(second.result(timeout=10), (self.model_entry, self.model_sha256))
        self.download.assert_called_once()

    def test_model_zip_filenames_keep_separate_entries_and_replace_incomplete_entry(self):
        self.model_entry.mkdir(parents=True)
        (self.model_entry/MODEL_ZIP_NAME).write_bytes(b'incomplete ZIP')
        self.assertEqual(prepare_model(MODEL_ZIP_URI, MODEL_ZIP_URI, MODEL_ZIP_NAME),
                         (self.model_entry, self.model_sha256))

        other_name = 'dep-agc-model-2.zip'
        other_response = model_zip_response(MODEL_DATA | {'README.txt': b'different ZIP'})
        other_digest = sha256(b''.join(other_response.iter_content.return_value)).hexdigest()
        self.download.return_value = nullcontext(other_response)
        other_uri = f'https://example.org/models/{other_name}'
        other_entry = self.model_cache/other_name/sha256(other_uri.encode()).hexdigest()
        self.assertEqual(prepare_model(other_uri, other_uri, other_name), (other_entry, other_digest))
        self.assertEqual(sha256((self.model_entry/MODEL_ZIP_NAME).read_bytes()).hexdigest(), self.model_sha256)
        self.assertEqual(self.download.call_count, 2)

    def test_model_urls_sharing_a_filename_keep_separate_stac_provenance(self):
        result = self.invoke('2024')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assert_annual_products(TILE, 2024)
        for year, uri in (
            (2025, f'https://other.example.org/releases/{MODEL_ZIP_NAME}'),
            (2026, f'{MODEL_ZIP_URI}?versionId=other-object-version'),
        ):
            with self.subTest(uri=uri):
                response = model_zip_response(MODEL_DATA | {'README.txt': uri.encode()})
                digest = sha256(b''.join(response.iter_content.return_value)).hexdigest()
                self.assertNotEqual(digest, self.model_sha256)
                self.download.return_value = nullcontext(response)
                separator = '&' if '?' in uri else '?'
                result = self.invoke(str(year), **{'model-zip-uri': f'{uri}{separator}X-Amz-Signature=temporary'})
                self.assertEqual(result.exit_code, 0, result.output)
                self.assert_annual_products(TILE, year, model_uri=uri, model_sha256=digest)

        result = self.invoke('2027')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assert_annual_products(TILE, 2027)
        self.assertEqual(self.download.call_count, 3)

    def test_geomad_discovery_queries_the_native_buffer_and_year(self):
        search = self.catalogue.return_value.search
        search.side_effect = None
        search.return_value = Mock(items=Mock(return_value=[]),
                                   item_collection=Mock(return_value=pystac.ItemCollection([])))
        with patch('src.run_task.PACIFIC_GRID_10', PACIFIC_GRID_10), patch('src.run_task.MODEL_BUFFER', MODEL_BUFFER):
            for tile_id, bounds in (
                (TILE, [[-172.29088, -14.33618, -171.38825, -13.45441]]),
                ('066_025', [[179.94768, -14.33618, 180, -13.45441],
                             [-180, -14.33618, -179.14970, -13.45441]]),
            ):
                with self.subTest(tile=tile_id):
                    search.reset_mock()
                    result = self.invoke('2024', tile_id)
                    self.assertEqual(result.exit_code, 1, result.output)
                    self.assertIn('No annual GeoMAD Items', result.stderr)
                    self.assertEqual(search.call_count, len(bounds))
                    for call, expected in zip(search.call_args_list, bounds, strict=True):
                        query = call.kwargs
                        self.assertEqual(set(query), {'bbox', 'collections', 'datetime'})
                        self.assertEqual(query['collections'], ['dep_s2_geomad'])
                        self.assertEqual(query['datetime'], '2024-01-01T00:00:00Z/2024-12-31T23:59:59Z')
                        np.testing.assert_allclose(query['bbox'], expected, atol=1e-5, rtol=0)
        self.load.assert_not_called()

    def assert_annual_products(self, tile_id, year, *, model_uri=MODEL_ZIP_URI, model_sha256=None):
        paths = keys(tile_id, year)
        path = self.store/paths['item']
        json.dumps(json.loads(path.read_text(encoding='utf-8')), allow_nan=False)
        item = pystac.Item.from_file(path)
        self.assertEqual(item.id, f'local-agc_s2_aboveground-carbon_{tile_id}_{year}')
        self.assertEqual(item.get_single_link('collection').get_href(),
                         f'{COLLECTION_ROOT}/local-agc_s2_aboveground-carbon')
        self.assertEqual(path.relative_to(self.store).parts[-4:-1], (*tile_id.split('_'), str(year)))
        self.assertEqual(item.properties['start_datetime'], f'{year}-01-01T00:00:00Z')
        self.assertEqual(item.properties['end_datetime'], f'{year}-12-31T23:59:59Z')
        self.assertIsNone(item.properties['datetime'])
        self.assertEqual(item.properties['agc:model_uri'], model_uri)
        self.assertEqual(item.properties['agc:model_sha256'], model_sha256 or self.model_sha256)
        self.assertEqual(set(item.assets), {'carbon', 'status'})
        self.assertEqual(set(path.parent.iterdir()), {self.store/key for key in paths.values()})
        self.assertEqual(item.assets['carbon'].extra_fields['raster:bands'][0]['unit'], 't C/ha')
        self.assertEqual(item.assets['carbon'].extra_fields['raster:bands'][0]['nodata'], 'nan')
        self.assertEqual(item.assets['status'].extra_fields['raster:bands'][0]['nodata'], STATUS_NODATA)

        tile = PACIFIC_GRID_10[tuple(map(int, tile_id.split('_')))]
        grid = tile.zoom_out(150)
        col, row = map(int, tile_id.split('_'))
        sources = [f'dep_s2_geomad_{x:03d}_{y:03d}_{year}'
                   for x in range(col-1, col+2) for y in range(row-1, row+2)]
        self.assertEqual([link.get_href() for link in item.get_links('derived_from')],
                         [f'{GEOMAD_ITEMS_URL}/{source}' for source in sources])
        self.assert_footprint(item, tile, tile_id)

        moved = self.root/f'relocated-{tile_id}-{year}'
        path.parent.rename(moved)
        relocated = pystac.Item.from_file(moved/path.name)
        expected, valid = fixture_values()
        for name, asset in relocated.assets.items():
            filename = Path(paths[name]).name
            self.assertEqual(Path(asset.href), Path(filename))
            self.assertEqual(asset.roles, ['data'])
            target = Path(asset.get_absolute_href())
            self.assertEqual(target, moved/filename)
            self.assertTrue(cog_validate(target, strict=True, quiet=True)[0])
            with rasterio.open(target) as cog:
                self.assertEqual((cog.crs, cog.transform, cog.shape, cog.dtypes),
                                 (rasterio.crs.CRS.from_epsg(3832), grid.transform, tuple(grid.shape),
                                  (expected[name].dtype.name,)))
                values = cog.read(1)
                np.testing.assert_array_equal(values, expected[name])
                np.testing.assert_array_equal(cog.read_masks(1) > 0, valid)
                self.assertTrue(np.any((values == 0) & valid))
                self.assertEqual(cog.overviews(1), [2, 4, 8])
                if name == 'carbon':
                    self.assertTrue(np.isnan(cog.nodata))
                    np.testing.assert_array_equal(np.isnan(values), ~valid)
                else:
                    self.assertEqual(cog.nodata, STATUS_NODATA)
                    np.testing.assert_array_equal(values == STATUS_NODATA, ~valid)
                    for level in cog.overviews(1):
                        options = dict(out_shape=(64//level, 64//level), resampling=Resampling.nearest)
                        overview = cog.read(1, **options)
                        covered = cog.read_masks(1, **options) > 0
                        self.assertLessEqual(set(overview[covered]), {0, 1, 4, 5, 7})
                        np.testing.assert_array_equal(overview == STATUS_NODATA, ~covered)
        return item

    def assert_footprint(self, item, tile, tile_id):
        footprint = shape(item.geometry)
        self.assertTrue(footprint.is_valid)
        if tile_id == '066_025':
            self.assertEqual(footprint.geom_type, 'MultiPolygon')
            self.assertEqual(len(footprint.geoms), 2)
            np.testing.assert_allclose(item.bbox, [179.967797878, -14.316558336, -179.169819449, -13.474104441],
                                       atol=1e-8, rtol=0)
            west, south, east, north = item.bbox
            envelope = box(west, south, 180, north).union(box(-180, south, east, north))
            self.assertTrue(envelope.covers(footprint))
            parts = footprint.geoms
        else:
            self.assertEqual(footprint.geom_type, 'Polygon')
            np.testing.assert_allclose(item.bbox, [-172.270758067, -14.316558336, -171.408375394, -13.474104441],
                                       atol=1e-8, rtol=0)
            parts = [footprint]
        project = Transformer.from_crs(4326, str(tile.crs), always_xy=True).transform
        projected = unary_union([transform(project, part).simplify(.001) for part in parts])
        self.assertLess(projected.symmetric_difference(tile.extent.geom).area, 1)

    def test_output_version_selects_s3_keys(self):
        stem = 'local-agc_s2_aboveground-carbon_075_025_2024'
        for version, folder in ((VERSION, '0-1-0-review-5'), ('0.1.0-review.6', '0-1-0-review-6')):
            with self.subTest(version=version):
                # Check the established keys independently of the namer.
                base = f'local-agc_s2_aboveground-carbon/{folder}/075/025/2024/{stem}'
                expected = [f'{base}_carbon.tif', f'{base}_status.tif', f'{base}.stac-item.json']
                start = len(self.writes)
                result = self.invoke(version=version)
                self.assertEqual(result.exit_code, 0, result.output)
                self.assertEqual([key for key, _ in self.writes[start:]], expected)
                self.assertEqual(result.stdout.splitlines(), [PREFIX + expected[-1]])
                self.assertEqual(self.client.head_object.call_args.kwargs['Key'], expected[-1])

    def test_resume_rewrites_partial_delivery_and_skips_completed_years(self):
        completed = keys(TILE, 2024)['item']
        partial = keys(TILE, 2025)
        self.seed(completed, b'previous Item')
        self.seed(partial['carbon'], b'incomplete COG')
        result = self.invoke('2024-2025')
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.stdout.splitlines(), [PREFIX + partial['item']])
        self.assertEqual([call.args[0] for call in self.process.call_args_list], [2025])
        self.assertEqual(self.writes, self.expected_writes(TILE, 2025))
        self.assertEqual((self.store/completed).read_bytes(), b'previous Item')
        self.assertTrue(cog_validate(self.store/partial['carbon'], strict=True, quiet=True)[0])

        cold_cache = self.root/'skipped-models'
        with patch('src.run_task.MODEL_CACHE', cold_cache):
            skipped = self.invoke('2024-2025')
        self.assertEqual(skipped.exit_code, 0, skipped.output)
        self.assertEqual(skipped.stdout, '')
        self.assertFalse(cold_cache.exists())
        self.download.assert_called_once()
        self.process.assert_called_once()

    def test_unexpected_head_errors_stop_the_range_before_any_writes(self):
        later_item = keys(TILE, 2025)['item']
        for status, code in ((403, '403'), (500, 'InternalError'), (404, 'NoSuchBucket')):
            with self.subTest(status=status, code=code):
                self.client.head_object.side_effect = [
                    ClientError({'Error': {'Code': '404'}, 'ResponseMetadata': {'HTTPStatusCode': 404}}, 'HeadObject'),
                    ClientError({'Error': {'Code': code}, 'ResponseMetadata': {'HTTPStatusCode': status}}, 'HeadObject'),
                ]
                result = self.invoke('2024-2025')
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertIn(later_item, result.stderr)
                self.assertIn(code, result.stderr)
                self.assertEqual(result.stdout, '')
                self.assertFalse(self.model_cache.exists())
                self.download.assert_not_called()
                self.process.assert_not_called()
                self.client.put_object.assert_not_called()

    def test_overwrite_replaces_existing_item(self):
        item = keys(TILE, 2024)['item']
        self.seed(item, b'previous Item')
        result = self.invoke(overwrite=True)
        self.assertEqual(result.exit_code, 0, result.output)
        self.assertEqual(result.stdout.splitlines(), [PREFIX + item])
        self.assertEqual(self.writes, self.expected_writes(TILE, 2024, item_condition=None))
        self.client.head_object.assert_not_called()
        self.assertEqual(json.loads((self.store/item).read_text())['type'], 'Feature')

    def test_failures_preserve_earlier_delivery_and_stop_the_range(self):
        failing = keys(TILE, 2025)
        for stage, attempts in (('inference', ()), ('status', ('carbon', 'status')), ('item', ('carbon', 'status', 'item'))):
            with self.subTest(stage=stage):
                self.store = self.root/f'store-{stage}'
                self.writes.clear()
                self.failed_year = 2025 if stage == 'inference' else None
                self.failed_asset = failing.get(stage)
                result = self.invoke('2024-2026')
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertTrue(result.stderr)
                self.assertTrue((self.model_entry/MODEL_ZIP_NAME).is_file())
                first_item = keys(TILE, 2024)['item']
                self.assertEqual(result.stdout.splitlines(), [PREFIX + first_item])
                self.assertTrue((self.store/first_item).exists())
                self.assertFalse((self.store/failing['item']).exists())
                self.assertEqual([key for key, _ in self.writes],
                                 list(keys(TILE, 2024).values()) + [failing[name] for name in attempts])

    def test_rejects_wrong_tile_grid_or_nodata_before_upload(self):
        def invalid_dataset(*args, **kwargs):
            data = self.tile_dataset(*args, **kwargs)
            if error == 'grid':
                return data.assign_coords(x=data.x + 1500)
            data.status.attrs['nodata'] = 0
            return data

        self.process.side_effect = invalid_dataset
        for error, layer in (('grid', 'carbon'), ('nodata', 'status')):
            with self.subTest(error=error):
                result = self.invoke()
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertIn(f'Invalid tile grid or encoding for {layer}', result.stderr)
                self.assertEqual(self.writes, [])

    def test_bad_cli_options_fail_before_s3(self):
        self.assertEqual(parse_years('1-9999'), range(1, 10000))
        for options in (
            *({'year': value} for value in ('2024-', '2024-2025-2026', '2025-2024', '0', '10000', '0-2024', '2024-10000')),
            {'tile-id': '075-025'}, {'model-zip-uri': 'http://example.org/model.zip'},
            {'model-zip-uri': 'https:///model.zip'}, {'model-zip-uri': 'https://example.org/models/'},
            *({'output-bucket': value} for value in (
                None, 's3://example-output/', 'ab', 'Uppercase-bucket', 'bucket..name', '192.0.2.1', 'bucket.mrap',
            )),
            *({'collection-url-root': value} for value in (
                None, '/collections', 'http://catalog.example.org/collections', 'https:///collections',
                'https://catalog.example.org:invalid/collections', 'https://user@catalog.example.org/collections',
                f'{COLLECTION_ROOT}?query=1', f'{COLLECTION_ROOT}#fragment',
            )),
            {'version': None},
        ):
            with self.subTest(options=options):
                result = self.invoke(**options)
                self.assertEqual(result.exit_code, 2, result.output)
                self.session.assert_not_called()
                self.assertFalse(self.model_cache.exists())

    def test_bad_model_download_stops_before_processing(self):
        for label, response in (
            ('unavailable', Mock(raise_for_status=Mock(side_effect=HTTPError('503 model ZIP unavailable')))),
            ('not a zip', Mock(iter_content=Mock(return_value=[b'not a ZIP']))),
            ('missing config', model_zip_response({'Delta1_model_traced.pth': b'weights'})),
            ('missing weights', model_zip_response({'training_config.cfg': b'config'})),
        ):
            with self.subTest(label=label):
                self.download.return_value = nullcontext(response)
                result = self.invoke('2024-2025')
                self.assertEqual(result.exit_code, 1, result.output)
                self.assertFalse(self.model_entry.exists())
                self.assertEqual(list(self.model_cache.rglob('.staging-*')), [])
                self.catalogue.assert_not_called()
                self.process.assert_not_called()
                self.assertEqual(self.writes, [])
