"""Official tile grids, GeoMAD source validation, and WorldCover alignment."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from affine import Affine
from dep_tools.grids import PACIFIC_GRID_10
from dep_tools.loaders import OdcLoader
import numpy as np
from odc.geo.geobox import GeoBox
import pystac
import rasterio
from rasterio.errors import RasterioIOError
from rasterio.transform import array_bounds
from rasterio.windows import Window
from shapely.geometry import mapping

from src.task import run_tile, tile_sources
from src.utils import (
    GEOMAD_NODATA, MODEL_BUFFER, SUPPORTED_VERSION, WORLDCOVER, WORLDCOVER_INDEX, align_worldcover,
    check_geomad_sources, prepare, prepare_worldcover,
)


TILE, YEAR = '075_025', 2024


def samoa_plan():
    tile = PACIFIC_GRID_10.tile_geobox((75, 25))
    return dict(tile_id=TILE, tile=tile, buffered_tile=tile.pad(MODEL_BUFFER))


def source_item(col, row, *, year=YEAR, version=SUPPORTED_VERSION):
    left, bottom = -3000000 + col * 96000, -4000000 + row * 96000
    top = bottom + 96000
    key = f'dep_s2_geomad_{col:03d}_{row:03d}_{year}'
    version_path = version.replace('.', '-')
    return dict(id=key, collection='dep_s2_geomad', properties={
        'start_datetime': f'{year}-01-01T00:00:00Z', 'end_datetime': f'{year}-12-31T23:59:59Z',
        'proj:epsg': 3832, 'proj:shape': [9600, 9600],
        'proj:transform': [10, 0, left, 0, -10, top], 'proj:bbox': [left, bottom, left+96000, top],
    }, assets={band: dict(
        href=(f'https://dep-public-data.s3.us-west-2.amazonaws.com/dep_s2_geomad/'
              f'{version_path}/{col:03d}/{row:03d}/{year:04d}/{key}_{band}.tif'),
        **{'raster:bands': [dict(scale=1 if band == 'count' else .0001, offset=0,
                                 nodata=9999 if band == 'count' else 0, data_type='uint16')]},
    ) for band in ('red', 'green', 'blue', 'nir', 'count')})


def stac_item(fields):
    return pystac.Item.from_dict(fields | {'type': 'Feature', 'stac_version': '1.0.0', 'geometry': None, 'links': []})


class InputTests(unittest.TestCase):
    def test_official_tile_buffer_and_crop(self):
        for tile_id in ('075_025', '066_025'):
            with self.subTest(tile_id=tile_id):
                tile = PACIFIC_GRID_10.tile_geobox(tuple(map(int, tile_id.split('_'))))
                buffered = tile.pad(MODEL_BUFFER)
                self.assertEqual(tuple(tile.shape), (9600, 9600))
                self.assertEqual(tuple(buffered.shape), (10048, 10048))
                self.assertEqual(tile.crs.epsg, 3832)
                bounds = array_bounds(*buffered.shape, buffered.transform)
                left, bottom, right, top = tile.boundingbox.bbox
                self.assertEqual(bounds, (left-2240, bottom-2240, right+2240, top+2240))
                self.assertEqual(buffered.transform * Affine.translation(224, 224), tile.transform)
                self.assertEqual(buffered.overlap_roi(tile), (slice(224, 9824), slice(224, 9824)))

    def test_worldcover_unknown_class_and_missing_extent_have_distinct_masks(self):
        affine = Affine(.001, 0, -172, 0, -.001, -13)
        tile = GeoBox((2, 3), affine, 'EPSG:4326')
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source.tif'
            with rasterio.open(source, 'w', driver='GTiff', crs=str(tile.crs), transform=affine,
                               width=2, height=2, count=1, dtype='uint8', nodata=0) as dst:
                dst.write(np.array([[0, 10], [80, 255]], np.uint8), 1)
            values, mask = align_worldcover(tile, tile.extent.geom, [source])
            np.testing.assert_array_equal(values, [[0, 10, 0], [80, 255, 0]])
            np.testing.assert_array_equal(mask, [[True, True, False], [True, True, False]])
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(source, 'r+') as dst:
                dst.write_mask(np.array([[255, 255], [0, 255]], np.uint8))
            values, mask = align_worldcover(tile, tile.extent.geom, [source])
            np.testing.assert_array_equal(values, [[0, 10, 0], [0, 255, 0]])
            np.testing.assert_array_equal(mask, [[True, True, False], [False, True, False]])

            overlap = root/'overlap.tif'
            with rasterio.open(overlap, 'w', driver='GTiff', crs=str(tile.crs), transform=affine,
                               width=2, height=2, count=1, dtype='uint8', nodata=0) as dst:
                dst.write(np.array([[0, 10], [80, 255]], np.uint8), 1)
            values, mask = align_worldcover(tile, tile.extent.geom, [source, overlap])
            np.testing.assert_array_equal(values, [[0, 10, 0], [80, 255, 0]])
            np.testing.assert_array_equal(mask, [[True, True, False], [True, True, False]])
            with rasterio.open(overlap, 'r+') as dst:
                dst.write(np.array([[10, 10], [80, 255]], np.uint8), 1)
            with self.assertRaisesRegex(ValueError, 'Overlapping WorldCover sources disagree'):
                align_worldcover(tile, tile.extent.geom, [source, overlap])

    def test_worldcover_opens_only_indexed_tiles_and_masks_unlisted_areas(self):
        tile = GeoBox((2, 4), Affine(.001, 0, -171.002, 0, -.001, -13), 'EPSG:4326')
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source = root/'source.tif'
            with rasterio.open(source, 'w', driver='GTiff', crs=str(tile.crs), transform=tile.transform,
                               width=2, height=2, count=1, dtype='uint8', nodata=0) as dst:
                dst.write(np.array([[0, 10], [80, 20]], np.uint8), 1)

            for listed, expected_urls in (('S15W174', [WORLDCOVER.format('S15W174')]), ('S54E168', [])):
                with self.subTest(listed=listed):
                    response = Mock(json=Mock(return_value={'features': [{'properties': {'ll_tile': listed}}]}))
                    opened = []
                    real_open = rasterio.open

                    def open_raster(path, *args, **kwargs):
                        if isinstance(path, str) and path.startswith('https://'):
                            opened.append(path)
                            return real_open(source, *args, **kwargs)
                        return real_open(path, *args, **kwargs)

                    with patch('src.utils.requests.get') as get, patch('rasterio.open', side_effect=open_raster):
                        get.return_value.__enter__.return_value = response
                        classes, available = prepare_worldcover(tile)
                        get.assert_called_once_with(WORLDCOVER_INDEX, timeout=(10, 60))
                        response.raise_for_status.assert_called_once()

                    self.assertEqual(opened, expected_urls)
                    if expected_urls:
                        np.testing.assert_array_equal(classes, [[0, 10, 0, 0], [80, 20, 0, 0]])
                        np.testing.assert_array_equal(available, [[True, True, False, False]] * 2)
                    else:
                        self.assertFalse(available.any())

    def test_worldcover_reports_an_unreadable_indexed_tiff(self):
        tile = GeoBox((2, 2), Affine(.001, 0, -172, 0, -.001, -13), 'EPSG:4326')
        response = Mock(json=Mock(return_value={'features': [{'properties': {'ll_tile': 'S15W174'}}]}))
        with patch('src.utils.requests.get') as get, patch(
            'rasterio.open', side_effect=RasterioIOError('remote read failed'),
        ) as open_raster:
            get.return_value.__enter__.return_value = response
            with self.assertRaisesRegex(ValueError, 'Could not read indexed WorldCover TIFF .*S15W174_Map.tif'):
                prepare_worldcover(tile)
            open_raster.assert_called_once_with(WORLDCOVER.format('S15W174'))

    def geomad_fixture(self, root, *, masked=False):
        tile = samoa_plan()['tile']
        src = stac_item(source_item(75, 25))
        footprint = tile.extent.to_crs('EPSG:4326').geom
        src.geometry, src.bbox = mapping(footprint), list(footprint.bounds)
        src.stac_extensions = ['https://stac-extensions.github.io/projection/v1.1.0/schema.json',
                               'https://stac-extensions.github.io/raster/v1.1.0/schema.json']
        window = Window(tile.width - 2, 0, 2, 2)
        for band, nodata in GEOMAD_NODATA.items():
            with rasterio.Env(GDAL_TIFF_INTERNAL_MASK=True), rasterio.open(
                root/f'{band}.tif', 'w', driver='GTiff', crs=str(tile.crs), transform=tile.transform,
                width=tile.width, height=tile.height, count=1, dtype='uint16', nodata=nodata,
                tiled=True, SPARSE_OK='YES',
            ) as dst:
                dst.write(np.array([[100, 200], [300, nodata]], np.uint16), 1, window=window)
                if masked and band == 'red':
                    # An explicit mask marks 100 invalid while preserving a stored value distinct from NoData.
                    dst.write_mask(np.array([[0, 255], [255, 255]], np.uint8), window=window)

        real_open = rasterio.open

        def open_fixture(path, *args, **kwargs):
            if isinstance(path, str) and path.startswith('https://'):
                band = Path(path).stem.rsplit('_', 1)[1]
                path = root/f'{band}.tif'
            return real_open(path, *args, **kwargs)

        return src, tile[:2, tile.width-2:tile.width], open_fixture

    def test_geomad_checks_reject_metadata_the_loader_accepts(self):
        loader = OdcLoader(bands=['red'], groupby='solar_day', resampling='nearest', dtype='uint16', fail_on_error=True)
        with TemporaryDirectory() as temp:
            src, target, open_fixture = self.geomad_fixture(Path(temp))
            with patch('rasterio.open', side_effect=open_fixture):
                check_geomad_sources([src], YEAR)
                changed = src.clone()
                changed.properties.pop('proj:epsg')
                changed.properties['proj:code'] = 'EPSG:3832'
                changed.assets['red'].extra_fields['raster:bands'][0]['data_type'] = 'uint8'
                check_geomad_sources([changed], YEAR)
                with loader.load([changed], target) as geomad:
                    self.assertEqual(geomad.red.dtype, np.uint16)
                    np.testing.assert_array_equal(geomad.red.values[0], [[100, 200], [300, 0]])

                for invalid in (source_item(75, 25, year=2023), source_item(75, 25, version='0.5.0')):
                    changed = src.clone()
                    changed.assets['red'].href = invalid['assets']['red']['href']
                    with self.subTest(asset=changed.assets['red'].href):
                        with loader.load([changed], target) as geomad:
                            np.testing.assert_array_equal(geomad.red.values[0], [[100, 200], [300, 0]])
                        with self.assertRaisesRegex(ValueError, 'asset year or version'):
                            check_geomad_sources([changed], YEAR)

                for key, value in (('scale', .01), ('offset', 50), ('nodata', 100)):
                    changed = src.clone()
                    changed.assets['red'].extra_fields['raster:bands'][0][key] = value
                    with self.subTest(metadata=key):
                        with loader.load([changed], target) as geomad:
                            nodata = value if key == 'nodata' else 0
                            np.testing.assert_array_equal(geomad.red.values[0], [[100, 200], [300, nodata]])
                            self.assertEqual(geomad.red.rio.nodata, nodata)
                        with self.assertRaisesRegex(ValueError, 'scale, offset or NoData'):
                            check_geomad_sources([changed], YEAR)

                changed = src.clone()
                changed.properties['proj:transform'][2] += 2 * 96000
                with loader.load([changed], target) as geomad:
                    np.testing.assert_array_equal(geomad.red.values[0], [[0, 0], [0, 0]])
                with self.assertRaisesRegex(ValueError, 'TIFF grid, encoding or mask'):
                    check_geomad_sources([changed], YEAR)

                for key, value, error in (
                    ('proj:shape', [9600, 9601], 'grid metadata'),
                    ('proj:code', 'EPSG:4326', 'grid metadata'),
                    ('start_datetime', '2023-01-01T00:00:00Z', 'full 2024 annual interval'),
                    ('end_datetime', '2024-11-30T23:59:59Z', 'full 2024 annual interval'),
                ):
                    changed = src.clone()
                    changed.properties[key] = value
                    with self.subTest(metadata=key), self.assertRaisesRegex(ValueError, error):
                        check_geomad_sources([changed], YEAR)

                changed = src.clone()
                changed.assets['red'].extra_fields['bands'] = [{'nodata': 100}]
                check_geomad_sources([changed], YEAR)
                with loader.load([changed], target) as geomad:
                    with self.assertRaisesRegex(ValueError, 'Loaded GeoMAD NoData differs for red'):
                        prepare(geomad, target)

    def test_odc_loads_explicitly_masked_pixels_and_fills_missing_coverage(self):
        loader = OdcLoader(bands=['red'], groupby='solar_day', resampling='nearest', dtype='uint16', fail_on_error=True)
        with TemporaryDirectory() as temp:
            src, target, open_fixture = self.geomad_fixture(Path(temp), masked=True)
            buffered = GeoBox((2, 3), target.transform, target.crs)
            with patch('rasterio.open', side_effect=open_fixture), loader.load([src], buffered) as geomad:
                np.testing.assert_array_equal(geomad.red.values[0], [[100, 200, 0], [300, 0, 0]])
                self.assertEqual(geomad.red.rio.nodata, 0)
            with (patch('rasterio.open', side_effect=open_fixture),
                  self.assertRaisesRegex(ValueError, r'Unexpected GeoMAD TIFF grid, encoding or mask for .*/red')):
                check_geomad_sources([src], YEAR)

    def test_geomad_selection_keeps_only_positive_overlap_in_search_order(self):
        plan = samoa_plan()
        items = [stac_item(source_item(col, row)) for col in range(76, 73, -1) for row in range(26, 23, -1)]
        touching = source_item(77, 25)
        touching['properties']['proj:bbox'][0] = plan['buffered_tile'].boundingbox.bbox[2]
        selected = tile_sources(plan, YEAR, [stac_item(touching), *items])
        self.assertEqual([item.id for item in selected], [item.id for item in items])

    def test_invalid_geomad_rosters_fail_before_loading(self):
        plan = samoa_plan()
        centre = source_item(75, 25)
        missing_neighbour = [source_item(col, row) for col in range(74, 77) for row in range(24, 27)
                             if (col, row) != (74, 26)]
        for label, items, reason in (
            ('empty', [], 'No annual GeoMAD Items'),
            ('missing neighbour', missing_neighbour, 'full buffered tile'),
            ('wrong collection', [centre | {'collection': 'other'}], 'Unexpected GeoMAD collection'),
            ('duplicate', [centre, centre], 'Duplicate annual GeoMAD Item'),
        ):
            with self.subTest(label=label):
                searcher = Mock(search=Mock(return_value=[stac_item(item) for item in items]))
                loader, worldcover = Mock(), Mock()
                with self.assertRaisesRegex(ValueError, reason):
                    run_tile(plan, YEAR, searcher=searcher, loader=loader,
                             worldcover=worldcover, processor=Mock(), writer=Mock(),
                             stac_creator=Mock(), stac_writer=Mock(), model_uri='https://example.org/model.zip',
                             model_sha256='a' * 64)
                worldcover.assert_not_called()
                loader.load.assert_not_called()
