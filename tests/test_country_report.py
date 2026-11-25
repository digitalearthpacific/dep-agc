"""Tile masks and geodesic country weights without a model or provider access."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from affine import Affine
import numpy as np
import rasterio
from shapely.geometry import box

from spc_agb_dep.country_report import report_countries
from spc_agb_inputs.physical_area import ground_area_ha
from spc_agb_worker.accounting import read_worldcover, tile_products
from spc_agb_worker.io import sha256, write_raster
from spc_agb_worker.policy import RULE


CRS = 'EPSG:3832'
X, Y = 4200000, -1540000
GRID = Affine(10, 0, X, 0, -10, Y)


class CountryReportTests(unittest.TestCase):
    def test_tile_uses_worldcover_and_imagery_masks(self):
        task = {'grid': {'retained_shape': [1, 6], 'context': 0, 'crs': CRS}, 'policy': RULE}
        cover = np.array([[10, 10, 80, 0, 10, 10]], np.uint8)
        processed = np.array([[True, True, True, True, True, False]])
        imagery = np.array([[True, True, True, True, False, True]])
        arrays = {'rgb': np.ones((3, 1, 6), np.uint8), 'rgb_valid': imagery,
                  'red_valid': imagery, 'nir_valid': imagery, 'ndvi_valid': imagery,
                  'count_valid': imagery, 'count': np.ones((1, 6), np.uint16),
                  'ndvi': np.array([[.6, .2, .6, .6, .6, .6]], np.float32)}
        height = np.full((1, 6), 10, np.float32)
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'cover.tif'
            write_raster(path, cover, processed, GRID, CRS)
            output = Path(temp) / 'native'
            tile_products(task, arrays, height, GRID, read_worldcover(path, task, GRID), output)
            with rasterio.open(output / 'status.tif') as status:
                np.testing.assert_array_equal(status.read(1)[0, :5], [4, 7, 0, 5, 1])
                np.testing.assert_array_equal(status.read_masks(1) > 0, processed)
            with rasterio.open(output / 'carbon.tif') as carbon:
                np.testing.assert_array_equal(carbon.read_masks(1)[0] > 0, [1, 1, 0, 0, 0, 0])
                self.assertAlmostEqual(float(carbon.read(1)[0, 0]), 3.744 * 10**1.391, places=4)
                self.assertEqual(carbon.read(1)[0, 1], 0)

    def test_disjoint_countries_split_a_pixel_and_reporting_is_read_only(self):
        left = box(X, Y - 10, X + 4, Y)
        right = box(X + 4, Y - 10, X + 10, Y)
        whole = box(X, Y - 10, X + 10, Y)
        with TemporaryDirectory() as temp:
            carbon, status = (Path(temp) / f'{name}.tif' for name in ('carbon', 'status'))
            valid = np.ones((1, 1), bool)
            write_raster(carbon, np.array([[20]], np.float32), valid, GRID, CRS, -9999)
            write_raster(status, np.array([[4]], np.uint8), valid, GRID, CRS)
            before = [sha256(p) for p in (carbon, status)]
            report = report_countries(carbon, status, {'L': left, 'R': right}, pilot_footprint=whole)
            self.assertEqual(report['scope'], 'pilot-footprint totals')
            parts = report['countries']
            for name, piece in [('L', left), ('R', right)]:
                area = ground_area_ha(piece, CRS)
                self.assertAlmostEqual(parts[name]['assessed_ha'], area, places=10)
                self.assertAlmostEqual(parts[name]['stock_tC'], area * 20, places=9)
                self.assertEqual(parts[name]['coverage_percent'], 100)
            self.assertAlmostEqual(sum(r['assessed_ha'] for r in parts.values()),
                                   ground_area_ha(whole, CRS), places=9)
            with self.assertRaisesRegex(ValueError, 'overlap'):
                report_countries(carbon, status, {'L': left, 'R': whole})
            self.assertEqual([sha256(p) for p in (carbon, status)], before)

    def test_coverage_includes_excluded_and_unprocessed_pixels(self):
        footprint = box(X, Y - 10, X + 50, Y)
        with TemporaryDirectory() as temp:
            carbon, status = (Path(temp) / f'{name}.tif' for name in ('carbon', 'status'))
            write_raster(carbon, np.array([[0, -9999, -9999, -9999, -9999]], np.float32),
                         np.array([[1, 0, 0, 0, 0]], bool), GRID, CRS, -9999)
            write_raster(status, np.array([[7, 0, 5, 1, 0]], np.uint8),
                         np.array([[1, 1, 1, 1, 0]], bool), GRID, CRS)
            result = report_countries(carbon, status, {'A': footprint})['countries']['A']
            self.assertAlmostEqual(result['coverage_percent'], 20, places=6)
            self.assertEqual(result['stock_tC'], 0)
            self.assertEqual(result['assessed_pixels'], 1)
            self.assertAlmostEqual(result['footprint_ha'], sum(result[k] for k in (
                'assessed_ha', 'water_ha', 'unknown_ha', 'unavailable_ha', 'unprocessed_ha')))
