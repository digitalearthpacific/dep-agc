"""Tile product masks without a model or provider access."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from affine import Affine
import numpy as np
import rasterio
from carbon.accounting import read_worldcover, tile_products
from carbon.io import write_raster
from carbon.policy import RULE


CRS = 'EPSG:3832'
X, Y = 4200000, -1540000
GRID = Affine(10, 0, X, 0, -10, Y)


class TileProductTests(unittest.TestCase):
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
