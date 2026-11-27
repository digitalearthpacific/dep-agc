"""Carbon calculations, raster masks, and height patch stitching."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock, patch

from affine import Affine
import numpy as np
from odc.geo.geobox import GeoBox
import rasterio
from src.utils import (
    MODEL_BUFFER, STATUS_NODATA, carbon_and_status, infer_height, read_raw_height,
    read_worldcover, tile_products, write_raster,
)


CRS = 'EPSG:3832'
X, Y = 4200000, -1540000
GRID = Affine(10, 0, X, 0, -10, Y)


class ProductTests(unittest.TestCase):
    def test_height_validation_checks_rgb_valid_pixels_in_the_buffer(self):
        buffered = GeoBox((1, 2), GRID, CRS).pad(MODEL_BUFFER)
        height = np.ones(buffered.shape, np.float32)
        height[-1, :3] = [-1, np.nan, np.inf]
        valid = np.ones(buffered.shape, bool)
        valid[-1, :3] = False
        with TemporaryDirectory() as temp:
            path = Path(temp)/'height.tif'
            write_raster(path, height, valid, buffered.transform, CRS)
            np.testing.assert_array_equal(read_raw_height(path, buffered, valid), height)
            for col in range(3):
                with self.subTest(height=height[-1, col]):
                    valid[-1, col] = True
                    with self.assertRaisesRegex(ValueError, 'Nonfinite or negative runtime height'):
                        read_raw_height(path, buffered, valid)
                    valid[-1, col] = False

    def test_carbon_formula_ndvi_cutoff_zeros_and_nodata(self):
        tile = GeoBox((1, 10), GRID, CRS)
        cutoff = np.float32(0.3)
        below = np.nextafter(cutoff, np.float32(-np.inf))
        above = np.nextafter(cutoff, np.float32(np.inf))
        cover = np.array([[10, 10, 80, 0, 10, 10, 10, 10, 10, 10]], np.uint8)
        processed = np.array([[True, True, True, True, True, False, True, True, True, True]])
        # Pixel 4 has invalid RGB. Pixel 9 has valid RGB but invalid NDVI.
        arrays = {'rgb_valid': np.array([[True, True, True, True, False, True, True, True, True, True]]),
                  'ndvi_valid': np.array([[True, True, True, True, True, True, True, True, True, False]]),
                  'ndvi': np.array([[.6, .2, .6, .6, .6, .6, below, cutoff, above, -9999]], np.float32)}
        height = np.full((1, 10), 10, np.float32)
        height[0, 7] = 0
        # Use distinct buffer values so a misplaced crop changes the results.
        arrays = {name: np.pad(values, MODEL_BUFFER, constant_values=False) for name, values in arrays.items()}
        height = np.pad(height, MODEL_BUFFER, constant_values=123)
        buffered = tile.pad(MODEL_BUFFER)
        with TemporaryDirectory() as temp:
            path = Path(temp) / 'cover.tif'
            write_raster(path, cover, processed, GRID, CRS)
            height_path = Path(temp) / 'height.tif'
            write_raster(height_path, height, arrays['rgb_valid'], buffered.transform, CRS)
            validated_height = read_raw_height(height_path, buffered, arrays['rgb_valid'])
            output = Path(temp) / 'native'
            tile_products(tile, buffered, arrays, validated_height, read_worldcover(path, tile), output)
            self.assertEqual({path.name for path in output.iterdir()}, {'carbon.tif', 'status.tif'})
            with rasterio.open(output / 'status.tif') as status:
                self.assertEqual(status.transform, GRID)
                self.assertEqual(status.shape, (1, 10))
                self.assertEqual(status.nodata, STATUS_NODATA)
                values = status.read(1)
                valid = status.read_masks(1) > 0
                np.testing.assert_array_equal(valid, processed)
                np.testing.assert_array_equal(values[valid], [4, 7, 0, 5, 1, 7, 4, 4, 1])
                np.testing.assert_array_equal(values[~valid], STATUS_NODATA)
            with rasterio.open(output / 'carbon.tif') as carbon:
                valid = carbon.read_masks(1)[0] > 0
                np.testing.assert_array_equal(valid, [1, 1, 0, 0, 0, 0, 1, 1, 1, 0])
                density = carbon.read(1)[0]
                np.testing.assert_array_equal(valid, density != carbon.nodata)
                self.assertEqual(density[0], np.float32(3.744 * 10**1.391))
                np.testing.assert_array_equal(density[[1, 6, 7]], 0)
                np.testing.assert_array_equal(density[[4, 9]], -9999)
                self.assertEqual(density[8], density[0])

    def test_modelled_density_must_fit_float32(self):
        height = np.array([[np.finfo(np.float32).max]], dtype=np.float32)
        valid = np.array([[True]])
        ndvi = np.array([[.5]], dtype=np.float32)
        with self.assertRaisesRegex(ValueError, 'Modelled carbon density exceeds the float32 range'):
            carbon_and_status(valid, height, valid, ndvi, valid, ~valid)

    def test_height_inference_skips_empty_patches_and_stitches_centres(self):
        import torch

        # A 128-pixel patch at stride 64 splits these axes at 96 and 160.
        rgb = np.full((3, 192, 320), 128, dtype=np.uint8)
        rgb[:, :, 192:] = 0
        rgb[:, 40, 65] = 0
        offsets = torch.arange(128, dtype=torch.float32)
        ramp = offsets[:, None] / 64 + offsets / 128
        patch_numbers = iter(range(6))

        def predict(normalized):
            expected = (torch.tensor([128 / 255] * 3) - torch.tensor([.485, .456, .406]))
            expected /= torch.tensor([.229, .224, .225])
            torch.testing.assert_close(normalized[0, :, 0, 0], expected)
            height = ramp + next(patch_numbers) / 4
            return torch.stack([torch.full_like(height, 99), height])[None]

        fabric = Mock(device=torch.device('cpu'), setup=lambda model: model,
                      setup_dataloaders=lambda batches: batches)
        model = Mock(side_effect=predict)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'training_config.cfg').write_text('[general]\nn_classes = 2\n[data]\ns = 128\n')
            path = root/'rgb.tif'
            with rasterio.open(path, 'w', driver='GTiff', width=320, height=192, count=3,
                               dtype='uint8', transform=GRID, crs=CRS) as dst:
                dst.write(rgb)
            with patch('lightning.Fabric', return_value=fabric), patch('torch.jit.load', return_value=model):
                height_path = infer_height(root, path, root/'height')

            self.assertEqual(model.call_count, 6)
            y, x = np.ogrid[:192, :320]
            patch_row = (y >= 96).astype(int)
            patch_col = (x >= 96).astype(int) + (x >= 160)
            prediction = (3 * patch_row + patch_col) / 4 + (y - 64 * patch_row) / 64 + (x - 64 * patch_col) / 128
            expected = np.maximum(prediction * 60, 1) - 1
            expected[rgb[0] == 0] = 0
            with rasterio.open(height_path) as raster:
                self.assertEqual(raster.transform, GRID)
                self.assertEqual(raster.dtypes, ('float32',))
                np.testing.assert_array_equal(raster.read(1), expected)
