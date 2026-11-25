"""Exercise tile selection and its internal source roster without CUDA."""

import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from affine import Affine
import numpy as np
from odc.geo.geobox import GeoBox
import rasterio
from shapely.geometry import box

from carbon.io import read_json, sha256, write_json, write_raster
from tasks.__main__ import main
from tasks.inputs.geometry import pacific_tile_plan, provider_tiles
from tasks.inputs.live_tile import align_worldcover
from tasks.inputs.metadata import validate_item
from tasks.tile import run_tile


TILE, YEAR = '075_025', 2024


def source_item(col, row):
    left, bottom = -3000000 + col * 96000, -4000000 + row * 96000
    top = bottom + 96000
    key = f'dep_s2_geomad_{col:03d}_{row:03d}_{YEAR}'
    return dict(id=key, collection='dep_s2_geomad', properties={
        'start_datetime': f'{YEAR}-01-01T00:00:00Z', 'end_datetime': f'{YEAR}-12-31T23:59:59Z',
        'proj:epsg': 3832, 'proj:shape': [9600, 9600],
        'proj:transform': [10, 0, left, 0, -10, top], 'proj:bbox': [left, bottom, left+96000, top],
    }, assets={band: dict(
        href=f'https://dep-public-data.s3.us-west-2.amazonaws.com/dep_s2_geomad/0-4-0/{YEAR}/{key}/{band}.tif',
        **{'raster:bands': [dict(scale=1 if band == 'count' else .0001, offset=0,
                                 nodata=9999 if band == 'count' else 0, data_type='uint16')]},
    ) for band in ('red', 'green', 'blue', 'nir', 'count')})


class TileCommandTests(unittest.TestCase):
    def test_tile_command_discovers_inputs_without_a_manifest(self):
        plan = pacific_tile_plan(TILE)
        edge = plan['windows'][0]
        plan['windows'] = [edge]
        items = [source_item(*pair) for pair in provider_tiles(None, box(*edge['input_bounds']))]
        with TemporaryDirectory() as temp:
            root = Path(temp)
            cover = object()
            with patch('tasks.tile.pacific_tile_plan', return_value=plan), \
                 patch('tasks.tile.discover_geomad', return_value=items) as discover, \
                 patch('tasks.tile.live_tile_sources', return_value=contextlib.nullcontext(cover)) as sources, \
                 patch('tasks.tile.prepare_discovered_core', return_value={}) as prepare, \
                 patch('tasks.tile.CarbonProcessor') as processor:
                processor.return_value.process.side_effect = AssertionError('stop before inference')
                with self.assertRaisesRegex(AssertionError, 'stop before inference'):
                    main(['tile', TILE, str(YEAR), '--output', str(root/'tile'),
                          '--model', str(root/'model'), '--source', str(root/'source')])
            discover.assert_called_once_with(plan, YEAR)
            self.assertEqual(sources.call_args.args[1], sorted(items, key=lambda item: item['id']))
            self.assertEqual(prepare.call_args.args[1:], (edge, items))
            self.assertIs(prepare.call_args.kwargs['worldcover'], cover)
            processor.return_value.process.assert_called_once()

    def test_worldcover_unknown_class_and_missing_extent_have_distinct_masks(self):
        affine = Affine(.001, 0, -172, 0, -.001, -13)
        tile = GeoBox((2, 3), affine, 'EPSG:4326')
        with TemporaryDirectory() as temp:
            root = Path(temp)
            source, result = root/'source.tif', root/'aligned.tif'
            with rasterio.open(source, 'w', driver='GTiff', crs=str(tile.crs), transform=affine,
                               width=2, height=2, count=1, dtype='uint8', nodata=0) as dst:
                dst.write(np.array([[0, 10], [80, 95]], np.uint8), 1)
            align_worldcover(tile, tile.extent.geom, [source], result)
            with rasterio.open(result) as ds:
                self.assertIsNone(ds.nodata)
                np.testing.assert_array_equal(ds.read(1), [[0, 10, 0], [80, 95, 0]])
                np.testing.assert_array_equal(ds.read_masks(1) > 0, [[True, True, False], [True, True, False]])

    def test_tile_command_plans_context_and_keeps_windows_in_work(self):
        plan = pacific_tile_plan(TILE)
        edge = plan['windows'][0]
        plan['windows'] = [edge]
        required = [source_item(*pair) for pair in provider_tiles(None, box(*edge['input_bounds']))]
        self.assertEqual(len(required), 4)
        extra = source_item(77, 25)
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inputs, output = root/'inputs.json', root/'tile'
            asset = root/'band.tif'
            write_raster(asset, np.ones((1, 1), np.uint16), np.ones((1, 1), bool),
                         Affine(10, 0, 4200000, 0, -10, -1504000), 'EPSG:3832')
            asset_pin = dict(path=asset.name, sha256=sha256(asset))
            geomad = []
            for item in [*required, extra]:
                item_path = root/(item['id']+'.json')
                write_json(item_path, item)
                geomad.append(dict(item=dict(path=item_path.name, sha256=sha256(item_path)),
                                   assets=dict.fromkeys(item['assets'], asset_pin)))
            write_json(inputs, dict(schema='agc-tile-inputs-v1', tile=TILE, year=YEAR,
                                    geomad=geomad, worldcover=asset_pin))

            def prepare(grid, window, items, route, worldcover, profile, year, package):
                self.assertEqual(items, required)
                package.mkdir(parents=True)
                write_json(package/'task.json', {})
                return {}

            def process(loaded):
                worker = loaded['package'].parent/'worker'
                (worker/'prepared').mkdir(parents=True)
                (worker/'prepared/tci-input.tif').write_bytes(b'RGB fixture')
                (worker/'height.tif').write_bytes(b'height fixture')
                return {'completion': {'height': {'raw': 'height.tif'}}}

            with patch('tasks.tile.pacific_tile_plan', return_value=plan), \
                 patch('tasks.tile.validate_item', wraps=validate_item) as check_item, \
                 patch('tasks.tile.read_worldcover'), patch('tasks.tile.prepare_core', side_effect=prepare), \
                 patch('tasks.tile.CarbonProcessor') as processor, \
                 patch('tasks.tile.assemble_tile', return_value={'item': TILE+'.stac-item.json'}) as assemble, \
                 contextlib.redirect_stdout(io.StringIO()) as stdout:
                processor.return_value.process.side_effect = process
                self.assertEqual(main(['tile', TILE, str(YEAR), '--inputs', str(inputs), '--output', str(output),
                                       '--model', str(root/'model'), '--source', str(root/'source')]), 0)
            self.assertEqual([call.args[0] for call in check_item.call_args_list], [*required, extra])
            processor.return_value.process.assert_called_once()
            self.assertEqual(stdout.getvalue().strip(), str(output/(TILE+'.stac-item.json')))
            args = assemble.call_args.args
            self.assertEqual((args[0], args[1], args[4]), (TILE, YEAR, output))
            self.assertEqual(set(args[2]), {edge['id']})
            self.assertEqual({item['id'] for item in args[3]}, {item['id'] for item in required})
            replay = read_json(root/'tile-work/replay.json')
            self.assertEqual(set(replay['windows']), {edge['id']})
            self.assertEqual(replay['windows'][edge['id']]['source_ids'], sorted(item['id'] for item in required))

    def test_replay_dispatches_saved_height_inputs_without_inference(self):
        edge = pacific_tile_plan(TILE)['windows'][0]
        items = [source_item(*pair) for pair in provider_tiles(None, box(*edge['input_bounds']))]
        saved = dict(source_ids=[item['id'] for item in items], raw_height={'path': 'saved/height.tif'})
        with TemporaryDirectory() as temp:
            root = Path(temp)
            inputs = root/'replay.json'
            write_json(inputs, dict(schema='agc-tile-replay-v1', tile=TILE, year=YEAR,
                                   source_items=items, windows={edge['id']: saved}))
            with patch('tasks.tile.replay_core') as replay, patch('tasks.tile.CarbonProcessor') as processor, \
                 patch('tasks.tile.validate_item', wraps=validate_item) as check_item, \
                 patch('tasks.tile.assemble_tile') as assemble:
                run_tile(TILE, YEAR, inputs, root/'tile', root/'work')
            processor.assert_not_called()
            replay.assert_called_once()
            self.assertEqual(replay.call_args.args[:2], (root, saved))
            self.assertEqual([call.args[0] for call in check_item.call_args_list], items)
            assemble.assert_called_once()

    def test_replay_rejects_missing_neighbour_before_processing(self):
        with TemporaryDirectory() as temp:
            root = Path(temp)
            item = source_item(75, 25)
            inputs = root/'inputs.json'
            write_json(inputs, dict(schema='agc-tile-replay-v1', tile=TILE, year=YEAR,
                                   source_items=[item], windows={'r00c00': {'source_ids': [item['id']]}}))
            with patch('tasks.tile.replay_core') as replay, patch('tasks.tile.CarbonProcessor') as processor:
                with self.assertRaisesRegex(ValueError, 'full imagery context'):
                    run_tile(TILE, YEAR, inputs, root/'tile', root/'work')
            replay.assert_not_called()
            processor.assert_not_called()
            self.assertFalse((root/'tile').exists())
