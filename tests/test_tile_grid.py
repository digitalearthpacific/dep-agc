"""Official tile partition and receiving-grid fixtures."""

import unittest

from dep_tools.grids import PACIFIC_GRID_10
from shapely.geometry import box
from shapely.ops import unary_union

from carbon.grid import validate_grid
from tasks.inputs.geometry import pacific_tile_plan


class TileGridTests(unittest.TestCase):
    def test_official_tile_partition_and_edge_context(self):
        grid = pacific_tile_plan('075_025')
        tile = box(4200000, -1600000, 4296000, -1504000)
        cores = [box(*w['bounds']) for w in grid['windows']]
        self.assertEqual(len(cores), 25)
        self.assertTrue(unary_union(cores).equals(tile))
        self.assertEqual(sum(c.area for c in cores), tile.area)
        for w, core in zip(grid['windows'], cores):
            self.assertEqual(w['retained_shape'], [1920, 1920])
            self.assertEqual(core.area, 19200**2)
            self.assertEqual(w['global_input_window'][2:], [2368, 2368])
            self.assertTrue(box(*w['input_bounds']).equals(core.buffer(2240, join_style='mitre')))
        for window, neighbours in [
            (grid['windows'][0], [(74, 25), (74, 26), (75, 25), (75, 26)]),
            (grid['windows'][-1], [(75, 24), (75, 25), (76, 24), (76, 25)]),
        ]:
            context = box(*window['input_bounds'])
            source_tiles = [box(*PACIFIC_GRID_10[key].boundingbox.bbox) for key in neighbours]
            self.assertTrue(unary_union(source_tiles).covers(context))
            self.assertTrue(all(source.intersection(context).area > 0 for source in source_tiles))

    def test_worker_accepts_equal_edge_core_and_rejects_wrong_context(self):
        grid = pacific_tile_plan('075_025')
        w = grid['windows'][-1]
        g = dict(grid, receiving_window=w['global_input_window'][:],
                 global_input_window=w['global_input_window'][:], canonical_global_offset=[0, 0],
                 retained_shape=w['retained_shape'], bounds=w['bounds'])
        task = dict(schema='spc-agb-window-v3', window=w['id'], grid=g)
        self.assertEqual(validate_grid(task).to_epsg(), 3832)
        task['grid']['receiving_window'][2] -= 1
        with self.assertRaisesRegex(ValueError, 'Full receiving'):
            validate_grid(task)
