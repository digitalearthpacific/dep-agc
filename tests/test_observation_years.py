"""Annual GeoMAD validation fixtures."""
from copy import deepcopy
import unittest


from carbon.observation import validate_observation, annual_interval
from tasks.inputs.metadata import validate_item


def item(year):
    start, end = annual_interval(year)
    return dict(id=f'dep_s2_geomad_075_025_{year:04d}', collection='dep_s2_geomad',
        properties={'start_datetime': start, 'end_datetime': end, 'proj:epsg': 3832,
                    'proj:shape': [9600, 9600], 'proj:transform': [10, 0, 0, 0, -10, 0]},
        assets={band: {'href': f'https://dep-public-data.s3.us-west-2.amazonaws.com/dep_s2_geomad/0-4-0/{year:04d}/{band}.tif',
                       'raster:bands': [dict(scale=1 if band == 'count' else .0001, offset=0,
                                           nodata=9999 if band == 'count' else 0, data_type='uint16')]}
                for band in ['red', 'green', 'blue', 'nir', 'count']})


class ObservationTests(unittest.TestCase):
    def test_projection_code_and_legacy_epsg_preserve_the_same_grid(self):
        normalized = item(2024)
        props = normalized['properties']
        props.pop('proj:epsg')
        props['proj:code'] = 'EPSG:3832'
        self.assertEqual(validate_item(normalized, 2024, '0.4.0'), normalized)
        props['proj:epsg'] = 4326
        with self.assertRaises(ValueError):
            validate_item(normalized, 2024, '0.4.0')

    def test_multiple_years_and_wrong_item_interval_asset_or_version(self):
        for year in [2022, 2024, 2025]:
            self.assertEqual(validate_observation(year, '0.4.0'), year)
            valid = item(year)
            self.assertEqual(validate_item(valid, year, '0.4.0'), valid)
            for changed in [item(year - 1), deepcopy(valid)]:
                if changed['id'] == valid['id']:
                    changed['assets']['red']['href'] = item(year - 1)['assets']['red']['href']
                with self.assertRaises(ValueError):
                    validate_item(changed, year, '0.4.0')
            changed = deepcopy(valid)
            changed['properties']['end_datetime'] = f'{year}-11-30T23:59:59Z'
            with self.assertRaises(ValueError):
                validate_item(changed, year, '0.4.0')
        for year, version in [(True, '0.4.0'), (0, '0.4.0'), (2024, '0.5.0')]:
            with self.assertRaises(ValueError):
                validate_observation(year, version)

if __name__ == '__main__':
    unittest.main()
