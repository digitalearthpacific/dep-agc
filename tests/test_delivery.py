from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from carbon.io import read_json
from tasks.tile import ROOT, scientific_profile


class DeliveryTests(unittest.TestCase):
    def test_release_and_scientific_profile_integrity(self):
        release = read_json(ROOT / 'release.json')
        self.assertEqual(release['name'], 'dep-agc')
        self.assertEqual(release['version'], '0.1.0-review.5')
        self.assertEqual(
            release['image_id'],
            'sha256:d0dbe6ec7fc6fd63f298cea6b7cb5c40155e14b4a515071de9a53b0f08c13887')
        self.assertNotIn('findings', release['qualification_record'])
        profile = scientific_profile()
        self.assertEqual(profile['policy']['version'], 'annual-geomad-carbon-v2')
        self.assertEqual(profile['policy']['low_ndvi_treatment'], 'assumed_zero')
        self.assertIsNone(profile['window'])
        for key in ['country', 'grid', 'inputs']:
            self.assertEqual(profile[key], {})

    def test_changed_profile_is_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'configs').mkdir()
            (root / 'release.json').write_bytes((ROOT / 'release.json').read_bytes())
            (root / 'configs/scientific-profile.json').write_text('{}')
            with patch('tasks.tile.ROOT', root), self.assertRaises(ValueError):
                scientific_profile()
