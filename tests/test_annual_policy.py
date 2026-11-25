"""Synthetic annual-policy cases prepared for later execution; not yet run."""
from copy import deepcopy
import unittest

import numpy as np

from carbon.policy import RULE, validate_policy
from carbon.science import product_masks


class AnnualPolicyTests(unittest.TestCase):
    def test_rejects_missing_or_changed_policy(self):
        for policy in [None, {}, dict(RULE, ndvi_threshold=0.4)]:
            with self.assertRaises(ValueError):
                validate_policy(policy)

    def test_zero_and_exclusion_categories(self):
        # Positive model, genuine model zero, assumed zero, unavailable,
        # unknown classification, water. No country-specific fixture.
        land = np.array([[True, True, True, True, False, False]])
        source = np.array([[True, True, True, False, True, True]])
        model_valid = source.copy()
        height = np.array([[10., 0., 5., 0., 0., 0.]], np.float32)
        vi = np.array([[.5, .5, .2, .5, .5, .5]], np.float32)
        vi_valid = np.ones(land.shape, bool)
        unknown = np.array([[False, False, False, False, True, False]])
        assessed, status, carbon = product_masks(
            land, land & source, source, height, model_valid, vi, vi_valid,
            unknown, deepcopy(RULE))
        np.testing.assert_array_equal(status, [[4, 4, 7, 1, 5, 0]])
        np.testing.assert_array_equal(assessed, [[True, True, True, False, False, False]])
        self.assertEqual(float(carbon[0, 1]), 0.)
        self.assertEqual(float(carbon[0, 2]), 0.)
        self.assertTrue((carbon[0, 3:] == -9999).all())
        expected = (3.744 * height[0, 0].astype(np.float64) ** 1.391).astype(np.float32)
        self.assertEqual(carbon[0, 0], expected)


if __name__ == '__main__':
    unittest.main()
