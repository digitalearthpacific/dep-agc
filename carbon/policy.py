"""Explicit annual GeoMAD eligibility and assumed-zero contract.

Numerical equivalence and container qualification are pending. Historical
policies and receipts are retained outside this runtime and are not migrated.
"""

VERSION = 'annual-geomad-carbon-v2'
LAND_CLASSES = [10, 20, 30, 40, 50, 60, 70, 90, 95, 100]
RULE = {
    'version': VERSION,
    'source_product': 'GeoMAD',
    'source_version': '0.4.0',
    'known_land_classes': LAND_CLASSES,
    'water_class': 80,
    'unknown_class': 0,
    'ndvi_threshold': 0.3,
    'low_ndvi_treatment': 'assumed_zero',
    'assumed_zero_status': 7,
    'unavailable_treatment': 'excluded',
    'unknown_treatment': 'excluded',
    'water_treatment': 'excluded',
}


def validate_policy(config):
    """Require the single supported annual policy; never silently choose a mode."""
    if not isinstance(config, dict) or config != RULE:
        raise ValueError('Explicit supported annual GeoMAD carbon policy required')
    return config
