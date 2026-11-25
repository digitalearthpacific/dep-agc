"""Observation identity, separate from reusable spatial plans and model settings."""

SUPPORTED_VERSION = '0.4.0'


def validate_observation(year, version):
    """Validate a selectable calendar year; provider availability is checked later."""
    if type(year) is not int or not 1 <= year <= 9999:
        raise ValueError('Observation year must be an integer calendar year (1–9999)')
    if version != SUPPORTED_VERSION:
        raise ValueError('Only GeoMAD v0.4.0 is supported; no version substitution')
    return year


def annual_interval(year):
    validate_observation(year, SUPPORTED_VERSION)
    return (f'{year:04d}-01-01T00:00:00Z', f'{year:04d}-12-31T23:59:59Z')
