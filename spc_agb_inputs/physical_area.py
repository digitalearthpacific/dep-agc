"""Horizontal ellipsoid helpers from forest_carbon_accounting at4277015.

Ordinary imports for the input producer; numerical operation order is unchanged.
The experimental script remains intact. Equivalence is checked in tests.
"""
import math
from functools import lru_cache
import numpy as np
from pyproj import Geod, Transformer

GEOD = Geod(ellps='WGS84')


@lru_cache(maxsize=16)
def coordinate_transformer(source, target='EPSG:4326'):
    return Transformer.from_crs(source, target, always_xy=True)


def ground_area_ha(geometry, crs):
    """Horizontal WGS84 ellipsoid area; projected edges densified at <=5m."""
    if geometry.is_empty or geometry.area==0:return 0.
    if geometry.geom_type in ['MultiPolygon','GeometryCollection']:
        return sum(ground_area_ha(g,crs) for g in geometry.geoms)
    if geometry.geom_type!='Polygon':return 0.
    project=coordinate_transformer(crs).transform
    def ring_area(ring):
        points=[];coordinates=list(ring.coords)
        for a,b in zip(coordinates,coordinates[1:]):
            n=max(1,math.ceil(math.dist(a[:2],b[:2])/5))
            for fraction in np.arange(n)/n:points.append((a[0]+fraction*(b[0]-a[0]),a[1]+fraction*(b[1]-a[1])))
        lon,lat=project(*zip(*points));area,_=GEOD.polygon_area_perimeter(lon,lat)
        return abs(area)
    return (ring_area(geometry.exterior)-sum(ring_area(r) for r in geometry.interiors))/10000
