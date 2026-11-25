"""Produce a DEP carbon tile or report country stock from a completed tile."""

import argparse
import os
from pathlib import Path
import sys

import rasterio

from carbon.io import write_json
from .country_report import report_tile
from .tile import run_tile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    tile = commands.add_parser('tile', help='Produce a tile for an official Pacific tile ID and year')
    tile.add_argument('tile_id', help='Official PACIFIC_GRID_10 tile ID')
    tile.add_argument('year', type=int)
    tile.add_argument('--inputs', type=Path, help='Optional pinned annual input or saved-height replay bundle')
    tile.add_argument('--output', type=Path, required=True, help='New directory for the tile COGs and STAC Item')
    tile.add_argument('--work', type=Path, help='New working directory, defaults to OUTPUT-work')
    tile.add_argument('--model', type=Path, help='Directory holding the pinned model files')
    tile.add_argument('--source', type=Path, help='Directory holding the pinned inference source')

    report = commands.add_parser('report', help='Report country stock and coverage from a completed tile')
    report.add_argument('item', type=Path, help='Local tile STAC Item JSON')
    report.add_argument('countries', type=Path, help='GeoJSON FeatureCollection of country polygons')
    report.add_argument('--polygon-crs', choices=['EPSG:4326', 'EPSG:3832'], default='EPSG:4326',
                        help='CRS of both country and footprint inputs')
    report.add_argument('--footprint', type=Path, help='Optional GeoJSON reporting footprint')
    report.add_argument('--output', type=Path, required=True, help='New country report JSON file')
    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    try:
        if args.command == 'tile':
            work = args.work or args.output.with_name(args.output.name+'-work')
            worker = os.environ.get('SPC_WORKER_PYTHON', sys.executable)
            result = run_tile(args.tile_id, args.year, args.inputs, args.output, work,
                              model=args.model, source=args.source, worker_python=worker)
            print(args.output/result['item'])
        else:
            if args.output.exists():
                raise FileExistsError(args.output)
            with rasterio.Env(GDAL_PAM_ENABLED='NO', PROJ_NETWORK='OFF'):
                result = report_tile(args.item, args.countries, args.polygon_crs, args.footprint)
            write_json(args.output, result)
            print(args.output)
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
