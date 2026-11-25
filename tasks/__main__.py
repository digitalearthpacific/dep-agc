"""Produce a DEP aboveground carbon tile."""

import argparse
import os
from pathlib import Path
import sys

from .tile import run_tile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(required=True)
    tile = commands.add_parser('tile', help='Produce a tile for an official Pacific tile ID and year')
    tile.add_argument('tile_id', help='Official PACIFIC_GRID_10 tile ID')
    tile.add_argument('year', type=int)
    tile.add_argument('--inputs', type=Path, help='Optional pinned annual input or saved-height replay bundle')
    tile.add_argument('--output', type=Path, required=True, help='New directory for the tile COGs and STAC Item')
    tile.add_argument('--work', type=Path, help='New working directory, defaults to OUTPUT-work')
    tile.add_argument('--model', type=Path, help='Directory holding the pinned model files')
    tile.add_argument('--source', type=Path, help='Directory holding the pinned inference source')

    args = parser.parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    try:
        work = args.work or args.output.with_name(args.output.name+'-work')
        worker = os.environ.get('SPC_WORKER_PYTHON', sys.executable)
        result = run_tile(args.tile_id, args.year, args.inputs, args.output, work,
                          model=args.model, source=args.source, worker_python=worker)
        print(args.output/result['item'])
    except (OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
