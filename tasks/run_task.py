"""One explicit GeoMAD -> fresh height/carbon -> native COG/STAC job."""

import argparse
import logging
import os
from pathlib import Path
import re
import sys
import time

from dep_tools.task import AreaTask
from odc.geo.geobox import GeoBox

from carbon.observation import validate_observation
from carbon.grid import receiving_transform, validate_grid
from carbon.io import read_json, sha256, validate, write_json
from .loading import NativeGeoMADLoader
from .job import resolve_job, verify_inputs
from .catalog import country_plan, resolve_window
from .completion import verify_completion
from .outputs import NativeCogStacWriter
from .processing import CarbonProcessor


def task_id(task):
    validate_observation(task['year'], task['source_version'])
    parts = [task['country']['code'], task['grid']['id'], task['window'], str(task['year'])]
    if any(not re.fullmatch(r'[A-Za-z0-9_+.-]+', part) or part in ['.', '..'] for part in parts):
        raise ValueError('Safe explicit country/grid/window identifiers required')
    return tuple(parts)


def run(args):
    if getattr(args, 'verify_inputs', False):
        binding = resolve_job(args.job, args.year, args.source_version)
        return dict(status='inputs-verified', task_id=list(task_id(read_json(binding['task']))),
                    job_sha256=sha256(args.job), **verify_inputs(binding, args.model, args.source))
    if getattr(args, 'skip_completed', False) and args.output.exists():
        binding = resolve_job(args.job, args.year, args.source_version)
        verify_inputs(binding, args.model, args.source)
        pin = getattr(args, 'completion_sha256', None)
        if ((args.output/'integration-failure.json').exists()
                or not (args.output/'integration-completion.json').is_file()):
            raise FileExistsError('Failed/incomplete output preserved; no automatic retry')
        result = verify_completion(args.output, args.job, binding, pin)
        return dict(result, status='skipped-verified-complete')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    try:
        binding = resolve_job(args.job, args.year, args.source_version)
        task = read_json(binding['task'])
        validate_grid(task)
        identifier = task_id(task)
        area = GeoBox(
            tuple(reversed(task['grid']['receiving_window'][2:])),
            receiving_transform(task['grid']), task['grid']['crs'],
        )
        loader = NativeGeoMADLoader(
            binding['native_index'], binding['native_index_sha256'],
            binding['package'], binding['task'], args.output/'input',
        )
        if args.prepare_only:
            loaded = loader.load(area)
            pins = validate(task, loaded['package'], args.model, args.source)
            result = dict(status='prepared', task_id=list(identifier), input_pins=len(pins),
                          seconds=time.monotonic()-started, model_loaded=False)
            write_json(args.output/'input-completion.json', result)
            return result
        job = AreaTask(
            id=identifier, area=area, loader=loader,
            processor=CarbonProcessor(
                python=args.worker_python, code=args.worker_code,
                model=args.model, source=args.source, output=args.output/'worker',
                timeout=args.worker_timeout,
            ),
            writer=NativeCogStacWriter(args.output/'product', binding['native_index'].parent),
            logger=logging.getLogger('AGC'),
        )
        generated = job.run()
        result = dict(
            status='complete', task_id=list(identifier), **generated,
            seconds=time.monotonic()-started, job_sha256=sha256(args.job),
            scientific_method_unchanged=True, DEP_execution_qualified=False,
        )
        write_json(args.output/'integration-completion.json', result)
        return result
    except Exception as exc:
        write_json(args.output/'integration-failure.json', dict(
            type=type(exc).__name__, error=str(exc), seconds=time.monotonic()-started,
            job=str(args.job), year=args.year, source_version=args.source_version, retry=False,
        ))
        raise


def select_arguments(args):
    """Resolve run selection; the existing run() remains the single runner."""
    args.completion_sha256 = None
    if args.catalog:
        if not args.country or not args.window:
            raise ValueError('--catalog requires --country and --window')
        selected = country_plan(args.catalog, args.country, args.year, args.source_version)
        resolved = resolve_window(selected, args.window)
        args.job = resolved['job']
        if args.output is None:
            args.output = resolved['output']
        if args.output == resolved['output']:
            args.completion_sha256 = resolved['completion_sha256']
    elif args.country or args.window:
        raise ValueError('--country/--window require --catalog; explicit --job remains supported')
    if args.verify_inputs:
        if args.prepare_only or args.skip_completed:
            raise ValueError('--verify-inputs does not generate inputs or execute/skip a job')
    elif args.output is None or args.model is None or args.source is None:
        raise ValueError('Execution/preparation requires --output, --model and --source')
    if args.skip_completed and args.prepare_only:
        raise ValueError('--skip-completed applies to full jobs, not prepare-only output')
    return args


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument('--job', type=Path, help='Pinned package-relative job JSON')
    choice.add_argument('--catalog', type=Path, help='Index of prepared country plans/jobs')
    parser.add_argument('--country')
    parser.add_argument('--window')
    parser.add_argument('--year', type=int, required=True)
    parser.add_argument('--source-version', required=True)
    for name in ['model', 'source', 'output']:
        parser.add_argument('--'+name, type=Path)
    parser.add_argument('--verify-inputs', action='store_true', help='Hash existing inputs only; no raster/model/output generation')
    parser.add_argument('--skip-completed', action='store_true', help='Skip only after identity and required-output verification')
    parser.add_argument('--prepare-only', action='store_true', help='Input/identity preflight; no model load')
    parser.add_argument('--worker-timeout', type=int, default=900)
    args = parser.parse_args()
    args.worker_python = Path(os.environ.get('SPC_WORKER_PYTHON', sys.executable))
    args.worker_code = Path(__file__).resolve().parents[1]
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.absolute())
    logging.basicConfig(level=logging.INFO)
    try:
        result = run(select_arguments(args))
        target = f' -> {args.output}' if args.output is not None else ''
        print(f"{result['status']}: {'/'.join(result['task_id'])}{target}")
    except Exception as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
