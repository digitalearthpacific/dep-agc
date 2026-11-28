"""Produce a DEP aboveground carbon tile."""

from functools import cache, partial
from hashlib import sha256
import logging
from pathlib import Path, PurePosixPath
import re
from shutil import copyfileobj
from tempfile import TemporaryDirectory
from typing import Annotated
from urllib.parse import urlsplit
from zipfile import BadZipFile, ZipFile

import boto3
from botocore.exceptions import BotoCoreError, ClientError
from dep_tools.aws import write_to_s3
from dep_tools.grids import PACIFIC_GRID_10
from dep_tools.loaders import OdcLoader
from dep_tools.namers import S3ItemPath
from dep_tools.searchers import PystacSearcher
from dep_tools.stac_utils import StacCreator
from dep_tools.writers import AwsDsCogWriter, AwsStacWriter
from filelock import FileLock
from odc.stac import configure_s3_access
from pystac_client import Client
from pystac_client.exceptions import APIError
from pystac_client.stac_api_io import StacApiIO
import requests
import typer

from src.task import run_tile
from src.utils import (
    DEP_STAC_URL, GEOMAD_COLLECTION, GEOMAD_NODATA, MODEL_BUFFER, MODEL_WEIGHTS,
    parse_years, prepare_worldcover, process_tile,
)


logger = logging.getLogger('dep-agc')
DATASET_ID = 'aboveground-carbon'
OUTPUT_PREFIX = 'local-agc'
MODEL_CACHE = Path('models')
MODEL_FILES = (MODEL_WEIGHTS, 'training_config.cfg')
MODEL_CHECKSUM = 'model.sha256'


def stable_model_uri(url):
    """Keep the HTTPS object URL and its optional S3 version selector."""
    version = '&'.join(param for param in url.query.split('&') if param.partition('=')[0] == 'versionId')
    return url._replace(netloc=url.netloc.rsplit('@', 1)[-1], query=version, fragment='').geturl()


def cached_model_sha256(entry, zip_name):
    if not all((entry/name).is_file() for name in (zip_name, *MODEL_FILES, MODEL_CHECKSUM)):
        return None
    try:
        digest = (entry/MODEL_CHECKSUM).read_text(encoding='ascii').strip()
    except (FileNotFoundError, UnicodeError):
        return None
    return digest if re.fullmatch(r'[0-9a-f]{64}', digest) else None


def download_model(uri, archive_path):
    digest = sha256()
    with requests.get(uri, stream=True, timeout=(10, 60)) as response, archive_path.open('xb') as download:
        response.raise_for_status()
        for chunk in response.iter_content(chunk_size=1024 * 1024):
            download.write(chunk)
            digest.update(chunk)
    with ZipFile(archive_path) as archive:
        for filename in MODEL_FILES:
            members = [entry for entry in archive.infolist()
                       if not entry.is_dir() and PurePosixPath(entry.filename).name == filename]
            if len(members) != 1:
                raise ValueError(f'Model ZIP must contain exactly one {filename}')
            with archive.open(members[0]) as source, archive_path.with_name(filename).open('xb') as output:
                copyfileobj(source, output)
    return digest.hexdigest()


def prepare_model(download_uri, model_uri, zip_name):
    """Return the model cached for the stable URI and its original ZIP checksum."""
    cache = MODEL_CACHE/zip_name
    key = sha256(model_uri.encode()).hexdigest()
    entry = cache/key
    digest = cached_model_sha256(entry, zip_name)
    if digest is not None:
        return entry, digest

    cache.mkdir(parents=True, exist_ok=True)
    with FileLock(str(cache/f'.{key}.lock')):
        # A job sharing the cache may have published this version while we waited.
        digest = cached_model_sha256(entry, zip_name)
        if digest is not None:
            return entry, digest
        with TemporaryDirectory(dir=cache, prefix='.staging-') as staging:
            staged = Path(staging)/'entry'
            staged.mkdir()
            digest = download_model(download_uri, staged/zip_name)
            (staged/MODEL_CHECKSUM).write_text(digest + '\n', encoding='ascii')
            # Replace an incomplete entry only after its replacement is ready.
            if entry.exists() or entry.is_symlink():
                entry.rename(Path(staging)/'incomplete')
            staged.rename(entry)
    return entry, digest


def main(
    tile_id: Annotated[str, typer.Option(help='Official PACIFIC_GRID_10 tile ID such as 075_025')],
    year: Annotated[str, typer.Option(help='Calendar year (2024) or inclusive range (2024-2025)')],
    model_zip_uri: Annotated[str, typer.Option(help='Public HTTPS URL for the ZIP with the pinned model files')],
    output_bucket: Annotated[str, typer.Option(help='S3 bucket name for the COGs and STAC Item')],
    version: Annotated[str, typer.Option(help='Output data version for S3 paths')],
    collection_url_root: Annotated[
        str, typer.Option(help='HTTPS collections URL for the output STAC catalogue')
    ],
    overwrite: Annotated[bool, typer.Option('--overwrite/--no-overwrite', help='Replace existing results')] = False,
):
    """Produce a DEP aboveground carbon tile."""
    # Send logs to stderr so stdout only shows Item URIs.
    logging.basicConfig(level=logging.INFO, format='%(asctime)s | %(name)s | %(levelname)s | %(message)s')
    logging.getLogger('botocore').setLevel(logging.WARNING)
    try:
        # Check the inputs before connecting to S3 or creating work folders.
        try:
            years = parse_years(year)
        except ValueError as exc:
            raise typer.BadParameter(str(exc), param_hint='--year') from exc
        try:
            model_url = urlsplit(model_zip_uri)
        except ValueError as exc:
            raise typer.BadParameter('Expected a public HTTPS URL', param_hint='--model-zip-uri') from exc
        if model_url.scheme != 'https' or not model_url.hostname:
            raise typer.BadParameter('Expected a public HTTPS URL', param_hint='--model-zip-uri')
        zip_name = model_url.path.rsplit('/', 1)[-1]
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9._-]*\.zip', zip_name, re.IGNORECASE):
            raise typer.BadParameter('Expected a versioned ZIP filename', param_hint='--model-zip-uri')
        if (not re.fullmatch(r'[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]', output_bucket)
                or '..' in output_bucket or re.fullmatch(r'(?:[0-9]{1,3}\.){3}[0-9]{1,3}', output_bucket)
                or output_bucket.startswith(('xn--', 'sthree-', 'amzn-s3-demo-'))
                or output_bucket.endswith(('-s3alias', '--ol-s3', '.mrap', '--x-s3', '--table-s3'))):
            raise typer.BadParameter('Expected a general purpose S3 bucket name', param_hint='--output-bucket')
        try:
            collection_url = urlsplit(collection_url_root)
            valid_collection_url = (collection_url.scheme == 'https' and collection_url.hostname
                                    and collection_url.username is None and collection_url.port != 0
                                    and not re.search(r'[\s\\?#]', collection_url_root))
        except ValueError:
            valid_collection_url = False
        if not valid_collection_url:
            raise typer.BadParameter('Expected an absolute HTTPS URL without credentials, query or fragment',
                                     param_hint='--collection-url-root')
        collection_url_root = collection_url_root.rstrip('/')
        if not re.fullmatch(r'\d{3}_\d{3}', tile_id):
            raise typer.BadParameter('Expected an official DEP tile ID such as 075_025', param_hint='--tile-id')
        tile_index = tuple(map(int, tile_id.split('_')))
        tile = PACIFIC_GRID_10.tile_geobox(tile_index)
        plan = dict(tile_id=tile_id, tile=tile, buffered_tile=tile.pad(MODEL_BUFFER))
        # Check which years still need processing before setting up the job.
        client = boto3.Session().client('s3')
        pending = []
        for observation_year in years:
            itempath = S3ItemPath(bucket=output_bucket, sensor='s2', dataset_id=DATASET_ID, version=version,
                                  time=str(observation_year), prefix=OUTPUT_PREFIX, make_hrefs_https=False)
            # Skip years with an existing STAC Item unless --overwrite is set.
            if not overwrite:
                key = itempath.stac_path(tile_index)
                try:
                    client.head_object(Bucket=output_bucket, Key=key)
                except ClientError as exc:
                    status = exc.response.get('ResponseMetadata', {}).get('HTTPStatusCode')
                    code = exc.response.get('Error', {}).get('Code')
                    if status != 404 or code not in ('404', 'NotFound', 'NoSuchKey'):
                        raise RuntimeError(f'Could not check s3://{output_bucket}/{key}: {exc}') from exc
                else:
                    logger.info('Skipping tile %s for %d because its STAC Item exists', tile_id, observation_year)
                    continue
            pending.append((observation_year, itempath))
        if not pending:
            return
        configure_s3_access(cloud_defaults=True)
        model_uri = stable_model_uri(model_url)
        model_dir, model_sha256 = prepare_model(model_zip_uri, model_uri, zip_name)

        # Prepare WorldCover the first time it's needed, then reuse it.
        worldcover = cache(partial(prepare_worldcover, plan['tile']))
        processor = partial(process_tile, model_dir=model_dir)
        # Use the same catalogue client and loader for all years.
        catalogue = Client.open(DEP_STAC_URL, stac_io=StacApiIO(timeout=(10, 30), max_retries=0))
        loader = OdcLoader(bands=list(GEOMAD_NODATA), groupby='solar_day', resampling='nearest',
                           dtype='uint16', fail_on_error=True)

        # Process one year at a time.
        for observation_year, itempath in pending:
            logger.info('Processing tile %s for %d', tile_id, observation_year)
            # Find this year's GeoMAD images for the tile and the area around it.
            searcher = PystacSearcher(
                client=catalogue, collections=[GEOMAD_COLLECTION], raise_empty_collection_error=False,
                datetime=f'{observation_year:04d}-01-01T00:00:00Z/{observation_year:04d}-12-31T23:59:59Z',
            )
            writer = AwsDsCogWriter(itempath, write_function=write_to_s3, client=client, overwrite=True,
                                    blocksize=512, overview_levels=[2, 4, 8], overview_resampling='nearest',
                                    compress='deflate', num_threads=1)
            stac_creator = StacCreator(itempath, collection_url_root=collection_url_root, with_raster=True,
                                       set_geometry_from_input=True)
            stac_writer = AwsStacWriter(itempath, client=client,
                                        s3_dump_kwargs={} if overwrite else {'IfNoneMatch': '*'})

            # Validate the worker rasters and publish the Item after both COG writes.
            item_uri = run_tile(plan, observation_year,
                                searcher=searcher, loader=loader, worldcover=worldcover,
                                processor=processor, writer=writer, stac_creator=stac_creator, stac_writer=stac_writer,
                                model_uri=model_uri, model_sha256=model_sha256)
            print(item_uri, flush=True)
            logger.info('Delivered tile %s for %d', tile_id, observation_year)
    except (APIError, BadZipFile, BotoCoreError, ClientError, OSError, KeyError, TypeError, ValueError, RuntimeError) as exc:
        typer.echo(f'{type(exc).__name__}: {exc}', err=True)
        raise typer.Exit(code=1) from exc


if __name__ == '__main__':
    typer.run(main)
