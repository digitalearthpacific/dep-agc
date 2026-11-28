# DEP Aboveground Carbon

## Overview and method

DEP AGC estimates living aboveground vegetation carbon density at 10 m
resolution, in tonnes of carbon per hectare. It uses annual GeoMAD imagery to
predict vegetation height and calculate NDVI, and ESA WorldCover to identify
eligible land. AGC uses predicted height as a proxy for top-of-canopy height and
converts it to carbon density using an equation calibrated on Hawaiʻi Island.
[Asner et al. (2016)](https://doi.org/10.1186/s13021-015-0043-4)

## Run a tile

From the repository root, run Samoa tile `075_025` for 2024 in the AGC Python
environment. In the container, pass the same arguments to the image entry point.
You need a GPU and credentials that can check for existing Items and write to
your output bucket.

Replace `your-dep-bucket` in both arguments with the DEP S3 bucket that hosts the
publicly readable model ZIP and receives the carbon and status COGs and STAC Item
JSON. Set the collections URL to your output catalogue. The ZIP for model
`24123113h21` must contain `Delta1_model_traced.pth` and `training_config.cfg`.

```sh
python -m src.run_task \
  --tile-id 075_025 --year 2024 \
  --model-zip-uri https://your-dep-bucket.s3.us-west-2.amazonaws.com/aboveground-carbon/models/dep-agc-model-24123113h21.zip \
  --output-bucket your-dep-bucket \
  --version 0.1.0-review.5 \
  --collection-url-root https://stac.example.org/collections
```

Use `--year 2024-2025` to process both years in order. `--version` selects the
output path in S3. A year with an existing STAC Item is skipped unless you pass
`--overwrite`.

The runner caches the ZIP, extracted model files, and ZIP checksum under
`models/<ZIP filename>/<stable URL hash>/` in its working directory. Use a new
ZIP filename for each model revision. The cache uses the same stable URL as the
STAC Item, so refreshing an S3 signature reuses that entry. Reuse across Argo pods
requires a shared `models/` directory with file locking and atomic renames, or
a complete cache entry prepopulated in each pod's `models/`.

## Outputs

Each processed year produces two Cloud Optimized GeoTIFFs, `carbon` and `status`, beside
the year-qualified STAC Item. The command prints each published Item's S3 URI.
The Item records the model ZIP's stable HTTPS URL in `agc:model_uri` and the
archive checksum in `agc:model_sha256`. The URL excludes signing credentials
and fragments and preserves an S3 `versionId` if supplied.

The `carbon` raster reports tonnes of carbon per hectare. The `status` raster
records these codes where WorldCover is available.

| Code | Meaning |
| --- | --- |
| 0 | Excluded water |
| 1 | Unavailable inputs |
| 4 | Modelled carbon, including valid zero |
| 5 | Unknown land cover |
| 7 | Zero assigned by the low-NDVI rule |

Modelled zeros and low-NDVI assumed zeros are valid carbon estimates, distinct
from NoData. Pixels outside WorldCover coverage are NoData in both rasters.
The status raster reserves `255` for NoData.

## Build and release

Build locally from the repository root with `docker build -t dep-agc:local .`.
CI runs offline smoke checks and GIS tests with network access for STAC schema
validation on pull requests targeting `main` and pushes to `main`. When a GitHub
release is created, CI builds and checks the image once, then pushes it to GHCR
under the release tag and `latest`.
