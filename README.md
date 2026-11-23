# DEP Aboveground Carbon

Generate indicative aboveground vegetation carbon estimates at 10 m resolution from annual DEP GeoMAD imagery.

The processor estimates vegetation height, calculates carbon density and stock, and produces country datasets as Cloud Optimized GeoTIFFs (COGs), STAC metadata and accounting summaries.

## Components

| Component | Purpose |
|---|---|
| Source code | Configuration, preparation, processing and country assembly |
| Docker image | Processing code and its software environment |
| Supporting dependencies | Trained model, inference source, country boundaries and processing plans |
| Input imagery | GeoMAD and supporting land-cover data, retrieved automatically |
| Outputs | COG datasets, STAC metadata, carbon summaries and provenance |

The dashboard and map-serving infrastructure are maintained separately.

## Workflow

A single launch runs three stages:

1. **Prepare:** discover imagery and prepare processing windows and supporting data.
2. **Process:** estimate height and calculate aboveground carbon.
3. **Assemble:** combine window results into country products and accounting summaries.

## Setup

Requires Linux, Python, Docker with NVIDIA GPU support, access to input data and sufficient working storage.

Follow the [setup instructions](docs/setup.md) to obtain the image and dependencies and configure the environment. Setup is required before the first run.

## Run

Edit [run.json](run.json) to select countries, observation year, GeoMAD version and storage directories. See the [configuration reference](docs/reference.md#configuration) for field definitions and supported inputs.

List available countries and their codes:

```bash
python3 launch-country.py --list-countries
```

Select one country, several countries, or `["ALL"]`. The maintained [country catalogue](configs/countries.json) supplies the codes and processing-plan references.

Launch from the source directory:

```bash
python3 launch-country.py --config run.json
```

Rerun the same command with unchanged configuration to verify and reuse completed work. Failures preserve partial results and stop processing; inference is not retried automatically.

## Results

Products are written beneath the configured output directory, grouped by country and year:

- COG layers, including vegetation height and carbon density.
- STAC metadata describing the datasets.
- Country area, coverage and carbon-stock summaries.
- Processing provenance and file inventories.

See the [output reference](docs/reference.md#outputs) for layers, units and file structure.

## Method and limitations

Estimates represent living aboveground vegetation carbon. Roots, soil and dead material are excluded. Water and unsupported areas are excluded from accounting without extrapolation.

Results are indicative; spatial resolution does not establish accuracy. See the [method and limitations](docs/reference.md#method-and-limitations), [scientific settings](configs/scientific-profile.json) and [troubleshooting guidance](docs/reference.md#troubleshooting).
