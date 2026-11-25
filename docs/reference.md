# Reference

## Pacific tile products

`pacific_tile_plan(tile_id)` in `tasks.inputs.geometry` plans the full
9,600 × 9,600 `PACIFIC_GRID_10` tile in EPSG:3832. The 25 retained
1,920 × 1,920 cores cover it once. Each receives 2,368 × 2,368 source pixels,
including 224 pixels of context on every side. The 512-pixel model patch and
64-pixel stride stay fixed. Whole-tile boundaries do not clip the imagery context.

`python -m tasks tile TILE_ID YEAR --model model --source inference-source
--output tile` searches the DEP catalogue for the requested annual GeoMAD v0.4.0
Items, including neighbours that cover the model context. It reads the required
COG windows, aligns ESA WorldCover v200 to the retained tile, runs the pinned
worker and assembles one tile dataset. The tile contains the seven COG
assets `carbon`, `height`, `NDVI`, `count`, `WorldCover`, `status` and `RGB`, plus
`TILE_ID.stac-item.json` and `tile-verification.json`. Unprocessed pixels are
NoData. The status raster distinguishes assessed zero, water, unknown land cover
and unavailable imagery.

Work files go in `--work` or the sibling directory `OUTPUT-work`. Both output and
work need fresh destinations. The work directory holds the discovered Items,
aligned WorldCover, SHA-256-pinned window inputs, prepared RGB,
raw heights and `replay.json`. Use that manifest as `--inputs` to
reconstruct the tile from saved heights. Replay takes no model or source arguments.

### Tile input bundle

To supply prepared inputs explicitly, pass `--inputs inputs.json`. The bundle
contains local, SHA-256-pinned inputs. Each file reference
has `path` (relative to the bundle JSON) and `sha256` (hex digest).

| Field | Content |
|---|---|
| `schema` | `agc-tile-inputs-v1` |
| `tile`, `year` | Official `PACIFIC_GRID_10` ID and integer calendar year |
| `geomad` | Array of objects with an `item` file reference to a GeoMAD v0.4.0 STAC Item and an `assets` object mapping `red`, `green`, `blue`, `nir` and `count` to local GeoTIFF file references |
| `worldcover` | File reference to a one-band `uint8` WorldCover GeoTIFF aligned to the full official tile in EPSG:3832. Use an internal validity mask and no NoData value so class 0 remains distinguishable from missing bytes. |

The GeoMAD roster must cover the tile plus its 224 pixels of context, including
required neighbouring Items. Keep source validity masks in the local GeoTIFFs.
The command reads these local files and validates their annual identity.

### Country reports

`python -m tasks report TILE_ITEM COUNTRIES_GEOJSON --output country-report.json`
takes the tile's STAC Item and a GeoJSON FeatureCollection. Each feature needs a
unique string `id` or `properties.code`. Polygons default to EPSG:4326. Use
`--polygon-crs EPSG:3832` for native tile coordinates and `--footprint` to apply
a reporting footprint in the same CRS. Footprint-limited reports are labelled
`reporting-footprint totals`.

Reporting clips each pixel to the country and optional footprint. The coverage
denominator is the full WGS84 ground area of the country within the reporting
footprint and tile, including water, unknown classes, unavailable imagery and
unprocessed pixels. Valid zero contributes assessed area and zero stock.

### Accounting layers and report fields

| Former COG | Where the information lives |
|---|---|
| `physical-area-ha` | Country reporting computes WGS84 ground-area weights in memory for each pixel's intersection with the country and optional reporting footprint. The report records aggregate hectares as `footprint_ha`, `assessed_ha` and the area by status, along with `stock_tC`. Per-pixel weights are transient. |
| `owner` | Legacy rectangular ownership IDs remain in legacy country outputs and saved baselines. For new reports, the input polygons define country membership and their IDs key the `countries` results. A pixel split between countries contributes its respective intersection area to each. The tile's `status` mask identifies processed pixels. |

## Legacy country configuration

[run.json](../run.json) is the normal six-field configuration. Relative paths
resolve against that file. The supplied country/year example is editable; it does
not restrict selection to that country or year.

| Field | Meaning |
|---|---|
| `countries` | One country code, several codes, or `["ALL"]` alone. Use the maintained [country catalogue](../configs/countries.json) and listing command for names/codes. Empty, duplicate, unknown and mixed-ALL selections are rejected. |
| `year` | Integer observation year. Availability and exact requested-year source identity are checked during preparation. |
| `geomad_version` | Supported annual GeoMAD version; see supported inputs below. |
| `dependency_directory` | Supplied/bound spatial metadata and verification records, with references to model/source directories that may reside elsewhere. |
| `working_directory` | Downloaded imagery, generated inputs, intermediate results, checkpoints and logs. |
| `output_directory` | Durable country products, separate from working data and supplied dependencies. |

List configured countries without dispatch:

```bash
python3 launch-country.py --list-countries
python3 launch-country.py --list-countries --config run.json
```

With a configuration, listing also reports supplied dependency bindings. Catalogue
execution labels are static reference metadata, not a live deployment check.
All selected countries' prerequisites are checked before dispatch; countries then
run serially with separate country/year state. `ALL` requires complete approved
supplies for every catalogue entry. No separate country table is maintained here.

### Supported inputs

Annual **GeoMAD v0.4.0** is supported. The requested year must match retrieved
annual metadata/assets; another year/version is never substituted. Selectable
years do not establish provider availability. Real-data execution has exercised
2024; other requested years require available compatible source data.

WorldCover remains **2021 v200**. Approved spatial plans provide grids, windows,
context and ownership definitions. Scientific settings, accepted model identity,
raw inference-source pins and eligibility policy are maintained in the
[scientific profile](../configs/scientific-profile.json); image identity and
resource defaults are maintained in [release.json](../release.json).
These definitions are internal inputs, not extra per-run configuration files.

### Resource settings

Optional site settings in `dependency_directory/environment.json` may set:

| Key | Unit or constraint |
|---|---|
| `minutes` | Positive execution allowance in minutes. |
| `model_attempts` | Nonnegative attempt cap; automatic inference retries are not enabled. |
| `source_transfer_gib` | Source-transfer allowance in GiB. |
| `minimum_free_gib` | Positive free-space reserve in GiB. |
| `free_space_directory` | Existing filesystem location used for the host free-space check. |
| `allow_acquisition` | Boolean; false declares an offline run with zero source-transfer allowance. |
| `image_reference` | Locally supplied image reference; its resolved ID must still match release.json. |

Otherwise release defaults derive time/transfer allowances from plan size and the
model cap from verified pending jobs. Defaults are not a regional capacity estimate.
Set site overrides before first launch; preserve them when resuming. These optional
settings do not add required fields to the normal configuration.

## Outputs

Each `working_directory/CODE/YEAR/` contains generated `configuration.json`,
`incoming/`, `intermediate/`, `logs/` and `assembly/`. Each
`output_directory/CODE/YEAR/` contains:

- `collection.json`: country STAC collection and links to window Items/assets.
- `country-summary.json`: completeness, country/per-owner accounting and verification.
- `product-manifest.json`: durable file inventory and identities.
- Window COG assets and associated metadata/provenance records.

| COG layer | Units / interpretation |
|---|---|
| `RGB` | Encoded RGB display values. |
| `height` | Vegetation height, m. |
| `carbon` | Carbon density, t C/ha. |
| `NDVI` | Dimensionless vegetation index. |
| `count` | Observation count. |
| `WorldCover` | Land-cover class. |
| `status` | Accounting/support category; descriptions accompany STAC metadata. |
| `owner` | Ownership identifier, interpreted with supplied owner mapping. |
| `physical-area-ha` | Physical pixel area, ha. |

Native-resolution values and masks are authoritative. Display overviews do not
drive accounting. Durable local assets/lineage references resolve within the final
product, allowing relocation independently of working inputs. Source-provider URLs
remain provenance; reading final products does not require refetching imagery.

Accounting uses physical hectares: `owned_ha` is the selected boundary area,
including water; `assessed_ha` is classified support; `stock_tC` is included carbon
stock in t C. Coverage is classified support divided by selected area. Modelled,
assumed-zero, unavailable, unknown and water areas retain separate breakdowns.
An incomplete country is not advertised as a complete country product.

## Method and limitations

Height is inferred using the supplied model and fixed preprocessing/profile.
The accepted height-to-carbon relationship is implemented in the
[scientific definitions](../carbon/science.py); stock is the native
carbon density multiplied by supplied physical-area weights and reduced over
classified support. The [profile](../configs/scientific-profile.json)
defines eligibility, class rules and low-NDVI treatment.

Genuine modelled zero and low-NDVI assumed zero are valid classified estimates
with distinct statuses. Unavailable imagery, unknown land-cover classification
and excluded water remain explicit and do not become zero carbon. Missing coverage
is excluded without gap filling or stock extrapolation.

Estimates cover living aboveground vegetation only; roots, soil and dead organic
matter are excluded. Results are indicative: model accuracy has not been
independently established for each area, and a 10 m grid does not establish
accuracy. Annual composites do not provide daily acquisition dates. Local
compatibility checks do not establish execution suitability for another deployment.
Data/model distribution rights must be arranged; GADM permission remains unverified.

## Relaunch behaviour

Repeat `python3 launch-country.py --config run.json` with unchanged source,
configuration, dependency bindings and locations. Prepared inputs and completed
jobs are verified before reuse. Assemblies and durable products are reused only
when their saved verification identities match; unverified existing destinations
are preserved rather than overwritten.

Deadlines and accumulated model/transfer expenditure persist. Failures stop the
batch, return nonzero and retain logs and partial outputs; inference is not retried
automatically. Cross-version checkpoint migration and automatic writer-only recovery
are not provided. Do not edit receipts, overwrite failed results or create new
roots to evade exhausted limits. Diagnosing a failed run is separate from resuming
verified successful work.

## Troubleshooting

Read `working_directory/CODE/YEAR/logs/launcher-state.json` and its numbered stdout/
stderr files for the failed stage, exact commands and paths.

| Problem | Action |
|---|---|
| Missing or changed image | Supply the release-defined image and inspect its local ID as described in [setup](setup.md#processor-image). Launch does not pull/build. |
| Missing/invalid dependency | Check the reported country/file and supplied approval records; use the [binding helper](setup.md#bind-dependencies) for a fresh supply location. Do not repair hashes, geometry or jobs manually. |
| Requested data unavailable/incompatible | Check requested year/version and source error. Missing data is handled explicitly; no automatic substitution is made. |
| Resource/permission failure | Check admitted limits, physical free space, Docker access and container UID permissions. Preserve original expenditure and failed evidence. |
| Failed/incomplete processing | Preserve logs/results and diagnose the reported cause. No automatic retry or old-output fallback is available. |
| Existing incompatible state/product | Preserve it. Configuration migration and unverified-destination overwrite are not automatic. |

Individual country stages remain available for diagnosis. The archived reference
image uses its original module names. To inspect the corresponding source module
in this checkout, run:

```bash
python -m tasks.inputs.prepare_country --help
```

Other modules are `tasks.catalog`, `tasks.print_tasks`,
`tasks.run_task` and `tasks.assemble_country`. Use their help and the
recorded stage arguments/mounts for diagnosis, preserving partial results.
Individual stages are not additional normal setup steps.
