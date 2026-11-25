# Reference

## Pacific tile windows

`pacific_tile_plan(tile_id)` in `spc_agb_inputs.geometry` plans the full
9,600 × 9,600 `PACIFIC_GRID_10` tile in EPSG:3832. The 25 retained
1,920 × 1,920 cores cover it once. Each receives 2,368 × 2,368 source pixels,
including 224 pixels of context on every side. The 512-pixel model patch and
64-pixel stride stay fixed. Whole-tile boundaries do not clip the imagery context.

`assemble_tile` in `spc_agb_dep.assemble_tile` takes finished cores, the
accounting footprint and their validated source Items. It writes one local
tile-sized COG set and STAC Item, with NoData outside the footprint. Legacy
country grids and saved tasks retain their existing dimensions.

## Configuration

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
[scientific definitions](../spc_agb_worker/science.py); stock is the native
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

Individual image stages remain available for diagnosis: preparation, catalogue
binding/listing, window processing and country assembly. For example, inspect
preparation arguments without processing:

```bash
processor_image=$(python3 -c 'import json; print(json.load(open("release.json"))["image_reference"])')
docker run --rm --pull never --network none --read-only --user 65534:65534 \
  --entrypoint /opt/adapter-venv/bin/python "$processor_image" \
  -B -m spc_agb_inputs.prepare_country --help
```

Other modules are `spc_agb_dep.catalog`, `spc_agb_dep.print_tasks`,
`spc_agb_dep.run_task` and `spc_agb_dep.assemble_country`. Use their help and the
recorded stage arguments/mounts for diagnosis, preserving partial results.
Individual stages are not additional normal setup steps.
