# Setup

Complete setup before following the [run instructions](../README.md#run).
The launcher prepares imagery and jobs automatically; it does not download the
processor image, trained model or external inference source.

## Prerequisites

- Linux, Python 3.10+ standard library, Bash and GNU coreutils, including `timeout`
  and `sha256sum`. A host GIS or machine-learning environment is not required.
- Access to a running Docker daemon with the NVIDIA runtime named `nvidia` and
  `--gpus all` support, plus compatible host GPU drivers.
- Access to the required GeoMAD and land-cover assets. Supply any required site
  credentials outside source files and logs.
- Sufficient CPU, RAM, GPU memory and storage for the selected processing plans.
  Preparation and assembly use CPU; pending processing jobs request GPU.

Image distribution, dependency storage/access and any required redistribution
permissions must be arranged with the supplier. Registry addresses, staging
locations and credentials are not prescribed by this package.

## Source

Use either a Git checkout or an issued ZIP, then continue from the source root.
The paths below are examples; substitute the locations for your environment.

### Git checkout

Use the approved source branch or revision. Run the remaining setup commands from
the repository root. Git checkouts do not contain the issued ZIP's export files,
so skip ZIP extraction and checksum verification.

### Issued ZIP

Check the archive against its supplied delivery manifest, then extract it and
verify its included files:

```bash
python3 -m zipfile -e /supplied/source/dep-agc-0.1.0-review.5-source.zip /work/processor
cd /work/processor/dep-agc-0.1.0-review.5
sha256sum --check SHA256SUMS
```

Keep the source unchanged while a run is active and when relaunching that run.

## Processor image

Obtain the image specified by [release.json](../release.json). If supplied
as a Docker-compatible archive, load it once:

```bash
docker load --input /supplied/images/dep-agc.tar
```

Inspect the locally supplied reference using the release definition:

```bash
processor_image=$(python3 -c 'import json; print(json.load(open("release.json"))["image_reference"])')
docker image inspect "$processor_image" --format '{{.Id}}'
```

`image_reference` and `image_id` in release.json are authoritative. Launch checks
that the local reference resolves to the required immutable identity before
processing. It neither pulls nor builds an image and rejects a different identity.
The image contains separate pinned inference and GIS/DEP software environments.

## Supporting dependencies

Supply these directories at stable, readable locations:

| Supply | Required contents |
|---|---|
| Model directory | Original weights and training configuration named by `inference.model` in the [scientific profile](../configs/scientific-profile.json). |
| Raw inference-source directory | Original source files listed by `inference.source_sha256` in the scientific profile, preserving their bytes and line endings. |
| Approved spatial metadata | A supplied country index with its referenced plans, approval certificates, ownership metadata and any required projected geometry. The included [country catalogue](../configs/countries.json) defines approved identities; it is not the external payload bundle. |
| Boundary directory | Approved `gadm41_CODE.gpkg` files for the selected codes, matching the supplied plans and certificates. |

Model/source payloads may remain outside the dependency directory. Their pins are
maintained in the profile and checked automatically; no checksum list needs to be
entered in the run configuration. Spatial definitions are reusable across
observation years where boundaries and grids are unchanged. Previous imagery,
jobs, checkpoints, height/carbon outputs and manually repaired geometries are not
required inputs.

### Bind dependencies

From the extracted source directory, run the included helper after supplied files
have their intended local locations. This labelled example matches the selection
in run.json; substitute codes from the catalogue as needed:

```bash
python3 bind-dependencies.py --destination dependencies \
  --catalog /supplied/spatial/countries.json --countries NIU \
  --boundary-directory /supplied/spatial/boundaries \
  --model /supplied/model --source /supplied/inference-source \
  --metadata-root /supplied/spatial/metadata
```

`--countries` also accepts several codes or `ALL`. Each selected country needs its
approved payloads. `--metadata-root` is required when approval certificates refer
to projected geometry; preserve their supplied relative locator layout beneath
that root. The helper validates containment, file identities, declared CRS and
ownership metadata, copies small approved metadata into `dependencies/CODE/` and
writes `dependency-manifest.json` with model/source references. It emits canonical
projected-geometry bindings where required, without repair or guessed paths.

Bind once per supply location, not per year or processing window. An existing
binding is preserved. For a new supply location, use the helper and a fresh
destination; never hand-edit generated manifests. Keep existing run bindings
unchanged for [relaunch](reference.md#relaunch-behaviour).

## Permissions and storage

The launching account needs Docker access and write access to the configured
working and output roots. Supplied files must be readable by container UID/GID
65534; container-bound temporary/result directories must allow its writes.
Arrange permissions for provisioned locations before launch. New working
subdirectories are created by the launcher with the required container access.

Use stable Linux filesystem storage supporting file locks, atomic rename and
sync. Keep supplied dependencies, working data and final outputs separate; neither
work nor output may contain the other or overlap dependencies. Avoid commas and
newlines in mount paths. Check physical backing-volume free space as well as
filesystem capacity when using virtual disks.

Downloaded imagery, static inputs and intermediates belong in `working_directory`;
products belong in `output_directory`. Internal folders are generated automatically.
Resource defaults come from release.json; optional site limits belong in
`dependency_directory/environment.json` as described in the
[reference](reference.md#resource-settings). Admit resources before the first run
and retain the settings for resume. External test observers and independent
acceptance validators are not runtime prerequisites.

After setup, edit the six-field [run configuration](../run.json) and follow
[Run](../README.md#run).
