# DEP Aboveground Carbon

Estimate vegetation height and aboveground carbon density at 10 m from annual
DEP GeoMAD imagery and ESA WorldCover. Produce one dataset per official Pacific
tile.

## Setup and inputs

Use the pinned GIS environment and the pinned CUDA inference runtime. Supply
the model and inference source from the scientific profile. The tile command
discovers and reads annual DEP GeoMAD v0.4.0 and ESA WorldCover during the run.
See the [setup guide](docs/setup.md).

## Run

```bash
python -m tasks tile 075_025 2024 \
  --model model --source inference-source --work runs/075_025-2024 \
  --output products/075_025
```

The [tile runner](tasks/tile.py) discovers annual GeoMAD, aligns WorldCover,
loads each window with neighbouring imagery where needed, and dispatches the
carbon worker. It assembles seven COGs, one STAC Item with annual GeoMAD
lineage, and a verification receipt. See the
[tile reference](docs/reference.md#pacific-tile-products).

Estimates cover living aboveground vegetation carbon. See the
[method and limitations](docs/reference.md#method-and-limitations).
