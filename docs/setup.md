# Setup

## Tile workflow

Use Python 3.12 on Linux. Install the pinned GIS dependencies in a virtual
environment from the source root.

```bash
python3.12 -m venv .venv-gis
. .venv-gis/bin/activate
python -m pip install -r requirements/dep-window.txt
```

For new predictions, use a separate CUDA environment with the pinned
[inference dependencies](../containers/window-worker-requirements.txt), and set
`SPC_WORKER_PYTHON` to its Python executable. The worker checks the model and
inference source hashes from the [scientific profile](../configs/scientific-profile.json).
The GIS environment is sufficient for saved-height replay.

The [tile command](../README.md#run) needs HTTPS access to the DEP STAC catalogue,
the GeoMAD COGs and ESA WorldCover. Allow disk space for prepared windows and
the final COGs. An [explicit local input bundle](reference.md#tile-input-bundle)
is also supported.
