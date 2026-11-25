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

## Method and limitations

Height is inferred using the supplied model and fixed preprocessing/profile.
The accepted height-to-carbon relationship is implemented in the
[scientific definitions](../carbon/science.py). The [profile](../configs/scientific-profile.json)
defines eligibility, class rules and low-NDVI treatment.

Genuine modelled zero and low-NDVI assumed zero are valid classified estimates
with distinct statuses. Unavailable imagery, unknown land-cover classification
and excluded water remain explicit and do not become zero carbon. Missing coverage
remains NoData.

Estimates cover living aboveground vegetation carbon. Independent accuracy
assessments are limited to the areas evaluated.
