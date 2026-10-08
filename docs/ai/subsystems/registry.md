<!-- project-mapper:generated -->
# Node registry and catalog

ID: registry. Snapshot: `7376f9e`. Coverage: inspected (static). Labels: see [START_HERE](../START_HERE.md#evidence-labels).

## Responsibility and boundaries

This component owns how node classes reach ComfyUI. That covers the import chain,
merging, failure reporting, menu branding, and dynamic gizmos. It does not own what
the nodes do.

## Implementation index

| Source/symbol | Role | Label |
| --- | --- | --- |
| `__init__.py:_bootstrap_package_context` | Aliases `sys.modules["radiance"]` so absolute `radiance.*` imports work whatever the folder is called | Implemented |
| `__init__.py:_load_comfyui_nodes` | Loads `.nodes` (`required=True`) and folds in `nodes.NODE_LOAD_FAILURES` | Implemented |
| `__init__.py:report_node_load_health` | Logs ERROR if any import failed or if the count is below `EXPECTED_MIN_NODE_COUNT` | Implemented |
| `nodes/catalog.py:NODE_GROUPS` | Ten groups: color, hdr, io, vfx, pipeline, monitor, upscale, video, ai, generate. None has an env flag | Implemented |
| `nodes/registry.py:load_node_mappings` | Imports each spec and merges with plain `dict.update`, so a later module overrides an earlier one (logged only at DEBUG) | Implemented |
| `nodes/aggregate.py:fold_in_module_nodes` | Picks up `NODE_CLASS_MAPPINGS` from leaf modules one level deep. Keys the group already lists win | Implemented |
| `nodes/branding.py:apply_radiance_branding` | Uses `NODE_SECTIONS` (152 keys) to set the menu section and display name, and **overwrites each class's `CATEGORY`** | Implemented |
| `nodes/gizmo.py:load_dynamic_gizmos` | Builds `RadianceGizmo_<name>` classes from `<repo>/gizmos/*.gizmo` JSON. `os.makedirs` runs at import time | Implemented |

## Contracts

- **Registration pattern:** each `nodes/<group>/__init__.py` explicitly imports classes and
  builds a dict, then calls `fold_in_module_nodes(__name__, ...)`. Implementation packages
  (`radiance.hdr`, `.color`, `.image`, `.film`) are deliberately **not** groups
  (`catalog.py:27-45`), because loading them as groups could silently replace keys.
- **Per-group key counts (152 total since 4.0, which removed SAM Loader, SAM Mask Generator, HDR Latent Encoder and HDR Turbo Encoder):**

  | Group | Keys |
  | --- | --- |
  | color | 17 |
  | hdr | 39 |
  | io | 6 |
  | vfx | 26 |
  | pipeline | 10 |
  | monitor | 9 |
  | upscale | 10 |
  | video | 13 |
  | ai | 3 |
  | generate | 19 |

  The counts are from a static reading of each `__init__`. They match `NODE_SECTIONS`.
- **Custom socket types** are bare strings; there is no registry. They are `RADIANCE_PASSES`,
  `RADIANCE_SHOT`, `RADIANCE_OCIO`, `RADIANCE_CAMERA`, `STITCHER_DATA`,
  `LORA_STACK`, `LORA_DICT`, `LATENT_UPSCALE_MODEL`, and `BOUNDING_BOX`. HDR travels as plain `IMAGE`.
- **One V3 node:** `CinematicPromptEncoder` (`nodes/generate/prompt.py`) uses
  `comfy_api.latest` and is branded through `schema_branding`.

## Important paths

- **Failure granularity:** group `__init__` imports are not wrapped in try/except, so one
  ImportError drops the whole group. Fold-in leaf modules and multipass sub-modules fail
  one at a time instead. Hard dependencies that can drop a group:
  - `cv2` through `hdr/panorama.py` affects hdr and generate.
  - `defusedxml` and `folder_paths` affect color.
  - `folder_paths` and `node_helpers` affect io.
  - `comfy_extras.*` and `comfy_api.latest` affect generate. An older ComfyUI is inferred to drop this group.
- **Branding runs more than once:** on every `load_node_mappings` call and once more in `nodes/__init__.py`.

## Tests

`tests/test_nodes_registry.py`, `test_node_*` (8 files), `test_node_publication_completeness.py`,
`test_implementation_package_nodes.py`, `test_menu_sections.py`, and `test_package_cleanup.py`
(reads `EXPECTED_MIN_NODE_COUNT`), plus `tests/node_keys_snapshot.json`. These were inspected
by filename only and not run.

## Open questions

- README says "147 visible nodes" but 152 keys register (156 before 4.0). Possible reasons: deprecated aliases
  (`ImageLoader`, `ControlApply`), gizmos, or hidden nodes. Not verified.
- When a gizmo key is missing from `NODE_SECTIONS`, the keyword fallback replaces its own
  category (inferred).
