<!-- project-mapper:generated -->
# Colour and HDR library layer

ID: color (plus the non-VAE parts of hdr). Snapshot: `7376f9e`. Coverage: partial (static). Labels: see [START_HERE](../START_HERE.md#evidence-labels).

## Responsibility and boundaries

This layer holds pure maths and OCIO plumbing used by the nodes: transfer curves, gamut
matrices, file encodings, LUT/CDL, tonemapping, and ACES 2.0. It imports nothing from the
node, io, or delivery layers.

## Implementation index

| Source/symbol | Role | Label |
| --- | --- | --- |
| `color/ops.py` | The stated "single source of truth" for torch maths: `apply_matrix_3x3`, PQ/HLG constants, BT.2408/BT.2100, `soft_knee_compress` | Implemented |
| `color/transfer.py` | About 58 numpy/torch curves: sRGB, 709/2020, LogC3/C4, S-Log3, V-Log, CLog3, Log3G10, ACEScct/cc, DaVinci Intermediate, PQ, HLG | Implemented |
| `color/matrices.py` | Camera gamut to ACEScg, and 2020/P3 matrices | Implemented |
| `color/encodings.py:ENCODINGS`, `WORKING_SPACES` | File encode/decode table. Uses OCIO when available, otherwise analytic curves plus a CAT matrix. PQ/HLG reference white is 203 nits | Implemented |
| `color/pipeline.py:apply_input_transform` | Input transform. For log spaces and ACEScct it applies the curve only, with no gamut matrix | Implemented (trace), see OPEN_QUESTIONS |
| `color/lut.py`, `color/luts.py`, `color/grading.py` | LUT Apply/Blend nodes, analytic looks and IDTs, numpy grade, CDL export | Implemented |
| `color/ocio_setup.py:configure_ocio` | Startup config choice: valid `$OCIO`, else the built-in studio config (written to `ACES/studio-config.ocio`), else the bundled `ACES/config.ocio` | Implemented |
| `radiance_ocio.py` | OCIO manager singleton, LUT bake for the viewer, `/radiance/ocio/*` routes, CPU fallback `apply_ocio_transform` | Implemented |
| `hdr/ocio.py` | OCIO Transform, ACES Config Manager, List Colorspaces nodes. Has its own `_resolve_config` | Implemented |
| `hdr/aces2_ocio.py` | ACES 2.0 output transform, pinned to `ocio://studio-config-v4.0.0_aces-v2.0_ocio-v2.5`. Raises `ACES2ReferenceUnavailable` | Implemented |
| `nodes/hdr/aces2.py` | ACES 2.0 nodes. Falls back to an analytic implementation when OCIO is unavailable | Implemented |
| `hdr/tonemap.py:HDRToneMap`, `HDRExpandDynamicRange` | Operators: ACES filmic, Uncharted2, AgX, Reinhard variants, linear clamp, exposure only | Implemented |
| `hdr/tonescale.py` | ACES 2.0 mid-grey and tonescale reference functions | Implemented |
| `core/tensor/alpha.py` | `split_rgb_alpha`, `merge_rgb_alpha`, `@alpha_passthrough`. Raises if C is greater than 4 | Implemented |
| `core/tensor/contract.py` | Latent `ensure_4d`/`ensure_5d`. Rejects NestedTensor | Implemented |
| `core/tensor/chunking.py` | Per-chunk memory budget: 1 GiB on CPU or 35% of free VRAM | Implemented |

## Contracts

- **IMAGE:** float32 `(B,H,W,C)`, C in {1,3,4}. Video may arrive as 5D at the VAE.
  Alpha is straight. MASK is `(B,H,W)`.
- **Range:** no global declaration. The working space default is "Linear Rec.709 (sRGB)",
  scene-linear and unbounded. The Viewer's Auto mode treats any value outside [0,1] as linear,
  decided per batch (`color/viewer_space.py`).
- **OCIO is optional everywhere** (`HAS_OCIO` guards). Exceptions: the `OCIOColorTransform`
  node raises without it, and `aces2_ocio` needs OCIO 2.5 or newer.
- **Each consumer resolves its OCIO config differently.** There are five resolvers:
  `ocio_setup`, `radiance_ocio._OCIO_SEARCH_PATHS`, `hdr/ocio._resolve_config`, `hdr/aces2_ocio`
  (pinned), and `encodings.ocio_config`. A change to config handling must check all five.

## Tests

About 21 colour, ACES, OCIO, CDL, and LUT test files (`tests/test_aces2*`, `test_cdl*`,
`test_color_*`, `test_ocio_*`, `test_lut*`). JS colour parity tests are in
`js/tests/viewer_color.test.mjs` and `ocio.test.mjs`. Inspected by name only.

## Open questions

See OPEN_QUESTIONS: the bundled config's version against the fallback, the ACESConfigManager
download, duplicated curves, and the input transform that applies no gamut matrix.
