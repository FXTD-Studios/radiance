<!-- project-mapper:generated -->
# Tests, CI, and release

ID: validation. Snapshot: `7376f9e`. Coverage: structure inspected. No tests were executed during mapping.

## Implementation index

| Source | Role |
| --- | --- |
| `tests/conftest.py` | Sets `RADIANCE_ALLOW_DOWNLOADS=0`. A meta-path finder maps `radiance` to the repo root. If torch is missing it installs a stub. Stubs `comfy.*`, `folder_paths`, `server.PromptServer` (recording `_FakeRoutes`), and `comfy_api.latest`. Gates tests marked `real_torch` |
| `tests/_sampler_harness.py` | Fake `comfy.sample` and `comfy.samplers` so `RadianceSamplerPro.sample()` can be driven on CPU |
| `tests/node_keys_snapshot.json` | Snapshot of the registered keys |
| `pyproject.toml [tool.pytest]` | `testpaths=tests`, `timeout=30`, `-m 'not slow'`, markers `slow`/`gpu`/`integration`. Coverage floor 54 (branch) |
| `.github/workflows/ci.yml` | Jobs: `test` (light deps, Python 3.10–3.13 × ubuntu/windows on a public repo), `test-full` (3.11 with torch CPU, OpenEXR, OIIO, OCIO, ffmpeg, coverage, then `-m slow`), `smoke`, `js-test` (Node 22), `gpu-test` (Playwright plus Deno WebGPU), `lint-config`, `build-install` (wheel contents), `lint` (ruff E9/F82/F63/F7), `security` (advisory) |
| `.github/workflows/publish.yml` | On a `vX.Y.Z` tag: gate tests, then `Comfy-Org/publish-node-action`, then a GitHub Release built from the CHANGELOG block |
| `tools/check_release_ready.py` | Release metadata and hygiene checks. Run only through `tests/test_release_ready.py` |
| `tools/build_node_reference.py` | Regenerates `docs/nodes/*.md` (`RADIANCE_UPDATE_DOCS=1 pytest tests/test_node_reference.py`) |
| `tools/check_mojibake.py` | Double-encoded UTF-8 scan. **Not run by CI** |
| `tools/gpu_acceptance.py` | Manual on-GPU sweep that writes `gpu_acceptance_report.md` |
| `tools/pin_models.py` | Checks model pins against Hugging Face and GitHub. Needs network |
| `tests/test_version_sync.py` | Checks that the version (3.5.3) matches across pyproject, constants, package.json, CHANGELOG, and the README badge |

## Test inventory (by filename, approximate)

168 `test_*.py` files:

| Area | Files |
| --- | --- |
| Colour | 21 |
| HDR/VAE | 25 |
| IO | 29 |
| Generate | 28 |
| Registry/packaging | 18 |
| VFX/upscale | 15 |
| Delivery/DCC | 10 |
| Viewer | 6 |
| Workspace/security | 10 |
| Audit catch-alls | 6 |

- **GPU:** only `test_production_cuda_parity.py`.
- **Slow:** only `test_video_frame_counts.py`.
- **`integration` marker:** declared but unused.

## Commands

See [START_HERE](../START_HERE.md#validation-from-ci-and-contributing-none-of-these-were-run-during-mapping).
Also: `python -m pytest tests/ -m slow --tb=short -q`, `python -m build --wheel`,
`python tools/check_release_ready.py`, `python tools/check_mojibake.py`.

## Drift between CONTRIBUTING and CI

Tracked once, in [OPEN_QUESTIONS.md](../OPEN_QUESTIONS.md#documentation-drift) (items D2 to D5, D12).
