# Radiance code review (risk-prioritized)

Snapshot: `124d8a6` on `version-4-Beta` (source identical to `7376f9e`; the only commits since are docs/ai).
Scope: whole repository, read statically. There was no source diff to review, so this pass starts from
`docs/ai/START_HERE.md` and `OPEN_QUESTIONS.md`, re-reads the cited code, and adds new findings.
Nothing was run, imported, or tested. Every location below was re-read at this revision.

Severity: **P0** data loss / RCE / crash on the default path. **P1** a feature is broken or ships a wrong
deliverable without telling anyone. **P2** wrong result on a common path, or a security gap that needs a
non-default setup. **P3** minor, cosmetic, or defence-in-depth.

No P0 was found. The P0 class from 3.5.3 (bridge `exec`, OCIO path oracle, deliver containment) is
closed in the code that was re-read.

## Fix status

A fix pass on `version-4-Beta` after `b645400` fixed the findings below. Each fix has a regression test
that failed on the old code and passes now. The finding text further down describes the code as it was
reviewed and is kept as the record.

| ID | Status | Commit | Regression test |
| --- | --- | --- | --- |
| P1-1 | Fixed. Tensor or dict from `vae.encode`; RGB only; first frame of a video VAE latent, with a warning when a temporal VAE drops reference frames. The test fake now returns a bare tensor, as ComfyUI does | `3314513`, `55afb69` | `tests/test_sdr_conditioning.py::TestEncodeSDRReference` |
| P1-2 | Fixed. `RadianceAIUpscale.used_fallback`; Delivery reports status "partial" with a warning. The fallback is bicubic, not Lanczos as written below | `b242fdf` | `tests/test_delivery_endpoints.py::test_a_silent_bicubic_fallback_is_reported_too` |
| P2-1 | Fixed. Longest pattern first | `1766a81` | `tests/test_detect_by_config.py` |
| P2-2 | Fixed. Stores `extra_pnginfo["workflow"]`; API graphs get stats; old API-format saves open through `app.loadApiJson` (`js/radiance_graph_format.js`) | `16afaf6` | `tests/test_workspace_api.py`, `js/tests/graph_format.test.mjs` |
| P2-3 | Fixed. Boundaries are any non-alphanumeric character | `16afaf6` | `tests/test_workspace_api.py::test_shot_from_name`, `test_version_from_name` |
| P2-4 | **Open.** A restore-on-next-Loader fix (`6b5bd9c`) was reverted (`33a30cb`): with two Loaders in one graph, one on "sequential" and one on "none", both run before any sampler, so the second undid the first before its weights loaded. Nothing Radiance can hook is scoped to one prompt. The tooltip and log now say the effect lasts for the session (`057bab0`) | | |
| P2-5 | Fixed. A taken name becomes `name_1.ext`, created exclusively; an aborted upload removes its partial file | `fb88615` | `tests/test_workspace_api.py::test_uploading_a_taken_name_keeps_both_files` |
| P2-6 | Fixed in 4.0. On a non-loopback bind `queue` must be signed with the shared DCC token (`RADIANCE_DCC_AUTH_TOKEN`, else `~/.radiance/dcc_token`) by `core/dcc_auth.sign_queue`: timestamp within 120 s, single-use nonce, HMAC over the prompt, compared in constant time. The token is never sent (the first fix sent it in clear, and it is also the Nuke signing key). HTTP request lines are refused, so a browser cannot queue through a loopback bridge. Loopback is otherwise unchanged. The uncapped threads remain | `d8acf73`, review fix | `tests/test_bridge_security.py::test_a_remote_bridge_refuses_queue_that_is_not_signed_right`, `test_a_browser_request_to_a_loopback_bridge_is_dropped` |
| P3-1 | Fixed in 4.0. One suffix, `Shot_v0002`: the counter is four digits and passes as `write_frames`' version. 3.x names still count | `9c450af` | `tests/test_delivery_endpoints.py::test_the_delivered_name_carries_one_version_suffix` |
| P3-2 | Fixed. Containment before existence | `c4492a8` | `tests/test_read_surface.py::test_outside_the_roots_a_file_and_a_missing_path_look_the_same` |
| P3-3 | Fixed. A lexical containment check; the route answers 400 | `945458a`, `77075f5` | `tests/test_workspace_api.py::test_a_project_name_from_a_shared_rad_cannot_place_status_outside_the_library` |
| P3-4 | Fixed. One encoder, `core/rhdr.py`, always clamps fp16; the VAE uses the real `safe_join` | `e8c0e4d`, `7705193`, `2af342b` | `tests/test_rhdr_writers.py`, `tests/test_rhdr_format.py` |
| B12 | Fixed. ffmpeg now encodes to a hidden sibling that replaces the output only on success, so a failure, timeout or cancel removes the partial encode and leaves any previous master intact | `e4f084c`, `619f0ae` | `tests/test_write_path_defects.py::test_a_failed_encode_removes_the_partial_file` |

Validation for the pass: the full Python suite with real torch, OpenEXR, OCIO and ffmpeg, the no-torch
suite, the CI ruff command and the JS tests, all run locally. An independent review of the whole diff
found one regression (the P2-4 fix, reverted) and two smaller issues (fixed in `55afb69`, `77075f5`).
Structural debt and optional improvements below are unchanged.

---

## 1. Defects

### P1-1  SDR reference conditioning crashes with any real VAE
- **Where:** `nodes/generate/sampler.py:1504` (`_encode_sdr_reference`), called at `:1726`.
- **Trigger:** connect `sdr_reference` + `sdr_vae` on RadianceSamplerPro and set `sdr_blend > 0`.
- **Defect:** `res = vae.encode(ref); ref_latent = res["samples"]`. `comfy.sd.VAE.encode` returns a
  tensor, not a LATENT dict (only the `VAEEncode` node wraps it). Indexing a tensor with a string
  raises, so the whole sample fails. A video VAE would also fail at `expand(B,-1,-1,-1)` on a 5-D result.
- **Why tests miss it:** `tests/test_sdr_conditioning.py:354-367` `_FakeVAE.encode` returns
  `{"samples": ...}`, the shape the bug assumes.
- **Test to add:** make the fake VAE return a bare tensor (as ComfyUI does) and assert the 4-D and 5-D
  shapes. Fix: accept both (`res["samples"] if isinstance(res, dict) else res`).

### P1-2  Delivery "AI upscale 2x" can silently ship a Lanczos master
- **Where:** `delivery/handler.py:691-715` and `image/upscale.py:2377-2379, 2393, 2536-2545` (`_fallback_upscale` at `:2241`).
- **Trigger:** `upscale_2x` on, with any of these: the model is missing, the download fails or is
  disabled, CUDA OOM, or any inference exception.
- **Defect:** `RadianceAIUpscale.upscale` catches every failure and returns a Lanczos resize.
  The delivery code only records the warning "AI upscale did not run, the master is 1x" when an
  exception escapes, so that path never runs. The master is 2x by resampling and no receipt warning says so.
- **Test to add:** patch `_load_model` to return `(None, "missing")` and assert that delivery records a
  warning. The fix is a `strict` / `raise_on_fallback` flag, or having `upscale` return a fallback marker in `info`.

### P2-1  Flux.2 auto-detects as Flux.1 when `model_meta` is not connected
- **Where:** `sampler_utils.py:592-603` (`detect_by_config`).
- **Defect:** `for pattern, mtype in config_map.items(): if pattern in config_cls`. `"Flux"` comes
  before `"Flux2"`, so `"Flux" in "Flux2"` returns `"flux"` first. The `ALBABIT-FIX` comment on the line
  above says this is fixed, but the fix has no effect.
- **Impact:** Flux.2 picks up the Flux.1 defaults (guidance 3.5, not 4.0) and any `"flux"`-specific
  branch, such as the cfg check at `nodes/generate/sampler.py:1183`. Connecting the Loader's `model_meta` hides the bug.
- **Test to add:** a fake model whose `model_config` class is named `Flux2` should detect as `"flux2"`.
  The fix is an exact `config_map.get(config_cls)` lookup first, or matching the longest pattern first.

### P2-2  ProjectManager saves API-format graphs that the dashboard cannot reopen
- **Where:** `nodes/pipeline/workspace.py:155-157` (`graph_data = prompt or extra_pnginfo or {}`),
  `:279-281` (`_inspect_graph_content` reads `data["nodes"]`), and `js/radiance_workspace.js:855` (`app.loadGraphData(content)`).
- **Defect:** `prompt` is the API-format dict (`{id: {class_type, inputs}}`) and has no `nodes` key.
  As a result `node_count`, models, colour spaces and `is_hdr` are always empty in the saved metadata.
  The library's "open" passes an API prompt to `loadGraphData`, which expects the UI workflow
  (`extra_pnginfo["workflow"]`). The open is therefore expected to fail or load an empty canvas. This is Inferred: it depends on the frontend version.
- **Test to add:** run `_save` with a real API prompt plus `extra_pnginfo={"workflow": {...,"nodes":[...]}}`
  and assert that `stats.node_count > 0` and that the stored content has a `nodes` key.

### P2-3  The dashboard does not parse the ProjectManager node's own filenames
- **Where:** `nodes/pipeline/workspace.py:662-675` (`_shot_from_name`, `_version_from_name`) against `:146` (`f"{filename}_{artist}_v{version:04d}"`).
- **Defect:** both regexes start with `\b`, and `_` is a word character, so `sh010_v002` and
  `x_artist_v0003` match neither. Checked: `re.search(r"\bv(\d{3,4})\b", "sh010_v002")` returns `None`.
  Every workflow the node saves shows as version `v001` and shot `GENERAL`, and its shot status is keyed under `GENERAL`.
- **Tests:** `tests/test_workspace_api.py:410-428` only use `-`- and space-separated names. Add the cases `"sh010_v002"` and `"comp_artist_v0003"`.

### P2-4  Sequential offload permanently switches the whole ComfyUI process to LOW_VRAM
- **Where:** `loader_utils.py:230-238` (`setup_offload_mode`).
- **Defect:** `comfy.model_management.vram_state = LOW_VRAM` is a process global and is never restored.
  One Loader run with `offload_mode="sequential"` slows every later model load in the session,
  including unrelated workflows, until ComfyUI restarts.
- **Test to add:** after a `sequential` run, a run with `offload_mode="none"` should leave `vram_state` as it was.

### P2-5  `/radiance/assets/upload` silently overwrites existing inputs
- **Where:** `nodes/pipeline/workspace.py:1866-1890`.
- **Defect:** `dest = dest_dir / os.path.basename(part.filename)` is opened with `"wb"` and no existence check.
  Uploading a second `plate.exr` destroys the first, and any workflow that still references it then reads
  the new file. The write is also not atomic, so an aborted upload leaves a truncated file under the original name.
- **Test to add:** upload the same name twice and assert that both survive (suffixed) or that the second request is rejected with 409.

### P3-1  Delivery filenames carry two version suffixes
- **Where:** `delivery/handler.py:470-474` adds `_v02` to `filename_prefix`. `io/writer.py:1319-1321` then adds `_v{version:04d}`.
- **Impact:** the master is named `Radiance_Deliver_v02_v0001.mov`. `get_next_version` still increments,
  but the naming contract is inconsistent for editorial and conform. Pass `version=` to `write_frames`
  rather than baking the suffix into `filename`. `tests/test_delivery_handler.py` should assert the final name.

### P3-2  `/radiance/media/*` is still a file-existence oracle
- **Where:** `nodes/io/write.py:1407-1418` (`_resolve_query_path`).
- **Defect:** `os.path.isfile` runs **before** `_is_inside_allowed_read_root`. Any path outside the
  roots returns 404 `"not a file"` with the path echoed when it is missing, and 403 when it exists. The
  docstring at `:1375-1380` says the root check exists to prevent exactly this oracle. `radiance_ocio.py:626-643`
  orders the two checks correctly.
- **Test to add:** request `/radiance/media/info?path=/nonexistent` and `?path=/etc/hostname`. Both should return 403.

### P3-3  Shot-status path is built from untrusted `.rad` metadata
- **Where:** `nodes/pipeline/workspace.py:745-753, 797-798, 815-821`.
- **Defect:** the project name comes from `metadata["project"]` or `metadata["show"]` inside any library `.rad`
  (or its v1 `.json` sidecar). `_shot_status_path` joins it into `WORKFLOW_DIR` without a containment check.
  A shared `.rad` with `"project": "../../.."` makes `POST /radiance/projects/<slug>/shots/x/status` create
  `.shot_status.json` outside the library. The content is limited to `{shot: status}`.
- **Fix:** route the path through `_resolve_safe_path`, or key it by `project["id"]` (the slug).

### P3-4  HDR VAE `.rhdr` export: dead traversal guard, and fp16 overflow to inf
- **Where:** `hdr/vae.py:2696-2700, 2703-2709`.
- **Defect:** `from .path_utils import safe_join` always raises ImportError (`hdr/path_utils.py` doesn't exist;
  the helper is `core/system/path_utils.py:15`), so the guard is dead. Today's prefixes are internal, so this
  is not exploitable. The fp16 branch also casts without the ±65504 clamp that `nodes/monitor/viewer.py:870`
  applies, so specular values above 65504 become `inf` in the viewer and poison tonemap and scopes.
- **Test to add:** `_save_rhdr` with a pixel at 1e5 and `precision="f16"` must not contain inf.

### P2-6  Remote DCC bridge relays ComfyUI `/prompt` with no authentication
- **Severity:** P2, not P3 as first listed (P3-5): it needs the non-default `RADIANCE_ALLOW_REMOTE_BRIDGE=1`, and the scale above puts a security gap behind a non-default setup at P2.
- **Where:** `nodes/pipeline/dcc.py:86-101` (`queue`), with the bind policy at `:117-149`.
- **Defect:** when the opt-in `RADIANCE_ALLOW_REMOTE_BRIDGE=1` is set, anyone who can reach the port can queue
  any workflow. A queued workflow can run any installed node, which is effectively code execution. Every
  connection gets an unbounded thread, and `stop_server` has no caller. Add a shared-secret token, or refuse `queue` on a non-loopback bind.

---

## 2. Structural debt (not defects today)

- **No auth on `/radiance/*` routes.** This is safe only while ComfyUI binds to localhost. Deliver containment uses `abspath`,
  not `realpath` (`delivery/handler.py:404-417`), so a symlink inside `output/` escapes it.
- **Every dashboard API call re-reads every `.rad`.** `_read_workflow_records` (`workspace.py:678-735`) reads each v3 file
  twice (`read_bytes` plus `ZipFile`) per request, and `/projects`, `/versions`, `/outputs`, `/notes` each call it. Cost grows with library size.
- **Five OCIO resolvers and several download paths** with different integrity rules (OPEN_QUESTIONS B17, P1, P2). Addressed in 4.0: one active-config source, and MoGe, Marigold, the ACES config and whisper verified or consent-gated; the multipass registry pins remain.
- **`hdr/vae.py` is 3.6k lines.** (The `.rhdr` part of this item is resolved: all writers now go through `core/rhdr.py`.)
- **Test doubles that encode the wrong contract** (`_FakeVAE` in P1-1, now corrected) give false confidence. Audit other fakes against the real ComfyUI return types.
- **Error JSON returned with HTTP 200** by `/radiance/ocio/*` and `/radiance/media/info`, so the JS cannot tell a failure from a success by status code.

## 3. Optional improvements

- Atomic writes (temp file + `os.replace`) for uploads, workflow saves, shot status, and media (OPEN_QUESTIONS P5).
- A single `ModelType` lookup table shared by `sampler_utils.detect_by_config` and `model/detect.py`.
- The VideoSampler uses `latent_noise` as both the noise and the start latent (`nodes/video/t2v.py:998-1001`).
  The tooltip documents this, but a separate `noise_seed` input would make real v2v denoise possible.

---

## Reviewed areas

`delivery/handler.py` (route, versioning, upscale), `nodes/pipeline/workspace.py` (all routes, the project index, the ProjectManager node),
`nodes/io/write.py` (media routes), `radiance_ocio.py` (routes), `nodes/pipeline/dcc.py`, `nodes/pipeline/studio_integrations.py`
(Nuke/Resolve send), `sampler_utils.py` (registry, temporal windows), `nodes/generate/sampler.py` (SDR conditioning, defaults),
`loader_utils.py` (offload), `image/upscale.py` (AI upscale fallback), `hdr/vae.py` (`_save_rhdr`), `hdr/decode_meta.py`,
`color/encodings.py` (PQ/HLG/LogC3/S-Log3 maths checked against the specs; no defects), the `nodes/monitor/viewer.py` `.rhdr` writer,
`js/radiance_viewer.js` `_parseRHDR`, `js/radiance_workspace.js` library load, `io/writer.py` (`write_frames` naming, ffmpeg failure).

Re-checked and downgraded or excluded: OPEN_QUESTIONS B1 is not exploitable (the prefixes are internal), B20 is moot (the user
already picks the folder), and B25 is documented behaviour (see Optional improvements).

Correction (reconciled with OPEN_QUESTIONS): this report first said B12 was fixed. It isn't. The partial file is removed after a
timeout (`io/writer.py:1010`) or a mid-write exception (`:986`), but a non-zero ffmpeg exit only raises (`:1015`) and leaves the
partial file in place. Severity P3: the error is reported, but a truncated master stays on disk under its final name.

## Unreviewed areas

- `nodes/vfx/*` (depth, flow, multipass estimate, relight). The polarity and axis conventions in B24 were not re-verified.
- `nodes/video/*` beyond the T2V and VideoSampler sampling call, `nodes/ai/scene_cut.py`, `temporal_rudra.py`, `pixel_sdr2hdr.py`.
- `io/reader.py` (B5, B9, B10, B11 not re-verified), `core/exr.py`, `core/video.py`.
- `hdr/color.py`, `hdr/processing.py`, `hdr/panorama.py`, `hdr/recovery.py`, `hdr/tonemap.py`, `hdr/tonescale.py`, `hdr/aces2_ocio.py`.
- `nodes/gizmo.py` executor and routes, `nodes/pipeline/audio.py` (B18, B26), `tools/nuke_connector.py` (B19), `start_nuke_server.py`.
- The `js/` bodies beyond the entry points named above (WebGL renderer, scopes, dashboards).
- The test suite and CI were not run. No finding here is reproduced at runtime.

## Suggested verification

```
python -m pytest tests/test_sdr_conditioning.py tests/test_workspace_api.py tests/test_delivery_handler.py -q
node --test js/tests/*.test.mjs
```
Add the tests named under each finding before changing the code. Each one should fail on the current revision.
