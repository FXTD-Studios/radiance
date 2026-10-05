<!-- project-mapper:generated -->
# DCC integrations (Nuke, Resolve, bridge, NDI) and gizmo execution

ID: dcc (plus gizmo runtime). Snapshot: `7376f9e`. Coverage: partial. Agent trace of `nodes/pipeline/{dcc,studio_integrations,studio,overlay,metadata}.py`, `tools/nuke_connector.py`, `tools/resolve_import.py`, `scripts/start_nuke_server.py`, `core/dcc_auth.py`, `nodes/gizmo.py`, and `NDISender`. The mapper re-read the connector's timeout handling and the Nuke send paths. Labels are explained in [START_HERE](../START_HERE.md#evidence-labels).

## Responsibility and boundaries

These are outbound integrations with external tools, and the runtime for saved subgraphs ("gizmos").
Workspace persistence (`workspace.py`) is in [frontend-routes](frontend-routes.md#persistence).

## Implementation index

| Source/symbol | Role | Label |
| --- | --- | --- |
| `studio_integrations.py:RadianceNukeSend.run` (:147-220) | Writes EXRs **directly into `nuke_folder`** (not atomic, overwrites) through `io/writer._save_exr`, plus a `.nk` Read snippet. Pushes to Nuke if asked. Never raises; returns a status string | Implemented (mapper re-read paths) |
| `tools/nuke_connector.py` | TCP framing: `RCMD`, version byte, HMAC-SHA256(token, command), length, JSON payload. Connect timeout 5 s, read timeout 15 s. **A read timeout after send returns success with a partial reply** (:176-186) | Implemented (mapper) |
| `core/dcc_auth.py:load_or_create_token` | `RADIANCE_DCC_AUTH_TOKEN`, else `~/.radiance/dcc_token` (O_EXCL, 0600) | Implemented |
| `scripts/start_nuke_server.py` | Runs inside Nuke. Binds to loopback by default. Mandatory HMAC (`compare_digest`), 1 MB payload cap. Actions: `ping`, `set_frame`, `get_info`, `load_exr`; anything else goes only to `ast.literal_eval`. **No eval/exec.** No nonce, so messages can be replayed | Implemented |
| `studio_integrations.py:RadianceDaVinciSend.run` (:339-382), `import_into_resolve` (:257-282) | Writes TIFF16, PNG8, or EXR, then runs `tools/resolve_import.py` as a subprocess with a 30 s timeout and a JSON request on stdin | Implemented |
| `tools/resolve_import.py` | `DaVinciResolveScript` → current project → `MediaPool.ImportMedia`. Explicit messages for each failure | Implemented |
| `dcc.py:RadianceMCP` "Bridge Server" | Starts **only when the node runs** in that mode. Newline-JSON over TCP (`ping`, `status`, `exec` always refused, `queue` POSTs to ComfyUI `/prompt`). Loopback unless `RADIANCE_ALLOW_REMOTE_BRIDGE`. **No authentication.** Daemon threads with no cap. `stop_server` is never called | Implemented |
| `nodes/generate/engine.py:NDISender` (:779-938) | `NDIlib` singleton sender, 8-bit BGRA, `clock_video=True`. Destroyed at `atexit` | Implemented |
| `nodes/gizmo.py:run_subgraph_executor` (:56-209) | Kahn topological sort, then each node class is called directly from ComfyUI `nodes.NODE_CLASS_MAPPINGS`. **Bypasses the ComfyUI executor**: no caching, hidden inputs, lists, lazy evaluation, `ui` results, or `VALIDATE_INPUTS`. A cycle is only logged | Implemented |
| `studio.py:RadianceCinemaStudio`, `overlay.py:RadianceBlendComposite`, `metadata.py:RadianceLinearCheck` | Prompt and tech-data text; an 8-mode fp32 blend; a check of the `RADIANCE_SHOT` colourspace tag | Implemented |

## Contracts

- **Nuke send** uses the configured host and port (default 127.0.0.1:1986). The token is shared through `~/.radiance/dcc_token`.
- **Gizmo schema** (`.gizmo` JSON): `name`, `display_name`, `category`, `description`, `inputs[]`, `outputs[]`, `widgets[]`, `nodes[]`, `links[]` (`create_gizmo_api` :345-355). Classes are built at import, so a **new gizmo needs a ComfyUI restart**. Implemented.
- **Failure semantics:** DCC nodes return status strings instead of raising. The gizmo executor re-raises node errors as `RuntimeError`. Implemented.

## Important paths

Nuke and Resolve sends are traced in [WORKFLOWS W11](../WORKFLOWS.md#w11-send-to-nuke-or-resolve) and drawn in [diagrams/flows/dcc.mmd](../diagrams/flows/dcc.mmd).

## Tests (inspected by agent, not run)

- `tests/test_bridge_security.py` (exec refused, no eval scaffolding).
- `test_dcc_send.py` (the real Nuke listener against a stand-in `nuke` module, HMAC, Resolve isolation).
- `test_dcc_studio.py`, `test_secret_utils.py`, `test_ndi_sender_batch.py`.
- `test_gizmo_storage.py` only checks `GIZMOS_DIR`. **There are no executor tests.**

## Open questions

See [OPEN_QUESTIONS](../OPEN_QUESTIONS.md): B20 (`filename` path separators in Nuke/Resolve send), B19 (connector timeout reports success), B21 (bridge has no auth when remote binding is allowed), D19 (replay comment), and the gizmo-executor design question.
