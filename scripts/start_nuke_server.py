import nuke
import nukescripts
import socket
import struct
import threading
import json
import ast
import os
import urllib.request
import urllib.error
import random

PORT = int(os.environ.get("RADIANCE_NUKE_PORT", "1986"))
# v1.1: Configurable via env vars for studio/farm deployments.
# Bind host: 127.0.0.1 by default (loopback only — safe).
# Set RADIANCE_NUKE_BIND_HOST=0.0.0.0 only when Nuke runs on a separate machine.
BIND_HOST = os.environ.get("RADIANCE_NUKE_BIND_HOST", os.environ.get("RADIANCE_NUKE_HOST", "127.0.0.1"))
# ComfyUI base URL for history/prompt API calls.
COMFY_URL = os.environ.get("RADIANCE_COMFY_URL", "http://127.0.0.1:8188")
def load_or_create_token():
    """The token shared with ComfyUI: RADIANCE_DCC_AUTH_TOKEN, else
    ~/.radiance/dcc_token, created on first use. A copy of
    radiance.core.dcc_auth.load_or_create_token (this script runs inside Nuke,
    where the package is not importable); keep the two identical."""
    import secrets
    tok = (os.environ.get("RADIANCE_DCC_AUTH_TOKEN") or "").strip()
    if tok:
        return tok
    path = os.path.join(os.path.expanduser("~"), ".radiance", "dcc_token")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            tok = fh.read().strip()
        if tok:
            return tok
    except OSError:
        pass
    tok = secrets.token_hex(32)
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(tok)
    except FileExistsError:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read().strip()
    except OSError:
        return ""
    return tok


# Commands must be signed with this token. It used to be read only from
# RADIANCE_DCC_AUTH_TOKEN, unset by default, so the listener refused every
# command until both Nuke and ComfyUI were given the same variable.
DCC_AUTH_TOKEN = load_or_create_token()
RUNNING = True
_SERVER_THREAD = None

# --- Bridge Server Logic ---


def safe_message(msg):
    """Safely show message in Nuke UI from any thread."""
    if nuke.GUI:
        nuke.executeInMainThread(lambda: nuke.message(msg))


# execute_in_main removed - logic inlined in handle_client

RADIANCE_MAGIC = b"RCMD"
RADIANCE_END = "\n__RADIANCE_END__\n"

# ═══════════════════════════════════════════════════════════════════════════════
#                       COMMAND SANDBOX (Security)
# ═══════════════════════════════════════════════════════════════════════════════

# Dangerous patterns that must never appear in incoming commands
_BLOCKED_PATTERNS = [
    "import os",
    "import sys",
    "import subprocess",
    "import shutil",
    "import socket",
    "import http",
    "import urllib",
    "__import__",
    "__builtins__",
    "__class__",
    "__subclasses__",
    "getattr(",
    "setattr(",
    "delattr(",
    "os.system",
    "os.popen",
    "os.exec",
    "os.remove",
    "os.unlink",
    "os.rmdir",
    "os.makedirs",
    "os.rename",
    "subprocess.",
    "shutil.rmtree",
    "shutil.move",
    "open(",
    "file(",
    "compile(",
    "globals()[",
    "locals()[",
    "eval(",
    "exec(",  # Prevent nested eval/exec
]

# Allowed top-level module names (Nuke API + standard safe modules)
_ALLOWED_IMPORTS = {"nuke", "nukescripts", "json", "math", "random"}
_ALLOWED_STRUCTURED_ACTIONS = {"ping", "set_frame", "get_info", "load_exr"}


def _validate_command(command):
    """
    Validate a command against the security blocklist.

    Returns (is_safe: bool, reason: str).
    Commands containing dangerous system-access patterns are rejected.
    Only Nuke API and safe modules are allowed.
    """
    cmd_lower = command.lower()

    # Check blocklist
    for pattern in _BLOCKED_PATTERNS:
        if pattern.lower() in cmd_lower:
            return False, f"Blocked pattern: '{pattern}'"

    # Check imports — only allow safe modules
    import re

    for match in re.finditer(r"import\s+(\w+)", command):
        module = match.group(1)
        if module not in _ALLOWED_IMPORTS:
            return False, f"Blocked import: '{module}'"

    # Check "from X import" patterns
    for match in re.finditer(r"from\s+(\w+)", command):
        module = match.group(1)
        if module not in _ALLOWED_IMPORTS:
            return False, f"Blocked from-import: '{module}'"

    return True, "OK"


def _parse_structured_command(command):
    try:
        payload = json.loads(command)
    except json.JSONDecodeError:
        return None

    if not isinstance(payload, dict):
        return None
    if payload.get("action") not in _ALLOWED_STRUCTURED_ACTIONS:
        return None
    return payload


def _set_knob_if_exists(node, knob_name, value):
    if knob_name in node.knobs():
        node[knob_name].setValue(value)


def _execute_structured_command(payload):
    """Run whitelisted bridge actions without eval/exec."""

    if not isinstance(payload, dict):
        return None

    action = payload.get("action")
    if action == "ping":
        return "RADIANCE_PONG"

    if action == "set_frame":
        frame = int(payload.get("frame", nuke.frame()))
        nuke.frame(frame)
        return f"frame={frame}"

    if action == "get_info":
        root = nuke.root()
        fmt = root.format()
        return json.dumps({
            "version": nuke.NUKE_VERSION_STRING,
            "project": root.name(),
            "fps": root["fps"].value(),
            "first": int(root["first_frame"].value()),
            "last": int(root["last_frame"].value()),
            "format": str(fmt.name()),
            "width": fmt.width(),
            "height": fmt.height(),
        })

    if action == "load_exr":
        filepath = str(payload.get("filepath", "")).replace("\\", "/").strip()
        node_name = str(payload.get("node_name", "RadianceStream")).strip()
        if not filepath:
            raise ValueError("load_exr requires filepath")
        if not node_name:
            node_name = "RadianceStream"

        first_frame = int(payload.get("first_frame", 1))
        last_frame = int(payload.get("last_frame", first_frame))
        current_frame = int(payload.get("current_frame", first_frame))
        raw = bool(payload.get("raw", True))
        color_space = str(payload.get("color_space", "linear"))
        connect_viewer = bool(payload.get("connect_viewer", True))

        node = nuke.toNode(node_name)
        if node is None or node.Class() != "Read":
            node = nuke.createNode("Read", inpanel=False)
            node.setName(node_name)

        _set_knob_if_exists(node, "file", filepath)
        _set_knob_if_exists(node, "first", first_frame)
        _set_knob_if_exists(node, "last", last_frame)
        _set_knob_if_exists(node, "origfirst", first_frame)
        _set_knob_if_exists(node, "origlast", last_frame)
        _set_knob_if_exists(node, "raw", raw)
        if not raw:
            _set_knob_if_exists(node, "colorspace", color_space)
        if "reload" in node.knobs():
            node["reload"].execute()

        if connect_viewer:
            viewer = nuke.toNode("Viewer1")
            if viewer is None:
                viewer = nuke.createNode("Viewer", inpanel=False)
            viewer.setInput(0, node)
            nuke.frame(current_frame)

        return f"{node_name}: loaded {filepath} [{first_frame}-{last_frame}]"

    return None


def handle_client(conn):
    """Handle individual client connection with binary protocol support."""
    try:
        conn.settimeout(5.0)

        # First, read the first 5 bytes of the header to determine magic and version
        prefix_data = b""
        while len(prefix_data) < 5:
            chunk = conn.recv(5 - len(prefix_data))
            if not chunk:
                break
            prefix_data += chunk

        # v1.1 FIX-4: Reject connections that don't present the RCMD magic.
        if len(prefix_data) < 5 or prefix_data[:4] != RADIANCE_MAGIC:
            msg = "ERROR: Protocol mismatch — expected RCMD magic header"
            conn.sendall((msg + RADIANCE_END).encode("utf-8"))
            return

        version = prefix_data[4]

        if version == 1:
            # Read remaining 4 bytes of length for Version 1
            length_data = b""
            while len(length_data) < 4:
                chunk = conn.recv(4 - len(length_data))
                if not chunk:
                    break
                length_data += chunk
            if len(length_data) < 4:
                conn.sendall(("ERROR: Incomplete header" + RADIANCE_END).encode("utf-8"))
                return
            cmd_length = struct.unpack("<I", length_data)[0]
            sig_received = None
        elif version == 2:
            # Read remaining 36 bytes (32 signature + 4 length) for Version 2
            remaining_data = b""
            while len(remaining_data) < 36:
                chunk = conn.recv(36 - len(remaining_data))
                if not chunk:
                    break
                remaining_data += chunk
            if len(remaining_data) < 36:
                conn.sendall(("ERROR: Incomplete header" + RADIANCE_END).encode("utf-8"))
                return
            sig_received = remaining_data[:32]
            cmd_length = struct.unpack("<I", remaining_data[32:36])[0]
        else:
            conn.sendall((f"ERROR: Unsupported protocol version: {version}" + RADIANCE_END).encode("utf-8"))
            return

        # Sanity-cap: reject absurdly large payloads (> 1 MB)
        if cmd_length > 1_048_576:
            conn.sendall(("ERROR: Payload too large" + RADIANCE_END).encode("utf-8"))
            return

        # Read command payload
        cmd_data = b""
        while len(cmd_data) < cmd_length:
            chunk = conn.recv(min(32768, cmd_length - len(cmd_data)))
            if not chunk:
                break
            cmd_data += chunk
        command = cmd_data.decode("utf-8", errors="replace").strip()

        # Authentication is mandatory. It used to be skipped entirely whenever
        # RADIANCE_DCC_AUTH_TOKEN was unset -- which is the default -- so any
        # client that could reach the port was trusted.
        if not DCC_AUTH_TOKEN:
            msg = ("ERROR: No shared token: set RADIANCE_DCC_AUTH_TOKEN or make "
                   "~/.radiance/dcc_token writable; refusing all commands.")
            print(f"[Radiance Security] {msg}")
            conn.sendall((msg + RADIANCE_END).encode("utf-8"))
            return

        if version != 2 or not sig_received:
            msg = "ERROR: Authentication required — connection is unauthenticated."
            print(f"[Radiance Security] {msg}")
            conn.sendall((msg + RADIANCE_END).encode("utf-8"))
            return

        # HMAC, not SHA256(token || command): the bare-hash construction was
        # length-extendable and had no nonce, so any observed pair replayed.
        import hmac
        import hashlib
        expected_sig = hmac.new(
            DCC_AUTH_TOKEN.encode("utf-8"), command.encode("utf-8"), hashlib.sha256
        ).digest()
        if not hmac.compare_digest(sig_received, expected_sig):
            msg = "ERROR: Authentication failed — Invalid security token signature."
            print(f"[Radiance Security] {msg}")
            conn.sendall((msg + RADIANCE_END).encode("utf-8"))
            return

        if command:
            structured_payload = _parse_structured_command(command)
            if structured_payload is None:
                # ── Security: Validate raw Python before execution ──
                is_safe, reason = _validate_command(command)
                if not is_safe:
                    msg = f"ERROR: Command rejected by security sandbox — {reason}"
                    print(f"[Radiance Security] {msg}")
                    conn.sendall((msg + RADIANCE_END).encode("utf-8"))
                    return

            # Execute in main thread and capture result
            result_box = [None]
            error_box = [None]
            done_event = threading.Event()

            # There is deliberately no eval/exec sandbox here any more.
            # The restricted-__builtins__ dict that used to sit at this
            # point was dead weight once the eval/exec fallback was removed:
            # it protected nothing and read like an active defence, which is
            # worse than no defence at all. Commands are structured or
            # ast.literal_eval'd; nothing else runs.

            def execute_wrapper():
                try:
                    structured = _parse_structured_command(command)
                    structured_result = _execute_structured_command(structured)
                    if structured_result is not None:
                        result_box[0] = structured_result
                        return

                    # Literals only. The eval/exec fallback that used to live
                    # here was guarded by a substring blocklist, which a single
                    # space defeated ("open (" does not contain "open("), as did
                    # module chaining through the permitted `json` global. Both
                    # bypasses gave arbitrary code execution inside Nuke. Only
                    # structured commands and literal payloads are accepted now.
                    try:
                        val = ast.literal_eval(command)
                        result_box[0] = str(val)
                    except (ValueError, SyntaxError):
                        error_box[0] = (
                            "ERROR: Only structured commands and literal values are accepted. "
                            "Arbitrary Python execution has been removed for security reasons."
                        )
                        return
                except Exception as e:
                    error_box[0] = f"ERROR: {e}"
                finally:
                    done_event.set()

            if nuke.GUI:
                nuke.executeInMainThread(execute_wrapper)
            else:
                execute_wrapper()
            done_event.wait(timeout=10.0)

            response = (
                error_box[0]
                if error_box[0]
                else (result_box[0] if result_box[0] else "ERROR: Timeout")
            )
            conn.sendall((response + RADIANCE_END).encode("utf-8"))

    except Exception as e:
        print(f"Radiance Bridge Error: {e}")
        try:
            conn.sendall((f"ERROR: {e}" + RADIANCE_END).encode("utf-8"))
        except Exception:  # nosec B110
            pass
    finally:
        conn.close()


def start_radiance_server():
    """Listen for commands; terminal Nuke must call this on its main thread."""
    global PORT

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.settimeout(0.5)

    if BIND_HOST not in ("127.0.0.1", "localhost", "::1") and not DCC_AUTH_TOKEN:
        msg = (f"Radiance Bridge: refusing to bind {BIND_HOST} without "
               f"RADIANCE_DCC_AUTH_TOKEN set. Falling back to 127.0.0.1.")
        print(msg)
        safe_message(msg)
        bind_host = "127.0.0.1"
    else:
        bind_host = BIND_HOST

    try:
        # v1.1: Bind to bind_host (default 127.0.0.1 — loopback only).
        # Set RADIANCE_NUKE_BIND_HOST=0.0.0.0 for cross-machine studio pipelines;
        # a token is mandatory in that case.
        server.bind((bind_host, PORT))  # nosec B104
        server.listen(5)

        msg = f"Radiance Bridge listening on {bind_host}:{PORT}..."
        print(msg)
        safe_message(msg)

        while RUNNING:
            try:
                conn, addr = server.accept()
                # v1.1 FIX-3: Spawn a thread per connection so slow Nuke commands
                # don't block the accept loop. Previously a single slow client
                # would prevent any second connection until it finished.
                if nuke.GUI:
                    t = threading.Thread(target=handle_client, args=(conn,), daemon=True)
                    t.start()
                else:
                    # Terminal Nuke has no GUI dispatcher; serialize node edits.
                    handle_client(conn)
            except socket.timeout:
                continue
            except Exception as e:
                print(f"Connection error: {e}")

    except OSError as e:
        err = f"Socket error (Port {PORT} in use?): {e}"
        print(err)
        safe_message(err)
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server.close()
        print("Radiance Bridge Server Stopped")


def start_bridge():
    """Start the GUI listener in a thread, or run terminal Nuke synchronously."""
    global _SERVER_THREAD, RUNNING

    if not nuke.GUI:
        RUNNING = True
        start_radiance_server()
        return

    if _SERVER_THREAD and _SERVER_THREAD.is_alive():
        nuke.message(f"Radiance Bridge is already running on {BIND_HOST}:{PORT}")
        return

    RUNNING = True
    _SERVER_THREAD = threading.Thread(target=start_radiance_server)
    _SERVER_THREAD.daemon = True
    _SERVER_THREAD.start()


def stop_bridge():
    """Stop the bridge server cleanly."""
    global RUNNING, _SERVER_THREAD
    if not (_SERVER_THREAD and _SERVER_THREAD.is_alive()):
        nuke.message("Radiance Bridge is not running.")
        return
    RUNNING = False
    _SERVER_THREAD.join(timeout=2.0)
    _SERVER_THREAD = None
    print("Radiance Bridge stopped.")
    nuke.message("Radiance Bridge stopped.")


# --- ComfyUI Interaction Logic ---


def fetch_last_history_item():
    """Fetch the most recent history item from ComfyUI."""
    try:
        url = f"{COMFY_URL}/history"
        if not url.startswith(("http://", "https://")):
            return None
        with urllib.request.urlopen(url, timeout=10) as response:  # nosec B310
            if response.status != 200:
                return None
            data = json.loads(response.read().decode("utf-8"))
            if not data:
                return None
            last_key = list(data.keys())[-1]
            return data[last_key]
    except Exception as e:
        print(f"ComfyUI API Error: {e}")
        return None


def find_node_by_class(graph, class_name):
    """Find a node by its class type."""
    if not graph:
        return None, None
    for node_id, node_data in graph.items():
        if node_data.get("class_type") == class_name:
            return node_id, node_data
    return None, None


def find_cinematic_node(graph):
    """Find the compatible Cinematic node (Encoder or Studio) in the prompt data."""
    if not graph:
        return None, None, None

    nid, ndata = find_node_by_class(graph, "RadianceCinematicPromptEncoder")
    if nid:
        return nid, ndata, "ENCODER"

    nid, ndata = find_node_by_class(graph, "RadianceCinemaStudio")
    if nid:
        return nid, ndata, "STUDIO"

    return None, None, None


def submit_prompt(graph):
    """Submit a graph to ComfyUI."""
    try:
        # v1.1 FIX-6: Include client_id so ComfyUI routes execution events
        # back correctly. Without it, execution_start/cached events are broadcast
        # to all clients and the 'fired before prompt was made' warning fires.
        data = json.dumps({"prompt": graph, "client_id": "radiance_nuke_bridge"}).encode("utf-8")
        req = urllib.request.Request(
            f"{COMFY_URL}/prompt",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        url = req.full_url
        if not url.startswith(("http://", "https://")):
            return
        with urllib.request.urlopen(req, timeout=10) as resp:  # nosec B310
            if resp.status == 200:
                print("ComfyUI Queued Successfully")
            else:
                nuke.message(f"ComfyUI Error: {resp.status}")
    except Exception as e:
        nuke.message(f"Failed to queue prompt: {e}")


def randomize_seed(graph):
    """Randomize seed in sampler nodes to force re-generation."""
    for node in graph.values():
        if "inputs" in node:
            if "seed" in node["inputs"]:
                node["inputs"]["seed"] = random.randint(1, 18446744073709551615)  # nosec B311
            elif "noise_seed" in node["inputs"]:
                node["inputs"]["noise_seed"] = random.randint(1, 18446744073709551615)  # nosec B311


def queue_last_run():
    item = fetch_last_history_item()
    if not item:
        nuke.message("No history found in ComfyUI.")
        return

    prompt_payload = item.get("prompt")
    if not prompt_payload:
        nuke.message("Invalid history item format.")
        return

    graph = None
    if isinstance(prompt_payload, list) and len(prompt_payload) >= 3:
        graph = prompt_payload[2]
    elif isinstance(prompt_payload, dict):
        graph = prompt_payload
        # Verify
        valid = False
        for k, v in graph.items():
            if isinstance(v, dict) and "inputs" in v:
                valid = True
                break
        if not valid:
            graph = None

    if graph:
        randomize_seed(graph)
        submit_prompt(graph)
    else:
        nuke.message("Could not find graph in history.")


# --- Cinematic Encoder Panel ---


class RadiancePromptPanel(nukescripts.PythonPanel):
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    def __init__(self):
        super(RadiancePromptPanel, self).__init__("Radiance Cinematic")

        # --- Data Lists ---
        self.cameras = [
            "None",
            "ARRI Alexa 35",
            "ARRI Alexa Mini LF",
            "ARRI Alexa 65 (IMAX)",
            "Sony Venice 2",
            "RED V-Raptor XL",
            "Panavision Millennium DXL2",
            "IMAX 15/70mm Film Camera",
        ]
        self.lenses = [
            "None",
            "50mm Standard Prime",
            "35mm Classic Wide",
            "85mm Portrait Prime",
            "24mm Wide Angle",
            "Anamorphic Lens",
            "ARRI/Zeiss Signature Prime",
            "Panavision Primo 70",
            "Cooke S7/i Full Frame",
        ]
        self.apertures = [
            "None",
            "f/1.2 (Dreamy Bokeh)",
            "f/1.8 (Soft Background)",
            "f/2.8 (Cinematic Separation)",
            "f/4.0 (Balanced)",
            "f/5.6 (Sharp Subject)",
            "f/8.0 (Deep Focus)",
        ]
        self.framings = [
            "None",
            "Medium Shot (MS)",
            "Close-Up (CU)",
            "Wide Shot (WS)",
            "Extreme Wide Shot (EWS)",
            "Low Angle (Hero Shot)",
        ]

        self.presets = [
            "SDXL Square (1024×1024)",
            "SDXL 16:9 (1216×832)",
            "SDXL 9:16 (832×1216)",
            "SDXL 4:3 (1152×896)",
            "SDXL 3:4 (896×1152)",
            "Flux Square (1024×1024)",
            "Flux 16:9 (1360×768)",
            "Flux 9:16 (768×1360)",
            "Flux 3:2 (1256×832)",
            "Flux 2:3 (832×1256)",
            "Flux 21:9 (1536×656)",
            "Flux 4:3 (1184×888)",
            "Flux 2.39:1 (1568×656)",
            "HD 1080p (1920×1080)",
            "HD 720p (1280×720)",
            "4K UHD (3840×2160)",
            "4K DCI (4096×2160)",
            "2K DCI (2048×1080)",
            "Instagram Square (1080×1080)",
            "Instagram Story (1080×1920)",
            "TikTok (1080×1920)",
        ]

        self.sampler_modes = [
            "Standard",
            "Phase-Shift (Euler→DPM)",
            "Phase-Shift (Euler→SGM)",
            "CFG++ (Perpendicular)",
        ]

        # v1.1 FIX: Query available VAEs from ComfyUI at panel init instead of
        # a hardcoded 3-item list. Falls back to the static list if unavailable.
        _vae_fallback = [
            "ae.safetensors",
            "vae-ft-mse-840000-ema-pruned.safetensors",
            "sdxl_vae.safetensors",
        ]
        try:
            _vae_url = f"{COMFY_URL}/models/vae"
            with urllib.request.urlopen(_vae_url, timeout=5) as _r:  # nosec B310
                _fetched = json.loads(_r.read().decode("utf-8"))
                self.vaes = _fetched if isinstance(_fetched, list) and _fetched else _vae_fallback
        except Exception:
            self.vaes = _vae_fallback

        self.samplers = [
            "euler",
            "euler_ancestral",
            "heun",
            "dpm_2",
            "dpm_2_ancestral",
            "dpmpp_2s_ancestral",
            "dpmpp_2m",
            "dpmpp_2m_sde",
            "dpmpp_sde",
            "uni_pc",
        ]
        self.schedulers = [
            "normal",
            "karras",
            "exponential",
            "sgm_uniform",
            "simple",
            "ddim_uniform",
        ]

        # RadianceDepthMapGenerator inputs
        self.depth_models = [
            "Small (25M - Fast)",
            "Base (98M - Balanced)",
            "Large (335M - Best)",
        ]

        # --- UI Elements & Layout ---

        # --- TAB 1: CREATIVE & COMPOSITION ---
        self.creative_tab = nuke.Tab_Knob("creative_tab", "Creative & Composition")
        self.addKnob(self.creative_tab)

        self.creative_hdr = nuke.Text_Knob(
            "creative_hdr", "",
            "<span style='color:#ffa500; font-size:12px; font-weight:bold;'>◎ CREATIVE & COMPOSITION</span>"
        )
        self.addKnob(self.creative_hdr)

        self.prompt_knob = nuke.Multiline_Eval_String_Knob("base_prompt", "Prompt")
        self.addKnob(self.prompt_knob)

        self.camera_knob = nuke.Enumeration_Knob("camera", "Camera", self.cameras)
        self.addKnob(self.camera_knob)

        self.lens_knob = nuke.Enumeration_Knob("lens", "Lens / Focal", self.lenses)
        self.addKnob(self.lens_knob)

        self.aperture_knob = nuke.Enumeration_Knob("aperture", "Aperture", self.apertures)
        self.addKnob(self.aperture_knob)

        self.framing_knob = nuke.Enumeration_Knob("framing", "Framing", self.framings)
        self.addKnob(self.framing_knob)

        # --- TAB 2: RENDER & SAMPLER ---
        self.sampler_tab = nuke.Tab_Knob("sampler_tab", "Render & Sampler")
        self.addKnob(self.sampler_tab)

        self.sampler_hdr = nuke.Text_Knob(
            "sampler_hdr", "",
            "<span style='color:#00ffff; font-size:12px; font-weight:bold;'>◎ CORE RENDER & SAMPLING</span>"
        )
        self.addKnob(self.sampler_hdr)

        self.res_knob = nuke.Enumeration_Knob("resolution", "Resolution Preset", self.presets)
        self.addKnob(self.res_knob)

        self.steps_knob = nuke.Int_Knob("steps", "Steps")
        self.addKnob(self.steps_knob)

        self.cfg_knob = nuke.Double_Knob("cfg", "CFG Scale")
        self.cfg_knob.setRange(1.0, 20.0)
        self.addKnob(self.cfg_knob)

        self.denoise_knob = nuke.Double_Knob("denoise", "Denoise")
        self.denoise_knob.setRange(0.0, 1.0)
        self.addKnob(self.denoise_knob)

        self.sampler_knob = nuke.Enumeration_Knob("sampler_name", "Sampler", self.samplers)
        self.addKnob(self.sampler_knob)

        self.scheduler_knob = nuke.Enumeration_Knob("scheduler_name", "Scheduler", self.schedulers)
        self.addKnob(self.scheduler_knob)

        self.sampler_mode_knob = nuke.Enumeration_Knob("sampler_mode", "Mode", self.sampler_modes)
        self.addKnob(self.sampler_mode_knob)

        self.vae_knob = nuke.Enumeration_Knob("vae", "VAE Model", self.vaes)
        self.addKnob(self.vae_knob)

        # --- TAB 3: FLUX & AI DEPTH ---
        self.flux_depth_tab = nuke.Tab_Knob("flux_depth_tab", "Flux & AI Depth Settings")
        self.addKnob(self.flux_depth_tab)

        self.flux_hdr = nuke.Text_Knob(
            "flux_hdr", "",
            "<span style='color:#39ff14; font-size:12px; font-weight:bold;'>◎ FLUX CORE CONFIGURATION</span>"
        )
        self.addKnob(self.flux_hdr)

        self.guidance_knob = nuke.Double_Knob("guidance", "Guidance (Distilled)")
        self.guidance_knob.setRange(0.0, 10.0)
        self.guidance_knob.setValue(3.5)
        self.addKnob(self.guidance_knob)

        self.mu_knob = nuke.Double_Knob("mu", "Mu (Flux Shift)")
        self.mu_knob.setRange(0.0, 10.0)
        self.mu_knob.setValue(1.0)
        self.addKnob(self.mu_knob)

        self.depth_hdr = nuke.Text_Knob(
            "depth_hdr", "",
            "<span style='color:#ff00ff; font-size:12px; font-weight:bold;'>◎ AI DEPTH GENERATION</span>"
        )
        self.addKnob(self.depth_hdr)

        self.depth_label = nuke.Text_Knob("depth_label", "Depth Pass Status")
        self.addKnob(self.depth_label)

        self.depth_model_knob = nuke.Enumeration_Knob("depth_model", "Depth Model", self.depth_models)
        self.addKnob(self.depth_model_knob)

        self.depth_blur_knob = nuke.Double_Knob("depth_blur", "Edge Blur")
        self.depth_blur_knob.setRange(0.0, 5.0)
        self.addKnob(self.depth_blur_knob)

        self.depth_invert_knob = nuke.Boolean_Knob("depth_invert", "Invert Depth")
        self.addKnob(self.depth_invert_knob)

        # Bottom Global Actions Block
        self.actions_tab = nuke.Tab_Knob("actions_tab", "◎ Trigger Controls")
        self.addKnob(self.actions_tab)

        self.refresh_btn = nuke.PyScript_Knob("refresh", "Sync from Last Run")
        self.addKnob(self.refresh_btn)

        self.generate_btn = nuke.PyScript_Knob("generate", "Generate (Queue)")
        self.addKnob(self.generate_btn)

        # State
        self.last_graph = None
        self.node_id = None
        self.node_type = None
        self.res_node_id = None
        self.sampler_node_id = None
        self.depth_node_id = None

        # Attempt initial sync & visibility refresh
        self.sync_from_last()
        self.update_flux_visibility()

    def knobChanged(self, knob):
        if knob == self.refresh_btn:
            self.sync_from_last()
        elif knob == self.generate_btn:
            self.generate()
        elif knob == self.res_knob:
            self.update_flux_visibility()

    def update_flux_visibility(self):
        is_flux = "Flux" in self.res_knob.value()
        self.flux_hdr.setVisible(is_flux)
        self.guidance_knob.setVisible(is_flux)
        self.mu_knob.setVisible(is_flux)

        has_depth = self.depth_node_id is not None
        self.depth_hdr.setVisible(has_depth)
        self.depth_label.setVisible(has_depth)
        self.depth_model_knob.setVisible(has_depth)
        self.depth_blur_knob.setVisible(has_depth)
        self.depth_invert_knob.setVisible(has_depth)

    def sync_from_last(self):
        item = fetch_last_history_item()
        if not item:
            print("No history to sync.")
            return

        prompt_payload = item.get("prompt")
        graph = None
        if isinstance(prompt_payload, list) and len(prompt_payload) >= 3:
            graph = prompt_payload[2]
        elif isinstance(prompt_payload, dict):
            graph = prompt_payload

        if not graph:
            print("Could not extract graph for sync.")
            return

        self.last_graph = graph

        # 1. Sync Cinematic Node
        nid, ndata, ntype = find_cinematic_node(graph)
        if nid:
            self.node_id = nid
            self.node_type = ntype
            inputs = ndata.get("inputs", {})

            self.prompt_knob.setValue(inputs.get("base_prompt", ""))

            camera_val = (
                inputs.get("camera_type")
                if ntype == "ENCODER"
                else inputs.get("camera")
            )
            lens_val = (
                inputs.get("lens_focal")
                if ntype == "ENCODER"
                else inputs.get("focal_length")
            )
            ap_val = (
                inputs.get("aperture_dof")
                if ntype == "ENCODER"
                else inputs.get("aperture")
            )
            frame_val = inputs.get("framing") if ntype == "ENCODER" else "None"

            if camera_val in self.cameras:
                self.camera_knob.setValue(camera_val)
            if lens_val in self.lenses:
                self.lens_knob.setValue(lens_val)
            if ap_val in self.apertures:
                self.aperture_knob.setValue(ap_val)
            if frame_val in self.framings:
                self.framing_knob.setValue(frame_val)

            print(f"Synced with {ntype} Node {nid}")
        else:
            print("No Radiance cinematic node found.")
            self.node_id = None

        # 2. Sync Resolution Node
        rid, rdata = find_node_by_class(graph, "RadianceResolution")
        self.res_node_id = rid
        if rid:
            rinputs = rdata.get("inputs", {})
            preset = rinputs.get("preset")
            if preset in self.presets:
                self.res_knob.setValue(preset)
            print(f"Synced with Resolution Node {rid}")

        # 3. Sync Sampler Node
        sid, sdata = find_node_by_class(graph, "RadianceSamplerPro")
        self.sampler_node_id = sid
        if sid:
            sinputs = sdata.get("inputs", {})
            self.steps_knob.setValue(sinputs.get("steps", 20))
            self.cfg_knob.setValue(sinputs.get("cfg", 1.0))
            self.denoise_knob.setValue(sinputs.get("denoise", 1.0))

            sname = sinputs.get("sampler")
            if sname in self.samplers:
                self.sampler_knob.setValue(sname)

            schname = sinputs.get("scheduler")
            if schname in self.schedulers:
                self.scheduler_knob.setValue(schname)
            
            smode = sinputs.get("sampler_mode", "Standard")
            if smode in self.sampler_modes:
                self.sampler_mode_knob.setValue(smode)
            
            self.guidance_knob.setValue(sinputs.get("flux_guidance", 3.5))
            self.mu_knob.setValue(sinputs.get("flux_shift", 1.0))

            print(f"Synced with Sampler Node {sid}")

        # 4. Sync Loader / VAE
        lid, ldata = find_node_by_class(graph, "RadianceUnifiedLoader")
        if lid:
            linputs = ldata.get("inputs", {})
            vname = linputs.get("vae_name")
            if vname in self.vaes:
                self.vae_knob.setValue(vname)
            print(f"Synced with Loader Node {lid}")

        # 5. Sync Depth Node
        did, ddata = find_node_by_class(graph, "RadianceDepthMapGenerator")
        self.depth_node_id = did
        if did:
            dinputs = ddata.get("inputs", {})
            self.depth_model_knob.setValue(
                dinputs.get("model_size", "Base (98M - Balanced)")
            )
            self.depth_blur_knob.setValue(dinputs.get("blur_edges", 0.0))
            self.depth_invert_knob.setValue(dinputs.get("invert", False))
            self.depth_label.setValue("Depth Pass Status: Active")
            print(f"Synced with Depth Node {did}")
        else:
            self.depth_label.setValue("Depth Pass Status: Not Found")

        # Update visibility states dynamically
        self.update_flux_visibility()

    def generate(self):
        if not self.last_graph:
            nuke.message("No graph loaded. Sync first.")
            return

        # Update Prompt / Camera
        if self.node_id:
            inputs = self.last_graph[self.node_id]["inputs"]
            inputs["base_prompt"] = self.prompt_knob.value()

            if self.node_type == "ENCODER":
                inputs["camera_type"] = self.camera_knob.value()
                inputs["lens_focal"] = self.lens_knob.value()
                inputs["aperture_dof"] = self.aperture_knob.value()
                inputs["framing"] = self.framing_knob.value()
            elif self.node_type == "STUDIO":
                inputs["camera"] = self.camera_knob.value()
                val = self.lens_knob.value()
                if "mm" in val:
                    inputs["focal_length"] = val
                else:
                    inputs["lens_series"] = val
                inputs["aperture"] = self.aperture_knob.value()

        # Update Resolution
        if self.res_node_id:
            r_inputs = self.last_graph[self.res_node_id]["inputs"]
            r_inputs["preset"] = self.res_knob.value()

        # Update Sampler
        if self.sampler_node_id:
            s_inputs = self.last_graph[self.sampler_node_id]["inputs"]
            s_inputs["steps"] = int(self.steps_knob.value())
            s_inputs["cfg"] = self.cfg_knob.value()
            s_inputs["denoise"] = self.denoise_knob.value()
            s_inputs["sampler"] = self.sampler_knob.value()
            s_inputs["scheduler"] = self.scheduler_knob.value()
            s_inputs["sampler_mode"] = self.sampler_mode_knob.value()
            s_inputs["flux_guidance"] = self.guidance_knob.value()
            s_inputs["flux_shift"] = self.mu_knob.value()

        # Update VAE in Loader
        lid, ldata = find_node_by_class(self.last_graph, "RadianceUnifiedLoader")
        if lid:
            l_inputs = self.last_graph[lid]["inputs"]
            l_inputs["vae_name"] = self.vae_knob.value()

        # Update Depth
        if self.depth_node_id:
            d_inputs = self.last_graph[self.depth_node_id]["inputs"]
            d_inputs["model_size"] = self.depth_model_knob.value()
            d_inputs["blur_edges"] = self.depth_blur_knob.value()
            d_inputs["invert"] = self.depth_invert_knob.value()

        # Randomize Seed
        randomize_seed(self.last_graph)

        # Submit
        submit_prompt(self.last_graph)


def show_radiance_panel():
    p = RadiancePromptPanel()
    p.show()


# --- Menu ---


def add_radiance_menu():
    radiance_menu = nuke.menu("Nuke").addMenu("Radiance")
    radiance_menu.addCommand("Start Bridge", start_bridge)
    radiance_menu.addCommand("Stop Bridge", stop_bridge)
    radiance_menu.addCommand("Queue Last Run", queue_last_run, "F5")
    radiance_menu.addCommand("Cinematic Encoder", show_radiance_panel, "Shift+F5")


if nuke.GUI:
    add_radiance_menu()
    start_bridge()
    print("Radiance Bridge started and menu installed.")
