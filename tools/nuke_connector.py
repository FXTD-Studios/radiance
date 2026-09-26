from __future__ import annotations

import re
import socket
import struct
import json
import logging
import os
from typing import Dict, Any, Tuple

logger = logging.getLogger("radiance.nuke.connector")

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 1986
BUFFER_SIZE = 65536
CONNECT_TIMEOUT = 5.0
READ_TIMEOUT = 15.0

# Wire protocol
HEADER_MAGIC = b"RCMD"  # 4-byte magic identifier
HEADER_VERSION = 1  # Protocol version
END_MARKER = b"\n__RADIANCE_END__\n"  # Response terminator

# Security: regex pattern for safe Nuke identifiers (node names, stream names)
_SAFE_IDENTIFIER_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")
# Security: characters allowed in Nuke strings (paths, color space names)
_SAFE_STRING_RE = re.compile(r"^[a-zA-Z0-9_\-./: ()#+]+$")


def _sanitize_nuke_string(value: str, context: str = "value") -> str:
    """
    Sanitize a string before embedding it in a Nuke Python command.

    Removes/escapes characters that could break out of a Python string literal
    and inject arbitrary code. This is a defense-in-depth measure.

    Args:
        value: The string to sanitize
        context: Description for error messages (e.g., 'node_name', 'filepath')

    Returns:
        Sanitized string safe for embedding in single-quoted Python literals
    """
    # Remove null bytes
    value = value.replace("\x00", "")
    # Escape backslashes first, then single quotes
    value = value.replace("\\", "/")
    value = value.replace("'", "")
    value = value.replace('"', "")
    # Remove newlines and other control characters that could break commands
    value = re.sub(r"[\n\r\t]", "", value)
    # Strip leading/trailing whitespace
    value = value.strip()
    return value


def nuke_node_name(name: str, fallback: str = "RadianceStream") -> str:
    """A valid Nuke node name made from `name`: letters, digits and
    underscores, not starting with a digit. Nuke refuses anything else (a
    hyphen included), and the file name a user types is often not one."""
    cleaned = re.sub(r"[^A-Za-z0-9_]", "_", (name or "").strip())[:128].strip("_")
    if not cleaned:
        return fallback
    if cleaned[0].isdigit():
        cleaned = "R_" + cleaned
    return cleaned


def validate_nuke_identifier(name: str, context: str = "identifier") -> str:
    """
    Validate that a string is a safe Nuke identifier (node name, stream name).

    Only allows: a-z, A-Z, 0-9, underscore, hyphen.

    Raises:
        ValueError: If the name contains invalid characters
    """
    name = name.strip()
    if not name:
        raise ValueError(f"Empty {context} is not allowed")
    if not _SAFE_IDENTIFIER_RE.match(name):
        raise ValueError(
            f"Invalid {context}: '{name}'. "
            f"Only letters, digits, underscore, and hyphen are allowed."
        )
    if len(name) > 128:
        raise ValueError(f"{context} too long (max 128 chars): '{name[:32]}...'")
    return name


class NukeConnector:
    CATEGORY = "FXTD STUDIOS/Radiance/◎ Pipeline"
    """
    TCP client that sends Python commands to Nuke's Radiance listener.

    Every public method returns (success: bool, result: str|dict).
    No exceptions are raised to the caller — all errors are captured
    and returned as (False, "ERROR: ...") tuples.

    Usage:
        conn = NukeConnector("127.0.0.1", 1986)
        ok, msg = conn.ping()
        if ok:
            conn.load_exr("/renders/stream.0001.exr", "RadianceStream")
    """

    def __init__(self, host: str = None, port: int = None):
        self.host = host if host is not None else os.environ.get("RADIANCE_NUKE_HOST", DEFAULT_HOST)
        self.port = port if port is not None else int(os.environ.get("RADIANCE_NUKE_PORT", str(DEFAULT_PORT)))
        self._last_error = ""

    @property
    def last_error(self) -> str:
        return self._last_error

    # ═════════════════════════════════════════════════════════════════════
    #  LOW-LEVEL: send raw Python command to Nuke
    # ═════════════════════════════════════════════════════════════════════

    def send_command(
        self, command: str, timeout: float = READ_TIMEOUT
    ) -> Tuple[bool, str]:
        """
        Send a Python command string to Nuke and return (success, result).

        The command is executed inside Nuke's script context via
        nuke.executeInMainThread() for thread safety.
        """
        sock = None
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(CONNECT_TIMEOUT)
            sock.connect((self.host, self.port))
            sock.settimeout(timeout)

            # The shared token: RADIANCE_DCC_AUTH_TOKEN, or ~/.radiance/dcc_token,
            # which the Nuke listener reads too (radiance.core.dcc_auth). The
            # listener refuses unsigned commands, so without a token nothing
            # reached Nuke unless both environments set the variable.
            try:
                from radiance.core.dcc_auth import load_or_create_token
                token = load_or_create_token()
            except ImportError:
                token = os.environ.get("RADIANCE_DCC_AUTH_TOKEN", "")

            payload = command.encode("utf-8")
            if token:
                # Protocol Version 2: MAGIC(4) + VERSION(1) + SIGNATURE(32) + LENGTH(4 LE) + PAYLOAD
                # HMAC rather than SHA256(token || command) -- the bare-hash form
                # was length-extendable and replayable. Must match the server.
                import hmac
                import hashlib
                sig = hmac.new(
                    token.encode("utf-8"), command.encode("utf-8"), hashlib.sha256
                ).digest()
                header = HEADER_MAGIC + struct.pack("<B", 2) + sig + struct.pack("<I", len(payload))
            else:
                # Protocol Version 1 (unauthenticated). The server now refuses
                # these; kept only so the failure is a clear server-side message.
                header = HEADER_MAGIC + struct.pack("<BI", 1, len(payload))

            sock.sendall(header + payload)

            # Read response chunks until END_MARKER or disconnect
            chunks = []
            accumulated = b""
            while True:
                try:
                    data = sock.recv(BUFFER_SIZE)
                    if not data:
                        break
                    chunks.append(data)
                    accumulated = b"".join(chunks)
                    if END_MARKER in accumulated:
                        break
                except socket.timeout:
                    break

            response = accumulated.decode("utf-8", errors="replace")
            response = response.replace(END_MARKER.decode(), "").strip()

            if response.startswith("ERROR:"):
                self._last_error = response
                return (False, response)

            return (True, response)

        except ConnectionRefusedError:
            # Generate absolute, forward-slashed path for Nuke copy-paste
            try:
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                script_path = os.path.join(base_dir, "scripts", "start_nuke_server.py").replace("\\", "/")
                cmd = f"exec(open('{script_path}').read())"
            except Exception:
                cmd = "exec(open('scripts/start_nuke_server.py').read())"

            msg = (
                f"Nuke not reachable at {self.host}:{self.port}. "
                "Ensure the Radiance listener is running in Nuke:\n"
                f"  Script Editor → run: {cmd}"
            )
            self._last_error = msg
            logger.warning(msg)
            return (False, f"CONNECTION_REFUSED: {msg}")

        except socket.timeout:
            msg = f"Timeout connecting to {self.host}:{self.port}"
            self._last_error = msg
            logger.warning(msg)
            return (False, f"TIMEOUT: {msg}")

        except OSError as e:
            msg = f"Network error: {e}"
            self._last_error = msg
            logger.warning(msg)
            return (False, f"OS_ERROR: {msg}")

        finally:
            if sock:
                try:
                    sock.close()
                except Exception:  # nosec B110
                    pass

    def send_action(
        self,
        action: str,
        payload: Dict[str, Any] | None = None,
        timeout: float = READ_TIMEOUT,
    ) -> Tuple[bool, str]:
        """Send a structured command understood by the Radiance Nuke bridge.

        Structured commands avoid arbitrary Python execution on the Nuke side and
        keep production bridge operations available while RADIANCE_DEV is off.
        """

        body = {"action": action, **(payload or {})}
        return self.send_command(json.dumps(body), timeout=timeout)

    # ═════════════════════════════════════════════════════════════════════
    #  HIGH-LEVEL: Nuke operations
    # ═════════════════════════════════════════════════════════════════════

    def ping(self) -> Tuple[bool, str]:
        """Check if a compatible Nuke listener is alive. Fast 3-second timeout."""
        ok, result = self.send_action("ping", timeout=3.0)
        return (ok and "PONG" in result, result)

    def load_exr(
        self,
        filepath: str,
        node_name: str = "RadianceStream",
        first_frame: int = 1,
        last_frame: int = 1,
        current_frame: int = 1,
        color_space: str = "linear",
        connect_viewer: bool = True,
        raw: bool = True,
    ) -> Tuple[bool, str]:
        """
        Create or update a Read node in Nuke pointing to the EXR,
        and optionally connect it to the Viewer.

        This is the replacement for the broken sync_image() method.

        Args:
            filepath:       EXR path. Use #### for sequences: stream.####.exr
            node_name:      Nuke node name for the Read (reused across updates)
            first_frame:    First frame of sequence
            last_frame:     Last frame of sequence
            current_frame:  Frame to display in Viewer
            color_space:    Nuke colorspace knob value ('linear', 'sRGB', etc.)
            connect_viewer: If True, wire Read → Viewer1 input 0
            raw:            If True, bypass Nuke's internal color management
        """
        # The payload is JSON handled by a whitelisted action (no code is
        # built from it), so the path is sent as it is apart from separators;
        # stripping quotes, as before, pointed Read nodes at paths that do not
        # exist. The node name is made valid for Nuke instead of raising.
        safe_path = str(filepath).replace("\\", "/").replace("\x00", "").strip()
        safe_node_name = nuke_node_name(node_name)
        safe_color_space = _sanitize_nuke_string(color_space, "color_space")

        # Validate numeric parameters
        first_frame = int(first_frame)
        last_frame = int(last_frame)
        current_frame = int(current_frame)

        return self.send_action("load_exr", {
            "filepath": safe_path,
            "node_name": safe_node_name,
            "first_frame": first_frame,
            "last_frame": last_frame,
            "current_frame": current_frame,
            "color_space": safe_color_space,
            "connect_viewer": bool(connect_viewer),
            "raw": bool(raw),
        })

    def set_frame(self, frame: int) -> Tuple[bool, str]:
        """Set Nuke's current frame."""
        frame = int(frame)  # Ensure integer — no injection
        return self.send_action("set_frame", {"frame": frame})

    def get_info(self) -> Tuple[bool, Dict[str, Any]]:
        """Query Nuke for version, project, format, frame range, fps."""
        ok, result = self.send_action("get_info")
        if ok:
            try:
                return (True, json.loads(result))
            except (json.JSONDecodeError, TypeError):
                return (True, {"raw": result})
        return (False, {"error": result})
