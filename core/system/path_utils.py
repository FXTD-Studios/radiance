"""Safe filesystem path resolution — prevents directory traversal attacks."""
from __future__ import annotations

import os
import re


# ALBABIT-FIX: restored from radiance.disabled. Windows "Copy as path"
# (Shift+Right-click) wraps paths in double-quotes; users also sometimes
# paste single-quoted paths. Strip both so path widgets accept either form.
def strip_path_quotes(path: str) -> str:
    return path.strip().strip('"').strip("'")


def safe_join(base: str, *paths: str) -> str:
    base = os.path.normpath(os.path.abspath(base))
    full_path = os.path.normpath(os.path.abspath(os.path.join(base, *paths)))
    base_with_sep = base if base.endswith(os.sep) else base + os.sep

    if not (full_path.startswith(base_with_sep) or full_path == base):
        raise ValueError(
            f"Path traversal detected: '{os.path.join(*paths)}' "
            f"escapes base directory '{base}'"
        )
    return full_path


def validate_output_path(
    base_dir: str, subfolder: str, filename: str, allow_absolute: bool = False
) -> str:
    if subfolder and os.path.isabs(subfolder):
        if allow_absolute:
            return os.path.normpath(os.path.join(subfolder, filename))
        raise ValueError(
            f"Absolute subfolder paths not allowed for security: '{subfolder}'. "
            f"Use relative paths only."
        )
    if subfolder:
        return safe_join(base_dir, subfolder, filename)
    return safe_join(base_dir, filename)


def get_safe_output_dir(base_dir: str, subfolder: str = "", allow_absolute: bool = False) -> str:
    if subfolder and os.path.isabs(subfolder):
        if allow_absolute:
            output_dir = os.path.normpath(subfolder)
            os.makedirs(output_dir, exist_ok=True)
            return output_dir
        raise ValueError(
            f"Absolute subfolder paths not allowed for security: '{subfolder}'. "
            f"Use relative paths only."
        )
    output_dir = (
        safe_join(base_dir, subfolder) if subfolder else os.path.abspath(base_dir)
    )
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_safe_input_path(base_dir: str, filename: str, allow_absolute: bool = False) -> str:
    if os.path.isabs(filename):
        if allow_absolute:
            return os.path.normpath(filename)
        raise ValueError(
            f"Absolute input paths are not permitted by default: '{filename}'. "
            f"Pass allow_absolute=True to explicitly allow unrestricted paths."
        )
    base_dir = os.path.normpath(os.path.abspath(base_dir))
    return safe_join(base_dir, filename)


def _comfy_dir(kind: str) -> str:
    """ComfyUI's output/input directory, or a temp dir when outside ComfyUI.

    Imported lazily: `folder_paths` only exists when running inside ComfyUI,
    and these helpers are reachable from tests and standalone tools.
    """
    try:
        import folder_paths  # type: ignore
    except ImportError:
        import tempfile
        return tempfile.gettempdir()

    getter = getattr(folder_paths, f"get_{kind}_directory", None)
    if getter is None:
        import tempfile
        return tempfile.gettempdir()
    return getter()


def resolve_output_path(path: str) -> str:
    """Anchor a user-supplied output path to ComfyUI's output directory.

    A node whose path widget defaults to something like "grading/shot.cdl" was
    being resolved with `os.path.abspath()`, which anchors to the *process
    working directory* — for ComfyUI that is the install root. Running such a
    node on its default therefore scattered `grading/` and `preview/`
    directories through the ComfyUI installation, and through this repository
    whenever the test suite exercised those nodes.

    Absolute paths are honoured untouched — someone who types a full path means
    it. Relative paths land under `output/`, with `..` traversal rejected.
    """
    path = strip_path_quotes(path)
    if not path:
        return _comfy_dir("output")
    if os.path.isabs(path):
        return os.path.normpath(path)
    return safe_join(_comfy_dir("output"), path)


def resolve_input_path(path: str) -> str:
    """Resolve a user-supplied input path for reading.

    Absolute paths pass through. A relative path is looked for under ComfyUI's
    `input/` and then `output/` — the second because Radiance's own exporters
    (CDL, flipbooks, sidecars) write to `output/`, so "read back what I just
    wrote" is the common case. Falls back to the working directory so an
    explicitly relative invocation from a shell still resolves.
    """
    path = strip_path_quotes(path)
    if not path:
        return path
    if os.path.isabs(path):
        return os.path.normpath(path)

    for base in (_comfy_dir("input"), _comfy_dir("output")):
        try:
            candidate = safe_join(base, path)
        except ValueError:
            continue
        if os.path.isfile(candidate):
            return candidate

    if os.path.isfile(path):
        return os.path.abspath(path)

    # Nothing exists yet — hand back the input-dir candidate so the caller's
    # "file not found" message names the directory users are meant to look in.
    try:
        return safe_join(_comfy_dir("input"), path)
    except ValueError:
        return os.path.abspath(path)


def get_next_index(directory: str, prefix: str, extension: str) -> int:
    if not os.path.isdir(directory):
        return 0

    max_idx = -1
    pattern = re.compile(rf"^{re.escape(prefix)}(\d+){re.escape(extension)}$")

    try:
        for f in os.listdir(directory):
            match = pattern.match(f)
            if match:
                try:
                    idx = int(match.group(1))
                    if idx > max_idx:
                        max_idx = idx
                except ValueError:
                    continue
    except Exception:
        return 0
    return max_idx + 1
