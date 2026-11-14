"""ComfyUI-Manager runs this after installing (or updating) Radiance.

It makes Radiance work with nothing to configure:

* the libraries the colour pipeline and SDR -> HDR are built on are present
  and new enough: OpenEXR (EXR read/write), OpenColorIO 2.5+ (colour
  management, the ACES 2.0 view RUDRA is built on), diffusers 0.35+ (Wan VACE
  for the RUDRA AI fill, Marigold for Multipass) and accelerate;
* RUDRA ACES 2, the learned SDR -> HDR model (2.2 MB, Apache 2.0), is fetched
  into ComfyUI/models/radiance/RUDRA_ACES_2, pinned and hash-checked, so SDR ->
  HDR Universal and Recover run their learned path offline from the first
  queue. The AI fill's larger models download the first time that node runs.

At startup Radiance uses your $OCIO if you have one, otherwise OpenColorIO's
built-in ACES config. Set RADIANCE_ALLOW_DOWNLOADS=0 to skip every download.
"""
import importlib.util
import os
import subprocess
import sys
from pathlib import Path

REQUIRED = [
    # (import name, distribution, minimum version, pip requirement)
    ("OpenEXR", "OpenEXR", "3.2.0", "OpenEXR>=3.2.0,<4.0.0"),
    ("PyOpenColorIO", "opencolorio", "2.5.0", "opencolorio>=2.5.0,<3.0.0"),
    ("diffusers", "diffusers", "0.35.0", "diffusers>=0.35.0"),
    ("accelerate", "accelerate", "0.26.0", "accelerate>=0.26.0"),
]


def _version_tuple(v: str):
    out = []
    for part in v.split("."):
        digits = "".join(ch for ch in part if ch.isdigit())
        if not digits:
            break
        out.append(int(digits))
    return tuple(out)


def _too_old(dist: str, minimum: str) -> bool:
    try:
        from importlib.metadata import version
        return _version_tuple(version(dist)) < _version_tuple(minimum)
    except Exception:  # noqa: BLE001 - unknown version: leave it alone
        return False


def _missing():
    todo = [pip for mod, dist, minimum, pip in REQUIRED
            if importlib.util.find_spec(mod) is None or _too_old(dist, minimum)]
    if sys.version_info >= (3, 14):
        # No OpenEXR wheel for 3.14 yet; a source build fails. Radiance reads
        # and writes EXR through OpenImageIO (then OpenCV) without it.
        todo = [t for t in todo if not t.startswith("OpenEXR")]
    return todo


def _downloads_off() -> bool:
    def flag(name):
        return os.environ.get(name, "").strip().lower() in ("1", "true", "yes", "on")
    allow = os.environ.get("RADIANCE_ALLOW_DOWNLOADS", "").strip().lower()
    return allow in ("0", "false", "no", "off") or flag("HF_HUB_OFFLINE") or flag("TRANSFORMERS_OFFLINE")


def _prefetch_rudra() -> None:
    """RUDRA ACES 2 into <ComfyUI>/models/radiance (best effort; the nodes fetch it otherwise)."""
    if _downloads_off():
        print("[Radiance] downloads are off; RUDRA ACES 2 will not be fetched.")
        return
    here = Path(__file__).resolve().parent
    target = here.parents[1] / "models" / "radiance"          # custom_nodes/radiance -> ComfyUI/models
    if not (here.parent.name == "custom_nodes" and target.parent.is_dir()):
        return
    spec = importlib.util.spec_from_file_location("_radiance_rudra_download",
                                                  here / "model" / "rudra_download.py")
    try:
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod                          # dataclasses look the module up
        spec.loader.exec_module(mod)                          # standard library only
        folder = target / mod.ACES2.dirname
        if mod.package_complete(folder, mod.ACES2):
            print(f"[Radiance] RUDRA ACES 2 present in {folder}")
            return
        mod.download_to(mod.ACES2, folder)
    except Exception as exc:  # noqa: BLE001 - never fail the install over a model
        print(f"[Radiance] RUDRA ACES 2 not fetched now ({exc}); it downloads on first use.")


def main() -> int:
    rc = 0
    todo = _missing()
    if not todo:
        print("[Radiance] Required libraries present; OCIO is configured automatically at startup.")
    else:
        print(f"[Radiance] installing {', '.join(todo)} ...")
        rc = subprocess.call([sys.executable, "-m", "pip", "install", "-U", *todo])
        if rc != 0:
            print(f"[Radiance] pip failed ({rc}). Install manually: {sys.executable} -m pip install -U "
                  + " ".join(f'"{t}"' for t in todo))
    _prefetch_rudra()
    return rc


if __name__ == "__main__":
    sys.exit(main())
