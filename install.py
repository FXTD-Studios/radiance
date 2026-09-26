"""ComfyUI-Manager runs this after installing Radiance.

It makes sure the two libraries Radiance's colour pipeline is built on are
present: OpenEXR (EXR read/write) and OpenColorIO (colour management), plus
diffusers and accelerate for Multipass Estimate's Marigold models. Once
they are, there is nothing to configure: at startup Radiance uses your $OCIO
if you have one, otherwise OpenColorIO's built-in ACES studio config, and
OpenCV's EXR codec is switched on (OPENCV_IO_ENABLE_OPENEXR=1).
"""
import importlib.util
import subprocess
import sys

REQUIRED = [
    # (import name, pip requirement)
    ("OpenEXR", "OpenEXR>=3.2.0,<4.0.0"),
    ("PyOpenColorIO", "opencolorio>=2.3.0,<3.0.0"),
    ("diffusers", "diffusers>=0.33.0"),
    ("accelerate", "accelerate>=0.26.0"),
]


def _missing():
    todo = [pip for mod, pip in REQUIRED if importlib.util.find_spec(mod) is None]
    if sys.version_info >= (3, 14):
        # No OpenEXR wheel for 3.14 yet; a source build fails. Radiance reads
        # and writes EXR through OpenImageIO (then OpenCV) without it.
        todo = [t for t in todo if not t.startswith("OpenEXR")]
    return todo


def main() -> int:
    todo = _missing()
    if not todo:
        print("[Radiance] Required libraries present; OCIO is configured automatically at startup.")
        return 0
    print(f"[Radiance] installing {', '.join(todo)} ...")
    rc = subprocess.call([sys.executable, "-m", "pip", "install", *todo])
    if rc != 0:
        print(f"[Radiance] pip failed ({rc}). Install manually: {sys.executable} -m pip install "
              + " ".join(f'"{t}"' for t in todo))
    return rc


if __name__ == "__main__":
    sys.exit(main())
