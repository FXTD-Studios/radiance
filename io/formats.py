import numpy as np
import struct
import zlib
import os
import logging
from typing import Tuple, Dict, Any, Optional

logger = logging.getLogger("radiance.io.formats")

# v1.21: cv2 for 16-bit PNG support
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    # Bind the name, not just the flag. `viewer_utils.py` does
    # `from radiance.io.formats import HAS_CV2, cv2` — with `cv2` left
    # undefined on this branch that import raised "cannot import name 'cv2'
    # from 'radiance.io.formats'", so the guard protected nothing: every
    # consumer of this module died anyway, just with a more confusing error
    # than the ImportError it was written to absorb.
    cv2 = None
    HAS_CV2 = False
    logger.warning(
        "cv2 not available — 16-bit PNG disabled, falling back to 8-bit. "
        "Install opencv-python for 16-bit support."
    )

PICK_MAX_DIM = 256
RPICK_MAGIC = b"RPIC"
CV2_PNG_COMPRESSION = 4


def _save_pick_buffer(
    frame: np.ndarray,
    filepath: str,
    max_dim: int = PICK_MAX_DIM,
) -> bool:
    """Save a scene-linear fp32 picking buffer (.rpick) for the color picker."""
    try:
        h, w = frame.shape[:2]
        c = frame.shape[2] if frame.ndim == 3 else 1
        rgb = frame[..., :3] if c >= 3 else frame

        flags = 1  # bit0 = fp32
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            if HAS_CV2:
                rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)
            else:
                from PIL import Image as PILImage
                pil = PILImage.fromarray(
                    np.clip(rgb, 0.0, 1.0).astype(np.float32), mode="RGB"
                    if c >= 3 else "L"
                )
                pil = pil.resize((new_w, new_h), PILImage.LANCZOS)
                rgb = np.array(pil, dtype=np.float32)
            h, w = rgb.shape[:2]
            c = rgb.shape[2] if rgb.ndim == 3 else 1
            flags |= 2  # bit1 = downsampled

        buf = rgb.astype(np.float32).tobytes()
        compressed = zlib.compress(buf, level=3)
        header = struct.pack("<4sHHHH", RPICK_MAGIC, w, h, c, flags)
        with open(filepath, "wb") as f:
            f.write(header)
            f.write(compressed)
        return True
    except Exception as e:
        logger.debug(f"[Radiance v3.0.0] pick buffer save failed: {e}")
        return False


#: The ASC CDL XML namespace. ASC CDL v1.2 still uses the v1.01 schema URN;
#: "urn:ASC:CDL:v1.2" is not a published namespace.
ASC_CDL_NAMESPACE = "urn:ASC:CDL:v1.01"


def _cdl_triplet(name: str, values, *, positive: bool = False, non_negative: bool = False):
    import math
    vals = [float(v) for v in (values if isinstance(values, (list, tuple)) else [values] * 3)]
    if len(vals) != 3 or not all(math.isfinite(v) for v in vals):
        raise ValueError(f"ASC CDL {name} must be three finite numbers, got {values!r}.")
    if positive and any(v <= 0 for v in vals):
        raise ValueError(f"ASC CDL {name} must be > 0, got {vals}.")
    if non_negative and any(v < 0 for v in vals):
        raise ValueError(f"ASC CDL {name} must be >= 0, got {vals}.")
    return vals


def build_cdl_xml(
    slope: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    power: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    saturation: float = 1.0,
    description: str = "Radiance Viewer Grade",
    cc_id: str = "radiance_grade",
    container: str = "cdl",
) -> str:
    """The one ASC CDL writer for Radiance (FIX-006).

    ``container`` picks the document type by file extension:

    * ``"cdl"``: ``<ColorDecisionList><ColorDecision><ColorCorrection>``
    * ``"cc"``:  a single ``<ColorCorrection>``
    * ``"ccc"``: ``<ColorCorrectionCollection><ColorCorrection>``

    Elements follow the ASC CDL v1.2 schema (SOPNode Slope/Offset/Power,
    SatNode Saturation) in the ``urn:ASC:CDL:v1.01`` namespace, which is what
    OCIO, Nuke and Resolve read. Values are validated: slope and saturation
    >= 0, power > 0, everything finite. The id is ASCII so every reader can
    select it.
    """
    import math
    import re
    import xml.etree.ElementTree as _ET

    s = _cdl_triplet("Slope", slope, non_negative=True)
    o = _cdl_triplet("Offset", offset)
    p = _cdl_triplet("Power", power, positive=True)
    sat = float(saturation)
    if not math.isfinite(sat) or sat < 0:
        raise ValueError(f"ASC CDL Saturation must be a finite number >= 0, got {saturation!r}.")
    container = container.lower().lstrip(".")
    if container not in ("cdl", "cc", "ccc"):
        raise ValueError(f"ASC CDL container must be cdl, cc or ccc, got {container!r}.")
    cc_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", cc_id).strip("_") or "radiance_grade"

    def _fmt(vals):
        return " ".join(f"{v:.6f}" for v in vals)

    def _cc(parent=None):
        attrs = {"id": cc_id}
        cc = _ET.SubElement(parent, "ColorCorrection", attrs) if parent is not None \
            else _ET.Element("ColorCorrection", dict(attrs, xmlns=ASC_CDL_NAMESPACE))
        sop = _ET.SubElement(cc, "SOPNode")
        if description:
            _ET.SubElement(sop, "Description").text = description
        _ET.SubElement(sop, "Slope").text = _fmt(s)
        _ET.SubElement(sop, "Offset").text = _fmt(o)
        _ET.SubElement(sop, "Power").text = _fmt(p)
        sat_node = _ET.SubElement(cc, "SatNode")
        _ET.SubElement(sat_node, "Saturation").text = f"{sat:.6f}"
        return cc

    if container == "cc":
        root = _cc()
    elif container == "ccc":
        root = _ET.Element("ColorCorrectionCollection", {"xmlns": ASC_CDL_NAMESPACE})
        _cc(root)
    else:
        root = _ET.Element("ColorDecisionList", {"xmlns": ASC_CDL_NAMESPACE})
        _cc(_ET.SubElement(root, "ColorDecision"))
    _ET.indent(root, space="  ")
    return '<?xml version="1.0" encoding="UTF-8"?>\n' + _ET.tostring(root, encoding="unicode") + "\n"


def write_cdl_file(path: str, slope, offset, power, saturation, description: str = "Radiance grade",
                   cc_id: str = "radiance_grade") -> str:
    """Write an ASC CDL file; the extension (.cdl, .cc, .ccc) picks the container."""
    ext = os.path.splitext(path)[1].lower().lstrip(".")
    if ext not in ("cdl", "cc", "ccc"):
        raise ValueError(f"ASC CDL path must end in .cdl, .cc or .ccc: {path!r}")
    xml = build_cdl_xml(slope, offset, power, saturation, description=description,
                        cc_id=cc_id, container=ext)
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(xml)
    return path


def save_16bit_png(filepath: str, img_uint16: np.ndarray) -> bool:
    """Save a uint16 numpy array as a 16-bit PNG using OpenCV."""
    if not HAS_CV2:
        logger.error("cv2 required for 16-bit PNG save but not available")
        return False

    try:
        if img_uint16.ndim == 3 and img_uint16.shape[2] == 3:
            bgr = cv2.cvtColor(img_uint16, cv2.COLOR_RGB2BGR)
            return cv2.imwrite(
                filepath, bgr, [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        elif img_uint16.ndim == 3 and img_uint16.shape[2] == 4:
            bgra = cv2.cvtColor(img_uint16, cv2.COLOR_RGBA2BGRA)
            return cv2.imwrite(
                filepath, bgra, [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        elif img_uint16.ndim == 3 and img_uint16.shape[2] == 1:
            return cv2.imwrite(
                filepath,
                img_uint16[:, :, 0],
                [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION],
            )
        elif img_uint16.ndim == 2:
            return cv2.imwrite(
                filepath, img_uint16, [cv2.IMWRITE_PNG_COMPRESSION, CV2_PNG_COMPRESSION]
            )
        else:
            logger.warning(f"Unsupported shape for 16-bit save: {img_uint16.shape}")
            return False
    except Exception as e:
        logger.error(f"cv2 16-bit PNG save failed: {e}")
        return False
