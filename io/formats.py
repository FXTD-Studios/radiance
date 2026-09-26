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


def build_cdl_xml(
    slope: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    offset: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    power: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    saturation: float = 1.0,
    description: str = "◎ Radiance Viewer Grade",
) -> str:
    """Build an ASC CDL XML string compatible with Nuke, DaVinci Resolve, OCIO."""
    s = " ".join(f"{v:.6f}" for v in slope)
    o = " ".join(f"{v:.6f}" for v in offset)
    p = " ".join(f"{v:.6f}" for v in power)
    return (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<ColorDecisionList xmlns="urn:ASC:CDL:v1.2">\n'
        '  <ColorDecision>\n'
        f'    <!-- {description} -->\n'
        '    <ColorCorrection id="◎ Radiance_grade">\n'
        '      <SOPNode>\n'
        f'        <Slope>{s}</Slope>\n'
        f'        <Offset>{o}</Offset>\n'
        f'        <Power>{p}</Power>\n'
        '      </SOPNode>\n'
        '      <SatNode>\n'
        f'        <Saturation>{saturation:.6f}</Saturation>\n'
        '      </SatNode>\n'
        '      </ColorCorrection>\n'
        '  </ColorDecision>\n'
        '</ColorDecisionList>\n'
    )


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
