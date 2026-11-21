"""What a still says about its own colour and orientation (RUDRA 1.1, roadmap 1.1).

Ported from FXTD-Studios/RUDRA ``native/media/src/still_metadata.cpp``, commit
74f03b6 (Apache 2.0). OpenCV's IMREAD_UNCHANGED keeps every code but reads no
metadata, so a Display P3 phone photo was mastered as Rec.709 and a portrait
one came out sideways. This reads, from the file's bytes and without a colour
library:

* the embedded ICC profile (PNG iCCP, JPEG APP2, TIFF tag 34675, WebP ICCP):
  its rXYZ / gXYZ / bXYZ against the D50-adapted primaries of sRGB / Rec.709,
  Display P3, Rec.2020 and Adobe RGB (1998), and its description;
* PNG cICP (colour primaries 1, 9, 12), sRGB and cHRM chunks;
* the EXIF orientation (JPEG APP1, PNG eXIf, TIFF IFD0, WebP EXIF).

An untagged file is Rec.709. A profile that matches none of the four is
reported (``recognised`` False) and read as Rec.709: never a silent
conversion, never a silent guess.
"""
from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

PRIMARIES = ("rec709", "p3d65", "rec2020", "adobe_rgb")
PRIMARIES_NAME = {"rec709": "Rec.709 / sRGB", "p3d65": "Display P3", "rec2020": "Rec.2020",
                  "adobe_rgb": "Adobe RGB (1998)"}
# xy of R, G, B and the white (all D65)
CHROMATICITIES = {
    "rec709": (0.64, 0.33, 0.30, 0.60, 0.15, 0.06, 0.3127, 0.3290),
    "p3d65": (0.680, 0.320, 0.265, 0.690, 0.150, 0.060, 0.3127, 0.3290),
    "rec2020": (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290),
    "adobe_rgb": (0.64, 0.33, 0.21, 0.71, 0.15, 0.06, 0.3127, 0.3290),
}
# D50-adapted (Bradford) colorants as ICC v2/v4 profiles store them: rXYZ, gXYZ, bXYZ
_ICC_KNOWN = {
    "rec709": (0.4360, 0.2225, 0.0139, 0.3851, 0.7169, 0.0971, 0.1430, 0.0606, 0.7139),
    "p3d65": (0.5151, 0.2412, -0.0011, 0.2920, 0.6922, 0.0419, 0.1571, 0.0666, 0.7841),
    "rec2020": (0.6735, 0.2790, -0.0019, 0.1657, 0.6753, 0.0300, 0.1250, 0.0456, 0.7969),
    "adobe_rgb": (0.6097, 0.3111, 0.0195, 0.2053, 0.6257, 0.0609, 0.1492, 0.0632, 0.7446),
}


@dataclass
class StillColour:
    primaries: str = "rec709"       # what the master converts from
    tagged: bool = False            # the file said something about its colour
    recognised: bool = True         # ... and it was one of the four
    tag: str = ""                   # "icc", "cicp", "srgb", "chrm" or ""
    profile: str = ""               # the ICC description, when there is one
    orientation: int = 1            # EXIF 1..8, 1 = as stored

    def note(self) -> str:
        name = PRIMARIES_NAME[self.primaries]
        if not self.tagged:
            s = "no colour profile: read as Rec.709 / sRGB"
        elif not self.recognised:
            s = (f"colour profile '{self.profile or 'unnamed'}' is not one RUDRA knows: read as Rec.709 / sRGB, "
                 "check the colours")
        else:
            s = f"{name} ({self.profile or self.tag}): mastered from {name}"
        if self.orientation != 1:
            s += f"; EXIF orientation {self.orientation} applied"
        return s


def _be32(b, o):
    return struct.unpack_from(">I", b, o)[0]


def _be16(b, o):
    return struct.unpack_from(">H", b, o)[0]


def _icc_xyz(icc: bytes, off: int, size: int):
    if size < 20 or off + size > len(icc) or icc[off:off + 4] != b"XYZ ":
        return None
    return tuple(struct.unpack_from(">i", icc, off + 8 + 4 * i)[0] / 65536.0 for i in range(3))


def _icc_text(icc: bytes, off: int, size: int) -> str:
    if off + size > len(icc) or size < 12:
        return ""
    p = icc[off:off + size]
    if p[:4] == b"desc":
        n = _be32(p, 8)
        if 12 + n > size:
            return ""
        return p[12:12 + n].rstrip(b"\0").decode("latin-1")
    if p[:4] == b"mluc" and size >= 28:
        ln, at = _be32(p, 20), _be32(p, 24)
        if at + ln > size:
            return ""
        return "".join(chr(u) if u < 0x80 else "?" for u in
                       (_be16(p, at + i) for i in range(0, ln - 1, 2)))
    return ""


def icc_primaries(icc: bytes) -> Tuple[Optional[str], str]:
    """(primaries id or None, description) from an ICC profile's colorant tags, within 0.01."""
    desc = ""
    if len(icc) < 132 or icc[36:40] != b"acsp":
        return None, desc
    n = _be32(icc, 128)
    r = g = b = None
    for i in range(n):
        e = 132 + 12 * i
        if e + 12 > len(icc):
            break
        sig, off, size = icc[e:e + 4], _be32(icc, e + 4), _be32(icc, e + 8)
        if sig == b"rXYZ":
            r = _icc_xyz(icc, off, size)
        elif sig == b"gXYZ":
            g = _icc_xyz(icc, off, size)
        elif sig == b"bXYZ":
            b = _icc_xyz(icc, off, size)
        elif sig == b"desc":
            desc = _icc_text(icc, off, size)
    if icc[16:20] != b"RGB " or r is None or g is None or b is None:
        return None, desc
    got = r + g + b
    for pid, k in _ICC_KNOWN.items():
        if all(abs(x - y) <= 0.01 for x, y in zip(got, k)):
            return pid, desc
    return None, desc


def _tiff_entries(t: bytes):
    if len(t) < 8:
        return None, []
    le = t[:2] == b"II"
    if not le and t[:2] != b"MM":
        return None, []
    e16, e32 = ("<H", "<I") if le else (">H", ">I")
    ifd = struct.unpack_from(e32, t, 4)[0]
    if ifd + 2 > len(t):
        return (e16, e32), []
    n = struct.unpack_from(e16, t, ifd)[0]
    out = []
    for i in range(n):
        e = ifd + 2 + 12 * i
        if e + 12 > len(t):
            break
        out.append(e)
    return (e16, e32), out


def tiff_orientation(t: bytes) -> int:
    fmt, entries = _tiff_entries(t)
    for e in entries:
        if struct.unpack_from(fmt[0], t, e)[0] == 0x0112:
            v = struct.unpack_from(fmt[0], t, e + 8)[0]
            return v if 1 <= v <= 8 else 1
    return 1


def _tiff_icc(t: bytes) -> bytes:
    fmt, entries = _tiff_entries(t)
    for e in entries:
        if struct.unpack_from(fmt[0], t, e)[0] == 34675:
            count, off = struct.unpack_from(fmt[1], t, e + 4)[0], struct.unpack_from(fmt[1], t, e + 8)[0]
            if off + count <= len(t):
                return t[off:off + count]
    return b""


def _classify_xy(c) -> Optional[str]:
    for pid, k in CHROMATICITIES.items():
        if all(abs(x - y) <= 0.005 for x, y in zip(c, k)):
            return pid
    return None


def _from_icc(out: StillColour, icc: bytes) -> None:
    out.tagged, out.tag = True, "icc"
    pid, out.profile = icc_primaries(icc)
    out.primaries, out.recognised = (pid, True) if pid else ("rec709", False)


def _read_png(b: bytes, out: StillColour) -> None:
    pos, have_icc, have_cicp = 8, False, False
    while pos + 12 <= len(b):
        ln, typ = _be32(b, pos), b[pos + 4:pos + 8]
        if pos + 12 + ln > len(b):
            break
        data = b[pos + 8:pos + 8 + ln]
        if typ == b"cICP" and ln >= 4:
            have_cicp = True
            cp = data[0]
            out.tagged, out.tag = True, "cicp"
            out.recognised = cp in (1, 9, 12)
            out.primaries = "rec2020" if cp == 9 else "p3d65" if cp == 12 else "rec709"
            out.profile = f"cICP colour primaries {cp}"
        elif typ == b"iCCP" and not have_cicp:
            nul = data.find(b"\0")
            if nul >= 0 and nul + 2 <= len(data):
                try:
                    icc = zlib.decompress(data[nul + 2:], bufsize=1 << 16)
                except zlib.error:
                    icc = b""
                if icc:
                    _from_icc(out, icc)
                    have_icc = True
                else:
                    out.tagged, out.tag, out.recognised, out.profile = True, "icc", False, "unreadable iCCP"
        elif typ == b"sRGB" and not have_icc and not have_cicp:
            out.tagged, out.tag, out.primaries, out.recognised, out.profile = True, "srgb", "rec709", True, "sRGB"
        elif typ == b"cHRM" and ln >= 32 and not have_icc and not have_cicp and out.tag != "srgb":
            v = [_be32(data, 4 * i) / 100000.0 for i in range(8)]   # white, red, green, blue
            c = v[2:] + v[:2]
            pid = _classify_xy(c)
            out.tagged, out.tag, out.recognised = True, "chrm", pid is not None
            out.primaries, out.profile = pid or "rec709", "cHRM chromaticities"
        elif typ == b"eXIf":
            out.orientation = tiff_orientation(data)
        elif typ in (b"IDAT", b"IEND"):
            break
        pos += 12 + ln


def _read_jpeg(b: bytes, out: StillColour) -> None:
    pos, parts = 2, []
    while pos + 4 <= len(b) and b[pos] == 0xFF:
        marker = b[pos + 1]
        if marker == 0xD8 or 0xD0 <= marker <= 0xD7 or marker == 0x01:
            pos += 2
            continue
        if marker in (0xDA, 0xD9):
            break
        ln = _be16(b, pos + 2)
        if ln < 2 or pos + 2 + ln > len(b):
            break
        seg = b[pos + 4:pos + 2 + ln]
        if marker == 0xE1 and len(seg) > 6 and seg[:6] == b"Exif\0\0":
            out.orientation = tiff_orientation(seg[6:])
        elif marker == 0xE2 and len(seg) > 14 and seg[:12] == b"ICC_PROFILE\0":
            parts.append((seg[12], seg[14:]))
        pos += 2 + ln
    if parts:
        _from_icc(out, b"".join(p for _, p in sorted(parts, key=lambda x: x[0])))


def _read_webp(b: bytes, out: StillColour) -> None:
    pos = 12
    while pos + 8 <= len(b):
        ln = struct.unpack_from("<I", b, pos + 4)[0]
        if pos + 8 + ln > len(b):
            break
        data = b[pos + 8:pos + 8 + ln]
        if b[pos:pos + 4] == b"ICCP":
            _from_icc(out, data)
        if b[pos:pos + 4] == b"EXIF":
            out.orientation = tiff_orientation(data[6:] if data[:6] == b"Exif\0\0" else data)
        pos += 8 + ln + (ln & 1)


def read_still_colour(b: bytes) -> StillColour:
    out = StillColour()
    if b[:8] == b"\x89PNG\r\n\x1a\n":
        _read_png(b, out)
    elif b[:2] == b"\xff\xd8":
        _read_jpeg(b, out)
    elif len(b) >= 8 and ((b[:2] == b"II" and b[2] == 42) or (b[:2] == b"MM" and b[3] == 42)):
        out.orientation = tiff_orientation(b)
        icc = _tiff_icc(b)
        if icc:
            _from_icc(out, icc)
    elif b[:4] == b"RIFF" and b[8:12] == b"WEBP":
        _read_webp(b, out)
    return out


def apply_orientation(a: np.ndarray, orientation: int) -> np.ndarray:
    """EXIF orientation applied to an (H, W, ...) array: what a viewer shows. 5..8 swap H and W."""
    o = int(orientation)
    if o == 2:
        return a[:, ::-1]
    if o == 3:
        return a[::-1, ::-1]
    if o == 4:
        return a[::-1]
    if o == 5:
        return np.swapaxes(a, 0, 1)
    if o == 6:
        return np.swapaxes(a, 0, 1)[:, ::-1]
    if o == 7:
        return np.swapaxes(a, 0, 1)[::-1, ::-1]
    if o == 8:
        return np.swapaxes(a, 0, 1)[::-1]
    return a


def _rgb_to_xyz(pid: str) -> np.ndarray:
    rx, ry, gx, gy, bx, by, wx, wy = CHROMATICITIES[pid]
    m = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in ((rx, ry), (gx, gy), (bx, by))]).T
    w = np.array([wx / wy, 1.0, (1 - wx - wy) / wy])
    return m * np.linalg.solve(m, w)


def matrix_to_rec709(pid: str) -> np.ndarray:
    """Linear RGB in ``pid`` primaries -> linear Rec.709 (same D65 white)."""
    if pid == "rec709":
        return np.eye(3)
    return np.linalg.inv(_rgb_to_xyz("rec709")) @ _rgb_to_xyz(pid)
