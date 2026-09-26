"""Read EXR the way a comp application reads it: all of it.

The previous reader pulled ``R``, ``G``, ``B`` and ``A`` out of part 0 and
raised on anything else:

    '/renders/sh010.exr' has no standard R/G/B channels (found: ['Z', ...]).
    This looks like a depth/data-only or multi-layer AOV file, which
    RadianceRead does not support -- it only reads standard RGB(A) EXR.

A multi-layer EXR *is* the normal output of a Nuke or Arnold render. Refusing
the format the package exists to serve, and telling the user their file is the
problem, is the wrong way round.

This module reads:

* **multi-part** files (each part its own layer, as Nuke writes them),
* **multi-channel** single-part files (``diffuse.R``, ``specular.B``, ``N.X``),
* **data-only** files -- a Z-depth or a normal pass with no beauty at all,
* and plain RGB(A), which still takes the same path.

It also honours the **display window**. An overscan render -- data window larger
than display window, which is what Nuke writes by default -- used to come back
at the data-window resolution and offset, silently. The frame is now conformed
to the display window, cropping the overscan or padding a short data window,
with a line in the log saying so. ``raw=True`` gives you the data window
untouched, for the cases where you want exactly what is in the file.

Backends, in order: the OpenEXR 3.x ``File`` API (which groups channels into
layers for us and reads every part), the legacy ``InputFile`` + ``Imath``
bindings, then OpenImageIO, then OpenCV. Each fallback covers less; the log says
which one ran when it matters.
"""
from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger("radiance.exr")

#: Channel names that mean "this is the beauty", in the order we prefer them.
_BEAUTY_NAMES = ("RGBA", "RGB", "rgba", "rgb", "beauty", "Beauty", "C", "")

#: Single-channel names that are data, not colour. Worth knowing so the reader
#: can say "this file is a depth pass" instead of "this file has no R channel".
_DATA_CHANNELS = frozenset({"Z", "z", "depth", "Depth", "ZBack", "id", "ID"})

_LAYER_SUFFIX_RE = re.compile(r"\.(R|G|B|A|X|Y|Z|U|V|W|r|g|b|a|x|y|z)$")


class EXRReadError(RuntimeError):
    """The file could not be read, and the message says what would help."""


@dataclass(frozen=True)
class LayerInfo:
    """One addressable layer inside an EXR."""

    name: str                 #: what the user picks, e.g. "diffuse" or "rgba"
    part: int                 #: which part it lives in
    channels: Tuple[str, ...]
    width: int
    height: int
    is_beauty: bool = False
    is_data: bool = False

    @property
    def channel_count(self) -> int:
        return len(self.channels)

    def label(self) -> str:
        kind = " (beauty)" if self.is_beauty else (" (data)" if self.is_data else "")
        return f"{self.name}{kind} · {self.channel_count}ch · {self.width}x{self.height}"


@dataclass(frozen=True)
class EXRInfo:
    """What the file is, before you decide what to pull out of it."""

    path: str
    width: int = 0
    height: int = 0
    display_width: int = 0
    display_height: int = 0
    data_offset: Tuple[int, int] = (0, 0)
    parts: int = 1
    layers: Tuple[LayerInfo, ...] = ()
    compression: str = ""
    pixel_type: str = ""
    attributes: Dict[str, Any] = None  # type: ignore[assignment]
    backend: str = ""

    @property
    def has_overscan(self) -> bool:
        return (self.display_width, self.display_height) != (self.width, self.height) \
            or self.data_offset != (0, 0)

    @property
    def layer_names(self) -> List[str]:
        return [layer.name for layer in self.layers]

    def beauty(self) -> Optional[LayerInfo]:
        for layer in self.layers:
            if layer.is_beauty:
                return layer
        colour = [layer for layer in self.layers if not layer.is_data]
        return colour[0] if colour else (self.layers[0] if self.layers else None)

    def summary(self) -> str:
        bits = [f"{self.width}x{self.height}"]
        if self.has_overscan:
            bits.append(
                f"overscan (display {self.display_width}x{self.display_height})")
        if self.parts > 1:
            bits.append(f"{self.parts} parts")
        bits.append(f"{len(self.layers)} layer(s): "
                    + ", ".join(self.layer_names[:8])
                    + (f", +{len(self.layers) - 8}" if len(self.layers) > 8 else ""))
        if self.compression:
            bits.append(self.compression)
        return " · ".join(bits)


# ── layer naming ───────────────────────────────────────────────────────────

def _group_channels(names: List[str]) -> Dict[str, List[str]]:
    """``["diffuse.R", "diffuse.G", "Z"]`` -> ``{"diffuse": [...], "Z": ["Z"]}``.

    The OpenEXR 3.x bindings do this themselves; the legacy bindings and OIIO
    do not, so the fallbacks share this.
    """
    groups: Dict[str, List[str]] = {}
    for name in names:
        if "." in name:
            layer, _, _suffix = name.rpartition(".")
        elif name in ("R", "G", "B", "A"):
            layer = "RGBA"
        else:
            layer = name
        groups.setdefault(layer, []).append(name)
    for channels in groups.values():
        channels.sort(key=_channel_sort_key)
    return groups


def _channel_sort_key(name: str) -> Tuple[int, str]:
    """R, G, B, A order — not alphabetical, which would give A, B, G, R."""
    order = {"r": 0, "x": 0, "g": 1, "y": 1, "b": 2, "z": 2,
             "a": 3, "w": 3}
    suffix = name.rpartition(".")[2] if "." in name else name
    return order.get(suffix.lower(), 9), name


def _is_beauty(layer_name: str) -> bool:
    return layer_name in _BEAUTY_NAMES


def _is_data(layer_name: str, channels: List[str]) -> bool:
    if layer_name in _DATA_CHANNELS:
        return True
    return len(channels) == 1 and not _is_beauty(layer_name)


# ── probing ────────────────────────────────────────────────────────────────

def probe(path: str) -> EXRInfo:
    """List the parts, layers and windows without decoding pixels."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"No such EXR: {path}")
    for backend in (_probe_modern, _probe_legacy, _probe_oiio):
        try:
            info = backend(path)
        except ImportError:
            continue
        except Exception as exc:
            logger.debug("%s could not probe %s: %s", backend.__name__, path, exc)
            continue
        if info is not None:
            return info
    raise EXRReadError(
        f"Cannot read {os.path.basename(path)}: no EXR backend is available. "
        "pip install OpenEXR"
    )


def _window_size(window) -> Tuple[int, int, int, int]:
    """``(x, y, width, height)`` from an OpenEXR box, whatever shape it takes."""
    try:
        lo, hi = window
        x0, y0 = int(lo[0]), int(lo[1])
        x1, y1 = int(hi[0]), int(hi[1])
    except (TypeError, ValueError, IndexError):
        x0, y0 = int(window.min.x), int(window.min.y)
        x1, y1 = int(window.max.x), int(window.max.y)
    return x0, y0, x1 - x0 + 1, y1 - y0 + 1


def _probe_modern(path: str) -> Optional[EXRInfo]:
    """OpenEXR >= 3.1 Python bindings: every part, channels pre-grouped."""
    import OpenEXR

    if not hasattr(OpenEXR, "File"):
        raise ImportError("this OpenEXR build has no File API")

    layers: List[LayerInfo] = []
    attributes: Dict[str, Any] = {}
    width = height = dw_w = dw_h = 0
    off = (0, 0)
    compression = pixel_type = ""

    with OpenEXR.File(path) as handle:
        parts = list(handle.parts)
        for index, part in enumerate(parts):
            header = dict(part.header)
            dx, dy, w, h = _window_size(header.get("dataWindow"))
            if index == 0:
                width, height, off = w, h, (dx, dy)
                _, _, dw_w, dw_h = _window_size(
                    header.get("displayWindow", header.get("dataWindow")))
                compression = str(header.get("compression", "")).split(".")[-1]
                attributes = {
                    k: _jsonable(v) for k, v in header.items()
                    if k not in ("channels",)
                }
            part_name = ""
            try:
                part_name = part.name() or ""
            except Exception:
                part_name = ""
            for name, channel in part.channels.items():
                pixels = getattr(channel, "pixels", None)
                count = 1 if pixels is None or pixels.ndim == 2 else pixels.shape[-1]
                if pixel_type == "" and pixels is not None:
                    pixel_type = str(pixels.dtype)
                # A multi-part file names its layers by part; a single-part file
                # names them by channel prefix. Prefer the part name when the
                # channel name is just "RGBA".
                label = name
                if len(parts) > 1 and part_name and _is_beauty(name):
                    label = part_name
                layers.append(LayerInfo(
                    name=label,
                    part=index,
                    channels=tuple(f"{name}.{c}" for c in "RGBA"[:count])
                    if count > 1 else (name,),
                    width=w, height=h,
                    is_beauty=_is_beauty(name) and index == 0,
                    is_data=_is_data(name, [name] * count),
                ))

    return EXRInfo(
        path=path, width=width, height=height,
        display_width=dw_w or width, display_height=dw_h or height,
        data_offset=off, parts=(max(layer.part for layer in layers) + 1) if layers else 1,
        layers=tuple(layers), compression=compression, pixel_type=pixel_type,
        attributes=attributes, backend="OpenEXR.File",
    )


def _probe_legacy(path: str) -> Optional[EXRInfo]:
    """OpenEXR 2.x bindings. Part 0 only — that is all they expose."""
    import OpenEXR

    if not hasattr(OpenEXR, "InputFile"):
        raise ImportError("no legacy OpenEXR bindings")

    handle = OpenEXR.InputFile(path)
    try:
        header = handle.header()
        dx, dy, w, h = _window_size(header["dataWindow"])
        _, _, dw_w, dw_h = _window_size(header.get("displayWindow", header["dataWindow"]))
        groups = _group_channels(list(header["channels"].keys()))
        layers = tuple(
            LayerInfo(name=name, part=0, channels=tuple(channels),
                      width=w, height=h,
                      is_beauty=_is_beauty(name), is_data=_is_data(name, channels))
            for name, channels in sorted(groups.items())
        )
        attributes = {k: _jsonable(v) for k, v in header.items() if k != "channels"}
    finally:
        handle.close()

    return EXRInfo(
        path=path, width=w, height=h,
        display_width=dw_w, display_height=dw_h, data_offset=(dx, dy),
        parts=1, layers=layers,
        compression=str(header.get("compression", "")),
        pixel_type="float32", attributes=attributes, backend="OpenEXR.InputFile",
    )


def _probe_oiio(path: str) -> Optional[EXRInfo]:
    import OpenImageIO as oiio

    handle = oiio.ImageInput.open(path)
    if handle is None:
        raise EXRReadError(f"OpenImageIO cannot open {path}: {oiio.geterror()}")
    try:
        spec = handle.spec()
        groups = _group_channels(list(spec.channelnames))
        layers = tuple(
            LayerInfo(name=name, part=0, channels=tuple(channels),
                      width=spec.width, height=spec.height,
                      is_beauty=_is_beauty(name), is_data=_is_data(name, channels))
            for name, channels in sorted(groups.items())
        )
    finally:
        handle.close()
    return EXRInfo(
        path=path, width=spec.width, height=spec.height,
        display_width=spec.full_width or spec.width,
        display_height=spec.full_height or spec.height,
        data_offset=(spec.x, spec.y), parts=1, layers=layers,
        attributes={}, backend="OpenImageIO",
    )


def _jsonable(value: Any) -> Any:
    """EXR attributes are a zoo of C++ types. Make them printable."""
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    return str(value)


# ── reading ────────────────────────────────────────────────────────────────

def read_layer(
    path: str,
    layer: Optional[str] = None,
    *,
    info: Optional[EXRInfo] = None,
    raw: bool = False,
) -> Tuple[np.ndarray, Optional[np.ndarray], EXRInfo, str]:
    """Read one layer as ``(H, W, 3) float32`` plus its alpha, if any.

    Parameters
    ----------
    layer
        Which layer to pull. ``None`` or ``"auto"`` takes the beauty, falling
        back to the first colour layer and then to the first layer of any kind,
        so a Z-only file still reads rather than raising.
    raw
        Return the data window exactly as stored, skipping the conform to the
        display window.

    Returns ``(rgb, alpha, info, layer_name)``.
    """
    if info is None:
        info = probe(path)
    if not info.layers:
        raise EXRReadError(f"{os.path.basename(path)} has no readable channels.")

    chosen = _pick_layer(info, layer)
    pixels = _read_layer_pixels(path, chosen, info)

    if pixels.ndim == 2:
        pixels = pixels[..., None]
    channels = pixels.shape[-1]

    alpha = None
    if channels >= 4:
        alpha = np.ascontiguousarray(pixels[..., 3].astype(np.float32))
        rgb = pixels[..., :3]
    elif channels == 3:
        rgb = pixels
    elif channels == 2:
        # A UV or motion pass. Third channel zero rather than a broadcast of U.
        rgb = np.concatenate([pixels, np.zeros_like(pixels[..., :1])], axis=-1)
    else:
        # Depth, an ID, a single-channel matte. Replicate so it is viewable;
        # the values are untouched, so it is still the data you asked for.
        rgb = np.repeat(pixels, 3, axis=-1)

    rgb = np.ascontiguousarray(rgb.astype(np.float32))

    if not raw and info.has_overscan:
        rgb = _conform_to_display_window(rgb, info)
        if alpha is not None:
            alpha = _conform_to_display_window(alpha[..., None], info)[..., 0]
        logger.info(
            "[Radiance/EXR] %s has a data window of %dx%d at (%d, %d) inside a "
            "%dx%d display window; conformed to the display window. Set raw to "
            "keep the overscan.",
            os.path.basename(path), info.width, info.height,
            info.data_offset[0], info.data_offset[1],
            info.display_width, info.display_height,
        )

    return rgb, alpha, info, chosen.name


def _pick_layer(info: EXRInfo, requested: Optional[str]) -> LayerInfo:
    if requested in (None, "", "auto", "Auto", "auto (beauty)"):
        chosen = info.beauty()
        if chosen is None:
            raise EXRReadError(
                f"{os.path.basename(info.path)} has no layer to read.")
        if len(info.layers) > 1:
            others = [n for n in info.layer_names if n != chosen.name]
            logger.info(
                "[Radiance/EXR] %s: reading layer %r. Also present: %s. Set the "
                "layer widget to pick another.",
                os.path.basename(info.path), chosen.name, ", ".join(others[:12]),
            )
        return chosen

    for candidate in info.layers:
        if candidate.name == requested:
            return candidate
    # Tolerate a user typing "diffuse.R" or a case difference.
    normalised = _LAYER_SUFFIX_RE.sub("", str(requested)).lower()
    for candidate in info.layers:
        if candidate.name.lower() == normalised:
            return candidate
    # Nuke names a layer by its prefix: "depth" for a single "depth.Z"
    # channel, which the File API reports whole.
    for candidate in info.layers:
        if candidate.name.rpartition(".")[0].lower() == normalised:
            return candidate
    raise EXRReadError(
        f"{os.path.basename(info.path)} has no layer named {requested!r}. "
        f"It has: {', '.join(info.layer_names)}"
    )


def _read_layer_pixels(path: str, layer: LayerInfo, info: EXRInfo) -> np.ndarray:
    if info.backend == "OpenEXR.File":
        import OpenEXR

        with OpenEXR.File(path) as handle:
            part = handle.parts[layer.part]
            for name, channel in part.channels.items():
                if name == layer.name or (
                    len(handle.parts) > 1 and _is_beauty(name)
                    and (part.name() or "") == layer.name
                ):
                    return np.asarray(channel.pixels)
            raise EXRReadError(
                f"{os.path.basename(path)} part {layer.part} no longer has a "
                f"channel {layer.name!r}; the file changed while it was being read."
            )

    if info.backend == "OpenEXR.InputFile":
        import Imath
        import OpenEXR

        handle = OpenEXR.InputFile(path)
        try:
            header = handle.header()
            _dx, _dy, w, h = _window_size(header["dataWindow"])
            pt = Imath.PixelType(Imath.PixelType.FLOAT)
            planes = [
                np.frombuffer(handle.channel(name, pt), dtype=np.float32).reshape(h, w)
                for name in layer.channels
            ]
        finally:
            handle.close()
        return np.stack(planes, axis=-1)

    import OpenImageIO as oiio

    handle = oiio.ImageInput.open(path)
    if handle is None:
        raise EXRReadError(f"OpenImageIO cannot open {path}: {oiio.geterror()}")
    try:
        spec = handle.spec()
        full = np.asarray(handle.read_image(format=oiio.FLOAT), dtype=np.float32)
        full = full.reshape(spec.height, spec.width, spec.nchannels)
        indices = [spec.channelnames.index(c) for c in layer.channels
                   if c in spec.channelnames]
    finally:
        handle.close()
    if not indices:
        raise EXRReadError(f"{os.path.basename(path)} has no channels for "
                           f"layer {layer.name!r}")
    return full[..., indices]


def _conform_to_display_window(pixels: np.ndarray, info: EXRInfo) -> np.ndarray:
    """Place the data window inside the display window: crop overscan, pad short.

    Nuke keeps both windows and carries the bounding box through the graph.
    ComfyUI has one rectangle per image, so the display window -- the format the
    shot is actually delivered at -- is the one to honour.
    """
    out = np.zeros(
        (info.display_height, info.display_width, pixels.shape[-1]), np.float32)
    ox, oy = info.data_offset
    # Source region that lands inside the display window.
    sx0 = max(0, -ox)
    sy0 = max(0, -oy)
    dx0 = max(0, ox)
    dy0 = max(0, oy)
    w = min(pixels.shape[1] - sx0, info.display_width - dx0)
    h = min(pixels.shape[0] - sy0, info.display_height - dy0)
    if w > 0 and h > 0:
        out[dy0:dy0 + h, dx0:dx0 + w] = pixels[sy0:sy0 + h, sx0:sx0 + w]
    return out


def layer_choices(path: str) -> List[str]:
    """Layer names for the node's dropdown, beauty first."""
    try:
        info = probe(path)
    except Exception as exc:
        logger.debug("cannot list layers of %s: %s", path, exc)
        return []
    beauty = info.beauty()
    names = list(info.layer_names)
    if beauty and beauty.name in names:
        names.remove(beauty.name)
        names.insert(0, beauty.name)
    return names


__all__ = [
    "EXRInfo",
    "EXRReadError",
    "LayerInfo",
    "layer_choices",
    "probe",
    "read_layer",
]
