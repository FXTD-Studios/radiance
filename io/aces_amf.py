"""ACES Metadata File (AMF v2.0) sidecars for ACES EXR masters.

The same document RUDRA 1.1.0 writes beside its masters
(native/core/src/aces_amf.cpp, Apache 2.0): the pipeline's input transform
(ACEScg to ACES2065-1, not applied, when the master is ACEScg) and the ACES 2.0
SDR 100-nit Rec.709 output transform the master is viewed through, so a
grading or review tool opens it in the right ACES pipeline. Every text value
is XML-escaped; the document validates against the Academy's AMF v2.0 schema
(tests/test_aces_amf.py).
"""

from __future__ import annotations

import datetime as _dt
import re
import uuid as _uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

AMF_NAMESPACE = "urn:ampas:aces:amf:v2.0"
ACESCG_TO_ACES_ID = "urn:ampas:aces:transformId:v2.0:CSC.Academy.ACEScg_to_ACES.a2.v1"
REC709_BT1886_OUTPUT_ID = (
    "urn:ampas:aces:transformId:v2.0:Output.Academy.Rec709-D65_100nit_in_Rec709-D65_BT1886.a2.v1")

#: Output colour spaces an AMF can describe: the ACES containers.
ACES_LABELS = {"ACEScg": True, "ACES2065-1": False}   # label -> is ACEScg


@dataclass
class AmfRequest:
    description: str = "Radiance ACES master"
    author_name: str = ""
    author_email: str = ""
    created_utc: str = ""
    amf_uuid: str = ""
    pipeline_uuid: str = ""
    acescg: bool = True
    clip_name: Optional[str] = None
    sequence: Optional[Tuple[str, int, int]] = None   # (pattern with '#', first, last)
    file: Optional[str] = None
    #: RUDRA writes "...: the view RUDRA's appearance constraint holds the master to".
    output_description: str = "ACES 2.0 SDR 100 nits Rec.709 BT.1886: the reference view of the master"


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
            .replace('"', "&quot;").replace("'", "&apos;"))


def _esc_text(s: str) -> str:
    """Element text needs only & < > escaped (RUDRA writes its fixed texts so)."""
    return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _datetime(indent: str, when: str) -> str:
    return (f"{indent}<aces:dateTime>\n"
            f"{indent}    <aces:creationDateTime>{_esc(when)}</aces:creationDateTime>\n"
            f"{indent}    <aces:modificationDateTime>{_esc(when)}</aces:modificationDateTime>\n"
            f"{indent}</aces:dateTime>\n")


def make_amf_uuid() -> str:
    """A random (version 4) UUID as urn:uuid:..., the schema's pattern."""
    return f"urn:uuid:{_uuid.uuid4()}"


def amf_now_utc() -> str:
    return _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def aces_amf_xml(r: AmfRequest) -> str:
    """The AMF document, UTF-8 text, in RUDRA's layout (byte for byte with the same request)."""
    when = r.created_utc or amf_now_utc()
    x = '<?xml version="1.0" encoding="UTF-8"?>\n'
    x += (f'<aces:acesMetadataFile xmlns:aces="{AMF_NAMESPACE}" '
          'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" '
          'xmlns:cdl="urn:ASC:CDL:v1.01" version="2.0">\n')
    x += "    <aces:amfInfo>\n"
    x += f"        <aces:description>{_esc(r.description)}</aces:description>\n"
    # The schema's author needs a name and an email; without an email none is written.
    if r.author_name and r.author_email:
        x += (f"        <aces:author>\n            <aces:name>{_esc(r.author_name)}</aces:name>\n"
              f"            <aces:emailAddress>{_esc(r.author_email)}</aces:emailAddress>\n"
              "        </aces:author>\n")
    x += _datetime("        ", when)
    x += f"        <aces:uuid>{_esc(r.amf_uuid or make_amf_uuid())}</aces:uuid>\n"
    x += "    </aces:amfInfo>\n"
    if r.clip_name and (r.sequence or r.file):
        x += f"    <aces:clipId>\n        <aces:clipName>{_esc(r.clip_name)}</aces:clipName>\n"
        if r.sequence:
            pattern, first, last = r.sequence
            x += (f'        <aces:sequence idx="#" min="{int(first)}" max="{int(last)}">'
                  f"{_esc(pattern)}</aces:sequence>\n")
        else:
            x += f"        <aces:file>{_esc(r.file)}</aces:file>\n"
        x += "    </aces:clipId>\n"
    x += "    <aces:pipeline>\n        <aces:pipelineInfo>\n"
    x += _datetime("            ", when)
    x += f"            <aces:uuid>{_esc(r.pipeline_uuid or make_amf_uuid())}</aces:uuid>\n"
    x += ("            <aces:systemVersion>\n                <aces:majorVersion>2</aces:majorVersion>\n"
          "                <aces:minorVersion>0</aces:minorVersion>\n"
          "                <aces:patchVersion>0</aces:patchVersion>\n"
          "            </aces:systemVersion>\n        </aces:pipelineInfo>\n")
    if r.acescg:
        x += ('        <aces:inputTransform applied="false">\n'
              "            <aces:description>ACEScg to ACES2065-1 (the master is ACEScg)</aces:description>\n"
              f"            <aces:transformId>{ACESCG_TO_ACES_ID}</aces:transformId>\n"
              "        </aces:inputTransform>\n")
    x += ('        <aces:outputTransform applied="false">\n'
          f"            <aces:description>{_esc_text(r.output_description)}</aces:description>\n"
          f"            <aces:transformId>{REC709_BT1886_OUTPUT_ID}</aces:transformId>\n"
          "        </aces:outputTransform>\n")
    x += "    </aces:pipeline>\n</aces:acesMetadataFile>\n"
    return x


_TRAILING_DIGITS = re.compile(r"^(.*?)(\d+)$")


def amf_for_files(paths: Iterable[str], colour_label: str,
                  description: str = "Radiance ACES master") -> Optional[Tuple[Path, str]]:
    """(sidecar path, AMF text) for EXR files just written in an ACES colour
    space, or None when there is nothing an AMF can describe (no EXR, or
    another colour space). A sequence (frames named <stem><digits>.exr) gets
    one AMF named after the stem; a single file gets <name>.amf beside it."""
    if colour_label not in ACES_LABELS:
        return None
    exrs = [Path(p) for p in paths if str(p).lower().endswith(".exr")]
    if not exrs:
        return None
    acescg = ACES_LABELS[colour_label]
    if len(exrs) == 1:
        f = exrs[0]
        req = AmfRequest(description=description, acescg=acescg, clip_name=f.stem, file=f.name)
        return f.with_suffix(".amf"), aces_amf_xml(req)
    first = exrs[0]
    m = _TRAILING_DIGITS.match(first.stem)
    frames = []
    for p in exrs:
        mm = _TRAILING_DIGITS.match(p.stem)
        if not mm or mm.group(1) != (m.group(1) if m else None) or p.parent != first.parent:
            frames = []
            break
        frames.append(int(mm.group(2)))
    if not m or not frames:
        return None
    prefix, digits = m.group(1), len(m.group(2))
    pattern = f"{prefix}{'#' * digits}.exr"
    clip = prefix.rstrip("_.-") or first.stem
    req = AmfRequest(description=description, acescg=acescg, clip_name=clip,
                     sequence=(pattern, min(frames), max(frames)))
    return first.parent / f"{clip}.amf", aces_amf_xml(req)
