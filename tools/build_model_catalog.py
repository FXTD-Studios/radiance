#!/usr/bin/env python3
"""Build config/model_catalog.json: every model the shipped workflows need.

    python tools/build_model_catalog.py           # rebuild from workflows/*.json
    python tools/build_model_catalog.py --check   # exit 1 when it is out of date
    python tools/build_model_catalog.py --pin     # also pin revision, SHA-256 and
                                                  # size from the Hugging Face API
                                                  # (network; HF_TOKEN for gated repos)

The Radiance setup check (core/setup_check.py, the Models page and the tour)
reads the catalog to say which workflows are ready on this machine and to
download what is missing. It is built from what the workflows themselves say:

* the model files a workflow's nodes name in their widgets (what it loads);
* its "Model Links" note, for each file's Hugging Face URL, and whether the
  link is marked used, optional or an alternative, or 🔒 (gated);
* its "Model Storage Location" tree, for the folder under ComfyUI/models;
* the Radiance nodes in it, for Radiance's own models (RUDRA ACES 2, the AI
  fill, MoGe-2 and Marigold), which their own downloaders fetch and verify.

Pins (revision, sha256, size) already in the catalog are kept when the file's
URL has not changed, so a rebuild never drops them. Unpinned files are pinned
at download time from the Hub's own metadata (see core/setup_check.py).
Standard library only.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
CATALOG = ROOT / "config" / "model_catalog.json"

MODEL_EXT = re.compile(r"\.(safetensors|ckpt|pth|pt|gguf|bin|sft)$", re.I)
LINK = re.compile(r"\[([^\]]+)\]\((https://huggingface\.co/[^)\s]+)\)([^\[\n]*)")
KNOWN_FOLDERS = ("diffusion_models", "text_encoders", "vae", "loras", "latent_upscale_models",
                 "upscale_models", "checkpoints", "clip", "clip_vision", "unet", "controlnet",
                 "geometry_estimation", "audio_encoders", "model_patches")

# Files a workflow loads that its widgets do not name: the loader picks them
# itself (Wan 2.2's low-noise expert is paired with the high-noise one).
IMPLIED = {
    "wan22_hdr_shot_plates": ["wan2.2_t2v_low_noise_14B_fp8_scaled.safetensors"],
}

# Radiance's own models, fetched and verified by their own downloaders.
RADIANCE_PACKS = {
    "radiance/RUDRA_ACES_2": {"title": "RUDRA ACES 2 (SDR to HDR)", "size_mb": 3, "licence": "Apache 2.0",
                               "page": "https://huggingface.co/fxtdstudios/rudra-models"},
    "radiance/rudra_genfill_v3": {"title": "RUDRA AI fill v3", "size_mb": 271, "licence": "Apache 2.0",
                                  "page": "https://huggingface.co/fxtdstudios/rudra-models"},
    "radiance/wan_vace_base": {"title": "Wan2.1-VACE-1.3B (AI fill base model)", "size_mb": 7700,
                               "licence": "Apache 2.0",
                               "page": "https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers"},
    "radiance/wan_vace_text_encoder": {"title": "umT5 text encoder (AI fill custom prompts)", "size_mb": 11400,
                                       "licence": "Apache 2.0",
                                       "page": "https://huggingface.co/Wan-AI/Wan2.1-VACE-1.3B-diffusers"},
    "radiance/moge2": {"title": "MoGe-2 (Multipass geometry)", "size_mb": 662, "licence": "MIT",
                       "page": "https://huggingface.co/Comfy-Org/MoGe"},
    "radiance/marigold_appearance": {"title": "Marigold IID Appearance (albedo, materials)", "size_mb": 2500,
                                     "licence": "OpenRAIL++-M",
                                     "page": "https://huggingface.co/prs-eth/marigold-iid-appearance-v1-1"},
    "radiance/marigold_lighting": {"title": "Marigold IID Lighting (diffuse, specular)", "size_mb": 1700,
                                   "licence": "OpenRAIL++-M",
                                   "page": "https://huggingface.co/prs-eth/marigold-iid-lighting-v1-1"},
    "radiance/depth_anything_v2_small": {"title": "Depth Anything V2 Small (Depth)", "size_mb": 100,
                                         "licence": "Apache 2.0",
                                         "page": "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf"},
    "radiance/depth_anything_v2_base": {"title": "Depth Anything V2 Base (Depth)", "size_mb": 390,
                                        "licence": "CC-BY-NC-4.0 (non-commercial)",
                                        "page": "https://huggingface.co/depth-anything/Depth-Anything-V2-Base-hf"},
    "radiance/depth_anything_v2_large": {"title": "Depth Anything V2 Large (Depth)", "size_mb": 1340,
                                         "licence": "CC-BY-NC-4.0 (non-commercial)",
                                         "page": "https://huggingface.co/depth-anything/Depth-Anything-V2-Large-hf"},
}
DEPTH_SIZES = {"Small": "radiance/depth_anything_v2_small", "Base": "radiance/depth_anything_v2_base",
               "Large": "radiance/depth_anything_v2_large"}
# Node type -> (required packs, optional packs) when the node is active.
PACK_NODES = {
    "RadianceSDRToHDRUniversal": (["radiance/RUDRA_ACES_2"], []),
    "RadianceSDRToHDRRecover": (["radiance/RUDRA_ACES_2"], []),
    "RadianceSDRToHDRAIFill": (["radiance/rudra_genfill_v3", "radiance/wan_vace_base"],
                               ["radiance/wan_vace_text_encoder"]),
    "RadianceSDRToHDRAIFillRerender": (["radiance/rudra_genfill_v3", "radiance/wan_vace_base"],
                                       ["radiance/wan_vace_text_encoder"]),
    "RadianceMultipassEstimate": (["radiance/moge2", "radiance/marigold_appearance",
                                   "radiance/marigold_lighting"], []),
}
# Files covered by a Radiance pack, so a note's link to them is not listed twice.
PACK_FILES = {"moge_2_vitl_normal_fp16.safetensors": "radiance/moge2"}


def _all_nodes(graph: dict) -> list:
    nodes = list(graph.get("nodes") or [])
    for sg in (graph.get("definitions") or {}).get("subgraphs") or []:
        nodes.extend(sg.get("nodes") or [])
    return nodes


def _notes(graph: dict) -> str:
    return "\n".join(str((n.get("widgets_values") or [""])[0]) for n in _all_nodes(graph)
                     if n.get("type") in ("MarkdownNote", "Note") and n.get("widgets_values"))


def _tree_folders(text: str) -> dict:
    """{file name: folder under models/} from the Model Storage Location tree."""
    m = re.search(r"Model Storage Location[^\n]*\n+```(.*?)```", text, re.S)
    out = {}
    if not m:
        return out
    stack: list = []
    for line in m.group(1).splitlines():
        if not line.strip():
            continue
        name = line.strip("│ ├└─\t").strip()
        depth = len(line) - len(line.lstrip("│ ├└─\t"))
        while stack and stack[-1][0] >= depth:
            stack.pop()
        if name.startswith("📂"):
            stack.append((depth, name.lstrip("📂").strip().rstrip("/")))
        elif MODEL_EXT.search(name):
            folders = [f for _, f in stack if f not in ("ComfyUI", "models")]
            out[name] = "/".join(folders)
    return out


def _folder_from_url(url: str) -> str:
    parts = url.split("/resolve/", 1)[-1].split("/")[1:-1]      # after the revision
    for p in reversed(parts):
        if p in KNOWN_FOLDERS:
            return p
    return ""


def _links(text: str) -> dict:
    """{file name: {url, gated, role}} from the Model Links section."""
    out = {}
    for line in text.splitlines():
        prefix = re.match(r"\s*-\s*([a-z_]+):", line)
        for label, url, tail in LINK.findall(line):
            fname = url.rsplit("/", 1)[-1]
            if "/resolve/" not in url or not MODEL_EXT.search(fname):
                continue
            t = tail.lower()
            role = ("alternative" if "alternative" in t else "optional" if "optional" in t
                    else "used" if "(used" in t else "")
            folder = _folder_from_url(url) or (prefix.group(1) if prefix and prefix.group(1) in KNOWN_FOLDERS else "")
            # ALBABIT-FIX: a file listed twice keeps the link marked (used), the one the workflow loads.
            if fname not in out or (role == "used" and out[fname]["role"] != "used"):
                out[fname] = {"url": url, "gated": "🔒" in tail, "role": role, "folder": folder}
    return out


def _card_title(stem: str) -> str:
    """The workflow's title on its Workflow Library card (workflows/official/Local/<stem>.rad)."""
    import zipfile
    card = WORKFLOWS / "official" / "Local" / f"{stem}.rad"
    if not card.is_file():
        return ""
    raw = card.read_bytes()
    try:
        with zipfile.ZipFile(io.BytesIO(raw[4:] if raw[:4] == b"RADZ" else raw)) as z:
            meta = json.loads(z.read("manifest.json"))
        return str((meta.get("metadata") or {}).get("stats", {}).get("title") or meta.get("stats", {}).get("title") or "")
    except Exception:  # noqa: BLE001 - a damaged card only loses its title here
        return ""


def _model_id(folder: str, fname: str) -> str:
    return f"{folder}/{fname}" if folder else fname


def _hf_parts(url: str):
    m = re.match(r"https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$", url)
    return (m.group(1), m.group(2), m.group(3)) if m else (None, None, None)


def build(previous: dict | None = None) -> dict:
    prev_models = (previous or {}).get("models", {})
    models: dict = {}
    workflows: dict = {}
    for path in sorted(WORKFLOWS.glob("*.json")):
        if path.name.startswith("api_"):
            continue                                  # API workflows run in the cloud
        graph = json.loads(path.read_text(encoding="utf-8"))
        stem = path.stem
        text = _notes(graph)
        links = _links(text)
        tree = _tree_folders(text)
        nodes = _all_nodes(graph)
        named, named_off = set(), set()          # by an active node / only by bypassed ones
        for n in nodes:
            active = n.get("mode", 0) not in (2, 4)       # 2 muted, 4 bypassed
            for v in n.get("widgets_values") or []:
                if isinstance(v, str) and MODEL_EXT.search(v):
                    (named if active else named_off).add(v)
        named_off -= named
        named |= set(IMPLIED.get(stem, []))
        required, optional = [], []
        for fname in sorted(set(named) | set(links) | set(tree)):
            if fname in PACK_FILES:
                continue
            link = links.get(fname)
            if link is None:
                if fname in named or fname in named_off:
                    raise SystemExit(f"{path.name}: {fname} is loaded but its note has no Hugging Face link")
                continue
            folder = tree.get(fname) or link["folder"]
            if not folder:
                raise SystemExit(f"{path.name}: no models/ folder for {fname}")
            mid = _model_id(folder, fname)
            repo, rev, repo_path = _hf_parts(link["url"])
            entry = {"file": fname, "folder": folder, "repo": repo, "path": repo_path,
                     "url": link["url"], "gated": bool(link["gated"])}
            old = prev_models.get(mid) or {}
            if old.get("url") == link["url"]:
                for k in ("revision", "sha256", "size"):
                    if old.get(k) is not None:
                        entry[k] = old[k]
            if mid in models and models[mid]["url"] != entry["url"]:
                # The same file linked from two repositories: keep the first; it is the same file.
                pass
            else:
                models.setdefault(mid, entry)
            if fname in named or link["role"] == "used":
                required.append(mid)
            else:
                optional.append(mid)
        packs_req, packs_opt = [], []
        for n in nodes:
            req, opt = PACK_NODES.get(n.get("type"), ([], []))
            wv = [v for v in (n.get("widgets_values") or []) if isinstance(v, str)]
            if n.get("type") == "RadianceHDRVAEDecode" and "Direct HDR" in wv:
                req = ["radiance/RUDRA_ACES_2"]          # Direct HDR runs SDR -> HDR Universal inside
            if n.get("type") == "RadianceDepthMapGenerator":
                size = next((v.split()[0] for v in wv if v.split()[0] in DEPTH_SIZES), "Base")
                req = [DEPTH_SIZES[size]]
            active = n.get("mode", 0) not in (2, 4)       # 2 muted, 4 bypassed
            (packs_req if active else packs_opt).extend(req)
            packs_opt.extend(opt)
        packs_req = sorted(set(packs_req))
        packs_opt = sorted(set(packs_opt) - set(packs_req))
        title = _card_title(stem) or stem.replace("_", " ")
        workflows[stem] = {
            "title": title,
            "required": sorted(set(required)) + packs_req,
            "optional": sorted(set(optional) - set(required)) + packs_opt,
        }
    return {
        "_about": "Built by tools/build_model_catalog.py from workflows/*.json. Do not edit by hand.",
        "version": 1,
        "radiance_packs": RADIANCE_PACKS,
        "models": dict(sorted(models.items())),
        "workflows": workflows,
    }


def pin(catalog: dict) -> int:
    """Fill revision, sha256 and size from the Hub for every unpinned model."""
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    failed = 0
    for mid, m in catalog["models"].items():
        if m.get("sha256") and m.get("revision") and m.get("size"):
            continue
        headers = {"User-Agent": "radiance-comfyui"}
        if token:
            headers["Authorization"] = f"Bearer {token}"
        req = urllib.request.Request(m["url"], method="HEAD", headers=headers)
        try:
            opener = urllib.request.build_opener(_NoRedirect)
            try:
                resp = opener.open(req, timeout=30)
            except urllib.error.HTTPError as e:
                if e.code not in (301, 302, 303, 307, 308):
                    raise
                resp = e
            h = resp.headers
            sha = (h.get("X-Linked-Etag") or "").strip('"')
            m["revision"] = h.get("X-Repo-Commit") or m.get("revision")
            m["sha256"] = sha if re.fullmatch(r"[0-9a-f]{64}", sha or "") else m.get("sha256")
            size = h.get("X-Linked-Size") or h.get("Content-Length")
            m["size"] = int(size) if size else m.get("size")
            print(f"pinned {mid}: {m['revision']} {m['sha256']} {m['size']}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"could not pin {mid}: {e}", file=sys.stderr)
    return failed


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--pin", action="store_true")
    args = ap.parse_args(argv)
    previous = json.loads(CATALOG.read_text(encoding="utf-8")) if CATALOG.is_file() else None
    catalog = build(previous)
    failed = pin(catalog) if args.pin else 0
    text = json.dumps(catalog, indent=2, ensure_ascii=False) + "\n"
    if args.check:
        current = CATALOG.read_text(encoding="utf-8") if CATALOG.is_file() else ""
        if current != text:
            print("config/model_catalog.json is out of date: run python tools/build_model_catalog.py", file=sys.stderr)
            return 1
        return 0
    CATALOG.write_text(text, encoding="utf-8", newline="\n")
    print(f"wrote {CATALOG.relative_to(ROOT)}: {len(catalog['models'])} models, {len(catalog['workflows'])} workflows")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
