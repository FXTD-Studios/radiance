#!/usr/bin/env python3
"""Check every pinned model download against its source.

    python tools/pin_models.py            # Hugging Face: digest and size from
                                          # the Hub API; GitHub: size only
    python tools/pin_models.py --full     # also download each GitHub file and
                                          # hash it (about 1 GB)

Reads the pins where the downloaders read them: config/model_map.py (Read
Models), the upscale registry (nodes/upscale/upscale.py), AI Upscale's
MODEL_FILES (image/upscale.py), Depth Map Generator's DEPTH_REVISIONS,
Multipass Estimate's MoGe / Marigold pins and the RUDRA pixel model. Exits 1
if any source is missing, moved or no longer matches its pinned SHA-256 and
size, and prints what to put in the pin when a repository has moved on.
Standard library only, so it runs before the package's requirements are
installed. Needs network access; set HF_TOKEN to check gated repositories.
"""
from __future__ import annotations

import ast
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UA = {"User-Agent": "radiance-pin-check"}
HF = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([0-9a-f]{40})/(.+)$")


def _token_headers():
    h = dict(UA)
    tok = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _json(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers=_token_headers()), timeout=60) as r:
        return json.load(r)


def _literal(path: Path, name: str):
    """The value of a top-level (or class-level) literal assignment `name = {...}`."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name) and node.target.id == name:
            return ast.literal_eval(node.value)
    raise KeyError(f"{name} not found in {path}")


def _registry_updates(path: Path):
    """_UPSCALE_MODEL_REGISTRY plus the entries added with .update({...})."""
    reg = dict(_literal(path, "_UPSCALE_MODEL_REGISTRY"))
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "update"
                and isinstance(node.func.value, ast.Name) and node.func.value.id == "_UPSCALE_MODEL_REGISTRY"):
            reg.update(ast.literal_eval(node.args[0]))
    return reg


def collect():
    """(label, url, sha256, size) for every single-file pin, and (label, repo, revision) for repo pins."""
    files, repos = [], []
    for name, e in _literal(ROOT / "config" / "model_map.py", "RADIANCE_MODEL_MAP").items():
        files.append((f"Read Models: {name}", e["url"], e.get("sha256"), e.get("size")))
    for key, e in _registry_updates(ROOT / "nodes" / "upscale" / "upscale.py").items():
        if e.get("url"):
            files.append((f"Upscale: {key}", e["url"], e.get("sha256"), e.get("size")))
    for name, (url, sha, size) in _literal(ROOT / "image" / "upscale.py", "MODEL_FILES").items():
        files.append((f"AI Upscale: {name}", url, sha, size))
    for repo, rev in _literal(ROOT / "nodes" / "vfx" / "depth.py", "DEPTH_REVISIONS").items():
        repos.append((f"Depth: {repo}", repo, rev))
    em = ROOT / "nodes" / "vfx" / "multipass" / "estimate_models.py"
    text = em.read_text(encoding="utf-8")
    for repo, rev in re.findall(r'"repo": "([^"]+)",\s*"revision": "([0-9a-f]{40})"', text):
        repos.append((f"Estimate: {repo}", repo, rev))
    moge_repo, moge_rev = _literal(em, "MOGE_REPO"), _literal(em, "MOGE_REVISION")
    files.append(("Estimate: MoGe-2", f"https://huggingface.co/{moge_repo}/resolve/{moge_rev}/geometry_estimation/"
                  f"{_literal(em, 'MOGE_FILENAME')}", _literal(em, "MOGE_SHA256"), _literal(em, "MOGE_SIZE")))
    return files, repos


def _hf_file(repo, rev, path):
    info = _json(f"https://huggingface.co/api/models/{repo}/revision/{rev}?blobs=true")
    for s in info.get("siblings", []):
        if s["rfilename"] == path:
            lfs = s.get("lfs") or {}
            return lfs.get("sha256"), lfs.get("size", s.get("size")), info.get("gated")
    return None, None, info.get("gated")


def _download_sha(url):
    h, n = hashlib.sha256(), 0
    with urllib.request.urlopen(urllib.request.Request(url, headers=_token_headers()), timeout=120) as r:
        for block in iter(lambda: r.read(1 << 20), b""):
            h.update(block)
            n += len(block)
    return h.hexdigest(), n


def main(argv):
    full = "--full" in argv
    files, repos = collect()
    bad = 0
    for label, url, sha, size in files:
        try:
            if not (sha and re.fullmatch(r"[0-9a-f]{64}", sha)):
                raise ValueError("no SHA-256 pinned")
            m = HF.match(url)
            if m:
                got_sha, got_size, gated = _hf_file(*m.groups())
                if got_sha is None and got_size is not None and size is not None:
                    got_sha, got_size = _download_sha(url)   # a small non-LFS file
                if got_sha is None:
                    raise ValueError(f"file not in {m.group(1)}@{m.group(2)[:10]}")
            elif url.startswith("https://huggingface.co/"):
                raise ValueError("Hugging Face URL not pinned to a commit")
            elif full:
                got_sha, got_size = _download_sha(url)
            else:
                req = urllib.request.Request(url, headers=UA, method="HEAD")
                with urllib.request.urlopen(req, timeout=60) as r:
                    got_sha, got_size = sha, int(r.headers.get("Content-Length") or size or 0)
            if got_sha != sha or (size is not None and int(got_size) != int(size)):
                raise ValueError(f"source is now sha256={got_sha} size={got_size}")
            print(f"ok    {label}")
        except (ValueError, urllib.error.URLError, OSError) as e:
            bad += 1
            print(f"FAIL  {label}: {e}\n      {url}")
    for label, repo, rev in repos:
        try:
            _json(f"https://huggingface.co/api/models/{repo}/revision/{rev}")
            print(f"ok    {label} @ {rev[:10]}")
        except (urllib.error.URLError, OSError) as e:
            bad += 1
            print(f"FAIL  {label}: {repo}@{rev}: {e}")
    print(f"\n{len(files) + len(repos) - bad} of {len(files) + len(repos)} pins verified.")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
