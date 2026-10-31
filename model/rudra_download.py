"""First-use download of the RUDRA 1.0 model packages (Apache 2.0).

Two packages from ``fxtdstudios/rudra-models`` at tag ``v1.0.0``, pinned to
the commit that tag points at, every file checked by size and SHA-256 before
it is moved into place:

``RUDRA_ACES_2``      the learned ACES 2.0 reconstruction (2.2 MB): a
                      TorchScript network stage (``rudra_g1.ts``), its ONNX
                      twin and the staged-contract manifest. Small and
                      first-party, so it downloads by default like the old
                      pixel checkpoint did.
``rudra_genfill_v3``  AI fill v3 (271 MB): the FXTD LoRA for
                      Wan2.1-VACE-1.3B, the brightness estimator and the base
                      prompt's cached embeddings. Large, so it downloads only
                      when ``RADIANCE_ALLOW_DOWNLOADS=1`` is set or the node's
                      own download switch is on.

Both land in ``<models>/radiance/<package>/``. Turn every download off with
``RADIANCE_ALLOW_DOWNLOADS=0`` (or ``HF_HUB_OFFLINE=1``).

The module imports no torch, so it is testable in the lightweight CI lane.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger("radiance.model.rudra_download")

REPO = "fxtdstudios/rudra-models"
TAG = "v1.0.0"
REVISION = "78a0b21bdc606fd78b51ab279bf3c6672ee812aa"   # what tag v1.0.0 points at
LICENSE_URL = f"https://huggingface.co/{REPO}/blob/{TAG}/LICENSE"

_TIMEOUT_S = 60
_CHUNK = 1 << 20


@dataclass(frozen=True)
class Package:
    name: str
    title: str
    repo_dir: str
    files: Tuple[Tuple[str, int, str], ...]   # (relative path, size, sha256)
    size_mb: int
    default_allowed: bool

    @property
    def page_url(self) -> str:
        return f"https://huggingface.co/{REPO}/tree/{TAG}/{self.repo_dir}"

    def url(self, rel: str) -> str:
        return f"https://huggingface.co/{REPO}/resolve/{REVISION}/{self.repo_dir}/{rel}"


ACES2 = Package(
    name="RUDRA_ACES_2",
    title="RUDRA ACES 2",
    repo_dir="models/RUDRA_ACES_2",
    files=(
        ("manifest.json", 3934, "aade83bb28cbd5607226484daf9f331dc9a9a21ced77be236ba5d777c96c47da"),
        ("parity.json", 1408, "ef1295d85c93000df4c44be7615830e8d23d3d107b8a1a17f8c66b9e454682a1"),
        ("rudra_g1.ts", 1037772, "fb30e3df748f4fa9ae17f7db189c8fe49a58d9863af6c1c45a408a72f3b7a59b"),
        ("rudra_g1.onnx", 1157294, "8ba97cb0ed36f5033958e752afad95d585d50bcefefc3bc66a61b27a6618f237"),
    ),
    size_mb=3,
    default_allowed=True,
)

GENFILL_V3 = Package(
    name="rudra_genfill_v3",
    title="RUDRA AI fill v3",
    repo_dir="models/rudra_genfill_v3",
    files=(
        ("brightness.json", 34855, "34892fb840de138fb3ddd2e2c8e6e3b13c232973ed59ea23338444fcaff2a95a"),
        ("prompt_cache/c9138bdd2a8e8feabc812558.json", 133,
         "995d82dd6f27b0b3de6bd526d585a8b12aa6ae461fa0de9d1ccc63b7c2653150"),
        ("prompt_cache/c9138bdd2a8e8feabc812558.pt", 8390432,
         "6af2658b9e0c99ed3e3446eae0484fd84ec2bdeee3bb90d1f9312eaae3202fe6"),
        ("rudra_genfill.safetensors", 262570552,
         "9fff7eb8492ae62619a126d302d9c940403b2e4aa03e02daf79d98142008238a"),
    ),
    size_mb=271,
    default_allowed=False,
)

PACKAGES: Dict[str, Package] = {p.name: p for p in (ACES2, GENFILL_V3)}

_lock = threading.Lock()
_failed: Dict[str, bool] = {}
_refusal_logged: Dict[str, bool] = {}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def package_complete(folder: Path, pkg: Package, *, verify_hash: bool = False) -> bool:
    """Every file of *pkg* is in *folder* with its pinned size (and hash, if asked)."""
    for rel, size, digest in pkg.files:
        f = folder / rel
        try:
            if not f.is_file() or f.stat().st_size != size:
                return False
        except OSError:
            return False
        if verify_hash and sha256_file(f) != digest:
            return False
    return True


def find_package(pkg: Package, explicit: str = "", env: Optional[str] = None) -> Optional[Path]:
    """The installed package folder: *explicit*, then *env*, then every models/radiance dir."""
    from radiance.model.paths import radiance_model_dirs
    candidates = []
    for value in (explicit, os.environ.get(env, "") if env else ""):
        value = (value or "").strip().strip('"').strip("'")
        if value:
            p = Path(value).expanduser()
            candidates.append(p.parent if p.is_file() else p)
    candidates += [d / pkg.name for d in radiance_model_dirs()]
    for c in candidates:
        if package_complete(c, pkg):
            return c
    return None


def _target_dir(pkg: Package) -> Optional[Path]:
    from radiance.model.paths import radiance_model_dirs
    for d in radiance_model_dirs():
        try:
            target = d / pkg.name
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".radiance_write_probe"
            probe.write_bytes(b"")
            probe.unlink()
            return target
        except OSError:
            continue
    return None


def _fetch(url: str, dest: Path) -> None:
    """Stream *url* to *dest*. urllib honours HTTPS_PROXY / HTTP_PROXY."""
    req = urllib.request.Request(url, headers={"User-Agent": "radiance-comfyui"})
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp, open(dest, "wb") as out:
        while True:
            chunk = resp.read(_CHUNK)
            if not chunk:
                break
            out.write(chunk)


def download_package(pkg: Package, *, force: bool = False, allow: Optional[bool] = None) -> Optional[Path]:
    """Download *pkg* into ``models/radiance/<name>``; the folder, or None.

    *allow* overrides the package's default when nothing in the environment
    says otherwise (the AI fill node's download switch). ``RADIANCE_ALLOW_DOWNLOADS=0``
    always wins. After one failure the process does not try again unless
    *force* is set.
    """
    from radiance.core.consent import downloads_allowed, refusal_message

    default = pkg.default_allowed if allow is None else bool(allow)
    if not downloads_allowed(default=default):
        if not _refusal_logged.get(pkg.name):
            _refusal_logged[pkg.name] = True
            logger.warning(refusal_message(
                pkg.title, size_mb=pkg.size_mb, dest=f"ComfyUI/models/radiance/{pkg.name}/",
                url=pkg.page_url,
            ))
        return None

    with _lock:
        if _failed.get(pkg.name) and not force:
            return None
        target = _target_dir(pkg)
        if target is None:
            logger.error("[Radiance] No writable models/radiance folder for %s. Download it from %s "
                         "into ComfyUI/models/radiance/%s/.", pkg.title, pkg.page_url, pkg.name)
            _failed[pkg.name] = True
            return None
        logger.info("[Radiance] Downloading %s (~%d MB) to %s", pkg.title, pkg.size_mb, target)
        for rel, size, digest in pkg.files:
            final = target / rel
            if final.is_file() and final.stat().st_size == size and sha256_file(final) == digest:
                continue
            final.parent.mkdir(parents=True, exist_ok=True)
            part = final.with_name(final.name + ".part")
            try:
                _fetch(pkg.url(rel), part)
                got = part.stat().st_size
                if got != size:
                    raise ValueError(f"{rel}: size {got} bytes, expected {size}")
                got_digest = sha256_file(part)
                if got_digest != digest:
                    raise ValueError(f"{rel}: sha256 {got_digest}, expected {digest}")
                os.replace(part, final)
            except Exception as exc:  # noqa: BLE001 - network, disk, verification
                _failed[pkg.name] = True
                try:
                    part.unlink()
                except OSError:
                    pass
                logger.error("[Radiance] Could not download %s: %s\n"
                             "           Download it manually from %s\n"
                             "           and put its files in %s", pkg.title, exc, pkg.page_url, target)
                return None
        logger.info("[Radiance] Installed %s in %s (Apache 2.0: %s)", pkg.title, target, LICENSE_URL)
        return target


def _reset_for_tests() -> None:
    _failed.clear()
    _refusal_logged.clear()
