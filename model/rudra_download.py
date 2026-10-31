"""First-use download of everything SDR → HDR needs, so it works on a fresh install.

Every file is pinned to a Hugging Face commit and checked by size and hash
(SHA-256 for weights, the git blob id for small text files) before it is
moved into place; a large file resumes where an interrupted download stopped.

``RUDRA_ACES_2``      RUDRA ACES 2, the learned ACES 2.0 reconstruction
                      (fxtdstudios/rudra-models v1.0.0, Apache 2.0, 2.2 MB).
                      Used by SDR → HDR Universal and Recover. Also fetched at
                      install time by ``install.py``.
``rudra_genfill_v3``  RUDRA AI fill v3: the FXTD LoRA, brightness estimator
                      and cached base-prompt embedding (Apache 2.0, 271 MB).
``Wan2.1-VACE-1.3B-diffusers``
                      the AI fill's base model (Wan-AI, Apache 2.0): VAE,
                      transformer, scheduler and tokenizer (7.7 GB). The umT5
                      text encoder (11.4 GB) is a separate package, fetched only
                      the first time a custom prompt has to be encoded.

The AI fill packages download when the AI Fill node first runs on a frame with
clipped highlights, never at startup. Everything lands in
``<ComfyUI>/models/radiance/<folder>/``. ``RADIANCE_ALLOW_DOWNLOADS=0`` (or
``HF_HUB_OFFLINE=1``) turns every download off; the nodes then say which files
to put where.

The download core (:func:`download_to`) uses only the standard library, so
``install.py`` can run it before Radiance is importable.
"""
from __future__ import annotations

import hashlib
import logging
import os
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

logger = logging.getLogger("radiance.model.rudra_download")

REPO = "fxtdstudios/rudra-models"
TAG = "v1.0.0"
REVISION = "78a0b21bdc606fd78b51ab279bf3c6672ee812aa"   # what tag v1.0.0 points at
LICENSE_URL = f"https://huggingface.co/{REPO}/blob/{TAG}/LICENSE"

WAN_REPO = "Wan-AI/Wan2.1-VACE-1.3B-diffusers"
WAN_REVISION = "ec4d2cb062b548996b179d493fdd05340de702a1"

_TIMEOUT_S = 60
_CHUNK = 1 << 20
_RETRIES = 3


@dataclass(frozen=True)
class Package:
    name: str                                 # key, and the folder name unless `folder` is set
    title: str
    repo_dir: str                             # folder inside the repository ('' = its root)
    files: Tuple[Tuple[str, int, str], ...]   # (relative path, size, sha256 or git blob sha1)
    size_mb: int
    default_allowed: bool
    repo: str = REPO
    revision: str = REVISION
    folder: str = ""
    licence: str = "Apache 2.0"

    @property
    def dirname(self) -> str:
        return self.folder or self.name

    @property
    def page_url(self) -> str:
        sub = f"/{self.repo_dir}" if self.repo_dir else ""
        return f"https://huggingface.co/{self.repo}/tree/{self.revision}{sub}"

    def url(self, rel: str) -> str:
        sub = f"{self.repo_dir}/" if self.repo_dir else ""
        return f"https://huggingface.co/{self.repo}/resolve/{self.revision}/{sub}{rel}"


ACES2 = Package(
    name="RUDRA_ACES_2", title="RUDRA ACES 2", repo_dir="models/RUDRA_ACES_2",
    files=(
        ("manifest.json", 3934, "aade83bb28cbd5607226484daf9f331dc9a9a21ced77be236ba5d777c96c47da"),
        ("parity.json", 1408, "ef1295d85c93000df4c44be7615830e8d23d3d107b8a1a17f8c66b9e454682a1"),
        ("rudra_g1.ts", 1037772, "fb30e3df748f4fa9ae17f7db189c8fe49a58d9863af6c1c45a408a72f3b7a59b"),
        ("rudra_g1.onnx", 1157294, "8ba97cb0ed36f5033958e752afad95d585d50bcefefc3bc66a61b27a6618f237"),
    ),
    size_mb=3, default_allowed=True,
)

GENFILL_V3 = Package(
    name="rudra_genfill_v3", title="RUDRA AI fill v3", repo_dir="models/rudra_genfill_v3",
    files=(
        ("brightness.json", 34855, "34892fb840de138fb3ddd2e2c8e6e3b13c232973ed59ea23338444fcaff2a95a"),
        ("prompt_cache/c9138bdd2a8e8feabc812558.json", 133,
         "995d82dd6f27b0b3de6bd526d585a8b12aa6ae461fa0de9d1ccc63b7c2653150"),
        ("prompt_cache/c9138bdd2a8e8feabc812558.pt", 8390432,
         "6af2658b9e0c99ed3e3446eae0484fd84ec2bdeee3bb90d1f9312eaae3202fe6"),
        ("rudra_genfill.safetensors", 262570552,
         "9fff7eb8492ae62619a126d302d9c940403b2e4aa03e02daf79d98142008238a"),
    ),
    size_mb=271, default_allowed=True,
)

WAN_DIRNAME = "Wan2.1-VACE-1.3B-diffusers"

WAN_BASE = Package(
    name="wan_vace_base", title="Wan2.1-VACE-1.3B (AI fill base model)", repo_dir="",
    files=(
        ("model_index.json", 408, "f3d1c8ba9a419e17bdbe301508dd7417614089f3"),
        ("scheduler/scheduler_config.json", 751, "9bf3f38c31c3621c45cf6f2b0dd57a80cd7052fb"),
        ("tokenizer/special_tokens_map.json", 7079, "2ed25bf989a28d20b5d4b5822fbc24666d12a6f7"),
        ("tokenizer/spiece.model", 4548313, "e3909a67b780650b35cf529ac782ad2b6b26e6d1f849d3fbb6a872905f452458"),
        ("tokenizer/tokenizer.json", 16837459, "20a46ac256746594ed7e1e3ef733b83fbc5a6f0922aa7480eda961743de080ef"),
        ("tokenizer/tokenizer_config.json", 61786, "847870cb2653176fbb3d4661b7ca971c1251ce55"),
        ("transformer/config.json", 662, "02c41b34934400784826e4c4871a4f0bd4dc2380"),
        ("transformer/diffusion_pytorch_model.safetensors.index.json", 114130,
         "1a6ddd856b5f37550fab21400eaebbd0d2fa3495"),
        ("transformer/diffusion_pytorch_model-00001-of-00002.safetensors", 4999964504,
         "192804a4e10b5bb0a13f5c224bc4ec9707b3b8cc0def8eea005dbce7c9d6752a"),
        ("transformer/diffusion_pytorch_model-00002-of-00002.safetensors", 2146103400,
         "dadaf1d5e9ccf826e2c497aa9fc2756a044f606d0aa919e5405a734a121bcf7e"),
        ("vae/config.json", 724, "a5be94e360479a8b513ced12e5d5032bcbd9e60d"),
        ("vae/diffusion_pytorch_model.safetensors", 507591892,
         "d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793"),
    ),
    size_mb=7700, default_allowed=True, repo=WAN_REPO, revision=WAN_REVISION, folder=WAN_DIRNAME,
)

WAN_TEXT_ENCODER = Package(
    name="wan_vace_text_encoder", title="umT5 text encoder (AI fill custom prompts)", repo_dir="",
    files=(
        ("text_encoder/config.json", 850, "6847b0fc2756dd0f1278fd1dde852fdeb1bb6622"),
        ("text_encoder/model.safetensors.index.json", 22476, "f3d3d4da90eb33e14c92d88ea346370fa3c0b5b2"),
        ("text_encoder/model-00001-of-00003.safetensors", 4935812536,
         "a8e861969c7433e707cc5a74065d795d36cca07ec96eb6763eb4083df7248f58"),
        ("text_encoder/model-00002-of-00003.safetensors", 4983103192,
         "d57d948ece4837d850b7a859a4415121d57cacf8b9ee1d4db200c67f592902d7"),
        ("text_encoder/model-00003-of-00003.safetensors", 1442935480,
         "0da9ee284e21d1406df708788db1d502d95d75f69faa25cd26151bf8829b7c5f"),
    ),
    size_mb=11400, default_allowed=True, repo=WAN_REPO, revision=WAN_REVISION, folder=WAN_DIRNAME,
)

PACKAGES: Dict[str, Package] = {p.name: p for p in (ACES2, GENFILL_V3, WAN_BASE, WAN_TEXT_ENCODER)}

_lock = threading.Lock()
_failed: Dict[str, bool] = {}
_refusal_logged: Dict[str, bool] = {}


# ─────────────────────────────────────────────────────────────────────────────
#  verification (stdlib only)
# ─────────────────────────────────────────────────────────────────────────────

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def file_digest(path: Path, expected: str) -> str:
    """The digest of *path* in the form *expected* uses: SHA-256 (64 hex) or
    the git blob SHA-1 Hugging Face lists for small text files (40 hex)."""
    if len(expected) == 40:
        h = hashlib.sha1()
        h.update(f"blob {Path(path).stat().st_size}\0".encode())
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(_CHUNK), b""):
                h.update(chunk)
        return h.hexdigest()
    return sha256_file(path)


def package_complete(folder: Path, pkg: Package, *, verify_hash: bool = False) -> bool:
    """Every file of *pkg* is in *folder* with its pinned size (and hash, if asked)."""
    for rel, size, digest in pkg.files:
        f = Path(folder) / rel
        try:
            if not f.is_file() or f.stat().st_size != size:
                return False
        except OSError:
            return False
        if verify_hash and file_digest(f, digest) != digest:
            return False
    return True


# ─────────────────────────────────────────────────────────────────────────────
#  download core (stdlib only)
# ─────────────────────────────────────────────────────────────────────────────

def _fetch(url: str, dest: Path, size: int = 0, report: Optional[Callable[[int], None]] = None) -> None:
    """Stream *url* to *dest*, resuming a partial *dest*. urllib honours HTTPS_PROXY."""
    have = dest.stat().st_size if dest.exists() else 0
    if size and have >= size:
        have = 0
        dest.unlink()
    headers = {"User-Agent": "radiance-comfyui"}
    if have:
        headers["Range"] = f"bytes={have}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:
        resumed = have and getattr(resp, "status", 200) == 206
        with open(dest, "ab" if resumed else "wb") as out:
            done = have if resumed else 0
            while True:
                chunk = resp.read(_CHUNK)
                if not chunk:
                    break
                out.write(chunk)
                done += len(chunk)
                if report:
                    report(done)


def download_to(pkg: Package, target: Path, log: Callable[[str], None] = print) -> Path:
    """Fetch every missing or wrong file of *pkg* into *target*; raises on failure.

    Verified files are kept, so a second call after an interruption only
    fetches what is left, and a large file resumes from its ``.part``.
    """
    target = Path(target)
    target.mkdir(parents=True, exist_ok=True)
    todo = []
    for rel, size, digest in pkg.files:
        f = target / rel
        if f.is_file() and f.stat().st_size == size:
            continue
        todo.append((rel, size, digest))
    total = sum(s for _, s, _ in todo)
    if not todo:
        return target
    log(f"[Radiance] Downloading {pkg.title} ({total / 1e6:,.0f} MB) to {target}")
    fetched = 0
    for rel, size, digest in todo:
        final = target / rel
        final.parent.mkdir(parents=True, exist_ok=True)
        part = final.with_name(final.name + ".part")
        last = [time.monotonic()]

        def report(done, _base=fetched, _size=size, _rel=rel):
            now = time.monotonic()
            if _size > 50_000_000 and now - last[0] > 10:
                last[0] = now
                log(f"[Radiance]   {_rel}: {done / 1e6:,.0f} / {_size / 1e6:,.0f} MB "
                    f"({100 * (_base + done) / max(total, 1):.0f} % of {pkg.title})")

        error = None
        for attempt in range(_RETRIES):
            try:
                _fetch(pkg.url(rel), part, size, report)
                got = part.stat().st_size
                if got != size:
                    raise ValueError(f"{rel}: size {got} bytes, expected {size}")
                got_digest = file_digest(part, digest)
                if got_digest != digest:
                    part.unlink()
                    raise ValueError(f"{rel}: hash {got_digest}, expected {digest}")
                os.replace(part, final)
                error = None
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as exc:
                error = exc            # network: retry, resuming from the .part
                time.sleep(2 * (attempt + 1))
            except ValueError as exc:
                error = exc            # wrong size or hash: the .part is gone, retry from zero
        if error is not None:
            raise RuntimeError(f"{pkg.title}: {error}") from error
        fetched += size
    log(f"[Radiance] Installed {pkg.title} in {target} ({pkg.licence})")
    return target


# ─────────────────────────────────────────────────────────────────────────────
#  Radiance side: where packages live, consent
# ─────────────────────────────────────────────────────────────────────────────

def find_package(pkg: Package, explicit: str = "", env: Optional[str] = None) -> Optional[Path]:
    """The installed package folder: *explicit*, then *env*, then every models/radiance dir."""
    from radiance.model.paths import radiance_model_dirs
    candidates = []
    for value in (explicit, os.environ.get(env, "") if env else ""):
        value = (value or "").strip().strip('"').strip("'")
        if value:
            p = Path(value).expanduser()
            candidates.append(p.parent if p.is_file() else p)
    candidates += [d / pkg.dirname for d in radiance_model_dirs()]
    for c in candidates:
        if package_complete(c, pkg):
            return c
    return None


def _target_dir(pkg: Package) -> Optional[Path]:
    from radiance.model.paths import radiance_model_dirs
    for d in radiance_model_dirs():
        try:
            target = d / pkg.dirname
            target.mkdir(parents=True, exist_ok=True)
            probe = target / ".radiance_write_probe"
            probe.write_bytes(b"")
            probe.unlink()
            return target
        except OSError:
            continue
    return None


def downloads_allowed_for(pkg: Package, allow: Optional[bool] = None) -> bool:
    from radiance.core.consent import downloads_allowed
    default = pkg.default_allowed if allow is None else bool(allow)
    return downloads_allowed(default=default)


def download_package(pkg: Package, *, force: bool = False, allow: Optional[bool] = None,
                     target: Optional[Path] = None) -> Optional[Path]:
    """Download *pkg* into ``models/radiance/<folder>`` (or *target*); the folder, or None.

    *allow* overrides the package's default when nothing in the environment
    says otherwise (a node's download switch). ``RADIANCE_ALLOW_DOWNLOADS=0``
    always wins. After one failure the process does not try again unless
    *force* is set.
    """
    from radiance.core.consent import refusal_message

    if not downloads_allowed_for(pkg, allow):
        if not _refusal_logged.get(pkg.name):
            _refusal_logged[pkg.name] = True
            logger.warning(refusal_message(
                pkg.title, size_mb=pkg.size_mb, dest=f"ComfyUI/models/radiance/{pkg.dirname}/",
                url=pkg.page_url,
            ))
        return None

    with _lock:
        if _failed.get(pkg.name) and not force:
            return None
        target = Path(target) if target is not None else _target_dir(pkg)
        if target is None:
            logger.error("[Radiance] No writable models/radiance folder for %s. Download it from %s "
                         "into ComfyUI/models/radiance/%s/.", pkg.title, pkg.page_url, pkg.dirname)
            _failed[pkg.name] = True
            return None
        try:
            return download_to(pkg, target, log=logger.info)
        except Exception as exc:  # noqa: BLE001 - network, disk, verification
            _failed[pkg.name] = True
            logger.error("[Radiance] Could not download %s: %s\n"
                         "           Download it manually from %s\n"
                         "           and put its files in %s", pkg.title, exc, pkg.page_url, target)
            return None


def _reset_for_tests() -> None:
    _failed.clear()
    _refusal_logged.clear()
