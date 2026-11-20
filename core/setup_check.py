"""Radiance setup check: this machine, the models every workflow needs, and their download.

Used by the tour's "Set up" page and the Models page (js/studio.mjs) through
the /radiance/setup/* routes (nodes/pipeline/setup_routes.py).

* `hardware()`  GPU and its memory, RAM, free disk where the models go, and
  the versions that matter (Python, PyTorch, CUDA, ComfyUI, Radiance).
* `scan()`      every model in config/model_catalog.json, installed or not,
  and per shipped workflow what is missing and how much it is to download.
* `QUEUE`       one download at a time, in the background, with progress and
  cancel. Workflow models go through core/model_fetch.fetch(): pinned to a
  commit and a SHA-256 (from the catalog, or from the Hub's own metadata for
  the file at download time), resumable, gated repositories with the user's
  Hugging Face token. Radiance's own models go through their own downloaders
  (RUDRA from fxtdstudios/rudra-models into models/radiance, MoGe-2,
  Marigold, Depth Anything), which verify every file.

Nothing downloads unless the person asks for it on the page, and the usual
consent switch (RADIANCE_ALLOW_DOWNLOADS=0, HF_HUB_OFFLINE=1) still wins.
"""
from __future__ import annotations

import json
import logging
import os
import platform
import re
import shutil
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("radiance.setup_check")

ROOT = Path(__file__).resolve().parents[1]
CATALOG_PATH = ROOT / "config" / "model_catalog.json"

_HF_RESOLVE = re.compile(r"^https://huggingface\.co/([^/]+/[^/]+)/resolve/([^/]+)/(.+)$")


# ── Catalog ─────────────────────────────────────────────────────────────────

_catalog_cache: Dict[str, Any] = {}


def load_catalog() -> Dict[str, Any]:
    if not _catalog_cache:
        _catalog_cache.update(json.loads(CATALOG_PATH.read_text(encoding="utf-8")))
    return _catalog_cache


def is_pack(mid: str) -> bool:
    return mid.startswith("radiance/")


# ── Where models live ───────────────────────────────────────────────────────

def _folder_paths():
    try:
        import folder_paths  # type: ignore
        return folder_paths
    except ImportError:
        return None


def models_dir() -> Path:
    fp = _folder_paths()
    d = getattr(fp, "models_dir", None) if fp else None
    return Path(d) if d else ROOT.parents[1] / "models"


def radiance_dir(create: bool = False) -> Path:
    """ComfyUI/models/radiance, where Radiance keeps RUDRA and its other models."""
    d = models_dir() / "radiance"
    if create:
        d.mkdir(parents=True, exist_ok=True)
    return d


def _folder_dirs(folder: str) -> List[Path]:
    """Every directory ComfyUI reads `folder` models from (extra_model_paths included)."""
    fp = _folder_paths()
    dirs: List[Path] = []
    if fp is not None:
        try:
            dirs = [Path(p) for p in fp.get_folder_paths(folder)]
        except Exception:  # noqa: BLE001 - a folder name this ComfyUI does not register
            dirs = []
    default = models_dir() / folder
    if default not in dirs:
        dirs.append(default)
    return dirs


def destination(entry: Dict[str, Any]) -> Path:
    """Where a workflow model is downloaded to: the folder's first models directory."""
    return _folder_dirs(entry["folder"])[0] / entry["file"]


def find_installed(entry: Dict[str, Any]) -> Optional[Path]:
    for d in _folder_dirs(entry["folder"]):
        p = d / entry["file"]
        if p.is_file() and p.stat().st_size > 0:
            return p
    return None


# ── Radiance's own models ───────────────────────────────────────────────────

def _rudra(name: str):
    from radiance.model import rudra_download as rd
    return rd, rd.PACKAGES[name]


def _depth_repo(mid: str):
    from radiance.nodes.vfx.depth import DEPTH_MODELS, DEPTH_REVISIONS
    size = mid.rsplit("_", 1)[-1].capitalize()
    repo = next(r for k, r in DEPTH_MODELS.items() if k.startswith(size))
    return repo, DEPTH_REVISIONS.get(repo, "main")


def _pack_dir(mid: str) -> Optional[Path]:
    key = mid.split("/", 1)[1]
    if key in ("RUDRA_ACES_2", "rudra_genfill_v3", "wan_vace_base", "wan_vace_text_encoder"):
        rd, pkg = _rudra(key)
        return radiance_dir() / pkg.dirname
    if key == "moge2":
        return _folder_dirs("geometry_estimation")[0]
    if key.startswith("marigold_"):
        return radiance_dir() / "marigold"
    return None


def pack_installed(mid: str) -> bool:
    key = mid.split("/", 1)[1]
    try:
        if key in ("RUDRA_ACES_2", "rudra_genfill_v3", "wan_vace_base", "wan_vace_text_encoder"):
            rd, pkg = _rudra(key)
            return rd.find_package(pkg) is not None
        if key == "moge2":
            from radiance.nodes.vfx.multipass import estimate_models as em
            return em.find_moge() is not None
        if key.startswith("marigold_"):
            from radiance.nodes.vfx.multipass import estimate_models as em
            name = em.MARIGOLD_APPEARANCE if key.endswith("appearance") else em.MARIGOLD_LIGHTING
            return em.find_marigold(name) is not None
        if key.startswith("depth_anything_v2_"):
            from huggingface_hub import try_to_load_from_cache  # type: ignore
            repo, rev = _depth_repo(mid)
            hit = try_to_load_from_cache(repo, "model.safetensors", revision=rev)
            return isinstance(hit, str) and os.path.isfile(hit)
    except Exception as exc:  # noqa: BLE001 - an optional dependency missing reads as not installed
        logger.debug("[Radiance setup] %s: %s", mid, exc)
    return False


def _download_pack(mid: str, job: "Job") -> None:
    key = mid.split("/", 1)[1]
    if key in ("RUDRA_ACES_2", "rudra_genfill_v3", "wan_vace_base", "wan_vace_text_encoder"):
        rd, pkg = _rudra(key)
        radiance_dir(create=True)
        rd._failed.pop(pkg.name, None)        # a failure earlier in this process is not final here
        if rd.download_package(pkg, allow=True, force=True) is None:
            raise RuntimeError(f"{pkg.title} did not download; the ComfyUI console says why.")
        return
    if key == "moge2":
        from radiance.nodes.vfx.multipass import estimate_models as em
        em.ensure_moge(True)
        return
    if key.startswith("marigold_"):
        from radiance.nodes.vfx.multipass import estimate_models as em
        radiance_dir(create=True)
        em.ensure_marigold(em.MARIGOLD_APPEARANCE if key.endswith("appearance") else em.MARIGOLD_LIGHTING, True)
        return
    if key.startswith("depth_anything_v2_"):
        from huggingface_hub import snapshot_download  # type: ignore
        repo, rev = _depth_repo(mid)
        snapshot_download(repo_id=repo, revision=rev,
                          allow_patterns=["*.json", "*.safetensors", "*.txt"])
        return
    raise RuntimeError(f"unknown Radiance model {mid}")


# ── This machine ────────────────────────────────────────────────────────────

def _disk_free(path: Path) -> Optional[int]:
    p = path
    while not p.exists() and p != p.parent:
        p = p.parent
    try:
        return int(shutil.disk_usage(p).free)
    except OSError:
        return None


def hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {"os": f"{platform.system()} {platform.release()}", "python": platform.python_version(),
                            "gpus": [], "torch": None, "cuda": None}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda"] = getattr(torch.version, "cuda", None)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                free = None
                try:
                    free, _total = torch.cuda.mem_get_info(i)
                except Exception:  # noqa: BLE001
                    pass
                info["gpus"].append({"name": p.name, "kind": "cuda", "vram": int(p.total_memory),
                                     "vram_free": int(free) if free is not None else None,
                                     "capability": f"{p.major}.{p.minor}"})
        elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            info["gpus"].append({"name": "Apple GPU (Metal)", "kind": "mps", "vram": None, "vram_free": None})
    except Exception as exc:  # noqa: BLE001
        info["torch_error"] = str(exc)
    try:
        import psutil  # type: ignore
        vm = psutil.virtual_memory()
        info["ram"], info["ram_free"] = int(vm.total), int(vm.available)
    except Exception:  # noqa: BLE001 - psutil is optional
        info["ram"] = info["ram_free"] = None
    info["cpu"] = platform.processor() or platform.machine()
    info["cpu_count"] = os.cpu_count()
    info["disk_free"] = _disk_free(models_dir())
    try:
        import comfyui_version  # type: ignore
        info["comfyui"] = comfyui_version.__version__
    except Exception:  # noqa: BLE001
        info["comfyui"] = None
    try:
        from radiance.config.constants import VERSION
        info["radiance"] = VERSION
    except Exception:  # noqa: BLE001
        info["radiance"] = None
    return info


def gpu_memory(hw: Dict[str, Any]) -> Optional[int]:
    return max((g["vram"] for g in hw.get("gpus") or [] if g.get("vram")), default=None)


# ── The scan ────────────────────────────────────────────────────────────────

def model_status(mid: str) -> Dict[str, Any]:
    cat = load_catalog()
    if is_pack(mid):
        meta = cat["radiance_packs"][mid]
        d = _pack_dir(mid)
        return {"id": mid, "kind": "radiance", "title": meta["title"], "installed": pack_installed(mid),
                "size": int(meta["size_mb"]) * 1_000_000, "size_exact": False, "gated": False,
                "page": meta["page"], "licence": meta.get("licence", ""), "path": str(d) if d else ""}
    m = cat["models"][mid]
    hit = find_installed(m)
    dest = destination(m)
    part = Path(str(dest) + ".part")
    repo = m.get("repo") or ""
    return {"id": mid, "kind": "file", "title": m["file"], "folder": m["folder"], "installed": hit is not None,
            "path": str(hit or dest), "size": m.get("size"), "size_exact": m.get("size") is not None,
            "partial": part.stat().st_size if part.is_file() else 0, "gated": bool(m.get("gated")),
            "page": f"https://huggingface.co/{repo}" if repo else m["url"], "pinned": bool(m.get("sha256"))}


def scan(create_radiance_dir: bool = False, refresh: bool = False) -> Dict[str, Any]:
    from radiance.core import first_start
    from radiance.core.consent import downloads_allowed
    from radiance.core.model_fetch import hf_token
    cat = load_catalog()
    rdir = radiance_dir(create=create_radiance_dir)
    hw = hardware()
    vram = gpu_memory(hw)
    ids = sorted({i for w in cat["workflows"].values() for i in w["required"] + w["optional"]})
    status = {i: model_status(i) for i in ids}
    workflows = []
    for stem, w in cat["workflows"].items():
        missing = [i for i in w["required"] if not status[i]["installed"]]
        sizes = [status[i]["size"] for i in missing]
        known = all(s is not None for s in sizes)
        biggest = max((status[i]["size"] or 0 for i in w["required"] if not is_pack(i)
                       and cat["models"][i]["folder"] in ("diffusion_models", "checkpoints", "unet")), default=0)
        note = ""
        if vram and biggest and biggest > vram:
            note = "The main model is larger than this GPU's memory: ComfyUI offloads part of it to RAM (slower)."
        if vram and any(i in w["required"] for i in ("radiance/rudra_genfill_v3",)) and vram < 15.5e9:
            note = "The AI fill needs about 16 GB of GPU memory."
        workflows.append({
            "id": stem, "title": w["title"], "required": w["required"], "optional": w["optional"],
            "missing": missing, "ready": not missing,
            "missing_bytes": sum(s for s in sizes if s) if missing else 0, "missing_bytes_known": known,
            "gated": [i for i in missing if status[i].get("gated")], "gpu_note": note,
        })
    return {
        "hardware": hw,
        "paths": {"models": str(models_dir()), "radiance": str(rdir), "radiance_exists": rdir.is_dir()},
        "downloads_allowed": downloads_allowed(),
        "hf_token": bool(hf_token()),
        "models": status,
        "workflows": workflows,
        "core": [status.get("radiance/RUDRA_ACES_2") or model_status("radiance/RUDRA_ACES_2")],
        "summary": {"workflows": len(workflows), "ready": sum(1 for w in workflows if w["ready"])},
        "checks": first_start.checks(refresh),
    }


# ── Pinning an unpinned file at download time ───────────────────────────────

class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, *a, **k):
        return None

def _hub_head(url: str, token: str = "", timeout: float = 30.0, opener=None):
    """Headers of the Hub's answer for a resolve URL, following the Hub's own
    redirects (a renamed or differently-cased repository answers 307 with a
    relative Location and no file headers) until the answer names the file:
    X-Linked-Etag / X-Linked-Size for an LFS file. Redirects to the CDN are not
    followed; their headers are what is read. Raises urllib.error.HTTPError."""
    import urllib.parse
    headers = {"User-Agent": "radiance-comfyui"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    opener = opener or urllib.request.build_opener(_NoRedirect)
    for _ in range(6):
        req = urllib.request.Request(url, method="HEAD", headers=headers)
        try:
            resp = opener.open(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code not in (301, 302, 303, 307, 308):
                raise
            resp = e
        h = resp.headers
        code = getattr(resp, "code", None) or getattr(resp, "status", 200)
        loc = h.get("Location") or ""
        if h.get("X-Linked-Etag") or code not in (301, 302, 303, 307, 308) or not loc:
            return h
        nxt = urllib.parse.urljoin(url, loc)
        if urllib.parse.urlparse(nxt).netloc not in ("huggingface.co", "www.huggingface.co"):
            return h                                           # off to the CDN: these headers are the file's
        url = nxt
    return h


def hub_metadata(url: str, timeout: float = 30.0) -> Dict[str, Any]:
    """{revision, sha256, size} of a Hugging Face file, from the Hub's response
    headers (X-Repo-Commit, X-Linked-Etag = the LFS SHA-256, X-Linked-Size)."""
    from radiance.core.model_fetch import ModelFetchError, hf_token
    try:
        h = _hub_head(url, hf_token() or "", timeout)
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            m = _HF_RESOLVE.match(url)
            page = f"https://huggingface.co/{m.group(1)}" if m else url
            raise ModelFetchError(
                f"The repository is gated (HTTP {e.code}). Open {page}, accept its licence, then set "
                f"HF_TOKEN (or run `hf auth login`) and restart ComfyUI.") from e
        else:
            raise ModelFetchError(f"HTTP {e.code} from {url}") from e
    except (urllib.error.URLError, OSError) as e:
        raise ModelFetchError(f"Hugging Face could not be reached ({e}).") from e
    sha = (h.get("X-Linked-Etag") or h.get("ETag") or "").strip().strip('"').removeprefix("W/").strip('"')
    if not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ModelFetchError(f"the Hub did not give a SHA-256 for {url}; it was not downloaded.")
    # Only the file's own size: Content-Length here is a redirect's body (about 140 bytes).
    size = h.get("X-Linked-Size")
    return {"revision": h.get("X-Repo-Commit") or "", "sha256": sha, "size": int(size) if size else None}


def pinned_url(entry: Dict[str, Any], revision: str) -> str:
    m = _HF_RESOLVE.match(entry["url"])
    if not m or not revision:
        return entry["url"]
    return f"https://huggingface.co/{m.group(1)}/resolve/{revision}/{m.group(3)}"


# ── Downloads ───────────────────────────────────────────────────────────────

class Job:
    def __init__(self, mid: str, title: str, total: Optional[int]):
        self.id, self.title, self.total = mid, title, total
        self.done = 0
        self.status = "queued"            # queued, running, done, failed, cancelled
        self.error = ""
        self.cancel = False
        self.started = self.finished = 0.0

    def as_dict(self) -> Dict[str, Any]:
        return {"id": self.id, "title": self.title, "status": self.status, "done": self.done,
                "total": self.total, "error": self.error, "started": self.started, "finished": self.finished}


class DownloadQueue:
    """One download at a time, on a daemon thread; the page polls `jobs()`."""

    def __init__(self, fetcher: Optional[Callable] = None, pack_downloader: Optional[Callable] = None,
                 metadata: Optional[Callable] = None):
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}
        self._order: List[str] = []
        self._thread: Optional[threading.Thread] = None
        self._fetch = fetcher
        self._pack = pack_downloader or _download_pack
        self._meta = metadata or hub_metadata

    def enqueue(self, ids: List[str]) -> List[str]:
        from radiance.core.consent import ALLOW_ENV, downloads_allowed
        if not downloads_allowed():
            raise PermissionError(f"Downloads are off ({ALLOW_ENV}=0 or an offline flag is set).")
        cat = load_catalog()
        added = []
        with self._lock:
            for mid in ids:
                if not (is_pack(mid) and mid in cat["radiance_packs"]) and mid not in cat["models"]:
                    raise KeyError(mid)
                cur = self._jobs.get(mid)
                if cur and cur.status in ("queued", "running"):
                    continue
                st = model_status(mid)
                if st["installed"]:
                    continue
                self._jobs[mid] = Job(mid, st["title"], st["size"])
                if mid in self._order:
                    self._order.remove(mid)
                self._order.append(mid)
                added.append(mid)
            if added and (self._thread is None or not self._thread.is_alive()):
                self._thread = threading.Thread(target=self._work, name="radiance-setup-downloads", daemon=True)
                self._thread.start()
        return added

    def cancel(self, mid: str) -> bool:
        with self._lock:
            job = self._jobs.get(mid)
            if not job or job.status not in ("queued", "running"):
                return False
            job.cancel = True
            if job.status == "queued":
                job.status, job.finished = "cancelled", time.time()
            return True

    def jobs(self) -> List[Dict[str, Any]]:
        with self._lock:
            out = []
            for mid in self._order:
                j = self._jobs[mid]
                if j.status == "running" and is_pack(mid):
                    d = _pack_dir(mid)
                    if d is not None and d.is_dir():
                        j.done = sum(f.stat().st_size for f in d.rglob("*") if f.is_file())
                out.append(j.as_dict())
            return out

    def _next(self) -> Optional[Job]:
        with self._lock:
            for mid in self._order:
                j = self._jobs[mid]
                if j.status == "queued":
                    j.status, j.started = "running", time.time()
                    return j
        return None

    def _work(self) -> None:
        while True:
            job = self._next()
            if job is None:
                return
            try:
                if is_pack(job.id):
                    self._pack(job.id, job)
                else:
                    self._download_file(job)
                job.status = "cancelled" if job.cancel else "done"
                if job.status == "done" and job.total:
                    job.done = job.total
            except Exception as exc:  # noqa: BLE001 - reported on the page
                job.status = "cancelled" if job.cancel else "failed"
                job.error = str(exc).replace("[Radiance] ", "").strip()
                logger.warning("[Radiance setup] %s: %s", job.id, job.error)
            job.finished = time.time()

    def _download_file(self, job: Job) -> None:
        from radiance.core import model_fetch
        entry = dict(load_catalog()["models"][job.id])
        if not entry.get("sha256"):
            entry.pop("size", None)           # a size with no digest was not read from the file
            entry.update({k: v for k, v in self._meta(entry["url"]).items() if v})
        if entry.get("size"):
            job.total = int(entry["size"])
        dest = destination(entry)
        free = _disk_free(dest.parent)
        if job.total and free is not None and free < job.total - (Path(str(dest) + ".part").stat().st_size
                                                                   if Path(str(dest) + ".part").is_file() else 0):
            raise OSError(f"Not enough free disk space in {dest.parent}: {job.total / 1e9:.1f} GB needed, "
                          f"{free / 1e9:.1f} GB free.")
        dest.parent.mkdir(parents=True, exist_ok=True)

        def progress(done: int, total: int) -> None:
            job.done = done
            if total:
                job.total = total
        fetch = self._fetch or model_fetch.fetch
        fetch(pinned_url(entry, entry.get("revision", "")), str(dest), sha256=entry["sha256"],
              size=entry.get("size"), label=entry["file"], progress=progress,
              should_cancel=lambda: job.cancel)


QUEUE = DownloadQueue()


def missing_for(workflow: str, include_optional: bool = False) -> List[str]:
    cat = load_catalog()
    w = cat["workflows"][workflow]
    ids = w["required"] + (w["optional"] if include_optional else [])
    return [i for i in ids if not model_status(i)["installed"]]
