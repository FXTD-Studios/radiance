"""First-start checks shown on the Models page and the tour's Set up page (4.0).

Each check answers one question a new user would otherwise ask on day one:

    ocio     Is there an OCIO config, and does it have the ACES 2.0 SDR and
             HDR views the viewer and the HDR nodes use?
    ffmpeg   Can Write make 10-bit H.265 and ProRes?
    output   Does a new Write node's default folder (~/radiance_output) exist and
             take files? (The shipped workflows write under ComfyUI's output.)
    hub      Is huggingface.co reachable, is a token set (yes/no only), and
             which licences for gated models still need accepting?

The display and WebGL check runs in the browser (js/studio_model.mjs).

A check is ``{"id", "title", "status": "ok"|"warn"|"bad", "detail", "fix"}``;
``fix`` is ``None``, ``{"action": name, "label": text}`` for a fix the server
can make (POST /radiance/setup/fix), or ``{"url": ..., "label": ...}`` for a
page the user opens. The token itself is never returned, logged or stored.
"""
from __future__ import annotations

import logging
import os
import shutil
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

log = logging.getLogger("radiance.first_start")

OK, WARN, BAD = "ok", "warn", "bad"
LOW_SPACE = 20 * 1024 ** 3          # under 20 GB free: a few minutes of 4K EXR
HUB_TIMEOUT = 8.0
HUB_CACHE_SECONDS = 600


def _check(cid: str, title: str, status: str, detail: str, fix: Optional[dict] = None) -> Dict[str, Any]:
    return {"id": cid, "title": title, "status": status, "detail": detail, "fix": fix}


# ── OCIO ─────────────────────────────────────────────────────────────────────

def _aces_views(cfg) -> Dict[str, bool]:
    names = set()
    for d in cfg.getDisplays():
        names.update(cfg.getViews(d))
    return {"sdr": any(v.startswith("ACES 2.0 - SDR") for v in names),
            "hdr": any(v.startswith("ACES 2.0 - HDR") for v in names)}


def check_ocio(environ: Optional[dict] = None) -> Dict[str, Any]:
    title = "Colour management (OCIO + ACES 2.0)"
    try:
        from radiance.color import ocio_setup
    except Exception as exc:  # noqa: BLE001
        return _check("ocio", title, BAD, f"Radiance's OCIO setup did not load: {exc}")
    state = ocio_setup.STATE
    if not state.get("configured"):
        ocio_setup.configure_ocio(environ)
    if not state.get("configured"):
        err = state.get("error") or "no OCIO config available"
        return _check("ocio", title, BAD, f"{err}. Run: pip install \"opencolorio>=2.5.1\", then restart ComfyUI.")
    path = ocio_setup.active_config_path(environ)
    try:
        cfg = ocio_setup.active_config(path)
    except Exception as exc:  # noqa: BLE001
        cfg = None
        log.debug("[Radiance first start] active config did not load: %s", exc)
    version = state.get("ocio_version") or ""
    where = state.get("source") or "OCIO config"
    fix_ocio = {"action": "ocio", "label": "Use the ACES studio config"}
    if cfg is None:
        return _check("ocio", title, BAD, f"The OCIO config at {path} does not load.", fix_ocio)
    views = _aces_views(cfg)
    old = _version_lt(version, (2, 5, 1))
    if not views["sdr"] or not views["hdr"]:
        missing = " and ".join(k.upper() for k in ("sdr", "hdr") if not views[k])
        return _check("ocio", title, WARN,
                      f"{where} has no ACES 2.0 {missing} view. Radiance's HDR nodes still use the built-in "
                      f"ACES studio config. Unset OCIO, or point it at an ACES 2.0 config, to see the same views everywhere.",
                      fix_ocio if state.get("source") == "$OCIO (yours)" else None)
    if old:
        return _check("ocio", title, WARN,
                      f"OpenColorIO {version} is older than 2.5.1. Run: pip install -U \"opencolorio>=2.5.1\".")
    return _check("ocio", title, OK, f"OpenColorIO {version} · {cfg.getName() or 'config'} · {where}, "
                                     f"with ACES 2.0 SDR and HDR views")


def _version_lt(text: str, floor) -> bool:
    try:
        parts = tuple(int(p) for p in text.split(".")[:3])
    except ValueError:
        return False
    return parts < tuple(floor)


# ── ffmpeg ───────────────────────────────────────────────────────────────────

ENCODERS = (("libx265", "H.265 10-bit"), ("prores_ks", "ProRes"))


def check_ffmpeg() -> Dict[str, Any]:
    title = "Video encoding (ffmpeg)"
    from radiance.core import ffmpeg as ff
    found: Dict[str, str] = {}
    for enc, label in ENCODERS:
        try:
            found[label] = ff.ffmpeg_with_encoder(enc)
        except RuntimeError:
            pass
    if not found:
        try:
            ff.ffmpeg_exe()
        except RuntimeError:
            return _check("ffmpeg", title, BAD,
                          "ffmpeg was not found, so Read and Write can't open or make video files. "
                          "Run: pip install imageio-ffmpeg, or install ffmpeg and put it on PATH, then restart.")
    missing = [label for _, label in ENCODERS if label not in found]
    if missing:
        return _check("ffmpeg", title, WARN,
                      f"The ffmpeg here has no {' or '.join(missing)} encoder. Image sequences (EXR, DPX, TIFF) "
                      f"still work. Install a 'full' or 'essentials' ffmpeg build, or set RADIANCE_FFMPEG to one.")
    return _check("ffmpeg", title, OK, "H.265 10-bit and ProRes available · " + os.path.basename(found["H.265 10-bit"]))


# ── output folder ────────────────────────────────────────────────────────────

def default_output() -> Path:
    return Path.home() / "radiance_output"


def check_output(path: Optional[Path] = None) -> Dict[str, Any]:
    title = "Output folder"
    p = Path(path) if path else default_output()
    if not p.is_dir():
        parent = p.parent
        if not os.access(parent, os.W_OK):
            return _check("output", title, BAD,
                          f"Write's default folder {p} can't be made ({parent} is read-only). "
                          f"Set output_path on the Write node to a folder you can write to.")
        return _check("output", title, WARN, f"Write's default folder {p} doesn't exist yet. Write makes it on the "
                                             f"first render, or make it now.",
                      {"action": "output", "label": "Make the folder"})
    probe = p / ".radiance_write_test"
    try:
        probe.write_bytes(b"")
        probe.unlink()
    except OSError as exc:
        return _check("output", title, BAD, f"Write's default folder {p} doesn't take files ({exc.strerror or exc}). "
                                            f"Set output_path on the Write node to another folder.")
    try:
        free = shutil.disk_usage(p).free
    except OSError:
        free = None
    if free is not None and free < LOW_SPACE:
        return _check("output", title, WARN, f"{p} · only {free / 1024 ** 3:.1f} GB free; EXR sequences fill "
                                             f"a disk quickly.")
    space = f" · {free / 1024 ** 3:.0f} GB free" if free is not None else ""
    return _check("output", title, OK, f"{p}{space}. New Write nodes save here; the shipped workflows "
                                        f"save under ComfyUI's output folder (radiance_examples, radiance_api).")


# ── Hugging Face ─────────────────────────────────────────────────────────────

_hub_cache: Dict[str, Any] = {"at": 0.0, "result": None}
#: How the Hub is asked; None means setup_check._hub_head. The tests swap it
#: so no test reaches the network.
HEAD: Optional[Callable] = None
_hub_lock = threading.Lock()


def _gated_repos(cat: Dict[str, Any]) -> Dict[str, str]:
    """{repo: one file URL} for each gated repository a workflow needs."""
    used = {i for w in cat["workflows"].values() for i in w["required"] + w["optional"]}
    out: Dict[str, str] = {}
    for mid in sorted(used):
        m = cat["models"].get(mid) or {}
        if m.get("gated") and m.get("repo") and m.get("url"):
            out.setdefault(m["repo"], m["url"])
    return out


def _probe(url: str, token: str, head: Callable) -> str:
    """'ok', 'denied' or 'unreachable' for one file URL."""
    try:
        head(url, token=token, timeout=HUB_TIMEOUT)
        return "ok"
    except urllib.error.HTTPError as e:
        return "denied" if e.code in (401, 403) else "ok" if e.code < 400 else "unreachable"
    except Exception:  # noqa: BLE001 - DNS, timeout, proxy
        return "unreachable"


def check_hub(refresh: bool = False, *, head: Optional[Callable] = None,
              token_fn: Optional[Callable[[], Optional[str]]] = None,
              allowed_fn: Optional[Callable[[], bool]] = None) -> Dict[str, Any]:
    with _hub_lock:
        cached = _hub_cache["result"]
        if cached and not refresh and head is None and time.time() - _hub_cache["at"] < HUB_CACHE_SECONDS:
            return cached
    result = _check_hub(head, token_fn, allowed_fn)
    if head is None:
        with _hub_lock:
            _hub_cache.update(at=time.time(), result=result)
    return result


def _check_hub(head, token_fn, allowed_fn) -> Dict[str, Any]:
    title = "Hugging Face access"
    from radiance.core import setup_check
    from radiance.core.consent import downloads_allowed
    from radiance.core.model_fetch import hf_token
    head = head or HEAD or setup_check._hub_head
    allowed = (allowed_fn or downloads_allowed)()
    if not allowed:
        return _check("hub", title, WARN, "Downloads are off (RADIANCE_ALLOW_DOWNLOADS=0 or offline). Copy model "
                                          "files into ComfyUI/models yourself.")
    token = (token_fn or hf_token)() or ""
    cat = setup_check.load_catalog()
    gated = _gated_repos(cat)
    public = next((m["url"] for m in cat["models"].values() if not m.get("gated") and m.get("url", "").startswith(
        "https://huggingface.co/")), "")
    if public and _probe(public, "", head) == "unreachable":
        return _check("hub", title, BAD, "huggingface.co can't be reached from this computer (offline, firewall or "
                                         "proxy). Set HTTPS_PROXY if your network needs one.")
    if not gated:
        return _check("hub", title, OK, "huggingface.co reachable")
    if not token:
        return _check("hub", title, WARN,
                      f"huggingface.co reachable · no token set. {len(gated)} model repositories need their licence "
                      f"accepted and a token: run `hf auth login` in a terminal (or set HF_TOKEN), then restart.",
                      {"url": "https://huggingface.co/settings/tokens", "label": "Get a token"})
    with ThreadPoolExecutor(max_workers=min(8, len(gated))) as pool:
        answers = dict(zip(gated, pool.map(lambda u: _probe(u, token, head), gated.values())))
    denied = sorted(r for r, a in answers.items() if a == "denied")
    if denied:
        more = f" and {len(denied) - 3} more" if len(denied) > 3 else ""
        return _check("hub", title, WARN,
                      f"Token found · accept the licence for {len(denied)} of {len(gated)} gated repositories: "
                      f"{', '.join(denied[:3])}{more}.",
                      {"url": f"https://huggingface.co/{denied[0]}", "label": "Open licence page"})
    return _check("hub", title, OK, f"Token found · all {len(gated)} gated licences accepted")


# ── all, and fixes ───────────────────────────────────────────────────────────

def checks(refresh: bool = False) -> List[Dict[str, Any]]:
    out = []
    for cid, title, run in (("ocio", "Colour management (OCIO + ACES 2.0)", lambda: check_ocio()),
                            ("ffmpeg", "Video encoding (ffmpeg)", lambda: check_ffmpeg()),
                            ("output", "Output folder", lambda: check_output()),
                            ("hub", "Hugging Face access", lambda: check_hub(refresh))):
        try:
            out.append(run())
        except Exception as exc:  # noqa: BLE001 - one failing check must not hide the others
            log.exception("[Radiance first start] the %s check failed", cid)
            out.append(_check(cid, title, WARN, f"The check failed: {exc}"))
    return out


def fix(action: str) -> Dict[str, Any]:
    """Make one server-side fix; returns the check it fixes, run again.

    "ocio" only changes this ComfyUI session: the OCIO environment variable is
    the user's, so it is not edited on disk."""
    if action == "output":
        default_output().mkdir(parents=True, exist_ok=True)
        return check_output()
    if action == "ocio":
        from radiance.color import ocio_setup
        os.environ.pop("OCIO", None)
        ocio_setup.STATE["configured"] = False
        ocio_setup.configure_ocio()
        return check_ocio()
    raise KeyError(action)


__all__ = ["checks", "fix", "check_ocio", "check_ffmpeg", "check_output", "check_hub", "default_output"]
