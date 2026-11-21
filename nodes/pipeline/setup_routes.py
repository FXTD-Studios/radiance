"""HTTP routes for the Radiance setup check (core/setup_check.py).

    POST /radiance/setup/scan        this machine and every workflow's models;
                                     creates ComfyUI/models/radiance if missing
    POST /radiance/setup/download    {"ids": [...]} or {"workflow": stem,
                                     "optional": false}: queue the missing ones
    POST /radiance/setup/cancel      {"id": model id}
    GET  /radiance/setup/downloads   the queue, with progress
    POST /radiance/setup/fix         {"action": "output"|"ocio"}: one first-start
                                     fix (core/first_start.py); returns the check

Only ids from config/model_catalog.json are accepted: the page cannot make
the server download an arbitrary URL. No node classes live here.
"""
from __future__ import annotations

import asyncio
import logging

try:
    from aiohttp import web
    from server import PromptServer
    _SERVER = True
except ImportError:
    web = None                # type: ignore[assignment]
    PromptServer = None       # type: ignore[assignment]
    _SERVER = False

logger = logging.getLogger("radiance.setup_routes")

NODE_CLASS_MAPPINGS: dict = {}
NODE_DISPLAY_NAME_MAPPINGS: dict = {}


def _route(method: str, path: str):
    if not _SERVER or PromptServer is None or getattr(PromptServer, "instance", None) is None:
        return lambda f: f
    reg = getattr(PromptServer.instance, "_radiance_registered_routes", None)
    if reg is None:
        reg = set()
        setattr(PromptServer.instance, "_radiance_registered_routes", reg)
    key = (method.lower(), path)
    if key in reg:
        return lambda f: f
    reg.add(key)
    return getattr(PromptServer.instance.routes, method)(path)


async def _body(request) -> dict:
    try:
        data = await request.json()
    except Exception:  # noqa: BLE001
        return {}
    return data if isinstance(data, dict) else {}


@_route("post", "/radiance/setup/scan")
async def setup_scan(request):
    from radiance.core import setup_check
    refresh = bool((await _body(request)).get("refresh"))
    try:
        result = await asyncio.get_running_loop().run_in_executor(
            None, lambda: setup_check.scan(create_radiance_dir=True, refresh=refresh))
        result["downloads"] = setup_check.QUEUE.jobs()
        return web.json_response(result)
    except Exception as exc:  # noqa: BLE001
        logger.exception("[Radiance setup] scan failed")
        return web.json_response({"error": str(exc)}, status=500)


@_route("post", "/radiance/setup/download")
async def setup_download(request):
    from radiance.core import setup_check
    data = await _body(request)
    try:
        if data.get("workflow"):
            ids = setup_check.missing_for(str(data["workflow"]), bool(data.get("optional")))
        else:
            ids = [str(i) for i in (data.get("ids") or [])]
        added = setup_check.QUEUE.enqueue(ids)
        return web.json_response({"queued": added, "downloads": setup_check.QUEUE.jobs()})
    except PermissionError as exc:
        return web.json_response({"error": str(exc)}, status=403)
    except KeyError as exc:
        return web.json_response({"error": f"not a Radiance model or workflow: {exc}"}, status=400)


@_route("post", "/radiance/setup/cancel")
async def setup_cancel(request):
    from radiance.core import setup_check
    data = await _body(request)
    ok = setup_check.QUEUE.cancel(str(data.get("id", "")))
    return web.json_response({"cancelled": ok, "downloads": setup_check.QUEUE.jobs()})


@_route("get", "/radiance/setup/downloads")
async def setup_downloads(request):
    from radiance.core import setup_check
    return web.json_response({"downloads": setup_check.QUEUE.jobs()})


@_route("post", "/radiance/setup/fix")
async def setup_fix(request):
    from radiance.core import first_start
    data = await _body(request)
    try:
        check = await asyncio.get_running_loop().run_in_executor(
            None, lambda: first_start.fix(str(data.get("action", ""))))
        return web.json_response({"check": check})
    except KeyError:
        return web.json_response({"error": "not a Radiance setup fix"}, status=400)
    except OSError as exc:
        return web.json_response({"error": str(exc)}, status=500)
