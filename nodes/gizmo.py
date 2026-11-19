"""
Radiance Dynamic Gizmos Engine.
Enables packaging of node subgraphs into reusable, styled custom ComfyUI nodes (Gizmos).
Created by FXTD Studios.
"""

from __future__ import annotations
import os
import shutil
import json
import time
import logging
import traceback
import torch
from typing import Tuple, Dict, Any, List, Optional

logger = logging.getLogger("radiance.gizmo")

# ═══════════════════════════════════════════════════════════════════════════════
#                           SERVER ROUTE BOOTSTRAPPING
# ═══════════════════════════════════════════════════════════════════════════════

try:
    from aiohttp import web
    from server import PromptServer
    _SERVER_AVAILABLE = True
except ImportError:
    web = None                # type: ignore[assignment]
    PromptServer = None       # type: ignore[assignment]
    _SERVER_AVAILABLE = False

def _route(method: str, path: str):
    """Decorator: register an aiohttp route only when PromptServer is present."""
    if not _SERVER_AVAILABLE or PromptServer is None:
        return lambda f: f
    # Idempotent across duplicate module imports — see nodes_workspace._route.
    _reg = getattr(PromptServer.instance, "_radiance_registered_routes", None)
    if _reg is None:
        _reg = set()
        setattr(PromptServer.instance, "_radiance_registered_routes", _reg)
    _key = (method.lower(), path)
    if _key in _reg:
        return lambda f: f
    _reg.add(_key)
    return getattr(PromptServer.instance.routes, method)(path)

# Ensure local gizmos folder exists
# ALBABIT-FIX: the package's gizmos/ folder, not one beside this file. Moving the
# module into nodes/ had sent saved gizmos to a stray nodes/gizmos/.
GIZMOS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "gizmos")
os.makedirs(GIZMOS_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════════
#                      TOPOLOGICAL SUBGRAPH EXECUTOR
# ═══════════════════════════════════════════════════════════════════════════════

def run_subgraph_executor(gizmo_data: Dict[str, Any], outer_inputs: Dict[str, Any]) -> Tuple[Any, ...]:
    """
    Executes a saved node subgraph sequentially in Python using a topological sort.
    Routes outer inputs and promoted widgets to their target internal node ports.
    """
    # 1. Map target inputs and promoted widgets
    # (node_id, input_name) -> value
    overrides = {}

    # Map outer boundary inputs
    for inp in gizmo_data.get("inputs", []):
        name = inp["name"]
        if name in outer_inputs:
            node_id = inp["target_node_id"]
            target_input = inp["target_input"]
            overrides[(node_id, target_input)] = outer_inputs[name]

    # Map promoted widgets / knobs
    for widget in gizmo_data.get("widgets", []):
        name = widget["name"]
        if name in outer_inputs:
            node_id = widget["target_node_id"]
            target_widget = widget["target_widget"]
            overrides[(node_id, target_widget)] = outer_inputs[name]

    # 2. Build dependency graph for topological sorting
    nodes_list = gizmo_data.get("nodes", [])
    nodes_map = {n["id"]: n for n in nodes_list}

    adj: Dict[int, List[int]] = {n["id"]: [] for n in nodes_list}
    in_degree = {n["id"]: 0 for n in nodes_list}

    # Map incoming links: (target_node_id, target_input) -> (origin_node_id, origin_output_slot)
    incoming_links = {}

    for link in gizmo_data.get("links", []):
        origin = link["origin_node_id"]
        target = link["target_node_id"]

        if origin in nodes_map and target in nodes_map:
            adj[origin].append(target)
            in_degree[target] += 1
            target_input = link["target_input"]
            incoming_links[(target, target_input)] = (origin, link["origin_output_slot"])

    # 3. Topological Sort (Kahn's Algorithm)
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    execution_order = []

    while queue:
        u = queue.pop(0)
        execution_order.append(u)
        for v in adj[u]:
            in_degree[v] -= 1
            if in_degree[v] == 0:
                queue.append(v)

    # Cycle fallback
    if len(execution_order) < len(nodes_list):
        logger.warning(
            f"[Radiance Gizmo] Dynamic Gizmo '{gizmo_data['name']}' has a cycle in its internal graph. "
            "Attempting fallback execution order."
        )
        visited = set(execution_order)
        for n in nodes_list:
            nid = n["id"]
            if nid not in visited:
                execution_order.append(nid)

    # 4. State storage for execution outputs
    # node_outputs[node_id] = tuple of output values
    node_outputs: Dict[int, Tuple[Any, ...]] = {}

    try:
        # Import ComfyUI node mappings registry
        import nodes

        # 5. Sequentially execute internal nodes
        for node_id in execution_order:
            node_def = nodes_map[node_id]
            node_type = node_def["type"]

            node_class = nodes.NODE_CLASS_MAPPINGS.get(node_type)
            if not node_class:
                raise RuntimeError(
                    f"[Radiance Gizmo] Missing dependency! Internal node type '{node_type}' is not registered. "
                    "Is the required custom node suite or custom node package installed?"
                )

            # Instantiate node class
            node_instance = node_class()

            # Resolve input arguments
            resolved_inputs = {}

            # a) Populate saved static widgets values
            for w_name, w_val in node_def.get("widgets", {}).items():
                resolved_inputs[w_name] = w_val

            # b) Overwrite with outputs from preceding nodes
            for (tgt_id, tgt_input), (orig_id, orig_slot) in incoming_links.items():
                if tgt_id == node_id:
                    if orig_id in node_outputs:
                        outputs_tuple = node_outputs[orig_id]
                        if isinstance(outputs_tuple, tuple) and orig_slot < len(outputs_tuple):
                            resolved_inputs[tgt_input] = outputs_tuple[orig_slot]
                        elif orig_slot == 0 and not isinstance(outputs_tuple, tuple):
                            resolved_inputs[tgt_input] = outputs_tuple
                    else:
                        resolved_inputs[tgt_input] = None

            # c) Overwrite with outer inputs (exposed boundary inputs and promoted knobs)
            for (tgt_id, tgt_input), val in overrides.items():
                if tgt_id == node_id:
                    resolved_inputs[tgt_input] = val

            # Locate class execution function
            func_name = getattr(node_class, "FUNCTION", "run")
            execution_method = getattr(node_instance, func_name, None)
            if not execution_method:
                raise RuntimeError(f"Internal node class '{node_type}' has no callable execution method '{func_name}'")

            # Execute
            try:
                res = _node_outputs(execution_method(**resolved_inputs), node_type)
                node_outputs[node_id] = res
            except Exception as exc:
                logger.error(f"[Radiance Gizmo] Execution failed on internal node '{node_type}' (ID: {node_id}) inside Gizmo: {exc}")
                logger.error(traceback.format_exc())
                raise RuntimeError(f"Internal error inside dynamic Gizmo subgraph: {exc}") from exc

        # 6. Retrieve boundary outputs
        outer_outputs = []
        for out in gizmo_data.get("outputs", []):
            node_id = out["target_node_id"]
            slot = out["target_output_slot"]

            val = None
            if node_id in node_outputs:
                outputs_tuple = node_outputs[node_id]
                if isinstance(outputs_tuple, tuple) and slot < len(outputs_tuple):
                    val = outputs_tuple[slot]
                elif slot == 0 and not isinstance(outputs_tuple, tuple):
                    val = outputs_tuple

            outer_outputs.append(val)

        return tuple(outer_outputs)
    finally:
        # Clear large intermediate cache references to ensure PyTorch releases VRAM instantly
        node_outputs.clear()


# ═══════════════════════════════════════════════════════════════════════════════
#                        DYNAMIC CLASS REGISTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def _node_outputs(res: Any, node_type: str = "") -> tuple:
    """A node's outputs as the tuple ComfyUI's executor reads, from any return form.

    Package audit R4: every non-tuple result used to be wrapped as
    ``(result,)``, so a node returning ``{"ui": ..., "result": (a, b)}`` (the
    Prompt, the Viewer) or a V3 ``NodeOutput`` put the whole dict or object in
    output slot 0 and left slot 1 empty. The UI part has no viewer inside a
    Gizmo and is dropped; forms the subgraph executor cannot honour (graph
    expansion, an execution blocker, a coroutine from an async node) stop with
    a message rather than passing the wrong type on.
    """
    import inspect as _inspect
    where = f" (internal node '{node_type}')" if node_type else ""
    if _inspect.isawaitable(res):
        if hasattr(res, "close"):
            res.close()
        raise RuntimeError(f"An async node cannot run inside a Gizmo{where}.")
    if isinstance(res, tuple):
        return res
    if res is None:
        return ()
    if isinstance(res, dict) and ("result" in res or "ui" in res or "expand" in res):
        if res.get("expand") is not None:
            raise RuntimeError(f"A node that expands into a graph cannot run inside a Gizmo{where}.")
        out = res.get("result", ())
        return tuple(out) if isinstance(out, (list, tuple)) else (out,)
    # V3 NodeOutput (comfy_api.latest.io), recognised by shape so older
    # ComfyUI builds without it still import this module.
    if hasattr(res, "args") and hasattr(res, "ui") and hasattr(res, "expand"):
        if getattr(res, "expand", None) is not None:
            raise RuntimeError(f"A node that expands into a graph cannot run inside a Gizmo{where}.")
        if getattr(res, "block_execution", None) is not None:
            raise RuntimeError(f"A node blocked execution inside a Gizmo{where}: {res.block_execution}")
        return tuple(res.args)
    return (res,)


def generate_gizmo_class(gizmo_data: Dict[str, Any]) -> Tuple[str, Any]:
    """
    Dynamically constructs a first-class Python class matching the ComfyUI
    custom node registry specification using the type() class factory.
    """
    name = gizmo_data["name"]
    category = gizmo_data.get("category", "FXTD STUDIOS/Radiance/◎ Gizmos").replace("\\", "/")
    description = gizmo_data.get("description", "A custom dynamic Radiance Gizmo.")

    # Standardize class naming convention to ensure style matching works cleanly
    if not name.startswith("RadianceGizmo_"):
        class_name = f"RadianceGizmo_{name}"
    else:
        class_name = name

    # Dynamic INPUT_TYPES construction
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        required = {}
        optional = {}

        # Add boundary input ports
        for inp in gizmo_data.get("inputs", []):
            required[inp["name"]] = (inp["type"], {"tooltip": f"Exposed input port: {inp['name']}"})

        # Add promoted widgets / controls
        for widget in gizmo_data.get("widgets", []):
            w_type = widget["type"]
            w_name = widget["name"]
            w_opts: Dict[str, Any] = {
                "default": widget.get("default"),
                "tooltip": f"Promoted control parameter: {w_name}"
            }
            if "min" in widget: w_opts["min"] = widget["min"]
            if "max" in widget: w_opts["max"] = widget["max"]
            if "step" in widget: w_opts["step"] = widget["step"]

            if "choices" in widget:
                required[w_name] = (widget["choices"], w_opts)
            elif w_type in ("FLOAT", "INT", "STRING", "BOOLEAN"):
                required[w_name] = (w_type, w_opts)
            else:
                required[w_name] = (w_type, w_opts)

        return {"required": required, "optional": optional}

    # Dynamic Return Types
    return_types = tuple(out["type"] for out in gizmo_data.get("outputs", []))
    return_names = tuple(out["name"] for out in gizmo_data.get("outputs", []))

    # Execution method mapping
    def execute_gizmo(self, **kwargs) -> Tuple[Any, ...]:
        return run_subgraph_executor(gizmo_data, kwargs)

    class_attrs = {
        "INPUT_TYPES": INPUT_TYPES,
        "RETURN_TYPES": return_types,
        "RETURN_NAMES": return_names,
        "FUNCTION": "execute_gizmo",
        "CATEGORY": category,
        "DESCRIPTION": description,
        "execute_gizmo": execute_gizmo,
        "IS_GIZMO": True  # styling flag
    }

    # Create dynamic class
    dynamic_class = type(class_name, (object,), class_attrs)
    return class_name, dynamic_class


#: Gizmos registered at this start-up: file stem -> node class name. A gizmo
#: saved since then is listed on the Gizmos page as needing a restart.
LOADED_GIZMOS: Dict[str, str] = {}


def load_dynamic_gizmos() -> Tuple[Dict[str, Any], Dict[str, str]]:
    """
    Scans the gizmos/ directory, compiles valid JSON .gizmo specs into dynamic
    classes, and returns mappings for ComfyUI registration.
    """
    class_mappings = {}
    display_mappings = {}

    if not os.path.exists(GIZMOS_DIR):
        return class_mappings, display_mappings

    for filename in os.listdir(GIZMOS_DIR):
        if filename.endswith(".gizmo"):
            filepath = os.path.join(GIZMOS_DIR, filename)
            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    gizmo_data = json.load(f)

                class_name, dynamic_class = generate_gizmo_class(gizmo_data)
                class_mappings[class_name] = dynamic_class
                LOADED_GIZMOS[filename[:-len(".gizmo")]] = class_name

                # Register display name prefixing with custom glyph
                display_name = gizmo_data.get("display_name", gizmo_data["name"])
                display_mappings[class_name] = f"◎ {display_name}"
                
                logger.info(f"[Radiance Gizmo] Dynamically loaded dynamic Gizmo: {display_name} -> {class_name}")
            except Exception as e:
                logger.error(f"[Radiance Gizmo] Failed to compile dynamic Gizmo file {filename}: {e}")

    return class_mappings, display_mappings


# ═══════════════════════════════════════════════════════════════════════════════
#                             API ROUTE HANDLERS
# ═══════════════════════════════════════════════════════════════════════════════

_MAX_GIZMO_BYTES = 5 * 1024 * 1024


def _save_gizmo(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate a gizmo definition and write it to gizmos/<name>.gizmo.

    Returns {"name", "display_name", "replaced"}; raises ValueError when it is
    not a gizmo. Re-saving one is how it is edited: the previous file is kept
    as .bak.
    """
    if not isinstance(data, dict):
        raise ValueError("A gizmo is a JSON object")
    name = str(data.get("name", "")).strip()
    display_name = str(data.get("display_name", "")).strip() or name
    category = (str(data.get("category", "")).strip() or "FXTD STUDIOS/Radiance/◎ Gizmos").replace("\\", "/")
    description = str(data.get("description", "")).strip() or "Custom Dynamic Gizmo."
    lists = {k: data.get(k, []) for k in ("inputs", "outputs", "widgets", "nodes", "links")}
    if not name or not lists["nodes"]:
        raise ValueError("Missing Gizmo name or internal node definitions")
    if not all(isinstance(v, list) for v in lists.values()):
        raise ValueError("inputs, outputs, widgets, nodes and links must be lists")
    safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    payload = {"name": safe_name, "display_name": display_name, "category": category,
               "description": description, **lists}
    filepath = os.path.join(GIZMOS_DIR, f"{safe_name}.gizmo")
    replaced = os.path.isfile(filepath)
    if replaced:
        try:
            shutil.copy2(filepath, filepath + ".bak")
        except OSError as exc:
            logger.warning("[Radiance Gizmo] could not back up %s: %s", filepath, exc)
    from radiance.core.sequence_safety import write_text_atomic
    write_text_atomic(filepath, json.dumps(payload, indent=2))
    logger.info("[Radiance Gizmo] Saved %s", filepath)
    return {"name": safe_name, "display_name": display_name, "replaced": replaced}


def _gizmo_path(name: str) -> Optional[str]:
    """gizmos/<name>.gizmo for a name the list returned; None for anything else."""
    safe = "".join(c if c.isalnum() or c == "_" else "_" for c in str(name or ""))
    if not safe or safe != name:
        return None
    path = os.path.join(GIZMOS_DIR, f"{safe}.gizmo")
    return path if os.path.isfile(path) else None


def gizmo_summaries() -> List[Dict[str, Any]]:
    """Every saved gizmo, for the Gizmos page of the Radiance window."""
    out: List[Dict[str, Any]] = []
    if not os.path.isdir(GIZMOS_DIR):
        return out
    for filename in sorted(os.listdir(GIZMOS_DIR)):
        if not filename.endswith(".gizmo"):
            continue
        path = os.path.join(GIZMOS_DIR, filename)
        stem = filename[:-len(".gizmo")]
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as err:  # noqa: BLE001
            out.append({"name": stem, "display_name": stem, "error": f"cannot be read: {err}",
                        "file": path, "loaded": False})
            continue
        st = os.stat(path)
        class_name = generate_class_name(data.get("name") or stem)
        out.append({
            "name": data.get("name") or stem,
            "display_name": data.get("display_name") or data.get("name") or stem,
            "category": data.get("category"),
            "description": data.get("description"),
            "inputs": [{"name": i.get("name"), "type": i.get("type")} for i in data.get("inputs", []) if isinstance(i, dict)],
            "outputs": [{"name": o.get("name"), "type": o.get("type")} for o in data.get("outputs", []) if isinstance(o, dict)],
            "knobs": [w.get("name") for w in data.get("widgets", []) if isinstance(w, dict)],
            "input_count": len(data.get("inputs", [])),
            "output_count": len(data.get("outputs", [])),
            "widget_count": len(data.get("widgets", [])),
            "node_count": len(data.get("nodes", [])),
            "node_types": sorted({str(n.get("type")) for n in data.get("nodes", []) if isinstance(n, dict)}),
            "class_name": class_name,
            "loaded": LOADED_GIZMOS.get(stem) == class_name,
            "file": path, "size": st.st_size, "mtime": st.st_mtime,
        })
    return out


def generate_class_name(name: str) -> str:
    return name if name.startswith("RadianceGizmo_") else f"RadianceGizmo_{name}"


@_route("post", "/radiance/gizmos/create")
async def create_gizmo_api(request) -> web.Response:
    """API Endpoint: Receives a dynamic Gizmo configuration, validates and saves it."""
    try:
        if request.content_length is not None and request.content_length > _MAX_GIZMO_BYTES:
            return web.json_response({"success": False, "error": "Gizmo definition is too large"}, status=413)
        data = await request.json()
        try:
            saved = _save_gizmo(data)
        except ValueError as exc:
            return web.json_response({"success": False, "error": str(exc)}, status=400)
        return web.json_response({
            "success": True,
            "message": f"Gizmo '{saved['display_name']}' {'replaced (previous kept as .bak)' if saved['replaced'] else 'created'} successfully. "
                       "Please restart your ComfyUI server to load it. It is listed in Radiance > Gizmos.",
            "name": saved["name"],
        })

    except Exception as e:
        logger.exception("[Radiance Gizmo] Create endpoint failed")
        return web.json_response({"success": False, "error": str(e)}, status=500)


@_route("get", "/radiance/gizmos/list")
async def list_gizmos_api(request) -> web.Response:
    """Every saved gizmo, with what it holds and whether this start-up loaded it."""
    try:
        return web.json_response({"success": True, "folder": GIZMOS_DIR, "gizmos": gizmo_summaries()})
    except Exception as e:  # noqa: BLE001
        return web.json_response({"success": False, "error": str(e)}, status=500)


@_route("get", "/radiance/gizmos/file")
async def gizmo_file_api(request) -> web.Response:
    """Download one .gizmo, to share it."""
    path = _gizmo_path(request.query.get("name", ""))
    if path is None:
        return web.json_response({"success": False, "error": "No such gizmo"}, status=404)
    return web.FileResponse(path, headers={
        "Content-Disposition": f'attachment; filename="{os.path.basename(path)}"'})


@_route("post", "/radiance/gizmos/import")
async def import_gizmo_api(request) -> web.Response:
    """Save a shared .gizmo (its JSON as the body) into gizmos/."""
    try:
        if request.content_length is not None and request.content_length > _MAX_GIZMO_BYTES:
            return web.json_response({"success": False, "error": "Gizmo definition is too large"}, status=413)
        saved = _save_gizmo(await request.json())
        return web.json_response({"success": True, **saved,
                                  "message": f"Gizmo '{saved['display_name']}' imported. Restart ComfyUI to use it."})
    except ValueError as e:
        return web.json_response({"success": False, "error": str(e)}, status=400)
    except Exception as e:  # noqa: BLE001
        return web.json_response({"success": False, "error": f"Not a gizmo file: {e}"}, status=400)


@_route("post", "/radiance/gizmos/delete")
async def delete_gizmo_api(request) -> web.Response:
    """Move a gizmo to gizmos/.trash (kept, not erased). Its node stays until a restart."""
    try:
        data = await request.json()
        path = _gizmo_path(str(data.get("name", "")))
        if path is None:
            return web.json_response({"success": False, "error": "No such gizmo"}, status=404)
        trash = os.path.join(GIZMOS_DIR, ".trash")
        os.makedirs(trash, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        dest = os.path.join(trash, f"{os.path.basename(path)[:-len('.gizmo')]}.{stamp}.gizmo")
        os.replace(path, dest)
        return web.json_response({"success": True, "trash": dest})
    except Exception as e:  # noqa: BLE001
        return web.json_response({"success": False, "error": str(e)}, status=500)
