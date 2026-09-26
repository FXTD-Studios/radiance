"""
Radiance Dynamic Gizmos Engine.
Enables packaging of node subgraphs into reusable, styled custom ComfyUI nodes (Gizmos).
Created by FXTD Studios.
"""

from __future__ import annotations
import os
import json
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
                res = execution_method(**resolved_inputs)
                # Ensure output is a tuple (ComfyUI standard)
                if not isinstance(res, tuple):
                    res = (res,)
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

@_route("post", "/radiance/gizmos/create")
async def create_gizmo_api(request) -> web.Response:
    """API Endpoint: Receives a dynamic Gizmo configuration, validates and saves it."""
    try:
        data = await request.json()
        name = data.get("name", "").strip()
        display_name = data.get("display_name", "").strip() or name
        category = (data.get("category", "").strip() or "FXTD STUDIOS/Radiance/◎ Gizmos").replace("\\", "/")
        description = data.get("description", "").strip() or "Custom Dynamic Gizmo."
        
        inputs = data.get("inputs", [])
        outputs = data.get("outputs", [])
        widgets = data.get("widgets", [])
        nodes_list = data.get("nodes", [])
        links = data.get("links", [])

        if not name or not nodes_list:
            return web.json_response({"success": False, "error": "Missing Gizmo name or internal node definitions"}, status=400)

        # Standardize naming to alphanumeric/underscores
        safe_name = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
        
        # Package JSON payload
        gizmo_payload = {
            "name": safe_name,
            "display_name": display_name,
            "category": category,
            "description": description,
            "inputs": inputs,
            "outputs": outputs,
            "widgets": widgets,
            "nodes": nodes_list,
            "links": links
        }

        # Write to disk
        filepath = os.path.join(GIZMOS_DIR, f"{safe_name}.gizmo")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(gizmo_payload, f, indent=2)

        logger.info(f"[Radiance Gizmo] Successfully wrote new dynamic Gizmo payload: {filepath}")
        return web.json_response({
            "success": True, 
            "message": f"Gizmo '{display_name}' created successfully. Please restart your ComfyUI server to load it.",
            "name": safe_name
        })

    except Exception as e:
        logger.exception("[Radiance Gizmo] Create endpoint failed")
        return web.json_response({"success": False, "error": str(e)}, status=500)


@_route("get", "/radiance/gizmos/list")
async def list_gizmos_api(request) -> web.Response:
    """API Endpoint: Lists all saved Gizmos with their visual and pipeline metadata."""
    try:
        gizmos_list = []
        if os.path.exists(GIZMOS_DIR):
            for filename in os.listdir(GIZMOS_DIR):
                if filename.endswith(".gizmo"):
                    filepath = os.path.join(GIZMOS_DIR, filename)
                    try:
                        with open(filepath, "r", encoding="utf-8") as f:
                            data = json.load(f)
                        gizmos_list.append({
                            "name": data.get("name"),
                            "display_name": data.get("display_name"),
                            "category": data.get("category"),
                            "description": data.get("description"),
                            "input_count": len(data.get("inputs", [])),
                            "output_count": len(data.get("outputs", [])),
                            "widget_count": len(data.get("widgets", []))
                        })
                    except Exception as err:
                        logger.warning(f"[Radiance Gizmo] Failed to read gizmo file metadata {filename}: {err}")

        return web.json_response({"success": True, "gizmos": gizmos_list})
    except Exception as e:
        return web.json_response({"success": False, "error": str(e)}, status=500)
