"""Gizmos read every node return form ComfyUI accepts (package audit R4).

The subgraph executor wrapped any non-tuple result as ``(result,)``, so a node
returning ``{"ui": ..., "result": (a, b)}`` (Radiance's Prompt and Viewer) or a
V3 ``NodeOutput`` put the whole dict in output slot 0 and left slot 1 empty.
"""

import sys
import types

import pytest

from radiance.nodes.gizmo import _node_outputs, run_subgraph_executor


class _NodeOutput:          # the shape of comfy_api.latest.io.NodeOutput
    def __init__(self, *args, ui=None, expand=None, block_execution=None):
        self.args, self.ui, self.expand, self.block_execution = args, ui, expand, block_execution


def _node(fn):
    return type("N", (), {"FUNCTION": "run", "run": lambda self, **kw: fn(**kw)})


@pytest.fixture
def registry(monkeypatch):
    mod = types.ModuleType("nodes")
    mod.NODE_CLASS_MAPPINGS = {}
    monkeypatch.setitem(sys.modules, "nodes", mod)
    return mod.NODE_CLASS_MAPPINGS


def _gizmo(outputs):
    return {
        "name": "t",
        "inputs": [{"name": "text", "target_node_id": 1, "target_input": "text"}],
        "nodes": [{"id": 1, "type": "Prompt", "widgets": {}}, {"id": 2, "type": "Use", "widgets": {}}],
        "links": [{"origin_node_id": 1, "origin_output_slot": 1, "target_node_id": 2, "target_input": "x"}],
        "outputs": outputs,
    }


@pytest.mark.parametrize("form", ["dict", "node_output", "tuple"])
def test_both_outputs_of_a_multi_output_node_reach_the_gizmo(registry, form):
    def prompt(text):
        pair = (f"pos:{text}", f"neg:{text}")
        if form == "dict":
            return {"ui": {"text": ["preview"]}, "result": pair}
        if form == "node_output":
            return _NodeOutput(*pair, ui={"text": ["preview"]})
        return pair
    registry["Prompt"] = _node(prompt)
    registry["Use"] = _node(lambda x: (f"used {x}",))
    out = run_subgraph_executor(_gizmo([
        {"target_node_id": 1, "target_output_slot": 0},
        {"target_node_id": 1, "target_output_slot": 1},
        {"target_node_id": 2, "target_output_slot": 0},
    ]), {"text": "fox"})
    assert out == ("pos:fox", "neg:fox", "used neg:fox")


def test_a_single_value_is_still_one_output():
    assert _node_outputs(5) == (5,)
    assert _node_outputs(None) == ()
    assert _node_outputs({"ui": {"images": []}}) == ()


@pytest.mark.parametrize("res, match", [
    ({"result": (1,), "expand": {"graph": {}}}, "expands into a graph"),
    (_NodeOutput(1, expand={"graph": {}}), "expands into a graph"),
    (_NodeOutput(block_execution="no model"), "blocked execution"),
])
def test_forms_a_gizmo_cannot_honour_stop_with_a_message(res, match):
    with pytest.raises(RuntimeError, match=match):
        _node_outputs(res, "X")


def test_an_async_node_is_refused_not_passed_on_as_a_coroutine():
    async def run():
        return (1,)
    with pytest.raises(RuntimeError, match="async node"):
        _node_outputs(run(), "X")
