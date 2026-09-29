"""
js/radiance_resolution.js mirrors two of resolution.py's tables so the
widgets react as soon as model_type changes, before any run. The copies
drifted when 3.5 added families: HunyuanImage 2.1 and HunyuanVideo 1.5
snapped to 8px, HunyuanVideo 1.5 and Kandinsky 5 Video did not switch to
video.

Reads both files as text, so it needs neither torch nor a browser.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "js" / "radiance_resolution.js").read_text(encoding="utf-8")


def _py_literal(name):
    tree = ast.parse((ROOT / "nodes" / "generate" / "resolution.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def test_widget_alignment_matches_generate_for_every_model_type():
    body = re.search(r"const SPATIAL_SCALE_JS = \{(.*?)\n\};", JS, re.S).group(1)
    js = {k: int(v) for k, v in re.findall(r'"([^"]+)":\s*(\d+)', body)}
    scale, align = _py_literal("SPATIAL_SCALE"), _py_literal("SPATIAL_ALIGN")
    for model_type in _py_literal("MODEL_TYPES"):
        assert js.get(model_type, 8) == (align.get(model_type) or scale.get(model_type, 8)), model_type


def test_video_model_types_match():
    body = re.search(r"const VIDEO_MODEL_TYPES_JS = new Set\(\[(.*?)\]\);", JS, re.S).group(1)
    assert set(re.findall(r'"([^"]+)"', body)) == _py_literal("VIDEO_MODEL_TYPES")


def test_model_type_by_loader_arch_matches():
    body = re.search(r"const ARCHS_BY_MODEL_TYPE_JS = \{(.*?)\n\};", JS, re.S).group(1)
    js = {k: re.findall(r'"([^"]+)"', v) for k, v in re.findall(r'"([^"]+)":\s*\[([^\]]*)\]', body)}
    assert js == _py_literal("ARCHS_BY_MODEL_TYPE")


def test_every_loader_architecture_has_a_model_type():
    """model_meta's arch is one of the Loader's model_types; a new family must
    get a Resolution model_type too, or model_meta would not set one."""
    tree = ast.parse((ROOT / "nodes" / "generate" / "loader.py").read_text(encoding="utf-8"))
    loader_types = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                        and any(isinstance(t, ast.Name) and t.id == "MODEL_TYPES" for t in n.targets))
    mapping = _py_literal("ARCHS_BY_MODEL_TYPE")
    archs = [arch for group in mapping.values() for arch in group]
    assert set(loader_types) - {"Auto-Detect"} <= set(archs)
    assert len(archs) == len(set(archs))
    assert set(mapping) <= set(_py_literal("MODEL_TYPES"))
