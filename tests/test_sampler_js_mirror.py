"""
js/radiance_sampler.js keeps its own copies of sampler_utils.py's model sets
and MODEL_DEFAULTS, and of config/model_map.py's Loader preset model types, so
the node shows the right widgets and values before any run. The copies
stopped at the families that existed before 3.5: none of the nine new ones had
live defaults, LongCat's flux_guidance stayed hidden, and MiniMax H3,
HunyuanVideo 1.5 and Kandinsky 5 video were offered Phase-Shift modes they
ignore.

Reads the files as text, so it needs neither torch nor a browser.
"""
import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JS = (ROOT / "js" / "radiance_sampler.js").read_text(encoding="utf-8")


def _py_literal(rel_path, name):
    tree = ast.parse((ROOT / rel_path).read_text(encoding="utf-8"))
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, ast.AnnAssign) else []
        if any(isinstance(t, ast.Name) and t.id == name for t in targets):
            return ast.literal_eval(node.value)
    raise KeyError(name)


def _js_set(name):
    body = re.search(r"const %s = new Set\(\[(.*?)\]\);" % name, JS, re.S).group(1)
    return set(re.findall(r'"([^"]+)"', body))


def _js_object(name):
    return re.search(r"const %s = \{(.*?)\n\};" % name, JS, re.S).group(1)


def test_model_sets_match_sampler_utils():
    for name in ("VIDEO_MODEL_TYPES", "GUIDANCE_EMBED_MODELS", "CFG_GUIDED_MODELS"):
        assert _js_set(name) == set(_py_literal("sampler_utils.py", name)), name


def test_live_defaults_match_model_defaults():
    js = {}
    for key, inner in re.findall(r'^\s*"?([a-z0-9_.-]+)"?:\s*\{([^}]*)\}', _js_object("MODEL_TYPE_SAMPLING_DEFAULTS"), re.M):
        js[key] = {f: v.strip('"') if v.startswith('"') else float(v)
                   for f, v in re.findall(r'(\w+):\s*("[^"]*"|[\d.]+)', inner)}
    for model_type, defaults in _py_literal("sampler_utils.py", "MODEL_DEFAULTS").items():
        assert model_type in js, model_type
        for field in ("cfg", "sampler", "scheduler", "guidance", "steps"):
            if field in defaults:
                assert js[model_type][field] == defaults[field], (model_type, field)


def test_loader_preset_model_types_match_checkpoint_presets():
    js = dict(re.findall(r'"([^"]+)":\s*"([^"]+)"', _js_object("LOADER_PRESET_MODEL_TYPE")))
    for preset, config in _py_literal("config/model_map.py", "CHECKPOINT_PRESETS").items():
        model_type = config.get("model_type")
        if model_type and model_type != "Auto-Detect":
            assert js.get(preset) == model_type, preset
