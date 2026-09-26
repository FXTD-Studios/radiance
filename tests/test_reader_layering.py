"""The read engine sits below the node layer, and that is checkable.

The mirror of `test_writer_layering.py`, for the other half of what was a
2972-line `nodes/io/write.py`. The writer left first and took it to 2201; the
reader was the larger remainder, and the more entangled one:

* `nodes/pipeline/dcc.py` and `nodes/pipeline/studio_integrations.py` import
  `_read_sequence` and `_load_video_to_numpy` out of a module whose declared
  job is to register two ComfyUI nodes.
* The writer could not take `coerce_to_frames` all the way down, because
  resolving a path to frames means *reading* media, and the decoders were up
  here. That is why `write_frames` has a `read_media` callback at all.

`radiance/io/reader.py` is that code, moved down a floor. These tests are what
stops it drifting back up — including the quiet version, where the import moves
inside a function body and the module header still claims independence.

The one thing that stayed up is the browse widget: `_resolve_browse` turns a
filename in ComfyUI's input directory into a path, which needs `folder_paths`.
`read_frames` takes a path and nothing else, and the last test below is what
holds that line.
"""
import ast
import pathlib
import re
import subprocess
import sys

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent

#: Third-party packages a bare interpreter may legitimately lack. `radiance`
#: is deliberately not here: if the package itself will not import, that is the
#: failure these tests exist to report.
_OPTIONAL = {"torch", "numpy", "imageio", "OpenEXR", "Imath", "cv2", "PIL"}

# Importing the package as `radiance` in a bare subprocess, whatever the
# checkout is called. Spelled out again rather than shared with
# test_writer_layering.py on purpose: these subprocesses run with no conftest
# and no import of this repo's test helpers, which is the whole point of them,
# so a shared fixture would be a dependency the thing under test does not have.
_BOOT = (
    "import importlib.util, sys;"
    f"_s = importlib.util.spec_from_file_location('radiance', {str(_ROOT / '__init__.py')!r},"
    f" submodule_search_locations=[{str(_ROOT)!r}]);"
    "_p = importlib.util.module_from_spec(_s);"
    "sys.modules['radiance'] = _p;"
    "_s.loader.exec_module(_p);"
)


def _tree(rel):
    return ast.parse((_ROOT / rel).read_text(encoding="utf-8"))


def _imported_modules(tree):
    """Every module named by an import, at any depth — including inside a
    function body, which is where an upward dependency goes to hide."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            found.append("." * (node.level or 0) + (node.module or ""))
    return found


def test_the_engine_never_imports_the_node_layer():
    # Relative and absolute both: `..nodes` and `radiance.nodes` are the same
    # dependency written two ways, and ast.walk reaches function bodies, so a
    # deferred import inside `except ImportError` does not escape this.
    for mod in _imported_modules(_tree("io/reader.py")):
        assert "nodes" not in mod.split("."), (
            f"radiance/io/reader.py imports {mod!r} — the engine is reaching "
            "back up into the node layer it was extracted from"
        )


def test_the_engine_never_imports_comfyui():
    # `folder_paths` is the specific one. It is what the browse widget needs,
    # it is the reason `read_frames` takes a path instead of a filename, and it
    # is the import that would quietly re-couple the engine to a running
    # ComfyUI — the thing that made the reader hard to exercise in the first
    # place.
    for mod in _imported_modules(_tree("io/reader.py")):
        assert mod.split(".")[0] not in ("folder_paths", "comfy", "nodes"), (
            f"radiance/io/reader.py imports {mod!r} — that is ComfyUI, and the "
            "engine is supposed to run without it"
        )


def test_the_engine_pulls_in_no_node_module_of_its_own():
    # The AST check above reads the file; this one runs it. Import `radiance`
    # first to establish what the package __init__ drags in by itself, then
    # import the engine and diff. Anything from radiance.nodes in the delta is
    # the engine's doing.
    code = (
        _BOOT
        + "before=set(sys.modules);"
        "import radiance.io.reader as r;"
        "delta=[m for m in set(sys.modules)-before if m.startswith('radiance.nodes')];"
        "assert not delta, delta;"
        "assert callable(r.read_frames);"
        "print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0 and "No module named 'torch'" in r.stderr:
        pytest.skip("torch is not importable in a bare interpreter here")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ok" in r.stdout


def test_the_engine_reads_a_file_without_comfyui(tmp_path):
    # The point of the extraction, exercised end to end: no conftest, no node
    # import, no ComfyUI stubs — just a file on disk and frames out. Written
    # with Pillow rather than through Radiance so the fixture does not depend
    # on the thing under test.
    src = tmp_path / "plate.png"
    code = (
        "import tempfile, pathlib, json;"
        + _BOOT
        + "import numpy as np;"
        "from PIL import Image;"
        f"p = {str(src)!r};"
        "Image.fromarray((np.linspace(0,255,16*16*3).reshape(16,16,3)).astype('uint8')).save(p);"
        "from radiance.io.reader import read_frames;"
        "img, mask, info = read_frames(path=p, color_space='sRGB');"
        "assert tuple(img.shape) == (1,16,16,3), img.shape;"
        "assert info['kind'] == 'image', info;"
        "assert float(img.max()) > 0.9, float(img.max());"
        "print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    # Third-party only. A bare `No module named` catch would also swallow
    # `No module named 'radiance'`, which is exactly how the writer's version
    # of this test spent every CI run reporting itself as skipped.
    missing = re.search(r"No module named '([\w.]+)'", r.stderr)
    if r.returncode != 0 and missing and missing.group(1).split(".")[0] in _OPTIONAL:
        pytest.skip(f"a dependency is missing in a bare interpreter: {missing.group(1)}")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ok" in r.stdout


def _params(tree, qualname):
    """Parameter names of a module-level function or a method, in order."""
    *cls, name = qualname.split(".")
    scope = tree
    if cls:
        scope = next(n for n in tree.body
                     if isinstance(n, ast.ClassDef) and n.name == cls[0])
    fn = next(n for n in ast.walk(scope)
              if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name == name)
    a = fn.args
    return [p.arg for p in (a.posonlyargs + a.args + a.kwonlyargs) if p.arg != "self"]


#: The two the node keeps to itself. `browse` needs folder_paths; `reload`
#: exists only to change the IS_CHANGED hash and has never been read by the
#: read path at all.
_NODE_ONLY = ("browse", "reload")


def test_the_node_signature_still_matches_the_engine():
    # The node's parameter list is the widget contract: saved workflows are
    # matched against it by name and order, so it must not drift. The engine
    # takes the same list minus the two the node keeps, in the same order —
    # order included, because a silent reordering is how keyword defaults start
    # landing on the wrong widget.
    node = _params(_tree("nodes/io/write.py"), "RadianceRead.read")
    engine = _params(_tree("io/reader.py"), "read_frames")
    assert engine == [p for p in node if p not in _NODE_ONLY], (
        "the node and the engine disagree about the read signature:\n"
        f"  node   {node}\n  engine {engine}"
    )
    for p in _NODE_ONLY:
        assert p in node, f"{p} vanished from the node — that is a widget break"
        assert p not in engine, f"{p} reached the engine, which cannot use it"


def test_the_node_delegates_rather_than_decoding():
    # RadianceRead.read was 74 lines that called the decoders directly. If any
    # of them reappears in the method body, the engine has been forked rather
    # than called, and the two will drift the way the four grade paths did.
    tree = _tree("nodes/io/write.py")
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RadianceRead")
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == "read")
    called = {n.func.id for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    for decoder in ("_read_image", "_read_exr_with_info", "_read_video",
                    "_read_sequence", "_apply_input_colorspace", "_path_kind"):
        assert decoder not in called, (
            f"RadianceRead.read calls {decoder} directly — it should call the engine"
        )
    assert "_read_frames" in called, "RadianceRead.read no longer calls the engine"


def test_the_browse_widget_stayed_in_the_node_layer():
    # The mirror of the assertion above: the one genuinely node-layer thing in
    # the reader has to remain node-layer, or the engine grows a folder_paths
    # dependency by the back door.
    node = _tree("nodes/io/write.py")
    defined = {n.name for n in ast.walk(node)
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))}
    assert {"_resolve_browse", "_get_input_files"} <= defined, (
        "the browse helpers left the node layer — they need folder_paths"
    )
    # Against the engine's code, not its prose: the module docstring names
    # these two while explaining why they stayed, so a substring search over
    # the file would pass for the wrong reason.
    engine = _tree("io/reader.py")
    for n in ast.walk(engine):
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            assert n.name not in ("_resolve_browse", "_get_input_files"), (
                f"{n.name} was defined in the engine — that needs folder_paths"
            )
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name):
            assert n.func.id not in ("_resolve_browse", "_get_input_files"), (
                f"the engine calls {n.func.id}"
            )
