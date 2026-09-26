"""The write engine sits below the node layer, and that is checkable.

`delivery/handler.py` used to do `from radiance.nodes.io.write import
RadianceWrite`, instantiate the node, and call `.write()` on it. The delivery
path reaching up into the node layer is backwards, and it cost something real:
the code that decides whether a master lands on disk correctly could not be
exercised without importing ComfyUI's node surface.

`radiance/io/writer.py` is that code, moved down a floor. These tests are what
stops it drifting back up — including the quiet version, where the import moves
inside a function body and the module header still claims independence.

The last two are the ones that would have caught the delivery bug that shipped:
the handler called `writer.write(filename_prefix=..., output_format=...,
output_color_space=...)`, none of which were parameters, and omitted the
required `format`. Every delivery raised TypeError, the handler swallowed it,
and the endpoint returned HTTP 200 with status "error".
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
# checkout is called.
#
# These tests used to do `sys.path.insert(0, _ROOT.parent); import radiance`,
# which only works when the repository directory happens to be named
# `radiance`. It is on a working copy cloned as `radiance`; it is not on CI,
# where GitHub checks the repo out as `radiance-beta/radiance-beta`, and
# `import radiance` there is ModuleNotFoundError. Loading from the __init__
# path under an explicit module name removes the dependency on the folder name
# entirely -- and it is the package `__init__` that runs either way, which is
# the whole point of the measurement below.
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
    for mod in _imported_modules(_tree("io/writer.py")):
        assert "nodes" not in mod.split("."), (
            f"radiance/io/writer.py imports {mod!r} — the engine is reaching "
            "back up into the node layer it was extracted from"
        )


def test_the_engine_pulls_in_no_node_module_of_its_own():
    # The AST check above reads the file; this one runs it. In a subprocess
    # with no conftest and no ComfyUI stubs, import `radiance` first to
    # establish what the package __init__ drags in by itself, then import the
    # engine and diff. Anything from radiance.nodes in the delta is the
    # engine's doing.
    #
    # Measured as a delta rather than an absolute because radiance/__init__.py
    # registers the node modules at import time -- that is the package's own
    # arrangement and not something the engine can or should change.
    code = (
        _BOOT
        + "before=set(sys.modules);"
        "import radiance.io.writer as w;"
        "delta=[m for m in set(sys.modules)-before if m.startswith('radiance.nodes')];"
        "assert not delta, delta;"
        "assert callable(w.write_frames);"
        "print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    if r.returncode != 0 and "No module named 'torch'" in r.stderr:
        pytest.skip("torch is not importable in a bare interpreter here")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ok" in r.stdout


def test_the_engine_writes_a_file_without_comfyui():
    # The point of the extraction, exercised end to end: no conftest, no node
    # import, no ComfyUI stubs -- just frames in and a file on disk.
    code = (
        "import tempfile, pathlib;"
        + _BOOT
        + "import numpy as np, torch;"
        "from radiance.io.writer import write_frames;"
        "d=tempfile.mkdtemp();"
        "img=torch.from_numpy(np.full((1,4,4,3), 0.5, dtype=np.float32));"
        "p,n=write_frames(image=img, output_path=str(pathlib.Path(d)/'m'),"
        " format='IMG \u2502 PNG (8-bit)');"
        "assert n==1 and pathlib.Path(p).is_file() and pathlib.Path(p).stat().st_size>0, (p,n);"
        "print('ok')"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    # Third-party only. A bare `No module named` catch also swallowed
    # `No module named 'radiance'` -- so when the folder-name assumption above
    # was wrong, this test reported itself as skipped rather than broken, and
    # went on doing that on every CI run.
    missing = re.search(r"No module named '([\w.]+)'", r.stderr)
    if r.returncode != 0 and missing and missing.group(1).split(".")[0] in _OPTIONAL:
        pytest.skip(f"a dependency is missing in a bare interpreter: {missing.group(1)}")
    assert r.returncode == 0, r.stderr[-2000:]
    assert "ok" in r.stdout


def test_the_delivery_handler_does_not_reach_for_the_node():
    src = (_ROOT / "delivery/handler.py").read_text(encoding="utf-8")
    assert "RadianceWrite()" not in src, (
        "delivery/handler.py is instantiating the node again"
    )
    for mod in _imported_modules(_tree("delivery/handler.py")):
        assert not mod.startswith("radiance.nodes"), (
            f"delivery/handler.py imports {mod!r}"
        )


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


def test_the_node_signature_still_matches_the_engine():
    # The node's parameter list is the widget contract: saved workflows are
    # matched against it by name and order, so it must not drift. The engine
    # is allowed the one extra parameter the node supplies for it.
    node = _params(_tree("nodes/io/write.py"), "RadianceWrite.write")
    engine = _params(_tree("io/writer.py"), "write_frames")
    assert engine[:len(node)] == node, (
        "the node and the engine disagree about the write signature:\n"
        f"  node   {node}\n  engine {engine}"
    )
    assert engine[len(node):] == ["read_media"], (
        f"unexpected extra engine parameters: {engine[len(node):]}"
    )


def test_the_handler_passes_only_real_parameters():
    # This is the shipped bug, as a test. Reads the actual call site rather
    # than a list of names retyped here.
    tree = _tree("delivery/handler.py")
    call = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "write_frames"
    )
    passed = {kw.arg for kw in call.keywords if kw.arg}
    engine = set(_params(_tree("io/writer.py"), "write_frames"))
    unknown = passed - engine
    assert not unknown, f"delivery/handler.py passes {sorted(unknown)} — not parameters of write_frames"
    assert not call.args, "the handler should pass write_frames' arguments by keyword"
    # `format` has no default; omitting it is a TypeError at delivery time.
    assert "format" in passed and "image" in passed and "output_path" in passed
