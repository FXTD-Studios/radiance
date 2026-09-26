"""Importing a node module must not require a running ComfyUI.

Retiring the root `nodes_*.py` layer moved six modules into packages, and that
changed what importing one of them costs. `radiance.nodes_realtime_preview` was
a top-level module: importing it ran that file and nothing else. Its
replacement, `radiance.nodes.monitor.realtime`, cannot be imported without
first running `radiance/nodes/monitor/__init__.py`, which imports the viewer,
which imported `delivery/handler.py`, which did:

    from aiohttp import web
    from server import PromptServer

unguarded, and then read `PromptServer.instance` at module scope. So the whole
Review group became unimportable outside ComfyUI. `nodes/gizmo.py` and
`nodes/pipeline/workspace.py` had guarded the same import for years; these two
had not, and the flat module had hidden it.

CI caught it — the import smoke-test job runs with only numpy, scipy, Pillow,
defusedxml and tqdm installed. This puts the same property in the normal suite,
generalised from one hand-listed module to every group, so the next module that
reaches for a server at import time fails here first.

aiohttp is a required ComfyUI dependency, so none of this degrades a real
install. The point is that the package stays readable by tooling that has
neither aiohttp nor a server: CI, linters, doc generators, packaging checks.
"""
import os
import subprocess
import sys
import textwrap

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.nodes.catalog import NODE_GROUPS  # noqa: E402

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_PARENT = os.path.dirname(REPO_ROOT)
#: Directory name of the checkout — "radiance" locally, "radiance-beta" on the
#: mirror CI clones. The harness needs it to map the import name.
REPO_DIRNAME = os.path.basename(REPO_ROOT)

#: Everything the import smoke-test stubs, and nothing else. In particular NOT
#: aiohttp and NOT `server` — those are the two this file exists to keep out.
_HARNESS = """
import sys, types
from unittest.mock import MagicMock

for name in ["torch", "torch.nn", "torch.nn.functional",
             "comfy", "comfy.samplers", "comfy.sample", "comfy.model_management",
             "comfy.utils", "comfy.model_base", "comfy.sd", "comfy.latent_formats",
             "comfy.nested_tensor", "comfy.cldm", "comfy.cldm.control_types",
             "folder_paths", "node_helpers"]:
    mod = types.ModuleType(name)
    mod.__dict__.update({a: MagicMock() for a in dir(MagicMock())})
    mod.__getattr__ = lambda n: MagicMock()
    sys.modules.setdefault(name, mod)

sys.modules["torch"].Tensor = object
for _n in ("float32", "float16", "bfloat16"):
    setattr(sys.modules["torch"], _n, _n)


# A real class, not MagicMock. `class X(SomeMixin, nn.Module)` raises
# "metaclass conflict" against a MagicMock base, which reads like a code defect
# and is purely an artefact of the stub — it is what makes CI's own smoke-test
# job log two module failures it then ignores.
class _Module:
    def __init__(self, *a, **k): pass
    def __call__(self, *a, **k): return MagicMock()
    def to(self, *a, **k): return self
    def eval(self): return self
    def train(self, *a, **k): return self
    def parameters(self): return iter(())
    def state_dict(self): return {}
    def load_state_dict(self, *a, **k): return None
    def register_buffer(self, *a, **k): return None


sys.modules["torch.nn"].Module = _Module

# `import torch.nn as nn` binds the name from `getattr(torch, "nn")`, and the
# stub's catch-all __getattr__ answers that with a fresh MagicMock — so `nn`
# was a mock whose `.Module` was a mock, and every `class X(Mixin, nn.Module)`
# raised "metaclass conflict". Wire the submodules onto the parent explicitly.
sys.modules["torch"].nn = sys.modules["torch.nn"]
sys.modules["torch.nn"].functional = sys.modules["torch.nn.functional"]

sys.modules["folder_paths"].models_dir = "/tmp"
sys.modules["folder_paths"].output_directory = "/tmp"
sys.modules["comfy.samplers"].SAMPLER_NAMES = []
sys.modules["comfy.samplers"].SCHEDULER_NAMES = []
sys.modules["node_helpers"].conditioning_set_values = lambda conditioning, values: conditioning
sys.modules["comfy.cldm.control_types"].UNION_CONTROLNET_TYPES = {}
sys.modules["comfy"].cldm = sys.modules["comfy.cldm"]
sys.modules["comfy.cldm"].control_types = sys.modules["comfy.cldm.control_types"]

# Make aiohttp and server unimportable even if the dev machine has them, so
# this test measures the same thing on a laptop as it does in CI.
#
# AUDIT-FIX (2026-09): this was a `find_module`/`load_module` finder. That is
# the pre-PEP-451 meta-path API and it was REMOVED in Python 3.12, which the CI
# matrix runs: `sys.meta_path` skips a finder with no `find_spec`, so on 3.12
# the blocker did nothing, aiohttp and server imported normally, and every test
# in this file measured nothing while still reporting pass. `find_spec` is the
# API that still exists. Raising straight out of it blocks the name for good,
# submodules included, and names itself in the message so the self-check below
# can tell "blocked" apart from "merely not installed".
class _Blocker:
    BLOCKED = {"aiohttp", "server"}
    def find_spec(self, name, path=None, target=None):
        if name.split(".")[0] in self.BLOCKED:
            raise ImportError(f"{name} is blocked by test_import_isolation")
        return None
sys.meta_path.insert(0, _Blocker())
for _m in [m for m in sys.modules if m.split(".")[0] in {"aiohttp", "server"}]:
    del sys.modules[_m]

# The blocker's own self-test, run in every subprocess this file spawns. A
# blocker that silently stops blocking is the defect above returning, and it
# turns all eleven group tests into green tests of nothing. The message check
# matters as much as the raise: on CI's lightweight lane aiohttp is not
# installed either, so a bare ModuleNotFoundError would prove nothing about
# whether the blocker is live.
for _blocked in sorted(_Blocker.BLOCKED):
    try:
        __import__(_blocked)
    except ImportError as _exc:
        if "blocked by test_import_isolation" not in str(_exc):
            print(f"HARNESS-BROKEN: {_blocked} is not blocked, only absent: {_exc!r}")
            raise SystemExit(2)
    else:
        print(f"HARNESS-BROKEN: the import blocker is inert, {_blocked} imported anyway")
        raise SystemExit(2)

sys.path.insert(0, %(parent)r)

# Map the import name `radiance` to this checkout regardless of the directory
# name. CI clones the mirror as "radiance-beta", where `import radiance` fails
# outright — so every group "failed" this test for a reason that had nothing to
# do with aiohttp, and the failure message blamed the code. conftest.py and
# ci.yml both carry this finder; a subprocess gets neither.
import importlib.abc, importlib.util, os
_ROOT = os.path.join(%(parent)r, %(pkgdir)r)
if os.path.basename(_ROOT) != "radiance" and os.path.exists(os.path.join(_ROOT, "__init__.py")):
    class _RadianceNameFinder(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name != "radiance":
                return None
            return importlib.util.spec_from_file_location(
                "radiance", os.path.join(_ROOT, "__init__.py"),
                submodule_search_locations=[_ROOT])
    sys.meta_path.insert(0, _RadianceNameFinder())

try:
    import radiance  # noqa: F401
except Exception as exc:
    print(f"HARNESS-BROKEN: the package itself will not import: {exc!r}")
    raise SystemExit(2)

__import__(%(module)r)
print("IMPORT-OK")
"""


def _import_in_clean_process(module: str):
    """Import *module* in a subprocess with no aiohttp and no server."""
    script = _HARNESS % {
        "parent": REPO_PARENT,
        "pkgdir": REPO_DIRNAME,
        "module": module,
    }
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        capture_output=True, text=True, timeout=180,
    )


GROUP_PATHS = [g.module_path for g in NODE_GROUPS]


def test_the_harness_can_import_the_package_at_all():
    """Fail here, loudly and once, when the harness is what is broken.

    Without this the first version of this file reported eleven identical
    "cannot be imported without aiohttp" failures on CI, when the real cause
    was that the mirror is checked out as `radiance-beta` and the subprocess
    had no name mapping, so `import radiance` failed before aiohttp entered
    into it. A test that misattributes its own breakage to the code under test
    is worse than no test.
    """
    result = _import_in_clean_process("radiance")

    assert "HARNESS-BROKEN" not in result.stdout, result.stdout.strip()
    assert "IMPORT-OK" in result.stdout, (
        "the isolation harness cannot import the package:\n"
        + (result.stderr or result.stdout).strip()[-1500:]
    )


def test_the_import_blocker_actually_blocks():
    """Import something the harness blocks, and require the blocker's refusal.

    Every other test in this file is worth exactly what the blocker is worth. It
    was a `find_module`/`load_module` finder, an API Python 3.12 removed, so on
    3.12 the harness quietly stopped blocking anything: aiohttp and `server`
    imported normally and eleven tests passed while measuring nothing.

    Both halves are asserted. The import must fail, and it must fail with the
    blocker's own message: on a lane where aiohttp is not installed a plain
    ModuleNotFoundError looks identical from outside and says nothing about
    whether the blocker is live.
    """
    for blocked in sorted(("aiohttp", "server")):
        result = _import_in_clean_process(blocked)

        assert "HARNESS-BROKEN" not in result.stdout, result.stdout.strip()
        assert "IMPORT-OK" not in result.stdout, (
            f"{blocked} imported inside the isolation harness, so the blocker "
            "is inert and every other test in this file measures nothing"
        )
        assert "blocked by test_import_isolation" in (result.stderr or ""), (
            f"{blocked} failed to import for some reason other than the "
            "blocker, so this proves nothing:\n"
            + (result.stderr or result.stdout).strip()[-800:]
        )


def test_the_blocker_uses_a_meta_path_api_that_still_exists():
    """`find_module` and `load_module` were removed in Python 3.12.

    The subprocess check above catches an inert blocker on the interpreter it
    runs under. This catches the same defect on 3.11, where the removed API
    still happens to work, so the CI matrix's 3.12 leg is not the only thing
    standing between the blocker and silence.
    """
    assert "def find_spec(self, name" in _HARNESS, (
        "the import blocker no longer implements find_spec, the only meta-path "
        "API that exists on Python 3.12+"
    )
    for removed in ("def find_module", "def load_module"):
        assert removed not in _HARNESS, (
            f"the harness is back on `{removed[4:]}`, removed in Python 3.12: "
            "sys.meta_path skips a finder without find_spec, so the blocker "
            "would do nothing there and this whole file would pass vacuously"
        )


@pytest.mark.parametrize("group", GROUP_PATHS)
def test_a_node_group_imports_without_aiohttp_or_a_server(group):
    result = _import_in_clean_process(group)

    if "IMPORT-OK" in result.stdout:
        return

    tail = "\n".join((result.stderr or result.stdout).strip().splitlines()[-12:])

    if "HARNESS-BROKEN" in result.stdout:
        pytest.fail(
            "the isolation harness is broken, so this says nothing about "
            f"{group}:\n\n{result.stdout.strip()}\n\n{tail}"
        )

    pytest.fail(
        f"{group} cannot be imported without aiohttp / ComfyUI's server.\n"
        "Guard the import (see nodes/gizmo.py) and do not touch PromptServer "
        f"at module scope.\n\n{tail}"
    )


def test_the_module_the_ci_smoke_test_names_is_one_that_exists():
    """The stale entry that produced the failure: a module path that moved.

    The CI job hard-codes a handful of module paths. One of them still read
    `radiance.nodes_realtime_preview` after that file was retired, which is a
    real failure but only discoverable by pushing. Check the list here instead.
    """
    import importlib.util
    import pathlib
    import re

    ci = pathlib.Path(__file__).resolve().parent.parent / ".github" / "workflows" / "ci.yml"
    if not ci.exists():
        pytest.skip("no CI workflow in this checkout")

    named = re.findall(r'\("(radiance\.[\w.]+)"', ci.read_text(encoding="utf-8"))
    assert named, "the smoke-test module list is no longer parseable from ci.yml"

    missing = [m for m in named if importlib.util.find_spec(m) is None]
    assert not missing, f"ci.yml smoke-tests modules that no longer exist: {missing}"


def test_the_guarded_imports_are_actually_guarded():
    """A grep-level backstop, so the reason survives a future edit."""
    import pathlib

    root = pathlib.Path(__file__).resolve().parent.parent
    for rel in ("nodes/monitor/viewer.py", "delivery/handler.py",
                "nodes/gizmo.py", "nodes/pipeline/workspace.py"):
        src = (root / rel).read_text(encoding="utf-8")
        for line_no, line in enumerate(src.splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith(("from aiohttp import", "import aiohttp",
                                    "from server import", "import server")):
                assert line.startswith((" ", "\t")), (
                    f"{rel}:{line_no} imports aiohttp/server at module scope, "
                    "unguarded — that makes the whole node group unimportable "
                    "without a running ComfyUI"
                )
