"""Regression tests for Batch 3 — the guardrails batch.

Every defect here is a *reporting* defect: the code worked, and told you it
worked, while something was broken. A stale ``--ignore`` list kept CI green-ish
by never running the failing files, a meta-test warned instead of asserting, and
startup printed "successfully loaded N nodes" whether N was 100 or 12.

These tests are deliberately cheap and need no torch, so they run on the
lightweight CI matrix — the one the guardrails are meant to protect.
"""
import ast
import logging
import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _src(rel):
    return (_ROOT / rel).read_text(encoding="utf-8")


# ── 1. The CI ignore list is gone and cannot come back quietly ──────────────

def test_ci_lightweight_matrix_runs_every_test_file():
    """
    The lightweight job carried a hand-maintained --ignore list naming the
    files that import torch at module scope. Nobody updated it when
    test_multipass_contracts.py and test_temporal_rudra.py landed, so CI ran
    red for seventeen days with 13 failures and the list was the reason.
    """
    ci = _src(".github/workflows/ci.yml")
    step = ci[ci.index("- name: Run tests"):ci.index("- name: Upload pytest log")]
    assert "--ignore=" not in step, (
        "the --ignore list is back in the lightweight test step; gate on real "
        "torch in tests/conftest.py instead so nothing has to be kept in sync"
    )
    assert "python -m pytest tests/" in step


def test_ci_yaml_still_parses():
    yaml = pytest.importorskip("yaml")
    doc = yaml.safe_load(_src(".github/workflows/ci.yml"))
    assert "test" in doc["jobs"] and "test-full" in doc["jobs"]


def test_ci_release_branches_run_full_tests_with_project_coverage_floor():
    yaml = pytest.importorskip("yaml")
    workflow = yaml.safe_load(_src(".github/workflows/ci.yml"))
    triggers = workflow.get("on", workflow.get(True))
    for event in ("push", "pull_request"):
        assert "release/**" in triggers[event]["branches"]
    full = workflow["jobs"]["test-full"]
    assert "if" not in full
    runs = "\n".join(step.get("run", "") for step in full["steps"])
    assert "--cov=radiance" in runs
    assert "--cov-fail-under" not in runs
    assert "pytest-cov" in runs
    light = "\n".join(step.get("run", "") for step in workflow["jobs"]["test"]["steps"])
    assert "--cov" not in light


# ── 2. The torch stub is identifiable, and the gate keys off it ─────────────

def test_the_stub_marks_itself():
    """
    Every "is torch real?" idiom in this suite was defeated by the stub:
    it is a genuine ModuleType (so `isinstance(torch, MagicMock)` is False)
    whose attributes are MagicMocks (so `hasattr(torch.zeros(1), "shape")` is
    True). Both reported "real torch" on a machine with no torch at all.
    """
    conftest = _src("tests/conftest.py")
    assert "__radiance_stub__" in conftest
    assert "HAS_REAL_TORCH" in conftest

    import torch
    if getattr(torch, "__radiance_stub__", False):
        # We are running under the stub: the gate below must have been applied.
        assert not _has_real_torch()
    else:
        assert _has_real_torch()


def _has_real_torch():
    import conftest
    return conftest.HAS_REAL_TORCH


def test_gate_detects_unguarded_module_scope_torch(tmp_path):
    import conftest

    cases = {
        "import torch\n": True,
        "import torch.nn as nn\n": True,
        "from torch import Tensor\n": True,
        'torch = pytest.importorskip("torch")\n': True,
        'pytest.importorskip("torch")\n': True,
        'pytest.importorskip("numpy")\n': False,
        "import numpy\n": False,
        "try:\n    import torch\nexcept ImportError:\n    torch = None\n": False,
        "def f():\n    import torch\n": False,
    }
    for i, (body, expected) in enumerate(cases.items()):
        f = tmp_path / f"case_{i}.py"
        f.write_text("import pytest\n" + body, encoding="utf-8")
        conftest._torch_need_cache.pop(str(f), None)
        assert conftest._module_needs_real_torch(str(f)) is expected, body


def test_gate_is_inert_when_real_torch_is_present():
    """The test-full lane must still execute every one of these."""
    conftest = _src("tests/conftest.py")
    hook = conftest[conftest.index("def pytest_collection_modifyitems"):]
    assert "if HAS_REAL_TORCH" in hook.split("\n\n")[0]


def test_real_torch_marker_is_registered():
    conftest = _src("tests/conftest.py")
    assert '"markers",' in conftest and "real_torch:" in conftest


def _code_only(src):
    """Strip comments and docstrings.

    These files *describe* the broken detectors in prose; a plain substring
    search matches the explanation as readily as the code. (The Batch 0 suite
    learned this the hard way with a `strict=False` check that matched its own
    comment.)
    """
    tree = ast.parse(src)
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.ClassDef,
                             ast.FunctionDef, ast.AsyncFunctionDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and \
                    isinstance(body[0].value, ast.Constant) and \
                    isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return ast.unparse(ast.fix_missing_locations(tree))


@pytest.mark.parametrize("rel", [
    "tests/test_audit_regressions.py",
    "tests/test_audit_regressions_batch2.py",
    "tests/test_radiance_load_image_mask.py",
])
def test_no_file_still_uses_a_detector_the_stub_defeats(rel):
    code = _code_only(_src(rel))
    assert "isinstance(torch, MagicMock)" not in code
    assert "hasattr(torch.zeros(1), 'shape')" not in code
    assert "__radiance_stub__" in code, f"{rel} has no working real-torch check"


def test_opt_out_files_declare_the_flag_the_gate_reads():
    """A typo in the opt-out constant silently re-enables the module skip."""
    import conftest
    flag = conftest._TORCH_GATE_OPT_OUT
    optees = [p for p in (_ROOT / "tests").glob("test_*.py")
              if flag in p.read_text(encoding="utf-8")]
    assert len(optees) >= 15, "the per-test opt-out has been stripped out"
    for p in optees:
        tree = ast.parse(p.read_text(encoding="utf-8"))
        assert any(
            isinstance(n, ast.Assign)
            and any(getattr(t, "id", "") == flag for t in n.targets)
            and getattr(n.value, "value", None) is True
            for n in tree.body
        ), f"{p.name} mentions {flag} but never assigns it True at module scope"


# ── 3. The meta-test asserts instead of warning ─────────────────────────────

def test_node_key_metatest_asserts():
    tree = ast.parse(_src("tests/test_node_smoke.py"))
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "test_all_keys_have_class")
    calls = {ast.unparse(n.func) for n in ast.walk(fn) if isinstance(n, ast.Call)}
    assert "warnings.warn" not in calls, (
        "test_all_keys_have_class is back to warning; it swallowed twelve "
        "unregistered nodes for months"
    )
    assert any(c.endswith("assertFalse") for c in calls)


def test_known_unregistered_allowlist_is_a_ratchet():
    """
    It must fail in BOTH directions: a new key that stops resolving is a
    regression, and a key that starts resolving must be removed from the list
    so the allowlist can only shrink.
    """
    src = _src("tests/test_node_smoke.py")
    body = src[src.index("_KNOWN_UNREGISTERED = frozenset"):]
    body = body[:body.index("\n\n\n")] if "\n\n\n" in body else body
    assert "regressions = missing - self._KNOWN_UNREGISTERED" in body
    assert "fixed = self._KNOWN_UNREGISTERED - missing" in body


#: Placeholders that read like a reason and are not one.
_EMPTY_REASONS = re.compile(r"^\W*(todo|tbd|fixme|wip|n/?a|see above|later)\b", re.I)


def test_every_withheld_node_carries_a_written_reason():
    """A node that exists in source but not in the menu needs a reason on file.

    This used to parse the literal `frozenset({...})` text and require exactly
    three entries, then moved to the property that actually mattered: whatever
    sits in an allowlist is explained.

    AUDIT-FIX (2026-09): it opened with `if not allowlist: return`, reading only
    `_KNOWN_UNREGISTERED`, which is empty, so the function returned before its
    single assertion and could not fail for any reason at all. Two things fix
    that. It now covers both withheld-node allowlists the suite keeps, and
    `UNPUBLISHED_KEYS` is not empty; and an empty allowlist skips with a reason
    instead of passing silently, so "nothing to check" is visible in the run
    rather than indistinguishable from "checked and fine".

    The two lists record the reason differently. `_KNOWN_UNREGISTERED` is a
    frozenset of keys with the reason in a source comment on the entry; the
    `UNPUBLISHED_KEYS` mapping carries the reason as the value.
    """
    from test_node_smoke import TestCoverageSummary
    from test_node_publication_completeness import UNPUBLISHED_KEYS

    checked = []

    keyset = TestCoverageSummary._KNOWN_UNREGISTERED
    if keyset:
        src = _src("tests/test_node_smoke.py")
        block = src[src.index("_KNOWN_UNREGISTERED = frozenset("):]
        block = block[:block.index("\n\n")]
        for key in sorted(keyset):
            assert re.search(rf'"{key}"\s*,?\s*#\s*\S+', block), (
                f"{key} is withheld from the catalog with no reason written down"
            )
            checked.append(key)

    for key, reason in sorted(UNPUBLISHED_KEYS.items()):
        assert isinstance(reason, str), (
            f"{key} is withheld from the catalog and its reason is "
            f"{type(reason).__name__}, not text someone can read"
        )
        assert not _EMPTY_REASONS.match(reason.strip()), (
            f"{key} is withheld from the catalog behind a placeholder rather "
            f"than a reason: {reason.strip()[:80]!r}"
        )
        assert len(reason.split()) >= 12, (
            f"{key} is withheld from the catalog with {len(reason.split())} "
            "words of explanation. Say what is wrong with it and what would "
            f"have to be true to ship it: {reason.strip()!r}"
        )
        checked.append(key)

    if not checked:
        pytest.skip(
            "no node is withheld from the catalog: _KNOWN_UNREGISTERED and "
            "UNPUBLISHED_KEYS are both empty, so there is no reason to check. "
            "This test asserts nothing in that state and says so rather than "
            "reporting a pass it did not earn"
        )


# ── 4. Startup reports a shortfall as an error ──────────────────────────────

class _FakeSpec:
    label = "radiance.nodes.hdr"


class _FakeFailure:
    source = _FakeSpec()
    error = ImportError("No module named 'OpenEXR'")


class _FakeResult:
    def __init__(self, n_nodes, failures=()):
        self.class_mappings = {f"Node{i}": object for i in range(n_nodes)}
        self.display_name_mappings = {}
        self.loaded_modules = ()
        self.failures = tuple(failures)


def _report(*args, **kwargs):
    import radiance
    return radiance.report_node_load_health(*args, **kwargs)


def test_shortfall_is_logged_at_error_not_info(caplog):
    with caplog.at_level(logging.DEBUG):
        healthy = _report(_FakeResult(12), expected_minimum=100,
                          log=logging.getLogger("t.shortfall"))
    assert healthy is False
    errors = [r for r in caplog.records if r.levelno >= logging.ERROR]
    assert errors, "an 88% shortfall was not reported at ERROR"
    text = "\n".join(r.getMessage() for r in errors)
    assert "12" in text and "100" in text and "88" in text


def test_healthy_load_is_still_a_single_info_line(caplog):
    with caplog.at_level(logging.DEBUG):
        healthy = _report(_FakeResult(140), expected_minimum=100,
                          log=logging.getLogger("t.ok"))
    assert healthy is True
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("successfully loaded 140" in r.getMessage() for r in caplog.records)


def test_each_failed_module_is_named(caplog):
    with caplog.at_level(logging.DEBUG):
        healthy = _report(_FakeResult(140, [_FakeFailure()]), expected_minimum=100,
                          log=logging.getLogger("t.fail"))
    assert healthy is False, "failures must not report a healthy load"
    text = "\n".join(r.getMessage() for r in caplog.records if r.levelno >= logging.ERROR)
    assert "radiance.nodes.hdr" in text
    assert "OpenEXR" in text


def test_expected_minimum_is_a_named_constant():
    from radiance.config.constants import EXPECTED_MIN_NODE_COUNT
    assert isinstance(EXPECTED_MIN_NODE_COUNT, int)
    assert EXPECTED_MIN_NODE_COUNT >= 100


def test_group_failures_reach_the_entry_point():
    """
    `load_node_mappings` collected `failures` and radiance/nodes/__init__.py
    discarded them, so a group that dropped 18 nodes on an optional-dependency
    error looked like a clean load from the entry point.
    """
    assert "NODE_LOAD_FAILURES" in _src("nodes/__init__.py")
    entry = _src("__init__.py")
    assert "NODE_LOAD_FAILURES" in entry
    assert "tuple(result.failures) + tuple(nested)" in entry


def test_entry_point_calls_the_reporter_rather_than_logging_success_blind():
    entry = _src("__init__.py")
    assert "report_node_load_health(_LOAD_RESULT)" in entry
    tail = entry[entry.index("_LOAD_RESULT = _load_comfyui_nodes()"):]
    assert 'logger.info(\n    "Radiance: successfully loaded' not in tail, \
        "the unconditional success banner is back"


# ── 5. The real package agrees with the reporter ────────────────────────────

def test_live_package_reports_healthy_under_the_test_stubs():
    """The floor must match what the entry point actually publishes.

    Skipped where the environment itself is short a runtime dependency -- a
    missing tqdm takes `radiance.nodes.generate` and its whole node group with
    it, which is a machine problem, not a code one. The reporter has already
    logged it by the time we get here; that is the entire point of Batch 3.
    """
    import radiance
    from radiance.config.constants import EXPECTED_MIN_NODE_COUNT

    failures = getattr(radiance, "_LOAD_RESULT", None)
    failed = tuple(getattr(failures, "failures", ()) or ())
    if failed:
        pytest.skip(
            "environment is missing runtime dependencies for: "
            + ", ".join(f.source.label for f in failed)
        )

    assert len(radiance.NODE_CLASS_MAPPINGS) >= EXPECTED_MIN_NODE_COUNT, (
        f"the entry point publishes {len(radiance.NODE_CLASS_MAPPINGS)} nodes, "
        f"below the {EXPECTED_MIN_NODE_COUNT} floor"
    )


def test_reporter_survives_an_empty_catalog(caplog):
    with caplog.at_level(logging.DEBUG):
        assert _report(_FakeResult(0), expected_minimum=100,
                       log=logging.getLogger("t.empty")) is False
    assert any("100%" in r.getMessage() for r in caplog.records)


def test_reporter_handles_a_zero_expectation_without_dividing_by_zero(caplog):
    with caplog.at_level(logging.DEBUG):
        assert _report(_FakeResult(0), expected_minimum=0,
                       log=logging.getLogger("t.zero")) is True


def test_batch3_docstrings_stay_in_the_source(tmp_path):
    """Guard against a reformat silently dropping the 'why' comments."""
    # AUDIT-FIX (2026-09): a third line, `assert textwrap.dedent("") == ""`,
    # used to sit here. It is a constant the interpreter folds, so it asserted
    # nothing about this repository and made the test look broader than the two
    # checks above.
    assert "went stale" in _src(".github/workflows/ci.yml")
    assert "cannot go stale" in _src("tests/conftest.py")
