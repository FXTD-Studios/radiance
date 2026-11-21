"""Startup acts on the runtime dependency check instead of discarding it.

`validate_runtime_dependencies(logger)` returned False when a required package
was missing and the entry point threw the result away, so the closing
"successfully loaded N nodes" line read as a clean start. An exception inside
the check (the status table, a metadata scan) also stopped Radiance loading
at all.
"""
import logging
import pathlib

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent


class _Spec:
    def __init__(self, name, hint):
        self.display_name, self.install_hint = name, hint


def test_missing_required_dependencies_are_named_in_one_warning(monkeypatch, caplog):
    import radiance
    monkeypatch.setattr(radiance, "validate_runtime_dependencies", lambda log: False)
    monkeypatch.setattr(radiance, "missing_dependencies", lambda specs: (
        _Spec("OpenEXR", "pip install OpenEXR"), _Spec("opencolorio", "pip install opencolorio")))
    log = logging.getLogger("t.deps.missing")
    with caplog.at_level(logging.DEBUG, logger="t.deps.missing"):
        missing = radiance.check_runtime_dependencies(log)
    assert missing == ("OpenEXR", "opencolorio")
    warnings = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert len(warnings) == 1, warnings
    assert "OpenEXR" in warnings[0] and "opencolorio" in warnings[0]
    assert "pip install OpenEXR" in warnings[0]


def test_all_present_logs_nothing_extra(monkeypatch, caplog):
    import radiance
    monkeypatch.setattr(radiance, "validate_runtime_dependencies", lambda log: True)
    with caplog.at_level(logging.DEBUG, logger="t.deps.ok"):
        assert radiance.check_runtime_dependencies(logging.getLogger("t.deps.ok")) == ()
    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_failing_check_never_stops_startup(monkeypatch, caplog):
    import radiance

    def boom(log):
        raise RuntimeError("table renderer broke")

    monkeypatch.setattr(radiance, "validate_runtime_dependencies", boom)
    with caplog.at_level(logging.DEBUG, logger="t.deps.boom"):
        assert radiance.check_runtime_dependencies(logging.getLogger("t.deps.boom")) == ()
    assert any("table renderer broke" in r.getMessage() for r in caplog.records
               if r.levelno >= logging.WARNING)


def test_entry_point_keeps_the_result_and_repeats_it_after_the_load_banner():
    entry = (_ROOT / "__init__.py").read_text(encoding="utf-8")
    assert "\nvalidate_runtime_dependencies(logger)\n" not in entry, "result discarded again"
    assert "_MISSING_REQUIRED_DEPENDENCIES = check_runtime_dependencies(" in entry
    tail = entry[entry.index("report_node_load_health(_LOAD_RESULT)"):]
    assert "_MISSING_REQUIRED_DEPENDENCIES" in tail


def test_package_exposes_the_missing_list():
    import radiance
    assert isinstance(radiance._MISSING_REQUIRED_DEPENDENCIES, tuple)
