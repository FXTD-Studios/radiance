"""RADIANCE_LOG_LEVEL must actually set the level.

The startup shortfall error tells users to "re-run with
RADIANCE_LOG_LEVEL=DEBUG for tracebacks". Nothing read that variable until
2026-08, so the instruction printed by an error message did nothing — the
same class of dead control tests/test_dead_controls.py exists to prevent.
"""
import logging
import os
import pathlib
import sys

import pytest

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from radiance.core.logging import resolve_log_level, setup_radiance_logging


class TestResolveLogLevel:

    def test_unset_falls_back_to_info(self, monkeypatch):
        monkeypatch.delenv("RADIANCE_LOG_LEVEL", raising=False)
        assert resolve_log_level() == logging.INFO

    @pytest.mark.parametrize("name,expected", [
        ("DEBUG", logging.DEBUG),
        ("debug", logging.DEBUG),
        ("WARNING", logging.WARNING),
        ("ERROR", logging.ERROR),
    ])
    def test_named_levels(self, monkeypatch, name, expected):
        monkeypatch.setenv("RADIANCE_LOG_LEVEL", name)
        assert resolve_log_level() == expected

    def test_numeric_levels(self, monkeypatch):
        monkeypatch.setenv("RADIANCE_LOG_LEVEL", "25")
        assert resolve_log_level() == 25

    def test_garbage_falls_back_rather_than_raising(self, monkeypatch):
        """This runs during import; raising would take the whole pack down."""
        monkeypatch.setenv("RADIANCE_LOG_LEVEL", "loud please")
        assert resolve_log_level() == logging.INFO

    def test_empty_falls_back(self, monkeypatch):
        monkeypatch.setenv("RADIANCE_LOG_LEVEL", "   ")
        assert resolve_log_level() == logging.INFO


class TestSetupHonoursIt:

    def test_the_logger_and_its_handler_both_move(self, monkeypatch):
        monkeypatch.setenv("RADIANCE_LOG_LEVEL", "DEBUG")
        logger = setup_radiance_logging()
        try:
            assert logger.level == logging.DEBUG
            assert logger.isEnabledFor(logging.DEBUG)
            assert all(h.level <= logging.DEBUG for h in logger.handlers)
        finally:
            monkeypatch.delenv("RADIANCE_LOG_LEVEL", raising=False)
            setup_radiance_logging()

    def test_an_explicit_argument_still_wins(self, monkeypatch):
        monkeypatch.setenv("RADIANCE_LOG_LEVEL", "DEBUG")
        try:
            assert setup_radiance_logging(logging.ERROR).level == logging.ERROR
        finally:
            monkeypatch.delenv("RADIANCE_LOG_LEVEL", raising=False)
            setup_radiance_logging()


def test_the_error_message_names_a_variable_that_exists():
    """The advice in __init__.py must stay true."""
    src = pathlib.Path(__file__).resolve().parent.parent / "__init__.py"
    text = src.read_text(encoding="utf-8")
    if "RADIANCE_LOG_LEVEL" not in text:
        pytest.skip("the startup error no longer advertises the variable")

    from radiance.config.env import ENV
    assert getattr(ENV, "RADIANCE_LOG_LEVEL", None) == "RADIANCE_LOG_LEVEL"
