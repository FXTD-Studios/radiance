"""Console diagnostics must not break a render on Windows redirected output."""
import io
import logging
import sys

import pytest

from radiance.core.logging import RadianceConsoleFormatter


@pytest.mark.parametrize("encoding", ["cp1252", "ascii", "utf-8"])
@pytest.mark.parametrize("color", [False, True])
def test_unicode_message_and_traceback_fit_the_stream(monkeypatch, encoding, color):
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding=encoding)
    monkeypatch.setattr("sys.stdout", stream)
    formatter = RadianceConsoleFormatter(use_color=color, use_unicode=False)
    handler = logging.StreamHandler(stream)
    handler.setFormatter(formatter)
    try:
        raise ValueError("shot \u25ce failed")
    except ValueError:
        record = logging.LogRecord("radiance.io", logging.ERROR, __file__, 1,
                                   "write \u2192 %s", ("\u65e5\u672c.exr",), sys.exc_info())
    handler.emit(record)
    stream.flush()
    output = raw.getvalue().decode(encoding)
    assert "write" in output and "ValueError" in output and ".exr" in output
    if encoding == "utf-8":
        assert "\u2192" in output and "\u65e5\u672c.exr" in output
