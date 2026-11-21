"""B's own frame rate reaches the viewer (VIDEO input only), so compare can follow A by time."""
from pathlib import Path
from types import SimpleNamespace

from radiance.nodes.monitor.viewer import _video_fps

SRC = (Path(__file__).resolve().parents[1] / "nodes" / "monitor" / "viewer.py").read_text(encoding="utf-8")


class _Video:
    def __init__(self, rate):
        self._rate = rate

    def get_components(self):
        return SimpleNamespace(frame_rate=self._rate, images=None)


def test_video_input_reports_its_rate():
    assert _video_fps(_Video(30)) == 30.0
    assert _video_fps(_Video(23.976)) == 23.976


def test_image_batch_and_unknown_rate_report_none():
    assert _video_fps(object()) is None
    assert _video_fps(_Video(0)) is None
    assert _video_fps(_Video(None)) is None


def test_payload_carries_compare_fps_only_when_known():
    assert 'compare_fps = _video_fps(compare_image) if compare_image is not None else None' in SRC
    assert 'ui_payload["compare_fps"] = [compare_fps]' in SRC
    assert SRC.index('compare_fps = _video_fps(compare_image)') < SRC.index('compare_image = _extract_tensor(compare_image)')
