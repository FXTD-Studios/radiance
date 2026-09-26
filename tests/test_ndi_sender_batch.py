"""NDI Sender streams the whole batch, not image[0]."""
import sys
import types

import pytest
import torch

engine = pytest.importorskip("radiance.nodes.generate.engine")


def test_every_frame_is_sent(monkeypatch):
    sent = []
    ndi = types.SimpleNamespace(
        initialize=lambda: True,
        SendCreate=lambda: types.SimpleNamespace(),
        send_create=lambda desc: ("sender", desc),
        VideoFrameV2=lambda: types.SimpleNamespace(),
        FOURCC_VIDEO_TYPE_BGRA="BGRA",
        send_destroy=lambda inst: None,
        send_send_video_v2=lambda inst, vf: sent.append((vf.xres, vf.yres, vf.p_data[0, 0].tolist())),
    )
    monkeypatch.setitem(sys.modules, "NDIlib", ndi)
    Sender = engine.RadianceNDISender
    monkeypatch.setattr(Sender, "_ndi_send_instance", None)
    monkeypatch.setattr(Sender, "_ndi_stream_name", None)
    frames = torch.stack([torch.full((4, 6, 3), v) for v in (0.0, 0.5, 1.0)])
    _, ok = Sender().apply(frames, "test", "None (SDR)", True, 24.0)
    assert ok is True
    assert [p for _, _, p in sent] == [[0, 0, 0, 255], [128, 128, 128, 255], [255, 255, 255, 255]]
    assert Sender._ndi_send_instance[1].clock_video is True
