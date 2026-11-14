"""Audio Transcribe: a long segment split by max_segment_chars gets its own
timings per chunk.

Every chunk but the last used to keep the segment's start, so the chunks
overlapped and all began at the same moment.
"""
import json

import pytest

from radiance.nodes.pipeline import audio


_LONG = ("alpha bravo charlie delta echo foxtrot golf hotel india juliet "
         "kilo lima mike november oscar papa quebec romeo sierra tango")


def _transcribe(monkeypatch, segments, max_chars, include_timings=True):
    monkeypatch.setattr(audio, "_validate_audio_path", lambda p: str(p))
    monkeypatch.setattr(audio, "_transcribe_whisper_cli",
                        lambda *a, **k: (" ".join(s["text"] for s in segments), segments))
    _text, seg_json, n, report = audio.RadianceAudioTranscribe().transcribe(
        "x.wav", "whisper_cli", "base", "auto",
        include_timings=include_timings, max_segment_chars=max_chars)
    out = json.loads(seg_json)
    assert n == len(out), report
    return out


def test_split_chunks_tile_the_segment_in_order(monkeypatch):
    out = _transcribe(monkeypatch, [{"start": 10.0, "end": 30.0, "text": _LONG}], 25)
    assert len(out) >= 4
    assert " ".join(c["text"] for c in out) == _LONG
    assert out[0]["start"] == pytest.approx(10.0)
    assert out[-1]["end"] == pytest.approx(30.0)
    for prev, cur in zip(out, out[1:]):
        # Contiguous, strictly advancing: no chunk starts where the segment did.
        assert cur["start"] == pytest.approx(prev["end"])
        assert cur["start"] > prev["start"]
    for c in out:
        assert c["end"] > c["start"]
        assert len(c["text"]) <= 25


def test_chunk_durations_follow_their_share_of_the_characters(monkeypatch):
    seg = {"start": 0.0, "end": 12.0, "text": "aaaa bbbb cccccccc"}
    out = _transcribe(monkeypatch, [seg], 9)
    assert [c["text"] for c in out] == ["aaaa bbbb", "cccccccc"]
    # 8 of 16 letters each: half the segment each.
    assert out[0]["start"] == pytest.approx(0.0) and out[0]["end"] == pytest.approx(6.0)
    assert out[1]["start"] == pytest.approx(6.0) and out[1]["end"] == pytest.approx(12.0)


def test_short_segments_and_untimed_output_are_unchanged(monkeypatch):
    segs = [{"start": 0.0, "end": 1.0, "text": "hi there"},
            {"start": 1.0, "end": 9.0, "text": _LONG}]
    out = _transcribe(monkeypatch, [dict(s) for s in segs], 30)
    assert out[0] == segs[0]
    assert out[1]["start"] == pytest.approx(1.0) and out[-1]["end"] == pytest.approx(9.0)
    untimed = _transcribe(monkeypatch, [dict(s) for s in segs], 30, include_timings=False)
    assert all(set(c) == {"text"} for c in untimed)
