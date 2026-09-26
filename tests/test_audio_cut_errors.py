"""Audio Cut fails loudly; an empty cut list only ever means no cuts."""
import pytest

from radiance.nodes.pipeline import audio


def test_missing_file_raises():
    with pytest.raises(ValueError, match="Radiance Audio Cut"):
        audio.RadianceAudioCut().detect_cuts("/path/to/audio.wav", 24.0, "beats", 0.5, 12)


def test_backend_failure_raises(tmp_path, monkeypatch):
    wav = tmp_path / "a.wav"
    wav.write_bytes(b"RIFF0000WAVE")
    monkeypatch.setattr(audio, "_validate_audio_path", lambda p: str(p))
    def boom(*a, **k):
        raise OSError("decoder missing")
    monkeypatch.setattr(audio, "_detect_scipy", boom)
    monkeypatch.setattr(audio, "HAS_LIBROSA", False)
    monkeypatch.setattr(audio, "HAS_SCIPY", True)
    monkeypatch.setattr(audio, "HAS_NUMPY", True)
    with pytest.raises(RuntimeError, match="scipy backend failed"):
        audio.RadianceAudioCut().detect_cuts(str(wav), 24.0, "beats", 0.5, 12)


def test_ignored_method_is_reported(tmp_path, monkeypatch):
    monkeypatch.setattr(audio, "_validate_audio_path", lambda p: str(p))
    monkeypatch.setattr(audio, "_detect_scipy", lambda *a: [0, 24])
    monkeypatch.setattr(audio, "HAS_LIBROSA", False)
    monkeypatch.setattr(audio, "HAS_SCIPY", True)
    monkeypatch.setattr(audio, "HAS_NUMPY", True)
    frames, _, n, report = audio.RadianceAudioCut().detect_cuts("x.wav", 24.0, "beats", 0.5, 12)
    assert n == 2 and "no beats detector" in report
