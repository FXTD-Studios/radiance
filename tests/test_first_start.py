"""The first-start checks (core/first_start.py): OCIO + ACES views, ffmpeg's
10-bit encoders, Write's default output folder, and Hugging Face access.

No test reaches the network: the Hub is a fake (see conftest).
"""
from __future__ import annotations

import urllib.error

import pytest

from radiance.core import first_start as fs
from radiance.core import ffmpeg as ff


# ── OCIO ─────────────────────────────────────────────────────────────────────

def test_the_built_in_aces_config_passes(monkeypatch):
    pytest.importorskip("PyOpenColorIO")
    from radiance.color import ocio_setup
    env = {}
    ocio_setup.STATE["configured"] = False
    c = fs.check_ocio(env)
    assert c["status"] == "ok", c
    assert "ACES 2.0 SDR and HDR" in c["detail"]


def test_a_config_without_aces_2_views_is_a_warning_with_a_fix(monkeypatch, tmp_path):
    ocio = pytest.importorskip("PyOpenColorIO")
    cfg = ocio.Config.CreateRaw()
    path = tmp_path / "plain.ocio"
    path.write_text(cfg.serialize())
    from radiance.color import ocio_setup
    env = {"OCIO": str(path)}
    ocio_setup.STATE["configured"] = False
    c = fs.check_ocio(env)
    assert c["status"] == "warn"
    assert "no ACES 2.0 SDR and HDR view" in c["detail"]
    assert c["fix"] == {"action": "ocio", "label": "Use the ACES studio config"}
    ocio_setup.STATE["configured"] = False


def test_no_opencolorio_is_bad_and_says_what_to_install(monkeypatch):
    from radiance.color import ocio_setup
    monkeypatch.setattr(ocio_setup, "configure_ocio", lambda env=None: ocio_setup.STATE)
    monkeypatch.setitem(ocio_setup.STATE, "configured", False)
    monkeypatch.setitem(ocio_setup.STATE, "error", "OpenColorIO not installed")
    c = fs.check_ocio({})
    assert c["status"] == "bad"
    assert "pip install" in c["detail"] and "2.5.1" in c["detail"]


def test_an_old_opencolorio_is_named():
    assert fs._version_lt("2.4.2", (2, 5, 1))
    assert not fs._version_lt("2.5.1", (2, 5, 1))
    assert not fs._version_lt("2.6.0", (2, 5, 1))


# ── ffmpeg ───────────────────────────────────────────────────────────────────

def _encoders(monkeypatch, have):
    def with_encoder(enc):
        if enc in have:
            return "/usr/bin/ffmpeg"
        raise RuntimeError("no")
    monkeypatch.setattr(ff, "ffmpeg_with_encoder", with_encoder)


def test_ffmpeg_with_both_encoders_is_ok(monkeypatch):
    _encoders(monkeypatch, {"libx265", "prores_ks"})
    c = fs.check_ffmpeg()
    assert c["status"] == "ok" and "H.265 10-bit and ProRes" in c["detail"]


def test_ffmpeg_without_x265_is_a_warning_naming_it(monkeypatch):
    _encoders(monkeypatch, {"prores_ks"})
    c = fs.check_ffmpeg()
    assert c["status"] == "warn" and "H.265 10-bit" in c["detail"] and "RADIANCE_FFMPEG" in c["detail"]


def test_no_ffmpeg_is_bad(monkeypatch):
    _encoders(monkeypatch, set())

    def missing():
        raise RuntimeError("ffmpeg was not found")
    monkeypatch.setattr(ff, "ffmpeg_exe", missing)
    c = fs.check_ffmpeg()
    assert c["status"] == "bad" and "imageio-ffmpeg" in c["detail"]


# ── output folder ────────────────────────────────────────────────────────────

def test_a_missing_output_folder_offers_to_make_it(monkeypatch, tmp_path):
    monkeypatch.setattr(fs, "default_output", lambda: tmp_path / "radiance_output")
    c = fs.check_output()
    assert c["status"] == "warn" and c["fix"]["action"] == "output"
    after = fs.fix("output")
    assert (tmp_path / "radiance_output").is_dir()
    assert after["status"] in ("ok", "warn") and after["fix"] is None
    assert not list((tmp_path / "radiance_output").iterdir()), "the write test leaves nothing behind"


def test_a_nearly_full_disk_is_a_warning(monkeypatch, tmp_path):
    import shutil
    monkeypatch.setattr(shutil, "disk_usage", lambda p: type("U", (), {"free": 5 * 1024 ** 3})())
    c = fs.check_output(tmp_path)
    assert c["status"] == "warn" and "5.0 GB free" in c["detail"]


def test_an_unknown_fix_is_refused():
    with pytest.raises(KeyError):
        fs.fix("rm -rf")


# ── Hugging Face ─────────────────────────────────────────────────────────────

def _hub(answers):
    """A fake Hub: answers[repo substring] -> HTTP code; anything else 200."""
    seen = []

    def head(url, token="", timeout=0):
        seen.append((url, token))
        for key, code in answers.items():
            if key in url:
                if code == "down":
                    raise urllib.error.URLError("down")
                raise urllib.error.HTTPError(url, code, "x", {}, None)
        return {}
    return head, seen


def test_downloads_off_skips_the_network():
    head, seen = _hub({})
    c = fs.check_hub(head=head, allowed_fn=lambda: False, token_fn=lambda: "t")
    assert c["status"] == "warn" and "Downloads are off" in c["detail"] and not seen


def test_an_unreachable_hub_is_bad():
    head, _ = _hub({"huggingface.co": "down"})
    c = fs.check_hub(head=head, allowed_fn=lambda: True, token_fn=lambda: "")
    assert c["status"] == "bad" and "can't be reached" in c["detail"]


def test_no_token_says_how_to_log_in_and_never_asks_for_one():
    head, seen = _hub({})
    c = fs.check_hub(head=head, allowed_fn=lambda: True, token_fn=lambda: None)
    assert c["status"] == "warn" and "hf auth login" in c["detail"]
    assert c["fix"]["url"].startswith("https://huggingface.co/")
    assert all(tok == "" for _, tok in seen)


def test_a_licence_not_accepted_is_named_and_the_token_never_shown():
    from radiance.core import setup_check
    gated = fs._gated_repos(setup_check.load_catalog())
    assert gated, "the catalog has gated models"
    repo = sorted(gated)[0]
    head, seen = _hub({repo: 403})
    c = fs.check_hub(head=head, allowed_fn=lambda: True, token_fn=lambda: "hf_secret_value")
    assert c["status"] == "warn" and repo in c["detail"]
    assert c["fix"]["url"] == f"https://huggingface.co/{repo}"
    assert "hf_secret_value" not in repr(c)
    assert {u for u, tok in seen if tok} <= set(gated.values()), "the token only goes to the Hub's own files"


def test_every_licence_accepted_is_ok():
    head, _ = _hub({})
    c = fs.check_hub(head=head, allowed_fn=lambda: True, token_fn=lambda: "t")
    assert c["status"] == "ok" and "licences accepted" in c["detail"]


# ── all together ─────────────────────────────────────────────────────────────

def test_the_scan_carries_four_checks_and_one_failing_check_hides_none(monkeypatch):
    def boom():
        raise ValueError("broken")
    monkeypatch.setattr(fs, "check_ffmpeg", boom)
    out = fs.checks()
    assert [c["id"] for c in out] == ["ocio", "ffmpeg", "output", "hub"]
    assert out[1]["status"] == "warn" and "broken" in out[1]["detail"]
    for c in out:
        assert c["status"] in ("ok", "warn", "bad") and c["title"] and c["detail"]
