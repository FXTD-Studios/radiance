"""install.py: libraries new enough, RUDRA ACES 2 in place, nothing for the user to do."""
import importlib.util
import shutil
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(path):
    spec = importlib.util.spec_from_file_location("_radiance_install_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def comfy_tree(tmp_path):
    """<tmp>/ComfyUI/custom_nodes/radiance with install.py and the download module."""
    pkg = tmp_path / "ComfyUI" / "custom_nodes" / "radiance"
    (pkg / "model").mkdir(parents=True)
    (tmp_path / "ComfyUI" / "models").mkdir()
    shutil.copy(ROOT / "install.py", pkg / "install.py")
    shutil.copy(ROOT / "model" / "rudra_download.py", pkg / "model" / "rudra_download.py")
    return pkg


def test_diffusers_minimum_is_the_one_wan_vace_needs():
    inst = _load(ROOT / "install.py")
    req = {mod: minimum for mod, _, minimum, _ in inst.REQUIRED}
    assert req["diffusers"] == "0.35.0" and req["PyOpenColorIO"] == "2.5.0"
    for f in ["requirements.txt", "requirements_windows.txt", "requirements_linux.txt",
              "requirements_mac_silicon.txt", "pyproject.toml"]:
        assert "diffusers>=0.35.0" in (ROOT / f).read_text(), f


def test_an_old_library_is_upgraded(monkeypatch):
    inst = _load(ROOT / "install.py")
    monkeypatch.setattr(inst.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(inst, "_too_old", lambda dist, minimum: dist == "diffusers")
    assert inst._missing() == ["diffusers>=0.35.0"]
    assert inst._version_tuple("0.34.0.dev0") < inst._version_tuple("0.35.0")


def test_prefetch_puts_rudra_aces2_in_models_radiance(comfy_tree, monkeypatch):
    monkeypatch.delenv("RADIANCE_ALLOW_DOWNLOADS", raising=False)
    monkeypatch.delenv("HF_HUB_OFFLINE", raising=False)
    monkeypatch.delenv("TRANSFORMERS_OFFLINE", raising=False)
    inst = _load(comfy_tree / "install.py")
    calls = []
    real = importlib.util.spec_from_file_location

    def spy(name, path, *a, **k):
        spec = real(name, path, *a, **k)
        orig = spec.loader.exec_module

        def exec_module(mod):
            orig(mod)
            mod.download_to = lambda pkg, folder, log=print: calls.append((pkg.name, Path(folder)))
        spec.loader.exec_module = exec_module
        return spec
    monkeypatch.setattr(inst.importlib.util, "spec_from_file_location", spy)
    inst._prefetch_rudra()
    target = comfy_tree.parents[1] / "models" / "radiance" / "RUDRA_ACES_2"
    assert calls == [("RUDRA_ACES_2", target)]


def test_prefetch_respects_downloads_off(comfy_tree, monkeypatch, capsys):
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "0")
    inst = _load(comfy_tree / "install.py")
    inst._prefetch_rudra()
    assert "downloads are off" in capsys.readouterr().out
    assert not (comfy_tree.parents[1] / "models" / "radiance").exists()


def test_prefetch_never_fails_the_install(comfy_tree, monkeypatch, capsys):
    monkeypatch.setenv("RADIANCE_ALLOW_DOWNLOADS", "1")
    inst = _load(comfy_tree / "install.py")
    (comfy_tree / "model" / "rudra_download.py").write_text("raise RuntimeError('boom')\n")
    inst._prefetch_rudra()
    assert "downloads on first use" in capsys.readouterr().out
