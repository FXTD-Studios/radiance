"""OCIO is configured automatically; OpenCV's EXR codec is always on."""
import os

import pytest

_ocio = pytest.importorskip("PyOpenColorIO")
# Another test can leave a stand-in module under this name; only the real
# library (a file on disk) can be configured.
if not getattr(_ocio, "__file__", None) or not hasattr(_ocio, "GetCurrentConfig"):
    pytest.skip("PyOpenColorIO is not the real library here", allow_module_level=True)

from radiance.color import ocio_setup


@pytest.fixture(autouse=True)
def _restore_ocio_current_config():
    import PyOpenColorIO as O
    previous = O.GetCurrentConfig()
    yield
    O.SetCurrentConfig(previous)


def test_no_ocio_env_gets_the_studio_config(tmp_path, monkeypatch):
    monkeypatch.setattr(ocio_setup, "ACES_DIR", str(tmp_path))
    monkeypatch.setattr(ocio_setup, "STUDIO_FILE", str(tmp_path / "studio-config.ocio"))
    env = {}
    state = ocio_setup.configure_ocio(env)
    assert state["configured"] and state["name"].startswith("studio-config")
    assert env["OCIO"] == str(tmp_path / "studio-config.ocio")
    assert os.path.isfile(env["OCIO"]) and state["colorspaces"] >= 50
    import PyOpenColorIO as O
    cfg = O.Config.CreateFromFile(env["OCIO"])
    for name in ("ARRI LogC4", "S-Log3 S-Gamut3.Cine", "ACEScg", "Linear Rec.709 (sRGB)"):
        assert cfg.getColorSpace(name) is not None, name
    assert O.GetCurrentConfig().getName() == state["name"]


def test_a_user_ocio_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(ocio_setup, "STUDIO_FILE", str(tmp_path / "s.ocio"))
    env = {"OCIO": ocio_setup.BUNDLED_CG}
    state = ocio_setup.configure_ocio(env)
    assert state["source"] == "$OCIO (yours)" and env["OCIO"] == ocio_setup.BUNDLED_CG


def test_a_broken_user_ocio_falls_back_without_overwriting_it(tmp_path, monkeypatch):
    monkeypatch.setattr(ocio_setup, "ACES_DIR", str(tmp_path))
    monkeypatch.setattr(ocio_setup, "STUDIO_FILE", str(tmp_path / "s.ocio"))
    env = {"OCIO": str(tmp_path / "missing.ocio")}
    state = ocio_setup.configure_ocio(env)
    assert state["configured"] and state["source"] == "built-in ACES studio config"
    assert env["OCIO"] == str(tmp_path / "missing.ocio")


def test_opencv_exr_is_forced_on():
    from radiance.config.env import configure_runtime_environment
    env = configure_runtime_environment({"OPENCV_IO_ENABLE_OPENEXR": "0"})
    assert env["OPENCV_IO_ENABLE_OPENEXR"] == "1"


def test_opencolorio_is_a_required_dependency():
    from radiance.config.dependencies import CORE_DEPENDENCIES
    assert any(d.module_name == "PyOpenColorIO" and d.required for d in CORE_DEPENDENCIES)
