"""One answer to "which OCIO config?" (OPEN_QUESTIONS P2, 4.0).

color/ocio_setup.active_config_path is the source; the Write/Read encodings,
the HDR OCIO nodes and the OCIO manager behind /radiance/ocio/* ask it. Until
4.0 they each had their own order, and two fell back to the bundled CG config
whose colour space names differ from the studio config the others used.
The ACES 2.0 reference renders stay pinned on purpose and are not covered.
"""
import os

import pytest

OCIO = pytest.importorskip("PyOpenColorIO")
if not hasattr(OCIO, "Config"):
    pytest.skip("PyOpenColorIO is a stub in this lane", allow_module_level=True)

import radiance.color.ocio_setup as setup  # noqa: E402
from radiance.color import encodings  # noqa: E402
from radiance.hdr import ocio as hdr_ocio  # noqa: E402
import radiance.radiance_ocio as manager_mod  # noqa: E402


def _names():
    return {
        "encodings": encodings.ocio_config().getName(),
        "hdr": hdr_ocio._resolve_config("").getName(),
        "manager": OCIO.Config.CreateFromFile(manager_mod.discover_ocio_config()).getName(),
    }


@pytest.fixture
def raw_config(tmp_path):
    cfg = OCIO.Config.CreateRaw()
    cfg.setName("radiance-test-raw")
    path = tmp_path / "raw.ocio"
    path.write_text(cfg.serialize(), encoding="utf-8")
    return str(path)


def test_with_no_ocio_every_consumer_gets_the_same_config(monkeypatch):
    monkeypatch.delenv("OCIO", raising=False)
    names = _names()
    assert len(set(names.values())) == 1, names
    assert next(iter(names.values())) == setup.STATE["name"]


def test_a_user_ocio_reaches_every_consumer_even_after_a_first_call(monkeypatch, raw_config):
    monkeypatch.delenv("OCIO", raising=False)
    _names()                                   # warm every cache on the default
    monkeypatch.setenv("OCIO", raw_config)
    assert set(_names().values()) == {"radiance-test-raw"}


def test_a_broken_ocio_falls_back_to_the_same_config_everywhere(monkeypatch, tmp_path):
    bad = tmp_path / "broken.ocio"
    bad.write_text("not: [a config", encoding="utf-8")
    monkeypatch.setenv("OCIO", str(bad))
    names = _names()
    assert len(set(names.values())) == 1, names
    assert setup.active_config_path() != str(bad)


def test_an_edited_config_is_read_again(monkeypatch, raw_config):
    monkeypatch.setenv("OCIO", raw_config)
    assert setup.active_config().getName() == "radiance-test-raw"
    cfg = OCIO.Config.CreateRaw()
    cfg.setName("radiance-test-edited")
    with open(raw_config, "w", encoding="utf-8") as f:
        f.write(cfg.serialize())
    st = os.stat(raw_config)
    os.utime(raw_config, (st.st_atime, st.st_mtime + 5))
    assert setup.active_config().getName() == "radiance-test-edited"


def test_the_manager_loads_a_builtin_uri():
    m = manager_mod.OCIOConfigManager.__new__(manager_mod.OCIOConfigManager)
    m.config = None
    m.config_path = None
    m.config_name = ""
    m._lut_cache = {}
    assert m.load_config("ocio://studio-config-latest") is True
    assert m.config_path == "ocio://studio-config-latest"


def test_an_explicit_path_still_wins(monkeypatch, raw_config):
    monkeypatch.delenv("OCIO", raising=False)
    assert encodings.ocio_config(raw_config).getName() == "radiance-test-raw"
    assert hdr_ocio._resolve_config(raw_config).getName() == "radiance-test-raw"
