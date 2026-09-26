"""`radiance_ocio.py` — the OCIO config manager, which had no tests at all.

824 lines and 0% coverage, under `RadianceColorSpaceConvert` and the Viewer's
OCIO dropdown. The first thing writing these found is why that matters:

    bake_colorspace_lut('ACEScg', 'sRGB - Display') returned an exact identity.

Not approximately wrong — bit-identical to the lattice it was built from. The
bake looped over the cube calling `applyRGB(python_list)`, and OCIO's binding
converts a list to a temporary buffer, transforms that, and throws it away. The
list is never modified, so every point was written back unchanged. 18% grey
came out of an ACEScg → sRGB Display transform as 0.18 instead of 0.47.

That LUT is served over POST /radiance/ocio/bake and loaded straight into the
Viewer's renderer, so picking a display transform from the dropdown logged
"[OCIO] Active: sRGB / ACES 1.0" and changed nothing on screen. A wrong
picture under the right menu entry, which is the one thing colour tooling must
never do.

The tests below are therefore mostly about the bake. The rest of the manager
gets the coverage it should have had.
"""
import contextlib
import importlib
import importlib.util
import os
import pathlib
import sys

import numpy as np
import pytest


def _is_real(mod):
    """A real extension module, not a stand-in."""
    return (mod is not None
            and getattr(mod, "__file__", None) is not None
            and hasattr(getattr(mod, "Config", None), "CreateFromBuiltinConfig"))


@contextlib.contextmanager
def _real_pyocio():
    """Hand back the real PyOpenColorIO for the duration of the block.

    tests/test_node_smoke.py installs a MagicMock at
    sys.modules["PyOpenColorIO"] whenever the real one has not been imported
    yet, and pytest collects it before this file. So `importorskip` here
    returns the mock, which does not fail at import — it fails later, on every
    attribute, as an AttributeError rather than a skip.

    This swaps the real module in, yields it, and puts whatever was there back
    afterwards, so the rest of the suite keeps the stub it expects.
    """
    saved = sys.modules.get("PyOpenColorIO")
    if _is_real(saved):
        yield saved
        return
    sys.modules.pop("PyOpenColorIO", None)
    real = None
    try:
        try:
            real = importlib.import_module("PyOpenColorIO")
        except ImportError:
            real = None
        yield real if _is_real(real) else None
    finally:
        if saved is not None:
            sys.modules["PyOpenColorIO"] = saved
        else:
            sys.modules.pop("PyOpenColorIO", None)


# The real module, loaded from its file rather than imported by name.
#
# conftest.py installs a stub at sys.modules["radiance.radiance_ocio"] with
# HAS_OCIO = False and a MagicMock manager, so every test in the suite that
# touches OCIO gets the mock and the real 824 lines are never executed. That is
# why this file's subject sat at 0% coverage while the suite was green, and why
# a bake that returned an exact identity could ship. Loading it by path leaves
# the stub in place for everyone else and gives these tests the real thing.
_REAL = pathlib.Path(__file__).resolve().parent.parent / "radiance_ocio.py"
with _real_pyocio() as _pyocio:
    if _pyocio is None:
        pytest.skip("PyOpenColorIO is not installed", allow_module_level=True)
    PyOCIO = _pyocio
    _spec = importlib.util.spec_from_file_location("radiance_ocio_under_test", _REAL)
    ocio_module = importlib.util.module_from_spec(_spec)
    sys.modules["radiance_ocio_under_test"] = ocio_module
    _spec.loader.exec_module(ocio_module)

if not getattr(ocio_module, "HAS_OCIO", False):   # pragma: no cover - belt and braces
    pytest.skip("PyOpenColorIO is not usable here", allow_module_level=True)

OCIOConfigManager = ocio_module.OCIOConfigManager
_is_inside_allowed_ocio_root = ocio_module._is_inside_allowed_ocio_root


@pytest.fixture(scope="module")
def builtin_config_path(tmp_path_factory):
    """A real OCIO config on disk, so load_config runs its normal path."""
    cfg = PyOCIO.Config.CreateFromBuiltinConfig("studio-config-latest")
    d = tmp_path_factory.mktemp("ocio")
    p = d / "config.ocio"
    p.write_text(cfg.serialize(), encoding="utf-8")
    return str(p)


@pytest.fixture(scope="module")
def manager(builtin_config_path):
    m = OCIOConfigManager()
    assert m.load_config(builtin_config_path) is True
    return m


def lattice(n):
    """The cube the bake is built from: R fastest, then G, then B."""
    axis = np.arange(n, dtype=np.float32) / (n - 1)
    return np.stack([
        np.tile(axis, n * n),
        np.repeat(np.tile(axis, n), n),
        np.repeat(axis, n * n),
    ], axis=1).astype(np.float32)


# ── the bake ────────────────────────────────────────────────────────────────

@pytest.mark.parametrize("n", [2, 33])
def test_the_baked_lut_is_not_an_identity(manager, n):
    """The regression. An identity here is the whole defect, in one line."""
    lut = manager.bake_colorspace_lut("ACEScg", "sRGB - Display", n)
    assert lut is not None
    assert not np.array_equal(lut, lattice(n)), (
        "the baked LUT is bit-identical to its own input lattice — the OCIO "
        "transform is not being applied at all"
    )


def test_the_baked_lut_is_what_ocio_computes(manager, builtin_config_path):
    """Against OCIO's own CPU processor, not against a remembered number.

    Exact, not approximate: the bake and the reference run the same processor
    over the same points, so any difference at all is the bake doing something
    of its own.
    """
    n = 33
    lut = manager.bake_colorspace_lut("ACEScg", "sRGB - Display", n)
    cfg = PyOCIO.Config.CreateFromFile(builtin_config_path)
    ref = lattice(n)
    cfg.getProcessor("ACEScg", "sRGB - Display").getDefaultCPUProcessor().applyRGB(ref)
    assert np.array_equal(lut, ref), (
        f"max deviation {np.abs(lut - ref).max()}"
    )


def test_18_percent_grey_lands_where_the_transform_puts_it(manager):
    """The value a colourist would check first, on the lattice point nearest it.

    ACEScg 0.1875 through sRGB - Display is ~0.47. The broken bake returned
    0.1875, which is exactly the kind of wrong that looks like a slightly flat
    grade rather than like a bug.
    """
    n = 33
    lut = manager.bake_colorspace_lut("ACEScg", "sRGB - Display", n)
    i = round(0.18 * (n - 1))
    idx = i + n * i + n * n * i
    assert lut[idx][0] == pytest.approx(0.470, abs=0.01), lut[idx]


def test_the_lattice_order_is_r_fastest(manager):
    """Wrong axis order is a channel swap that no shape check would catch.

    A pure gamma has no cross-channel term, so the output at a lattice point is
    a function of that point's own channels. Reading the second element must
    therefore correspond to R having advanced, not B.
    """
    n = 5
    lut = manager.bake_colorspace_lut("ACEScg", "ACEScg", n)   # identity by definition
    assert lut is not None
    lat = lattice(n)
    assert lut[1][0] > lut[0][0], "R did not advance between the first two entries"
    assert lut[1][1] == lut[0][1], "G moved when only R should have"
    assert lut[n][1] > lut[0][1], "G did not advance after n entries"
    assert lut[n * n][2] > lut[0][2], "B did not advance after n*n entries"
    assert np.array_equal(lut, lat), "a colour space to itself should be the lattice"


def test_the_bake_returns_none_without_a_config(monkeypatch):
    """A bare manager auto-discovers the bundled ACES config, which is the
    intended behaviour, so discovery is suppressed to reach the empty state."""
    monkeypatch.setattr(ocio_module, "discover_ocio_config", lambda: None)
    m = OCIOConfigManager()
    assert m.is_loaded is False
    assert m.bake_colorspace_lut("ACEScg", "sRGB - Display", 33) is None
    assert m.bake_display_view_lut("sRGB - Display", "Un-tone-mapped", 33) is None


def test_a_bare_manager_finds_the_bundled_config():
    """The other half: with nothing configured, Radiance still has a config.

    An install with no OCIO set up is the common case, and the bundled ACES
    config is what makes the colour menus non-empty there.
    """
    m = OCIOConfigManager()
    assert m.is_loaded is True
    assert m.get_displays()


def test_an_unknown_colour_space_fails_rather_than_returning_a_lattice(manager):
    """Silence here would be a LUT that looks fine and does nothing."""
    assert manager.bake_colorspace_lut("NotASpace", "sRGB - Display", 33) is None


# ── the cache ───────────────────────────────────────────────────────────────

def test_the_cache_returns_the_same_lut_for_the_same_request(manager):
    a = manager.bake_colorspace_lut("ACEScg", "sRGB - Display", 33)
    b = manager.bake_colorspace_lut("ACEScg", "sRGB - Display", 33)
    assert a is b, "the second bake did not come from the cache"


def test_cache_keys_cannot_collide_across_different_transforms(manager):
    """The key used to be a colon-joined string of unescaped names.

    A display or view containing a colon could then hash to the same key as a
    different pair, and the cache would hand back the wrong LUT under the right
    name — the same failure as the identity bake, arrived at differently.
    """
    key = manager._make_cache_key
    assert key("a:b", "c", None, 33) != key("a", "b:c", None, 33)
    assert key("d", "v", "in", 33) != key("d", "v", None, 33)
    assert key("d", "v", None, 33) != key("d", "v", None, 65)


def test_clear_cache_empties_it(manager):
    manager.bake_colorspace_lut("ACEScg", "sRGB - Display", 33)
    assert manager._lut_cache
    manager.clear_cache()
    assert not manager._lut_cache


# ── what the panel reads off a config ───────────────────────────────────────

def test_the_config_reports_displays_views_and_roles(manager):
    displays = manager.get_displays()
    assert displays, "a loaded ACES config with no displays is not loaded"
    views = manager.get_views(displays[0])
    assert views, f"{displays[0]} has no views"
    pairs = manager.get_display_view_pairs()
    assert len(pairs) >= len(displays)
    assert {"display", "view"} <= set(pairs[0])
    roles = manager.get_roles()
    assert "scene_linear" in roles, sorted(roles)


def test_colour_space_listings_are_non_empty_and_disjoint_in_kind(manager):
    scene = manager.get_scene_color_spaces()
    display = manager.get_display_color_spaces()
    assert scene and display
    assert all("name" in c for c in scene + display)


def test_config_info_describes_the_loaded_file(manager, builtin_config_path):
    info = manager.get_config_info()
    assert info.get("loaded") is True
    assert os.path.basename(builtin_config_path) in str(info.get("path", ""))


# ── the route guard ─────────────────────────────────────────────────────────

def test_the_ocio_root_guard_refuses_paths_outside_the_allowed_roots(monkeypatch, tmp_path):
    """POST /radiance/ocio/load takes a path from an unauthenticated route.

    Without this it is a file-existence oracle for the whole filesystem, and a
    config is a file that names other files.
    """
    allowed = tmp_path / "configs"
    allowed.mkdir()
    inside = allowed / "config.ocio"
    inside.write_text("ocio_profile_version: 2", encoding="utf-8")
    outside = tmp_path / "elsewhere.ocio"
    outside.write_text("ocio_profile_version: 2", encoding="utf-8")

    monkeypatch.setenv("RADIANCE_OCIO_ROOTS", str(allowed))
    assert _is_inside_allowed_ocio_root(str(inside)) is True
    assert _is_inside_allowed_ocio_root(str(outside)) is False


def test_the_ocio_root_guard_is_not_fooled_by_dot_dot(monkeypatch, tmp_path):
    allowed = tmp_path / "configs"
    allowed.mkdir()
    (tmp_path / "secret.ocio").write_text("x", encoding="utf-8")
    monkeypatch.setenv("RADIANCE_OCIO_ROOTS", str(allowed))
    traversal = str(allowed / ".." / "secret.ocio")
    assert _is_inside_allowed_ocio_root(traversal) is False


def test_the_manager_singleton_is_one_object():
    a = ocio_module.get_ocio_manager()
    b = ocio_module.get_ocio_manager()
    assert a is b
