"""Regression tests for the 2026-07-28 package cleanup.

Four things were true of this package and are no longer:

  * ffmpeg was invoked as the bare string "ffmpeg" from PATH, on a package that
    has shipped a bundled ffmpeg (imageio-ffmpeg) as a hard dependency since 3.0
    and never once called it. On Windows, where nothing puts ffmpeg on PATH,
    video export failed with a bare FileNotFoundError.
  * Three declared dependencies were imported nowhere; OpenImageIO, which *is*
    used, was missing from all three platform requirements files.
  * A widget helper existed in six copies that had drifted into four behaviours,
    and escapeHtml -- an XSS primitive -- existed in five.
  * Nine complete, importable node classes were never listed in any mapping
    dict, so they never appeared in ComfyUI's node menu.

These are cheap and need no torch, so they run on the lightweight CI matrix.
"""
import ast
import pathlib
import re

import pytest

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_PLATFORM_REQS = ("requirements_windows.txt", "requirements_linux.txt",
                  "requirements_mac_silicon.txt")


def _src(rel):
    return (_ROOT / rel).read_text(encoding="utf-8")


# ── 1. ffmpeg is resolved, not assumed ──────────────────────────────────────

def test_no_module_invokes_a_bare_ffmpeg_string():
    """
    Every command list started with the literal "ffmpeg". That works on a
    developer's Linux box and fails on most Windows installs.
    """
    offenders = []
    for p in _ROOT.rglob("*.py"):
        rel = p.relative_to(_ROOT).as_posix()
        if rel.startswith(("tests/", "build/", "_to_delete/", ".")) or "__pycache__" in rel:
            continue
        try:
            tree = ast.parse(p.read_text(encoding="utf-8"))
        except SyntaxError:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.List) or not node.elts:
                continue
            first = node.elts[0]
            if isinstance(first, ast.Constant) and first.value in ("ffmpeg", "ffprobe"):
                offenders.append(f"{rel}:{first.lineno}")
    assert not offenders, (
        "these build a command list starting with a bare binary name instead of "
        f"going through radiance.core.ffmpeg: {offenders}"
    )


def test_resolver_prefers_path_then_the_bundled_binary(monkeypatch, tmp_path):
    from radiance.core import ffmpeg as mod

    fake = tmp_path / "ffmpeg"
    fake.write_text("#!/bin/sh\n")
    fake.chmod(0o755)

    mod.reset_cache()
    monkeypatch.setattr(mod.shutil, "which", lambda n: str(fake) if n == "ffmpeg" else None)
    assert mod.ffmpeg_exe() == str(fake)
    assert mod.ffmpeg_available() is True

    # Nothing on PATH and no imageio-ffmpeg -> None, and require_ffmpeg explains.
    mod.reset_cache()
    monkeypatch.setattr(mod.shutil, "which", lambda n: None)
    monkeypatch.setattr(mod, "_from_imageio", lambda: None)
    assert mod.ffmpeg_exe() is None
    assert mod.ffmpeg_available() is False
    with pytest.raises(RuntimeError) as excinfo:
        mod.require_ffmpeg()
    assert "imageio-ffmpeg" in str(excinfo.value)
    assert "RADIANCE_FFMPEG" in str(excinfo.value)
    mod.reset_cache()


def test_env_override_is_honoured_and_validated(monkeypatch, tmp_path):
    from radiance.core import ffmpeg as mod

    real = tmp_path / "my-ffmpeg"
    real.write_text("#!/bin/sh\n")
    real.chmod(0o755)

    mod.reset_cache()
    monkeypatch.setenv("RADIANCE_FFMPEG", str(real))
    monkeypatch.setattr(mod.shutil, "which", lambda n: "/usr/bin/ffmpeg")
    assert mod.ffmpeg_exe() == str(real), "the override lost to PATH"

    # A bogus override must not shadow a working PATH binary.
    mod.reset_cache()
    monkeypatch.setenv("RADIANCE_FFMPEG", str(tmp_path / "nope"))
    assert mod.ffmpeg_exe() == "/usr/bin/ffmpeg"
    mod.reset_cache()


def test_ffprobe_absence_is_survivable():
    """imageio-ffmpeg ships ffmpeg but not ffprobe, so callers must degrade."""
    src = _src("core/ffmpeg.py")
    assert "def ffprobe_exe" in src
    assert "imageio-ffmpeg ships ffmpeg but *not* ffprobe" in src


# ── 2. Declared dependencies match imported ones ────────────────────────────

def _declared(text):
    names = set()
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        names.add(re.split(r"[<>=!\[;]", line)[0].strip().lower())
    return names


@pytest.mark.parametrize("rel", _PLATFORM_REQS)
def test_openimageio_is_in_every_platform_requirements_file(rel):
    """
    DPX has no Pillow plugin, so RadianceDigitalCinemaRead/Write need
    OpenImageIO. It was in requirements.txt and pyproject and in none of the
    three platform files, which are what the install docs point at.
    """
    assert "openimageio" in _declared(_src(rel)), f"{rel} is missing OpenImageIO"


@pytest.mark.parametrize("name", ["einops", "torchsde", "colour-science"])
@pytest.mark.parametrize("rel", ("requirements.txt",) + _PLATFORM_REQS)
def test_unused_dependencies_are_not_declared(rel, name):
    assert name not in _declared(_src(rel)), (
        f"{name} is back in {rel}; nothing in the package imports it"
    )


def test_platform_requirements_agree_with_each_other():
    sets = {rel: _declared(_src(rel)) for rel in _PLATFORM_REQS}
    first = next(iter(sets.values()))
    for rel, names in sets.items():
        assert names == first, f"{rel} has drifted: {names ^ first}"


def test_pyproject_runtime_deps_are_all_imported_somewhere():
    """A hard dependency nobody imports is install cost and resolver risk."""
    tomllib = pytest.importorskip("tomllib")
    deps = _declared("\n".join(
        tomllib.load(open(_ROOT / "pyproject.toml", "rb"))["project"]["dependencies"]
    ))
    # Import name differs from distribution name for these.
    aliases = {
        "opencv-python": "cv2", "pillow": "PIL", "openexr": "OpenEXR",
        "openimageio": "OpenImageIO", "opencolorio": "PyOpenColorIO",
        "imageio-ffmpeg": "imageio_ffmpeg",
    }
    sources = "\n".join(
        p.read_text(encoding="utf-8", errors="ignore")
        for p in _ROOT.rglob("*.py")
        if not p.relative_to(_ROOT).as_posix().startswith((".", "build/", "_to_delete/"))
    )
    unused = [d for d in deps
              if not re.search(rf"\b{re.escape(aliases.get(d, d))}\b", sources)]
    assert not unused, f"declared but imported nowhere: {unused}"


def test_dependency_table_does_not_advertise_a_feature_that_does_not_exist():
    src = _src("config/dependencies.py")
    assert '"colour"' not in src, (
        "the colour-science row is back; it claimed to unlock 'advanced color "
        "science' while nothing in the package imports colour"
    )
    for name in ("OpenImageIO", "PyOpenColorIO", "imageio_ffmpeg"):
        assert name in src, f"{name} gates a real feature but is not in the table"


# ── 3. One copy of each shared helper ───────────────────────────────────────

def test_escape_html_has_exactly_one_implementation():
    """Five copies of an XSS primitive means four keep the bug when one is fixed."""
    js = _ROOT / "js"
    defs = []
    for p in sorted(list(js.glob("*.js")) + list(js.glob("*.mjs"))):
        body = p.read_text(encoding="utf-8")
        if re.search(r"^\s*(export\s+)?function escapeHtml\s*\(", body, re.M):
            defs.append(p.name)
    assert defs == ["radiance_dom_utils.js"], f"escapeHtml is defined in {defs}"


def test_widget_helpers_have_exactly_one_implementation():
    js = _ROOT / "js"
    for helper in ("forceWidgetReinsert", "setWidgetVisible", "getWidget"):
        defs = [p.name for p in sorted(js.glob("*.js"))
                if re.search(rf"^\s*export function {helper}\s*\(", p.read_text(encoding="utf-8"), re.M)]
        assert defs == ["radiance_widget_utils.js"], f"{helper} exported from {defs}"

    # The six consumers must import it rather than carry a private copy.
    consumers = ["radiance_io.js", "radiance_loader.js", "radiance_resolution.js",
                 "radiance_sampler.js", "radiance_upscale.js", "radiance_vae_widgets.js"]
    for name in consumers:
        body = (js / name).read_text(encoding="utf-8")
        assert "radiance_widget_utils.js" in body, f"{name} does not import the shared helpers"
        assert not re.search(r"^function _forceWidgetReinsert\s*\(", body, re.M), \
            f"{name} still has a private _forceWidgetReinsert"


def test_each_consumer_keeps_its_own_fallback_type():
    """
    The four variants differed on the fallback type used when restoring a widget
    whose original type was never recorded. Collapsing that to one value would
    have been a silent behaviour change.
    """
    js = _ROOT / "js"
    expected = {
        "radiance_io.js": "number", "radiance_loader.js": "combo",
        "radiance_resolution.js": "INT", "radiance_sampler.js": "number",
        "radiance_upscale.js": "text", "radiance_vae_widgets.js": "number",
    }
    for name, fallback in expected.items():
        body = (js / name).read_text(encoding="utf-8")
        assert f'fallbackType: "{fallback}"' in body, f"{name} lost its {fallback!r} fallback"


def test_shared_widget_helper_is_the_union_of_the_old_variants():
    """
    Two of the four variants never suppressed draw/inputEl/element, so DOM-backed
    widgets kept painting after being hidden. The shared one does it for all six.
    """
    src = _src("js/radiance_widget_utils.js")
    for token in ("_origDraw", "inputEl.style.display", "element.style.display",
                  "_origComputeSize", "_origComputedHeight"):
        assert token in src, f"the shared helper dropped {token}"


# ── 4. Every written node reaches the menu ──────────────────────────────────

_RECOVERED = [
    "RadianceBitDepthDegrade", "RadiancePolicyGuard",
    "RadianceLUTApply", "RadianceLUTBlend",
    "RadianceDigitalCinemaRead", "RadianceDigitalCinemaWrite",
    "RadianceFlipbookGIF", "RadiancePreviewServer",
    "RadianceControlNetApply",
]


@pytest.mark.parametrize("key", _RECOVERED)
def test_recovered_nodes_are_published(key):
    """
    Nine complete node classes were written, importable and structurally valid,
    and the v3 reorganisation left them out of every mapping dict by hand.
    """
    import radiance
    if radiance._LOAD_RESULT.failures:
        pytest.skip("environment is short a runtime dependency; see the startup ERROR")
    assert key in radiance.NODE_CLASS_MAPPINGS, f"{key} is unreachable again"
    assert key in radiance.NODE_DISPLAY_NAME_MAPPINGS, f"{key} has no display name"


def test_expected_node_count_matches_what_is_published():
    import radiance
    from radiance.config.constants import EXPECTED_MIN_NODE_COUNT
    if radiance._LOAD_RESULT.failures:
        pytest.skip("environment is short a runtime dependency")
    assert len(radiance.NODE_CLASS_MAPPINGS) >= EXPECTED_MIN_NODE_COUNT
    # No second number here. This assertion used to carry its own literal 129
    # while the constant also read 129 and 131 nodes actually registered, so two
    # could have stopped registering with this test still green. The floor is
    # the constant; what this pins is that the floor is not slack.
    assert len(radiance.NODE_CLASS_MAPPINGS) == EXPECTED_MIN_NODE_COUNT, (
        "EXPECTED_MIN_NODE_COUNT in config/constants.py is not the real node "
        f"count ({len(radiance.NODE_CLASS_MAPPINGS)}); slack here is room for a "
        "node to stop registering unnoticed"
    )


def test_no_node_exists_in_source_without_reaching_the_menu():
    """Everything written is published, or is written down as withheld.

    The allowlist held three alias keys; widening the source scan past the
    retired root `nodes_*.py` layer found seventeen more finished nodes in the
    same state, none of them deliberate. All twenty ship now, the aliases as
    DEPRECATED subclasses so they load saved workflows without doubling up in
    the menu, and `nodes/aggregate.py` publishes leaf-module nodes by default
    so the gap cannot silently reopen.
    """
    from test_node_smoke import TestCoverageSummary

    assert TestCoverageSummary._KNOWN_UNREGISTERED == frozenset(), (
        "a node is being withheld from the catalog: "
        f"{sorted(TestCoverageSummary._KNOWN_UNREGISTERED)}"
    )


# ── 5. What ships ───────────────────────────────────────────────────────────

def test_package_data_does_not_ship_local_workflows():
    """
    workflows/*.rad swept up whatever the person building happened to have
    saved -- a wheel built on a working machine shipped their own shot files.
    """
    src = _src("pyproject.toml")
    block = src[src.index("[tool.setuptools.package-data]"):]
    block = block[:block.index("\n[")]
    assert '"workflows/*.rad"' not in block
    assert '"workflows/*/*.rad"' not in block
    assert '"workflows/TEMPLATES/*.rad"' in block


def test_scratch_directories_are_excluded_from_the_registry_archive():
    ignore = _src(".comfyignore")
    for entry in ("_audit_tmp/", "_to_delete/", ".ruff_cache/", ".phase3-venv/",
                  "workflows/*.rad"):
        assert entry in ignore, f"{entry} would ship to the Comfy Registry"


def test_caches_and_build_output_are_git_ignored():
    ignore = _src(".gitignore")
    for entry in ("__pycache__/", "build/", "dist/", ".ruff_cache/", ".phase3-venv/"):
        assert entry in ignore, f"{entry} is not ignored"
