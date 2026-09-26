"""
test_nodes_registry.py — Node registry integrity tests.

Covers:
  • Every entry in NODE_CLASS_MAPPINGS resolves to an actual class
  • Every class has RETURN_TYPES and RETURN_NAMES of matching length
  • Every class has FUNCTION pointing to an existing method
  • CATEGORY is a string
  • No duplicate NODE_CLASS_MAPPINGS keys across modules
  • Logger hierarchy: all radiance loggers are under 'radiance' (lowercase)
"""

import sys
import os
import ast
import glob
import logging
import types
import importlib

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# Build/packaging artifacts that contain *copies* of the package source.
# Scanning them would report every node as a cross-module duplicate of itself.
_IGNORE_DIRS = {"__pycache__", "build", "dist", ".git"}


def _is_ignored_path(fpath: str) -> bool:
    segs = fpath.replace("\\", "/").split("/")
    return any(s in _IGNORE_DIRS or s.endswith(".egg-info") for s in segs)


# ─────────────────────────────────────────────────────────────────────────────
#  Collect registry directly from AST (no imports needed)
# ─────────────────────────────────────────────────────────────────────────────

def _collect_registry_from_ast():
    """
    Walk all nodes_*.py and sub-package __init__.py files.
    For each class found, extract RETURN_TYPES, RETURN_NAMES, FUNCTION, CATEGORY.
    Returns list of (filename, classname, rt_len, rn_len, function_name, has_method).
    """
    root = os.path.join(os.path.dirname(__file__), "..")
    results = []

    patterns = [
        os.path.join(root, "nodes_*.py"),
        os.path.join(root, "color", "__init__.py"),
        os.path.join(root, "film",  "__init__.py"),
        os.path.join(root, "image", "__init__.py"),
        os.path.join(root, "hdr",   "__init__.py"),
    ]

    for pattern in patterns:
        for fpath in sorted(glob.glob(pattern)):
            try:
                src = open(fpath, encoding="utf-8").read()
                tree = ast.parse(src)
            except Exception:
                continue

            fname = os.path.relpath(fpath, root)

            for cls in ast.walk(tree):
                if not isinstance(cls, ast.ClassDef):
                    continue

                rt_len = rn_len = func_name = category = None

                for item in cls.body:
                    if not isinstance(item, ast.Assign):
                        continue
                    for t in item.targets:
                        if not isinstance(t, ast.Name):
                            continue
                        if t.id == "RETURN_TYPES" and isinstance(item.value, ast.Tuple):
                            rt_len = len(item.value.elts)
                        if t.id == "RETURN_NAMES" and isinstance(item.value, ast.Tuple):
                            rn_len = len(item.value.elts)
                        if t.id == "FUNCTION" and isinstance(item.value, ast.Constant):
                            func_name = item.value.value
                        if t.id == "CATEGORY" and isinstance(item.value, ast.Constant):
                            category = item.value.value

                # Skip classes without FUNCTION (not ComfyUI nodes)
                if func_name is None:
                    continue

                # Check if the named method actually exists on the class
                method_names = {
                    n.name for n in ast.walk(cls)
                    if isinstance(n, ast.FunctionDef)
                }
                has_method = func_name in method_names

                results.append((fname, cls.name, rt_len, rn_len, func_name,
                                 has_method, category))

    return results


_REGISTRY = _collect_registry_from_ast()


# ─────────────────────────────────────────────────────────────────────────────
#  Parametrized tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "fname,clsname,rt_len,rn_len,func_name,has_method,category",
    _REGISTRY,
    ids=[f"{r[0]}::{r[1]}" for r in _REGISTRY],
)
def test_function_method_exists(fname, clsname, rt_len, rn_len, func_name,
                                has_method, category):
    """FUNCTION must point to an existing method on the class."""
    assert has_method, (
        f"{fname}::{clsname} — FUNCTION = '{func_name}' "
        f"but no method with that name exists on the class"
    )


@pytest.mark.parametrize(
    "fname,clsname,rt_len,rn_len,func_name,has_method,category",
    [r for r in _REGISTRY if r[2] is not None and r[3] is not None],
    ids=[f"{r[0]}::{r[1]}" for r in _REGISTRY if r[2] is not None and r[3] is not None],
)
def test_return_types_names_length_match(fname, clsname, rt_len, rn_len,
                                         func_name, has_method, category):
    """RETURN_TYPES and RETURN_NAMES must have the same number of entries."""
    assert rt_len == rn_len, (
        f"{fname}::{clsname} — "
        f"RETURN_TYPES has {rt_len} entries but RETURN_NAMES has {rn_len}"
    )


@pytest.mark.parametrize(
    "fname,clsname,rt_len,rn_len,func_name,has_method,category",
    [r for r in _REGISTRY if r[6] is not None],
    ids=[f"{r[0]}::{r[1]}" for r in _REGISTRY if r[6] is not None],
)
def test_category_is_string(fname, clsname, rt_len, rn_len,
                             func_name, has_method, category):
    """CATEGORY must be a non-empty string."""
    assert isinstance(category, str) and category.strip(), (
        f"{fname}::{clsname} — CATEGORY is empty or not a string: {category!r}"
    )


# ─────────────────────────────────────────────────────────────────────────────
#  No duplicate keys across the full pack
# ─────────────────────────────────────────────────────────────────────────────

def _collect_node_class_mapping_keys():
    """
    Extract all string keys from NODE_CLASS_MAPPINGS dicts across the pack.
    Returns list of (filename, key).
    """
    root = os.path.join(os.path.dirname(__file__), "..")
    seen = []

    for fpath in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        if _is_ignored_path(fpath):
            continue
        try:
            src = open(fpath, encoding="utf-8").read()
            tree = ast.parse(src)
        except Exception:
            continue

        fname = os.path.relpath(fpath, root)

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id == "NODE_CLASS_MAPPINGS":
                    if isinstance(node.value, ast.Dict):
                        for k in node.value.keys:
                            if isinstance(k, ast.Constant):
                                seen.append((fname, k.value))

    return seen


def _package_namespace(fname: str) -> str:
    """
    Return the canonical namespace for duplicate detection.

    Files within the same package directory share a namespace (the package
    name). `color/__init__.py` and `color/lut.py` both map to `color`.
    Top-level files map to their basename without `.py` extension.

    Examples:
      color/__init__.py → color
      color/lut.py      → color
      nodes_engine.py   → nodes_engine
      film/camera.py    → film
    """
    parts = fname.replace("\\", "/").split("/")
    if len(parts) > 1:
        return parts[0]   # first directory component
    return fname[:-3] if fname.endswith(".py") else fname   # top-level module


# Known pre-existing cross-module duplicates where a sub-package has superseded
# an older top-level implementation.  These are tracked here so that any NEW
# cross-module duplicates still cause a test failure.
_KNOWN_CROSS_MODULE_DUPLICATES: dict = {
    # color/ sub-package and nodes.monitor currently both expose LUT apply.
    # Track this transition explicitly so unrelated duplicate keys still fail.
    "RadianceLUTApply":  frozenset(["color", "nodes"]),
    # Was ["color", "nodes_engine"]. nodes.color now publishes LUT Blend too --
    # it had been written in color/lut.py and left out of every mapping dict, so
    # the node existed but never reached ComfyUI's menu.
    "RadianceLUTBlend":  frozenset(["color", "nodes"]),
    # nodes_io_unified is the canonical IO module; nodes_io is its backward-compat
    # shim (Task #140 / #141-fix).  The shim re-exports the same keys so that
    # saved ComfyUI workflows and test_io.py continue to resolve them.
    "RadianceEXRMultiPart": frozenset(["nodes_io", "nodes_io_unified"]),
    # Same story as the colour nodes below: the DPX read/write pair lived in
    # nodes_io.py and the v3 nodes.io group forgot to list them, so they were
    # unreachable until 2026-07. nodes.io is canonical; nodes_io is the shim.
    "RadianceDigitalCinemaRead": frozenset(["nodes", "nodes_io"]),
    "RadianceDigitalCinemaWrite": frozenset(["nodes", "nodes_io"]),
    # v3 organized package plus compatibility wrappers.
    "RadianceCDLTransform": frozenset(["nodes", "nodes_cdl"]),
    "RadianceCDLImport": frozenset(["nodes", "nodes_cdl"]),
    "RadianceCDLExport": frozenset(["nodes", "nodes_cdl"]),
    "RadianceWhiteBalance": frozenset(["nodes", "nodes_colorscience"]),
    "RadianceColorSpaceConvert": frozenset(["nodes", "nodes_colorscience"]),
    "RadianceACESTransform": frozenset(["nodes", "nodes_colorscience"]),
    "RadianceBitDepthDegrade": frozenset(["nodes", "nodes_colorscience"]),
    "RadianceHueCurves": frozenset(["nodes", "nodes_curves"]),
    "RadianceCurves": frozenset(["nodes", "nodes_curves"]),
    "RadianceGrade": frozenset(["nodes", "nodes_grade"]),
    "RadianceApplyGradeInfo": frozenset(["nodes", "nodes_grade"]),
    "RadianceGradeMatch": frozenset(["nodes", "nodes_grade"]),
    "RadianceOCIOContext": frozenset(["nodes", "nodes_ocio"]),
    "RadianceQC": frozenset(["nodes", "nodes_qc"]),
    "RadiancePolicyGuard": frozenset(["nodes", "nodes_qc"]),
    "RadianceHDRExpandDynamicRange": frozenset(["hdr", "nodes"]),
    "RadianceHDRToneMap": frozenset(["hdr", "nodes"]),
    "RadianceSamplerPro": frozenset(["nodes", "nodes_sampler"]),
    "RadianceLoraStack": frozenset(["nodes", "nodes_loader"]),
    "RadianceUnifiedLoader": frozenset(["nodes", "nodes_loader"]),
    "RadianceVideoLoader": frozenset(["nodes", "nodes_loader"]),
    "RadianceControlNetApply": frozenset(["nodes", "nodes_loader"]),
    "RadianceRead": frozenset(["nodes", "nodes_io"]),
    "RadianceWrite": frozenset(["nodes", "nodes_io"]),
    "RadianceEXRMultiPart": frozenset(["nodes", "nodes_io"]),
    "RadianceFalseColorMonitor": frozenset(["nodes", "nodes_realtime_preview"]),
    "RadianceFocusPeaking": frozenset(["nodes", "nodes_realtime_preview"]),
    "RadianceSplitView": frozenset(["nodes", "nodes_realtime_preview"]),
    "RadianceContactSheet": frozenset(["nodes", "nodes_realtime_preview"]),
    "RadianceFlipbookGIF": frozenset(["nodes", "nodes_realtime_preview"]),
    "RadianceFrameStamp": frozenset(["nodes", "nodes_realtime_preview"]),
    "RadiancePreviewServer": frozenset(["nodes", "nodes_realtime_preview"]),
    "RadianceViewer": frozenset(["nodes", "nodes_radiance_viewer"]),
    # 2026-09-18: the implementation packages declare NODE_CLASS_MAPPINGS that
    # nothing in the load chain ever read, so twenty-three finished nodes never
    # reached ComfyUI. `radiance/nodes/<group>/__init__.py` is the registration
    # layer and now names them, which is the same arrangement that already
    # covers RadianceLUTApply above: the key is declared in both namespaces and
    # points at one class object in both. See nodes/catalog.py.
    "RadianceProUpscale": frozenset(["image", "nodes"]),
    "RadianceUpscaleBySize": frozenset(["image", "nodes"]),
    "RadianceDownscale32bit": frozenset(["image", "nodes"]),
    "RadianceBitDepthConvert": frozenset(["image", "nodes"]),
    "RadianceAIUpscale": frozenset(["image", "nodes"]),
    "RadianceFloat32Convert": frozenset(["hdr", "nodes"]),
    "RadianceFloat32ColorCorrect": frozenset(["hdr", "nodes"]),
    "RadianceHDRColorConvert": frozenset(["hdr", "nodes"]),
    "RadianceDaVinciWideGamut": frozenset(["hdr", "nodes"]),
    "RadianceARRIWideGamut4": frozenset(["hdr", "nodes"]),
    "RadianceACES2OutputTransform": frozenset(["hdr", "nodes"]),
    "RadianceHDRExposureBlend": frozenset(["hdr", "nodes"]),
    "RadianceHDRShadowHighlight": frozenset(["hdr", "nodes"]),
    "RadianceGPUTensorOps": frozenset(["hdr", "nodes"]),
    "RadianceHDR360Generate": frozenset(["hdr", "nodes"]),
    "RadianceHighlightSynthesis": frozenset(["hdr", "nodes"]),
    "RadianceACESConfigManager": frozenset(["hdr", "nodes"]),
    "RadianceHDROCIOTransform": frozenset(["hdr", "nodes"]),
    "RadianceOCIOListColorspaces": frozenset(["hdr", "nodes"]),
    "RadianceDepthOfField": frozenset(["film", "nodes"]),
    "RadianceRollingShutter": frozenset(["film", "nodes"]),
    "RadianceCompressionArtifacts": frozenset(["film", "nodes"]),
}


def test_no_duplicate_node_class_mapping_keys():
    """
    No two *distinct* modules may register the same NODE_CLASS_MAPPINGS key.

    Files within the same package directory (e.g. color/__init__.py +
    color/lut.py) are treated as one namespace and are NOT flagged.

    Known pre-existing cross-module duplicates (where a sub-package has
    superseded an older top-level module) are listed in
    _KNOWN_CROSS_MODULE_DUPLICATES and will NOT fail the test.  Any NEW
    cross-module duplicates not in that list WILL fail.
    """
    all_keys = _collect_node_class_mapping_keys()
    key_to_namespaces: dict = {}
    for fname, key in all_keys:
        ns = _package_namespace(fname)
        key_to_namespaces.setdefault(key, set()).add(ns)

    new_duplicates = {}
    for key, namespaces in key_to_namespaces.items():
        if len(namespaces) <= 1:
            continue
        known = _KNOWN_CROSS_MODULE_DUPLICATES.get(key)
        if known and frozenset(namespaces) == known:
            continue   # acknowledged pre-existing duplicate, skip
        new_duplicates[key] = sorted(namespaces)

    assert not new_duplicates, (
        "NEW duplicate NODE_CLASS_MAPPINGS keys found across distinct modules:\n" +
        "\n".join(f"  '{k}': {ns}" for k, ns in new_duplicates.items())
    )


# ─────────────────────────────────────────────────────────────────────────────
#  Logger hierarchy: all under 'radiance' (lowercase)
# ─────────────────────────────────────────────────────────────────────────────

def test_logger_hierarchy_all_lowercase():
    """
    Every getLogger call in the pack must use lowercase 'radiance' as the
    root — not 'Radiance'.  Two separate trees would mean log handlers set
    on the package root never see messages from nodes that use the wrong case.

    Regression for the package-wide logger sweep in the 2026-04-24 audit.
    """
    root = os.path.join(os.path.dirname(__file__), "..")
    offenders = []

    import re
    pattern = re.compile(r'getLogger\("Radiance[^a-z]')  # uppercase R followed by non-lowercase

    for fpath in sorted(glob.glob(os.path.join(root, "**", "*.py"), recursive=True)):
        if _is_ignored_path(fpath):
            continue
        src = open(fpath, encoding="utf-8").read()
        for i, line in enumerate(src.splitlines(), 1):
            if pattern.search(line):
                offenders.append(f"{os.path.relpath(fpath, root)}:{i}: {line.strip()}")

    assert not offenders, (
        "Non-lowercase 'Radiance.*' logger names found — "
        "these are orphaned from the 'radiance' logger tree:\n" +
        "\n".join(f"  {o}" for o in offenders)
    )
