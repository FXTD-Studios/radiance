"""docs/nodes is generated from the node definitions and must match them.

Regenerate after changing a node, its inputs or their tooltips:

    RADIANCE_UPDATE_DOCS=1 python -m pytest tests/test_node_reference.py
"""
from __future__ import annotations

import importlib.util
import os
import pathlib

import pytest

torch = pytest.importorskip("torch")
if not isinstance(getattr(torch, "__version__", None), str):
    pytest.skip("needs real torch: the stubbed lane under-registers nodes", allow_module_level=True)

pytestmark = pytest.mark.real_torch

_ROOT = pathlib.Path(__file__).resolve().parent.parent
_TOOL = _ROOT / "tools" / "build_node_reference.py"


def _builder():
    spec = importlib.util.spec_from_file_location("build_node_reference", _TOOL)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_node_reference_is_current():
    import radiance
    if radiance._LOAD_RESULT.failures:
        pytest.skip("environment is short a runtime dependency")
    tool = _builder()
    pages = tool.build(radiance)
    if os.environ.get("RADIANCE_UPDATE_DOCS"):
        tool.write(pages)
        pytest.skip(f"docs/nodes regenerated ({len(pages)} pages); commit them")
    on_disk = {p.name: p.read_text(encoding="utf-8") for p in tool.OUT_DIR.glob("*.md")}
    stale = sorted(set(pages) ^ set(on_disk)) + sorted(k for k in pages if k in on_disk and pages[k] != on_disk[k])
    assert not stale, (
        f"docs/nodes is out of date ({stale}). Regenerate with "
        "RADIANCE_UPDATE_DOCS=1 python -m pytest tests/test_node_reference.py and commit.")


def test_readme_section_counts_match_the_reference():
    """The README's node-map counts are typed by hand; the reference is generated."""
    import re
    readme = (_ROOT / "README.md").read_text(encoding="utf-8")
    ref = (_ROOT / "docs" / "nodes" / "README.md").read_text(encoding="utf-8")
    row = re.compile(r"^\| \[([^\]]+)\]\((?:docs/nodes/)?([a-z-]+)\.md\) \| (\d+) \|", re.M)
    in_readme = {m.group(2): int(m.group(3)) for m in row.finditer(readme)}
    in_ref = {m.group(2): int(m.group(3)) for m in row.finditer(ref)}
    assert in_readme and in_readme == in_ref, f"README node map {in_readme} != reference {in_ref}"


def test_local_links_in_the_docs_resolve():
    import re
    link = re.compile(r"\]\(([^)#\s]+)(#[^)\s]*)?\)")
    broken = []
    for md in [_ROOT / "README.md", _ROOT / "KNOWN_ISSUES.md", *sorted((_ROOT / "docs").rglob("*.md"))]:
        for m in link.finditer(md.read_text(encoding="utf-8")):
            target = m.group(1)
            if target.startswith(("http://", "https://", "mailto:")):
                continue
            if not (md.parent / target).exists():
                broken.append(f"{md.relative_to(_ROOT)} -> {target}")
    assert not broken, broken
