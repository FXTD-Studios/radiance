"""
One colour code (js/radiance_palette.js) across the shipped workflows.

Group colours come from ComfyUI's own group palette, so a workflow looks the
same with or without Radiance, and no node carries a theme colour saved by an
older Radiance (those painted every node dark outside Radiance).
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PALETTE = open(os.path.join(ROOT, "js", "radiance_palette.js"), encoding="utf-8").read()
GROUP_COLOURS = {c.lower() for c in re.findall(r'group: "(#[0-9a-fA-F]{3,6})"', PALETTE)}
THEME_HEADERS = {c.lower() for c in re.findall(r'"(#[0-9a-fA-F]{3,6})"', PALETTE[PALETTE.index("OLD_THEME_HEADERS"):PALETTE.index("OLD_THEME_BODIES")])}
WORKFLOWS = sorted(glob.glob(os.path.join(ROOT, "workflows", "*.json")))


def test_palette_is_comfyui_native_for_the_main_families():
    assert {"#3f789e", "#b58b2a", "#8a8", "#a1309b", "#444"} <= GROUP_COLOURS


def test_shipped_groups_use_the_palette():
    bad = []
    for p in WORKFLOWS:
        for g in json.load(open(p, encoding="utf-8")).get("groups", []):
            if str(g.get("color", "")).lower() not in GROUP_COLOURS:
                bad.append((os.path.basename(p), g.get("title"), g.get("color")))
    assert not bad, bad


def test_no_shipped_node_carries_a_theme_colour():
    bad = []
    for p in WORKFLOWS:
        for n in json.load(open(p, encoding="utf-8")).get("nodes", []):
            if str(n.get("color", "")).lower() in THEME_HEADERS or str(n.get("bgcolor", "")).lower() == "#141419":
                bad.append((os.path.basename(p), n.get("type")))
    assert not bad, bad[:10]
