"""
The Workflow Library's official cards: workflows/official/{API,Local}/*.rad.

Each card is a .rad v3 (zip) holding the same graph as the workflow of the same
name in workflows/ (the one ComfyUI's Templates menu lists), a 960 x 480 cover
and the card fields the Library reads: stats.title, stats.kicker and three
stats.tags. A card whose graph drifted from its JSON would open an older
workflow from the Library than from the Templates menu.
"""
import json
import struct
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
CARDS = sorted((WORKFLOWS / "official").glob("*/*.rad"))


def _png_size(data: bytes):
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    return struct.unpack(">II", data[16:24])


def test_there_are_official_cards():
    assert {p.parent.name for p in CARDS} == {"API", "Local"}


@pytest.mark.parametrize("card", CARDS, ids=lambda p: f"{p.parent.name}/{p.stem}")
def test_card_matches_its_workflow(card):
    assert card.read_bytes()[:4] == b"RADZ", "a .rad v3 starts with RADZ, or the Library reads it as plain text"
    z = zipfile.ZipFile(card)
    names = set(z.namelist())
    assert {"manifest.json", "workflow.json", "preview.png"} <= names
    manifest = json.loads(z.read("manifest.json"))
    assert manifest.get("rad_version") == 3
    meta = manifest["metadata"]
    stats = meta["stats"]
    assert stats.get("title"), "a card needs stats.title"
    assert stats.get("kicker"), "a card needs stats.kicker"
    assert isinstance(stats.get("tags"), list) and len(stats["tags"]) == 3
    assert meta.get("category") == card.parent.name.lower()
    assert meta.get("description")
    assert _png_size(z.read("preview.png")) == (960, 480)
    source = WORKFLOWS / f"{card.stem}.json"
    assert source.is_file(), f"{card.name} has no {source.name} in workflows/"
    assert json.loads(z.read("workflow.json")) == json.loads(source.read_text(encoding="utf-8")), (
        f"{card.relative_to(ROOT)} holds an older graph than {source.name}: rebuild the card")


def test_every_api_workflow_has_a_card():
    api = {p.stem for p in WORKFLOWS.glob("api_*.json")}
    cards = {p.stem for p in (WORKFLOWS / "official" / "API").glob("*.rad")}
    assert api == cards
