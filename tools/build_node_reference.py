#!/usr/bin/env python3
"""Build the node reference (docs/nodes/*.md) from the node definitions.

Everything on those pages comes from the loaded classes: the menu section,
display name, DESCRIPTION, each input's type, default, range or choices and
tooltip, and each output's name, type and tooltip. Nothing is written by
hand, so the reference cannot drift from the code; tests/test_node_reference.py
fails when the committed pages are stale.

Regenerate after changing a node:

    RADIANCE_UPDATE_DOCS=1 python -m pytest tests/test_node_reference.py

or, from a ComfyUI install with Radiance in custom_nodes:

    python tools/build_node_reference.py
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "nodes"
MAX_CHOICES = 10

# Choices read from the user's folders (models, LUTs, fonts...) differ per
# machine; the reference names the folder instead of listing one machine's files.
_FILE_LIKE = re.compile(r"\.(safetensors|ckpt|pt|pth|bin|cube|3dl|csp|ocio|json|ttf|otf|png|jpg|jpeg|exr|mp4|mov|wav|rad)$", re.I)


# Choice lists ComfyUI itself supplies. They change with the ComfyUI version,
# so the reference names them instead of copying one version's list.
_COMFY_LISTS = {
    "sampler": "ComfyUI's samplers",
    "sampler_name": "ComfyUI's samplers",
    "scheduler": "ComfyUI's schedulers",
    "control_type": "`auto`, or one of ComfyUI's Union ControlNet types",
}


def _is_list(kind: Any) -> bool:
    return isinstance(kind, (list, tuple))


def _slug(section: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", section.lower()).strip("-")


def _cell(text: Any) -> str:
    s = "" if text is None else str(text)
    s = s.replace("\r", " ").replace("\n", " ").replace("|", "\\|").strip()
    return re.sub(r"\s{2,}", " ", s)


def _portable(value: str) -> str:
    """Defaults built from the user's home folder read as ~ so the page is the same on every machine."""
    home = str(Path.home())
    for h in {home, home.replace("\\", "/")}:
        if h and h not in ("/", "") and value.startswith(h):
            return "~" + value[len(h):].replace("\\", "/")
    return value


def _fmt_number(v: Any) -> str:
    if isinstance(v, bool):
        return "on" if v else "off"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _describe_input(entry: Any, name: str = "") -> Tuple[str, str, str, str]:
    """(type, default, range or choices, tooltip) for one INPUT_TYPES entry."""
    kind = entry[0] if isinstance(entry, (list, tuple)) and entry else entry
    opts = entry[1] if isinstance(entry, (list, tuple)) and len(entry) > 1 and isinstance(entry[1], dict) else {}
    tooltip = opts.get("tooltip", "")
    default = opts.get("default", "")

    comfy_list = _COMFY_LISTS.get(name)
    if comfy_list and (not _is_list(kind) or name == "control_type"
                       or (_is_list(kind) and "euler" in [str(c) for c in kind])
                       or (_is_list(kind) and "karras" in [str(c) for c in kind])):
        dflt = f"`{default}`" if isinstance(default, str) and default and not _FILE_LIKE.search(default) else ""
        return "choice", dflt, comfy_list, tooltip

    if _is_list(kind):
        # Model, LUT and font files differ per machine: list only the fixed
        # entries, and say where the rest come from.
        allc = [str(c) for c in kind]
        choices = [c for c in allc if not _FILE_LIKE.search(c)]
        if isinstance(default, str) and _FILE_LIKE.search(default):
            default = ""
        if not choices:
            rng = "files found in the matching models or input folder"
        else:
            shown = choices[:MAX_CHOICES]
            rng = ", ".join(f"`{c}`" for c in shown)
            if len(choices) > MAX_CHOICES:
                rng += f", and {len(choices) - MAX_CHOICES} more"
        return "choice", (f"`{default}`" if default != "" else ""), rng, tooltip

    kind = str(kind)
    rng = ""
    if kind in ("INT", "FLOAT"):
        lo, hi = opts.get("min"), opts.get("max")
        if lo is not None and hi is not None:
            rng = f"{_fmt_number(lo)} to {_fmt_number(hi)}"
            if opts.get("step") is not None:
                rng += f", step {_fmt_number(opts['step'])}"
    elif kind == "STRING" and opts.get("multiline"):
        rng = "multi-line text"
    dflt = ""
    if default != "" and default is not None:
        if kind == "STRING":
            dflt = f"`{_portable(str(default))}`" if default else "empty"
        else:
            dflt = _fmt_number(default)
    return kind.lower() if kind in ("INT", "FLOAT", "STRING", "BOOLEAN") else kind, dflt, rng, tooltip


def _node_section(cls: Any) -> str:
    cat = str(getattr(cls, "CATEGORY", "") or "")
    return cat.rsplit("/", 1)[-1] if "/" in cat else (cat or "Other")


def _render_node(key: str, cls: Any, name: str) -> List[str]:
    lines = [f"## {name}", "", f"`{key}`", ""]
    desc = str(getattr(cls, "DESCRIPTION", "") or "").strip()
    if desc:
        lines += [_cell(desc), ""]

    spec = cls.INPUT_TYPES()
    rows = []
    for section in ("required", "optional"):
        for inp, entry in (spec.get(section) or {}).items():
            kind, dflt, rng, tip = _describe_input(entry, inp)
            label = f"`{inp}`" + (" (optional)" if section == "optional" else "")
            rows.append(f"| {label} | {_cell(kind)} | {_cell(dflt)} | {_cell(rng)} | {_cell(tip)} |")
    if rows:
        lines += ["**Inputs**", "", "| Input | Type | Default | Range or choices | What it does |",
                  "| :--- | :--- | :--- | :--- | :--- |", *rows, ""]

    types = list(getattr(cls, "RETURN_TYPES", ()) or ())
    if types:
        names = list(getattr(cls, "RETURN_NAMES", ()) or ()) or [str(t).lower() for t in types]
        tips = list(getattr(cls, "OUTPUT_TOOLTIPS", ()) or ())
        lines += ["**Outputs**", ""]
        if any(tips):
            lines += ["| Output | Type | What it is |", "| :--- | :--- | :--- |"]
            for i, t in enumerate(types):
                n = names[i] if i < len(names) else str(t).lower()
                tip = tips[i] if i < len(tips) else ""
                lines.append(f"| `{n}` | {_cell(t)} | {_cell(tip)} |")
        else:
            lines += ["| Output | Type |", "| :--- | :--- |"]
            for i, t in enumerate(types):
                n = names[i] if i < len(names) else str(t).lower()
                lines.append(f"| `{n}` | {_cell(t)} |")
        lines.append("")
    elif getattr(cls, "OUTPUT_NODE", False):
        lines += ["Output node: writes or displays, no graph outputs.", ""]
    return lines


def render(class_mappings: Mapping[str, Any], display_names: Mapping[str, str],
           section_blurbs: Mapping[str, str], version: str) -> Dict[str, str]:
    """Return {relative path: markdown} for every page."""
    by_section: Dict[str, List[Tuple[str, str, Any]]] = {}
    retired: List[Tuple[str, str]] = []
    for key, cls in class_mappings.items():
        name = str(display_names.get(key, key))
        if getattr(cls, "DEPRECATED", False):
            retired.append((name, key))
            continue
        by_section.setdefault(_node_section(cls), []).append((name, key, cls))

    order = [s for s in section_blurbs if s in by_section] + sorted(s for s in by_section if s not in section_blurbs)
    pages: Dict[str, str] = {}
    header = "<!-- Generated by tools/build_node_reference.py from the node definitions. Do not edit by hand. -->"

    index = [header, "", f"# Radiance {version} node reference", "",
             "Every node in the menu, by section: what it does, every input with its type, default, "
             "range and meaning, and every output. The same text appears on hover inside ComfyUI.", "",
             "| Section | Nodes | What is in it |", "| :--- | ---: | :--- |"]
    for sec in order:
        index.append(f"| [{sec}]({_slug(sec)}.md) | {len(by_section[sec])} | {_cell(section_blurbs.get(sec, ''))} |")
    total = sum(len(v) for v in by_section.values())
    index += ["", f"{total} nodes in the menu."]
    if retired:
        index += ["", "## Retired nodes", "",
                  "Hidden from the menu. They still load so saved graphs open; the changelog says what replaces each.", ""]
        index += [f"- {name} (`{key}`)" for name, key in sorted(retired)]
    pages["README.md"] = "\n".join(index) + "\n"

    for sec in order:
        nodes = sorted(by_section[sec], key=lambda t: (t[0].lower(), t[1]))
        body = [header, "", f"# {sec} nodes", ""]
        if section_blurbs.get(sec):
            body += [section_blurbs[sec], ""]
        body += [f"{len(nodes)} nodes. [All sections](README.md)", ""]
        body += [f"- [{name}](#{_anchor(name)})" for name, _k, _c in nodes]
        body.append("")
        for name, key, cls in nodes:
            body += _render_node(key, cls, name)
        pages[f"{_slug(sec)}.md"] = "\n".join(body).rstrip() + "\n"
    return pages


def _anchor(title: str) -> str:
    a = title.strip().lower()
    a = re.sub(r"[^\w\- ]", "", a)
    return a.replace(" ", "-")


def build(radiance_module: Any) -> Dict[str, str]:
    from radiance.config.constants import VERSION
    from radiance.nodes.branding import MENU_STRUCTURE
    return render(radiance_module.NODE_CLASS_MAPPINGS, radiance_module.NODE_DISPLAY_NAME_MAPPINGS,
                  MENU_STRUCTURE, VERSION)


def write(pages: Mapping[str, str], out_dir: Path = OUT_DIR) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    keep = set(pages)
    for old in out_dir.glob("*.md"):
        if old.name not in keep:
            old.unlink()
    for rel, text in pages.items():
        (out_dir / rel).write_text(text, encoding="utf-8", newline="\n")


def main(argv: Iterable[str] = ()) -> int:
    sys.path.insert(0, str(ROOT.parent))
    import radiance  # noqa: PLC0415
    pages = build(radiance)
    write(pages)
    print(f"wrote {len(pages)} pages to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
