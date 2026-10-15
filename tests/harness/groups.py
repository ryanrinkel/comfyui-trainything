"""Draw a titled, coloured group box around the TrainyThing part of a canvas
workflow: every TrainyThing node plus the nodes wired straight after them (the
Save Image and Preview Any that show the verdicts). make_examples.py calls it
on each example; it also runs on its own:

    python tests/harness/groups.py examples/gate.json [more.json ...]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Optional

TITLE = "TrainyThing: review on your phone"
COLOR = "#6c8cf5"  # the site's accent; groups draw it translucent
PREFIX = "TrainyThing"
NODE_TITLE_H = 30   # LiteGraph draws a node's title above node.pos
GROUP_HEAD_H = 44   # room for the group's own title
PAD = 20


def members(wf: dict, roots: Optional[Iterable[int]] = None) -> set:
    ids = set(roots) if roots is not None else {n["id"] for n in wf["nodes"] if n["type"].startswith(PREFIX)}
    for link in wf.get("links", []):  # [id, from_node, from_slot, to_node, to_slot, type]
        if link[1] in ids:
            ids.add(link[3])
    return ids


def add_group(wf: dict, title: str = TITLE, color: str = COLOR, roots: Optional[Iterable[int]] = None) -> dict:
    sel = [n for n in wf["nodes"] if n["id"] in members(wf, roots)]
    if not sel:
        return wf
    x0 = min(n["pos"][0] for n in sel) - PAD
    y0 = min(n["pos"][1] for n in sel) - NODE_TITLE_H - GROUP_HEAD_H
    x1 = max(n["pos"][0] + n["size"][0] for n in sel) + PAD
    y1 = max(n["pos"][1] + n["size"][1] for n in sel) + PAD
    groups = [g for g in wf.get("groups", []) if g.get("title") != title]
    groups.append({"id": len(groups) + 1, "title": title, "bounding": [x0, y0, x1 - x0, y1 - y0],
                   "color": color, "font_size": 24, "flags": {}})
    wf["groups"] = groups
    return wf


def main(paths: list) -> None:
    for p in map(Path, paths):
        wf = json.loads(p.read_text())
        add_group(wf)
        p.write_text(json.dumps(wf, indent=1) + "\n")
        g = wf["groups"][-1]
        print(f"{p.name}: group {g['bounding']} around {len(members(wf))} nodes")


if __name__ == "__main__":
    main(sys.argv[1:])
