"""Builds examples/*.json (canvas workflows) from example_prompts.py through the
harness ComfyUI's frontend, and (with --shoot) takes the README screenshot of
the Review node: docs/screenshots/review-node.png.

    ~/.cache/comfyui-trainything/venv/bin/python tests/harness/make_examples.py [--shoot]

A throwaway daemon and ComfyUI (tests/comfyrun.py), both killed at the end.
examples/two-runs-load.json is written by hand below: TrainyThing Load is K2,
so the frontend does not know it yet.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import conftest  # noqa: E402  (loads the pack as comfyui_trainything)
from comfyrun import Comfy  # noqa: E402
from conftest import start_daemon  # noqa: E402
from example_prompts import EXAMPLES  # noqa: E402

EXAMPLES_DIR = PACK / "examples"
SHOTS = PACK / "docs" / "screenshots"


def load_workflow() -> dict:
    """§5.2 run 2: TrainyThing Load by slug -> the keepers, the summary, the
    keepers' meta. Widgets in Load's order (K2): slug, wait, timeout_minutes.
    Outputs as Wait's minus accepted_latent."""
    outputs = [("accepted", "IMAGE"), ("rejected", "IMAGE"), ("images", "IMAGE"), ("labels", "STRING"),
               ("summary", "STRING"), ("notes", "STRING"), ("accepted_meta", "STRING"), ("accepted_indices", "INT")]
    links = [[1, 1, 0, 2, 0, "IMAGE"], [2, 1, 4, 3, 0, "STRING"], [3, 1, 6, 4, 0, "STRING"]]
    load_outputs = []
    for slot, (name, typ) in enumerate(outputs):
        out = {"name": name, "type": typ, "links": [ln[0] for ln in links if ln[2] == slot] or None}
        if name == "accepted_indices":
            out["shape"] = 6
        load_outputs.append(out)
    nodes = [
        {"id": 1, "type": "TrainyThingLoad", "title": "TrainyThing Load", "pos": [0, 0], "size": [320, 330],
         "flags": {}, "order": 0, "mode": 0, "inputs": [], "outputs": load_outputs,
         "properties": {"Node name for S&R": "TrainyThingLoad"}, "widgets_values": ["sprites-08", False, 45]},
        {"id": 2, "type": "SaveImage", "title": "Save the keepers", "pos": [420, 0], "size": [320, 300], "flags": {},
         "order": 1, "mode": 0, "inputs": [{"name": "images", "type": "IMAGE", "link": 1}], "outputs": [],
         "properties": {"Node name for S&R": "SaveImage"}, "widgets_values": ["trainything/run2-keepers"]},
        {"id": 3, "type": "PreviewAny", "title": "Summary (rewrite the prompt from the notes)", "pos": [420, 340],
         "size": [420, 260], "flags": {}, "order": 2, "mode": 0,
         "inputs": [{"name": "source", "type": "*", "link": 2}], "outputs": [{"name": "STRING", "type": "STRING", "links": None}],
         "properties": {"Node name for S&R": "PreviewAny"}, "widgets_values": [None, False]},
        {"id": 4, "type": "PreviewAny", "title": "Keepers' meta (seeds, prompts)", "pos": [420, 640],
         "size": [420, 260], "flags": {}, "order": 3, "mode": 0,
         "inputs": [{"name": "source", "type": "*", "link": 3}], "outputs": [{"name": "STRING", "type": "STRING", "links": None}],
         "properties": {"Node name for S&R": "PreviewAny"}, "widgets_values": [None, False]},
    ]
    return {"last_node_id": 4, "last_link_id": 3, "nodes": nodes, "links": links, "groups": [], "config": {},
            "extra": {"ds": {"scale": 1, "offset": [40, 40]}}, "version": 0.4}


def main() -> None:
    shoot = "--shoot" in sys.argv
    root = Path(tempfile.mkdtemp(prefix="k1-examples-", dir=str(Path.home() / ".cache" / "comfyui-trainything")))
    daemon = start_daemon(root)
    comfy = None
    try:
        comfy = Comfy(root / "comfy", daemon)
        for name, fn in EXAMPLES.items():
            api = root / f"{name}.api.json"
            api.write_text(json.dumps(fn()))
            subprocess.run(["node", str(HERE / "canvas.mjs"), "convert", comfy.url, str(api), str(EXAMPLES_DIR / f"{name}.json")], check=True)
        (EXAMPLES_DIR / "two-runs-load.json").write_text(json.dumps(load_workflow(), indent=1) + "\n")
        print("wrote two-runs-load.json")
        if shoot:
            SHOTS.mkdir(parents=True, exist_ok=True)
            subprocess.run(["node", str(HERE / "canvas.mjs"), "shoot", comfy.url, str(EXAMPLES_DIR / "gate.json"),
                            daemon.url, "sprites-gate", str(SHOTS)], check=True, timeout=600)
            print(comfy.log_text()[-1500:])
    finally:
        if comfy is not None:
            comfy.stop()
        daemon.stop()
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
