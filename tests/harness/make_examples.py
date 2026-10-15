"""Builds examples/*.json (canvas workflows) from example_prompts.py through the
harness ComfyUI's frontend, and (with --shoot) takes the README screenshot of
the Review node: docs/screenshots/review-node.png.

    ~/.cache/comfyui-trainything/venv/bin/python tests/harness/make_examples.py [--check] [--shoot]

A throwaway daemon and ComfyUI (tests/comfyrun.py), both killed at the end.
--check queues two-runs-send.json and then two-runs-load.json as the frontend
would (canvas.mjs prompt), with a scripted review between them, then
ab-pair.json and choose-reasons.json, reviewed while their Review waits.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
PACK = HERE.parent.parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE))

import conftest  # noqa: E402  (loads the pack as comfyui_trainything)
from comfyrun import Comfy  # noqa: E402
from conftest import start_daemon  # noqa: E402
from example_prompts import EXAMPLES  # noqa: E402
import groups  # noqa: E402

EXAMPLES_DIR = PACK / "examples"
SHOTS = PACK / "docs" / "screenshots"


def check(root: Path, comfy: Comfy, daemon) -> None:
    """§5.2 through the frontend: the API prompts the frontend would queue for
    two-runs-send.json and two-runs-load.json, submitted; run 1 makes
    sprites-08, a scripted reviewer keeps 8, run 2 loads and filters them."""
    from reviewer import Reviewer

    def queue(name: str) -> dict:
        out = root / f"{name}.queued.json"
        subprocess.run(["node", str(HERE / "canvas.mjs"), "prompt", comfy.url, str(EXAMPLES_DIR / f"{name}.json"), str(out)], check=True)
        h = comfy.wait_history(comfy.submit(json.loads(out.read_text())))
        assert h["status"]["status_str"] == "success", h["status"]
        return h

    queue("two-runs-send")
    Reviewer(daemon.url, "sprites-08", "Ada").review(lambda ref: "accept" if int(ref.rsplit("-", 1)[1]) < 8 else "reject",
                                                    lambda ref, v: "off-model" if v == "reject" else None)
    h = queue("two-runs-load")
    saves = [o for o in h["outputs"].values() if "images" in o]
    assert len(saves) == 1 and len(saves[0]["images"]) == 8, h["outputs"]
    print("check: two-runs-send then two-runs-load queued from the frontend; 8 keepers saved")

    # K3: the two Review examples hold the run, so the review happens while
    # they wait. ab-pair: A for the first 7 cards served, B for the next 4,
    # skip the last; choose-reasons: the four tags in turn (3 hair).
    def queue_and_review(name: str, slug: str, decide) -> dict:
        out = root / f"{name}.queued.json"
        subprocess.run(["node", str(HERE / "canvas.mjs"), "prompt", comfy.url, str(EXAMPLES_DIR / f"{name}.json"), str(out)], check=True)
        pid = comfy.submit(json.loads(out.read_text()))
        client = daemon.client()
        for _ in range(300):
            if client.find_queue(slug) is not None and client.status(slug)["cards"] >= 12:
                break
            time.sleep(0.3)
        seen: list = []
        Reviewer(daemon.url, slug, "Ada").review(lambda ref: seen.append(ref) or decide(len(seen) - 1))
        h = comfy.wait_history(pid)
        assert h["status"]["status_str"] == "success", h["status"]
        return h

    h = queue_and_review("ab-pair", "sprites-ab", lambda k: "a" if k < 7 else ("b" if k < 11 else "skip"))
    saves = {nid: o for nid, o in h["outputs"].items() if "images" in o and nid != "20"}
    assert sorted(len(o["images"]) for o in saves.values()) == [11, 11], h["outputs"]
    prefs = (comfy.output / "trainything" / "sprites-ab" / "preferences.jsonl").read_text().splitlines()
    assert len(prefs) == 11, prefs
    print("check: ab-pair queued from the frontend; 7 A + 4 B, 1 skipped: 11 winners, 11 losers, 11 preferences")

    tags = ["hair", "scarf", "hands", "ok"]
    h = queue_and_review("choose-reasons", "sprites-reasons", lambda k: tags[k % 4])
    saves = {nid: o for nid, o in h["outputs"].items() if "images" in o and nid != "20"}
    assert [len(o["images"]) for o in saves.values()] == [3], h["outputs"]  # cards 0, 4, 8 served: hair
    print("check: choose-reasons queued from the frontend; Filter option hair saved the hair cards")


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
            groups.main([str(EXAMPLES_DIR / f"{name}.json")])  # the titled box around the TrainyThing part
        if "--check" in sys.argv:
            check(root, comfy, daemon)
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
