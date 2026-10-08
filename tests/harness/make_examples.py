"""Builds examples/*.json (canvas workflows) from example_prompts.py through the
harness ComfyUI's frontend, and (with --shoot) takes the README screenshot of
the Review node: docs/screenshots/review-node.png.

    ~/.cache/comfyui-trainything/venv/bin/python tests/harness/make_examples.py [--check] [--shoot]

A throwaway daemon and ComfyUI (tests/comfyrun.py), both killed at the end.
--check queues two-runs-send.json and then two-runs-load.json as the frontend
would (canvas.mjs prompt), with a scripted review between them.
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
