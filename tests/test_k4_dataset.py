"""K4: examples/dataset.py (PLAN-COMFY §5.4, "a dataset of keepers with
captions"). A run's folder is built the way Review / Wait write it
(``write_run_folder_sync`` against a throwaway daemon, no ComfyUI), then the
script turns it into <ref>.png + <ref>.txt pairs. A second folder, written by
hand, covers pair and select mode. The script runs with ``python -S``: the
standard library only."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import PACK
from reviewer import Reviewer

np = pytest.importorskip("numpy")

from comfyui_trainything.cards import make_run_id, send_batch  # noqa: E402
from comfyui_trainything.nodes.common import write_run_folder_sync  # noqa: E402
from comfyui_trainything.trainything_client import connect  # noqa: E402
from comfyui_trainything.walk import walk_graph  # noqa: E402

SCRIPT = PACK / "examples" / "dataset.py"


def run_script(folder: Path, out: Path, *args: str) -> str:
    res = subprocess.run([sys.executable, "-S", str(SCRIPT), str(folder), "--out", str(out), *args],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    return res.stdout


def pairs(out: Path) -> dict:
    pngs = sorted(p.stem for p in out.glob("*.png"))
    assert pngs == sorted(p.stem for p in out.glob("*.txt"))
    return {s: (out / f"{s}.txt").read_text(encoding="utf-8") for s in pngs}


@pytest.fixture(scope="module")
def run_folder(daemon, sprite_batch, graph, tmp_path_factory):
    """Six sprites with the walk's facts; image 2 overrides prompt_text by the
    meta input, image 5 has a label; a reviewer keeps 0, 1, 2, 4 and rejects
    3 and 5. Then the run's folder, as the nodes write it."""
    client = connect(base_url=daemon.url, token=daemon.token, auto_start=False)
    facts = walk_graph(graph, "12")
    assert facts.get("prompt_text")
    run = make_run_id("k4")
    metas = [{} for _ in range(6)]
    metas[2] = {"prompt_text": "a witch, blue scarf"}
    handle = send_batch(
        client, sprite_batch[:6], {"title": "K4 dataset", "slug": "k4-dataset", "prompt": "Keep?", "order": "fixed",
                                   "project": "comfyui-k4"},
        run, labels="\n".join(f"sprite {i}" for i in range(6)), meta="\n".join(json.dumps(m) for m in metas),
        facts=facts,
    )
    keep = {f"{run}-{i}" for i in (0, 1, 2, 4)}
    Reviewer(daemon.url, handle["slug"], "Kay").review(
        lambda ref: "accept" if ref in keep else "reject", notes=lambda ref, v: "off model" if v == "reject" else None)
    root = tmp_path_factory.mktemp("k4-out")
    write_run_folder_sync(client, handle["queue_id"], handle["slug"], "single", str(root))
    return root / handle["slug"], run, facts["prompt_text"]


def test_keepers_become_captioned_pairs(run_folder, tmp_path):
    folder, run, prompt_text = run_folder
    out = tmp_path / "keepers"
    assert run_script(folder, out).strip() == f"4 pairs in {out}"
    got = pairs(out)
    assert set(got) == {f"{run}-{i}" for i in (0, 1, 2, 4)}
    assert got[f"{run}-0"] == prompt_text.strip() + "\n"
    assert got[f"{run}-2"] == "a witch, blue scarf\n"
    doc = json.loads((folder / "results.json").read_text())
    by_ref = {c["ref"]: c for c in doc["cards"]}
    for ref in got:
        assert (out / f"{ref}.png").read_bytes() == (folder / by_ref[ref]["items"][0]["src"]).read_bytes()


def test_rejected_rule_and_default_out(run_folder):
    folder, run, _ = run_folder
    res = subprocess.run([sys.executable, "-S", str(SCRIPT), str(folder), "--rule", "rejected"],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 0, res.stderr
    assert set(pairs(folder / "dataset-rejected")) == {f"{run}-3", f"{run}-5"}


def test_bad_rule_is_an_error(run_folder, tmp_path):
    folder, _, _ = run_folder
    res = subprocess.run([sys.executable, "-S", str(SCRIPT), str(folder), "--rule", "keepers"],
                         capture_output=True, text=True, timeout=60)
    assert res.returncode == 2 and "--rule" in res.stderr


def write_folder(folder: Path, mode: str, cards: list, lines: list, queue_prompt: str = "Which one?") -> None:
    (folder / "blobs").mkdir(parents=True)
    for card in cards:
        for it in card["items"]:
            (folder / it["src"]).write_bytes(it.pop("_bytes"))
    doc = {"queue": {"slug": folder.name, "mode": mode, "prompt": queue_prompt}, "cards": cards}
    (folder / "results.json").write_text(json.dumps(doc))
    (folder / "labels.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines))


def item(ref: str, data: bytes) -> dict:
    return {"kind": "image", "ref": ref, "src": f"blobs/{ref}.png", "_bytes": data}


def test_pair_takes_the_winning_side(tmp_path):
    folder = tmp_path / "ab"
    cards = [
        {"ref": "r-0", "label": "pair 0", "meta": {"prompt_text": "prompt A"}, "items": [item("r-0-a", b"A0"), item("r-0-b", b"B0")]},
        {"ref": "r-1", "label": "pair 1", "meta": {"prompt_text": "prompt A", "prompt_text_b": "prompt B"},
         "items": [item("r-1-b", b"B1"), item("r-1-a", b"A1")]},  # B listed first: the -a/-b refs decide
        {"ref": "r-2", "label": "pair 2", "meta": {"prompt_text": "prompt A"}, "items": [item("r-2-a", b"A2"), item("r-2-b", b"B2")]},
        {"ref": "r-3", "meta": {}, "items": [item("r-3-a", b"A3"), item("r-3-b", b"B3")]},
    ]
    lines = [{"ref": "r-0", "verdict": "a"}, {"ref": "r-1", "verdict": "b"}, {"ref": "r-2", "verdict": "b"},
             {"ref": "r-3", "verdict": "unjudged"}]
    write_folder(folder, "pair", cards, lines)
    out = tmp_path / "won"
    run_script(folder, out)
    got = pairs(out)
    assert got == {"r-0": "prompt A\n", "r-1": "prompt B\n", "r-2": "pair 2\n"}
    assert [(out / f"r-{i}.png").read_bytes() for i in range(3)] == [b"A0", b"B1", b"B2"]
    lost = tmp_path / "lost"
    run_script(folder, lost, "--rule", "rejected")
    assert [(lost / f"r-{i}.png").read_bytes() for i in range(3)] == [b"B0", b"A1", b"A2"]


def test_option_rule_in_select_and_contested(tmp_path):
    folder = tmp_path / "tags"
    cards = [{"ref": f"t-{i}", "meta": {}, "items": [item(f"t-{i}", bytes([i]))]} for i in range(4)]
    lines = [{"ref": "t-0", "verdict": ["hair", "hands"]}, {"ref": "t-1", "verdict": ["scarf"]},
             {"ref": "t-2", "verdict": ["hair"], "contested": True}, {"ref": "t-3", "verdict": ["hair"]}]
    write_folder(folder, "select", cards, lines, queue_prompt="What is wrong?")
    out = tmp_path / "hair"
    run_script(folder, out, "--rule", "option:hair")
    assert pairs(out) == {"t-0": "What is wrong?\n", "t-3": "What is wrong?\n"}
