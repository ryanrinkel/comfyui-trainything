"""Send, Wait and Review in a real ComfyUI on CPU.

One throwaway daemon and one throwaway ComfyUI (tests/comfyrun.py) for the
module; each test submits an API-format prompt from tests/fixtures/prompts
through POST /prompt, judges over the reviewer API (tests/reviewer.py), and
reads what the nodes did from /history: Save Image for the image sockets, Save
Latent for the latent, Preview as Text for the strings and the index list.

Clause -> test:
  link in /history; 8 accepted / 4 rejected with notes; notes 4 lines;
  accepted_meta 8 objects; the run's folder ............. test_gate_review
  same slug appends, 24 cards, run 2 splits its own 12 .... test_second_submission_appends
  cancel leaves the queue open and prints the slug ........ test_cancel_mid_wait
  timeout partial: judged split, the rest unjudged ........ test_timeout_partial
  timeout block: Save Image downstream does not run ....... test_timeout_block
  zero rejects: the rejected branch does not run, no error  test_zero_rejects
  /system_stats answers during the wait ................... test_gate_review, test_cancel_mid_wait
  pair send makes pair cards ............................... test_pair_send
  no token in examples, prompts, blobs or the QR .......... test_token_never_leaks
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path

import pytest

from conftest import PACK, SPRITES, start_daemon

pytest.importorskip("torch", reason="needs the harness venv (tests/run.sh)")
from comfyrun import COMFY, Comfy, load_prompt, messages, ui_images, ui_texts  # noqa: E402
from reviewer import Reviewer  # noqa: E402

pytestmark = [
    pytest.mark.comfy,
    pytest.mark.skipif(not (COMFY / "main.py").exists(), reason="no harness ComfyUI (tests/harness/setup_comfy.sh)"),
]

LOAD_LATER = "TrainyThing Load will read it later"
STATE: dict = {}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("k1")
    daemon = start_daemon(root)
    comfy = None
    try:
        comfy = Comfy(root / "comfy", daemon)
        yield daemon, comfy
    finally:
        if comfy is not None:
            comfy.stop()
        daemon.stop()


def index_of(ref: str) -> int:
    return int(ref.rsplit("-", 1)[1])


def run_of(ref: str) -> str:
    return ref.rsplit("-", 1)[0]


def wait_cards(daemon, slug: str, cards: int, timeout: float = 90) -> dict:
    client = daemon.client()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.find_queue(slug) is not None:
            st = client.status(slug)
            if st["cards"] >= cards:
                return st
        time.sleep(0.3)
    raise AssertionError(f"queue {slug} never reached {cards} cards")


def prompt(name: str, **inputs) -> dict:
    """A fixture prompt with widget values of the TrainyThing nodes replaced."""
    p = load_prompt(name)
    for n in p.values():
        if n["class_type"].startswith("TrainyThing"):
            for k, v in inputs.items():
                if k in n["inputs"]:
                    n["inputs"][k] = v
    return p


def stats_latency(comfy) -> float:
    t0 = time.monotonic()
    stats = comfy.get("/system_stats", timeout=5)
    assert stats["system"]["comfyui_version"]
    return time.monotonic() - t0


def no_errors(h):
    assert h["status"]["status_str"] == "success", h["status"]
    assert "execution_error" not in messages(h)
    assert "execution_interrupted" not in messages(h)


def text_of(h, node) -> str:
    texts = ui_texts(h, node)
    assert len(texts) == 1, (node, texts)
    return texts[0]


# ---------------------------------------------------------------------------


def test_gate_review(env):
    daemon, comfy = env
    pid = comfy.submit(prompt("gate", slug="k1-gate"))
    wait_cards(daemon, "k1-gate", 12)

    # The node is waiting; ComfyUI's server is not.
    lat = [stats_latency(comfy) for _ in range(5)]
    assert max(lat) < 1.0, lat
    assert comfy.history(pid) is None  # still running

    def decide(ref):
        return "accept" if index_of(ref) < 8 else "reject"

    def note(ref, verdict):
        return f"sprite {index_of(ref):02d} is off-model" if verdict == "reject" else None

    assert Reviewer(daemon.url, "k1-gate", "Ada").review(decide, note) == 12
    h = comfy.wait_history(pid)
    no_errors(h)

    # /history shows the link: on the node's own ui and through the link socket.
    link = f"{daemon.url}/r/k1-gate"
    assert ui_texts(h, "20") == [link]
    assert text_of(h, "40") == link
    assert text_of(h, "41") == "k1-gate"
    assert len(ui_images(h, "20")) == 1  # the QR

    assert len(ui_images(h, "30")) == 8  # accepted
    assert len(ui_images(h, "31")) == 4  # rejected
    notes = text_of(h, "35").splitlines()
    assert len(notes) == 4
    assert notes[0] == "sprite 08: sprite 08 is off-model"
    meta = json.loads(text_of(h, "36"))
    assert len(meta) == 8
    assert [m["index"] for m in meta] == list(range(8))
    assert all(m["source"] == "k1-test" and "walk" in m for m in meta)
    labels = [json.loads(ln) for ln in text_of(h, "33").splitlines()]
    assert len(labels) == 12 and [ln["verdict"] for ln in labels] == ["accept"] * 8 + ["reject"] * 4
    assert "8 accept" in text_of(h, "34")

    # accepted_indices is a ComfyUI list: the nodes after it ran once per keeper.
    assert ui_texts(h, "37") == [str(i) for i in range(8)]
    by_index = ui_images(h, "39")
    assert len(by_index) == 8
    import numpy as np
    from PIL import Image

    for k, img in enumerate(by_index):
        saved = np.asarray(Image.open(comfy.output / img["subfolder"] / img["filename"]).convert("RGB"))
        assert np.array_equal(saved, np.asarray(Image.open(SPRITES[k]).convert("RGB"))), k

    # accepted_latent: the latent's 8 keepers.
    from safetensors.torch import load_file

    lat_out = h["outputs"]["32"]["latents"][0]
    tensors = load_file(str(comfy.output / lat_out["subfolder"] / lat_out["filename"]))
    assert tuple(tensors["latent_tensor"].shape) == (8, 4, 8, 8)

    folder = comfy.output / "trainything" / "k1-gate"
    for name in ("results.json", "labels.jsonl", "summary.md"):
        assert (folder / name).is_file(), name
    blobs = list((folder / "blobs").iterdir())
    assert len(blobs) == 12
    doc = json.loads((folder / "results.json").read_text())
    assert len(doc["cards"]) == 12
    assert all(item["src"].startswith("blobs/") for c in doc["cards"] for item in c["items"])
    STATE["gate_run"] = run_of(labels[0]["ref"])
    assert "k1-gate" in comfy.log_text() and "8 accepted, 4 rejected" in comfy.log_text()


def test_second_submission_appends(env):
    daemon, comfy = env
    if "gate_run" not in STATE:
        pytest.skip("needs test_gate_review's queue")
    labels_in = "\n".join(f"second {i:02d}" for i in range(12))
    # Send + Wait this time: the queue handle travels on the custom socket.
    pid = comfy.submit(prompt("send-wait", slug="k1-gate", labels=labels_in))
    st = wait_cards(daemon, "k1-gate", 24)
    assert st["cards"] == 24 and st["done"] is False

    def decide(ref):
        assert run_of(ref) != STATE["gate_run"], "only the new cards are left to judge"
        return "accept" if index_of(ref) % 2 == 0 else "reject"

    assert Reviewer(daemon.url, "k1-gate", "Ada").review(decide, lambda r, v: "no" if v == "reject" else None) == 12
    h = comfy.wait_history(pid)
    no_errors(h)
    assert text_of(h, "41") == "k1-gate"
    assert len(ui_images(h, "20")) == 1  # Send's QR
    assert len(ui_images(h, "30")) == 6
    assert len(ui_images(h, "31")) == 6
    labels = [json.loads(ln) for ln in text_of(h, "33").splitlines()]
    assert len(labels) == 12
    assert {run_of(ln["ref"]) for ln in labels} != {STATE["gate_run"]}
    assert len({run_of(ln["ref"]) for ln in labels}) == 1
    assert [ln["label"] for ln in labels] == [f"second {i:02d}" for i in range(12)]
    meta = json.loads(text_of(h, "36"))
    assert [m["index"] for m in meta] == [0, 2, 4, 6, 8, 10]
    assert ui_texts(h, "37") == ["0", "2", "4", "6", "8", "10"]
    assert len(text_of(h, "35").splitlines()) == 6
    doc = json.loads((comfy.output / "trainything" / "k1-gate" / "results.json").read_text())
    assert len(doc["cards"]) == 24


def test_cancel_mid_wait(env):
    daemon, comfy = env
    pid = comfy.submit(prompt("gate", slug="k1-cancel"))
    wait_cards(daemon, "k1-cancel", 12)
    rev = Reviewer(daemon.url, "k1-cancel", "Ada")
    seen = []

    def three(ref):
        if len(seen) == 3:
            return None
        seen.append(ref)
        return "accept"

    assert rev.review(three) == 3
    lat = [stats_latency(comfy) for _ in range(3)]
    assert max(lat) < 1.0, lat
    comfy.interrupt()
    h = comfy.wait_history(pid, timeout=30)
    assert "execution_interrupted" in messages(h)
    assert "execution_error" not in messages(h)
    assert "30" not in h["outputs"] and "31" not in h["outputs"]
    st = daemon.client().status("k1-cancel")
    assert st["status"] == "open" and st["judged"] == 3
    log = comfy.log_text()
    line = next(ln for ln in log.splitlines() if "wait cancelled" in ln)
    assert "k1-cancel" in line and LOAD_LATER in line
    # The cancelled queue keeps taking verdicts.
    assert Reviewer(daemon.url, "k1-cancel", "Bea").review(lambda ref: "accept") == 12


def test_timeout_partial(env):
    daemon, comfy = env
    pid = comfy.submit(prompt("gate", slug="k1-partial", timeout_minutes=0.2, on_timeout="partial"))
    wait_cards(daemon, "k1-partial", 12)
    plan = {0: "accept", 1: "reject", 2: "accept", 3: "accept", 4: "reject"}
    rev = Reviewer(daemon.url, "k1-partial", "Ada")
    assert rev.review(lambda ref: plan.get(index_of(ref)), lambda r, v: "meh" if v == "reject" else None) == 5
    h = comfy.wait_history(pid, timeout=60)
    no_errors(h)
    assert len(ui_images(h, "30")) == 3
    assert len(ui_images(h, "31")) == 2
    labels = [json.loads(ln) for ln in text_of(h, "33").splitlines()]
    assert [ln["verdict"] for ln in labels] == ["accept", "reject", "accept", "accept", "reject"] + ["unjudged"] * 7
    assert ui_texts(h, "37") == ["0", "2", "3"]
    assert len(json.loads(text_of(h, "36"))) == 3
    assert "k1-partial: timed out" in comfy.log_text()
    assert daemon.client().status("k1-partial")["status"] == "open"
    assert (comfy.output / "trainything" / "k1-partial" / "labels.jsonl").is_file()


def test_timeout_block(env):
    daemon, comfy = env
    pid = comfy.submit(prompt("gate", slug="k1-block", timeout_minutes=0.1, on_timeout="block"))
    wait_cards(daemon, "k1-block", 12)
    plan = {0: "accept", 1: "reject"}
    Reviewer(daemon.url, "k1-block", "Ada").review(lambda ref: plan.get(index_of(ref)), lambda r, v: "meh" if v == "reject" else None)
    h = comfy.wait_history(pid, timeout=60)
    no_errors(h)
    # Nothing downstream ran: no Save Image, no Save Latent, no preview.
    for node in ("30", "31", "32", "33", "34", "35", "36", "37", "39", "40", "41"):
        assert node not in h["outputs"], node
    assert len(ui_images(h, "20")) == 1  # the node itself still shows its QR
    assert "k1-block: timed out" in comfy.log_text()


def test_zero_rejects(env):
    daemon, comfy = env
    pid = comfy.submit(prompt("gate", slug="k1-allgood"))
    wait_cards(daemon, "k1-allgood", 12)
    assert Reviewer(daemon.url, "k1-allgood", "Ada").review(lambda ref: "accept") == 12
    h = comfy.wait_history(pid)
    no_errors(h)
    assert len(ui_images(h, "30")) == 12
    assert "31" not in h["outputs"]  # the rejected branch did not run
    assert text_of(h, "35") == ""  # no notes, and no error for it
    assert len(json.loads(text_of(h, "36"))) == 12


def test_pair_send(env):
    daemon, comfy = env
    pid = comfy.submit(load_prompt("send-pair"))
    h = comfy.wait_history(pid)
    no_errors(h)
    assert ui_texts(h, "20") == [f"{daemon.url}/r/k1-pair"]
    doc = json.loads(daemon.client().results("k1-pair", "json"))
    assert doc["queue"]["mode"] == "pair"
    assert len(doc["cards"]) == 12
    for card in doc["cards"]:
        assert [it["ref"] for it in card["items"]] == [card["ref"] + "-a", card["ref"] + "-b"]


def test_token_never_leaks(env):
    daemon, comfy = env
    token = daemon.token.encode()
    assert len(token) >= 16
    files = list((PACK / "examples").glob("*.json")) + list((PACK / "tests" / "fixtures" / "prompts").glob("*.json"))
    assert files
    files += list((comfy.output / "trainything").rglob("*"))  # results, labels, summary, every uploaded PNG
    files += list(comfy.temp.rglob("*.png"))  # the QR previews
    pngs = 0
    for f in files:
        if f.is_file():
            data = f.read_bytes()
            assert token not in data, f
            pngs += data[:8] == b"\x89PNG\r\n\x1a\n"
    assert pngs >= 24
    # And the uploaded PNGs carry the graph, as Save Image's do.
    from comfyui_trainything.png import png_text

    for blob in (comfy.output / "trainything" / "k1-gate" / "blobs").iterdir():
        text = png_text(blob.read_bytes())
        graph = json.loads(text["prompt"])
        assert {"TrainyThingReview", "TrainyThingSend"} & {n["class_type"] for n in graph.values()}
        assert not re.search(re.escape(daemon.token), json.dumps(text))
