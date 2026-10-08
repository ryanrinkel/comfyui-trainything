"""Gate K2 (PLAN-COMFY §8): Load and Filter in a real ComfyUI on CPU.

One throwaway daemon and ComfyUI for the module (as K1), plus, for the remote
clause, a second daemon (its own TRAINYTHING_HOME, another port) and a second
ComfyUI pointed at it with TRAINYTHING_FORCE_REMOTE=1, so the pack treats it
as remote and reads results/bundle.zip; that daemon never writes a file into
ComfyUI's folders.

Clause -> test:
  Run 2 of §5.2 against the queue K1 left (two Send+Wait runs of 12, same
  slug, 8 accepted + 4 rejected each): Load returns 24 images matching the
  uploaded tensors to within PNG rounding; accepted_meta matches what Send
  attached; Filter accepted -> 16, rejected -> 8 ....... test_run2_load_filter
  Filter option hair on a choose fixture -> the right three  test_choose_option_hair
  Filter unjudged on a half-judged queue -> the right half . test_half_judged_unjudged
  re-queuing the same graph: Load runs again only after a
  new judgment ........................................... test_rerun_only_after_a_judgment
  a daemon on another port, no shared filesystem: Load
  reads the bundle and gives the same images ............. test_remote_bundle_same_images
  Load wait on (timeout -> partial) ....................... test_load_wait_times_out_partial
  unknown slug -> an error naming it ...................... test_unknown_slug
  no token in the new example / prompts / folders ......... test_token_never_leaks_k2
"""

from __future__ import annotations

import json
import time

import pytest

from conftest import PACK, start_daemon

pytest.importorskip("torch", reason="needs the harness venv (tests/run.sh)")
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from comfyrun import COMFY, Comfy, load_prompt, messages, ui_images, ui_texts  # noqa: E402
from reviewer import Reviewer  # noqa: E402

pytestmark = [
    pytest.mark.comfy,
    pytest.mark.skipif(not (COMFY / "main.py").exists(), reason="no harness ComfyUI (tests/harness/setup_comfy.sh)"),
]

CHOOSE_OPTIONS = "hair:Hair colour, scarf:Scarf, hands:Hands, ok:Fine"
STATE: dict = {}


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("k2")
    daemon = start_daemon(root)
    comfy = None
    try:
        comfy = Comfy(root / "comfy", daemon)
        yield daemon, comfy
    finally:
        if comfy is not None:
            comfy.stop()
        daemon.stop()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def index_of(ref: str) -> int:
    return int(ref.rsplit("-", 1)[1])


def no_errors(h):
    assert h["status"]["status_str"] == "success", h["status"]
    assert "execution_error" not in messages(h), h["status"]
    assert "execution_interrupted" not in messages(h)


def text_of(h, node) -> str:
    texts = ui_texts(h, node)
    assert len(texts) == 1, (node, texts)
    return texts[0]


def cached_nodes(h) -> list:
    for name, body in h["status"]["messages"]:
        if name == "execution_cached":
            return list(body["nodes"])
    return []


def saved(comfy, h, node) -> list:
    """The images a Save Image node wrote, as uint8 arrays, in batch order."""
    return [np.asarray(Image.open(comfy.output / im["subfolder"] / im["filename"]).convert("RGB"))
            for im in ui_images(h, node)]


def max_diff(a: list, b: list) -> int:
    assert len(a) == len(b), (len(a), len(b))
    return max(int(np.abs(x.astype(np.int16) - y.astype(np.int16)).max()) for x, y in zip(a, b))


def send_prompt(slug: str, **send) -> dict:
    p = load_prompt("send")
    p["20"]["inputs"].update(slug=slug, **send)
    return p


def send_wait_prompt(slug: str, run_no: int) -> dict:
    """K1's Send + Wait, and the batch Send was given saved as it went in.
    The labels differ per run, or ComfyUI would reuse the first Send."""
    p = load_prompt("send-wait")
    p["20"]["inputs"]["slug"] = slug
    p["20"]["inputs"]["labels"] = "\n".join(f"run {run_no} sprite {i:02d}" for i in range(12))
    p["42"] = {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "k2/sent"}}
    return p


def load_prompt_for(slug: str, batch: int, *, wait: bool = False, timeout_minutes: float = 45.0) -> dict:
    p = load_prompt("load-filter")
    p["70"]["inputs"].update(slug=slug, wait=wait, timeout_minutes=timeout_minutes)
    p["79"]["inputs"]["batch_size"] = batch
    return p


def run(comfy, p, timeout=120) -> dict:
    h = comfy.wait_history(comfy.submit(p), timeout=timeout)
    no_errors(h)
    return h


def wait_cards(daemon, slug, cards, timeout=90):
    client = daemon.client()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.find_queue(slug) is not None and client.status(slug)["cards"] >= cards:
            return
        time.sleep(0.3)
    raise AssertionError(f"{slug} never reached {cards} cards")


def filter_out(h, rule_node: str) -> tuple:
    """(indices, count) of Filter 8k from its Preview as Text nodes. An empty
    subset blocks indices, so its preview does not run: ([], 0)."""
    k = rule_node[1]
    count = int(text_of(h, f"11{k}"))
    if count == 0:
        assert f"10{k}" not in h["outputs"], "an empty indices list must not reach the node downstream"
    indices = [int(t) for t in ui_texts(h, f"10{k}")]
    return indices, count


def loads_in_log(comfy, slug) -> int:
    return sum(1 for ln in comfy.log_text().splitlines() if f"[TrainyThing] {slug}: loaded " in ln)


# ---------------------------------------------------------------------------


def test_run2_load_filter(env):
    daemon, comfy = env
    slug = "k2-sprites"
    sent, wait_meta = [], []
    # The queue K1 leaves: two Send + Wait runs of 12 with one slug, 8 + 4 each.
    for n in (12, 24):
        pid = comfy.submit(send_wait_prompt(slug, n // 12))
        wait_cards(daemon, slug, n)
        rev = Reviewer(daemon.url, slug, "Ada")
        assert rev.review(lambda ref: "accept" if index_of(ref) < 8 else "reject",
                          lambda ref, v: f"sprite {index_of(ref):02d} is off" if v == "reject" else None) == 12
        h = comfy.wait_history(pid)
        no_errors(h)
        sent += saved(comfy, h, "42")
        wait_meta += json.loads(text_of(h, "36"))
    assert len(sent) == 24 and len(wait_meta) == 16

    h = run(comfy, load_prompt_for(slug, 24))
    loaded = saved(comfy, h, "71")
    assert len(loaded) == 24
    STATE["load_images"] = loaded
    STATE["diff"] = max_diff(loaded, sent)
    print(f"\nLoad vs the uploaded batch: max pixel difference {STATE['diff']}")
    assert STATE["diff"] <= 1, STATE["diff"]  # PNG rounding: uint8 in, uint8 out
    assert len(ui_images(h, "72")) == 16 and len(ui_images(h, "73")) == 8

    # accepted_meta is what Send attached: the same objects Wait returned, and
    # each one the card's meta in the daemon (plus its ref).
    # position is where each keeper is in this node's batch: Wait's is per
    # run, Load's across both runs (= accepted_indices); index is Send's.
    meta = json.loads(text_of(h, "77"))
    unplaced = lambda ms: [{k: v for k, v in m.items() if k != "position"} for m in ms]  # noqa: E731
    assert unplaced(meta) == unplaced(wait_meta)
    assert [m["position"] for m in meta] == [i for i in range(24) if i % 12 < 8]
    assert [m["position"] for m in wait_meta] == [m["index"] for m in wait_meta]
    doc = json.loads(daemon.client().results(slug, "json"))
    by_ref = {c["ref"]: c for c in doc["cards"]}
    for m in meta:
        card_meta = by_ref[m["ref"]]["meta"]
        assert {k: v for k, v in m.items() if k not in ("ref", "position")} == card_meta
        assert card_meta["source"] == "k1-test" and "walk" in card_meta and "run" in card_meta
    keepers = [i for i in range(24) if i % 12 < 8]
    assert ui_texts(h, "78") == [str(i) for i in keepers]
    labels = [json.loads(ln) for ln in text_of(h, "74").splitlines()]
    assert len(labels) == 24 and [ln["ref"] for ln in labels] == [c["ref"] for c in doc["cards"]]
    assert len(text_of(h, "76").splitlines()) == 8  # the rejection notes

    # Filter accepted -> 16, with the latent sliced alongside; rejected -> 8.
    idx, count = filter_out(h, "80")
    assert count == 16 and idx == keepers
    assert len(ui_images(h, "90")) == 16
    assert max_diff(saved(comfy, h, "90"), [loaded[i] for i in keepers]) == 0
    from safetensors.torch import load_file

    lat = h["outputs"]["120"]["latents"][0]
    assert tuple(load_file(str(comfy.output / lat["subfolder"] / lat["filename"]))["latent_tensor"].shape) == (16, 4, 8, 8)
    idx, count = filter_out(h, "81")
    assert count == 8 and idx == [i for i in range(24) if i % 12 >= 8]
    assert len(ui_images(h, "91")) == 8
    # unjudged, contested and option hair are empty here: their Save Image and
    # indices are blocked, count is 0, and nothing errors.
    for node, save in (("82", "92"), ("83", "93"), ("84", "94")):
        assert filter_out(h, node) == ([], 0)
        assert save not in h["outputs"]

    folder = comfy.output / "trainything" / slug
    for name in ("results.json", "labels.jsonl", "summary.md"):
        assert (folder / name).is_file()
    assert len(list((folder / "blobs").iterdir())) == 24
    assert f"[TrainyThing] {slug}: loaded 24 images, 16 accepted, 8 rejected" in comfy.log_text()
    assert "(local)" in [ln for ln in comfy.log_text().splitlines() if f"{slug}: loaded" in ln][-1]


def test_choose_option_hair(env):
    daemon, comfy = env
    slug = "k2-choose"
    h = run(comfy, send_prompt(slug, mode="choose", options=CHOOSE_OPTIONS, feedback="optional",
                               prompt="What is wrong with it?"))
    sent = saved(comfy, h, "42")
    hair = {1, 4, 9}
    others = ["scarf", "hands", "ok"]
    assert Reviewer(daemon.url, slug, "Ada").review(
        lambda ref: "hair" if index_of(ref) in hair else others[index_of(ref) % 3]) == 12
    h = run(comfy, load_prompt_for(slug, 12))
    idx, count = filter_out(h, "83")
    assert idx == [1, 4, 9] and count == 3
    assert max_diff(saved(comfy, h, "93"), [sent[i] for i in (1, 4, 9)]) <= 1
    # choose has no yes or no: accepted and rejected are empty and blocked.
    assert "72" not in h["outputs"] and "73" not in h["outputs"]
    assert filter_out(h, "80") == ([], 0) and filter_out(h, "81") == ([], 0)
    counts = json.loads(daemon.client().results(slug, "json"))["summary"]["counts"]
    assert counts["hair"] == 3


def test_half_judged_unjudged(env):
    daemon, comfy = env
    slug = "k2-half"
    run(comfy, send_prompt(slug))
    judged = []

    def six(ref):
        if len(judged) == 6:
            return None
        judged.append(index_of(ref))
        return "accept" if len(judged) % 2 else "reject"

    assert Reviewer(daemon.url, slug, "Ada").review(six, lambda r, v: "no" if v == "reject" else None) == 6
    h = run(comfy, load_prompt_for(slug, 12))
    rest = sorted(set(range(12)) - set(judged))
    idx, count = filter_out(h, "82")
    assert idx == rest and count == 6
    assert len(ui_images(h, "92")) == 6
    labels = [json.loads(ln) for ln in text_of(h, "74").splitlines()]
    assert [i for i, ln in enumerate(labels) if ln["verdict"] == "unjudged"] == rest
    STATE["half_judged"] = judged


def test_rerun_only_after_a_judgment(env):
    daemon, comfy = env
    slug = "k2-half"
    if "half_judged" not in STATE:
        pytest.skip("needs test_half_judged_unjudged's queue")
    before = loads_in_log(comfy, slug)
    h1 = run(comfy, load_prompt_for(slug, 12))
    assert loads_in_log(comfy, slug) == before  # cached: the queue did not move since the last Load
    assert "70" in cached_nodes(h1)
    h2 = run(comfy, load_prompt_for(slug, 12))
    assert "70" in cached_nodes(h2) and loads_in_log(comfy, slug) == before
    assert filter_out(h2, "82")[1] == 6

    # One more verdict: the fingerprint moves and Load runs again.
    rev = Reviewer(daemon.url, slug, "Ada")
    done = []
    assert rev.review(lambda ref: None if done else done.append(ref) or "accept") == 1
    h3 = run(comfy, load_prompt_for(slug, 12))
    assert "70" not in cached_nodes(h3)
    assert loads_in_log(comfy, slug) == before + 1
    assert filter_out(h3, "82")[1] == 5
    # And again unchanged: cached again.
    h4 = run(comfy, load_prompt_for(slug, 12))
    assert "70" in cached_nodes(h4) and loads_in_log(comfy, slug) == before + 1


def test_load_wait_times_out_partial(env):
    daemon, comfy = env
    slug = "k2-half"
    if "half_judged" not in STATE:
        pytest.skip("needs test_half_judged_unjudged's queue")
    t0 = time.monotonic()
    h = run(comfy, load_prompt_for(slug, 12, wait=True, timeout_minutes=0.1))
    assert time.monotonic() - t0 >= 5.5  # it waited for the timeout (6 s), then loaded
    assert filter_out(h, "82")[1] == 5
    assert f"{slug}: timed out after 0.1 min" in comfy.log_text()


def test_unknown_slug(env):
    daemon, comfy = env
    pid = comfy.submit(load_prompt_for("k2-nowhere", 12))
    h = comfy.wait_history(pid)
    assert h["status"]["status_str"] == "error"
    err = next(body for name, body in h["status"]["messages"] if name == "execution_error")
    assert err["node_id"] == "70" and "no TrainyThing queue named 'k2-nowhere'" in err["exception_message"]


def test_remote_bundle_same_images(env, tmp_path_factory):
    daemon, comfy = env
    if "load_images" not in STATE:
        pytest.skip("needs test_run2_load_filter's images")
    root = tmp_path_factory.mktemp("k2-remote")
    remote = start_daemon(root)
    comfy2 = None
    try:
        assert remote.home != daemon.home and remote.port != daemon.port
        comfy2 = Comfy(root / "comfy", remote, extra_env={"TRAINYTHING_FORCE_REMOTE": "1"})
        slug = "k2-remote"
        run(comfy2, send_prompt(slug))
        assert Reviewer(remote.url, slug, "Ada").review(
            lambda ref: "accept" if index_of(ref) < 8 else "reject", lambda r, v: "no" if v == "reject" else None) == 12
        h = run(comfy2, load_prompt_for(slug, 12))
        loaded = saved(comfy2, h, "71")
        # The same sprites as the local Load's first run, pixel for pixel.
        assert max_diff(loaded, STATE["load_images"][:12]) == 0
        assert filter_out(h, "80")[1] == 8 and filter_out(h, "81")[1] == 4
        line = [ln for ln in comfy2.log_text().splitlines() if f"{slug}: loaded" in ln][-1]
        assert "(bundle)" in line, line
        folder = comfy2.output / "trainything" / slug
        assert (folder / "results.json").is_file() and len(list((folder / "blobs").iterdir())) == 12
    finally:
        if comfy2 is not None:
            comfy2.stop()
        remote.stop()


def test_token_never_leaks_k2(env):
    daemon, comfy = env
    token = daemon.token.encode()
    files = [PACK / "examples" / "two-runs-load.json"] + list((PACK / "tests" / "fixtures" / "prompts").glob("*.json"))
    files += list((comfy.output / "trainything").rglob("*")) + list((comfy.output / "k2").rglob("*"))
    checked = 0
    for f in files:
        if f.is_file():
            assert token not in f.read_bytes(), f
            checked += 1
    assert checked >= 50
