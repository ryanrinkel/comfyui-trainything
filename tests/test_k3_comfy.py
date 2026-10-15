"""pair mode and options in a real ComfyUI on CPU.

One throwaway daemon and ComfyUI for the module (as the other ComfyUI tests); the graphs
are tests/fixtures/prompts/ab-pair.json and options.json (build.py).

Clause -> test:
  The A/B workflow: twelve pairs, the reviewer picks A seven times, B four,
  skips one; accepted is the seven A and four B images in index order;
  preferences.jsonl has eleven lines; the skipped pair is unjudged ..... test_ab_pair_gate
  the pair rejected set is the losing side; accepted_latent blocked
  in pair mode, without error ..................................... test_ab_pair_gate
  Filter with both images and images_b on pair labels picks the
  winning (accepted) / losing (rejected) side; A only without
  images_b ........................................................ test_ab_pair_gate
  A choose queue with four reason tags: labels counts match, Filter
  per id matches, Review's accepted and rejected are both empty and
  both blocked without error ...................................... test_choose_reasons_gate
  select with the same tags: a card tagged with two ids is matched
  by Filter on either ............................................. test_select_tags
  options missing / malformed on choose: an error naming the problem test_choose_without_options
  the token never leaks into the new examples ..................... test_token_never_leaks_k3
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

TAGS = ["hair", "scarf", "hands", "ok"]


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    root = tmp_path_factory.mktemp("k3")
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
# helpers (as the Load tests')
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


def saved(comfy, h, node) -> list:
    return [np.asarray(Image.open(comfy.output / im["subfolder"] / im["filename"]).convert("RGB"))
            for im in ui_images(h, node)]


def same(a: list, b: list) -> bool:
    return len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b))


def wait_cards(daemon, slug, cards, timeout=90):
    client = daemon.client()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if client.find_queue(slug) is not None and client.status(slug)["cards"] >= cards:
            return
        time.sleep(0.3)
    raise AssertionError(f"{slug} never reached {cards} cards")


def filter_out(h, node: str) -> tuple:
    """(indices, count) of Filter 8k from its Preview as Text nodes; an
    empty subset blocks indices, so its preview does not run: ([], 0)."""
    k = node[1]
    count = int(text_of(h, f"11{k}"))
    if count == 0:
        assert f"10{k}" not in h["outputs"], "an empty indices list must not reach the node downstream"
    return [int(t) for t in ui_texts(h, f"10{k}")], count


def review_run(comfy, daemon, prompt, slug, decide, notes=None, timeout=120) -> dict:
    """Submit a graph whose Review waits, judge it through the reviewer API,
    and return its /history once the run finishes."""
    pid = comfy.submit(prompt)
    wait_cards(daemon, slug, 12)
    judged = Reviewer(daemon.url, slug, "Ada").review(decide, notes)
    assert judged == 12
    h = comfy.wait_history(pid, timeout=timeout)
    no_errors(h)
    return h


# ---------------------------------------------------------------------------
# Pair: A and B
# ---------------------------------------------------------------------------

B_WINS = {1, 3, 8, 10}
SKIPPED = 5
A_WINS = [i for i in range(12) if i not in B_WINS and i != SKIPPED]
JUDGED = sorted(A_WINS + sorted(B_WINS))


def pair_verdict(ref):
    i = index_of(ref)
    return "skip" if i == SKIPPED else ("b" if i in B_WINS else "a")


def pair_note(ref, verdict):
    i = index_of(ref)
    return {0: "A is cleaner", 3: "B keeps the outline"}.get(i)


def test_ab_pair_gate(env):
    daemon, comfy = env
    slug = "k3-ab"
    h = review_run(comfy, daemon, load_prompt("ab-pair"), slug, pair_verdict, pair_note)
    assert len(A_WINS) == 7 and len(B_WINS) == 4
    a, b = saved(comfy, h, "42"), saved(comfy, h, "43")
    assert len(a) == len(b) == 12 and not same(a, b)

    # accepted: the seven A and four B winners, in index order; rejected: the other side.
    winners = [b[i] if i in B_WINS else a[i] for i in JUDGED]
    losers = [a[i] if i in B_WINS else b[i] for i in JUDGED]
    assert same(saved(comfy, h, "30"), winners)
    assert same(saved(comfy, h, "31"), losers)
    assert ui_texts(h, "37") == [str(i) for i in JUDGED]  # accepted_indices: the 11 pairs with a winner
    assert [m["index"] for m in json.loads(text_of(h, "36"))] == JUDGED
    # accepted_latent: blocked in pair mode (the latent is A's), silently.
    assert "32" not in h["outputs"]
    assert "pair mode: accepted_latent is blocked" in comfy.log_text()

    # labels: one line per pair; the skipped pair is unjudged.
    labels = [json.loads(ln) for ln in text_of(h, "33").splitlines()]
    assert len(labels) == 12
    assert labels[SKIPPED]["verdict"] == "unjudged" and labels[SKIPPED]["skip"] == 1
    assert [ln["verdict"] for i, ln in enumerate(labels) if i != SKIPPED] == ["b" if i in B_WINS else "a" for i in JUDGED]

    # notes: every note, with the side chosen; summary: the daemon's summary.md as is.
    assert text_of(h, "35").splitlines() == ["sprite 00 (chose A): A is cleaner", "sprite 03 (chose B): B keeps the outline"]
    folder = comfy.output / "trainything" / slug
    assert text_of(h, "34") == (folder / "summary.md").read_text()
    assert "7 A" in text_of(h, "34") and "4 B" in text_of(h, "34")

    # The run's folder: preferences.jsonl with eleven lines, the winner as chosen.
    for name in ("results.json", "labels.jsonl", "summary.md", "preferences.jsonl"):
        assert (folder / name).is_file(), name
    prefs = [json.loads(ln) for ln in (folder / "preferences.jsonl").read_text().splitlines()]
    assert len(prefs) == 11
    assert [index_of(p["ref"]) for p in prefs] == JUDGED
    for p in prefs:
        side = "b" if index_of(p["ref"]) in B_WINS else "a"
        assert p["chosen"]["ref"].endswith(f"-{side}") and p["rejected"]["ref"].endswith("-" + ("a" if side == "b" else "b"))

    # Filter with both batches picks the winning / losing side; with A only, A.
    assert filter_out(h, "80") == (JUDGED, 11) and same(saved(comfy, h, "90"), winners)
    assert filter_out(h, "81") == (JUDGED, 11) and same(saved(comfy, h, "91"), losers)
    assert filter_out(h, "82") == (JUDGED, 11) and same(saved(comfy, h, "92"), [a[i] for i in JUDGED])
    assert "120" not in h["outputs"]  # Filter's latent: blocked when the picks come from both sides
    log = comfy.log_text()
    assert "filter accepted: 11 of 12 (4 from images_b)" in log and "filter rejected: 11 of 12 (7 from images_b)" in log


# ---------------------------------------------------------------------------
# Choose and select: the reasons, counted, and defects, tagged
# ---------------------------------------------------------------------------

CHOOSE = {1: "hair", 4: "hair", 9: "hair", 0: "scarf", 5: "scarf", 2: "hands", 7: "hands", 10: "hands", 11: "hands",
          3: "ok", 6: "ok", 8: "ok"}


def option_prompt(mode: str, slug: str) -> dict:
    p = load_prompt("options")
    p["20"]["inputs"].update(mode=mode, slug=slug, labels="\n".join(f"{mode} sprite {i:02d}" for i in range(12)))
    return p


def assert_keepers_blocked(h):
    """choose / select: no yes or no. accepted, rejected and every keeper
    socket are blocked, silently: nothing downstream of them ran."""
    no_errors(h)
    for node in ("30", "31", "32", "36", "37", "38", "39"):
        assert node not in h["outputs"], node
    assert filter_out(h, "84") == ([], 0) and "94" not in h["outputs"]


def test_choose_reasons_gate(env):
    daemon, comfy = env
    slug = "k3-choose"
    h = review_run(comfy, daemon, option_prompt("choose", slug), slug, lambda ref: CHOOSE[index_of(ref)],
                   lambda ref, v: "too red" if v == "hair" else None)
    sent = saved(comfy, h, "42")

    # labels: one line per card, its verdict, its counts; the totals match.
    labels = [json.loads(ln) for ln in text_of(h, "33").splitlines()]
    assert [ln["verdict"] for ln in labels] == [CHOOSE[i] for i in range(12)]
    assert all(ln["counts"] == {CHOOSE[i]: 1} for i, ln in enumerate(labels))
    totals = {t: sum(ln["counts"].get(t, 0) for ln in labels) for t in TAGS}
    assert totals == {"hair": 3, "scarf": 2, "hands": 4, "ok": 3}
    doc = json.loads(daemon.client().results(slug, "json"))
    assert {t: doc["summary"]["counts"].get(t, 0) for t in TAGS} == totals
    assert doc["queue"]["options"][0] == {"id": "hair", "label": "Hair colour"}
    assert doc["queue"]["feedback"] == "optional"  # on_reject has no reject in choose
    assert "| Hair colour | hair | 3 |" in text_of(h, "34")

    # Filter per id: the right cards, the right images.
    for k, tag in enumerate(TAGS):
        want = [i for i in range(12) if CHOOSE[i] == tag]
        assert filter_out(h, f"8{k}") == (want, len(want)), tag
        assert same(saved(comfy, h, f"9{k}"), [sent[i] for i in want]), tag

    # Review's accepted and rejected: both empty, both blocked, no error.
    assert_keepers_blocked(h)
    assert "choose mode: hair 3, scarf 2, hands 4, ok 3; accepted and rejected are empty by design" in comfy.log_text()
    assert text_of(h, "35").splitlines() == [f"choose sprite {i:02d} [hair]: too red" for i in (1, 4, 9)]


SELECT = {0: ["hair", "hands"], 1: ["hair"], 2: ["scarf"], 3: [], 4: ["hands", "ok"], 5: ["ok"], 6: ["hair", "scarf"],
          7: ["ok"], 8: ["hands"], 9: [], 10: ["scarf", "hands"], 11: ["ok"]}


def test_select_tags(env):
    daemon, comfy = env
    slug = "k3-select"
    h = review_run(comfy, daemon, option_prompt("select", slug), slug, lambda ref: SELECT[index_of(ref)],
                   lambda ref, v: "two things" if len(v) == 2 else None)
    sent = saved(comfy, h, "42")
    labels = [json.loads(ln) for ln in text_of(h, "33").splitlines()]
    assert [sorted(ln["verdict"]) for ln in labels] == [sorted(SELECT[i]) for i in range(12)]
    for k, tag in enumerate(TAGS):
        want = [i for i in range(12) if tag in SELECT[i]]
        assert filter_out(h, f"8{k}") == (want, len(want)), tag
        assert same(saved(comfy, h, f"9{k}"), [sent[i] for i in want]), tag
    # A card tagged with two ids is matched on either.
    for i, ids in SELECT.items():
        if len(ids) == 2:
            for tag in ids:
                assert i in filter_out(h, f"8{TAGS.index(tag)}")[0], (i, tag)
    assert_keepers_blocked(h)
    notes = text_of(h, "35").splitlines()
    assert notes == ["select sprite 00 [hair, hands]: two things", "select sprite 06 [hair, scarf]: two things",
                     "select sprite 10 [scarf, hands]: two things", "select sprite 04 [hands, ok]: two things"]


def test_choose_without_options(env):
    daemon, comfy = env
    for options, problem in (("", "mode choose needs options"), ("Hair:Hair colour, ok:Fine", "'Hair' is not an option id")):
        p = option_prompt("choose", "k3-no-options")
        p["20"]["inputs"]["options"] = options
        h = comfy.wait_history(comfy.submit(p))
        assert h["status"]["status_str"] == "error"
        err = next(body for name, body in h["status"]["messages"] if name == "execution_error")
        assert err["node_id"] == "20" and problem in err["exception_message"], err["exception_message"]
    assert daemon.client().find_queue("k3-no-options") is None  # nothing was created


def test_token_never_leaks_k3(env):
    daemon, comfy = env
    token = daemon.token.encode()
    files = [PACK / "examples" / "ab-pair.json", PACK / "examples" / "choose-reasons.json"]
    files += list((PACK / "tests" / "fixtures" / "prompts").glob("*.json"))
    files += list((comfy.output / "trainything").rglob("*")) + list((comfy.output / "k3").rglob("*"))
    checked = 0
    for f in files:
        if f.is_file():
            assert token not in f.read_bytes(), f
            checked += 1
    assert checked >= 50
