"""A judged fixture queue splits into the expected index sets in single,
pair, choose and select, with contested and unjudged in neither.

Driven end to end: each queue is created over the owner API with
min_reviews 2 and fixed order, the twelve sprites are uploaded and added with
refs, two reviewers join over the reviewer API and judge through /next and
/judge (disagreeing where a card should be contested, both skipping where it
should be unjudged, and stopping before the last card so it is never judged),
then results?format=labels feeds split.py."""

import pytest

np = pytest.importorskip("numpy")

from reviewer import Reviewer  # noqa: E402

from comfyui_trainything.cards import make_run_id, send_batch  # noqa: E402
from comfyui_trainything.split import split_labels  # noqa: E402

OPTIONS = [
    {"id": "hair", "label": "Hair colour"},
    {"id": "scarf", "label": "Scarf"},
    {"id": "hands", "label": "Hands"},
    {"id": "ok", "label": "Fine"},
]


def run_queue(daemon, sprite_batch, mode, slug, plan_a, plan_b, options=None, pair=False):
    """plan_x: per batch index, the reviewer's verdict; a missing index ends
    that reviewer's session (the card and those after it go unseen)."""
    client = daemon.client()
    cfg = {
        "title": f"Split {mode}",
        "slug": slug,
        "mode": mode,
        "order": "fixed",
        "feedback": "on_reject" if mode == "single" else "optional",
        "completion": {"min_reviews_per_card": 2},
        "options": options,
    }
    run = make_run_id(slug)
    handle = send_batch(
        client,
        sprite_batch,
        cfg,
        run,
        images_b=sprite_batch[::-1].copy() if pair else None,
    )
    assert handle["added"] == 12
    index = {ref: i for i, ref in enumerate(handle["refs"])}

    def decide(plan):
        return lambda ref: plan.get(index[ref])

    def note(ref, verdict):
        return f"note on {index[ref]}" if verdict == "reject" else None

    Reviewer(daemon.url, handle["slug"], "Ada").review(decide(plan_a), note)
    Reviewer(daemon.url, handle["slug"], "Bea").review(decide(plan_b), note)
    labels = client.results(handle["slug"], "labels")
    return split_labels(labels, handle["refs"], mode), labels, client.status(handle["slug"])


def plans(agree, disagree_a, disagree_b, skip=(10,)):
    """Both reviewers say `agree[i]`; on index 9 they split; on `skip` both
    skip; index 11 is never reached."""
    a = dict(agree)
    b = dict(agree)
    a[9], b[9] = disagree_a, disagree_b
    for i in skip:
        a[i] = b[i] = "skip"
    return a, b


def test_single(daemon, sprite_batch):
    agree = {i: "accept" for i in range(5)} | {i: "reject" for i in range(5, 8)}
    a, b = plans(agree, "accept", "reject")
    a[8], b[8] = "reject", "accept"
    s, labels, status = run_queue(daemon, sprite_batch, "single", "split-single", a, b)
    assert s.accepted == [0, 1, 2, 3, 4]
    assert s.rejected == [5, 6, 7]
    assert s.contested == [8, 9]
    assert s.unjudged == [10, 11]
    assert not set(s.contested + s.unjudged) & set(s.accepted + s.rejected)
    # Notes on rejects came back on the lines.
    assert [ln["notes"] for ln in s.lines[5:8]] == [["note on 5", "note on 5"], ["note on 6", "note on 6"], ["note on 7", "note on 7"]]
    # `judged` counts cards with any judgment, skips included: 0..10.
    assert status["done"] is False and status["judged"] == 11 and status["contested"] == 2


def test_pair(daemon, sprite_batch):
    agree = {i: "a" for i in range(6)} | {i: "b" for i in range(6, 9)}
    a, b = plans(agree, "a", "b")
    s, _, _ = run_queue(daemon, sprite_batch, "pair", "split-pair", a, b, pair=True)
    assert s.accepted == list(range(9))
    assert s.accepted_from == ["a"] * 6 + ["b"] * 3
    assert s.rejected == list(range(9))
    assert s.rejected_from == ["b"] * 6 + ["a"] * 3
    assert s.contested == [9] and s.unjudged == [10, 11]


def test_choose(daemon, sprite_batch):
    agree = {0: "hair", 1: "hair", 2: "hair", 3: "scarf", 4: "scarf", 5: "hands", 6: "ok", 7: "ok", 8: "ok"}
    a, b = plans(agree, "hair", "ok")
    s, _, _ = run_queue(daemon, sprite_batch, "choose", "split-choose", a, b, options=OPTIONS)
    assert s.options == {"hair": [0, 1, 2], "scarf": [3, 4], "hands": [5], "ok": [6, 7, 8]}
    assert s.accepted == [] and s.rejected == []
    assert s.contested == [9] and s.unjudged == [10, 11]
    assert s.indices("option", "hair") == [0, 1, 2]


def test_select(daemon, sprite_batch):
    agree = {
        0: ["hair"], 1: ["hair"], 2: ["hair"], 3: ["hair", "scarf"], 4: ["scarf"],
        5: [], 6: ["ok"], 7: ["ok"], 8: ["ok"],
    }
    # Index 9: one ticks hands, the other nothing -> hands at exactly half.
    a, b = plans(agree, ["hands"], [])
    s, _, _ = run_queue(daemon, sprite_batch, "select", "split-select", a, b, options=OPTIONS)
    assert s.options == {"hair": [0, 1, 2, 3], "scarf": [3, 4], "ok": [6, 7, 8]}
    assert s.accepted == [] and s.rejected == []
    assert s.contested == [9] and s.unjudged == [10, 11]
    # "none of these" is a judged answer, not unjudged.
    assert s.verdicts[5] == []
