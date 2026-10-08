"""K2 without ComfyUI: Filter's rules over labels.jsonl (split.filter_indices),
Load's card reading (nodes/common.py: image cards, blob paths, a whole queue
as one run, mixed sizes) and Load's fingerprint against a throwaway daemon."""

import asyncio
import json

import pytest

from comfyui_trainything.nodes import common as c
from comfyui_trainything.split import filter_indices, infer_mode, order_lines


def line(ref, verdict, **kw):
    out = {"ref": ref, "label": None, "verdict": verdict}
    out.update(kw)
    out.setdefault("n", 0 if verdict == "unjudged" else 1)
    out.setdefault("contested", verdict == "contested")
    out.setdefault("notes", [])
    return out


def jsonl(lines):
    return "".join(json.dumps(ln) + "\n" for ln in lines)


SINGLE = jsonl([line(f"r-{i}", v, accept=0, reject=0, skip=0) for i, v in enumerate(
    ["accept", "reject", "accept", "contested", "unjudged", "reject", "accept", "unjudged"])])


def test_single_rules_by_position():
    assert filter_indices(SINGLE, 8, "accepted") == [0, 2, 6]
    assert filter_indices(SINGLE, 8, "rejected") == [1, 5]
    assert filter_indices(SINGLE, 8, "contested") == [3]
    assert filter_indices(SINGLE, 8, "unjudged") == [4, 7]
    assert filter_indices(SINGLE, 8, "option", "hair") == []


def test_choose_option_is_the_consensus():
    verdicts = ["hair", "ok", "hair", "scarf", "unjudged", "hair", "hands", "contested", "ok", "ok", "scarf", "ok"]
    text = jsonl([line(f"r-{i}", v, counts={}) for i, v in enumerate(verdicts)])
    assert infer_mode([json.loads(ln) for ln in text.splitlines()]) == "choose"
    assert filter_indices(text, 12, "option", "hair") == [0, 2, 5]
    assert filter_indices(text, 12, "option", " scarf ") == [3, 10]
    assert filter_indices(text, 12, "unjudged") == [4]
    assert filter_indices(text, 12, "contested") == [7]
    assert filter_indices(text, 12, "accepted") == [] and filter_indices(text, 12, "rejected") == []


def test_select_option_includes_the_id():
    verdicts = [["hair", "hands"], ["ok"], "unjudged", ["hands"], ["hair"]]
    text = jsonl([line(f"r-{i}", v, counts={}) for i, v in enumerate(verdicts)])
    assert infer_mode([json.loads(ln) for ln in text.splitlines()]) == "select"
    assert filter_indices(text, 5, "option", "hair") == [0, 4]
    assert filter_indices(text, 5, "option", "hands") == [0, 3]
    assert filter_indices(text, 5, "unjudged") == [2]


def test_pair_lines_are_inferred():
    text = jsonl([line("p-0", "a", accept=0, reject=0, skip=0), line("p-1", "unjudged", accept=0, reject=0, skip=0),
                  line("p-2", "b", accept=0, reject=0, skip=0)])
    lines = [json.loads(ln) for ln in text.splitlines()]
    assert infer_mode(lines) == "pair"
    assert filter_indices(text, 3, "accepted") == [0, 2]
    assert filter_indices(text, 3, "unjudged") == [1]


def test_mode_inference():
    assert infer_mode([line("x", "accept", accept=1, reject=0, skip=0)]) == "single"
    assert infer_mode([line("x", "unjudged", accept=0, reject=0, skip=0)]) == "single"
    assert infer_mode([line("x", "unjudged", counts={})]) == "choose"


def test_line_count_must_match_the_batch():
    with pytest.raises(ValueError, match="labels has 8 lines for 12 images"):
        filter_indices(SINGLE, 12, "accepted")
    with pytest.raises(ValueError, match="option_id"):
        filter_indices(SINGLE, 8, "option", "")
    with pytest.raises(ValueError, match="unknown rule"):
        filter_indices(SINGLE, 8, "keepers")


def test_a_line_index_wins_over_position():
    lines = [line("b", "reject", index=1), line("a", "accept", index=0), line("c", "accept", meta={"index": 2})]
    assert [ln["ref"] for ln in order_lines(lines, 3)] == ["a", "b", "c"]
    assert filter_indices(jsonl(lines), 3, "accepted") == [0, 2]
    # Indices that are not exactly 0..B-1 (Send's per-run index in an appended
    # queue, say) are ignored: position decides.
    dup = [line("a", "accept", index=0), line("b", "reject", index=0)]
    assert filter_indices(jsonl(dup), 2, "rejected") == [1]


# ---------------------------------------------------------------------------
# Load's pieces
# ---------------------------------------------------------------------------


def card(ref, items, meta=None, judgments=(), label=None):
    return {"ref": ref, "label": label, "meta": meta or {}, "items": items, "judgments": list(judgments), "summary": {}}


def img(src, ref=None, side=None):
    return {"kind": "image", "src": src, "ref": ref, "side": side}


def test_image_cards_skip_other_kinds():
    doc = {"queue": {"mode": "single"}, "cards": [
        card("r-0", [img("blobs/a.png")]),
        card("t-0", [{"kind": "text", "src": None, "text": "hello"}]),
        card("r-1", [img("blobs/b.png")]),
        card("v-0", [{"kind": "video", "src": "blobs/v.mp4"}]),
    ]}
    kept, skipped = c.image_cards(doc)
    assert [k["ref"] for k in kept] == ["r-0", "r-1"]
    assert [s["ref"] for s in skipped] == ["t-0", "v-0"]


def test_pair_items_a_side_first():
    items = c.card_items(card("p", [img("blobs/b.png", "p-b", "b"), img("blobs/a.png", "p-a", "a")]))
    assert [it["src"] for it in items] == ["blobs/a.png", "blobs/b.png"]


def test_load_run_maps_the_whole_queue():
    doc = {"queue": {"mode": "single"}, "cards": [
        card("r1-0", [img("blobs/a.png")], {"index": 0, "seed": 1}),
        card("t-0", [{"kind": "text", "src": None}]),
        card("r1-1", [img("blobs/b.png")], {"index": 1, "seed": 2}, label="two"),
        card("r2-0", [img("blobs/c.png")], {"index": 0, "seed": 3}, judgments=[{"verdict": "reject", "note": "no"}], label="three"),
    ]}
    labels = jsonl([line("r1-0", "accept"), line("t-0", "accept"), line("r2-0", "reject")])
    run, skipped = c.load_run(doc, labels, "# summary\n")
    assert run.refs == ["r1-0", "r1-1", "r2-0"] and [s["ref"] for s in skipped] == ["t-0"]
    assert run.split.accepted == [0] and run.split.rejected == [2] and run.split.unjudged == [1]
    assert [ln["verdict"] for ln in run.lines] == ["accept", "unjudged", "reject"]
    # accepted_meta keeps what Send attached (Send's per-run index included).
    assert json.loads(c.meta_json(run, run.split.accepted)) == [{"index": 0, "seed": 1, "ref": "r1-0"}]
    assert c.notes_text(run) == "three: no"
    assert c.LOAD_OUTPUTS == ("accepted", "rejected", "images", "labels", "summary", "notes", "accepted_meta", "accepted_indices")


def test_blob_paths_stay_in_the_folder(tmp_path):
    assert c.blob_path(str(tmp_path), "blobs/x.png") == str((tmp_path / "blobs" / "x.png").resolve())
    with pytest.raises(ValueError):
        c.blob_path(str(tmp_path), "../escape.png")


def test_decode_batch_resizes_to_the_first(tmp_path):
    np = pytest.importorskip("numpy")
    pytest.importorskip("torch")
    from comfyui_trainything.png import encode_png

    a = np.zeros((8, 6, 3), np.float32)
    b = np.ones((4, 4, 3), np.float32)
    paths = []
    for name, arr in (("a", a), ("b", b), ("c", a)):
        p = tmp_path / f"{name}.png"
        p.write_bytes(encode_png(arr))
        paths.append(str(p))
    batch, resized = c.decode_batch(paths)
    assert tuple(batch.shape) == (3, 8, 6, 3) and resized == 1
    assert float(batch[1].min()) == 1.0 and float(batch[0].max()) == 0.0


def test_unknown_slug_is_named(daemon, monkeypatch):
    monkeypatch.setenv("TRAINYTHING_URL", daemon.url)
    monkeypatch.setenv("TRAINYTHING_TOKEN", daemon.token)

    async def go():
        client = await c.connect()
        await c.find_queue_by_slug(client, "no-such-queue-k2")

    with pytest.raises(c.QueueNotFound, match="no TrainyThing queue named 'no-such-queue-k2'"):
        asyncio.run(go())


def test_fingerprint_moves_with_judgments_and_close(daemon, monkeypatch):
    from reviewer import Reviewer

    monkeypatch.setenv("TRAINYTHING_URL", daemon.url)
    monkeypatch.setenv("TRAINYTHING_TOKEN", daemon.token)
    client = daemon.client()
    client.create_queue({"title": "fp", "slug": "k2-fingerprint", "mode": "single",
                         "completion": {"min_reviews_per_card": 1}})
    client.add_cards("k2-fingerprint", [{"ref": f"t-{i}", "kind": "text", "text": f"card {i}"} for i in range(3)])
    fp = lambda: asyncio.run(c.load_fingerprint("k2-fingerprint"))  # noqa: E731
    first = fp()
    assert fp() == first  # nothing moved: the same value, so ComfyUI reuses Load
    rev = Reviewer(daemon.url, "k2-fingerprint", "Ada")
    seen = []
    rev.review(lambda ref: None if seen else seen.append(ref) or "accept")
    second = fp()
    assert second != first and fp() == second
    client.close("k2-fingerprint")
    third = fp()
    assert third not in (first, second)
    # A queue that cannot be read never repeats: Load runs and says why.
    assert asyncio.run(c.load_fingerprint("no-such-queue-k2")) != asyncio.run(c.load_fingerprint("no-such-queue-k2"))
