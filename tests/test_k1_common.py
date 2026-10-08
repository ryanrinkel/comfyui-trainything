"""nodes/common.py without ComfyUI: the queue config, this run's labels, the
notes and meta outputs, the wait loop (non-blocking, interrupt, timeout), the
QR, and (with torch) the latent slicer against core's LatentFromBatch rules."""

import asyncio
import json
import time

import pytest

from comfyui_trainything.nodes import common as c
from comfyui_trainything.split import split_labels


def test_queue_config_defaults_and_min_reviews():
    cfg = c.queue_config(title="ComfyUI batch", prompt_text="a red fox")
    assert cfg["completion"] == {"min_reviews_per_card": 1}
    assert cfg["mode"] == "single" and cfg["feedback"] == "on_reject" and cfg["identity"] == "name_email"
    assert cfg["order"] == "coverage" and cfg["project"] == "comfyui"
    assert cfg["intro"] == "Generated from: a red fox. Reject anything that misses it."
    assert "options" not in cfg and "slug" not in cfg
    cfg = c.queue_config(title="t", intro="mine", min_reviews=3, slug=" s1 ", options="a:A, b:B")
    assert cfg["intro"] == "mine" and cfg["completion"]["min_reviews_per_card"] == 3 and cfg["slug"] == "s1"
    assert "options" not in cfg  # single: the daemon refuses options


def test_options_for_choose_and_select():
    cfg = c.queue_config(title="t", mode="choose", options="hair:Hair colour, scarf:Scarf\nok")
    assert cfg["options"] == [{"id": "hair", "label": "Hair colour"}, {"id": "scarf", "label": "Scarf"}, {"id": "ok", "label": "ok"}]
    with pytest.raises(ValueError):
        c.queue_config(title="t", mode="select", options="only")
    with pytest.raises(ValueError):
        c.queue_config(title="t", mode="rank")


def _labels(*lines):
    return "".join(json.dumps(ln) + "\n" for ln in lines)


def test_run_labels_one_line_per_index_unjudged_filled():
    daemon_labels = _labels(
        {"ref": "old-0", "label": None, "verdict": "accept", "n": 1, "contested": False, "notes": []},
        {"ref": "r-0", "label": "a", "verdict": "reject", "n": 1, "contested": False, "notes": ["dark"]},
        {"ref": "r-2", "label": "c", "verdict": "unjudged", "n": 0, "contested": False, "notes": []},
    )
    lines = c.run_labels(daemon_labels, ["r-0", "r-1", "r-2"])
    assert [ln["ref"] for ln in lines] == ["r-0", "r-1", "r-2"]
    assert [ln["verdict"] for ln in lines] == ["reject", "unjudged", "unjudged"]
    text = c.labels_text(lines)
    assert len(text.splitlines()) == 3 and text.count('"unjudged"') == 2


def _run(mode, cards, verdicts):
    refs = [f"r-{i}" for i in range(len(cards))]
    lines = [{"ref": r, "label": cards[i].get("label"), "verdict": v, "n": 1, "contested": False, "notes": []}
             for i, (r, v) in enumerate(zip(refs, verdicts))]
    for i, card in enumerate(cards):
        card.setdefault("ref", refs[i])
    return c.RunResults(mode=mode, refs=refs, split=split_labels(lines, refs, mode), lines=lines, cards=cards, summary_md="")


def test_notes_are_rejection_notes_prefixed_by_label():
    cards = [
        {"label": "s1", "judgments": [{"verdict": "accept", "note": "nice"}]},
        {"label": None, "judgments": [{"verdict": "reject", "note": "too\n dark"}, {"verdict": "reject", "note": " "}]},
        {"label": "s3", "judgments": [{"verdict": "reject", "note": "hands"}, {"verdict": "reject", "note": None}]},
    ]
    run = _run("single", cards, ["accept", "reject", "reject"])
    assert c.notes_text(run) == "r-1: too dark\ns3: hands"  # no label: the ref (K3)


def test_accepted_meta_keeps_the_card_meta():
    cards = [{"meta": {"seed": 5, "index": 0}}, {"meta": {"seed": 6, "index": 1}}, None]
    run = _run("single", cards[:2] + [{"meta": {}}], ["accept", "reject", "accept"])
    metas = json.loads(c.meta_json(run, run.split.accepted))
    assert [m["index"] for m in metas] == [0, 2] and metas[0]["seed"] == 5 and metas[1]["ref"] == "r-2"


class FakeClient:
    """Async ``status`` that becomes done after ``done_after`` polls."""

    def __init__(self, done_after=None):
        self.polls = 0
        self.done_after = done_after

    async def status(self, qid):
        self.polls += 1
        done = self.done_after is not None and self.polls >= self.done_after
        return {"cards": 4, "judged": min(4, self.polls), "done": done}


HANDLE = {"queue_id": "q1", "slug": "s", "refs": ["a", "b", "c", "d"]}


def test_wait_polls_until_done_without_blocking_the_loop():
    ticks = []

    async def main():
        async def ticker():
            while True:
                ticks.append(time.monotonic())
                await asyncio.sleep(0.01)

        t = asyncio.create_task(ticker())
        res = await c.wait_until_done(FakeClient(done_after=3), HANDLE, 30, interrupted=lambda: False, poll_s=0.1)
        t.cancel()
        return res

    res = asyncio.run(main())
    assert res.timed_out is False and res.status["done"] is True
    gaps = [b - a for a, b in zip(ticks, ticks[1:])]
    assert len(ticks) > 10 and max(gaps) < 0.25  # the loop kept running between polls


def test_wait_times_out():
    t0 = time.monotonic()
    res = asyncio.run(c.wait_until_done(FakeClient(), HANDLE, 0.3, interrupted=lambda: False, poll_s=0.1))
    assert res.timed_out is True and time.monotonic() - t0 < 2


def test_wait_sees_the_interrupt_between_polls():
    flag = {"on": False}
    client = FakeClient()

    async def main():
        async def press():
            await asyncio.sleep(0.4)
            flag["on"] = True

        asyncio.create_task(press())
        await c.wait_until_done(client, HANDLE, 60, interrupted=lambda: flag["on"], poll_s=10)

    t0 = time.monotonic()
    with pytest.raises(c.WaitInterrupted):
        asyncio.run(main())
    assert time.monotonic() - t0 < 1.5  # not after the 10 s poll
    assert client.polls == 1


def test_left_open_line_names_the_slug():
    line = c.left_open_line("sprites-08")
    assert "sprites-08" in line and "TrainyThing Load will read it later" in line


def test_qr_png_is_a_png_with_room_for_the_link():
    pytest.importorskip("segno")
    from PIL import Image
    import io

    data = c.qr_png("http://127.0.0.1:4747/r/sprites-08", "12 cards")
    img = Image.open(io.BytesIO(data))
    assert img.format == "PNG" and img.height > img.width * 0.9


def test_slice_latent_mirrors_latent_from_batch():
    torch = pytest.importorskip("torch")
    samples = torch.arange(6 * 4 * 2 * 2, dtype=torch.float32).reshape(6, 4, 2, 2)
    out = c.slice_latent({"samples": samples}, [1, 4])
    assert torch.equal(out["samples"], samples[[1, 4]]) and out["batch_index"] == [1, 4]
    one_mask = torch.ones(1, 1, 16, 16)
    out = c.slice_latent({"samples": samples, "noise_mask": one_mask, "batch_index": [10, 11, 12, 13, 14, 15]}, [0, 5])
    assert out["noise_mask"].shape[0] == 1 and out["batch_index"] == [10, 15]
    masks = torch.arange(3, dtype=torch.float32).reshape(3, 1, 1, 1).expand(3, 1, 4, 4)
    out = c.slice_latent({"samples": samples, "noise_mask": masks}, [4, 2])
    # repeated up to the batch (0,1,2,0,1,2), then sliced: index 4 -> 1, index 2 -> 2
    assert out["noise_mask"][:, 0, 0, 0].tolist() == [1.0, 2.0]
    with pytest.raises(ValueError):
        c.slice_latent({"samples": samples}, [6])


