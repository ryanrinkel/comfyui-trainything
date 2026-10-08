"""K3 without ComfyUI: options parsed and checked, feedback and the intro per
mode, notes per mode, Filter's sides on pair labels, select matched on either
id, and (with torch) the pair / choose sockets of split_outputs."""

import json

import pytest

from comfyui_trainything.cards import default_intro
from comfyui_trainything.nodes import common as c
from comfyui_trainything.split import filter_indices, filter_pick, split_labels

TAGS = "hair:Hair colour, scarf:Scarf, hands:Hands, ok:Fine"


def jsonl(lines):
    return "".join(json.dumps(ln) + "\n" for ln in lines)


def line(ref, verdict, **kw):
    out = {"ref": ref, "label": None, "verdict": verdict}
    out.update(kw)
    out.setdefault("n", 0 if verdict == "unjudged" else 1)
    out.setdefault("contested", verdict == "contested")
    out.setdefault("notes", [])
    return out


# --- intro (§10 question 7) --------------------------------------------------


def test_default_intro_reads_in_every_mode():
    assert default_intro("a fox") == "Generated from: a fox. Reject anything that misses it."
    assert default_intro("a fox", "pair") == "Generated from: a fox. Pick the one that matches it better."
    assert default_intro("a fox", "choose") == "Generated from: a fox. Pick the answer that fits each one best."
    assert default_intro("a fox", "select") == "Generated from: a fox. Tick every answer that applies to each one."
    assert default_intro(None, "pair") is None
    cfg = c.queue_config(title="t", mode="choose", options=TAGS, prompt_text="a fox")
    assert "Reject" not in cfg["intro"] and cfg["intro"].startswith("Generated from: a fox.")


def test_pair_intro_quotes_the_prompt_only_when_both_sides_share_it():
    def graph(text_a, text_b):
        g = {"9": {"class_type": "TrainyThingReview", "inputs": {"images": ["1", 0], "images_b": ["2", 0]}}}
        for nid, text, tid in (("1", text_a, "11"), ("2", text_b, "12")):
            g[nid] = {"class_type": "VAEDecode", "inputs": {"samples": [f"{nid}0", 0]}}
            g[f"{nid}0"] = {"class_type": "KSampler", "inputs": {"seed": 1, "steps": 2, "cfg": 7.0, "sampler_name": "euler",
                            "scheduler": "normal", "denoise": 1.0, "positive": [tid, 0], "negative": [tid, 0]}}
            g[tid] = {"class_type": "CLIPTextEncode", "inputs": {"text": text}}
        return g

    same = graph("a fox", "a fox")
    facts = c.graph_facts(same, "9")
    assert facts["prompt_text"] == "a fox"
    assert c.intro_prompt(same, "9", facts, "pair") == "a fox"
    differ = graph("a fox", "a wolf")
    assert c.intro_prompt(differ, "9", c.graph_facts(differ, "9"), "pair") is None
    assert c.intro_prompt(differ, "9", c.graph_facts(differ, "9"), "single") == "a fox"


# --- options -------------------------------------------------------------------


def test_options_sent_for_choose_and_select():
    for mode in ("choose", "select"):
        cfg = c.queue_config(title="t", mode=mode, options=TAGS)
        assert [o["id"] for o in cfg["options"]] == ["hair", "scarf", "hands", "ok"]
        assert cfg["options"][0] == {"id": "hair", "label": "Hair colour"}


@pytest.mark.parametrize("options, problem", [
    ("", "needs options"),
    ("   ", "needs options"),
    ("hair:Hair colour", "at least 2"),
    ("Hair:Hair colour, ok:Fine", "'Hair' is not an option id"),
    ("hair colour, ok", "'hair colour' is not an option id"),
    (":Hair, ok:Fine", "has no id"),
    ("skip:Skip it, ok:Fine", "'skip' is reserved"),
    ("ok:Fine, ok:Okay", "'ok' appears twice"),
    (", ".join(f"o{i}" for i in range(13)), "at most 12"),
    ("hair:" + "x" * 81 + ", ok", "longer than 80"),
])
def test_options_missing_or_malformed_name_the_problem(options, problem):
    for mode in ("choose", "select"):
        with pytest.raises(ValueError, match=problem):
            c.queue_config(title="t", mode=mode, options=options)


def test_options_ignored_elsewhere_with_a_line(capsys):
    cfg = c.queue_config(title="t", mode="single", options=TAGS)
    assert "options" not in cfg
    assert "options are for choose / select; ignored in single mode" in capsys.readouterr().out
    cfg = c.queue_config(title="t", mode="pair", options="Not Valid At All")  # not even parsed
    assert "options" not in cfg


def test_feedback_on_reject_is_optional_outside_single(capsys):
    assert c.queue_config(title="t", mode="single")["feedback"] == "on_reject"
    for mode, opts in (("pair", ""), ("choose", TAGS), ("select", TAGS)):
        assert c.queue_config(title="t", mode=mode, options=opts)["feedback"] == "optional"
        assert f"no reject in {mode} mode" in capsys.readouterr().out
        for other in ("off", "optional", "required"):
            assert c.queue_config(title="t", mode=mode, options=opts, feedback=other)["feedback"] == other


# --- notes per mode --------------------------------------------------------------


def _run(mode, cards, verdicts, options=None):
    refs = [f"r-{i}" for i in range(len(cards))]
    lines = [line(r, v) for r, v in zip(refs, verdicts)]
    for i, card in enumerate(cards):
        card.setdefault("ref", refs[i])
    results = {"queue": {"mode": mode, "options": options}, "cards": cards}
    return c.RunResults(mode=mode, refs=refs, split=split_labels(lines, refs, mode), lines=lines, cards=cards,
                        summary_md="", results=results)


def test_pair_notes_say_the_side_chosen():
    cards = [
        {"label": "p0", "judgments": [{"verdict": "a", "note": "cleaner\n lines"}]},
        {"label": None, "judgments": [{"verdict": "b", "note": "A has six fingers"}]},
        {"label": "p2", "judgments": [{"verdict": "skip", "note": "can't tell"}]},
        {"label": "p3", "judgments": [{"verdict": "a", "note": None}]},
    ]
    run = _run("pair", cards, ["a", "b", "unjudged", "a"])
    assert c.notes_text(run) == "p0 (chose A): cleaner lines\nr-1 (chose B): A has six fingers"


def test_choose_notes_grouped_by_option_in_the_queue_order():
    opts = [{"id": i, "label": i} for i in ("hair", "scarf", "hands", "ok")]
    cards = [
        {"label": "s0", "judgments": [{"verdict": "ok", "note": "fine"}]},
        {"label": "s1", "judgments": [{"verdict": "hands", "note": "six fingers"}]},
        {"label": "s2", "judgments": [{"verdict": "hair", "note": "too red"}]},
        {"label": "s3", "judgments": [{"verdict": "hair", "note": "orange"}, {"verdict": "skip", "note": "?"}]},
        {"label": "s4", "judgments": [{"verdict": "scarf", "note": None}]},
    ]
    run = _run("choose", cards, ["ok", "hands", "hair", "hair", "scarf"], opts)
    assert c.notes_text(run).splitlines() == [
        "s2 [hair]: too red", "s3 [hair]: orange", "s1 [hands]: six fingers", "s0 [ok]: fine"]


def test_select_notes_carry_every_id_ticked():
    opts = [{"id": i, "label": i} for i in ("hair", "scarf", "hands", "ok")]
    cards = [
        {"label": "s0", "judgments": [{"verdict": ["hands", "hair"], "note": "both off"}]},
        {"label": "s1", "judgments": [{"verdict": ["scarf"], "note": "scarf clips"}]},
        {"label": "s2", "judgments": [{"verdict": [], "note": "nothing wrong"}]},
    ]
    run = _run("select", cards, [["hair", "hands"], ["scarf"], []], opts)
    assert c.notes_text(run).splitlines() == [
        "s0 [hair, hands]: both off", "s1 [scarf]: scarf clips", "s2 [none]: nothing wrong"]


# --- Filter on pair and select labels ----------------------------------------------


PAIR = jsonl([line(f"p-{i}", v, accept=0, reject=0, skip=0) for i, v in enumerate(["a", "b", "unjudged", "b", "contested", "a"])])


def test_filter_pick_pair_sides():
    acc = filter_pick(PAIR, 6, "accepted")
    assert acc.mode == "pair" and acc.indices == [0, 1, 3, 5] and acc.sides == ["a", "b", "b", "a"]
    rej = filter_pick(PAIR, 6, "rejected")
    assert rej.indices == [0, 1, 3, 5] and rej.sides == ["b", "a", "a", "b"]
    assert filter_pick(PAIR, 6, "unjudged").sides == ["a"]
    assert filter_pick(PAIR, 6, "contested").indices == [4]


def test_select_card_with_two_ids_matched_on_either():
    text = jsonl([line(f"s-{i}", v, counts={}) for i, v in enumerate([["hair", "hands"], ["ok"], ["hands"], "unjudged"])])
    assert filter_indices(text, 4, "option", "hair") == [0]
    assert filter_indices(text, 4, "option", "hands") == [0, 2]
    assert filter_indices(text, 4, "option", "scarf") == []
    assert filter_pick(text, 4, "option", "hair").sides == ["a"]


# --- split_outputs with torch: pair and choose sockets ------------------------------


class _Block:
    def __repr__(self):
        return "BLOCK"


@pytest.fixture
def blocks(monkeypatch):
    monkeypatch.setattr(c, "blocker", lambda: _Block())


def _batch(n, value):
    torch = pytest.importorskip("torch")
    return torch.stack([torch.full((4, 4, 3), value + i / 100.0) for i in range(n)])


def test_pair_outputs_pick_sides_and_block_the_latent(blocks, capsys):
    torch = pytest.importorskip("torch")
    a, b = _batch(4, 0.1), _batch(4, 0.6)
    cards = [{"label": f"p{i}", "judgments": []} for i in range(4)]
    run = _run("pair", cards, ["a", "b", "skip", "b"])
    latent = {"samples": torch.zeros(4, 4, 1, 1)}
    outs = dict(zip(c.WAIT_OUTPUTS, c.split_outputs(run, a, latent, b)))
    assert torch.equal(outs["accepted"], torch.stack([a[0], b[1], b[3]]))
    assert torch.equal(outs["rejected"], torch.stack([b[0], a[1], a[3]]))
    assert isinstance(outs["accepted_latent"], _Block)
    assert "accepted_latent is blocked" in capsys.readouterr().out
    assert outs["accepted_indices"] == [0, 1, 3]
    assert [m["index"] for m in json.loads(outs["accepted_meta"])] == [0, 1, 3]
    # A B batch of another size is resized to A's.
    small = torch.nn.functional.interpolate(b.movedim(-1, 1), size=(2, 2)).movedim(1, -1)
    outs = dict(zip(c.WAIT_OUTPUTS, c.split_outputs(run, a, None, small)))
    assert tuple(outs["accepted"].shape) == (3, 4, 4, 3)


@pytest.mark.parametrize("mode, verdicts", [
    ("choose", ["hair", "ok", "hair", "unjudged"]),
    ("select", [["hair", "hands"], ["ok"], [], "unjudged"]),
])
def test_option_modes_block_every_keeper_socket(blocks, capsys, mode, verdicts):
    torch = pytest.importorskip("torch")
    a = _batch(4, 0.1)
    opts = [{"id": i, "label": i} for i in ("hair", "scarf", "hands", "ok")]
    run = _run(mode, [{"label": f"s{i}", "judgments": []} for i in range(4)], verdicts, opts)
    outs = dict(zip(c.WAIT_OUTPUTS, c.split_outputs(run, a, {"samples": torch.zeros(4, 4, 1, 1)})))
    for name in ("accepted", "rejected", "accepted_latent", "accepted_meta", "accepted_indices"):
        assert isinstance(outs[name], _Block), name
    assert outs["images"] is a and len(outs["labels"].splitlines()) == 4
    out = capsys.readouterr().out
    assert (f"{mode} mode: hair 2, scarf 0, hands 0, ok 1" if mode == "choose" else
            f"{mode} mode: hair 1, scarf 0, hands 1, ok 1") in out and "use TrainyThing Filter with rule option" in out
