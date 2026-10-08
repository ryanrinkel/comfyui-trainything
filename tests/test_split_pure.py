"""split.py against hand-written labels.jsonl lines (the daemon-driven
version is test_daemon_split.py)."""

import json

import pytest

from comfyui_trainything.split import parse_labels, split_labels


def line(ref, verdict, contested=False, **kw):
    d = {"ref": ref, "label": None, "verdict": verdict, "n": 1, "contested": contested, "notes": []}
    d.update(kw)
    return json.dumps(d)


def test_single_with_foreign_refs_ignored():
    text = "\n".join(
        [
            line("r-0", "accept"),
            line("r-1", "reject"),
            line("r-2", "contested", True),
            line("r-3", "unjudged"),
            line("other-0", "accept"),  # another run's card in the same queue
        ]
    )
    s = split_labels(text, ["r-0", "r-1", "r-2", "r-3", "r-4"], "single")
    assert s.accepted == [0] and s.rejected == [1]
    assert s.contested == [2] and s.unjudged == [3, 4]
    assert s.verdicts == ["accept", "reject", "contested", "unjudged", "unjudged"]
    assert s.indices("accepted") == [0]


def test_pair_maps_chosen_and_other():
    text = "\n".join([line("p-0", "a"), line("p-1", "b"), line("p-2", "contested", True)])
    s = split_labels(text, ["p-0", "p-1", "p-2"], "pair")
    assert list(zip(s.accepted, s.accepted_from)) == [(0, "a"), (1, "b")]
    assert list(zip(s.rejected, s.rejected_from)) == [(0, "b"), (1, "a")]
    assert s.contested == [2]


def test_choose_and_select_options():
    choose = "\n".join([line("c-0", "hair"), line("c-1", "ok"), line("c-2", "hair")])
    s = split_labels(choose, ["c-0", "c-1", "c-2"], "choose")
    assert s.options == {"hair": [0, 2], "ok": [1]}
    assert s.accepted == [] and s.rejected == []
    assert s.indices("option", "hair") == [0, 2]
    select = "\n".join(
        [line("s-0", ["hair", "scarf"]), line("s-1", []), line("s-2", ["hair"], True)]
    )
    s = split_labels(select, ["s-0", "s-1", "s-2"], "select")
    assert s.options == {"hair": [0], "scarf": [0]}
    assert s.contested == [2] and s.unjudged == []
    with pytest.raises(ValueError):
        s.indices("option")
    with pytest.raises(ValueError):
        s.indices("nope")


def test_parse_labels_accepts_dicts_and_blank_lines():
    assert parse_labels([{"ref": "x"}, "", ' {"ref": "y"} ']) == [{"ref": "x"}, {"ref": "y"}]
    with pytest.raises(ValueError):
        parse_labels('{"no_ref": 1}')
