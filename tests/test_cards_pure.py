"""cards.py without a daemon: the widget parsers, the card objects, slugs."""

import re

import pytest

from comfyui_trainything.cards import (
    build_cards,
    default_intro,
    make_run_id,
    normalize_slug,
    parse_labels_input,
    parse_meta_input,
)


def test_parse_labels_input():
    assert parse_labels_input("", 3) == [None, None, None]
    assert parse_labels_input("a\n\nc\n", 4) == ["a", None, "c", None]
    with pytest.raises(ValueError):
        parse_labels_input("a\nb\nc", 2)


def test_parse_meta_input():
    assert parse_meta_input(None, 2) == [{}, {}]
    one = parse_meta_input('{"k": 1}', 3)
    assert one == [{"k": 1}] * 3 and one[0] is not one[1]
    assert parse_meta_input('[{"i": 0}, {"i": 1}]', 2) == [{"i": 0}, {"i": 1}]
    assert parse_meta_input('{"i": 0}\n{"i": 1}\n', 2) == [{"i": 0}, {"i": 1}]
    with pytest.raises(ValueError):
        parse_meta_input('[{"i": 0}, {"i": 1}]', 3)
    with pytest.raises(ValueError):
        parse_meta_input("not json", 1)


def test_build_cards_meta_precedence():
    cards = build_cards(
        ["sha0", "sha1"],
        "RUN",
        labels=["first", None],
        metas=[{"cfg": 9, "index": 99}, {}],
        facts={"cfg": 6.5, "seed": 1, "walk": "complete"},
    )
    assert cards[0] == {
        "ref": "RUN-0",
        "label": "first",
        "meta": {"cfg": 9, "seed": 1, "walk": "complete", "index": 99, "run": "RUN"},
        "kind": "image",
        "src": "blob:sha0",
    }
    assert "label" not in cards[1] and cards[1]["meta"]["index"] == 1


def test_build_pair_cards():
    cards = build_cards(["a0"], "R", shas_b=["b0"])
    assert cards[0]["items"] == [
        {"kind": "image", "src": "blob:a0", "ref": "R-0-a"},
        {"kind": "image", "src": "blob:b0", "ref": "R-0-b"},
    ]
    with pytest.raises(ValueError):
        build_cards(["a0", "a1"], "R", shas_b=["b0"])


def test_slug_and_run_and_intro():
    assert normalize_slug("  Sprites 08 — Café!! ") == "sprites-08-cafe"
    run = make_run_id("12", now=1791460800.25)
    assert re.fullmatch(r"20261008T120000250-n12-[0-9a-f]{4}", run), run
    assert default_intro("a witch") == "Generated from: a witch. Reject anything that misses it."
    assert default_intro("  ") is None
