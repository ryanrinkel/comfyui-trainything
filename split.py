"""labels.jsonl -> batch indices. Pure.

A labels line is the daemon's ``LabelLine`` (``src/core/results/labels.ts``):
``{ref, label, verdict, (accept, reject, skip | counts), n, contested, notes}``
where ``verdict`` is the card's consensus, or ``"contested"`` / ``"unjudged"``.

Every function takes ``refs``, the refs *this run* added, in batch order, so
an appended queue maps only its own cards: index ``i`` is ``refs[i]``. A ref
with no line is unjudged.

    single   accept -> accepted, reject -> rejected
    pair     a / b  -> accepted is the chosen image of each pair, rejected the
                       other: ``accepted_from[k]`` says which batch ("a" =
                       ``images``, "b" = ``images_b``) ``accepted[k]`` comes from
    choose   options[id] = cards whose consensus is id; accepted/rejected empty
    select   options[id] = cards whose consensus includes id; accepted/rejected empty

``contested`` (a tie, or the line's ``contested`` flag) and ``unjudged`` are in
neither set and in no option set: Filter reaches them by rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Union

MODES = ("single", "pair", "choose", "select", "match")
RULES = ("accepted", "rejected", "contested", "unjudged", "option")


def parse_labels(labels: Union[str, Iterable[Any]]) -> List[Dict[str, Any]]:
    """labels.jsonl text (or an iterable of lines or dicts) -> list of dicts."""
    if isinstance(labels, str):
        items: Iterable[Any] = labels.splitlines()
    else:
        items = labels
    out: List[Dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(item)
            continue
        line = str(item).strip()
        if line == "":
            continue
        parsed = json.loads(line)
        if not isinstance(parsed, dict) or "ref" not in parsed:
            raise ValueError(f"not a labels.jsonl line: {line[:80]}")
        out.append(parsed)
    return out


def card_state(line: Optional[Dict[str, Any]]) -> str:
    """``unjudged``, ``contested``, or ``judged``."""
    if line is None:
        return "unjudged"
    verdict = line.get("verdict")
    if verdict == "unjudged" or verdict is None:
        return "unjudged"
    if verdict == "contested" or line.get("contested") is True:
        return "contested"
    return "judged"


@dataclass
class Split:
    mode: str
    accepted: List[int] = field(default_factory=list)
    rejected: List[int] = field(default_factory=list)
    #: pair mode: which batch each accepted / rejected index is taken from
    #: ("a" = images, "b" = images_b). Every other mode: all "a".
    accepted_from: List[str] = field(default_factory=list)
    rejected_from: List[str] = field(default_factory=list)
    contested: List[int] = field(default_factory=list)
    unjudged: List[int] = field(default_factory=list)
    #: choose / select: option id -> indices
    options: Dict[str, List[int]] = field(default_factory=dict)
    #: per index: the line's verdict ("unjudged" for a ref with no line)
    verdicts: List[Any] = field(default_factory=list)
    #: per index: the labels line, or None
    lines: List[Optional[Dict[str, Any]]] = field(default_factory=list)

    def indices(self, rule: str, option_id: Optional[str] = None) -> List[int]:
        """Filter's rules."""
        if rule == "accepted":
            return list(self.accepted)
        if rule == "rejected":
            return list(self.rejected)
        if rule == "contested":
            return list(self.contested)
        if rule == "unjudged":
            return list(self.unjudged)
        if rule == "option":
            if not option_id:
                raise ValueError("rule option needs an option_id")
            return list(self.options.get(option_id, []))
        raise ValueError(f"unknown rule: {rule} (want one of {', '.join(RULES)})")


def split_labels(
    labels: Union[str, Iterable[Any]],
    refs: Sequence[str],
    mode: str,
) -> Split:
    """The split of one run's cards. ``refs[i]`` is batch index ``i``."""
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    by_ref: Dict[str, Dict[str, Any]] = {}
    for line in parse_labels(labels):
        by_ref[str(line["ref"])] = line

    s = Split(mode=mode)
    for i, ref in enumerate(refs):
        line = by_ref.get(str(ref))
        s.lines.append(line)
        s.verdicts.append("unjudged" if line is None else line.get("verdict"))
        state = card_state(line)
        if state == "unjudged":
            s.unjudged.append(i)
            continue
        if state == "contested":
            s.contested.append(i)
            continue
        assert line is not None
        verdict = line.get("verdict")
        if mode == "single":
            if verdict == "accept":
                s.accepted.append(i)
                s.accepted_from.append("a")
            elif verdict == "reject":
                s.rejected.append(i)
                s.rejected_from.append("a")
        elif mode == "pair":
            if verdict in ("a", "b"):
                other = "b" if verdict == "a" else "a"
                s.accepted.append(i)
                s.accepted_from.append(verdict)
                s.rejected.append(i)
                s.rejected_from.append(other)
        elif mode == "choose":
            if isinstance(verdict, str):
                s.options.setdefault(verdict, []).append(i)
        elif mode == "select":
            if isinstance(verdict, list):
                for oid in dict.fromkeys(v for v in verdict if isinstance(v, str)):
                    s.options.setdefault(oid, []).append(i)
        # match: no image split; Filter has contested / unjudged.
    return s


def take(batch: Sequence[Any], indices: Sequence[int]) -> List[Any]:
    """The items of ``batch`` at ``indices``, in order (a list helper; a
    tensor is sliced with ``tensor[indices]`` directly)."""
    return [batch[i] for i in indices]


# ---------------------------------------------------------------------------
# Filter: labels lines -> indices, knowing nothing about the daemon
# ---------------------------------------------------------------------------


def infer_mode(lines: Sequence[Dict[str, Any]]) -> str:
    """The mode a labels.jsonl came from, from the lines' shape (the daemon
    does not write the mode on them): a list verdict is ``select``; a
    ``counts`` tally is ``choose`` (or ``select`` with nothing chosen yet);
    an ``a`` / ``b`` verdict is ``pair``; anything else is ``single``. Only
    ``option`` depends on the difference between choose and select, and it
    matches a string verdict by equality and a list by inclusion either way."""
    has_counts = False
    for line in lines:
        verdict = line.get("verdict")
        if isinstance(verdict, list):
            return "select"
        if "counts" in line:
            has_counts = True
    if has_counts:
        return "choose"
    if any(line.get("verdict") in ("a", "b") for line in lines):
        return "pair"
    return "single"


def line_index(line: Dict[str, Any]) -> Optional[int]:
    """A line's own batch index, when it carries one (``index``, or
    ``meta.index``); None otherwise."""
    for value in (line.get("index"), (line.get("meta") or {}).get("index") if isinstance(line.get("meta"), dict) else None):
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def order_lines(lines: Sequence[Dict[str, Any]], batch: int) -> List[Dict[str, Any]]:
    """Line ``i`` is batch index ``i``; when every line carries its own index
    (``line_index``) and they are exactly ``0..batch-1``, that wins. The line
    count must equal the batch size."""
    if len(lines) != batch:
        raise ValueError(
            f"labels has {len(lines)} lines for {batch} images: wire the labels and the images "
            f"from the same Wait, Review or Load (one line per image, in batch order)"
        )
    own = [line_index(line) for line in lines]
    if all(i is not None for i in own) and sorted(own) == list(range(batch)):  # type: ignore[type-var]
        out: List[Optional[Dict[str, Any]]] = [None] * batch
        for line, i in zip(lines, own):
            out[i] = line  # type: ignore[index]
        return [ln for ln in out if ln is not None]
    return list(lines)


@dataclass
class Picked:
    """What Filter keeps: the batch ``indices`` in order, and per index the
    batch it comes from (``sides``: "a" = images, "b" = images_b; only pair
    labels under ``accepted`` / ``rejected`` ever say "b")."""

    mode: str
    indices: List[int]
    sides: List[str]


def filter_pick(
    labels: Union[str, Iterable[Any]],
    batch: int,
    rule: str,
    option_id: Optional[str] = None,
) -> Picked:
    """Filter's rules over a labels.jsonl, with the side of each pick.

    ``accepted`` / ``rejected`` (single: the verdict; pair: every card with a
    winner, both sets, ``sides`` saying which image won or lost),
    ``contested``, ``unjudged``, and ``option`` (the cards whose consensus is
    ``option_id`` in choose, or includes it in select)."""
    if rule not in RULES:
        raise ValueError(f"unknown rule: {rule} (want one of {', '.join(RULES)})")
    if rule == "option" and not (option_id or "").strip():
        raise ValueError("rule option needs an option_id (one of the queue's option ids)")
    lines = order_lines(parse_labels(labels), batch)
    refs = [f"#{i}" for i in range(batch)]
    keyed = [dict(line, ref=refs[i]) for i, line in enumerate(lines)]
    mode = infer_mode(lines)
    split = split_labels(keyed, refs, mode)
    indices = split.indices(rule, (option_id or "").strip() or None)
    if rule == "accepted":
        sides = list(split.accepted_from)
    elif rule == "rejected":
        sides = list(split.rejected_from)
    else:
        sides = ["a"] * len(indices)
    return Picked(mode=mode, indices=indices, sides=sides)


def filter_indices(
    labels: Union[str, Iterable[Any]],
    batch: int,
    rule: str,
    option_id: Optional[str] = None,
) -> List[int]:
    """``filter_pick``'s indices only."""
    return filter_pick(labels, batch, rule, option_id).indices
