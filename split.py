"""labels.jsonl -> batch indices (PLAN-COMFY §4.3, §4.5). Pure.

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
        """Filter's rules (§4.5)."""
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
    """The §4.3 split of one run's cards. ``refs[i]`` is batch index ``i``."""
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
