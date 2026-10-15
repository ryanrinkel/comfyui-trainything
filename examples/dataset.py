#!/usr/bin/env python3
"""A dataset of keepers with captions: a run's folder ->
<ref>.png + <ref>.txt pairs, the convention LoRA trainers read.

    python examples/dataset.py ComfyUI/output/trainything/<slug> [--out DIR] [--rule accepted|rejected|option:<id>]

Caption: the card's meta.prompt_text (the graph walk's positive prompt, or your
meta input), else the card's label, else the queue's prompt. In pair mode the
meta describes A; a B winner uses meta.prompt_text_b if your meta set one.
Standard library only; contested and unjudged cards are never taken."""
import argparse, json, os, re, shutil
from pathlib import Path

SIDE = {"accepted": {"accept": 0, "a": 0, "b": 1}, "rejected": {"reject": 0, "a": 1, "b": 0}}


def side(verdict, rule):
    """Which image of the card the rule takes (0 = A or the only one), or None."""
    if rule.startswith("option:"):
        oid = rule[len("option:"):]
        return 0 if verdict == oid or (isinstance(verdict, list) and oid in verdict) else None
    return SIDE[rule].get(verdict) if isinstance(verdict, str) else None


def build(folder, out, rule="accepted"):
    doc = json.loads(Path(folder, "results.json").read_text(encoding="utf-8"))
    labels = Path(folder, "labels.jsonl")
    raw = labels.read_text(encoding="utf-8").splitlines() if labels.exists() else []
    lines = {l["ref"]: l for l in map(json.loads, filter(str.strip, raw))}
    os.makedirs(out, exist_ok=True)
    written = []
    for card in doc.get("cards") or []:
        s = card.get("summary") or {}
        line = lines.get(card.get("ref"), {"verdict": s.get("consensus"), "contested": s.get("contested")})
        items = [it for it in card.get("items") or [] if it.get("kind") == "image"]
        items.sort(key=lambda it: str(it.get("ref", "")).endswith("-b"))
        k = None if line.get("contested") else side(line.get("verdict"), rule)
        if k is None or k >= len(items):
            continue
        meta = card.get("meta") or {}
        caption = (meta.get("prompt_text_b") if k == 1 else None) or (meta.get("prompt_text") if k == 0 else None) \
            or card.get("label") or card.get("prompt") or (doc.get("queue") or {}).get("prompt") or ""
        name = re.sub(r"[^\w.-]", "_", str(card.get("ref") or card.get("id")))
        shutil.copyfile(os.path.join(folder, items[k]["src"]), os.path.join(out, name + ".png"))
        Path(out, name + ".txt").write_text(str(caption).strip() + "\n", encoding="utf-8")
        written.append(name)
    return written


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("folder", help="output/trainything/<slug>, as Review, Wait or Load wrote it")
    ap.add_argument("--out", help="where the pairs go (default: <folder>/dataset-<rule>)")
    ap.add_argument("--rule", default="accepted", help="accepted (default), rejected, or option:<id>")
    a = ap.parse_args()
    if a.rule not in SIDE and not a.rule.startswith("option:"):
        ap.error(f"--rule: want accepted, rejected or option:<id>, not {a.rule}")
    out = a.out or os.path.join(a.folder, "dataset-" + a.rule.replace(":", "-"))
    print(f"{len(build(a.folder, out, a.rule))} pairs in {out}")
