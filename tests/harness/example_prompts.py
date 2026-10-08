"""The API-format graphs behind examples/*.json (PLAN-COMFY §5), built on Load
Image and the twelve sprites so they run with no model. make_examples.py turns
them into canvas workflows through the frontend itself."""

from __future__ import annotations

SPRITES = [f"s{i:03d}.png" for i in range(1, 13)]


def _loads(g, first_id, names, title):
    for k, name in enumerate(names):
        g[str(first_id + k)] = {"class_type": "LoadImage", "inputs": {"image": name}, "_meta": {"title": f"{title} {k + 1:02d}"}}


def _batch(g, nid, first_id, title):
    g[str(nid)] = {"class_type": "BatchImagesNode", "_meta": {"title": title},
                   "inputs": {f"images.image{k}": [str(first_id + k), 0] for k in range(12)}}


SEND = {
    "title": "Sprites",
    "prompt": "Keep this sprite?",
    "intro": "",
    "mode": "single",
    "options": "",
    "feedback": "on_reject",
    "identity": "name_email",
    "order": "coverage",
    "min_reviews": 1,
    "public": False,
    "project": "comfyui",
    "labels": "",
    "meta": "",
}
WAIT = {"timeout_minutes": 45.0, "on_timeout": "partial"}


def _show(g, nid, src, title):
    g[str(nid)] = {"class_type": "PreviewAny", "inputs": {"source": src}, "_meta": {"title": title}}


def _save(g, nid, src, prefix, title):
    g[str(nid)] = {"class_type": "SaveImage", "inputs": {"images": src, "filename_prefix": prefix}, "_meta": {"title": title}}


def gate():
    """§5.1: review inside the run. The empty latent stands in for KSampler's."""
    g = {}
    _loads(g, 1, SPRITES, "Sprite")
    _batch(g, 13, 1, "Batch Images (stands in for VAE Decode)")
    g["14"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 12},
               "_meta": {"title": "Empty Latent (stands in for KSampler's latent)"}}
    g["20"] = {"class_type": "TrainyThingReview", "inputs": {"images": ["13", 0], "latent": ["14", 0], **SEND,
               "slug": "sprites-gate", **WAIT}, "_meta": {"title": "TrainyThing Review"}}
    _save(g, 30, ["20", 0], "trainything/accepted", "Save the keepers")
    _save(g, 31, ["20", 1], "trainything/rejected", "Save the rejects")
    _show(g, 32, ["20", 6], "Rejection notes")
    _show(g, 33, ["20", 5], "Summary")
    _show(g, 34, ["20", 7], "Keepers' meta (seeds, prompts)")
    return g


def two_runs_send():
    """§5.2 run 1: send and finish; queue it again and the queue grows."""
    g = {}
    _loads(g, 1, SPRITES, "Sprite")
    _batch(g, 13, 1, "Batch Images (stands in for VAE Decode)")
    g["20"] = {"class_type": "TrainyThingSend", "inputs": {"images": ["13", 0], **SEND, "slug": "sprites-08"},
               "_meta": {"title": "TrainyThing Send"}}
    _show(g, 21, ["20", 1], "Link")
    return g


def ab_pair():
    """§5.3: A and B on one card. Here A is the sprites and B the same reversed."""
    g = {}
    _loads(g, 1, SPRITES, "A")
    _batch(g, 13, 1, "Batch A (prompt A's VAE Decode)")
    _loads(g, 51, list(reversed(SPRITES)), "B")
    _batch(g, 63, 51, "Batch B (prompt B's VAE Decode)")
    g["20"] = {"class_type": "TrainyThingReview", "inputs": {"images": ["13", 0], "images_b": ["63", 0], **SEND,
               "slug": "sprites-ab", "mode": "pair", "prompt": "Which one?", "feedback": "optional", **WAIT},
               "_meta": {"title": "TrainyThing Review (pair)"}}
    _save(g, 30, ["20", 0], "trainything/ab-winners", "Save the winners")
    _show(g, 33, ["20", 5], "Summary")
    return g


EXAMPLES = {"gate": gate, "two-runs-send": two_runs_send, "ab-pair": ab_pair}
