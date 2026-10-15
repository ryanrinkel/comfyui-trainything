"""The API-format graphs behind examples/*.json, built on Load
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
    """Review inside the run. The empty latent stands in for KSampler's."""
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
    """Run 1: send and finish; queue it again and the queue grows."""
    g = {}
    _loads(g, 1, SPRITES, "Sprite")
    _batch(g, 13, 1, "Batch Images (stands in for VAE Decode)")
    g["20"] = {"class_type": "TrainyThingSend", "inputs": {"images": ["13", 0], **SEND, "slug": "sprites-08"},
               "_meta": {"title": "TrainyThing Send"}}
    _show(g, 21, ["20", 1], "Link")
    return g


def ab_pair():
    """Pair: A and B on one card, same seeds, two settings. Here A is the
    sprites and B the same sprites in another order (the second half first,
    so both batches are 64x64)."""
    g = {}
    _loads(g, 1, SPRITES, "A")
    _batch(g, 13, 1, "Batch A (prompt A's VAE Decode)")
    _loads(g, 51, SPRITES[6:] + SPRITES[:6], "B")
    _batch(g, 63, 51, "Batch B (prompt B's VAE Decode)")
    g["20"] = {"class_type": "TrainyThingReview", "inputs": {"images": ["13", 0], "images_b": ["63", 0], **SEND,
               "slug": "sprites-ab", "mode": "pair", "prompt": "Which one?", "feedback": "optional", **WAIT},
               "_meta": {"title": "TrainyThing Review (pair)"}}
    _save(g, 30, ["20", 0], "trainything/ab-winners", "Save the winners (A or B)")
    _save(g, 31, ["20", 1], "trainything/ab-losers", "Save the losers")
    _show(g, 32, ["20", 6], "Notes (with the side chosen)")
    _show(g, 33, ["20", 5], "Summary (preferences.jsonl is in output/trainything/sprites-ab/)")
    return g


REASONS = "hair:Hair colour, scarf:Scarf, hands:Hands, ok:Fine"


def choose_reasons():
    """The reasons, counted: choose with four reason tags; the labels
    carry the counts; Filter option hair -> the images with the hair problem."""
    g = {}
    _loads(g, 1, SPRITES, "Sprite")
    _batch(g, 13, 1, "Batch Images (stands in for VAE Decode)")
    g["20"] = {"class_type": "TrainyThingReview", "inputs": {"images": ["13", 0], **SEND, "slug": "sprites-reasons",
               "mode": "choose", "options": REASONS, "prompt": "What is wrong with it?", "feedback": "optional", **WAIT},
               "_meta": {"title": "TrainyThing Review (choose: the reasons)"}}
    g["21"] = {"class_type": "TrainyThingFilter", "inputs": {"images": ["20", 3], "labels": ["20", 4], "rule": "option",
               "option_id": "hair"}, "_meta": {"title": "TrainyThing Filter (option hair)"}}
    _save(g, 22, ["21", 0], "trainything/reasons-hair", "Save the hair problems")
    _show(g, 23, ["21", 3], "How many have the hair problem")
    _show(g, 24, ["20", 5], "Summary (the counts per reason)")
    _show(g, 25, ["20", 6], "Notes, by reason")
    return g


def two_runs_load():
    """Run 2: Load reads sprites-08 back; Filter accepted -> the keepers
    (Save Image stands in for VAE Encode -> KSampler); the summary and the
    keepers' meta shown, to rewrite the prompt and reuse the seeds."""
    g = {}
    g["1"] = {"class_type": "TrainyThingLoad", "inputs": {"slug": "sprites-08", "wait": False, "timeout_minutes": 45.0},
              "_meta": {"title": "TrainyThing Load"}}
    g["2"] = {"class_type": "TrainyThingFilter", "inputs": {"images": ["1", 2], "labels": ["1", 3], "rule": "accepted",
              "option_id": ""}, "_meta": {"title": "TrainyThing Filter (accepted)"}}
    _save(g, 3, ["2", 0], "trainything/run2-keepers", "Save the keepers (stands in for VAE Encode -> KSampler)")
    _show(g, 4, ["1", 4], "Summary (rewrite the prompt from the notes)")
    _show(g, 5, ["1", 6], "Keepers' meta (seeds, prompts)")
    _show(g, 6, ["2", 3], "How many keepers")
    return g


EXAMPLES = {"gate": gate, "two-runs-send": two_runs_send, "two-runs-load": two_runs_load, "ab-pair": ab_pair,
            "choose-reasons": choose_reasons}
