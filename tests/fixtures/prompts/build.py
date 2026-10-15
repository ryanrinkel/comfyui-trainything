"""Writes the API-format prompts the ComfyUI tests submit (``python build.py``).

Every graph starts from the twelve sprites (``tests/fixtures/sprites``, copied
into ComfyUI's input folder) through Load Image and Batch Images, so no model
is needed; Batch Images resizes the odd sizes to the first one's 64x64.
Each output of the TrainyThing node goes to something whose ``ui`` lands in
``/history``: Save Image for the image sockets, Save Latent for the latent,
Preview as Text (core's ``PreviewAny``) for the strings and the index list.
The tests patch ``slug``, ``labels``, ``timeout_minutes`` and ``on_timeout``.

The Load tests add ``send`` (Send only, the batch also saved as it went in, so Load's
images can be compared with it) and ``load-filter`` (Load, every socket
observed, and one Filter per rule over Load's images and labels).

The pair and options tests add ``ab-pair`` (Review in pair mode over A = the sprites and B =
the sprites reversed, a latent wired, both batches saved as they went in, and
Filter accepted / rejected with both batches and accepted with A only) and
``options`` (Review in choose mode with the four reason tags, one Filter per
tag over Review's images and labels, and Filter accepted; the tests switch
it to select).
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
SPRITES = [f"s{i:03d}.png" for i in range(1, 13)]


def sources(g, latent=True):
    for i, name in enumerate(SPRITES, start=1):
        g[str(i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
    g["13"] = {"class_type": "BatchImagesNode", "inputs": {f"images.image{i}": [str(i + 1), 0] for i in range(12)}}
    if latent:
        g["14"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 12}}


SEND = {
    "title": "K1 sprites",
    "slug": "k1-gate",
    "prompt": "Keep this sprite?",
    "intro": "",
    "mode": "single",
    "options": "",
    "feedback": "on_reject",
    "identity": "name_email",
    "order": "fixed",
    "min_reviews": 1,
    "public": False,
    "project": "comfyui-k1",
    "labels": "\n".join(f"sprite {i:02d}" for i in range(12)),
    "meta": '{"source": "k1-test"}',
}
WAIT = {"timeout_minutes": 10.0, "on_timeout": "partial"}


def observers(g, src):
    """Wire the nine Wait sockets (0..8) of node ``src``."""
    g["30"] = {"class_type": "SaveImage", "inputs": {"images": [src, 0], "filename_prefix": "k1/accepted"}}
    g["31"] = {"class_type": "SaveImage", "inputs": {"images": [src, 1], "filename_prefix": "k1/rejected"}}
    g["32"] = {"class_type": "SaveLatent", "inputs": {"samples": [src, 2], "filename_prefix": "k1/accepted_latent"}}
    g["33"] = {"class_type": "PreviewAny", "inputs": {"source": [src, 4]}}  # labels
    g["34"] = {"class_type": "PreviewAny", "inputs": {"source": [src, 5]}}  # summary
    g["35"] = {"class_type": "PreviewAny", "inputs": {"source": [src, 6]}}  # notes
    g["36"] = {"class_type": "PreviewAny", "inputs": {"source": [src, 7]}}  # accepted_meta
    g["37"] = {"class_type": "PreviewAny", "inputs": {"source": [src, 8]}}  # accepted_indices: once per keeper
    g["38"] = {"class_type": "ImageFromBatch", "inputs": {"image": [src, 3], "batch_index": [src, 8], "length": 1}}
    g["39"] = {"class_type": "SaveImage", "inputs": {"images": ["38", 0], "filename_prefix": "k1/by_index"}}


def gate():
    """on sprites: Review with the batch and a latent."""
    g = {}
    sources(g)
    g["20"] = {"class_type": "TrainyThingReview", "inputs": {"images": ["13", 0], "latent": ["14", 0], **SEND, **WAIT}}
    observers(g, "20")
    g["40"] = {"class_type": "PreviewAny", "inputs": {"source": ["20", 9]}}  # link
    g["41"] = {"class_type": "PreviewAny", "inputs": {"source": ["20", 10]}}  # slug
    return g


def send_wait():
    """Send and Wait apart: the queue handle on the custom socket."""
    g = {}
    sources(g)
    g["20"] = {"class_type": "TrainyThingSend", "inputs": {"images": ["13", 0], **SEND}}
    g["21"] = {"class_type": "TrainyThingWait", "inputs": {"queue": ["20", 0], "images": ["13", 0], "latent": ["14", 0], **WAIT}}
    observers(g, "21")
    g["40"] = {"class_type": "PreviewAny", "inputs": {"source": ["20", 1]}}  # link
    g["41"] = {"class_type": "PreviewAny", "inputs": {"source": ["20", 2]}}  # slug
    return g


def send_pair():
    """'s shape: images and images_b (the batch reversed) into a pair Send."""
    g = {}
    sources(g, latent=False)
    for i, name in enumerate(reversed(SPRITES), start=51):
        g[str(i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
    g["63"] = {"class_type": "BatchImagesNode", "inputs": {f"images.image{i}": [str(51 + i), 0] for i in range(12)}}
    g["20"] = {"class_type": "TrainyThingSend", "inputs": {
        "images": ["13", 0], "images_b": ["63", 0], **dict(SEND, mode="pair", prompt="Which one?", feedback="optional", slug="k1-pair", labels="")}}
    return g


def send():
    """Send only, and Save Image of the batch Send was given."""
    g = {}
    sources(g, latent=False)
    g["20"] = {"class_type": "TrainyThingSend", "inputs": {"images": ["13", 0], **dict(SEND, slug="k2-send")}}
    g["42"] = {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "k2/sent"}}
    return g


FILTER_RULES = {"80": ("accepted", ""), "81": ("rejected", ""), "82": ("unjudged", ""), "83": ("option", "hair"),
                "84": ("contested", "")}


def load_filter():
    """run 2: Load by slug, every socket observed; Filter per rule over
    Load's images and labels (80 also slices a latent: EmptyLatentImage 79,
    whose batch_size the tests set to the queue's size). Per Filter 8k:
    Save Image 9k, Preview as Text of indices 10k and count 11k."""
    g = {}
    g["70"] = {"class_type": "TrainyThingLoad", "inputs": {"slug": "k2-send", "wait": False, "timeout_minutes": 45.0}}
    g["71"] = {"class_type": "SaveImage", "inputs": {"images": ["70", 2], "filename_prefix": "k2/load_images"}}
    g["72"] = {"class_type": "SaveImage", "inputs": {"images": ["70", 0], "filename_prefix": "k2/load_accepted"}}
    g["73"] = {"class_type": "SaveImage", "inputs": {"images": ["70", 1], "filename_prefix": "k2/load_rejected"}}
    for nid, slot in (("74", 3), ("75", 4), ("76", 5), ("77", 6), ("78", 7)):  # labels summary notes meta indices
        g[nid] = {"class_type": "PreviewAny", "inputs": {"source": ["70", slot]}}
    g["79"] = {"class_type": "EmptyLatentImage", "inputs": {"width": 64, "height": 64, "batch_size": 12}}
    for nid, (rule, option) in FILTER_RULES.items():
        k = nid[1]
        inputs = {"images": ["70", 2], "labels": ["70", 3], "rule": rule, "option_id": option}
        if nid == "80":
            inputs["latent"] = ["79", 0]
        g[nid] = {"class_type": "TrainyThingFilter", "inputs": inputs}
        g[f"9{k}"] = {"class_type": "SaveImage", "inputs": {"images": [nid, 0], "filename_prefix": f"k2/filter_{rule}"}}
        g[f"10{k}"] = {"class_type": "PreviewAny", "inputs": {"source": [nid, 2]}}
        g[f"11{k}"] = {"class_type": "PreviewAny", "inputs": {"source": [nid, 3]}}
    g["120"] = {"class_type": "SaveLatent", "inputs": {"samples": ["80", 1], "filename_prefix": "k2/filter_latent"}}
    return g


def _b_side(g):
    for i, name in enumerate(reversed(SPRITES), start=51):
        g[str(i)] = {"class_type": "LoadImage", "inputs": {"image": name}}
    g["63"] = {"class_type": "BatchImagesNode", "inputs": {f"images.image{i}": [str(51 + i), 0] for i in range(12)}}
    # Batch Images sizes the batch by its first image (s012, 512x512): scaled
    # to A's 64x64 here, so the winners compare pixel for pixel.
    g["64"] = {"class_type": "ImageScale", "inputs": {"image": ["63", 0], "upscale_method": "nearest-exact",
                                                      "width": 64, "height": 64, "crop": "disabled"}}


def _filter(g, nid, inputs, prefix):
    """Filter ``nid`` (8k) with Save Image 9k, indices 10k and count 11k."""
    k = nid[1]
    g[nid] = {"class_type": "TrainyThingFilter", "inputs": inputs}
    g[f"9{k}"] = {"class_type": "SaveImage", "inputs": {"images": [nid, 0], "filename_prefix": f"k3/{prefix}"}}
    g[f"10{k}"] = {"class_type": "PreviewAny", "inputs": {"source": [nid, 2]}}
    g[f"11{k}"] = {"class_type": "PreviewAny", "inputs": {"source": [nid, 3]}}


def ab_pair():
    """Pair: twelve pairs on one Review; Filter over its labels with both
    batches (80 accepted, with a latent; 81 rejected) and with A only (82)."""
    g = {}
    sources(g)
    _b_side(g)
    g["20"] = {"class_type": "TrainyThingReview", "inputs": {
        "images": ["13", 0], "images_b": ["64", 0], "latent": ["14", 0],
        **dict(SEND, title="K3 A/B", slug="k3-ab", mode="pair", prompt="Which one?", feedback="optional"), **WAIT}}
    observers(g, "20")
    g["42"] = {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "k3/a"}}
    g["43"] = {"class_type": "SaveImage", "inputs": {"images": ["64", 0], "filename_prefix": "k3/b"}}
    _filter(g, "80", {"images": ["13", 0], "images_b": ["64", 0], "latent": ["14", 0], "labels": ["20", 4],
                      "rule": "accepted", "option_id": ""}, "filter_ab_accepted")
    _filter(g, "81", {"images": ["13", 0], "images_b": ["64", 0], "labels": ["20", 4],
                      "rule": "rejected", "option_id": ""}, "filter_ab_rejected")
    _filter(g, "82", {"images": ["13", 0], "labels": ["20", 4], "rule": "accepted", "option_id": ""},
            "filter_a_accepted")
    g["120"] = {"class_type": "SaveLatent", "inputs": {"samples": ["80", 1], "filename_prefix": "k3/filter_latent"}}
    return g


TAG_IDS = ["hair", "scarf", "hands", "ok"]
TAGS = "hair:Hair colour, scarf:Scarf, hands:Hands, ok:Fine"


def options():
    """"the reasons, counted": Review in choose mode with four reason
    tags; Filter 80..83 per tag, 84 accepted (empty by design)."""
    g = {}
    sources(g)
    g["20"] = {"class_type": "TrainyThingReview", "inputs": {
        "images": ["13", 0], "latent": ["14", 0],
        **dict(SEND, title="K3 reasons", slug="k3-choose", mode="choose", options=TAGS, prompt="What is wrong with it?",
               feedback="on_reject"), **WAIT}}
    observers(g, "20")
    g["42"] = {"class_type": "SaveImage", "inputs": {"images": ["13", 0], "filename_prefix": "k3/sent"}}
    for k, tag in enumerate(TAG_IDS):
        _filter(g, f"8{k}", {"images": ["20", 3], "labels": ["20", 4], "rule": "option", "option_id": tag}, f"filter_{tag}")
    _filter(g, "84", {"images": ["20", 3], "labels": ["20", 4], "rule": "accepted", "option_id": ""}, "filter_accepted")
    return g


if __name__ == "__main__":
    for name, fn in {"gate": gate, "send-wait": send_wait, "send-pair": send_pair, "send": send,
                     "load-filter": load_filter, "ab-pair": ab_pair, "options": options}.items():
        (HERE / f"{name}.json").write_text(json.dumps(fn(), indent=1) + "\n")
        print(f"wrote {name}.json")
