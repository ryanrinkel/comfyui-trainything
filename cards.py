"""A batch of images -> cards in a queue (PLAN-COMFY §4.2). No ComfyUI import.

    run = make_run_id(node_id)                       # "<time>-n<node id>-<4 hex>"
    handle = send_batch(client, images, queue_config, run=run, prompt=..., ...)

Each card is ``{ref: "<run>-<index>", kind: "image", src: "blob:<sha>",
label?, meta}``; in pair mode ``{ref, label?, meta, items: [A, B]}``.
``meta`` is the walk's facts, then ``index`` and ``run``, then the user's
``meta`` input on top: the user wins.

The queue is created when the slug is new (or empty / ``random``), appended to
when a queue with that slug is open, and a closed one is an error that names
it, never a silent new queue. An append checks the queue against the widgets:
another mode is an error, other options are the queue's (with a console
line), and a queue with no ``min_reviews`` (one the CLI, the dashboard or the
MCP server made) gets the widget's, or the wait could only end on a close or
the timeout.
"""

from __future__ import annotations

import json
import re
import secrets
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence

from .png import encode_png, iter_batch
from .trainything_client import AsyncTrainyThingClient, ClientError, TrainyThingClient

SLUG_MIN = 3
SLUG_MAX = 40
RANDOM_SLUG = "random"


def say(line: str) -> None:
    """One console line, the way ComfyUI's own nodes print."""
    print(f"[TrainyThing] {line}", flush=True)


class QueueModeError(ValueError):
    """An append whose mode is not the queue's: a ``choose`` batch in a
    ``single`` queue would be judged by the wrong question."""


class QueueClosedError(ClientError):
    def __init__(self, slug: str) -> None:
        super().__init__(
            f"queue {slug} is closed; reopen it on the dashboard or pick another slug "
            f"(a closed queue never becomes a new one silently)",
            409,
            None,
        )
        self.slug = slug


def normalize_slug(text: str) -> str:
    """``normalizeSlug`` in src/core/slug.ts: lowercase, accents stripped,
    every run of anything else one dash, trimmed, at most 40."""
    s = unicodedata.normalize("NFKD", text)
    s = re.sub("[̀-ͯ]", "", s).lower()
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")
    if len(s) > SLUG_MAX:
        s = s[:SLUG_MAX].rstrip("-")
    return s


def make_run_id(node_id: Any, now: Optional[float] = None) -> str:
    """``<UTC time to the millisecond>-n<node id>-<4 hex>``. The random
    suffix keeps two runs of one node in the same millisecond apart (ComfyUI
    runs a node once per item of a list input; the daemon does not enforce
    unique refs)."""
    t = time.time() if now is None else now
    stamp = time.strftime("%Y%m%dT%H%M%S", time.gmtime(t)) + f"{int((t % 1) * 1000):03d}"
    node = re.sub(r"[^A-Za-z0-9]+", "_", str(node_id)) or "x"
    return f"{stamp}-n{node}-{secrets.token_hex(2)}"


def card_ref(run: str, index: int) -> str:
    return f"{run}-{index}"


def parse_labels_input(text: Optional[str], count: int) -> List[Optional[str]]:
    """The ``labels`` widget: one line per image; missing lines and blank ones
    mean no label. More lines than images is an error."""
    if text is None or text.strip() == "":
        return [None] * count
    lines = text.splitlines()
    while lines and lines[-1].strip() == "":
        lines.pop()
    if len(lines) > count:
        raise ValueError(f"labels has {len(lines)} lines for {count} images")
    out: List[Optional[str]] = [ln.strip() or None for ln in lines]
    return out + [None] * (count - len(out))


def parse_meta_input(text: Optional[str], count: int) -> List[Dict[str, Any]]:
    """The ``meta`` widget: empty; one JSON object for all images; a JSON
    array of ``count`` objects; or JSONL, one object per image."""
    if text is None or text.strip() == "":
        return [{} for _ in range(count)]
    try:
        parsed = json.loads(text)
    except ValueError:
        parsed = None
        rows = [ln for ln in text.splitlines() if ln.strip() != ""]
        try:
            parsed = [json.loads(ln) for ln in rows]
        except ValueError as err:
            raise ValueError(f"meta is neither JSON nor JSON lines: {err}") from None
    if isinstance(parsed, dict):
        return [dict(parsed) for _ in range(count)]
    if isinstance(parsed, list):
        if not all(isinstance(m, dict) for m in parsed):
            raise ValueError("meta: every entry must be a JSON object")
        if len(parsed) == 1:
            return [dict(parsed[0]) for _ in range(count)]
        if len(parsed) != count:
            raise ValueError(f"meta has {len(parsed)} objects for {count} images (want 1 or {count})")
        return [dict(m) for m in parsed]
    raise ValueError("meta must be a JSON object, an array of objects, or JSON lines")


#: §10 question 7, per mode: what the reviewer is asked to do with the prompt.
INTRO_ASK = {
    "single": "Reject anything that misses it.",
    "pair": "Pick the one that matches it better.",
    "choose": "Pick the answer that fits each one best.",
    "select": "Tick every answer that applies to each one.",
}


def default_intro(prompt_text: Optional[str], mode: str = "single") -> Optional[str]:
    """§10 question 7: the intro when the widget is left empty, worded for
    the mode (single rejects, pair picks one of two, choose picks one answer,
    select ticks any). No prompt found: no intro."""
    if not prompt_text or not str(prompt_text).strip():
        return None
    ask = INTRO_ASK.get(mode, INTRO_ASK["single"])
    return f"Generated from: {str(prompt_text).strip()}. {ask}"


def build_cards(
    shas: Sequence[str],
    run: str,
    labels: Optional[Sequence[Optional[str]]] = None,
    metas: Optional[Sequence[Dict[str, Any]]] = None,
    facts: Optional[Dict[str, Any]] = None,
    shas_b: Optional[Sequence[str]] = None,
) -> List[Dict[str, Any]]:
    """N blob shas -> N card objects for ``POST /cards``. ``shas_b`` makes
    pair cards (A from ``shas``, B from ``shas_b``, one card per index)."""
    n = len(shas)
    if shas_b is not None and len(shas_b) != n:
        raise ValueError(f"images_b has {len(shas_b)} images for {n} in images; pair mode needs the same batch size")
    cards: List[Dict[str, Any]] = []
    for i, sha in enumerate(shas):
        ref = card_ref(run, i)
        meta: Dict[str, Any] = dict(facts or {})
        meta["index"] = i
        meta["run"] = run
        if metas is not None:
            meta.update(metas[i])
        card: Dict[str, Any] = {"ref": ref}
        label = labels[i] if labels is not None else None
        if label:
            card["label"] = label
        card["meta"] = meta
        if shas_b is None:
            card["kind"] = "image"
            card["src"] = f"blob:{sha}"
        else:
            card["items"] = [
                {"kind": "image", "src": f"blob:{sha}", "ref": f"{ref}-a"},
                {"kind": "image", "src": f"blob:{shas_b[i]}", "ref": f"{ref}-b"},
            ]
        cards.append(card)
    return cards


def _options_key(options: Any) -> List[Any]:
    return [(o.get("id"), o.get("label")) for o in options or [] if isinstance(o, dict)]


def reconcile_queue(client: TrainyThingClient, queue: Dict[str, Any], body: Dict[str, Any]) -> Dict[str, Any]:
    """An append: check the open queue against the widgets before any upload.

    - ``mode`` differs: ``QueueModeError`` naming both.
    - ``options`` differ: the queue's stand (the daemon keeps them), with a
      console line.
    - The queue has no ``completion.min_reviews_per_card``: PATCH it to the
      widget's (``completion`` replaces the whole object, so a deadline is
      kept); if the daemon refuses, a console line says the wait can only end
      on a close or the timeout. A queue with its own value keeps it, with a
      console line when it differs.

    Returns the queue row (the patched one when it was patched)."""
    name = queue.get("share_token") or queue.get("id")
    want_mode = body.get("mode")
    if want_mode is not None and queue.get("mode") and queue.get("mode") != want_mode:
        raise QueueModeError(
            f"queue {name} is a {queue.get('mode')} queue and this node's mode is {want_mode}: "
            f"set mode to {queue.get('mode')} or pick another slug (cards of one mode never go into another's queue)"
        )
    want_options = body.get("options")
    if want_options and _options_key(want_options) != _options_key(queue.get("options")):
        have = ", ".join(str(o.get("id")) for o in queue.get("options") or []) or "none"
        say(f"queue {name} keeps its own options ({have}); the options widget is ignored on an append")
    want_min = (body.get("completion") or {}).get("min_reviews_per_card")
    if want_min is None:
        return queue
    completion = dict(queue.get("completion") or {})
    have_min = completion.get("min_reviews_per_card")
    if have_min is not None:
        if int(have_min) != int(want_min):
            say(f"queue {name} keeps its own min_reviews {have_min} (the widget says {want_min})")
        return queue
    completion["min_reviews_per_card"] = int(want_min)
    try:
        patched = client.patch_queue(queue["id"], {"completion": completion})
    except ClientError as err:
        say(f"queue {name} has no min_reviews and the daemon would not set it ({err}); "
            f"the wait can only end on a close or the timeout")
        return queue
    say(f"queue {name} had no min_reviews; set it to {want_min} so the wait can end")
    return patched.get("queue") or queue


def ensure_queue(client: TrainyThingClient, config: Dict[str, Any]) -> Dict[str, Any]:
    """Create or append. ``config`` is the ``POST /api/queues`` body; its
    ``slug`` decides: empty or ``random`` -> create; an open queue under the
    (normalized) slug -> append (``reconcile_queue``); a closed one ->
    ``QueueClosedError``; none -> create under it. Returns ``{queue,
    local_url, share_url, created}``."""
    body = {k: v for k, v in config.items() if v is not None and v != ""}
    slug = str(body.get("slug", "")).strip()
    if slug and slug.lower() != RANDOM_SLUG:
        wanted = normalize_slug(slug)
        found = client.find_queue(wanted) if len(wanted) >= SLUG_MIN else None
        if found is not None:
            queue = found["queue"]
            if queue.get("status") == "closed":
                raise QueueClosedError(queue.get("share_token") or wanted)
            queue = reconcile_queue(client, queue, body)
            return {
                "queue": queue,
                "local_url": found.get("local_url"),
                "share_url": found.get("share_url"),
                "created": False,
            }
    created = client.create_queue(body)
    return {
        "queue": created["queue"],
        "local_url": created.get("local_url"),
        "share_url": created.get("share_url"),
        "created": True,
    }


def share_queue(client: TrainyThingClient, q: Dict[str, Any]) -> None:
    """``public``: start the tunnel for the queue ``q`` (from
    ``ensure_queue``) before any card is added. A failure (no cloudflared,
    its 60 s timeout) is one console line and the local link, never an
    error: a retry would add the batch twice."""
    name = q["queue"].get("share_token") or q["queue"]["id"]
    try:
        shared = client.share(q["queue"]["id"])
    except ClientError as err:
        say(f"public: could not share {name} ({err}); the link is the local one, this machine only")
        return
    if shared.get("share_url"):
        q["share_url"] = shared["share_url"]


def encode_batch(images: Any, prompt: Any = None, extra_pnginfo: Optional[Dict[str, Any]] = None) -> List[bytes]:
    return [encode_png(img, prompt=prompt, extra_pnginfo=extra_pnginfo) for img in iter_batch(images)]


def send_batch(
    client: TrainyThingClient,
    images: Any,
    queue_config: Dict[str, Any],
    run: str,
    *,
    images_b: Any = None,
    labels: Optional[str] = None,
    meta: Optional[str] = None,
    facts: Optional[Dict[str, Any]] = None,
    prompt: Any = None,
    extra_pnginfo: Optional[Dict[str, Any]] = None,
    public: bool = False,
) -> Dict[str, Any]:
    """Send's body, blocking: resolve the queue (before any upload, so a
    closed one or another mode fails fast), share it when ``public`` (also
    before any upload; a failure falls back to the local link), encode and
    upload each image, add the cards. ``prompt`` / ``extra_pnginfo`` None
    writes no ``prompt`` / ``workflow`` chunk (``--disable-metadata``).

    Returns the queue handle §4.2 describes: ``{base_url, slug, queue_id,
    link, share_url, mode, refs, added, total, created}``."""
    pngs = encode_batch(images, prompt, extra_pnginfo)
    pngs_b = encode_batch(images_b, prompt, extra_pnginfo) if images_b is not None else None
    n = len(pngs)
    if pngs_b is not None and len(pngs_b) != n:
        raise ValueError(f"images_b has {len(pngs_b)} images for {n} in images")
    label_list = parse_labels_input(labels, n)
    meta_list = parse_meta_input(meta, n)

    q = ensure_queue(client, queue_config)
    queue = q["queue"]
    if public:
        share_queue(client, q)
    shas = [client.upload_blob(p, "image/png")["sha256"] for p in pngs]
    shas_b = [client.upload_blob(p, "image/png")["sha256"] for p in pngs_b] if pngs_b is not None else None
    cards = build_cards(shas, run, label_list, meta_list, facts, shas_b)
    added = client.add_cards(queue["id"], cards)
    return {
        "base_url": client.base_url,
        "slug": queue.get("share_token"),
        "queue_id": queue["id"],
        "link": q.get("share_url") or q.get("local_url"),
        "local_url": q.get("local_url"),
        "share_url": q.get("share_url"),
        "mode": queue.get("mode"),
        "refs": [c["ref"] for c in cards],
        "added": added.get("added"),
        "total": added.get("total"),
        "created": q["created"],
    }


async def send_batch_async(client: AsyncTrainyThingClient, *args: Any, **kwargs: Any) -> Dict[str, Any]:
    import asyncio

    return await asyncio.to_thread(send_batch, client.sync, *args, **kwargs)
