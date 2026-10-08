"""What Send, Wait and Review share (PLAN-COMFY §4.2, §4.3, §4.6, §4.7).

Nothing here imports ComfyUI at module level: torch, ``comfy`` and
``folder_paths`` are imported inside the functions that need them, so the pure
parts (the queue config, the notes, the meta, the labels for a run) are tested
in a plain venv.

    client  = await connect()                        # §4.7: health, one auto-start, the one line
    config  = queue_config(title=..., mode=..., ...) # the POST /api/queues body
    handle  = await send(client, images, config, ...)
    status  = await wait_until_done(client, handle, timeout_s, interrupted=...)
    run     = await collect(client, handle)          # labels, results.json, summary for this run
    outputs = split_outputs(run, images, latent, images_b)
    folder  = await write_run_folder(client, handle["slug"], root)
"""

from __future__ import annotations

import asyncio
import io as _io
import json
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Optional, Sequence, Tuple

from ..cards import default_intro, make_run_id, send_batch
from ..split import Split, split_labels
from ..trainything_client import (
    AsyncTrainyThingClient,
    ClientError,
    TrainyThingClient,
    connect_async,
    is_local_url,
)
from ..walk import walk_graph

log = logging.getLogger("trainything")

CATEGORY = "TrainyThing"
QUEUE_TYPE = "TRAINYTHING_QUEUE"
MODES = ["single", "pair", "choose", "select"]
FEEDBACKS = ["on_reject", "optional", "off", "required"]
IDENTITIES = ["name_email", "anonymous"]
ORDERS = ["coverage", "fixed"]
ON_TIMEOUT = ["partial", "block"]
#: §4.3: poll /status this often.
POLL_S = 2.0
#: How often the wait looks at ComfyUI's interrupt flag between polls.
INTERRUPT_CHECK_S = 0.25
DEFAULT_TIMEOUT_MIN = 45.0
#: The folder under ComfyUI's output directory (§4.3 "The run's folder").
OUTPUT_SUBDIR = "trainything"
LOAD_LATER = "TrainyThing Load will read it later"

#: The outputs of Wait, in socket order (§4.6); Review adds link and slug.
WAIT_OUTPUTS = (
    "accepted",
    "rejected",
    "accepted_latent",
    "images",
    "labels",
    "summary",
    "notes",
    "accepted_meta",
    "accepted_indices",
)


def say(line: str) -> None:
    """One console line, the way ComfyUI's own nodes print."""
    print(f"[TrainyThing] {line}", flush=True)


# ---------------------------------------------------------------------------
# §4.7: find the daemon
# ---------------------------------------------------------------------------


async def connect() -> AsyncTrainyThingClient:
    """The daemon from ``TRAINYTHING_URL`` / ``_TOKEN``, then config.json, then
    127.0.0.1:4747; one ``npx -y trainything list`` when a local one does not
    answer; else ``NoDaemonError`` whose message is the one line of §4.7.
    Never a widget: the token is read from the environment or the config."""
    return await connect_async()


# ---------------------------------------------------------------------------
# The queue config (§4.2)
# ---------------------------------------------------------------------------


def parse_options(text: Optional[str]) -> Optional[List[Dict[str, str]]]:
    """``id:Label, id2:Label 2`` (commas or new lines) -> ``[{id, label}]``.
    A bare ``id`` is its own label. Empty -> None."""
    if text is None or text.strip() == "":
        return None
    out: List[Dict[str, str]] = []
    for part in text.replace("\n", ",").split(","):
        part = part.strip()
        if not part:
            continue
        oid, _, label = part.partition(":")
        oid, label = oid.strip(), label.strip()
        if not oid:
            raise ValueError(f"options: an entry has no id: {part!r}")
        out.append({"id": oid, "label": label or oid})
    return out or None


def queue_config(
    *,
    title: str,
    slug: str = "",
    prompt: str = "",
    intro: str = "",
    mode: str = "single",
    options: str = "",
    feedback: str = "on_reject",
    identity: str = "name_email",
    order: str = "coverage",
    min_reviews: int = 1,
    project: str = "comfyui",
    prompt_text: Optional[str] = None,
) -> Dict[str, Any]:
    """The ``POST /api/queues`` body. ``completion.min_reviews_per_card`` is
    always sent: without it ``/status`` never says ``done`` before a close or a
    deadline. ``options`` only for choose / select (the daemon refuses them
    elsewhere). An empty ``intro`` is §10 question 7's default."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r} (want one of {', '.join(MODES)})")
    opts = parse_options(options)
    if mode in ("choose", "select"):
        if not opts or len(opts) < 2:
            raise ValueError(f"mode {mode} needs at least two options, as id:Label, id:Label")
    else:
        opts = None
    intro_text = intro.strip() if intro and intro.strip() else default_intro(prompt_text)
    body: Dict[str, Any] = {
        "title": title.strip() or "ComfyUI batch",
        "slug": slug.strip() or None,
        "prompt": prompt.strip() or None,
        "intro": intro_text,
        "mode": mode,
        "feedback": feedback,
        "identity": identity,
        "order": order,
        "completion": {"min_reviews_per_card": max(1, int(min_reviews))},
        "project": project.strip() or None,
        "options": opts,
    }
    return {k: v for k, v in body.items() if v is not None}


# ---------------------------------------------------------------------------
# Send
# ---------------------------------------------------------------------------


async def send(
    client: AsyncTrainyThingClient,
    images: Any,
    config: Dict[str, Any],
    *,
    node_id: Any,
    images_b: Any = None,
    labels: Optional[str] = None,
    meta: Optional[str] = None,
    facts: Optional[Dict[str, Any]] = None,
    prompt: Any = None,
    extra_pnginfo: Optional[Dict[str, Any]] = None,
    public: bool = False,
) -> Dict[str, Any]:
    """Encode, upload, create-or-append (cards.send_batch, on a worker thread),
    then share when ``public``. Returns the queue handle (§4.2)."""
    run = make_run_id(node_id)
    handle = await asyncio.to_thread(
        send_batch,
        client.sync,
        images,
        config,
        run,
        images_b=images_b,
        labels=labels,
        meta=meta,
        facts=facts,
        prompt=prompt,
        extra_pnginfo=extra_pnginfo,
    )
    handle["run"] = run
    if public:
        shared = await client.share(handle["queue_id"])
        if shared.get("share_url"):
            handle["share_url"] = shared["share_url"]
            handle["link"] = shared["share_url"]
    return handle


def graph_facts(prompt: Any, node_id: Any) -> Dict[str, Any]:
    """The §6.4 walk from this node's ``images`` input; never raises."""
    try:
        return walk_graph(prompt, node_id, "images")
    except Exception as err:  # best effort by design
        return {"walk": f"stopped: {type(err).__name__}: {err}"}


def send_line(handle: Dict[str, Any]) -> str:
    verb = "new queue" if handle.get("created") else "appended to"
    return (
        f"{handle.get('added')} cards, {verb} {handle.get('slug')} "
        f"({handle.get('total')} in all): {handle.get('link')}"
    )


# ---------------------------------------------------------------------------
# The QR (§4.2 "On the node", §10 question 4)
# ---------------------------------------------------------------------------


def qr_png(link: str, caption: Optional[str] = None, scale: int = 8) -> bytes:
    """A QR of ``link`` (segno, pure Python) with the link and an optional
    status line printed under it, so the link is readable on the canvas
    without a JavaScript extension."""
    import segno
    from PIL import Image, ImageDraw, ImageFont

    buf = _io.BytesIO()
    segno.make(link, error="m").save(buf, kind="png", scale=scale, border=4)
    qr = Image.open(_io.BytesIO(buf.getvalue())).convert("RGB")
    lines = [link] + ([caption] if caption else [])
    try:
        font = ImageFont.load_default(size=14)
    except TypeError:  # Pillow < 10.1
        font = ImageFont.load_default()
    probe = ImageDraw.Draw(qr)
    widths = [probe.textlength(ln, font=font) for ln in lines]
    width = max(qr.width, int(max(widths)) + 24)
    line_h = 20
    img = Image.new("RGB", (width, qr.height + line_h * len(lines) + 10), "white")
    img.paste(qr, ((width - qr.width) // 2, 0))
    draw = ImageDraw.Draw(img)
    for k, (ln, w) in enumerate(zip(lines, widths)):
        draw.text(((width - w) / 2, qr.height + k * line_h), ln, fill="black", font=font)
    out = _io.BytesIO()
    img.save(out, format="PNG")
    return out.getvalue()


def qr_pil(link: str, caption: Optional[str] = None) -> Any:
    """The QR as a PIL image: the running node's preview (``set_progress``)."""
    from PIL import Image

    return Image.open(_io.BytesIO(qr_png(link, caption))).convert("RGB")


def qr_tensor(link: str, caption: Optional[str] = None) -> Any:
    """The QR as a ComfyUI IMAGE, ``[1, H, W, 3]`` float in 0..1."""
    import numpy as np
    import torch

    from ..png import decode_png

    arr, _ = decode_png(qr_png(link, caption))
    return torch.from_numpy(np.ascontiguousarray(arr))[None, ...]


# ---------------------------------------------------------------------------
# Progress and interrupt (§4.3)
# ---------------------------------------------------------------------------


async def set_progress(value: float, max_value: float, node_id: Optional[str] = None, preview: Any = None) -> None:
    """``ComfyAPI().execution.set_progress``; ``comfy.utils.ProgressBar`` if
    that is missing. ``preview`` (a PIL image) shows on the node while it runs."""
    try:
        from comfy_api.latest import ComfyAPI

        await ComfyAPI().execution.set_progress(
            value, max(1, max_value), node_id=node_id, preview_image=preview, ignore_size_limit=True
        )
    except Exception as err:
        try:
            import comfy.utils

            comfy.utils.ProgressBar(max(1, int(max_value)), node_id=node_id).update_absolute(int(value))
        except Exception:
            log.debug("trainything: no progress (%s)", err)


def comfy_interrupted() -> bool:
    """ComfyUI's interrupt flag (``POST /interrupt`` sets it via
    ``nodes.interrupt_processing``). It never cancels an async node's task;
    a node has to look. False outside ComfyUI."""
    try:
        import comfy.model_management as mm
    except Exception:
        return False
    return bool(mm.processing_interrupted())


def raise_interrupt() -> None:
    """Consume the flag and raise ComfyUI's own ``InterruptProcessingException``,
    so the executor reports ``execution_interrupted`` as for any core node."""
    import comfy.model_management as mm

    mm.throw_exception_if_processing_interrupted()
    raise mm.InterruptProcessingException()


class WaitInterrupted(Exception):
    pass


@dataclass
class WaitResult:
    status: Dict[str, Any]
    timed_out: bool


async def wait_until_done(
    client: AsyncTrainyThingClient,
    handle: Dict[str, Any],
    timeout_s: float,
    *,
    node_id: Optional[str] = None,
    interrupted: Callable[[], bool] = comfy_interrupted,
    preview: Any = None,
    poll_s: float = POLL_S,
    on_status: Optional[Callable[[Dict[str, Any]], Awaitable[None]]] = None,
) -> WaitResult:
    """Poll ``/status`` every ``poll_s`` with ``asyncio.sleep`` (the event loop
    keeps serving), report "judged N of M" as progress, and return when
    ``done`` or at the timeout. ``interrupted()`` is checked every quarter
    second; when it is true, ``WaitInterrupted`` is raised: the queue is left
    open, as it is on ``CancelledError``."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    qid = handle["queue_id"]
    last_judged = None
    while True:
        if interrupted():
            raise WaitInterrupted()
        st = await client.status(qid)
        cards = int(st.get("cards") or 0)
        judged = int(st.get("judged") or 0)
        await set_progress(judged, cards, node_id=node_id, preview=preview)
        if judged != last_judged:
            say(f"{handle.get('slug')}: judged {judged} of {cards}")
            last_judged = judged
        if on_status is not None:
            await on_status(st)
        if st.get("done"):
            return WaitResult(st, False)
        now = time.monotonic()
        if now >= deadline:
            return WaitResult(st, True)
        step_until = min(deadline, now + poll_s)
        while time.monotonic() < step_until:
            if interrupted():
                raise WaitInterrupted()
            await asyncio.sleep(min(INTERRUPT_CHECK_S, max(0.0, step_until - time.monotonic())))


def left_open_line(slug: Any) -> str:
    return f"wait cancelled; queue {slug} is still open and keeps its verdicts. {LOAD_LATER}."


# ---------------------------------------------------------------------------
# This run's verdicts (§4.3 "The split", §4.6)
# ---------------------------------------------------------------------------


@dataclass
class RunResults:
    """What one run's cards came to: ``refs[i]`` is batch index ``i``."""

    mode: str
    refs: List[str]
    split: Split
    #: one labels.jsonl line per batch index, in batch order
    lines: List[Dict[str, Any]]
    #: the results.json card per batch index (None when the daemon lost it)
    cards: List[Optional[Dict[str, Any]]]
    summary_md: str
    results: Dict[str, Any] = field(default_factory=dict)


def run_labels(labels_text: str, refs: Sequence[str]) -> List[Dict[str, Any]]:
    """The daemon's labels.jsonl narrowed to this run's refs, in batch order;
    a ref with no line (or no verdict) gets an ``unjudged`` line, so the
    ``labels`` output always has one line per image and says ``unjudged``
    for every card nobody judged (a timeout's partial result)."""
    by_ref: Dict[str, Dict[str, Any]] = {}
    for raw in labels_text.splitlines():
        raw = raw.strip()
        if raw:
            line = json.loads(raw)
            by_ref[str(line.get("ref"))] = line
    out: List[Dict[str, Any]] = []
    for ref in refs:
        line = by_ref.get(str(ref))
        if line is None:
            line = {"ref": ref, "label": None, "verdict": "unjudged", "n": 0, "contested": False, "notes": []}
        elif line.get("verdict") in (None, ""):
            line = dict(line, verdict="unjudged")
        out.append(line)
    return out


def labels_text(lines: Sequence[Dict[str, Any]]) -> str:
    return "".join(json.dumps(line) + "\n" for line in lines)


def notes_text(run: "RunResults") -> str:
    """§4.6 ``notes``: the rejection notes only, one per line, prefixed by the
    card's label (``#<index>`` when it has none). In the modes with no
    reject (pair, choose, select) every note counts."""
    out: List[str] = []
    for i, card in enumerate(run.cards):
        if card is None:
            continue
        label = card.get("label") or f"#{i}"
        for j in card.get("judgments") or []:
            note = j.get("note")
            if not isinstance(note, str) or note.strip() == "":
                continue
            if run.mode == "single" and j.get("verdict") != "reject":
                continue
            out.append(f"{label}: {' '.join(note.split())}")
    return "\n".join(out)


def meta_json(run: "RunResults", indices: Sequence[int]) -> str:
    """§4.6 ``accepted_meta``: a JSON array of those cards' ``meta``."""
    metas = []
    for i in indices:
        card = run.cards[i]
        meta = dict(card.get("meta") or {}) if card else {}
        meta.setdefault("index", i)
        meta.setdefault("ref", run.refs[i])
        metas.append(meta)
    return json.dumps(metas, indent=1)


async def collect(client: AsyncTrainyThingClient, handle: Dict[str, Any]) -> RunResults:
    """Labels, results.json and summary.md from the daemon, narrowed to the
    refs this run added (an appended queue maps only its own cards)."""
    qid = handle["queue_id"]
    refs = [str(r) for r in handle["refs"]]
    labels_raw, results_raw, summary_md = await asyncio.gather(
        client.results(qid, "labels"),
        client.results(qid, "json"),
        client.results(qid, "summary"),
    )
    doc = json.loads(results_raw)
    mode = handle.get("mode") or doc.get("queue", {}).get("mode") or "single"
    lines = run_labels(labels_raw, refs)
    by_ref = {str(c.get("ref")): c for c in doc.get("cards", [])}
    cards = [by_ref.get(ref) for ref in refs]
    return RunResults(
        mode=mode,
        refs=refs,
        split=split_labels(lines, refs, mode),
        lines=lines,
        cards=cards,
        summary_md=summary_md,
        results=doc,
    )


# ---------------------------------------------------------------------------
# Tensors (§6.1): slice IMAGE and LATENT by batch index, as core does
# ---------------------------------------------------------------------------


def blocker() -> Any:
    """A silent ``ExecutionBlocker``: with a message ComfyUI reports an
    ``execution_error``, and a run with nothing rejected is not an error."""
    from comfy_execution.graph_utils import ExecutionBlocker

    return ExecutionBlocker(None)


def batch_size(images: Any) -> int:
    return int(images.shape[0]) if getattr(images, "ndim", 0) == 4 else 1


def slice_images(images: Any, indices: Sequence[int]) -> Any:
    """``images[indices]`` (a copy), in the order given."""
    import torch

    idx = torch.tensor(list(indices), dtype=torch.long, device=images.device)
    return images.index_select(0, idx)


def pick_images(images: Any, images_b: Any, indices: Sequence[int], sides: Sequence[str]) -> Any:
    """Pair mode: index ``i`` from ``images`` when its side is ``a``, from
    ``images_b`` when ``b``."""
    import torch

    rows = []
    for i, side in zip(indices, sides):
        src = images_b if side == "b" else images
        if src is None:
            raise ValueError("pair mode: wire images_b so the B side can be returned")
        rows.append(src[i])
    return torch.stack(rows, dim=0)


def slice_latent(latent: Dict[str, Any], indices: Sequence[int]) -> Dict[str, Any]:
    """Core's ``LatentFromBatch`` (nodes.py), generalised from a range to any
    index list: ``samples`` sliced and cloned; ``noise_mask`` kept whole when
    it has one entry, else repeated up to the batch and sliced; ``batch_index``
    sliced when present, else the positions taken."""
    import math

    import torch

    s = latent.copy()
    s_in = latent["samples"]
    n = int(s_in.shape[0])
    bad = [i for i in indices if i < 0 or i >= n]
    if bad:
        raise ValueError(f"latent has {n} samples; the batch needs index {bad[0]} (wire the latent of the same batch)")
    idx = torch.tensor(list(indices), dtype=torch.long, device=s_in.device)
    s["samples"] = s_in.index_select(0, idx).clone()
    if "noise_mask" in latent and latent["noise_mask"] is not None:
        masks = latent["noise_mask"]
        if masks.shape[0] == 1:
            s["noise_mask"] = masks.clone()
        else:
            if masks.shape[0] < n:
                masks = masks.repeat(math.ceil(n / masks.shape[0]), 1, 1, 1)[:n]
            s["noise_mask"] = masks.index_select(0, idx.to(masks.device)).clone()
    if "batch_index" not in latent:
        s["batch_index"] = list(indices)
    else:
        s["batch_index"] = [latent["batch_index"][i] for i in indices]
    return s


def split_outputs(
    run: RunResults,
    images: Any,
    latent: Optional[Dict[str, Any]] = None,
    images_b: Any = None,
) -> List[Any]:
    """The nine Wait sockets of §4.6, in order. An empty ``accepted`` or
    ``rejected`` (and ``accepted_latent`` / ``accepted_indices`` with it) is a
    silent blocker on that socket only."""
    n = batch_size(images)
    if n != len(run.refs):
        raise ValueError(f"the queue handle has {len(run.refs)} cards for {n} images: wire the batch Send was given")
    s = run.split
    if s.accepted:
        if run.mode == "pair":
            accepted = pick_images(images, images_b, s.accepted, s.accepted_from)
        else:
            accepted = slice_images(images, s.accepted)
    else:
        accepted = blocker()
    if s.rejected:
        if run.mode == "pair":
            rejected = pick_images(images, images_b, s.rejected, s.rejected_from)
        else:
            rejected = slice_images(images, s.rejected)
    else:
        rejected = blocker()
    if latent is not None and s.accepted:
        accepted_latent = slice_latent(latent, s.accepted)
    else:
        accepted_latent = blocker()
    indices: Any = list(s.accepted) if s.accepted else blocker()
    return [
        accepted,
        rejected,
        accepted_latent,
        images,
        labels_text(run.lines),
        run.summary_md,
        notes_text(run),
        meta_json(run, s.accepted),
        indices,
    ]


def blocked_outputs(count: int = len(WAIT_OUTPUTS)) -> List[Any]:
    return [blocker() for _ in range(count)]


def result_line(run: RunResults) -> str:
    s = run.split
    parts = [f"{len(s.accepted)} accepted", f"{len(s.rejected)} rejected"]
    if s.contested:
        parts.append(f"{len(s.contested)} contested")
    if s.unjudged:
        parts.append(f"{len(s.unjudged)} unjudged")
    return ", ".join(parts)


# ---------------------------------------------------------------------------
# The run's folder (§4.3): output/trainything/<slug>/
# ---------------------------------------------------------------------------


def output_root() -> str:
    """``<ComfyUI output>/trainything``; ``./output/trainything`` outside ComfyUI."""
    try:
        import folder_paths

        base = folder_paths.get_output_directory()
    except Exception:
        base = os.path.join(os.getcwd(), "output")
    return os.path.join(os.path.abspath(base), OUTPUT_SUBDIR)


def _write_text(path: str, text: str) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def write_run_folder_sync(client: TrainyThingClient, slug: str, mode: str, root: str) -> Dict[str, Any]:
    """``<root>/<slug>/``: ``results.json`` + ``blobs/`` (written by the daemon
    via ``results?out=`` when it is local; else, or when that fails, the
    unpacked ``results/bundle.zip``), ``labels.jsonl``, ``summary.md``, and
    ``preferences.jsonl`` in pair mode. ``out`` is absolute: the daemon
    resolves it against its own cwd otherwise."""
    folder = os.path.abspath(os.path.join(root, slug))
    os.makedirs(folder, exist_ok=True)
    how = "bundle"
    if is_local_url(client.base_url):
        try:
            client.results(slug, "json", out=os.path.join(folder, "results.json"))
            how = "local"
        except ClientError as err:
            log.info("trainything: results --out failed (%s); reading the bundle", err)
    if how == "bundle":
        client.bundle(slug, folder)
    _write_text(os.path.join(folder, "labels.jsonl"), str(client.results(slug, "labels")))
    _write_text(os.path.join(folder, "summary.md"), str(client.results(slug, "summary")))
    if mode == "pair":
        _write_text(os.path.join(folder, "preferences.jsonl"), str(client.results(slug, "preferences")))
    return {"folder": folder, "how": how}


async def write_run_folder(client: AsyncTrainyThingClient, slug: str, mode: str, root: Optional[str] = None) -> Dict[str, Any]:
    return await asyncio.to_thread(write_run_folder_sync, client.sync, slug, mode, root or output_root())


# ---------------------------------------------------------------------------
# The whole wait, for Wait and Review
# ---------------------------------------------------------------------------


async def wait_and_split(
    client: AsyncTrainyThingClient,
    handle: Dict[str, Any],
    images: Any,
    *,
    latent: Optional[Dict[str, Any]] = None,
    images_b: Any = None,
    timeout_minutes: float = DEFAULT_TIMEOUT_MIN,
    on_timeout: str = "partial",
    node_id: Optional[str] = None,
    preview: Any = None,
) -> Tuple[List[Any], bool]:
    """Wait, then the split and the folder: ``(the nine Wait sockets,
    blocked)``, where ``blocked`` means a timeout with ``block``, every socket
    a silent blocker, and a caller with more sockets blocks them too. On
    ComfyUI's interrupt (or task cancellation) the queue is left open, the
    console says so with the slug, and the interrupt propagates."""
    slug = handle.get("slug")
    n = batch_size(images)
    if n != len(handle.get("refs") or []):
        raise ValueError(f"the queue handle has {len(handle.get('refs') or [])} cards for {n} images: wire the batch Send was given")
    try:
        waited = await wait_until_done(client, handle, float(timeout_minutes) * 60.0, node_id=node_id, preview=preview)
    except WaitInterrupted:
        say(left_open_line(slug))
        raise_interrupt()
        raise  # unreachable: raise_interrupt always raises
    except asyncio.CancelledError:
        say(left_open_line(slug))
        raise
    if waited.timed_out:
        say(f"{slug}: timed out after {timeout_minutes:g} min ({on_timeout}); the queue stays open. {LOAD_LATER}.")
    run = await collect(client, handle)
    folder = await write_run_folder(client, str(slug), run.mode)
    say(f"{slug}: {result_line(run)}; wrote {folder['folder']} ({folder['how']})")
    if waited.timed_out and on_timeout == "block":
        return blocked_outputs(), True
    return split_outputs(run, images, latent, images_b), False


# ---------------------------------------------------------------------------
# Load (§4.4): a queue read back by name, images from the blob store
# ---------------------------------------------------------------------------

#: The outputs of Load, in socket order: Wait's minus ``accepted_latent``.
LOAD_OUTPUTS = tuple(name for name in WAIT_OUTPUTS if name != "accepted_latent")


class QueueNotFound(ValueError):
    pass


async def find_queue_by_slug(client: AsyncTrainyThingClient, slug: str) -> Dict[str, Any]:
    """The queue by slug or id (as typed, then normalized as the daemon does);
    ``QueueNotFound`` naming it when the daemon has neither."""
    from ..cards import normalize_slug

    wanted = (slug or "").strip()
    if not wanted:
        raise QueueNotFound("TrainyThing Load needs a slug: the queue's link name (or its id)")
    tries = [wanted] + [s for s in [normalize_slug(wanted)] if s and s != wanted]
    for name in tries:
        found = await client.find_queue(name)
        if found is not None:
            return found
    raise QueueNotFound(f"no TrainyThing queue named {wanted!r} at {client.base_url} (check the slug on the dashboard)")


def load_handle(found: Dict[str, Any], base_url: str) -> Dict[str, Any]:
    """A queue handle (the shape Send makes) for a queue found by name; its
    ``refs`` are filled in once the cards are read."""
    queue = found["queue"]
    return {
        "base_url": base_url,
        "slug": queue.get("share_token") or queue.get("id"),
        "queue_id": queue["id"],
        "link": found.get("share_url") or found.get("local_url"),
        "local_url": found.get("local_url"),
        "share_url": found.get("share_url"),
        "mode": queue.get("mode"),
        "refs": [],
    }


async def load_fingerprint(slug: str) -> str:
    """§4.4: what decides whether ComfyUI may reuse Load's last outputs. The
    queue's id, card count, judgment count, status and closed time (``/status``
    and the queue row): it moves when a verdict lands, a card is added, or
    the queue is closed or reopened, and not otherwise. When the daemon or the
    queue cannot be read, a value that never repeats, so Load runs and reports
    the problem itself. No auto-start here: that is execute's job."""
    try:
        client = await connect_async(auto_start=False, timeout=2.0)
        found = await find_queue_by_slug(client, slug)
        qid = found["queue"]["id"]
        st = await client.status(qid)
        return json.dumps(
            [qid, st.get("cards"), st.get("judgments"), st.get("status"), found["queue"].get("closed_at")]
        )
    except Exception as err:  # never fail validation; execute says what is wrong
        return f"unreadable {time.time()!r}: {type(err).__name__}"


def card_items(card: Dict[str, Any]) -> Optional[List[Dict[str, Any]]]:
    """A card's image items, A side first (one for single, two for pair);
    None when the card is not all images (text, audio or video the CLI could
    have added)."""
    items = list(card.get("items") or [])
    if not items or len(items) > 2 or any(it.get("kind") != "image" or not it.get("src") for it in items):
        return None
    if len(items) == 2:
        items.sort(key=lambda it: 0 if (it.get("side") == "a" or str(it.get("ref", "")).endswith("-a")) else 1)
    return items


def image_cards(doc: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """``(the image cards in card order, the cards skipped)``."""
    kept: List[Dict[str, Any]] = []
    skipped: List[Dict[str, Any]] = []
    for card in doc.get("cards") or []:
        (kept if card_items(card) is not None else skipped).append(card)
    return kept, skipped


def blob_path(folder: str, src: str) -> str:
    """``blobs/<sha><ext>`` (relative, as results.json writes it next to the
    blobs) or an absolute path, kept inside ``folder`` when relative."""
    if os.path.isabs(src):
        return src
    root = os.path.realpath(folder)
    path = os.path.realpath(os.path.join(root, src))
    if os.path.commonpath([root, path]) != root:
        raise ValueError(f"results.json names a blob outside the folder: {src}")
    return path


def decode_batch(paths: Sequence[str]) -> Tuple[Any, int]:
    """PNG files -> one IMAGE ``[B, H, W, 3]`` float32 in 0..1, in order.
    A ComfyUI batch has one size, so any image whose size differs from the
    first one's is resized to it (bilinear); returns ``(tensor, resized)``."""
    import numpy as np
    import torch
    from PIL import Image

    from ..png import decode_png

    arrays = []
    size = None
    resized = 0
    for path in paths:
        with open(path, "rb") as fh:
            arr, _ = decode_png(fh.read())
        h, w = arr.shape[:2]
        if size is None:
            size = (w, h)
        elif (w, h) != size:
            img = Image.fromarray(np.clip(arr * 255.0 + 0.5, 0, 255).astype(np.uint8))
            arr = np.asarray(img.resize(size, Image.BILINEAR)).astype(np.float32) / 255.0
            resized += 1
        arrays.append(arr)
    return torch.from_numpy(np.ascontiguousarray(np.stack(arrays))), resized


def read_folder(folder: str) -> Tuple[Dict[str, Any], str, str]:
    """``(results.json, labels.jsonl, summary.md)`` of a run's folder."""
    with open(os.path.join(folder, "results.json"), encoding="utf-8") as fh:
        doc = json.load(fh)
    with open(os.path.join(folder, "labels.jsonl"), encoding="utf-8") as fh:
        labels_raw = fh.read()
    with open(os.path.join(folder, "summary.md"), encoding="utf-8") as fh:
        summary_md = fh.read()
    return doc, labels_raw, summary_md


def load_run(doc: Dict[str, Any], labels_raw: str, summary_md: str, mode: Optional[str] = None) -> Tuple[RunResults, List[Dict[str, Any]]]:
    """A whole queue as one run: the image cards in card order are batch
    indices ``0..B-1`` (cards that are not images are left out of every
    index). Returns ``(run, skipped cards)``."""
    kept, skipped = image_cards(doc)
    refs = [str(c.get("ref")) for c in kept]
    mode = mode or (doc.get("queue") or {}).get("mode") or "single"
    lines = run_labels(labels_raw, refs)
    run = RunResults(
        mode=mode,
        refs=refs,
        split=split_labels(lines, refs, mode),
        lines=lines,
        cards=list(kept),
        summary_md=summary_md,
        results=doc,
    )
    return run, skipped


def load_images(run: RunResults, folder: str) -> Tuple[Any, Any, int]:
    """``(images, images_b or None, resized)``: the A side (or the only image)
    of every card, and the B side in pair mode, each one batch."""
    a_paths, b_paths = [], []
    for card in run.cards:
        items = card_items(card or {}) or []
        a_paths.append(blob_path(folder, items[0]["src"]))
        if len(items) > 1:
            b_paths.append(blob_path(folder, items[1]["src"]))
    images, resized = decode_batch(a_paths)
    images_b = None
    if b_paths and len(b_paths) == len(a_paths):
        images_b, more = decode_batch(b_paths)
        resized += more
        if tuple(images_b.shape[1:3]) != tuple(images.shape[1:3]):
            import torch

            images_b = torch.nn.functional.interpolate(
                images_b.movedim(-1, 1), size=tuple(images.shape[1:3]), mode="bilinear", align_corners=False
            ).movedim(1, -1).clamp(0, 1)
    return images, images_b, resized


async def load_queue(
    client: AsyncTrainyThingClient,
    slug: str,
    *,
    wait: bool = False,
    timeout_minutes: float = DEFAULT_TIMEOUT_MIN,
    node_id: Optional[str] = None,
    root: Optional[str] = None,
) -> List[Any]:
    """Load's body: find the queue, wait when asked (interruptible, as Wait),
    write the run's folder, read the cards back from it and split them.
    Returns the eight Load sockets (``LOAD_OUTPUTS``)."""
    found = await find_queue_by_slug(client, slug)
    handle = load_handle(found, client.base_url)
    name = str(handle["slug"])
    if wait:
        try:
            waited = await wait_until_done(client, handle, float(timeout_minutes) * 60.0, node_id=node_id)
        except WaitInterrupted:
            say(left_open_line(name))
            raise_interrupt()
            raise
        except asyncio.CancelledError:
            say(left_open_line(name))
            raise
        if waited.timed_out:
            say(f"{name}: timed out after {timeout_minutes:g} min; loading what is judged (the rest unjudged).")
    folder = await write_run_folder(client, name, str(handle.get("mode") or "single"), root)
    doc, labels_raw, summary_md = await asyncio.to_thread(read_folder, folder["folder"])
    run, skipped = load_run(doc, labels_raw, summary_md, handle.get("mode"))
    for card in skipped:
        kinds = ",".join(sorted({str(it.get("kind")) for it in card.get("items") or []})) or "empty"
        say(f"{name}: skipped card {card.get('ref')} ({kinds}): Load returns image cards only")
    if not run.refs:
        raise ValueError(f"queue {name} has no image cards to load")
    images, images_b, resized = await asyncio.to_thread(load_images, run, folder["folder"])
    if resized:
        h, w = int(images.shape[1]), int(images.shape[2])
        say(f"{name}: {resized} image(s) resized to the first card's {w}x{h} (a ComfyUI batch has one size)")
    outs = split_outputs(run, images, None, images_b)
    del outs[WAIT_OUTPUTS.index("accepted_latent")]
    say(f"{name}: loaded {len(run.refs)} images, {result_line(run)}; wrote {folder['folder']} ({folder['how']})")
    return outs
