"""TrainyThing Send: the batch becomes a queue; the link and a
QR show on the node; the run goes on."""

from __future__ import annotations

from comfy_api.latest import io, ui

from . import common as c
from .inputs import HIDDEN, TrainyThingQueue, send_inputs


async def send_from_inputs(cls, images, *, title, slug, prompt, intro, mode, options, feedback, identity, order,
                           min_reviews, public, project, images_b=None, labels="", meta=""):
    """Send's body, shared with Review: connect, build the config, upload.
    Returns ``(client, handle)``."""
    node_id = cls.hidden.unique_id
    graph = cls.hidden.prompt
    facts = c.graph_facts(graph, node_id)
    if mode == "pair" and images_b is None:
        raise ValueError("pair mode needs images_b (the B side of each card)")
    if mode != "pair" and images_b is not None:
        c.say(f"images_b is for pair mode; ignored in {mode} mode")
        images_b = None
    if images_b is not None and c.batch_size(images_b) != c.batch_size(images):
        raise ValueError(f"pair mode: images_b has {c.batch_size(images_b)} images for {c.batch_size(images)} "
                         f"in images; one card per index needs the same batch size")
    config = c.queue_config(
        title=title, slug=slug, prompt=prompt, intro=intro, mode=mode, options=options, feedback=feedback,
        identity=identity, order=order, min_reviews=min_reviews, project=project,
        prompt_text=c.intro_prompt(graph, node_id, facts, mode),
    )
    # --disable-metadata: no prompt / workflow chunk in the uploaded PNGs, as
    # Save Image; the walk above has already filled meta from the graph.
    no_meta = c.metadata_disabled()
    client = await c.connect()
    handle = await c.send(
        client, images, config, node_id=node_id, images_b=images_b, labels=labels, meta=meta, facts=facts,
        prompt=None if no_meta else graph, extra_pnginfo=None if no_meta else cls.hidden.extra_pnginfo,
        public=public,
    )
    c.say(c.send_line(handle))
    return client, handle


def link_ui(cls, handle, caption):
    """The QR (with the link printed under it) as an image preview, and the
    link as text for /history and for any frontend that shows it."""
    qr = c.qr_tensor(handle["link"], caption)
    return {**ui.PreviewImage(qr, cls=cls).as_dict(), "text": (handle["link"],)}


class TrainyThingSend(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainyThingSend",
            display_name="TrainyThing Send",
            category=c.CATEGORY,
            description="Send the batch to TrainyThing for review on a phone and go on; wire Wait (or use Load later) for the verdicts.",
            inputs=send_inputs(),
            outputs=[
                TrainyThingQueue.Output("queue", display_name="queue"),
                io.String.Output("link", display_name="link"),
                io.String.Output("slug", display_name="slug"),
            ],
            hidden=HIDDEN,
            is_output_node=True,
        )

    @classmethod
    async def execute(cls, images, title, slug, prompt, intro, mode, options, feedback, identity, order,
                      min_reviews, public, project, images_b=None, labels="", meta="") -> io.NodeOutput:
        _, handle = await send_from_inputs(
            cls, images, title=title, slug=slug, prompt=prompt, intro=intro, mode=mode, options=options,
            feedback=feedback, identity=identity, order=order, min_reviews=min_reviews, public=public,
            project=project, images_b=images_b, labels=labels, meta=meta,
        )
        caption = f"{handle['added']} cards in {handle['slug']}"
        return io.NodeOutput(handle, handle["link"], handle["slug"], ui=link_ui(cls, handle, caption))
