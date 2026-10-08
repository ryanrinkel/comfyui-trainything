"""TrainyThing Review (PLAN-COMFY §4.1, §4.3): Send then Wait in one node. The
batch and its latent stay in memory across the wait; the QR is on the node
(as the running node's preview) before the wait starts."""

from __future__ import annotations

from comfy_api.latest import io

from . import common as c
from .inputs import HIDDEN, send_inputs, wait_inputs, wait_outputs
from .send import link_ui, send_from_inputs


class TrainyThingReview(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        inputs = send_inputs()
        # latent right after images, as the plan lists it
        inputs.insert(1, io.Latent.Input("latent", optional=True, tooltip="The batch's latent: accepted_latent is sliced from it."))
        return io.Schema(
            node_id="TrainyThingReview",
            display_name="TrainyThing Review",
            category=c.CATEGORY,
            description="Send the batch to a phone, wait for the verdicts, pass on the keepers, the rejects, the notes and the seeds.",
            inputs=inputs + wait_inputs(),
            outputs=wait_outputs() + [
                io.String.Output("link", display_name="link"),
                io.String.Output("slug", display_name="slug"),
            ],
            hidden=HIDDEN,
            is_output_node=True,
        )

    @classmethod
    async def execute(cls, images, title, slug, prompt, intro, mode, options, feedback, identity, order,
                      min_reviews, public, project, timeout_minutes, on_timeout,
                      latent=None, images_b=None, labels="", meta="") -> io.NodeOutput:
        # What the split will need, before anything is uploaded or waited on.
        c.check_wait_inputs(mode, images, latent, images_b if mode == "pair" else None)
        client, handle = await send_from_inputs(
            cls, images, title=title, slug=slug, prompt=prompt, intro=intro, mode=mode, options=options,
            feedback=feedback, identity=identity, order=order, min_reviews=min_reviews, public=public,
            project=project, images_b=images_b, labels=labels, meta=meta,
        )
        caption = f"{handle['added']} cards in {handle['slug']}: scan to review"
        # The QR as the running node's preview, so the phone can scan it while
        # the node waits; the same QR is the node's output when it finishes.
        preview = c.qr_pil(handle["link"], caption)
        outs, blocked = await c.wait_and_split(
            client, handle, images, latent=latent, images_b=images_b if mode == "pair" else None,
            timeout_minutes=timeout_minutes, on_timeout=on_timeout, node_id=cls.hidden.unique_id, preview=preview,
        )
        extra = [c.blocker(), c.blocker()] if blocked else [handle["link"], handle["slug"]]
        return io.NodeOutput(*outs, *extra, ui=link_ui(cls, handle, caption))
