"""TrainyThing Wait (PLAN-COMFY §4.3): hold the run until the queue a Send made
is done (or the timeout), then split the batch by the verdicts."""

from __future__ import annotations

from comfy_api.latest import io

from . import common as c
from .inputs import TrainyThingQueue, wait_inputs, wait_outputs


class TrainyThingWait(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainyThingWait",
            display_name="TrainyThing Wait",
            category=c.CATEGORY,
            description="Wait for the verdicts on a Send's queue, then pass on the keepers, the rejects, the notes and the seeds.",
            inputs=[
                TrainyThingQueue.Input("queue", tooltip="From TrainyThing Send."),
                io.Image.Input("images", tooltip="The batch that Send was given, to slice."),
                io.Latent.Input("latent", optional=True, tooltip="The batch's latent: accepted_latent is sliced from it."),
                io.Image.Input("images_b", optional=True, tooltip="pair mode: the B batch, so the winners can come from either side."),
                *wait_inputs(),
            ],
            outputs=wait_outputs(),
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    async def execute(cls, queue, images, timeout_minutes, on_timeout, latent=None, images_b=None) -> io.NodeOutput:
        client = await c.connect()
        outs, _ = await c.wait_and_split(
            client, queue, images, latent=latent, images_b=images_b, timeout_minutes=timeout_minutes,
            on_timeout=on_timeout, node_id=cls.hidden.unique_id,
        )
        return io.NodeOutput(*outs)
