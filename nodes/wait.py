"""TrainyThing Wait: hold the run until the queue a Send made
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
                TrainyThingQueue.Input("queue", tooltip="Connect from TrainyThing Send's queue output."),
                io.Image.Input("images", tooltip="The same pictures that went into Send, so they can be split into keepers and rejects."),
                io.Latent.Input("latent", optional=True, tooltip="Optional: the latent from the KSampler that made these pictures. accepted_latent then carries the keepers' latents to a next step."),
                io.Image.Input("images_b", optional=True, tooltip="pair mode only: the B pictures, so the winners can come from either side.", advanced=True),
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
