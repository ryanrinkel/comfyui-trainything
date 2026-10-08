"""TrainyThing Load (PLAN-COMFY §4.4): a later run reads a queue back by name,
images included, so verdicts collected an hour or a week ago feed a new graph."""

from __future__ import annotations

from comfy_api.latest import io

from . import common as c


class TrainyThingLoad(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainyThingLoad",
            display_name="TrainyThing Load",
            category=c.CATEGORY,
            description="Read a TrainyThing queue back by its slug: the images from the blob store, the keepers, the rejects, the notes and the seeds.",
            inputs=[
                io.String.Input("slug", default="", dynamic_prompts=False,
                                tooltip="The queue's link name (as on the dashboard), or its id."),
                io.Boolean.Input("wait", default=False,
                                 tooltip="Off: read what is judged now. On: wait until the queue is done (or the timeout), as Wait does."),
                io.Float.Input("timeout_minutes", default=c.DEFAULT_TIMEOUT_MIN, min=0.1, max=10080.0, step=0.1, round=0.01,
                               tooltip="With wait on: give up waiting after this long and load what is judged."),
            ],
            outputs=[
                io.Image.Output("accepted", display_name="accepted"),
                io.Image.Output("rejected", display_name="rejected"),
                io.Image.Output("images", display_name="images"),
                io.String.Output("labels", display_name="labels"),
                io.String.Output("summary", display_name="summary"),
                io.String.Output("notes", display_name="notes"),
                io.String.Output("accepted_meta", display_name="accepted_meta"),
                io.Int.Output("accepted_indices", display_name="accepted_indices", is_output_list=True),
            ],
            hidden=[io.Hidden.unique_id],
        )

    @classmethod
    async def fingerprint_inputs(cls, slug, wait=False, timeout_minutes=c.DEFAULT_TIMEOUT_MIN, **kwargs):
        # ComfyUI reuses the last outputs while this value repeats (§4.4).
        return await c.load_fingerprint(slug)

    @classmethod
    async def execute(cls, slug, wait, timeout_minutes) -> io.NodeOutput:
        client = await c.connect()
        outs = await c.load_queue(client, slug, wait=bool(wait), timeout_minutes=timeout_minutes,
                                  node_id=cls.hidden.unique_id)
        return io.NodeOutput(*outs)
