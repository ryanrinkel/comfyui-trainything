"""TrainyThing Filter (PLAN-COMFY §4.5): any batch, the labels, a rule -> the
subset. The only node that knows nothing about the daemon."""

from __future__ import annotations

from comfy_api.latest import io

from ..split import RULES, filter_indices
from . import common as c


class TrainyThingFilter(io.ComfyNode):
    @classmethod
    def define_schema(cls) -> io.Schema:
        return io.Schema(
            node_id="TrainyThingFilter",
            display_name="TrainyThing Filter",
            category=c.CATEGORY,
            description="Slice a batch (and its latent) by the verdicts in labels: accepted, rejected, contested, unjudged, or an option id.",
            inputs=[
                io.Image.Input("images", tooltip="The batch the labels describe, one line per image."),
                io.Latent.Input("latent", optional=True, tooltip="The batch's latent, sliced the same way."),
                io.String.Input("labels", multiline=True, force_input=True, dynamic_prompts=False,
                                tooltip="labels from Wait, Review or Load (labels.jsonl, one line per image)."),
                io.Combo.Input("rule", options=list(RULES), default="accepted",
                               tooltip="Which cards: by consensus, contested, unjudged, or 'option' with option_id."),
                io.String.Input("option_id", default="", dynamic_prompts=False,
                                tooltip="rule option: the option id (choose: the consensus is it; select: the consensus includes it)."),
            ],
            outputs=[
                io.Image.Output("images", display_name="images"),
                io.Latent.Output("latent", display_name="latent"),
                io.Int.Output("indices", display_name="indices", is_output_list=True),
                io.Int.Output("count", display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, images, labels, rule, option_id="", latent=None) -> io.NodeOutput:
        n = c.batch_size(images)
        indices = filter_indices(labels, n, rule, option_id)
        what = f"{rule} {option_id}".strip() if rule == "option" else rule
        c.say(f"filter {what}: {len(indices)} of {n}")
        if not indices:
            # A blocker on indices too: ComfyUI 0.39.0 feeds an empty list
            # output to the node downstream as one call with None, which
            # breaks Image From Batch and the like. count still says 0.
            return io.NodeOutput(c.blocker(), c.blocker(), c.blocker(), 0)
        out_latent = c.slice_latent(latent, indices) if latent is not None else c.blocker()
        return io.NodeOutput(c.slice_images(images, indices), out_latent, list(indices), len(indices))
