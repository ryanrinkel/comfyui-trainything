"""TrainyThing Filter: any batch, the labels, a rule -> the
subset. The only node that knows nothing about the daemon."""

from __future__ import annotations

from comfy_api.latest import io

from ..split import RULES, filter_pick
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
                io.Image.Input("images", tooltip="The pictures the labels describe, in the same order."),
                io.Latent.Input("latent", optional=True, tooltip="Optional: their latents, cut the same way.", advanced=True),
                io.Image.Input("images_b", optional=True,
                               tooltip="pair mode only: the B pictures, so accepted and rejected pick the winning and losing picture of each pair. Not wired: the A picture is used.", advanced=True),
                io.String.Input("labels", multiline=True, force_input=True, dynamic_prompts=False,
                                tooltip="The labels output of Review, Wait or Load: one line per picture with its verdict."),
                io.Combo.Input("rule", options=list(RULES), default="accepted",
                               tooltip="Which pictures to pass on: accepted, rejected, contested (the reviewers disagreed), unjudged (nobody looked), or option (together with option_id)."),
                io.String.Input("option_id", default="", dynamic_prompts=False,
                                tooltip="With rule option: the option's id, e.g. hair. choose: the pictures where that option won. select: the pictures where it was ticked."),
            ],
            outputs=[
                io.Image.Output("images", display_name="images"),
                io.Latent.Output("latent", display_name="latent"),
                io.Int.Output("indices", display_name="indices", is_output_list=True),
                io.Int.Output("count", display_name="count"),
            ],
        )

    @classmethod
    def execute(cls, images, labels, rule, option_id="", latent=None, images_b=None) -> io.NodeOutput:
        n = c.batch_size(images)
        picked = filter_pick(labels, n, rule, option_id)
        indices = picked.indices
        what = f"{rule} {option_id}".strip() if rule == "option" else rule
        both = picked.mode == "pair" and rule in ("accepted", "rejected") and images_b is not None
        if images_b is not None and c.batch_size(images_b) != n:
            raise ValueError(f"images_b has {c.batch_size(images_b)} images for {n} in images: wire the A and B batches of the same pairs")
        if images_b is not None and not both:
            c.say(f"filter {what}: images_b is used only for accepted / rejected on pair labels; ignored")
        c.say(f"filter {what}: {len(indices)} of {n}" + (f" ({picked.sides.count('b')} from images_b)" if both else ""))
        if not indices:
            # A blocker on indices too: ComfyUI 0.39.0 feeds an empty list
            # output to the node downstream as one call with None, which
            # breaks Image From Batch and the like. count still says 0.
            return io.NodeOutput(c.blocker(), c.blocker(), c.blocker(), 0)
        if both:
            out_images = c.pick_images(images, c.match_size(images_b, images), indices, picked.sides)
            if latent is not None:
                # As Review's accepted_latent: the latent is one side's, and
                # the picks come from both, so a slice would not line up.
                c.say("filter: latent blocked (pair labels with images_b: the picks come from both sides, the latent is one side's)")
            out_latent = c.blocker()
        else:
            out_images = c.slice_images(images, indices)
            out_latent = c.slice_latent(latent, indices) if latent is not None else c.blocker()
        return io.NodeOutput(out_images, out_latent, list(indices), len(indices))
