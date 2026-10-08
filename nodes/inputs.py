"""The V3 schema pieces Send, Wait and Review share (PLAN-COMFY §4.2, §4.3, §4.6).
Imported only inside ComfyUI (it needs ``comfy_api``)."""

from __future__ import annotations

from typing import List

from comfy_api.latest import io

from . import common as c

#: Only a Send's ``queue`` connects to a Wait's (§4.2 "Out").
TrainyThingQueue = io.Custom(c.QUEUE_TYPE)


def _text(id: str, default: str, tooltip: str, multiline: bool = False, optional: bool = False) -> io.String.Input:
    # dynamic_prompts off: the frontend would rewrite {a|b} inside JSON meta.
    return io.String.Input(id, default=default, multiline=multiline, dynamic_prompts=False, optional=optional, tooltip=tooltip)


def send_inputs() -> List[io.Input]:
    """§4.2: images and the queue settings. The token is never one of them."""
    return [
        io.Image.Input("images", tooltip="The batch to review: one card per image."),
        _text("title", "ComfyUI batch", "The queue's title on the phone and the dashboard."),
        _text("slug", "", "The link name. Empty: from the title. A name that exists and is open: the cards are appended. 'random': an unguessable one."),
        _text("prompt", "Keep this one?", "The question on every card."),
        _text("intro", "", "Shown before the first card. Empty: 'Generated from: <the positive prompt>'.", multiline=True),
        io.Combo.Input("mode", options=c.MODES, default="single", tooltip="single: keep or reject. pair: A or B (wire images_b). choose / select: one or any of the options."),
        _text("options", "", "choose / select only: id:Label, id:Label, ..."),
        io.Combo.Input("feedback", options=c.FEEDBACKS, default="on_reject", tooltip="When the reviewer is asked why."),
        io.Combo.Input("identity", options=c.IDENTITIES, default="name_email"),
        io.Combo.Input("order", options=c.ORDERS, default="coverage"),
        io.Int.Input("min_reviews", default=1, min=1, max=50, tooltip="Reviews per card before the queue is done."),
        io.Boolean.Input("public", default=False, tooltip="Start the tunnel now so the link works off this machine."),
        _text("project", "comfyui", "The project tag on the dashboard."),
        io.Image.Input("images_b", optional=True, tooltip="pair mode: the B side, same batch size."),
        _text("labels", "", "One line per image, shown above its card.", multiline=True, optional=True),
        _text("meta", "", "One JSON object for all images, or one per line / an array; merged over what the node finds in the graph.", multiline=True, optional=True),
    ]


def wait_inputs() -> List[io.Input]:
    """§4.3: how long, and what a timeout does."""
    return [
        io.Float.Input("timeout_minutes", default=c.DEFAULT_TIMEOUT_MIN, min=0.1, max=10080.0, step=0.1, round=0.01,
                       tooltip="Give up waiting after this long; the queue stays open."),
        io.Combo.Input("on_timeout", options=c.ON_TIMEOUT, default="partial",
                       tooltip="partial: return what was judged, the rest 'unjudged'. block: nothing downstream runs."),
    ]


def wait_outputs() -> List[io.Output]:
    """§4.6, in socket order (``common.WAIT_OUTPUTS``)."""
    return [
        io.Image.Output("accepted", display_name="accepted"),
        io.Image.Output("rejected", display_name="rejected"),
        io.Latent.Output("accepted_latent", display_name="accepted_latent"),
        io.Image.Output("images", display_name="images"),
        io.String.Output("labels", display_name="labels"),
        io.String.Output("summary", display_name="summary"),
        io.String.Output("notes", display_name="notes"),
        io.String.Output("accepted_meta", display_name="accepted_meta"),
        io.Int.Output("accepted_indices", display_name="accepted_indices", is_output_list=True),
    ]


HIDDEN = [io.Hidden.unique_id, io.Hidden.prompt, io.Hidden.extra_pnginfo]
