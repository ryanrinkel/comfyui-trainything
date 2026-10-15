"""The V3 schema pieces Send, Wait and Review share (PLAN-COMFY §4.2, §4.3, §4.6).
Imported only inside ComfyUI (it needs ``comfy_api``)."""

from __future__ import annotations

from typing import List

from comfy_api.latest import io

from . import common as c

#: Only a Send's ``queue`` connects to a Wait's (§4.2 "Out").
TrainyThingQueue = io.Custom(c.QUEUE_TYPE)


def _text(id: str, default: str, tooltip: str, multiline: bool = False, optional: bool = False,
          advanced: bool = False) -> io.String.Input:
    # dynamic_prompts off: the frontend would rewrite {a|b} inside JSON meta.
    # advanced: hidden until "Show advanced inputs" (frontend >= 1.5x), so a
    # fresh node shows the essentials only (§4.2 "a few things if wanted").
    return io.String.Input(id, default=default, multiline=multiline, dynamic_prompts=False, optional=optional,
                           tooltip=tooltip, advanced=advanced or None)


def send_inputs() -> List[io.Input]:
    """§4.2: images and the queue settings. The token is never one of them.
    The essentials come first (images, title, prompt, mode, public); the rest
    is advanced and web/trainything.js keeps it folded until "more settings"."""
    return [
        io.Image.Input("images", tooltip="The pictures to review. Each one becomes a card on the phone."),
        _text("title", "ComfyUI batch", "A name for this batch. Shown at the top of the phone page and on the dashboard."),
        _text("prompt", "Keep this one?", "The question written on every card, e.g. Keep this one?"),
        io.Combo.Input("mode", options=c.MODES, default="single", tooltip="How the reviewer answers. single: keep or reject each picture. pair: pick A or B (wire the B pictures into images_b). choose: pick one of your options per picture. select: tick any of your options per picture."),
        io.Boolean.Input("public", default=False, tooltip="On: make a link that works away from this computer (your phone off the home Wi-Fi), through a tunnel. Off: the link only works on this machine."),
        _text("slug", "", "The short name in the link, like a folder name. Leave empty to make one from the title. Use the name of a batch that is still open to add these pictures to it. Type random for a name nobody can guess.", advanced=True),
        _text("intro", "", "A short message the reviewer reads before the first card. Leave empty for an automatic one that quotes your positive prompt and says what to do.", multiline=True, advanced=True),
        _text("options", "", "Only for choose and select: the buttons the reviewer sees, as id:Label pairs separated by commas, e.g. hair:Hair colour, hands:Hands, ok:Fine. An id is lowercase letters, digits, - or _. Ignored in single and pair.", advanced=True),
        io.Combo.Input("feedback", options=c.FEEDBACKS, default="on_reject", tooltip="When the reviewer is asked to type a reason. on_reject: only when they reject (single mode; a note box on every card in the other modes). optional: a note box on every card. required: they must write something. off: never.", advanced=True),
        io.Combo.Input("identity", options=c.IDENTITIES, default="name_email", tooltip="name_email: the phone asks for a name and email before the first card, so you know who said what. anonymous: it asks nothing.", advanced=True),
        io.Combo.Input("order", options=c.ORDERS, default="coverage", tooltip="The order the cards are shown in. coverage: the pictures with the fewest reviews first, so several reviewers spread out. fixed: the batch order.", advanced=True),
        io.Int.Input("min_reviews", default=1, min=1, max=50, tooltip="How many people must judge each picture before the batch counts as done. 1 when it is just you.", advanced=True),
        _text("project", "comfyui", "A tag that groups batches on the dashboard.", advanced=True),
        io.Image.Input("images_b", optional=True, tooltip="pair mode only: the B pictures, one for each A picture, same count.", advanced=True),
        _text("labels", "", "Optional: a caption above each card. One line per picture, in batch order.", multiline=True, optional=True, advanced=True),
        _text("meta", "", "Optional: extra details to attach to every card as JSON, e.g. {\"lora\": \"witch-v2\"}. One object for all pictures, or one per line. It comes back with the verdicts and overrides what the node reads from the graph.", multiline=True, optional=True, advanced=True),
    ]


def wait_inputs() -> List[io.Input]:
    """§4.3: how long, and what a timeout does."""
    return [
        io.Float.Input("timeout_minutes", default=c.DEFAULT_TIMEOUT_MIN, min=0.1, max=10080.0, step=0.1, round=0.01,
                       tooltip="How long to wait for the phone before the run moves on. The batch stays open on the dashboard either way.", advanced=True),
        io.Combo.Input("on_timeout", options=c.ON_TIMEOUT, default="partial",
                       tooltip="What happens when the time runs out. partial: carry on with what was judged, the rest counts as unjudged. block: stop here; nothing after this node runs.", advanced=True),
    ]


def wait_outputs() -> List[io.Output]:
    """§4.6, in socket order (``common.WAIT_OUTPUTS``)."""
    return [
        io.Image.Output("accepted", display_name="accepted", tooltip="The pictures the reviewer kept."),
        io.Image.Output("rejected", display_name="rejected", tooltip="The pictures the reviewer rejected."),
        io.Latent.Output("accepted_latent", display_name="accepted_latent", tooltip="The keepers' latents, cut from the latent you wired in. Nothing flows here if no latent was wired."),
        io.Image.Output("images", display_name="images", tooltip="All the pictures, unchanged, in batch order. Wire with labels into TrainyThing Filter."),
        io.String.Output("labels", display_name="labels", tooltip="One line per picture: its verdict, the votes and the notes. Made for TrainyThing Filter."),
        io.String.Output("summary", display_name="summary", tooltip="A short readable report: how many were kept, rejected or contested, and the notes. Wire to a Preview Any node."),
        io.String.Output("notes", display_name="notes", tooltip="The reasons the reviewer typed, one per line, with the picture each one was about."),
        io.String.Output("accepted_meta", display_name="accepted_meta", tooltip="For each keeper: its seed, batch position, prompt, checkpoint, LoRAs and sampler settings, as JSON. For reproducing or refining."),
        io.Int.Output("accepted_indices", display_name="accepted_indices", is_output_list=True, tooltip="The batch positions of the keepers, counting from 0."),
    ]


HIDDEN = [io.Hidden.unique_id, io.Hidden.prompt, io.Hidden.extra_pnginfo]
