"""TrainyThing for ComfyUI: send a batch to a phone for review, get the keepers back.

ComfyUI loads this file as the pack's entrypoint and calls ``comfy_entrypoint``
(V3 node API, ``comfy_api.latest``; see docs/V3-NOTES.md). K0 ships the client
and the contract only, so the node list is empty; K1 fills ``nodes/``.

``comfy_api`` is imported inside the entrypoint so the package also imports
outside ComfyUI (the tests load it that way).
"""

from __future__ import annotations

__version__ = "0.0.1"


async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension

    from .nodes import NODE_LIST

    class TrainyThingExtension(ComfyExtension):
        async def get_node_list(self):
            return list(NODE_LIST)

    return TrainyThingExtension()
