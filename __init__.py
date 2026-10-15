"""TrainyThing for ComfyUI: send a batch to a phone for review, get the keepers back.

ComfyUI loads this file as the pack's entrypoint and calls ``comfy_entrypoint``
(V3 node API, ``comfy_api.latest``): Review, Send, Wait,
Load and Filter (``nodes/``).

``comfy_api`` is imported inside the entrypoint so the package also imports
outside ComfyUI (the tests load it that way).
"""

from __future__ import annotations

import re
from typing import Optional, Tuple

__version__ = "0.1.0"

#: ComfyUI serves this folder at /extensions/<pack>/ and loads its .js files
#: (web/trainything.js: the nodes' default colour).
WEB_DIRECTORY = "./web"

#: pyproject.toml's ``requires-comfyui`` (``>=`` this); a test keeps them equal.
REQUIRES_COMFYUI = "0.39.0"


def _version_tuple(text: str) -> Tuple[int, ...]:
    return tuple(int(part) for part in re.findall(r"\d+", text)[:3])


def comfyui_too_old(running: Optional[str], pin: str = REQUIRES_COMFYUI) -> Optional[str]:
    """The one console line when ``running`` is older than the pin, else None
    (also None when the version is unknown or unreadable)."""
    have = _version_tuple(running or "")
    if not have or have >= _version_tuple(pin):
        return None
    return (f"[TrainyThing] this ComfyUI is {running}; the pack was built and tested on >= {pin} "
            f"(the V3 node API it uses may differ). Update ComfyUI if a TrainyThing node fails.")


def _check_comfyui_version() -> None:
    """Warn once at import, never raise: a pack that fails to import shows
    nothing on the canvas, which is worse than a node that may work."""
    try:
        import comfyui_version  # ComfyUI's root module; absent outside ComfyUI

        line = comfyui_too_old(str(getattr(comfyui_version, "__version__", "")))
    except Exception:
        return
    if line:
        print(line, flush=True)


_check_comfyui_version()


async def comfy_entrypoint():
    from comfy_api.latest import ComfyExtension

    from .nodes import NODE_LIST

    class TrainyThingExtension(ComfyExtension):
        async def get_node_list(self):
            return list(NODE_LIST)

    return TrainyThingExtension()
