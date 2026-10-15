"""The nodes: Review, Send, Wait, Load and Filter.

``NODE_LIST`` is resolved on first use (PEP 562), because the node classes
need ``comfy_api`` and torch while ``nodes.common`` must also import in a
plain Python (the tests do). ComfyUI reads it from ``comfy_entrypoint``."""

from typing import Any, List


def node_list() -> List[Any]:
    from .filter import TrainyThingFilter
    from .load import TrainyThingLoad
    from .review import TrainyThingReview
    from .send import TrainyThingSend
    from .wait import TrainyThingWait

    return [TrainyThingReview, TrainyThingSend, TrainyThingWait, TrainyThingLoad, TrainyThingFilter]


def __getattr__(name: str) -> Any:
    if name == "NODE_LIST":
        return node_list()
    raise AttributeError(name)
