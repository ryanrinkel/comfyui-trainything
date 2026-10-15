"""The graph walk.

ComfyUI's hidden ``prompt`` is the API-format graph, ``{id: {class_type,
inputs}}``, where a linked input is ``[source_id, output_index]``. From the
node's own id the walk follows ``images`` back to the decoder, ``samples`` to
the sampler, ``latent_image`` to the empty latent, ``positive`` / ``negative``
to the text encoders, and ``model`` through LoRA loaders to the checkpoint.

It is best effort. Each chain stops after ``MAX_HOPS`` links or at a class it
does not know, and ``walk`` says ``"complete"`` or where it stopped. The node's
``meta`` input overrides anything it got wrong. Pure: no ComfyUI import.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

MAX_HOPS = 12

#: Image-to-image steps between the decoder and the node: followed, not read.
IMAGE_PASSTHROUGH = {
    "ImageScale": "image",
    "ImageScaleBy": "image",
    "ImageScaleToTotalPixels": "image",
    "ImageUpscaleWithModel": "image",
    "ImageSharpen": "image",
    "ImageBlur": "image",
    "ImageInvert": "image",
    "RepeatImageBatch": "image",
}
DECODERS = {"VAEDecode": "samples", "VAEDecodeTiled": "samples"}
SAMPLERS = {"KSampler", "KSamplerAdvanced"}
EMPTY_LATENTS = {"EmptyLatentImage", "EmptySD3LatentImage", "EmptyHunyuanLatentVideo"}
TEXT_ENCODERS = {"CLIPTextEncode": "text", "CLIPTextEncodeSDXL": "text_g", "CLIPTextEncodeFlux": "t5xxl"}
CHECKPOINTS = {
    "CheckpointLoaderSimple": "ckpt_name",
    "CheckpointLoader": "ckpt_name",
    "ImageOnlyCheckpointLoader": "ckpt_name",
    "UNETLoader": "unet_name",
}
LORAS = {"LoraLoader", "LoraLoaderModelOnly"}
#: Model patches that sit between a LoRA and the sampler; followed through ``model``.
MODEL_PASSTHROUGH = {"ModelSamplingDiscrete", "ModelSamplingSD3", "ModelSamplingFlux", "FreeU", "FreeU_V2"}
#: Scalar keys a "primitive" node holds its value under (seed / text widgets
#: converted to inputs and fed from another node).
SCALAR_KEYS = ("value", "seed", "noise_seed", "int", "float", "string", "text", "number")


def is_link(value: Any) -> bool:
    return (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and isinstance(value[0], (str, int))
        and isinstance(value[1], int)
        and not isinstance(value[1], bool)
    )


class _Walk:
    def __init__(self, prompt: Dict[str, Any]) -> None:
        self.prompt = prompt
        self.facts: Dict[str, Any] = {}
        self.stops: List[str] = []

    def node(self, node_id: Any) -> Optional[Dict[str, Any]]:
        n = self.prompt.get(str(node_id))
        return n if isinstance(n, dict) else None

    def stop(self, why: str) -> None:
        if why not in self.stops:
            self.stops.append(why)

    def follow(self, node_id: Any, key: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """The node an input links to, or (None, None) and a stop."""
        n = self.node(node_id)
        value = (n or {}).get("inputs", {}).get(key)
        if not is_link(value):
            self.stop(f"node {node_id} has no linked {key}")
            return None, None
        target = str(value[0])
        found = self.node(target)
        if found is None:
            self.stop(f"node {node_id}.{key} links to missing node {target}")
            return None, None
        return target, found

    def scalar(self, node_id: str, key: str) -> Any:
        """A literal input, or the value of a primitive node it links to."""
        value = self.node(node_id).get("inputs", {}).get(key)  # type: ignore[union-attr]
        if not is_link(value):
            return value
        src = self.node(value[0])
        if src is None:
            return None
        inputs = src.get("inputs", {})
        literals = {k: v for k, v in inputs.items() if not is_link(v)}
        for k in SCALAR_KEYS:
            if k in literals:
                return literals[k]
        if len(literals) == 1:
            return next(iter(literals.values()))
        self.stop(f"node {node_id}.{key} comes from {src.get('class_type')} (node {value[0]}), not read")
        return None

    # -- chains -------------------------------------------------------------

    def images(self, start_id: str, key: str) -> None:
        nid, n = self.follow(start_id, key)
        hops = 1
        while n is not None:
            if hops > MAX_HOPS:
                self.stop(f"hop limit ({MAX_HOPS}) on the images chain at node {nid}")
                return
            ct = n.get("class_type")
            if ct in DECODERS:
                self.sampler(nid, DECODERS[ct], hops)
                return
            if ct in IMAGE_PASSTHROUGH:
                nid, n = self.follow(nid, IMAGE_PASSTHROUGH[ct])
                hops += 1
                continue
            self.stop(f"unknown class {ct} (node {nid}) on the images chain")
            return

    def sampler(self, decoder_id: str, key: str, hops: int) -> None:
        nid, n = self.follow(decoder_id, key)
        hops += 1
        if n is None:
            return
        ct = n.get("class_type")
        if ct not in SAMPLERS:
            self.stop(f"unknown class {ct} (node {nid}) where a sampler was expected")
            return
        seed_key = "noise_seed" if ct == "KSamplerAdvanced" else "seed"
        for fact, k in (
            ("seed", seed_key),
            ("steps", "steps"),
            ("cfg", "cfg"),
            ("sampler", "sampler_name"),
            ("scheduler", "scheduler"),
            ("denoise", "denoise"),
        ):
            if k in n.get("inputs", {}):
                v = self.scalar(nid, k)
                if v is not None:
                    self.facts[fact] = v
        self.latent(nid, hops)
        self.text(nid, "positive", "prompt_text", hops)
        self.text(nid, "negative", "negative_text", hops)
        self.model(nid, hops)

    def latent(self, sampler_id: str, hops: int) -> None:
        nid, n = self.follow(sampler_id, "latent_image")
        if n is None:
            return
        ct = n.get("class_type")
        if ct not in EMPTY_LATENTS:
            self.stop(f"latent_image comes from {ct} (node {nid}), not an empty latent")
            return
        for k in ("batch_size", "width", "height"):
            if k in n.get("inputs", {}):
                v = self.scalar(nid, k)
                if v is not None:
                    self.facts[k] = v

    def text(self, sampler_id: str, key: str, fact: str, hops: int) -> None:
        nid, n = self.follow(sampler_id, key)
        if n is None:
            return
        ct = n.get("class_type")
        if ct not in TEXT_ENCODERS:
            self.stop(f"{key} comes from {ct} (node {nid}), not a text encoder")
            return
        v = self.scalar(nid, TEXT_ENCODERS[ct])
        if v is not None:
            self.facts[fact] = v

    def model(self, sampler_id: str, hops: int) -> None:
        loras: List[Dict[str, Any]] = []
        self.facts["loras"] = loras
        nid, n = self.follow(sampler_id, "model")
        chain = 1
        while n is not None:
            if chain > MAX_HOPS:
                self.stop(f"hop limit ({MAX_HOPS}) on the model chain at node {nid}")
                break
            ct = n.get("class_type")
            if ct in CHECKPOINTS:
                v = self.scalar(nid, CHECKPOINTS[ct])
                if v is not None:
                    self.facts["checkpoint"] = v
                break
            if ct in LORAS:
                lora: Dict[str, Any] = {"name": self.scalar(nid, "lora_name")}
                lora["strength_model"] = self.scalar(nid, "strength_model")
                if ct == "LoraLoader":
                    lora["strength_clip"] = self.scalar(nid, "strength_clip")
                loras.append(lora)
            elif ct not in MODEL_PASSTHROUGH:
                self.stop(f"unknown class {ct} (node {nid}) on the model chain")
                break
            nid, n = self.follow(nid, "model")
            chain += 1
        # Walked from the sampler back; report them in the order they apply.
        loras.reverse()


def walk_graph(prompt: Optional[Dict[str, Any]], node_id: Any, images_input: str = "images") -> Dict[str, Any]:
    """The facts lists, from the node ``node_id`` back along its
    ``images_input``. Keys present only when found: ``seed``, ``steps``,
    ``cfg``, ``sampler``, ``scheduler``, ``denoise``, ``prompt_text``,
    ``negative_text``, ``checkpoint``, ``batch_size``, ``width``, ``height``;
    ``loras`` (a list of ``{name, strength_model, strength_clip}``) when the
    model chain was reached; always ``walk``: ``"complete"`` or where it
    stopped. ``index`` is the batch position, added per card (cards.py)."""
    if not isinstance(prompt, dict) or not prompt:
        return {"walk": "no graph (the hidden prompt was empty)"}
    w = _Walk(prompt)
    if w.node(node_id) is None:
        return {"walk": f"node {node_id} is not in the graph"}
    w.images(str(node_id), images_input)
    facts = dict(w.facts)
    facts["walk"] = "complete" if not w.stops else "stopped: " + "; ".join(w.stops)
    return facts
