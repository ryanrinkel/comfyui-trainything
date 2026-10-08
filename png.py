"""PNG in and out (PLAN-COMFY §6.3).

``encode_png`` writes an image the way ComfyUI's Save Image does
(``nodes.py`` ``SaveImage.save_images``): ``255 * image``, clipped, uint8,
and a ``PngInfo`` with a ``prompt`` text chunk (``json.dumps(prompt)``) and
one chunk per key of ``extra_pnginfo`` (``json.dumps(value)``), which is how
``workflow`` gets in; compress level 4. A blob dropped on the canvas restores
the graph that made it.

``decode_png`` reads one back into a ``[H, W, C]`` float32 array in 0..1 (for
Load, K2) with its text chunks.

numpy and Pillow are imported lazily, and torch never: a tensor is accepted
by duck typing (``.cpu().numpy()``), so the module imports without either.
"""

from __future__ import annotations

import io
import json
from typing import Any, Dict, Iterator, Optional, Tuple

#: ``SaveImage.compress_level`` in ComfyUI's nodes.py.
COMPRESS_LEVEL = 4


def to_numpy(image: Any) -> Any:
    """A numpy array from a numpy array or a torch tensor (any device)."""
    import numpy as np

    if hasattr(image, "detach") and hasattr(image, "cpu"):
        image = image.detach().cpu().numpy()
    elif hasattr(image, "cpu") and hasattr(image, "numpy"):
        image = image.cpu().numpy()
    return np.asarray(image)


def iter_batch(images: Any) -> Iterator[Any]:
    """``[B, H, W, C]`` -> each ``[H, W, C]`` as numpy; a single ``[H, W, C]`` too."""
    arr = to_numpy(images)
    if arr.ndim == 3:
        yield arr
        return
    if arr.ndim != 4:
        raise ValueError(f"expected [B, H, W, C] or [H, W, C], got shape {tuple(arr.shape)}")
    for i in range(arr.shape[0]):
        yield arr[i]


def pnginfo_for(prompt: Any = None, extra_pnginfo: Optional[Dict[str, Any]] = None) -> Any:
    """The ``PngInfo`` Save Image builds; None when there is nothing to write
    (Save Image passes None under ``--disable-metadata``)."""
    from PIL.PngImagePlugin import PngInfo

    if prompt is None and not extra_pnginfo:
        return None
    metadata = PngInfo()
    if prompt is not None:
        metadata.add_text("prompt", json.dumps(prompt))
    if extra_pnginfo is not None:
        for x in extra_pnginfo:
            metadata.add_text(x, json.dumps(extra_pnginfo[x]))
    return metadata


def encode_png(
    image: Any,
    prompt: Any = None,
    extra_pnginfo: Optional[Dict[str, Any]] = None,
    compress_level: int = COMPRESS_LEVEL,
) -> bytes:
    """One ``[H, W, C]`` float image (0..1; numpy or torch) -> PNG bytes with
    the ``prompt`` and ``workflow`` chunks. ``extra_pnginfo`` is ComfyUI's
    hidden ``EXTRA_PNGINFO`` (``{"workflow": {...}}``), written key by key."""
    import numpy as np
    from PIL import Image

    arr = to_numpy(image)
    if arr.ndim == 4 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim != 3:
        raise ValueError(f"expected one [H, W, C] image, got shape {tuple(arr.shape)}")
    if arr.shape[2] == 1:
        arr = arr[:, :, 0]
    i = 255.0 * arr
    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=pnginfo_for(prompt, extra_pnginfo), compress_level=compress_level)
    return buf.getvalue()


def png_text(data: bytes) -> Dict[str, str]:
    """The PNG's text chunks, raw (``prompt`` and ``workflow`` are JSON strings)."""
    from PIL import Image

    with Image.open(io.BytesIO(data)) as img:
        img.load()
        return dict(getattr(img, "text", {}) or {})


def decode_png(data: bytes, channels: int = 3) -> Tuple[Any, Dict[str, str]]:
    """PNG bytes -> (``[H, W, channels]`` float32 array in 0..1, text chunks).
    ``channels`` 3 is ComfyUI's IMAGE; 4 keeps alpha (Load Image's mask)."""
    import numpy as np
    from PIL import Image, ImageOps

    with Image.open(io.BytesIO(data)) as img:
        img.load()
        text = dict(getattr(img, "text", {}) or {})
        img = ImageOps.exif_transpose(img) or img
        if img.mode == "I":
            img = img.point(lambda v: v * (1 / 255))
        img = img.convert("RGBA" if channels == 4 else "RGB")
        arr = np.asarray(img).astype(np.float32) / 255.0
    return arr, text


def chunk_json(text: Dict[str, str], key: str) -> Any:
    """``json.loads`` of one text chunk, or None when absent or not JSON."""
    raw = text.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except ValueError:
        return None
