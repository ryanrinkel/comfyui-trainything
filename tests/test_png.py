"""The PNG carries `prompt` and `workflow` as Save Image writes them, and
decodes back to the same pixels."""

import json

import pytest

np = pytest.importorskip("numpy")

from comfyui_trainything.png import chunk_json, decode_png, encode_png, iter_batch, png_text  # noqa: E402


def test_round_trip_chunks_and_pixels(sprite_batch, graph, workflow):
    data = encode_png(sprite_batch[0], prompt=graph, extra_pnginfo={"workflow": workflow})
    text = png_text(data)
    assert set(text) == {"prompt", "workflow"}
    # Exactly json.dumps, as SaveImage writes them.
    assert text["prompt"] == json.dumps(graph)
    assert text["workflow"] == json.dumps(workflow)
    assert chunk_json(text, "prompt") == graph
    arr, text2 = decode_png(data)
    assert text2 == text
    assert arr.shape == (64, 64, 3) and arr.dtype == np.float32
    assert np.abs(arr - sprite_batch[0]).max() <= 1 / 255 + 1e-6


def test_no_metadata_writes_no_chunks(sprite_batch):
    assert png_text(encode_png(sprite_batch[1])) == {}


def test_shapes():
    one = np.zeros((1, 4, 5, 3), dtype=np.float32)
    assert len(list(iter_batch(one))) == 1
    assert len(list(iter_batch(one[0]))) == 1
    gray = encode_png(np.ones((4, 5, 1), dtype=np.float32))
    arr, _ = decode_png(gray)
    assert arr.shape == (4, 5, 3) and arr.min() == 1.0
    over = encode_png(np.full((2, 2, 3), 2.0, dtype=np.float32))  # clipped like SaveImage
    assert decode_png(over)[0].max() == 1.0
    with pytest.raises(ValueError):
        list(iter_batch(np.zeros((2, 2))))


def test_torch_tensor(sprite_batch):
    torch = pytest.importorskip("torch")
    t = torch.from_numpy(sprite_batch.copy())
    assert encode_png(t[3]) == encode_png(sprite_batch[3])
    assert len(list(iter_batch(t))) == 12


def test_qr_dependency_makes_a_png():
    """One pure-Python QR encoder (segno, no dependencies of
    its own) renders the link to a PNG Pillow can read."""
    segno = pytest.importorskip("segno")
    import io

    from PIL import Image

    buf = io.BytesIO()
    segno.make("http://127.0.0.1:4747/r/witch-sprites", error="m").save(buf, kind="png", scale=6, border=2)
    img = Image.open(io.BytesIO(buf.getvalue()))
    assert img.size[0] == img.size[1] and img.size[0] > 100
