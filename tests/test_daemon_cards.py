"""Gate K0 against a throwaway daemon: a 12-image batch becomes 12 cards
with kind image, explicit refs, labels and meta; the uploaded PNG round-trips
its prompt and workflow chunks; the token is in no chunk; create, append and
closed; results inline, --out and the bundle; the async face."""

import asyncio
import json
import os
import urllib.request

import pytest

np = pytest.importorskip("numpy")

from comfyui_trainything.cards import QueueClosedError, make_run_id, send_batch  # noqa: E402
from comfyui_trainything.png import chunk_json, decode_png, png_text  # noqa: E402
from comfyui_trainything.split import split_labels  # noqa: E402
from comfyui_trainything.trainything_client import (  # noqa: E402
    MAX_BLOB_BYTES,
    ClientError,
    connect,
    connect_async,
)
from comfyui_trainything.walk import walk_graph  # noqa: E402

LABELS = "\n".join(f"witch {i + 1}" for i in range(12))


def results_json(client, slug):
    return json.loads(client.results(slug, "json"))


def test_twelve_images_become_twelve_cards(daemon, sprite_batch, graph, workflow):
    client = connect(base_url=daemon.url, token=daemon.token, auto_start=False)
    facts = walk_graph(graph, "12")
    run = make_run_id("12")
    handle = send_batch(
        client,
        sprite_batch,
        {"title": "Witch sprites", "slug": "witch-sprites", "prompt": "Keep this one?", "feedback": "on_reject", "project": "comfyui"},
        run,
        labels=LABELS,
        meta='{"cfg": 7.0, "note": "from the meta input"}',
        facts=facts,
        prompt=graph,
        extra_pnginfo={"workflow": workflow},
    )
    assert handle["created"] is True and handle["slug"] == "witch-sprites"
    assert handle["added"] == 12 and handle["total"] == 12
    assert handle["refs"] == [f"{run}-{i}" for i in range(12)]
    assert handle["link"] == f"{daemon.url}/r/witch-sprites"
    assert "token" not in json.dumps(handle).lower() and daemon.token not in json.dumps(handle)

    doc = results_json(client, "witch-sprites")
    assert len(doc["cards"]) == 12
    for i, card in enumerate(doc["cards"]):
        assert card["ref"] == f"{run}-{i}"
        assert card["label"] == f"witch {i + 1}"
        assert [it["kind"] for it in card["items"]] == ["image"]
        assert card["items"][0]["mime"] == "image/png"
        meta = card["meta"]
        assert meta["index"] == i and meta["run"] == run
        assert meta["seed"] == 156680208700286 and meta["steps"] == 28
        assert meta["cfg"] == 7.0  # the meta input wins over the walk
        assert meta["note"] == "from the meta input"
        assert meta["checkpoint"] == "sd_xl_base_1.0.safetensors"
        assert meta["loras"][0]["name"] == "pixel-art-xl.safetensors"
        assert meta["walk"] == "complete"

    # The uploaded PNG, as the daemon stored it (inline results give the
    # blob's local path), round-trips both chunks through Pillow.
    src = doc["cards"][4]["items"][0]["src"]
    data = open(src, "rb").read()
    text = png_text(data)
    assert chunk_json(text, "prompt") == graph
    assert chunk_json(text, "workflow") == workflow
    arr, _ = decode_png(data)
    assert np.abs(arr - sprite_batch[4]).max() <= 1 / 255 + 1e-6

    # The token is in no chunk and nowhere in any uploaded file.
    for card in doc["cards"]:
        raw = open(card["items"][0]["src"], "rb").read()
        assert daemon.token.encode() not in raw
        assert all(daemon.token not in v for v in png_text(raw).values())


def test_append_then_closed(daemon, sprite_batch):
    client = daemon.client()
    run1 = make_run_id("7", now=1000.0)
    run2 = make_run_id("7", now=2000.0)
    cfg = {"title": "Appended", "slug": "Appended Batch"}  # normalized like slug.ts
    h1 = send_batch(client, sprite_batch[:5], cfg, run1)
    h2 = send_batch(client, sprite_batch[5:8], cfg, run2)
    assert h1["created"] is True and h2["created"] is False
    assert h1["slug"] == h2["slug"] == "appended-batch" and h1["queue_id"] == h2["queue_id"]
    assert h2["total"] == 8
    labels = client.results("appended-batch", "labels")
    s = split_labels(labels, h2["refs"], "single")
    assert s.unjudged == [0, 1, 2]  # only run 2's three cards

    client.close("appended-batch")
    with pytest.raises(QueueClosedError, match="appended-batch is closed"):
        send_batch(client, sprite_batch[:1], cfg, make_run_id("7"))
    assert client.status("appended-batch")["cards"] == 8  # nothing uploaded or added
    client.reopen("appended-batch")
    assert client.status("appended-batch")["status"] == "open"


def test_empty_and_random_slug_always_create(daemon, sprite_batch):
    client = daemon.client()
    a = send_batch(client, sprite_batch[:1], {"title": "Same Title"}, make_run_id("1"))
    b = send_batch(client, sprite_batch[:1], {"title": "Same Title", "slug": ""}, make_run_id("1"))
    c = send_batch(client, sprite_batch[:1], {"title": "Same Title", "slug": "random"}, make_run_id("1"))
    assert len({a["queue_id"], b["queue_id"], c["queue_id"]}) == 3
    assert a["slug"] == "same-title" and b["slug"] == "same-title-2"


def test_status_list_results_out_and_bundle(daemon, sprite_batch, tmp_path):
    client = daemon.client()
    h = send_batch(client, sprite_batch[:3], {"title": "Bundle me", "slug": "bundle-me"}, make_run_id("3"))
    st = client.status("bundle-me")
    assert st["cards"] == 3 and st["done"] is False and st["reason"] is None
    assert any(q["slug"] == "bundle-me" for q in client.list_queues())
    assert client.get_queue(h["queue_id"])["queue"]["share_token"] == "bundle-me"
    assert client.find_queue("no-such-queue") is None

    for fmt in ("labels", "summary", "preferences"):
        assert isinstance(client.results("bundle-me", fmt), str)
    written = client.results("bundle-me", "json", out=str(tmp_path / "local" / "results.json"))
    assert written["format"] == "json" and os.path.exists(written["path"])
    assert len(os.listdir(tmp_path / "local" / "blobs")) == 3

    paths = client.bundle("bundle-me", str(tmp_path / "remote"))
    names = sorted(os.path.relpath(p, tmp_path / "remote") for p in paths)
    assert names[0] == "blobs/" + names[0][6:] and "results.json" in names and len(names) == 4
    doc = json.loads((tmp_path / "remote" / "results.json").read_text())
    assert doc["cards"][0]["items"][0]["src"].startswith("blobs/")


def test_errors_carry_status_and_body(daemon):
    client = daemon.client()
    with pytest.raises(ClientError) as err:
        client.status("nope-nope")
    assert err.value.status == 404 and isinstance(err.value.body, dict) and "not found" in str(err.value)
    with pytest.raises(ClientError) as err:
        client.create_queue({"title": "x", "mode": "nonsense"})
    assert err.value.status == 400
    from comfyui_trainything.trainything_client import TrainyThingClient

    with pytest.raises(ClientError) as err:
        TrainyThingClient(daemon.url, "wrong-token").list_queues()
    assert err.value.status == 401


def test_blob_cap_is_checked_before_sending(daemon, monkeypatch):
    client = daemon.client()
    sent = []
    monkeypatch.setattr(client, "_send", lambda *a, **k: sent.append(a))

    class Huge(bytes):
        def __len__(self):
            return MAX_BLOB_BYTES + 1

    with pytest.raises(ClientError, match="blob too large") as err:
        client.upload_blob(Huge(b"x"))
    assert err.value.status == 413 and sent == []


def test_async_face(daemon):
    async def go():
        client = await connect_async(base_url=daemon.url, token=daemon.token, auto_start=False)
        health, queues = await asyncio.gather(client.health(), client.list_queues())
        return health, queues

    health, queues = asyncio.run(go())
    assert health["ok"] is True and isinstance(queues, list)


def test_health_needs_no_token(daemon):
    with urllib.request.urlopen(f"{daemon.url}/health", timeout=5) as res:
        assert json.loads(res.read())["ok"] is True
