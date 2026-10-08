"""The twelve findings of the K0-K4 review, without ComfyUI or a daemon:
fakes for the client, numpy arrays for the batches. The daemon half is in
test_review_fixes_daemon.py, the ComfyUI half in test_review_fixes_comfy.py.

Finding -> test:
   1 pair Wait without images_b / a short latent fail before any poll . test_f1_*
   2 an append checks mode, options and min_reviews ................... test_f2_*
   3 the wait retries status 0 / 5xx, fails at once on 4xx ............ test_f3_*
   4 the split before the folder; the folder best effort; by queue_id . test_f4_*
   5 auto-start: the PATH binary first; no token without the URL ...... test_f5_*
   7 public: share before any upload; a failure is the local link ..... test_f7_*
   8 Load's accepted_meta gains position; index untouched ............. test_f8_*
   9 results?out= that leaves no fresh file here -> the bundle ........ test_f9_*
  10 the fingerprint carries last_activity_at ......................... test_f10_*
  11 the bundle is streamed, the traversal guard kept ................. test_f11_*
  12 two run ids in the same millisecond differ ....................... test_f12_*
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import re
import time
import zipfile

import pytest

np = pytest.importorskip("numpy")

from comfyui_trainything import cards as cards_mod  # noqa: E402
from comfyui_trainything import trainything_client as tc  # noqa: E402
from comfyui_trainything.cards import QueueModeError, ensure_queue, make_run_id, send_batch  # noqa: E402
from comfyui_trainything.nodes import common as c  # noqa: E402
from comfyui_trainything.split import split_labels  # noqa: E402
from comfyui_trainything.trainything_client import ClientError  # noqa: E402


def batch(n, h=4, w=4):
    return np.zeros((n, h, w, 3), np.float32)


class NoPolls:
    """An async client that fails the test if the wait ever polls."""

    def __init__(self):
        self.calls = 0

    async def status(self, qid):
        self.calls += 1
        raise AssertionError("polled before the inputs were checked")


# ---------------------------------------------------------------------------
# 1
# ---------------------------------------------------------------------------


def test_f1_pair_wait_without_images_b_fails_before_polling():
    client = NoPolls()
    handle = {"queue_id": "q", "slug": "s", "mode": "pair", "refs": ["r-0", "r-1", "r-2"]}
    with pytest.raises(ValueError, match="pair mode: wire images_b"):
        asyncio.run(c.wait_and_split(client, handle, batch(3), timeout_minutes=60))
    with pytest.raises(ValueError, match="images_b has 2 images for 3"):
        asyncio.run(c.wait_and_split(client, handle, batch(3), images_b=batch(2), timeout_minutes=60))
    assert client.calls == 0


def test_f1_short_latent_fails_before_polling():
    client = NoPolls()
    handle = {"queue_id": "q", "slug": "s", "mode": "single", "refs": ["r-0", "r-1", "r-2"]}
    latent = {"samples": np.zeros((2, 4, 1, 1), np.float32)}
    with pytest.raises(ValueError, match="latent has 2 samples for 3 images"):
        asyncio.run(c.wait_and_split(client, handle, batch(3), latent=latent, timeout_minutes=60))
    assert client.calls == 0


def test_f1_review_checks_before_send():
    # Review calls this before send_from_inputs (nodes/review.py).
    with pytest.raises(ValueError, match="pair mode: wire images_b"):
        c.check_wait_inputs("pair", batch(3))
    with pytest.raises(ValueError, match="latent has 1 samples"):
        c.check_wait_inputs("single", batch(3), {"samples": np.zeros((1, 4, 1, 1))})
    c.check_wait_inputs("pair", batch(3), {"samples": np.zeros((5, 4, 1, 1))}, batch(3))  # more samples is fine
    c.check_wait_inputs("single", batch(3), None, None)


# ---------------------------------------------------------------------------
# 2
# ---------------------------------------------------------------------------


class QueueClient:
    """A sync client with one open queue under ``slug``; records calls."""

    base_url = "http://127.0.0.1:1"

    def __init__(self, queue, patch_error=None):
        self.queue = dict(queue)
        self.calls = []
        self.patch_error = patch_error

    def find_queue(self, slug):
        self.calls.append(("find", slug))
        return {"queue": dict(self.queue), "local_url": f"{self.base_url}/r/{slug}", "share_url": None}

    def patch_queue(self, qid, patch):
        self.calls.append(("patch", qid, patch))
        if self.patch_error is not None:
            raise self.patch_error
        self.queue.update(patch)
        return {"queue": dict(self.queue)}

    def create_queue(self, body):  # pragma: no cover - the tests append
        raise AssertionError("created a queue")


def queue_row(**kw):
    row = {"id": "q1", "share_token": "sprites", "status": "open", "mode": "single", "options": None, "completion": {}}
    row.update(kw)
    return row


def test_f2_append_with_another_mode_is_an_error():
    client = QueueClient(queue_row(mode="single"))
    cfg = c.queue_config(title="t", slug="sprites", mode="choose", options="a:A, b:B")
    with pytest.raises(QueueModeError, match="sprites is a single queue and this node's mode is choose"):
        ensure_queue(client, cfg)
    assert not [k for k in client.calls if k[0] == "patch"]


def test_f2_append_sets_a_missing_min_reviews_and_keeps_the_deadline(capsys):
    client = QueueClient(queue_row(completion={"deadline": "2030-01-01T00:00:00Z"}))
    q = ensure_queue(client, c.queue_config(title="t", slug="sprites", min_reviews=2))
    assert ("patch", "q1", {"completion": {"deadline": "2030-01-01T00:00:00Z", "min_reviews_per_card": 2}}) in client.calls
    assert q["queue"]["completion"]["min_reviews_per_card"] == 2 and q["created"] is False
    assert "had no min_reviews; set it to 2" in capsys.readouterr().out


def test_f2_a_refused_patch_is_a_console_line(capsys):
    client = QueueClient(queue_row(), patch_error=ClientError("nope", 409, None))
    q = ensure_queue(client, c.queue_config(title="t", slug="sprites"))
    assert q["queue"]["id"] == "q1"
    assert "the wait can only end on a close or the timeout" in capsys.readouterr().out


def test_f2_a_queue_with_its_own_min_reviews_keeps_it(capsys):
    client = QueueClient(queue_row(completion={"min_reviews_per_card": 3}))
    ensure_queue(client, c.queue_config(title="t", slug="sprites", min_reviews=1))
    assert not [k for k in client.calls if k[0] == "patch"]
    assert "keeps its own min_reviews 3 (the widget says 1)" in capsys.readouterr().out


def test_f2_other_options_are_the_queues(capsys):
    opts = [{"id": "hair", "label": "Hair"}, {"id": "ok", "label": "Fine"}]
    client = QueueClient(queue_row(mode="choose", options=opts, completion={"min_reviews_per_card": 1}))
    q = ensure_queue(client, c.queue_config(title="t", slug="sprites", mode="choose", options="x:X, y:Y"))
    assert q["queue"]["options"] == opts
    assert "keeps its own options (hair, ok)" in capsys.readouterr().out
    ensure_queue(client, c.queue_config(title="t", slug="sprites", mode="choose", options="hair:Hair, ok:Fine"))
    assert "options" not in capsys.readouterr().out  # the same list: no line


# ---------------------------------------------------------------------------
# 3
# ---------------------------------------------------------------------------


class Flaky:
    def __init__(self, errors, done=True):
        self.errors = list(errors)
        self.polls = 0
        self.done = done

    async def status(self, qid):
        self.polls += 1
        if self.errors:
            raise self.errors.pop(0)
        return {"cards": 2, "judged": 2, "done": self.done}


HANDLE = {"queue_id": "q1", "slug": "s", "refs": ["a", "b"]}


def test_f3_status_0_and_5xx_are_retried_with_one_line(capsys):
    client = Flaky([ClientError("cannot reach", 0), ClientError("boom", 503), ClientError("cannot reach", 0)])
    res = asyncio.run(c.wait_until_done(client, HANDLE, 30, interrupted=lambda: False, poll_s=0.05, retry_base_s=0.01))
    assert res.timed_out is False and client.polls == 4
    out = capsys.readouterr().out
    assert out.count("retrying until the timeout") == 1


@pytest.mark.parametrize("status", [401, 403, 404, 409])
def test_f3_4xx_fails_at_once(status):
    client = Flaky([ClientError("no", status)])
    with pytest.raises(ClientError):
        asyncio.run(c.wait_until_done(client, HANDLE, 30, interrupted=lambda: False, retry_base_s=0.01))
    assert client.polls == 1


def test_f3_backoff_doubles_to_the_cap_and_gives_up_at_the_deadline(monkeypatch):
    client = Flaky([ClientError("down", 0)] * 1000)
    t0 = time.monotonic()
    with pytest.raises(ClientError, match="down"):
        asyncio.run(c.wait_until_done(client, HANDLE, 0.5, interrupted=lambda: False, retry_base_s=0.05))
    assert 0.4 < time.monotonic() - t0 < 2.0
    assert 3 <= client.polls <= 6  # 0.05, 0.1, 0.2, then the deadline: not a busy loop
    assert [min(c.RETRY_MAX_S, c.RETRY_BASE_S * 2 ** k) for k in range(6)] == [2, 4, 8, 16, 30, 30]


# ---------------------------------------------------------------------------
# 4
# ---------------------------------------------------------------------------


class Collectable:
    """Async client with results for one run of two single cards."""

    def __init__(self):
        self.sync = self
        self.base_url = "http://127.0.0.1:1"

    async def status(self, qid):
        return {"cards": 2, "judged": 2, "done": True}

    async def results(self, qid, fmt, **kw):
        if fmt == "labels":
            return "".join(json.dumps({"ref": r, "verdict": "accept", "n": 1, "contested": False, "notes": []}) + "\n"
                           for r in ("r-0", "r-1"))
        if fmt == "json":
            return json.dumps({"queue": {"mode": "single"}, "cards": [{"ref": "r-0"}, {"ref": "r-1"}]})
        return "# summary\n"


def test_f4_a_failed_folder_never_loses_the_wait(monkeypatch, capsys):
    order = []

    def fake_split(run, images, latent=None, images_b=None, default_index=True):
        order.append("split")
        return ["the split"]

    def broken_folder(*a, **k):
        order.append("folder")
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(c, "split_outputs", fake_split)
    monkeypatch.setattr(c, "write_run_folder_sync", broken_folder)
    handle = {"queue_id": "q1", "slug": "s", "mode": "single", "refs": ["r-0", "r-1"]}
    outs, blocked = asyncio.run(c.wait_and_split(Collectable(), handle, batch(2), timeout_minutes=1))
    assert outs == ["the split"] and blocked is False
    assert order == ["split", "folder"]
    out = capsys.readouterr().out
    assert "could not write the run's folder (OSError" in out and "No space left on device" in out


class Recorder:
    """A sync client that records which id every call used; remote, so the
    folder comes from the bundle."""

    base_url = "http://trainything.invalid:4747"

    def __init__(self):
        self.ids = []

    def results(self, qid, fmt, **kw):
        self.ids.append(qid)
        return ""

    def bundle(self, qid, dest):
        self.ids.append(qid)
        open(os.path.join(dest, "results.json"), "w").write("{}")
        return []


def test_f4_the_folder_is_read_by_queue_id(tmp_path):
    client = Recorder()
    out = c.write_run_folder_sync(client, "q-123", "old-slug", "pair", str(tmp_path))
    assert set(client.ids) == {"q-123"} and len(client.ids) == 4  # bundle, labels, summary, preferences
    assert out["folder"] == str(tmp_path / "old-slug") and out["how"] == "bundle"


# ---------------------------------------------------------------------------
# 5
# ---------------------------------------------------------------------------


def test_f5_the_path_binary_comes_first(monkeypatch, capsys):
    runs = []
    monkeypatch.setattr(tc.shutil, "which", lambda name: {"trainything": "/opt/bin/trainything", "npx": "/opt/bin/npx"}.get(name))
    monkeypatch.setattr(tc.subprocess, "run", lambda cmd, **kw: runs.append((cmd, kw)))
    tc._run_autostart("http://127.0.0.1:4747", None, 15.0)
    assert runs[0][0] == ["/opt/bin/trainything", "list"] and runs[0][1]["timeout"] == 15.0
    assert "starting one with trainything list (/opt/bin/trainything)" in capsys.readouterr().out


def test_f5_npx_only_when_there_is_no_binary(monkeypatch, capsys):
    runs = []
    monkeypatch.setattr(tc.shutil, "which", lambda name: "/opt/bin/npx" if name == "npx" else None)
    monkeypatch.setattr(tc.subprocess, "run", lambda cmd, **kw: runs.append(cmd))
    tc._run_autostart("http://127.0.0.1:4747", None, 15.0)
    assert runs == [["/opt/bin/npx", "-y", "trainything", "list"]]
    assert "starting one with npx -y trainything list" in capsys.readouterr().out
    monkeypatch.setattr(tc.shutil, "which", lambda name: None)
    tc._run_autostart("http://127.0.0.1:4747", None, 15.0)
    assert len(runs) == 1 and "neither trainything nor npx" in capsys.readouterr().out


def test_f5_the_token_goes_along_only_with_the_url(monkeypatch):
    monkeypatch.setenv("TRAINYTHING_TOKEN", "secret-token")
    env = tc.autostart_env("http://127.0.0.1:4747", None)
    assert "TRAINYTHING_TOKEN" not in env and env["TRAINYTHING_URL"] == "http://127.0.0.1:4747"
    monkeypatch.setenv("TRAINYTHING_URL", "http://127.0.0.1:5000")
    env = tc.autostart_env("http://127.0.0.1:5000", None)
    assert env["TRAINYTHING_TOKEN"] == "secret-token"


# ---------------------------------------------------------------------------
# 7
# ---------------------------------------------------------------------------


class SendClient:
    base_url = "http://127.0.0.1:1"

    def __init__(self, share_error=None):
        self.calls = []
        self.share_error = share_error

    def find_queue(self, slug):
        return None

    def create_queue(self, body):
        self.calls.append("create")
        return {"queue": {"id": "q1", "share_token": "pub", "mode": "single"},
                "local_url": "http://127.0.0.1:1/r/pub", "share_url": None}

    def share(self, qid, slug=None):
        self.calls.append("share")
        if self.share_error is not None:
            raise self.share_error
        return {"share_url": "https://x.trycloudflare.com/r/pub", "changed": False}

    def upload_blob(self, data, mime):
        self.calls.append("upload")
        return {"sha256": f"{len(self.calls):064x}"}

    def add_cards(self, qid, cards):
        self.calls.append("add")
        return {"added": len(cards), "total": len(cards)}


def test_f7_share_comes_before_any_upload():
    client = SendClient()
    h = send_batch(client, batch(2), {"title": "pub"}, "run", public=True)
    assert client.calls == ["create", "share", "upload", "upload", "add"]
    assert h["link"] == "https://x.trycloudflare.com/r/pub"


def test_f7_a_failed_share_is_the_local_link(capsys):
    client = SendClient(share_error=ClientError("cloudflared did not report a trycloudflare.com URL within 60000 ms", 500))
    h = send_batch(client, batch(2), {"title": "pub"}, "run", public=True)
    assert client.calls.count("add") == 1 and h["added"] == 2
    assert h["link"] == "http://127.0.0.1:1/r/pub" and h["share_url"] is None
    assert "public: could not share pub" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# 8
# ---------------------------------------------------------------------------


def test_f8_load_meta_has_position_and_keeps_index():
    doc = {"queue": {"mode": "single"}, "cards": [
        {"ref": "r1-0", "items": [{"kind": "image", "src": "blobs/a.png"}], "meta": {"index": 0, "seed": 1}},
        {"ref": "r1-1", "items": [{"kind": "image", "src": "blobs/b.png"}], "meta": {"index": 1, "seed": 1}},
        {"ref": "r2-0", "items": [{"kind": "image", "src": "blobs/c.png"}], "meta": {"index": 0, "seed": 9}},
        {"ref": "cli-0", "items": [{"kind": "image", "src": "blobs/d.png"}], "meta": {}},
    ]}
    labels = "".join(json.dumps({"ref": r, "verdict": "accept"}) + "\n" for r in ("r1-1", "r2-0", "cli-0"))
    run, _ = c.load_run(doc, labels, "")
    metas = json.loads(c.meta_json(run, run.split.accepted, default_index=False))
    assert [m["position"] for m in metas] == run.split.accepted == [1, 2, 3]
    assert [m.get("index") for m in metas] == [1, 0, None]  # Send's index, never Load's position
    # Wait (default_index): position is the batch index, and a missing index is filled with it.
    wait = json.loads(c.meta_json(run, [3]))
    assert wait == [{"index": 3, "ref": "cli-0", "position": 3}]


# ---------------------------------------------------------------------------
# 9
# ---------------------------------------------------------------------------


class ElsewhereDaemon:
    """A "local" daemon that answers ``results?out=`` but writes the file
    into a filesystem this process cannot see."""

    base_url = "http://127.0.0.1:1"

    def __init__(self):
        self.bundled = False

    def results(self, qid, fmt, out=None, **kw):
        if out is not None:
            return {"path": out, "bytes": 10, "format": "json"}
        return ""

    def bundle(self, qid, dest):
        self.bundled = True
        open(os.path.join(dest, "results.json"), "w").write('{"fresh": true}')
        return []


def test_f9_out_that_lands_elsewhere_falls_back_to_the_bundle(tmp_path, capsys):
    folder = tmp_path / "s"
    folder.mkdir()
    stale = folder / "results.json"
    stale.write_text('{"stale": true}')
    os.utime(stale, (time.time() - 3600, time.time() - 3600))
    client = ElsewhereDaemon()
    out = c.write_run_folder_sync(client, "q1", "s", "single", str(tmp_path))
    assert out["how"] == "bundle" and client.bundled
    assert json.loads(stale.read_text()) == {"fresh": True}
    assert "somewhere this machine cannot see" in capsys.readouterr().out


def test_f9_a_fresh_file_counts_as_local(tmp_path):
    class Here(ElsewhereDaemon):
        def results(self, qid, fmt, out=None, **kw):
            if out is not None:
                open(out, "w").write("{}")
                return {"path": out}
            return ""

    client = Here()
    assert c.write_run_folder_sync(client, "q1", "s", "single", str(tmp_path))["how"] == "local"
    assert client.bundled is False


# ---------------------------------------------------------------------------
# 10
# ---------------------------------------------------------------------------


def test_f10_fingerprint_moves_with_last_activity(monkeypatch):
    state = {"at": "2026-10-08T10:00:00.000Z"}

    class FP:
        base_url = "http://127.0.0.1:1"

        async def find_queue(self, slug):
            return {"queue": {"id": "q1", "closed_at": None}}

        async def status(self, qid):
            return {"cards": 3, "judgments": 2, "status": "open"}

        async def list_queues(self, project=None):
            return [{"queue_id": "other", "last_activity_at": "x"}, {"queue_id": "q1", "last_activity_at": state["at"]}]

    async def fake_connect(**kw):
        return FP()

    monkeypatch.setattr(c, "connect_async", fake_connect)
    first = asyncio.run(c.load_fingerprint("s"))
    assert asyncio.run(c.load_fingerprint("s")) == first
    state["at"] = "2026-10-08T10:00:05.000Z"  # an undo, then another verdict: same counts
    assert asyncio.run(c.load_fingerprint("s")) != first


# ---------------------------------------------------------------------------
# 11
# ---------------------------------------------------------------------------


class ChunkedResponse:
    """A response that refuses a whole-body ``read()``: only chunks."""

    status = 200
    headers = {}

    def __init__(self, data):
        self.buf = io.BytesIO(data)
        self.reads = 0

    def read(self, n=-1):
        if n is None or n < 0:
            raise AssertionError("the bundle was read whole into memory")
        self.reads += 1
        return self.buf.read(n)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def zip_bytes(entries):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
    return buf.getvalue()


def test_f11_the_bundle_is_streamed(monkeypatch, tmp_path):
    payload = os.urandom(300_000)
    res = ChunkedResponse(zip_bytes({"results.json": "{}", "blobs/a.png": payload}))
    monkeypatch.setattr(tc.urllib.request, "urlopen", lambda req, timeout=None: res)
    written = tc.TrainyThingClient("http://127.0.0.1:1", "t").bundle("q1", str(tmp_path / "out"))
    assert sorted(os.path.relpath(p, tmp_path / "out") for p in written) == ["blobs/a.png", "results.json"]
    assert (tmp_path / "out" / "blobs" / "a.png").read_bytes() == payload
    assert res.reads >= 2
    assert sorted(os.listdir(tmp_path / "out")) == ["blobs", "results.json"]  # no temp file left


def test_f11_the_traversal_guard_stays(monkeypatch, tmp_path):
    res = ChunkedResponse(zip_bytes({"../escape.txt": "x"}))
    monkeypatch.setattr(tc.urllib.request, "urlopen", lambda req, timeout=None: res)
    with pytest.raises(ClientError, match="escapes the folder"):
        tc.TrainyThingClient("http://127.0.0.1:1", "t").bundle("q1", str(tmp_path / "out"))
    assert not (tmp_path / "escape.txt").exists()


# ---------------------------------------------------------------------------
# 12
# ---------------------------------------------------------------------------


def test_f12_run_ids_in_one_millisecond_differ():
    ids = {make_run_id("7", now=1000.0) for _ in range(50)}
    assert len(ids) > 40  # 4 hex: 65536 values
    for run in ids:
        assert re.fullmatch(r"19700101T001640000-n7-[0-9a-f]{4}", run), run
    assert cards_mod.card_ref(next(iter(ids)), 3).rsplit("-", 1)[1] == "3"
