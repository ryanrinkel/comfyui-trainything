"""The review fixes against throwaway daemons (no ComfyUI).

Finding -> test:
   2 a queue the CLI made (no min_reviews, cards already in it) gets the
     widget's min_reviews by PATCH, so the wait can end; another mode is an
     error before any upload ......................................... test_f2_*
   4 a slug renamed on the dashboard mid-wait: the folder is still written test_f4_*
   7 share on an empty queue works (a fake cloudflared on the daemon's
     PATH); a failing cloudflared is the local link, the batch added once test_f7_*
  10 an undo then another verdict moves Load's fingerprint ............. test_f10_*
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess

import pytest

np = pytest.importorskip("numpy")

from conftest import start_daemon  # noqa: E402
from comfyui_trainything.cards import QueueModeError, make_run_id, send_batch  # noqa: E402
from comfyui_trainything.nodes import common as c  # noqa: E402
from reviewer import Reviewer  # noqa: E402


def test_f2_a_cli_queue_with_cards_gets_min_reviews_and_finishes(daemon, sprite_batch, capsys):
    client = daemon.client()
    made = client.create_queue({"title": "From the CLI", "slug": "rf-cli-queue"})  # no completion, as the CLI
    assert made["queue"]["completion"].get("min_reviews_per_card") is None
    client.add_cards("rf-cli-queue", [{"ref": "cli-0", "kind": "text", "text": "a card the CLI added"}])
    h = send_batch(client, sprite_batch[:2], c.queue_config(title="t", slug="rf-cli-queue", min_reviews=1), make_run_id("9"))
    assert h["created"] is False and h["total"] == 3
    # PATCH completion on a queue that already has cards: the daemon allows it.
    assert client.get_queue("rf-cli-queue")["queue"]["completion"] == {"min_reviews_per_card": 1}
    assert "had no min_reviews; set it to 1" in capsys.readouterr().out
    assert client.status("rf-cli-queue")["done"] is False
    Reviewer(daemon.url, "rf-cli-queue", "Ada").review(lambda ref: "accept")
    st = client.status("rf-cli-queue")
    assert st["done"] is True and st["reason"] == "min_reviews"


def test_f2_another_mode_fails_before_any_upload(daemon, sprite_batch):
    client = daemon.client()
    send_batch(client, sprite_batch[:1], c.queue_config(title="t", slug="rf-single"), make_run_id("1"))
    cfg = c.queue_config(title="t", slug="rf-single", mode="choose", options="a:A, b:B")
    with pytest.raises(QueueModeError, match="rf-single is a single queue and this node's mode is choose"):
        send_batch(client, sprite_batch[:2], cfg, make_run_id("1"))
    assert client.status("rf-single")["cards"] == 1


def test_f4_a_renamed_slug_still_writes_the_folder(daemon, sprite_batch, tmp_path):
    client = daemon.client()
    h = send_batch(client, sprite_batch[:2], {"title": "t", "slug": "rf-before"}, make_run_id("4"))
    client.patch_queue(h["queue_id"], {"slug": "rf-after"})  # the dashboard renames the link mid-wait
    assert client.find_queue("rf-before") is None
    out = c.write_run_folder_sync(client, h["queue_id"], h["slug"], "single", str(tmp_path))
    assert out["folder"] == str(tmp_path / "rf-before") and out["how"] == "local"
    doc = json.loads((tmp_path / "rf-before" / "results.json").read_text())
    assert [card["ref"] for card in doc["cards"]] == h["refs"]
    assert (tmp_path / "rf-before" / "labels.jsonl").read_text().count("\n") == 2


FAKE_CLOUDFLARED = """#!/bin/sh
# A stand-in for cloudflared: prints a quick-tunnel URL and stays up, or fails
# at once while the flag file exists.
if [ -f "{flag}" ]; then echo "failed to connect" >&2; exit 1; fi
echo "INF |  https://rf-fake-tunnel.trycloudflare.com  |" >&2
while :; do sleep 1; done
"""


@pytest.fixture(scope="module")
def tunnel_daemon(tmp_path_factory):
    root = tmp_path_factory.mktemp("rf-tunnel")
    bindir = root / "bin"
    bindir.mkdir()
    flag = root / "fail"
    script = bindir / "cloudflared"
    script.write_text(FAKE_CLOUDFLARED.format(flag=flag))
    script.chmod(0o755)
    d = start_daemon(root, {"PATH": f"{bindir}{os.pathsep}{os.environ.get('PATH', '')}"})
    try:
        yield d, flag
    finally:
        d.stop()
        subprocess.run(["pkill", "-f", str(script)], check=False)


def test_f7_a_failing_tunnel_is_the_local_link_and_one_batch(tunnel_daemon, sprite_batch, capsys):
    d, flag = tunnel_daemon
    client = d.client()
    # First in the module: no tunnel is running yet, so share starts one.
    flag.write_text("1")
    try:
        h = send_batch(client, sprite_batch[:3], {"title": "t", "slug": "rf-public"}, make_run_id("7"), public=True)
    finally:
        flag.unlink()
    st = client.status("rf-public")
    assert h["share_url"] is None
    assert h["link"] == f"{d.url}/r/rf-public" and st["cards"] == 3 and h["added"] == 3
    assert "public: could not share rf-public" in capsys.readouterr().out


def test_f7_share_on_an_empty_queue(tunnel_daemon):
    d, flag = tunnel_daemon
    client = d.client()
    made = client.create_queue({"title": "Empty", "slug": "rf-empty"})
    assert client.status(made["queue"]["id"])["cards"] == 0
    shared = client.share(made["queue"]["id"])
    assert shared["share_url"] == "https://rf-fake-tunnel.trycloudflare.com/r/rf-empty"


def test_f10_undo_then_another_verdict_moves_the_fingerprint(daemon, monkeypatch):
    monkeypatch.setenv("TRAINYTHING_URL", daemon.url)
    monkeypatch.setenv("TRAINYTHING_TOKEN", daemon.token)
    client = daemon.client()
    client.create_queue({"title": "fp", "slug": "rf-undo", "completion": {"min_reviews_per_card": 1}})
    client.add_cards("rf-undo", [{"ref": "t-0", "kind": "text", "text": "one"}, {"ref": "t-1", "kind": "text", "text": "two"}])
    rev = Reviewer(daemon.url, "rf-undo", "Ada")
    rev.join()
    card = rev.next()["card"]
    rev.judge(card["id"], "accept")
    fp = lambda: asyncio.run(c.load_fingerprint("rf-undo"))  # noqa: E731
    before, judgments = fp(), client.status("rf-undo")["judgments"]
    assert fp() == before
    rev._call("POST", "/undo", {"card_id": card["id"]})
    rev.judge(card["id"], "reject")
    assert client.status("rf-undo")["judgments"] == judgments  # the count alone would not move
    assert fp() != before
