"""Review finding 6 in a real ComfyUI: started with ``--disable-metadata``,
Send uploads PNGs with no ``prompt`` / ``workflow`` chunk (as Save Image
writes none), while the graph walk still fills the card's ``meta``."""

from __future__ import annotations

import json

import pytest

from conftest import start_daemon

pytest.importorskip("torch", reason="needs the harness venv (tests/run.sh)")
from comfyrun import COMFY, Comfy, load_prompt, messages  # noqa: E402

from comfyui_trainything.png import png_text  # noqa: E402

pytestmark = [
    pytest.mark.comfy,
    pytest.mark.skipif(not (COMFY / "main.py").exists(), reason="no harness ComfyUI (tests/harness/setup_comfy.sh)"),
]


def test_f6_disable_metadata_keeps_the_chunks_out(tmp_path, workflow):
    daemon = start_daemon(tmp_path)
    comfy = None
    try:
        comfy = Comfy(tmp_path / "comfy", daemon, extra_args=["--disable-metadata"])
        p = load_prompt("send")
        p["20"]["inputs"]["slug"] = "rf-no-metadata"
        h = comfy.wait_history(comfy.submit(p, workflow=workflow))
        assert h["status"]["status_str"] == "success", h["status"]
        assert "execution_error" not in messages(h)

        doc = json.loads(daemon.client().results("rf-no-metadata", "json"))
        assert len(doc["cards"]) == 12
        for card in doc["cards"]:
            with open(card["items"][0]["src"], "rb") as fh:
                chunks = png_text(fh.read())
            assert "prompt" not in chunks and "workflow" not in chunks, sorted(chunks)
            # The walk still ran: meta comes from the graph, not from the PNG.
            assert card["meta"]["source"] == "k1-test" and "walk" in card["meta"] and "run" in card["meta"]
    finally:
        if comfy is not None:
            comfy.stop()
        daemon.stop()
