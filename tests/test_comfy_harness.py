"""The ComfyUI side (§6.1), against the CPU checkout tests/harness/setup_comfy.sh
installs at ~/.cache/comfyui-trainything. Runs only in the harness venv (torch
importable); elsewhere it skips. Every claim in docs/V3-NOTES.md that can be
checked by running code is checked here."""

import json
import os
import random
import signal
import subprocess
import sys
import textwrap
import time
import urllib.request
from pathlib import Path

import pytest

from conftest import PACK, _port_free

HARNESS = Path(os.environ.get("COMFY_HARNESS", Path.home() / ".cache" / "comfyui-trainything"))
COMFY = HARNESS / "ComfyUI"
PINNED = "0.39.0"

pytestmark = pytest.mark.skipif(
    not (COMFY / "main.py").exists() or __import__("importlib.util").util.find_spec("torch") is None,
    reason="needs the harness ComfyUI and torch (run tests/run.sh)",
)


def run_in_comfy(code: str) -> str:
    """Run ``code`` in a fresh interpreter rooted at the checkout (the code
    sets its own argv: comfy.cli_args parses sys.argv on import)."""
    out = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)],
        cwd=str(COMFY),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout


def test_checkout_version_is_the_pin():
    ns = {}
    exec((COMFY / "comfyui_version.py").read_text(), ns)
    assert ns["__version__"] == PINNED
    assert f'requires-comfyui = ">={PINNED}"' in (PACK / "pyproject.toml").read_text()


def test_v3_api_facts():
    out = run_in_comfy(
        f"""
        import sys, json, asyncio, importlib.util
        sys.argv = ["main.py", "--cpu"]
        sys.path.insert(0, ".")
        import comfy.options
        comfy.options.enable_args_parsing()  # as main.py does, so --cpu applies
        from comfy_api.latest import ComfyAPI, ComfyAPI_latest, ComfyExtension, io, ui
        from comfy_api.version_list import supported_versions
        from comfy_execution.graph import ExecutionBlocker
        from comfy_execution.graph_utils import ExecutionBlocker as EB2
        import comfy.utils

        facts = {{}}
        Queue = io.Custom("TRAINYTHING_QUEUE")
        facts["custom_io_type"] = Queue.io_type
        facts["custom_has_input_output"] = hasattr(Queue, "Input") and hasattr(Queue, "Output")
        facts["list_output"] = io.Int.Output(is_output_list=True).is_output_list
        facts["preview_text"] = ui.PreviewText("hello").as_dict() == {{"text": ("hello",)}}
        facts["blocker_same_class"] = ExecutionBlocker is EB2
        facts["blocker_silent"] = ExecutionBlocker(None).message is None
        facts["latest_version"] = ComfyAPI_latest.VERSION
        facts["versions"] = [v.VERSION for v in supported_versions]
        facts["set_progress"] = asyncio.iscoroutinefunction(ComfyAPI_latest.Execution.set_progress)
        facts["progress_bar"] = hasattr(comfy.utils, "ProgressBar")
        facts["node_output_block"] = io.NodeOutput(block_execution="x").block_execution == "x"

        spec = importlib.util.spec_from_file_location(
            "comfyui_trainything_probe", {str(PACK / "__init__.py")!r},
            submodule_search_locations=[{str(PACK)!r}])
        mod = importlib.util.module_from_spec(spec); sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        ext = asyncio.run(mod.comfy_entrypoint())
        facts["extension"] = isinstance(ext, ComfyExtension)
        facts["node_list"] = asyncio.run(ext.get_node_list())
        print(json.dumps(facts))
        """
    )
    facts = json.loads(out.strip().splitlines()[-1])
    assert facts == {
        "custom_io_type": "TRAINYTHING_QUEUE",
        "custom_has_input_output": True,
        "list_output": True,
        "preview_text": True,
        "blocker_same_class": True,
        "blocker_silent": True,
        "latest_version": "latest",
        "versions": ["latest", "0.0.2", "0.0.1"],
        "set_progress": True,
        "progress_bar": True,
        "node_output_block": True,
        "extension": True,
        "node_list": [],
    }


def test_comfyui_starts_on_cpu_and_loads_the_pack(tmp_path):
    for _ in range(100):
        port = random.randint(18000, 19999)
        if _port_free(port):
            break
    log = tmp_path / "comfy.log"
    with open(log, "w") as fh:
        proc = subprocess.Popen(
            [sys.executable, "main.py", "--cpu", "--port", str(port), "--listen", "127.0.0.1",
             "--disable-auto-launch", "--output-directory", str(tmp_path / "out")],
            cwd=str(COMFY),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    try:
        stats = None
        deadline = time.monotonic() + 180
        while time.monotonic() < deadline and proc.poll() is None:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{port}/system_stats", timeout=2) as res:
                    stats = json.loads(res.read())
                    break
            except OSError:
                time.sleep(1)
        text = log.read_text()
        assert stats is not None, text[-3000:]
        assert stats["system"]["comfyui_version"] == PINNED
        assert any(d["type"] == "cpu" for d in stats["devices"])
        load_lines = [ln for ln in text.splitlines() if "custom_nodes/comfyui-trainything" in ln]
        assert load_lines and "IMPORT FAILED" not in " ".join(load_lines), text[-3000:]
        assert "comfy_entrypoint in" not in text  # no entrypoint warning
    finally:
        os.killpg(proc.pid, signal.SIGTERM)
        try:
            proc.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
