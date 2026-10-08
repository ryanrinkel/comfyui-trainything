"""A throwaway ComfyUI for the node tests: the harness checkout on CPU, on a
random port in 18000-19999, its input / output / temp folders in a temp dir,
and the throwaway daemon in its environment (TRAINYTHING_URL, TRAINYTHING_TOKEN
and TRAINYTHING_HOME), so it never sees ~/.trainything or port 4747."""

from __future__ import annotations

import json
import os
import shutil
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Dict, Optional

from conftest import FIXTURES, SPRITES, free_port

HARNESS = Path(os.environ.get("COMFY_HARNESS", Path.home() / ".cache" / "comfyui-trainything"))
COMFY = HARNESS / "ComfyUI"
PROMPTS = FIXTURES / "prompts"


def load_prompt(name: str) -> Dict[str, Any]:
    return json.loads((PROMPTS / f"{name}.json").read_text())


class Comfy:
    def __init__(self, root: Path, daemon: Any, python: str = sys.executable) -> None:
        self.root = root
        self.input = root / "input"
        self.output = root / "output"
        self.temp = root / "temp"
        self.user = root / "user"
        for d in (self.input, self.output, self.temp, self.user):
            d.mkdir(parents=True, exist_ok=True)
        for p in SPRITES:
            shutil.copy(p, self.input / p.name)
        self.port = free_port(18000, 19999)
        self.url = f"http://127.0.0.1:{self.port}"
        self.log = root / "comfy.log"
        self.client_id = uuid.uuid4().hex
        env = dict(os.environ)
        env["TRAINYTHING_URL"] = daemon.url
        env["TRAINYTHING_TOKEN"] = daemon.token
        env["TRAINYTHING_HOME"] = str(daemon.home)
        env["PYTHONUNBUFFERED"] = "1"
        self._fh = open(self.log, "w")
        self.proc = subprocess.Popen(
            [python, "main.py", "--cpu", "--port", str(self.port), "--listen", "127.0.0.1", "--disable-auto-launch",
             "--input-directory", str(self.input), "--output-directory", str(self.output),
             "--temp-directory", str(self.temp), "--user-directory", str(self.user)],
            cwd=str(COMFY), env=env, stdout=self._fh, stderr=subprocess.STDOUT, start_new_session=True,
        )
        deadline = time.monotonic() + 240
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError("ComfyUI exited:\n" + self.log_text()[-4000:])
            try:
                self.get("/system_stats", timeout=2)
                return
            except OSError:
                time.sleep(0.5)
        self.stop()
        raise RuntimeError("ComfyUI did not answer /system_stats:\n" + self.log_text()[-4000:])

    def log_text(self) -> str:
        try:
            return self.log.read_text(errors="replace")
        except OSError:
            return ""

    def get(self, path: str, timeout: float = 10) -> Any:
        with urllib.request.urlopen(self.url + path, timeout=timeout) as res:
            return json.loads(res.read())

    def post(self, path: str, body: Any, timeout: float = 30) -> Any:
        req = urllib.request.Request(self.url + path, data=json.dumps(body).encode(), method="POST",
                                     headers={"content-type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as res:
                raw = res.read()
                return json.loads(raw) if raw.strip() else None
        except urllib.error.HTTPError as err:
            raise AssertionError(f"POST {path} -> {err.code}: {err.read().decode(errors='replace')[:3000]}") from None

    def submit(self, prompt: Dict[str, Any], workflow: Optional[Dict[str, Any]] = None) -> str:
        body: Dict[str, Any] = {"prompt": prompt, "client_id": self.client_id}
        if workflow is not None:
            body["extra_data"] = {"extra_pnginfo": {"workflow": workflow}}
        res = self.post("/prompt", body)
        assert not res.get("node_errors"), res
        return res["prompt_id"]

    def history(self, prompt_id: str) -> Optional[Dict[str, Any]]:
        return self.get(f"/history/{prompt_id}").get(prompt_id)

    def wait_history(self, prompt_id: str, timeout: float = 120) -> Dict[str, Any]:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            h = self.history(prompt_id)
            if h is not None and h.get("status", {}).get("completed") is not None:
                if h["status"].get("status_str") in ("success", "error"):
                    return h
            time.sleep(0.5)
        raise AssertionError(f"prompt {prompt_id} did not finish in {timeout}s:\n" + self.log_text()[-4000:])

    def interrupt(self) -> None:
        self.post("/interrupt", {})

    def stop(self) -> None:
        if self.proc.poll() is None:
            try:
                os.killpg(self.proc.pid, signal.SIGTERM)
                self.proc.wait(timeout=20)
            except subprocess.TimeoutExpired:
                os.killpg(self.proc.pid, signal.SIGKILL)
                self.proc.wait()
            except ProcessLookupError:
                pass
        self._fh.close()


def ui_texts(history: Dict[str, Any], node: str) -> list:
    """The ``text`` a Preview as Text (or our node) put in /history."""
    return list(history.get("outputs", {}).get(node, {}).get("text", []))


def ui_images(history: Dict[str, Any], node: str) -> list:
    return list(history.get("outputs", {}).get(node, {}).get("images", []))


def messages(history: Dict[str, Any]) -> list:
    return [m[0] for m in history.get("status", {}).get("messages", [])]
