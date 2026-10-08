"""Shared fixtures. Every daemon here is a throwaway: TRAINYTHING_HOME in a
temp dir, a random port in 20000-40000, killed after. The owner's daemon on
4747 and ~/.trainything are never touched: TRAINYTHING_URL / _TOKEN are
cleared and TRAINYTHING_HOME is pointed at an empty temp dir for every test.
"""

from __future__ import annotations

import importlib.util
import json
import os
import random
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import pytest

PACK = Path(__file__).resolve().parent.parent
FIXTURES = Path(__file__).resolve().parent / "fixtures"
SPRITES = sorted((FIXTURES / "sprites").glob("s*.png"))
PKG = "comfyui_trainything"


def _load_pack():
    """Import the repo as a package (its folder name has a dash), the way
    ComfyUI does: spec_from_file_location on __init__.py."""
    if PKG in sys.modules:
        return sys.modules[PKG]
    spec = importlib.util.spec_from_file_location(
        PKG, PACK / "__init__.py", submodule_search_locations=[str(PACK)]
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules[PKG] = mod
    spec.loader.exec_module(mod)
    return mod


_load_pack()


@pytest.fixture(autouse=True)
def _isolated_env(monkeypatch, tmp_path):
    monkeypatch.delenv("TRAINYTHING_URL", raising=False)
    monkeypatch.delenv("TRAINYTHING_TOKEN", raising=False)
    empty = tmp_path / "no-home"
    empty.mkdir()
    monkeypatch.setenv("TRAINYTHING_HOME", str(empty))
    yield


def trainything_repo() -> Path:
    env = os.environ.get("TRAINYTHING_REPO")
    return Path(env).resolve() if env else (PACK.parent / "trainything").resolve()


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
            return True
        except OSError:
            return False


def free_port(lo: int = 20000, hi: int = 40000) -> int:
    for _ in range(200):
        port = random.randint(lo, hi)
        if port != 4747 and _port_free(port):
            return port
    raise RuntimeError("no free port")


class Daemon:
    def __init__(self, home: Path, port: int) -> None:
        self.home = home
        self.port = port
        self.url = f"http://127.0.0.1:{port}"
        self.token = json.loads((home / "config.json").read_text())["owner_token"]

    def client(self):
        from comfyui_trainything.trainything_client import TrainyThingClient

        return TrainyThingClient(self.url, self.token)

    def stop(self) -> None:
        pidfile = self.home / "server.pid"
        if pidfile.exists():
            try:
                pid = int(pidfile.read_text().strip())
                os.kill(pid, signal.SIGTERM)
                for _ in range(50):
                    os.kill(pid, 0)
                    time.sleep(0.1)
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, ValueError):
                pass


def start_daemon(root: Path) -> Daemon:
    repo = trainything_repo()
    cli = repo / "dist" / "cli.js"
    if not cli.exists():
        pytest.skip(f"no daemon build at {cli} (run `pnpm build` in {repo})")
    if shutil.which("node") is None:
        pytest.skip("node is not on PATH")
    home = root / "home"
    port = free_port()
    env = dict(os.environ)
    env["TRAINYTHING_HOME"] = str(home)
    env.pop("TRAINYTHING_URL", None)
    env.pop("TRAINYTHING_TOKEN", None)
    subprocess.run(
        ["node", str(cli), "serve", "--daemon", "--port", str(port)],
        env=env,
        cwd=str(root),
        check=True,
        stdout=subprocess.DEVNULL,
        timeout=30,
    )
    d = Daemon(home, port)
    with urllib.request.urlopen(f"{d.url}/health", timeout=5) as res:
        assert json.loads(res.read())["ok"] is True
    return d


@pytest.fixture(scope="session")
def daemon(tmp_path_factory):
    d = start_daemon(tmp_path_factory.mktemp("daemon"))
    try:
        yield d
    finally:
        d.stop()


@pytest.fixture(scope="session")
def sprite_batch():
    """The twelve sprites as one [12, 64, 64, 3] float32 batch in 0..1. The
    fixtures are of mixed sizes and a ComfyUI batch has one, so each is
    resized (nearest) to 64x64."""
    np = pytest.importorskip("numpy")
    from PIL import Image

    return np.stack(
        [np.asarray(Image.open(p).convert("RGB").resize((64, 64), Image.NEAREST)).astype(np.float32) / 255.0 for p in SPRITES]
    )


@pytest.fixture(scope="session")
def graph():
    return json.loads((FIXTURES / "graph-sdxl-lora.json").read_text())


@pytest.fixture(scope="session")
def workflow():
    return json.loads((FIXTURES / "workflow-sdxl-lora.json").read_text())
