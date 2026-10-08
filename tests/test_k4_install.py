"""Gate K4: the stranger install (PLAN-COMFY §6.5, §12 "installs by `git
clone` into `custom_nodes/`"), done the way the README says.

A second ComfyUI checkout (``git clone --shared`` of the harness one, so no
download; the harness venv is its python), the pack ``git clone``d from this
repository (the local path stands in for the GitHub URL, so the clone is the
committed HEAD, not the working tree) into its ``custom_nodes/``,
``pip install -r requirements.txt`` with ``--no-index`` (already satisfied, so
nothing is fetched), then ComfyUI on CPU with a throwaway daemon's
environment. ``/object_info`` must list the five nodes with their display
names and category, and the log must show the pack imported from the clone
with no IMPORT FAILED and no version warning. Both processes are killed on
teardown. The version guard of ``__init__.py`` is checked without ComfyUI."""

import re
import subprocess
import sys
import types

import pytest

from conftest import PACK, PKG, start_daemon

NODES = {
    "TrainyThingReview": "TrainyThing Review",
    "TrainyThingSend": "TrainyThing Send",
    "TrainyThingWait": "TrainyThing Wait",
    "TrainyThingLoad": "TrainyThing Load",
    "TrainyThingFilter": "TrainyThing Filter",
}


# --- the version guard (any venv) -------------------------------------------


def test_pin_matches_pyproject():
    pack = sys.modules[PKG]
    text = (PACK / "pyproject.toml").read_text()
    assert re.search(r'^requires-comfyui = ">=([^"]+)"', text, re.M).group(1) == pack.REQUIRES_COMFYUI
    assert re.search(r'^version = "([^"]+)"', text, re.M).group(1) == pack.__version__


@pytest.mark.parametrize("running, warns", [
    ("0.38.9", True), ("0.3.60", True), ("0.39.0", False), ("0.39.1", False), ("0.40.0", False),
    ("1.0.0", False), ("", False), (None, False), ("unknown", False),
])
def test_too_old_line(running, warns):
    line = sys.modules[PKG].comfyui_too_old(running)
    assert (line is not None) == warns
    if warns:
        assert line.startswith("[TrainyThing] ") and running in line and ">= 0.39.0" in line and "\n" not in line


def test_guard_prints_once_and_never_raises(monkeypatch, capsys):
    pack = sys.modules[PKG]
    monkeypatch.setitem(sys.modules, "comfyui_version", types.SimpleNamespace(__version__="0.30.0"))
    pack._check_comfyui_version()
    out = capsys.readouterr().out
    assert out.count("[TrainyThing]") == 1 and "0.30.0" in out
    monkeypatch.setitem(sys.modules, "comfyui_version", types.SimpleNamespace(__version__="0.39.0"))
    pack._check_comfyui_version()
    monkeypatch.setitem(sys.modules, "comfyui_version", object())  # no __version__
    pack._check_comfyui_version()
    assert capsys.readouterr().out == ""


# --- the install (harness venv) ---------------------------------------------


def git(*args, cwd=None):
    res = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=300)
    assert res.returncode == 0, res.stderr
    return res.stdout


@pytest.mark.comfy
def test_git_clone_into_custom_nodes(tmp_path):
    pytest.importorskip("torch", reason="needs the harness venv (tests/run.sh)")
    from comfyrun import COMFY, Comfy

    if not (COMFY / "main.py").exists():
        pytest.skip("no harness ComfyUI (tests/harness/setup_comfy.sh)")
    comfy_dir = tmp_path / "ComfyUI"
    git("clone", "-q", "--shared", str(COMFY), str(comfy_dir))
    assert not (comfy_dir / "custom_nodes" / "comfyui-trainything").exists()

    # The README's Install, step by step.
    pack_dir = comfy_dir / "custom_nodes" / "comfyui-trainything"
    git("clone", "-q", str(PACK), "comfyui-trainything", cwd=comfy_dir / "custom_nodes")
    assert not pack_dir.is_symlink() and (pack_dir / "__init__.py").exists()
    pip = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "-r", "requirements.txt"],
                         cwd=pack_dir, capture_output=True, text=True, timeout=300)
    assert pip.returncode == 0, pip.stdout + pip.stderr
    imports = subprocess.run([sys.executable, "-c", "import segno, PIL, numpy"], capture_output=True, text=True)
    assert imports.returncode == 0, imports.stderr

    daemon = start_daemon(tmp_path)
    comfy = None
    try:
        comfy = Comfy(tmp_path / "run", daemon, comfy_dir=comfy_dir)
        info = comfy.get("/object_info", timeout=60)
        for node_id, display in NODES.items():
            assert node_id in info, f"{node_id} missing from /object_info"
            assert info[node_id]["display_name"] == display
            assert info[node_id]["category"] == "TrainyThing"
            assert "comfyui-trainything" in info[node_id]["python_module"]
        log = comfy.log_text()
        assert "IMPORT FAILED" not in log, log[-4000:]
        assert str(pack_dir) in log, "the log's import times should list the cloned pack"
        assert "[TrainyThing] this ComfyUI is" not in log
    finally:
        if comfy is not None:
            comfy.stop()
        daemon.stop()
