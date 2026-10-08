"""A missing daemon fails with the one line of §4.7."""

import asyncio
import time

import pytest

from comfyui_trainything.trainything_client import (
    NoDaemonError,
    connect,
    connect_async,
    no_daemon_message,
)

from conftest import free_port


def test_the_one_line():
    assert (
        no_daemon_message("http://127.0.0.1:4747")
        == "no TrainyThing at http://127.0.0.1:4747; run `npx -y trainything dashboard` once"
    )


def test_local_with_auto_start_disabled():
    url = f"http://127.0.0.1:{free_port()}"
    with pytest.raises(NoDaemonError) as err:
        connect(base_url=url, auto_start=False)
    assert str(err.value) == f"no TrainyThing at {url}; run `npx -y trainything dashboard` once"
    assert err.value.status == 0


def test_non_local_never_runs_npx():
    calls = []
    url = "http://trainything.invalid:4747"
    with pytest.raises(NoDaemonError) as err:
        connect(base_url=url, auto_start=True, _starter=lambda *a: calls.append(a))
    assert str(err.value) == f"no TrainyThing at {url}; run `npx -y trainything dashboard` once"
    assert calls == []


def test_local_tries_once_then_the_line():
    calls = []
    url = f"http://127.0.0.1:{free_port()}"
    t0 = time.monotonic()
    with pytest.raises(NoDaemonError) as err:
        connect(base_url=url, timeout=1.0, _starter=lambda *a: calls.append(a))
    assert len(calls) == 1 and calls[0][0] == url and calls[0][2] == 1.0
    assert time.monotonic() - t0 < 5
    assert str(err.value) == no_daemon_message(url)


def test_env_url_is_used(monkeypatch):
    url = f"http://127.0.0.1:{free_port()}"
    monkeypatch.setenv("TRAINYTHING_URL", url + "/")
    with pytest.raises(NoDaemonError, match=f"no TrainyThing at {url};"):
        connect(auto_start=False)


def test_async_face_fails_the_same_way():
    url = f"http://127.0.0.1:{free_port()}"
    with pytest.raises(NoDaemonError, match="run `npx -y trainything dashboard` once"):
        asyncio.run(connect_async(base_url=url, auto_start=False))
