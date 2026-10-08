"""URL and token precedence match src/client.ts (resolveBaseUrl, connect)
and src/infra/home.ts (resolveHome, readConfig) in all four cases:
explicit, env, config.json, default."""

import json

from comfyui_trainything.trainything_client import (
    is_local_url,
    read_config,
    resolve_base_url,
    resolve_home,
    resolve_token,
)


def write_config(home, **extra):
    home.mkdir(parents=True, exist_ok=True)
    body = {"owner_token": "cfg-token", "cookie_secret": "s"}
    body.update(extra)
    (home / "config.json").write_text(json.dumps(body))
    return home


def test_explicit_beats_env_and_config(monkeypatch, tmp_path):
    home = write_config(tmp_path / "h", base_url="http://cfg.example:1")
    monkeypatch.setenv("TRAINYTHING_HOME", str(home))
    monkeypatch.setenv("TRAINYTHING_URL", "http://env.example:2/")
    monkeypatch.setenv("TRAINYTHING_TOKEN", "env-token")
    assert resolve_base_url("http://explicit.example:3//") == "http://explicit.example:3"
    assert resolve_token("explicit-token") == "explicit-token"


def test_env_beats_config(monkeypatch, tmp_path):
    home = write_config(tmp_path / "h", base_url="http://cfg.example:1")
    monkeypatch.setenv("TRAINYTHING_HOME", str(home))
    monkeypatch.setenv("TRAINYTHING_URL", "http://env.example:2/")
    monkeypatch.setenv("TRAINYTHING_TOKEN", "env-token")
    assert resolve_base_url() == "http://env.example:2"
    assert resolve_token() == "env-token"
    # An empty string is "not set", as `env !== ''` in client.ts.
    assert resolve_base_url("") == "http://env.example:2"
    assert resolve_token("") == "env-token"


def test_config_beats_default(monkeypatch, tmp_path):
    home = write_config(tmp_path / "h", base_url="http://cfg.example:1/")
    monkeypatch.setenv("TRAINYTHING_HOME", str(home))
    monkeypatch.setenv("TRAINYTHING_URL", "")
    monkeypatch.setenv("TRAINYTHING_TOKEN", "")
    assert resolve_base_url() == "http://cfg.example:1"
    assert resolve_token() == "cfg-token"


def test_config_port_without_base_url(monkeypatch, tmp_path):
    home = write_config(tmp_path / "h", port=5151)
    monkeypatch.setenv("TRAINYTHING_HOME", str(home))
    assert resolve_base_url() == "http://127.0.0.1:5151"


def test_default(monkeypatch, tmp_path):
    # The autouse fixture points TRAINYTHING_HOME at an empty folder.
    assert resolve_base_url() == "http://127.0.0.1:4747"
    assert resolve_token() is None


def test_home_override_order(monkeypatch, tmp_path):
    a = write_config(tmp_path / "a", base_url="http://a.example:1")
    b = write_config(tmp_path / "b", base_url="http://b.example:2")
    monkeypatch.setenv("TRAINYTHING_HOME", str(a))
    assert resolve_base_url(home=str(b)) == "http://b.example:2"
    assert resolve_base_url() == "http://a.example:1"
    # No TRAINYTHING_HOME: ~/.trainything (HOME redirected to a temp dir).
    monkeypatch.delenv("TRAINYTHING_HOME")
    monkeypatch.setenv("HOME", str(tmp_path / "fakehome"))
    write_config(tmp_path / "fakehome" / ".trainything", base_url="http://home.example:3")
    assert resolve_home() == str(tmp_path / "fakehome" / ".trainything")
    assert resolve_base_url() == "http://home.example:3"
    assert resolve_token() == "cfg-token"


def test_unusable_config_is_absent(monkeypatch, tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    monkeypatch.setenv("TRAINYTHING_HOME", str(home))
    # No cookie_secret: readConfig returns undefined, base_url ignored.
    (home / "config.json").write_text(json.dumps({"owner_token": "t", "base_url": "http://x.example:9"}))
    assert read_config() is None
    assert resolve_base_url() == "http://127.0.0.1:4747"
    assert resolve_token() is None
    (home / "config.json").write_text("{not json")
    assert read_config() is None
    # Wrong types are dropped field by field.
    (home / "config.json").write_text(json.dumps({"owner_token": "t", "cookie_secret": "s", "port": "80", "base_url": 7}))
    assert read_config() == {"owner_token": "t", "cookie_secret": "s"}


def test_is_local_url():
    for url in ("http://127.0.0.1:4747", "http://localhost:1", "http://[::1]:4747", "https://LOCALHOST"):
        assert is_local_url(url), url
    for url in ("http://192.168.1.2:4747", "https://x.trycloudflare.com", "not a url", "http://127.0.0.2:1"):
        assert not is_local_url(url), url
