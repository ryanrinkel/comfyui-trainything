"""The owner API client for the TrainyThing daemon (PLAN-COMFY §6.2, §4.7).

Standard library only. It mirrors the daemon's ``src/client.ts`` method for
method, so an error on one side reads like the other's:

    resolve_base_url  explicit -> TRAINYTHING_URL -> config.json base_url
                      -> http://127.0.0.1:<config.port or 4747>
    resolve_token     explicit -> TRAINYTHING_TOKEN -> config.json owner_token
    resolve_home      explicit -> TRAINYTHING_HOME -> ~/.trainything

``config.json`` counts only when it parses and carries both ``owner_token``
and ``cookie_secret`` as strings, exactly as ``readConfig`` in
``src/infra/home.ts`` reads it; otherwise it is as if it were absent.

The token is never a widget value, never a PNG chunk and never in a workflow:
it is read here, from the environment or the daemon's own config, and travels
only in the ``Authorization`` header.

Two faces: ``TrainyThingClient`` (blocking, urllib) and
``AsyncTrainyThingClient`` (each call on a worker thread via
``asyncio.to_thread``), because a ComfyUI V3 node's ``execute`` is async and
must never block the event loop.
"""

from __future__ import annotations

import asyncio
import io
import json
import os
import shutil
import subprocess
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from typing import Any, Callable, Dict, List, Optional, Union

DEFAULT_PORT = 4747
#: ``POST /api/blobs`` refuses anything larger (owner.ts ``MAX_BLOB_BYTES``).
MAX_BLOB_BYTES = 200 * 1024 * 1024
#: How long one ``/health`` probe may take (client.ts ``HEALTH_TIMEOUT_MS``).
HEALTH_TIMEOUT_S = 1.5
#: Budget for the one ``npx -y trainything list`` auto-start attempt (§4.7).
AUTO_START_TIMEOUT_S = 15.0
#: Ordinary request timeout. Uploads and bundles get longer.
REQUEST_TIMEOUT_S = 30.0
RESULT_FORMATS = ("json", "labels", "preferences", "summary")

__all__ = [
    "DEFAULT_PORT",
    "MAX_BLOB_BYTES",
    "ClientError",
    "NoDaemonError",
    "TrainyThingClient",
    "AsyncTrainyThingClient",
    "resolve_home",
    "read_config",
    "resolve_base_url",
    "resolve_token",
    "is_local_url",
    "FORCE_REMOTE_ENV",
    "no_daemon_message",
    "connect",
    "connect_async",
]


class ClientError(Exception):
    """An HTTP failure: ``status`` is the HTTP status (0 when unreachable),
    ``body`` the parsed JSON body, or the text, or None. The message is the
    server's ``{error}`` when it sent one, as ``messageFor`` in client.ts."""

    def __init__(self, message: str, status: int, body: Any = None) -> None:
        super().__init__(message)
        self.status = status
        self.body = body


class NoDaemonError(ClientError):
    """Nothing answered ``/health``; the message is the one line of §4.7."""

    def __init__(self, base_url: str) -> None:
        super().__init__(no_daemon_message(base_url), 0, None)
        self.base_url = base_url


def no_daemon_message(base_url: str) -> str:
    return f"no TrainyThing at {base_url}; run `npx -y trainything dashboard` once"


# ---------------------------------------------------------------------------
# Where the daemon is (src/infra/home.ts, src/client.ts)
# ---------------------------------------------------------------------------


def _nonempty(value: Optional[str]) -> bool:
    return value is not None and value != ""


def resolve_home(home: Optional[str] = None) -> str:
    """``home`` (the ``ClientOptions.home`` override), then ``TRAINYTHING_HOME``,
    then ``~/.trainything``. Resolved to an absolute path like ``path.resolve``."""
    if _nonempty(home):
        return os.path.abspath(os.path.expanduser(str(home)))
    override = os.environ.get("TRAINYTHING_HOME")
    if _nonempty(override):
        return os.path.abspath(str(override))
    return os.path.join(os.path.expanduser("~"), ".trainything")


def read_config(home: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """``config.json`` from the home, or None when absent or unusable.

    Same rule as ``parseConfig``: it must be a JSON object with string
    ``owner_token`` and ``cookie_secret``; ``port`` (a number) and
    ``base_url`` (a string) are kept when they have the right type."""
    path = os.path.join(resolve_home(home), "config.json")
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
    except OSError:
        return None
    try:
        parsed = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(parsed, dict):
        return None
    if not isinstance(parsed.get("owner_token"), str) or not isinstance(parsed.get("cookie_secret"), str):
        return None
    config: Dict[str, Any] = {"owner_token": parsed["owner_token"], "cookie_secret": parsed["cookie_secret"]}
    port = parsed.get("port")
    if isinstance(port, (int, float)) and not isinstance(port, bool):
        config["port"] = port
    if isinstance(parsed.get("base_url"), str):
        config["base_url"] = parsed["base_url"]
    return config


def _strip_slash(url: str) -> str:
    return url.rstrip("/")


def _port_text(port: Any) -> str:
    # A JSON number: 4747 and 4747.0 both print as 4747, as in JavaScript.
    if isinstance(port, float) and port.is_integer():
        return str(int(port))
    return str(port)


def resolve_base_url(base_url: Optional[str] = None, home: Optional[str] = None) -> str:
    """Explicit, then ``TRAINYTHING_URL``, then ``base_url`` in config.json,
    then ``http://127.0.0.1:<config.port or 4747>``; trailing slashes dropped."""
    if _nonempty(base_url):
        return _strip_slash(str(base_url))
    env = os.environ.get("TRAINYTHING_URL")
    if _nonempty(env):
        return _strip_slash(str(env))
    config = read_config(home)
    if config is not None and _nonempty(config.get("base_url")):
        return _strip_slash(config["base_url"])
    port = config.get("port") if config is not None else None
    return f"http://127.0.0.1:{_port_text(port if port is not None else DEFAULT_PORT)}"


def resolve_token(token: Optional[str] = None, home: Optional[str] = None) -> Optional[str]:
    """Explicit, then ``TRAINYTHING_TOKEN``, then ``owner_token`` in config.json.

    None when there is none yet: ``connect`` in client.ts would create the
    config at that point, which is the daemon's job, not a client's in another
    language. ``connect`` here re-reads it after the auto-start attempt."""
    if _nonempty(token):
        return str(token)
    env = os.environ.get("TRAINYTHING_TOKEN")
    if _nonempty(env):
        return str(env)
    config = read_config(home)
    if config is not None:
        return config["owner_token"]
    return None


#: Test-only: when set to ``1``, ``is_local_url`` says no for every URL, so a
#: daemon on 127.0.0.1 is treated as remote (no auto-start; the run's folder
#: and Load read ``results/bundle.zip`` instead of asking the daemon to write
#: files). The K2 gate uses it for "remote in all but name".
FORCE_REMOTE_ENV = "TRAINYTHING_FORCE_REMOTE"


def is_local_url(url: str) -> bool:
    """Auto-start only makes sense for a server we could spawn (``isLocalUrl``).
    ``TRAINYTHING_FORCE_REMOTE=1`` (test-only) makes every URL remote."""
    if os.environ.get(FORCE_REMOTE_ENV) == "1":
        return False
    try:
        parsed = urllib.parse.urlsplit(url)
        host = (parsed.hostname or "").lower()
    except ValueError:
        return False
    return host in ("127.0.0.1", "localhost", "::1", "[::1]")


# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------


def _message_for(status: int, body: Any, fallback: str) -> str:
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, str) and err != "":
            return err
    if isinstance(body, str) and body.strip() != "":
        return body.strip()
    return f"{fallback} (HTTP {status})"


def _quote(segment: str) -> str:
    # encodeURIComponent
    return urllib.parse.quote(str(segment), safe="-_.!~*'()")


class _Response:
    def __init__(self, status: int, headers: Any, body: bytes) -> None:
        self.status = status
        self.headers = headers
        self.body = body

    def json(self) -> Any:
        return json.loads(self.body.decode("utf-8"))

    def text(self) -> str:
        return self.body.decode("utf-8")


class TrainyThingClient:
    """The owner API, blocking. Every method raises ``ClientError`` on a
    non-2xx answer or an unreachable server."""

    def __init__(self, base_url: str, token: Optional[str], timeout: float = REQUEST_TIMEOUT_S) -> None:
        self.base_url = _strip_slash(base_url)
        self._token = token or ""
        self.timeout = timeout

    def __repr__(self) -> str:  # never print the token
        return f"TrainyThingClient({self.base_url!r})"

    # -- plumbing -----------------------------------------------------------

    def _url(self, pathname: str, query: Optional[Dict[str, Optional[str]]] = None) -> str:
        url = f"{self.base_url}{pathname}"
        pairs = [(k, v) for k, v in (query or {}).items() if v is not None]
        if pairs:
            url += "?" + urllib.parse.urlencode(pairs)
        return url

    def _send(
        self,
        method: str,
        pathname: str,
        *,
        query: Optional[Dict[str, Optional[str]]] = None,
        json_body: Any = None,
        body: Optional[bytes] = None,
        content_type: Optional[str] = None,
        timeout: Optional[float] = None,
        auth: bool = True,
    ) -> _Response:
        headers: Dict[str, str] = {}
        if auth:
            headers["Authorization"] = f"Bearer {self._token}"
        data: Optional[bytes] = None
        if json_body is not None:
            headers["content-type"] = "application/json"
            data = json.dumps(json_body).encode("utf-8")
        elif body is not None:
            if content_type is not None:
                headers["content-type"] = content_type
            data = body
        req = urllib.request.Request(self._url(pathname, query), data=data, method=method, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout or self.timeout) as res:
                return _Response(res.status, res.headers, res.read())
        except urllib.error.HTTPError as err:
            raw = b""
            try:
                raw = err.read()
            except Exception:  # pragma: no cover - a broken socket mid-error
                pass
            text = raw.decode("utf-8", errors="replace")
            parsed: Any = text
            try:
                parsed = json.loads(text)
            except ValueError:
                pass
            raise ClientError(_message_for(err.code, parsed, f"{method} {pathname} failed"), err.code, parsed) from None
        except (urllib.error.URLError, OSError) as err:
            reason = getattr(err, "reason", err)
            raise ClientError(f"cannot reach {self.base_url}: {reason}", 0, None) from None

    def _json(self, method: str, pathname: str, **kw: Any) -> Any:
        return self._send(method, pathname, **kw).json()

    # -- the API ------------------------------------------------------------

    def health(self, timeout: float = HEALTH_TIMEOUT_S) -> Dict[str, Any]:
        """``GET /health`` -> ``{ok, version, tunnel}``. No auth."""
        res = self._send("GET", "/health", timeout=timeout, auth=False)
        return res.json()

    def probe(self, timeout: float = HEALTH_TIMEOUT_S) -> bool:
        try:
            return self.health(timeout=timeout).get("ok") is True
        except (ClientError, ValueError):
            return False

    def create_queue(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """``POST /api/queues`` -> ``{queue_id, local_url, share_url, queue}``."""
        return self._json("POST", "/api/queues", json_body=config)

    def list_queues(self, project: Optional[str] = None) -> List[Dict[str, Any]]:
        """``GET /api/queues`` -> ``[{queue_id, slug, title, project, status, progress, ...}]``."""
        return self._json("GET", "/api/queues", query={"project": project})

    def get_queue(self, id_or_slug: str) -> Dict[str, Any]:
        """``GET /api/queues/:id`` (id or slug) -> ``{queue, local_url, share_url}``."""
        return self._json("GET", f"/api/queues/{_quote(id_or_slug)}")

    def find_queue(self, id_or_slug: str) -> Optional[Dict[str, Any]]:
        """``get_queue``, with a 404 as None."""
        try:
            return self.get_queue(id_or_slug)
        except ClientError as err:
            if err.status == 404:
                return None
            raise

    def status(self, id_or_slug: str) -> Dict[str, Any]:
        """``GET /api/queues/:id/status`` -> ``{queue_id, status, cards, judged,
        judgments, reviewers, coverage, min_per_card, done, reason, contested,
        share_url, local_url}``."""
        return self._json("GET", f"/api/queues/{_quote(id_or_slug)}/status")

    def add_cards(self, id_or_slug: str, cards: List[Dict[str, Any]]) -> Dict[str, Any]:
        """``POST /api/queues/:id/cards`` with ``{cards}`` -> ``{added, total, warnings}``."""
        return self._json("POST", f"/api/queues/{_quote(id_or_slug)}/cards", json_body={"cards": cards})

    def upload_blob(self, data: bytes, mime: str = "image/png") -> Dict[str, Any]:
        """``POST /api/blobs``, raw body -> ``{sha256, bytes, mime}``. The cap is
        checked here first so a too-large image never leaves the machine."""
        if len(data) == 0:
            raise ClientError("empty blob upload", 400, None)
        if len(data) > MAX_BLOB_BYTES:
            raise ClientError(f"blob too large: {len(data)} bytes exceeds {MAX_BLOB_BYTES}", 413, None)
        return self._json(
            "POST",
            "/api/blobs",
            body=bytes(data),
            content_type=mime or "application/octet-stream",
            timeout=max(self.timeout, 120.0),
        )

    def share(self, id_or_slug: str, slug: Optional[str] = None) -> Dict[str, Any]:
        """``POST /api/queues/:id/share`` -> ``{share_url, changed}`` (starts the tunnel)."""
        return self._json(
            "POST",
            f"/api/queues/{_quote(id_or_slug)}/share",
            json_body={} if slug is None else {"slug": slug},
            timeout=max(self.timeout, 60.0),
        )

    def close(self, id_or_slug: str) -> Dict[str, Any]:
        """``POST /api/queues/:id/close`` -> ``{closed_at, queue}``."""
        return self._json("POST", f"/api/queues/{_quote(id_or_slug)}/close", json_body={})

    def reopen(self, id_or_slug: str) -> Dict[str, Any]:
        """``POST /api/queues/:id/reopen`` -> ``{queue}``."""
        return self._json("POST", f"/api/queues/{_quote(id_or_slug)}/reopen", json_body={})

    def results(
        self,
        id_or_slug: str,
        format: str = "json",
        out: Optional[str] = None,
        include_contested: bool = False,
    ) -> Union[str, Dict[str, Any]]:
        """``GET /api/queues/:id/results``.

        Without ``out``: the rendered document as text (``json``, ``labels``,
        ``preferences`` or ``summary``). With ``out`` (a file path the *daemon*
        writes, so only for a local daemon): ``{path, bytes, format}``; for
        ``json`` the daemon also copies ``blobs/`` next to it."""
        if format not in RESULT_FORMATS:
            raise ValueError(f"unknown format: {format} (want one of {', '.join(RESULT_FORMATS)})")
        query: Dict[str, Optional[str]] = {
            "format": format,
            "include_contested": "1" if include_contested else None,
        }
        if out is not None:
            query["out"] = os.path.abspath(out)
            query["cwd"] = os.getcwd()
        res = self._send("GET", f"/api/queues/{_quote(id_or_slug)}/results", query=query, timeout=max(self.timeout, 120.0))
        if out is not None:
            return res.json()
        return res.text()

    def bundle(self, id_or_slug: str, dest_dir: str) -> List[str]:
        """``GET /api/queues/:id/results/bundle.zip``, unpacked into ``dest_dir``:
        ``results.json`` plus ``blobs/<sha><ext>``. Works against a remote
        daemon (§10 question 5). Returns the paths written."""
        res = self._send("GET", f"/api/queues/{_quote(id_or_slug)}/results/bundle.zip", timeout=max(self.timeout, 600.0))
        os.makedirs(dest_dir, exist_ok=True)
        root = os.path.realpath(dest_dir)
        written: List[str] = []
        with zipfile.ZipFile(io.BytesIO(res.body)) as zf:
            for info in zf.infolist():
                if info.is_dir():
                    continue
                target = os.path.realpath(os.path.join(root, info.filename))
                if os.path.commonpath([root, target]) != root:
                    raise ClientError(f"bundle entry escapes the folder: {info.filename}", 0, None)
                os.makedirs(os.path.dirname(target), exist_ok=True)
                with zf.open(info) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                written.append(target)
        return written


class AsyncTrainyThingClient:
    """The same API, awaitable: each call runs the blocking one on a worker
    thread, so ComfyUI's event loop keeps serving while a request is out."""

    def __init__(self, sync: TrainyThingClient) -> None:
        self.sync = sync

    @property
    def base_url(self) -> str:
        return self.sync.base_url

    def __repr__(self) -> str:
        return f"AsyncTrainyThingClient({self.sync.base_url!r})"

    def __getattr__(self, name: str) -> Callable[..., Any]:
        attr = getattr(self.sync, name)
        if not callable(attr) or name.startswith("_"):
            return attr

        async def call(*args: Any, **kwargs: Any) -> Any:
            return await asyncio.to_thread(attr, *args, **kwargs)

        call.__name__ = name
        return call


# ---------------------------------------------------------------------------
# connect: health-check, one auto-start attempt, the one line
# ---------------------------------------------------------------------------


def _run_npx_list(base_url: str, home: Optional[str], budget: float) -> None:
    """``npx -y trainything list`` once; the CLI auto-starts the daemon for a
    local URL (design §8.2). Failure of any kind is fine: the probe decides."""
    npx = shutil.which("npx")
    if npx is None:
        return
    env = dict(os.environ)
    env["TRAINYTHING_URL"] = base_url
    if _nonempty(home):
        env["TRAINYTHING_HOME"] = resolve_home(home)
    try:
        subprocess.run(
            [npx, "-y", "trainything", "list"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=budget,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        pass


def connect(
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    home: Optional[str] = None,
    auto_start: bool = True,
    timeout: float = AUTO_START_TIMEOUT_S,
    _starter: Optional[Callable[[str, Optional[str], float], None]] = None,
) -> TrainyThingClient:
    """A ready client, or ``NoDaemonError`` with the one line of §4.7.

    When ``/health`` does not answer and the URL is local (and ``auto_start``),
    run ``npx -y trainything list`` once within ``timeout`` seconds, then
    re-probe until the same budget runs out. A non-local URL is never started
    from here."""
    url = resolve_base_url(base_url, home)
    client = TrainyThingClient(url, resolve_token(token, home))
    if client.probe():
        return client
    if not auto_start or not is_local_url(url):
        raise NoDaemonError(url)

    deadline = time.monotonic() + timeout
    (_starter or _run_npx_list)(url, home, timeout)
    while True:
        if client.probe(timeout=max(0.25, min(HEALTH_TIMEOUT_S, deadline - time.monotonic()))):
            # The daemon may have just written config.json with a fresh token.
            return TrainyThingClient(url, resolve_token(token, home))
        if time.monotonic() >= deadline:
            break
        time.sleep(0.25)
    raise NoDaemonError(url)


async def connect_async(
    base_url: Optional[str] = None,
    token: Optional[str] = None,
    home: Optional[str] = None,
    auto_start: bool = True,
    timeout: float = AUTO_START_TIMEOUT_S,
) -> AsyncTrainyThingClient:
    """``connect`` on a worker thread; the event loop is never blocked."""
    sync = await asyncio.to_thread(connect, base_url, token, home, auto_start, timeout)
    return AsyncTrainyThingClient(sync)
