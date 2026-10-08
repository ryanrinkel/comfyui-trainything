"""A scripted reviewer over the reviewer API (src/server/routes/reviewer.ts):
join, then next + judge until told to stop. Each Reviewer has its own cookie
jar, so two of them are two people."""

from __future__ import annotations

import http.cookiejar
import json
import urllib.request
from typing import Any, Callable, Dict, Optional


class Reviewer:
    def __init__(self, base_url: str, slug: str, name: str) -> None:
        self.base = f"{base_url}/api/r/{slug}"
        self.name = name
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(http.cookiejar.CookieJar()))

    def _call(self, method: str, path: str, body: Any = None) -> Any:
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(self.base + path, data=data, method=method)
        if data is not None:
            req.add_header("content-type", "application/json")
        with self.opener.open(req, timeout=10) as res:
            return json.loads(res.read())

    def join(self) -> Dict[str, Any]:
        return self._call("POST", "/join", {"name": self.name, "email": f"{self.name.lower()}@example.com"})

    def next(self) -> Dict[str, Any]:
        return self._call("GET", "/next")

    def judge(self, card_id: str, verdict: Any, note: Optional[str] = None) -> Dict[str, Any]:
        body: Dict[str, Any] = {"card_id": card_id, "verdict": verdict}
        if note is not None:
            body["note"] = note
        return self._call("POST", "/judge", body)

    def review(self, decide: Callable[[str], Any], notes: Optional[Callable[[str, Any], Optional[str]]] = None) -> int:
        """Judge served cards until ``decide(ref)`` returns None (stop) or
        the queue says done. Returns how many were judged."""
        self.join()
        judged = 0
        while True:
            nxt = self.next()
            if nxt.get("done"):
                return judged
            card = nxt["card"]
            verdict = decide(card["ref"])
            if verdict is None:
                return judged
            self.judge(card["id"], verdict, notes(card["ref"], verdict) if notes else None)
            judged += 1
