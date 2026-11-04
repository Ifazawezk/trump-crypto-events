"""HTTP client with an honest user agent, polite pacing and a file cache.

Every download goes through here, so the identity string and the pacing are
the same for all sources. It never sends browser headers and gives up on
403 instead of trying to get around it.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from pathlib import Path

import requests

USER_AGENT = "trump-crypto-events/0.1 (open-source research script; python-requests)"


class BlockedError(RuntimeError):
    """The server refused access. We stop rather than work around it."""


class Client:
    def __init__(self, cache_dir: str | Path, min_interval: float = 0.25,
                 session: requests.Session | None = None, timeout: float = 60.0,
                 retries: int = 4):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.min_interval = min_interval
        self.session = session or requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.timeout = timeout
        self.retries = retries
        self._last = 0.0
        self._lock = threading.Lock()  # pacing is shared across threads
        self.network_calls = 0

    def cache_path(self, url: str, params: dict | None, key: str | None) -> Path:
        if key:
            return self.cache_dir / key
        raw = url + "?" + "&".join(f"{k}={v}" for k, v in sorted((params or {}).items()))
        return self.cache_dir / hashlib.sha1(raw.encode()).hexdigest()

    def get(self, url: str, params: dict | None = None, key: str | None = None,
            expect: str | None = None) -> bytes | None:
        """Return the body, or None for 404. Results (and 404s) are cached.

        `expect` ("json", "zip" or "text") guards against access pages that
        come back with status 200: those raise BlockedError and are not cached.
        """
        path = self.cache_path(url, params, key)
        missing = path.with_name(path.name + ".404")
        if path.exists():
            return path.read_bytes()
        if missing.exists():
            return None
        for attempt in range(self.retries + 1):
            with self._lock:
                wait = self.min_interval - (time.monotonic() - self._last)
                if wait > 0:
                    time.sleep(wait)
                self._last = time.monotonic()
                self.network_calls += 1
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)
            except (requests.Timeout, requests.ConnectionError):
                time.sleep(2.0 * 2 ** attempt)
                continue
            if resp.status_code == 404:
                missing.parent.mkdir(parents=True, exist_ok=True)
                missing.touch()
                return None
            if resp.status_code in (401, 403):
                raise BlockedError(f"{url} returned {resp.status_code}; not retrying")
            if resp.status_code == 429 or resp.status_code >= 500:
                retry_after = resp.headers.get("Retry-After", "")
                delay = float(retry_after) if retry_after.isdigit() else 2.0 * 2 ** attempt
                time.sleep(min(delay, 60))
                continue
            resp.raise_for_status()
            check_body(url, resp.content, expect)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(resp.content)
            return resp.content
        raise RuntimeError(f"{url} still failing after {self.retries} retries")


def check_body(url: str, body: bytes, expect: str | None) -> None:
    ok = True
    if expect == "json":
        try:
            json.loads(body)
        except ValueError:
            ok = False
    elif expect == "zip":
        ok = body[:2] == b"PK"
    elif expect == "text":
        ok = not body.lstrip()[:1] == b"<"
    if not ok:
        raise BlockedError(f"{url} did not return {expect}; probably an access page, stopping")
