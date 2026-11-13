"""Polite, resumable HTTP helpers shared by the DataPull scripts.

Rules: identify ourselves, sleep between requests, retry transient errors with
backoff, never retry a 403 (a block is an answer, not a puzzle), and write files
atomically so an interrupted pull never leaves a half-written file behind.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
USER_AGENT = "TennisResearch/0.1 (personal non-commercial research; Tim Slanders)"


class BlockedError(RuntimeError):
    """Raised on HTTP 401/403: the source refuses automated access. Do not route around it."""


class PoliteSession:
    def __init__(self, min_interval_s: float = 1.0, max_retries: int = 4, timeout_s: float = 60.0):
        self.session = requests.Session()
        self.session.headers["User-Agent"] = USER_AGENT
        self.min_interval_s = min_interval_s
        self.max_retries = max_retries
        self.timeout_s = timeout_s
        self._last_request = 0.0
        self.throttled = 0

    def Get(self, url: str, params: dict | None = None) -> requests.Response:
        attempt = 0
        while True:
            wait_s = self.min_interval_s - (time.monotonic() - self._last_request)
            if wait_s > 0:
                time.sleep(wait_s)
            self._last_request = time.monotonic()
            try:
                response = self.session.get(url, params=params, timeout=self.timeout_s)
            except requests.RequestException:
                response = None
            if response is not None:
                if response.status_code in (401, 403):
                    raise BlockedError(f"HTTP {response.status_code} for {url}")
                if response.status_code == 404:
                    return response
                if response.status_code < 400:
                    return response
            attempt += 1
            if attempt > self.max_retries:
                if response is None:
                    raise RuntimeError(f"network error for {url}")
                response.raise_for_status()
            backoff_s = min(60.0, 2.0 ** attempt)
            if response is not None and response.status_code == 429:
                self.throttled += 1
                backoff_s = max(1.0 * attempt, float(response.headers.get("Retry-After", 0) or 0))
            time.sleep(backoff_s)


def WriteBytesAtomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".part")
    with open(temp_path, "wb") as handle:
        handle.write(payload)
    os.replace(temp_path, path)


def WriteJsonAtomic(path: Path, obj) -> None:
    WriteBytesAtomic(path, json.dumps(obj, indent=1, sort_keys=True).encode("utf-8"))


def AppendJsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")