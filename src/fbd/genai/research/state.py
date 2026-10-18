"""In-memory state: quotas, a search cache, and short-lived conversation memory.

All of it lives in this process and nothing is written to disk. That suits the
deployment (one FastAPI process, read-only, D-021) and the privacy rule: the
server keeps only what it needs to answer a follow-up, for thirty minutes, and
forgets it on restart.

* Quotas: a per-client sliding minute and day, and a global day, so a public
  demo cannot run up a bill or exhaust a free tier for everyone else.
* Search cache: keyed by the normalised query, the place and time scope, the
  provider and a config version. Safety and current questions are cached for
  five minutes at most, and every answer shows its retrieval time, so a cached
  advisory is never presented as fresher than it is.
* Memory: the last question per conversation id, so "Assam" can answer
  "Which place?" without the user retyping the question.
"""
from __future__ import annotations

import threading
import time
from collections import deque

_lock = threading.Lock()


class Quota:
    def __init__(self):
        self.minute: dict[str, deque] = {}
        self.day: dict[str, deque] = {}
        self.global_day: deque = deque()

    def check(self, client: str, per_min: int, per_day: int, global_day: int) -> tuple[bool, int, str]:
        """(allowed, retry_after_seconds, which limit)."""
        now = time.time()
        with _lock:
            m = self.minute.setdefault(client, deque())
            d = self.day.setdefault(client, deque())
            for q, span in ((m, 60), (d, 86400), (self.global_day, 86400)):
                while q and now - q[0] > span:
                    q.popleft()
            if len(self.global_day) >= global_day:
                return False, int(86400 - (now - self.global_day[0])) + 1, "global daily budget"
            if len(m) >= per_min:
                return False, int(60 - (now - m[0])) + 1, "per-minute limit"
            if len(d) >= per_day:
                return False, int(86400 - (now - d[0])) + 1, "daily limit"
            m.append(now); d.append(now); self.global_day.append(now)
            return True, 0, ""

    def reset(self) -> None:
        with _lock:
            self.minute.clear(); self.day.clear(); self.global_day.clear()


class TTLCache:
    def __init__(self, max_items: int = 256):
        self.items: dict[str, tuple[float, object]] = {}
        self.max = max_items
        self.hits = self.misses = 0

    def get(self, key: str):
        with _lock:
            hit = self.items.get(key)
            if hit and hit[0] > time.time():
                self.hits += 1
                return hit[1]
            self.items.pop(key, None)
            self.misses += 1
            return None

    def put(self, key: str, value, ttl: float) -> None:
        with _lock:
            if len(self.items) >= self.max:
                self.items.pop(min(self.items, key=lambda k: self.items[k][0]), None)
            self.items[key] = (time.time() + ttl, value)

    def clear(self) -> None:
        with _lock:
            self.items.clear(); self.hits = self.misses = 0


class Memory:
    TTL = 1800

    def __init__(self):
        self.items: dict[str, tuple[float, dict]] = {}

    def get(self, cid: str | None) -> dict | None:
        if not cid:
            return None
        with _lock:
            hit = self.items.get(cid)
            if hit and time.time() - hit[0] < self.TTL:
                return hit[1]
            self.items.pop(cid, None)
            return None

    def put(self, cid: str | None, value: dict) -> None:
        if not cid:
            return
        with _lock:
            if len(self.items) > 1000:
                for k in sorted(self.items, key=lambda k: self.items[k][0])[:200]:
                    self.items.pop(k, None)
            self.items[cid] = (time.time(), value)

    def forget(self, cid: str | None) -> None:
        with _lock:
            self.items.pop(cid or "", None)


QUOTA = Quota()
SEARCH_CACHE = TTLCache()
MEMORY = Memory()
