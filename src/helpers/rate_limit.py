"""
In-memory sliding-window rate limiter.

The backend runs as ONE process (embedded Qdrant holds a file lock, so uvicorn must use --workers 1),
therefore an in-process limiter is sufficient and needs no extra infrastructure. If the backend is ever
scaled to several processes, replace this with a shared store (e.g. Redis) behind the same `hit()` API.
"""
import threading
import time
from collections import defaultdict, deque


class RateLimiter:

    def __init__(self):
        self._hits = defaultdict(deque)
        self._lock = threading.Lock()

    def hit(self, key: str, limit: int, window_seconds: int):
        """Register one request for `key`. Returns (allowed: bool, retry_after_seconds: int).
        limit <= 0 disables the limit."""
        if not limit or limit <= 0:
            return True, 0
        now = time.monotonic()
        with self._lock:
            hits = self._hits[key]
            while hits and hits[0] <= now - window_seconds:
                hits.popleft()
            if len(hits) >= limit:
                retry_after = int(window_seconds - (now - hits[0])) + 1
                return False, max(1, retry_after)
            hits.append(now)
            if len(self._hits) > 10000:  # bound memory: forget keys with no recent hits
                for stale in [k for k, v in self._hits.items() if not v or v[-1] <= now - 86400]:
                    del self._hits[stale]
            return True, 0

    def reset(self):
        with self._lock:
            self._hits.clear()


rate_limiter = RateLimiter()
