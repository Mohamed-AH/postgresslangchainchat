"""In-memory rate limiting and a global daily budget.

These protect the shared provider keys (Phase 1 runs on our Cohere/Gemini keys) from a
single session hammering the service or the instance's daily cost running away. State is
process-local, which is the right trade-off for a single free-tier instance; a multi-
instance deployment would back these with Redis (the interface would not change).
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from ragchat.config import Settings


class RateLimiter:
    """Fixed-window rate limiter keyed by an arbitrary string (e.g. a session id)."""

    def __init__(self, limit: int, window_seconds: float) -> None:
        self._limit = limit
        self._window = window_seconds
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, int]] = {}  # key -> (window_start, count)

    def allow(self, key: str, *, now: float | None = None) -> bool:
        """Return True if ``key`` may proceed, counting this call against its window."""
        now = time.monotonic() if now is None else now
        with self._lock:
            start, count = self._state.get(key, (now, 0))
            if now - start >= self._window:
                start, count = now, 0
            count += 1
            self._state[key] = (start, count)
            return count <= self._limit


class DailyBudget:
    """A global cap on operations per UTC day (0 == unlimited)."""

    def __init__(self, budget: int) -> None:
        self._budget = budget
        self._lock = threading.Lock()
        self._day: int = -1
        self._count = 0

    def allow(self, *, now: float | None = None) -> bool:
        """Return True if the instance is under budget for the current day."""
        if self._budget <= 0:
            return True
        now = time.time() if now is None else now
        day = int(now // 86400)
        with self._lock:
            if day != self._day:
                self._day, self._count = day, 0
            self._count += 1
            return self._count <= self._budget


@dataclass
class Guards:
    """Bundle of the guardrails applied to cost-incurring endpoints."""

    ask_limiter: RateLimiter
    ingest_limiter: RateLimiter
    daily_budget: DailyBudget

    @classmethod
    def from_settings(cls, settings: Settings) -> Guards:
        return cls(
            ask_limiter=RateLimiter(settings.rate_limit_asks_per_minute, 60.0),
            ingest_limiter=RateLimiter(settings.rate_limit_ingests_per_hour, 3600.0),
            daily_budget=DailyBudget(settings.daily_request_budget),
        )
