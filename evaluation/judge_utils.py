"""Shared utilities for HLE and DeepSearch evaluation scripts."""
import heapq
import os
import queue as _queue
import random
import threading
import time
from collections import deque

from openai import OpenAI

thread_local = threading.local()
API_KEY = os.getenv("API_KEY", "")
BASE_URL = os.getenv("API_BASE", "")


class _PriorityQueue:
    """
    Min-heap keyed on (available_at, retry_count, seq).
    Items ready sooner and with fewer prior retries are served first,
    so no item starves while high-retry items wait out long back-offs.
    Workers never sleep — they always pick the next ready item.
    """

    def __init__(self):
        self._heap = []
        self._cond = threading.Condition(threading.Lock())
        self._seq = 0

    def put(self, item, retry_count: int = 0, delay: float = 0.0):
        available_at = time.monotonic() + delay
        with self._cond:
            heapq.heappush(self._heap, (available_at, retry_count, self._seq, item))
            self._seq += 1
            self._cond.notify_all()

    def get(self, timeout: float = 1.0):
        """Return (retry_count, item). Raises queue.Empty on timeout."""
        deadline = time.monotonic() + timeout
        with self._cond:
            while True:
                now = time.monotonic()
                if self._heap:
                    available_at, retry_count, _, item = self._heap[0]
                    if available_at <= now:
                        heapq.heappop(self._heap)
                        return retry_count, item
                    remaining = min(available_at - now, deadline - now)
                else:
                    remaining = deadline - now
                if remaining <= 0:
                    raise _queue.Empty
                self._cond.wait(remaining)

    def qsize(self):
        with self._cond:
            return len(self._heap)


class SlidingWindowRateLimiter:
    """Thread-safe sliding window rate limiter.
    Blocks the caller until a request slot is available within the window."""

    def __init__(self, max_calls: int, window_seconds: float = 60.0):
        self._max_calls = max_calls
        self._window = window_seconds
        self._timestamps = deque()
        self._lock = threading.Lock()

    def acquire(self):
        """Block until a call slot is available."""
        while True:
            with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self._window:
                    self._timestamps.popleft()
                if len(self._timestamps) < self._max_calls:
                    self._timestamps.append(now)
                    return
                wait_time = self._window - (now - self._timestamps[0])
            time.sleep(max(0.0, wait_time))


def get_client():
    if not hasattr(thread_local, "client"):
        thread_local.client = OpenAI(api_key=API_KEY, base_url=BASE_URL)
    return thread_local.client


def _is_rate_limit_error(e: Exception) -> bool:
    err_str = str(e).lower()
    return any(
        k in err_str for k in ("rate limit", "ratelimit", "429", "too many requests")
    )


def _is_terminal_error(e: Exception) -> bool:
    err_str = str(e).lower()
    return any(
        k in err_str
        for k in ("length limit", "401", "unauthorized", "400", "bad request", "invalid_api_key")
    )


def _rate_limit_delay(retry_count: int, jitter: float = 0.5) -> float:
    """前 5 次固定约 2s，后续从 4s 起指数增长，上限 60s，均叠加 ±jitter 抖动。"""
    base = 2.0 if retry_count < 5 else min(60.0, 2.0 ** (retry_count - 3))
    return base * random.uniform(1 - jitter, 1 + jitter)
