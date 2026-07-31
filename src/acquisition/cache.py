"""FastF1 cache management and a persistent API budget ledger.

Two facts drive this module, both measured by ``scripts/probe_api_cost.py``:

1. **Cache hits never reach the rate limiter.**
   ``fastf1.req._CachedSessionWithRateLimiting`` is ``(CacheMixin,
   _SessionWithRateLimiting)``; on a hit ``CacheMixin.send`` returns without
   calling ``super().send()``. A repeat load therefore costs zero requests.
   Preserving ``cache/fastf1`` is the single highest-leverage rule in the
   project: deleting it is what made three previous collection runs
   unrecoverable.

2. **FastF1's own budget counter does not survive a restart.**
   ``fastf1.req`` holds its 500 request timestamps in an in-memory ``deque``,
   so restarting Python resets the client-side guard while the server-side
   limit persists. That is precisely how a silent 429 storm begins. This module
   therefore persists its own timestamps to ``logs/api_budget.jsonl`` and
   throttles against them across process boundaries.
"""
from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

from src import config

logger = logging.getLogger(__name__)

BUDGET_WINDOW_S = 3600


class RequestCounter:
    """Counts outbound (non-cached) FastF1 requests.

    Patches ``_SessionWithRateLimiting.send``, which a request only reaches
    when it genuinely goes to the network, so the count is exact rather than
    inferred from cache-hit log lines.
    """

    _original = None

    def __init__(self) -> None:
        self.count = 0

    def install(self) -> None:
        """Patch the FastF1 session class to increment this counter."""
        from fastf1 import req

        if RequestCounter._original is None:
            RequestCounter._original = req._SessionWithRateLimiting.send
        original = RequestCounter._original
        counter = self

        def counting_send(session_self, request, **kwargs):
            counter.count += 1
            return original(session_self, request, **kwargs)

        req._SessionWithRateLimiting.send = counting_send

    def reset(self) -> int:
        """Return the count since the last reset, then zero it."""
        n = self.count
        self.count = 0
        return n


class BudgetLedger:
    """Cross-process record of outbound API requests.

    :param path: JSONL file holding one record per collection unit.
    :param limit: hard requests-per-hour limit imposed by the API.
    :param soft_limit: level at which to start sleeping, leaving headroom.
    """

    def __init__(self, path: Path, limit: int = 500, soft_limit: int = 450) -> None:
        self.path = path
        self.limit = limit
        self.soft_limit = soft_limit
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, n_requests: int, label: str) -> None:
        """Append a request-count record.

        :param n_requests: outbound requests issued for this unit of work.
        :param label: identifier, e.g. ``"2023_R04"``.
        """
        if n_requests <= 0:
            return
        entry = {"ts": time.time(), "n": int(n_requests), "label": label,
                 "iso": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        with open(self.path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    def spent_in_window(self) -> int:
        """Return requests issued within the trailing budget window."""
        if not self.path.exists():
            return 0
        cutoff = time.time() - BUDGET_WINDOW_S
        total = 0
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue                      # tolerate a partial final line
                if entry.get("ts", 0) >= cutoff:
                    total += int(entry.get("n", 0))
        return total

    def seconds_until_headroom(self, needed: int) -> float:
        """Seconds to wait until ``needed`` requests fit under the soft limit.

        :param needed: requests the next unit of work is expected to issue.
        :returns: seconds to sleep; 0.0 if there is already headroom.
        """
        if not self.path.exists():
            return 0.0
        now = time.time()
        cutoff = now - BUDGET_WINDOW_S
        entries = []
        with open(self.path, "r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if entry.get("ts", 0) >= cutoff:
                    entries.append((float(entry["ts"]), int(entry["n"])))

        spent = sum(n for _, n in entries)
        if spent + needed <= self.soft_limit:
            return 0.0

        # Wait until enough of the oldest records age out of the window.
        must_free = spent + needed - self.soft_limit
        freed = 0
        for ts, n in sorted(entries):
            freed += n
            if freed >= must_free:
                return max(0.0, (ts + BUDGET_WINDOW_S) - now)
        return float(BUDGET_WINDOW_S)

    def wait_for_headroom(self, needed: int, label: str = "") -> None:
        """Block until ``needed`` requests fit under the soft limit."""
        delay = self.seconds_until_headroom(needed)
        if delay <= 0:
            return
        logger.warning(
            "Budget: %d/%d spent this hour. Sleeping %.0f s before %s.",
            self.spent_in_window(), self.limit, delay, label or "next unit")
        time.sleep(delay + 1)


def init_cache(*, allow_cold: bool = False) -> Path:
    """Enable the FastF1 cache, refusing to run cold unless told otherwise.

    A cold cache means every session must be re-fetched against the 500/hour
    limit. Since the previous cache was deleted between runs and cost three
    collection attempts, running cold is made deliberate rather than accidental.

    :param allow_cold: proceed even when the cache is empty or absent.
    :returns: the cache directory.
    :raises RuntimeError: if the cache is cold and ``allow_cold`` is False.
    """
    import fastf1

    cache_dir = config.resolve_path("fastf1_cache")
    existed = cache_dir.exists() and any(cache_dir.iterdir())

    if not existed and not allow_cold:
        raise RuntimeError(
            f"FastF1 cache at {cache_dir} is empty or absent. Every session "
            "would be re-fetched against the 500 calls/hour limit. Pass "
            "--allow-cold-cache to proceed deliberately.")

    cache_dir.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    logger.info("Cache enabled at %s (%s)", cache_dir,
                "warm" if existed else "COLD - all sessions will be fetched")
    return cache_dir
