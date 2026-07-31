"""Phase 0.5: measure the real FastF1 API cost of one race session.

Three collection runs were lost to the 500 calls/hour limit, so no bulk
collection may start until the per-session request cost is measured rather
than estimated.

Counting method: ``fastf1.req._CachedSessionWithRateLimiting`` is declared as
``(CacheMixin, _SessionWithRateLimiting)``. On a cache hit ``CacheMixin.send``
returns without calling ``super().send()``, so a request only reaches
``_SessionWithRateLimiting.send`` when it genuinely goes to the network.
Patching that method therefore counts outbound requests exactly.

Run:
    .venv\\Scripts\\python.exe scripts/probe_api_cost.py
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = PROJECT_ROOT / "cache" / "fastf1"

# Probe target: a mid-season 2023 race, avoiding round 1 (atypical schedules)
# and sprint weekends.
PROBE_YEAR = 2023
PROBE_ROUND = 4

logger = logging.getLogger("probe")


class RequestCounter:
    """Counts outbound (non-cached) FastF1 requests by patching the session."""

    def __init__(self) -> None:
        self.count = 0
        self.urls: list[str] = []
        self._original = None

    def install(self) -> None:
        """Patch ``_SessionWithRateLimiting.send`` to increment the counter."""
        from fastf1 import req

        counter = self

        if self._original is None:
            self._original = req._SessionWithRateLimiting.send

        original = self._original

        def counting_send(session_self, request, **kwargs):
            counter.count += 1
            counter.urls.append(request.url)
            return original(session_self, request, **kwargs)

        req._SessionWithRateLimiting.send = counting_send

    def reset(self) -> int:
        """Return the count since the last reset, then zero it."""
        n = self.count
        self.count = 0
        self.urls.clear()
        return n


def load_race(year: int, round_number: int, *, telemetry: bool) -> object:
    """Load one race session, returning the loaded Session object.

    :param year: season year.
    :param round_number: FIA round number (stable key, not event name).
    :param telemetry: whether to fetch car_data and position_data.
    :returns: the loaded fastf1 Session.
    """
    import fastf1

    session = fastf1.get_session(year, round_number, "R")
    session.load(laps=True, telemetry=telemetry, weather=True, messages=False)
    return session


def main() -> int:
    """Run the three-stage probe and report the branch decision."""
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)-7s %(message)s", stream=sys.stdout)
    # FastF1 logs one line per resource at INFO, which would drown the report.
    logging.getLogger("fastf1").setLevel(logging.WARNING)

    import fastf1

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(CACHE_DIR))
    logger.info("Cache enabled at %s", CACHE_DIR)

    counter = RequestCounter()
    counter.install()

    # ---- Stage 1: cold load, laps + weather only ----
    logger.info("Stage 1: cold load of %s round %s (telemetry=False)",
                PROBE_YEAR, PROBE_ROUND)
    session = load_race(PROBE_YEAR, PROBE_ROUND, telemetry=False)
    cold_requests = counter.reset()
    laps = session.laps

    logger.info("  requests        : %d", cold_requests)
    logger.info("  event           : %s", session.event["EventName"])
    logger.info("  laps shape      : %s", laps.shape)
    logger.info("  compounds       : %s", laps["Compound"].value_counts().to_dict())
    logger.info("  weather rows    : %d", len(session.weather_data))
    has_status = "TrackStatus" in laps.columns
    logger.info("  TrackStatus col : %s", has_status)
    if has_status:
        logger.info("  green laps      : %d / %d",
                    int((laps["TrackStatus"].astype(str) == "1").sum()), len(laps))

    # ---- Stage 2: identical load, must be free ----
    logger.info("Stage 2: repeat identical load, expecting zero requests")
    del session
    session = load_race(PROBE_YEAR, PROBE_ROUND, telemetry=False)
    warm_requests = counter.reset()
    logger.info("  requests        : %d", warm_requests)

    # ---- Stage 3: telemetry increment ----
    logger.info("Stage 3: same session with telemetry=True")
    del session
    session = load_race(PROBE_YEAR, PROBE_ROUND, telemetry=True)
    telemetry_requests = counter.reset()
    logger.info("  extra requests  : %d", telemetry_requests)
    try:
        sample = session.laps.pick_drivers(session.drivers[0]).iloc[5].get_telemetry()
        logger.info("  sample lap tel  : %s rows, cols=%s",
                    len(sample), sorted(sample.columns)[:8])
    except Exception as exc:  # defensive: telemetry shape varies by session
        logger.warning("  telemetry sample unavailable: %s", exc)

    # ---- Report ----
    races_2023_2024 = 46
    projected = cold_requests * races_2023_2024
    logger.info("=" * 62)
    logger.info("RESULT  cold=%d/session  warm=%d  telemetry=+%d",
                cold_requests, warm_requests, telemetry_requests)
    logger.info("Projected for %d races (2023+2024): %d requests, %.1f hourly windows",
                races_2023_2024, projected, projected / 500)

    if warm_requests != 0:
        logger.error("CACHE NOT PROTECTING THE BUDGET - stop and investigate.")
        return 2
    logger.info("Cache confirmed: repeat loads cost nothing.")

    if cold_requests <= 12:
        logger.info("BRANCH: proceed with the full 2023+2024 plan in one window.")
        return 0
    if cold_requests <= 25:
        logger.warning("BRANCH: split collection into half-season batches.")
        return 0
    logger.error("BRANCH: cost above 25/session - escalate before spending budget.")
    return 3


if __name__ == "__main__":
    sys.exit(main())
