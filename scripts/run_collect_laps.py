"""Collect race laps and weather, one parquet per session.

Resumable by design: already-written sessions are skipped, so recovering from
a crash or a rate limit is simply re-running the same command.

    .venv\\Scripts\\python.exe scripts/run_collect_laps.py --year 2023
    .venv\\Scripts\\python.exe scripts/run_collect_laps.py --year 2024

Exit codes:
    0  all requested sessions present
    1  finished with some sessions failed
    2  stopped on the API rate limit (re-run after the window resets)
    3  cache is cold and --allow-cold-cache was not passed
"""
from __future__ import annotations

import argparse
import gc
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                          # noqa: E402
from src.acquisition import cache, collector                    # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    settings = config.settings()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, action="append",
                        help="season to collect; repeatable. Defaults to the "
                             "configured batch order.")
    parser.add_argument("--rounds", type=str, default=None,
                        help="round filter, e.g. '1-11' or '3,7,9'")
    parser.add_argument("--allow-cold-cache", action="store_true",
                        help="proceed even with an empty FastF1 cache")
    parser.add_argument("--limit", type=int, default=None,
                        help="stop after this many newly collected sessions")
    parser.set_defaults(batch_order=settings["collection"]["batch_order"])
    return parser.parse_args()


def parse_rounds(spec: str | None) -> set[int] | None:
    """Parse a round filter such as ``"1-11"`` or ``"3,7,9"``."""
    if not spec:
        return None
    rounds: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            start, end = part.split("-", 1)
            rounds.update(range(int(start), int(end) + 1))
        elif part:
            rounds.add(int(part))
    return rounds


def collect_year(year: int, args: argparse.Namespace, logger,
                 counter: cache.RequestCounter, ledger: cache.BudgetLedger,
                 collected_so_far: int) -> tuple[int, int, bool]:
    """Collect every race of one season.

    :returns: (newly collected, failed, hit_rate_limit)
    """
    import fastf1

    settings = config.settings()
    out_dir = config.resolve_path("raw_laps", create=True)
    session_type = settings["project"]["session_type"]
    per_session = settings["collection"]["requests_per_session"]
    min_interval = settings["collection"]["min_request_interval_s"]

    manifest = collector.load_manifest(out_dir)
    round_filter = parse_rounds(args.rounds)

    logger.info("Fetching %d event schedule", year)
    ledger.wait_for_headroom(2, f"{year} schedule")
    counter.reset()
    schedule = fastf1.get_event_schedule(year, include_testing=False)
    ledger.record(counter.reset(), f"{year}_schedule")

    rounds = [int(r) for r in schedule["RoundNumber"].tolist() if int(r) > 0]
    if round_filter:
        rounds = [r for r in rounds if r in round_filter]
    logger.info("%d has %d rounds to consider", year, len(rounds))

    new, failed = 0, 0

    for round_number in rounds:
        key = f"{year}_R{round_number:02d}"
        path = collector.session_path(out_dir, year, round_number)

        if path.exists():
            logger.debug("skip %s (already collected)", key)
            continue
        if args.limit is not None and collected_so_far + new >= args.limit:
            logger.info("Reached --limit of %d newly collected sessions", args.limit)
            break

        ledger.wait_for_headroom(per_session, key)
        counter.reset()
        try:
            frame = collector.collect_session(year, round_number, session_type)
            collector.write_session(frame, path)
            manifest[key] = collector.manifest_entry(
                "ok", n_laps=len(frame), n_requests=counter.count,
                event_name=str(frame["EventName"].iloc[0]))
            new += 1
        except collector.RateLimitHit as exc:
            ledger.record(counter.reset(), key)
            manifest[key] = collector.manifest_entry(
                "rate_limited", n_requests=counter.count, detail=str(exc))
            collector.write_manifest(out_dir, manifest)
            logger.error("Rate limit reached at %s. Re-run after the window "
                         "resets; collected sessions are already safe on disk.", key)
            return new, failed, True
        except Exception as exc:                    # defensive: one bad session
            failed += 1                             # must not end the run
            manifest[key] = collector.manifest_entry(
                "failed", n_requests=counter.count,
                detail=f"{type(exc).__name__}: {exc}")
            logger.error("Failed %s: %s: %s", key, type(exc).__name__, exc)
        finally:
            ledger.record(counter.reset(), key)
            collector.write_manifest(out_dir, manifest)
            gc.collect()
            time.sleep(min_interval)

    return new, failed, False


def main() -> int:
    """Collect every configured season, resuming where a previous run stopped."""
    args = parse_args()
    logger = config.setup_logging("collect_laps")

    try:
        cache.init_cache(allow_cold=args.allow_cold_cache)
    except RuntimeError as exc:
        logger.error("%s", exc)
        return 3

    settings = config.settings()
    ledger = cache.BudgetLedger(
        config.resolve_path("logs", create=True) / "api_budget.jsonl",
        limit=settings["collection"]["budget_per_hour"],
        soft_limit=settings["collection"]["budget_soft_limit"])
    counter = cache.RequestCounter()
    counter.install()

    years = args.year or args.batch_order
    logger.info("Collecting seasons %s (budget spent this hour: %d)",
                years, ledger.spent_in_window())

    total_new, total_failed, rate_limited = 0, 0, False
    for year in years:
        new, failed, rate_limited = collect_year(
            year, args, logger, counter, ledger, total_new)
        total_new += new
        total_failed += failed
        logger.info("%d: %d newly collected, %d failed", year, new, failed)
        if rate_limited:
            break

    out_dir = config.resolve_path("raw_laps")
    n_files = len(list(out_dir.glob("laps_*.parquet")))
    logger.info("=" * 62)
    logger.info("Sessions on disk: %d | new this run: %d | failed: %d",
                n_files, total_new, total_failed)
    logger.info("Budget spent in the trailing hour: %d / %d",
                ledger.spent_in_window(), ledger.limit)

    if rate_limited:
        return 2
    return 1 if total_failed else 0


if __name__ == "__main__":
    sys.exit(main())
