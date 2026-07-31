"""Phase 5: collect telemetry and reduce it to per-lap physics aggregates.

Marginal API cost is +2 requests per session, because ``car_data`` and
``position_data`` are single bulk endpoints and everything else is already
cached. The reason to defer telemetry was never the request budget; it was
memory and scientific sequencing. Raw frames are therefore aggregated per lap
and discarded, never persisted: the superseded pipeline pickled them and paid
~245 MB per event, where this pays roughly 0.5 MB.

Defaults to a 10-race pilot, which is what the plan specifies when the
telemetry-free Stage-0 energy proxy comes back ambiguous.

    .venv\\Scripts\\python.exe scripts/run_collect_telemetry.py --limit 10
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                     # noqa: E402
from src.acquisition import cache                          # noqa: E402
from src.features import physics                           # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--limit", type=int, default=10,
                        help="number of sessions to process (pilot size)")
    parser.add_argument("--year", type=int, action="append",
                        help="restrict to these seasons")
    parser.add_argument("--window", type=int, default=11,
                        help="Savitzky-Golay window")
    return parser.parse_args()


def process_session(year: int, round_number: int, window: int) -> pd.DataFrame:
    """Load one session's telemetry and reduce it to per-lap aggregates.

    :param year: season year.
    :param round_number: FIA round number.
    :param window: Savitzky-Golay window.
    :returns: one row per lap of aggregates plus identifying keys.
    """
    import fastf1

    session = fastf1.get_session(year, round_number, "R")
    session.load(laps=True, telemetry=True, weather=False, messages=False)

    rows = []
    for _, lap in session.laps.iterlaps():
        try:
            telemetry = lap.get_telemetry()
        except Exception:                       # defensive: incomplete laps
            continue
        aggregates = physics.lap_physics_features(telemetry, window=window)
        aggregates.update({
            "Year": int(year), "RoundNumber": int(round_number),
            "DriverNumber": str(lap["DriverNumber"]),
            "LapNumber": float(lap["LapNumber"]),
            "Stint": float(lap["Stint"]) if pd.notna(lap["Stint"]) else float("nan"),
        })
        rows.append(aggregates)

    # Raw telemetry is dropped here deliberately; only aggregates survive.
    del session
    gc.collect()
    return pd.DataFrame(rows)


def main() -> int:
    """Run the telemetry pilot."""
    args = parse_args()
    logger = config.setup_logging("collect_telemetry")

    try:
        cache.init_cache()
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

    laps_dir = config.resolve_path("raw_laps")
    out_dir = config.resolve_path("raw_telemetry", create=True)

    sessions = []
    for path in sorted(laps_dir.glob("laps_*.parquet")):
        _, year, rnd = path.stem.split("_")
        year, rnd = int(year), int(rnd.lstrip("R"))
        if args.year and year not in args.year:
            continue
        sessions.append((year, rnd))
    sessions = sessions[:args.limit]
    logger.info("Telemetry pilot over %d sessions", len(sessions))

    for year, rnd in sessions:
        out_path = out_dir / f"lapagg_{year}_R{rnd:02d}.parquet"
        if out_path.exists():
            continue
        ledger.wait_for_headroom(3, f"{year}_R{rnd:02d}_telemetry")
        counter.reset()
        try:
            frame = process_session(year, rnd, args.window)
            if frame.empty:
                logger.warning("No telemetry rows for %s R%02d", year, rnd)
                continue
            tmp = out_path.with_suffix(".parquet.tmp")
            frame.to_parquet(tmp, engine="pyarrow", index=False)
            os.replace(tmp, out_path)
            logger.info("  %s R%02d: %d laps aggregated, %d requests, %.0f KB",
                        year, rnd, len(frame), counter.count,
                        out_path.stat().st_size / 1024)
        except Exception as exc:
            logger.error("Failed %s R%02d: %s: %s", year, rnd,
                         type(exc).__name__, exc)
        finally:
            ledger.record(counter.reset(), f"{year}_R{rnd:02d}_telemetry")
            gc.collect()

    files = sorted(out_dir.glob("lapagg_*.parquet"))
    if not files:
        logger.error("No telemetry aggregates produced.")
        return 1

    combined = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    logger.info("=" * 62)
    logger.info("Aggregated %d laps across %d sessions, %.1f MB total",
                len(combined), len(files),
                sum(p.stat().st_size for p in files) / 1e6)
    checks = physics.plausibility_report(combined)
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
