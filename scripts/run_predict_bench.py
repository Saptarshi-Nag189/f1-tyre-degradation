"""Time ``predict_strategy`` in process, with no HTTP in the way.

Separates the cost of the prediction itself from the cost of serving it. Run
this and ``run_load_test.py`` together: the difference between them is what the
web layer adds.

Latency varies by circuit, and not in the direction intuition suggests. The
search grid steps by ``race_laps // 25`` laps, so a 44-lap race is searched at
one-lap resolution and a 78-lap race at three, leaving the shortest races with
the widest search. Spa generates 5,382 candidates against Monaco's 480. The
per-circuit table below reports this rather than hiding it in an average.

    .venv\\Scripts\\python.exe scripts/run_predict_bench.py
    .venv\\Scripts\\python.exe scripts/run_predict_bench.py --repeats 500
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                  # noqa: E402
from src.core.pipeline import F1Pipeline                # noqa: E402

#: Circuits spanning the range of race lengths, and therefore of search width.
CIRCUITS = ["silverstone", "monaco", "monza", "spa", "singapore",
            "jeddah", "bahrain", "suzuka", "barcelona", "hungaroring"]


def request_for(circuit: str) -> dict:
    """Build a representative prediction request."""
    return {"season": 2024, "circuit": circuit, "air_temp": 22,
            "track_temp": 38, "conditions": "dry",
            "compounds": ["SOFT", "MEDIUM", "HARD"]}


def percentile(values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not values:
        return float("nan")
    index = min(len(values) - 1, int(fraction * len(values)))
    return values[index]


def time_requests(pipeline: F1Pipeline, requests: list[dict], repeats: int
                  ) -> list[float]:
    """Time ``repeats`` predictions, cycling through ``requests``.

    :returns: sorted latencies in milliseconds.
    """
    latencies = []
    for index in range(repeats):
        params = requests[index % len(requests)]
        start = time.perf_counter()
        pipeline.predict_strategy(params)
        latencies.append((time.perf_counter() - start) * 1000.0)
    latencies.sort()
    return latencies


def main() -> int:
    """Run the benchmark and write the results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=1000,
                        help="predictions across the mixed workload")
    parser.add_argument("--warmup", type=int, default=50,
                        help="untimed predictions before measuring")
    args = parser.parse_args()

    logger = config.setup_logging("predict_bench")
    import logging
    logging.getLogger("src.simulation.strategy").setLevel(logging.WARNING)

    start = time.perf_counter()
    pipeline = F1Pipeline().load()
    load_s = time.perf_counter() - start
    logger.info("Model and reference loaded in %.2f s", load_s)

    requests = [request_for(c) for c in CIRCUITS]
    for index in range(args.warmup):
        pipeline.predict_strategy(requests[index % len(requests)])

    per_circuit = {}
    for circuit in CIRCUITS:
        latencies = time_requests(pipeline, [request_for(circuit)], 200)
        race_laps = sum(s["laps"] for s in pipeline.predict_strategy(
            request_for(circuit))["best_strategy"]["stints"])
        searched = len(_search_width(pipeline, circuit, race_laps))
        per_circuit[circuit] = {
            "race_laps": race_laps,
            "candidates_searched": searched,
            "median_ms": round(statistics.median(latencies), 3),
            "p95_ms": round(percentile(latencies, 0.95), 3),
        }
        logger.info("  %-12s %2d laps, %5d candidates, median %6.2f ms, "
                    "p95 %6.2f ms", circuit, race_laps, searched,
                    per_circuit[circuit]["median_ms"],
                    per_circuit[circuit]["p95_ms"])

    mixed = time_requests(pipeline, requests, args.repeats)
    summary = {
        "load_s": round(load_s, 2),
        "repeats": args.repeats,
        "median_ms": round(statistics.median(mixed), 3),
        "p95_ms": round(percentile(mixed, 0.95), 3),
        "p99_ms": round(percentile(mixed, 0.99), 3),
        "max_ms": round(mixed[-1], 3),
        "single_thread_rps": round(1000.0 / statistics.mean(mixed), 1),
        "per_circuit": per_circuit,
    }
    logger.info("Mixed workload over %d predictions: median %.2f ms, "
                "p95 %.2f ms, p99 %.2f ms, %.0f predictions/s on one thread",
                args.repeats, summary["median_ms"], summary["p95_ms"],
                summary["p99_ms"], summary["single_thread_rps"])

    out = config.resolve_path("reports", create=True) / "predict_bench.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


def _search_width(pipeline: F1Pipeline, circuit: str, race_laps: int) -> list:
    """Number of candidates the search costs before pruning.

    Reported because it, not race length, is what drives latency.
    """
    from src.simulation import strategy as strat
    ui_map = config.ui_circuit_map()
    event_name = ui_map["circuits"][circuit]["event_name"]
    compounds = list(strat.DRY_COMPOUNDS)
    table, _ = pipeline._degradation_table(
        event_name, compounds, request_for(circuit), [])
    rates = {c: (table[c]["deg_rate"], table[c]["base_offset"], "x")
             for c in compounds}
    return strat.search_strategies(
        race_laps, compounds, rates, 90.0,
        pipeline.config["strategy"]["pit_loss_s"],
        pipeline.config["fuel_correction"])


if __name__ == "__main__":
    sys.exit(main())
