"""Compare the current strategy search against the one it replaced.

    .venv\\Scripts\\python.exe scripts/run_simulator_ab.py
    .venv\\Scripts\\python.exe scripts/run_simulator_ab.py --rounds 40

Why this exists rather than a plain before-and-after run.

The development machine is a laptop whose clock varies by a factor of three
between a cold turbo burst and sustained load. Measuring the old code, then
the new code, then subtracting, measures the thermal state as much as the
change: the same unchanged benchmark was recorded at 1.32 ms and 6.13 ms per
prediction twenty minutes apart.

So both implementations run **in one process, alternating**, many times over.
Drift hits the two arms equally and cancels in the ratio. The absolute
milliseconds still depend on what else the machine is doing, and are reported
as a spread rather than a single number.

The old implementation is read out of git rather than kept as a second copy in
the tree, so it cannot drift away from what was actually replaced.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                  # noqa: E402
from src.core.pipeline import F1Pipeline                # noqa: E402
from src.simulation import strategy as current          # noqa: E402

#: The commit whose simulator this replaced.
BASELINE_REF = "dc12da5"

#: Circuits spanning the range of race lengths, and so of search width.
CIRCUITS = ["silverstone", "monaco", "monza", "spa", "singapore",
            "jeddah", "bahrain", "suzuka", "barcelona", "hungaroring"]


def load_baseline(ref: str, logger):
    """Import the strategy module as it stood at ``ref``.

    :param ref: a git revision.
    :returns: the imported module.
    :raises SystemExit: if git cannot produce the file.
    """
    path = f"{ref}:src/simulation/strategy.py"
    try:
        source = subprocess.run(
            ["git", "show", path], cwd=PROJECT_ROOT, check=True,
            capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError) as exc:
        logger.error("Could not read %s from git: %s", path, exc)
        raise SystemExit(2) from exc

    scratch = PROJECT_ROOT / "artifacts" / "reports" / "_strategy_baseline.py"
    scratch.parent.mkdir(parents=True, exist_ok=True)
    scratch.write_text(source, encoding="utf-8")

    spec = importlib.util.spec_from_file_location("strategy_baseline", scratch)
    module = importlib.util.module_from_spec(spec)
    # Registered before execution because @dataclass resolves annotations
    # through sys.modules[cls.__module__], which is None for a module that is
    # still being built.
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_case(pipeline: F1Pipeline, circuit: str) -> dict:
    """Assemble everything both simulators need for one circuit.

    Resolved once, outside the timing loop, so the measurement covers the
    search and nothing else.
    """
    ui_map = config.ui_circuit_map()
    event_name = ui_map["circuits"][circuit]["event_name"]
    params = {"season": 2024, "circuit": circuit, "air_temp": 22,
              "track_temp": 38, "conditions": "dry"}
    compounds = list(current.DRY_COMPOUNDS)

    table, max_observed = pipeline._degradation_table(
        event_name, compounds, params, [])
    pace = pipeline.reference["pace_index"][event_name]

    def lookup(compound: str):
        entry = table[compound]
        return (max(entry["deg_rate"], current.MIN_SIMULATED_DEG_RATE),
                entry["base_offset"], entry["source"])

    return {
        "circuit": circuit,
        "race_laps": pace["race_laps"],
        "base_lap_s": pace["base_lap_s"],
        "compounds": compounds,
        "lookup": lookup,
        "max_observed": max_observed,
        "pit_loss_s": pipeline.config["strategy"]["pit_loss_s"],
        "fuel": pipeline.config["fuel_correction"],
    }


def run_current(case: dict) -> list:
    """Search with the current implementation."""
    return current.enumerate_strategies(
        case["race_laps"], case["compounds"], case["lookup"],
        case["base_lap_s"], case["pit_loss_s"], case["fuel"],
        max_observed_stint=case["max_observed"])


def run_baseline(module, case: dict) -> list:
    """Search with the implementation from ``BASELINE_REF``."""
    return module.enumerate_strategies(
        case["race_laps"], case["compounds"], case["lookup"],
        case["base_lap_s"], case["pit_loss_s"], case["fuel"],
        max_observed_stint=case["max_observed"])


#: Synthetic degradation and pace offsets for the scaling study. Held fixed so
#: race length is the only thing varying; real per-circuit rates would confound
#: it with how far apart the compounds sit.
SYNTHETIC_RATES = {
    "SOFT": (0.120, 0.0, "synthetic"),
    "MEDIUM": (0.065, 0.4, "synthetic"),
    "HARD": (0.030, 0.8, "synthetic"),
}


def synthetic_case(race_laps: int, pipeline: F1Pipeline) -> dict:
    """A costing case at a chosen race length, with everything else fixed."""
    return {
        "circuit": f"synthetic-{race_laps}",
        "race_laps": race_laps,
        "base_lap_s": 90.0,
        "compounds": list(current.DRY_COMPOUNDS),
        "lookup": lambda compound: SYNTHETIC_RATES[compound],
        "max_observed": {},
        "pit_loss_s": pipeline.config["strategy"]["pit_loss_s"],
        "fuel": pipeline.config["fuel_correction"],
    }


def scaling_study(baseline, pipeline: F1Pipeline, lengths: list[int],
                  rounds: int, logger) -> dict:
    """Test whether the speed-up is structural or a constant factor.

    The claim being checked is a complexity one. The old search projected a
    lap time for every lap of every candidate, so its cost goes as
    ``candidates x race_laps``. The new one costs ``candidates`` of
    arithmetic and then materialises exactly one lap-by-lap trace.

    That makes a falsifiable prediction: **the ratio between them should grow
    roughly in proportion to race length.** A constant-factor tidy-up, of the
    kind that comes from writing the same algorithm more carefully, would
    show a flat ratio instead. Measuring the ratio at one race length cannot
    tell the two apart, which is why the per-circuit table alone is not
    evidence for the claim.

    Race length is swept with everything else held fixed. Candidate count is
    recorded rather than assumed, because the search grid steps by
    ``race_laps // 25`` and so does not vary smoothly with length.

    :param baseline: the module from :func:`load_baseline`.
    :param pipeline: a loaded pipeline, for the pit loss and fuel settings.
    :param lengths: race lengths to sweep, in laps.
    :param rounds: alternating measurements at each length.
    :param logger: where to report progress.
    :returns: per-length measurements and the fitted cost constants.
    """
    rows = []
    for race_laps in lengths:
        case = synthetic_case(race_laps, pipeline)
        candidates = len(search_width(case))
        if not candidates:
            logger.warning("  %3d laps: no feasible strategy, skipped",
                           race_laps)
            continue

        mine, theirs = run_current(case), run_baseline(baseline, case)
        assert mine[0].label == theirs[0].label, race_laps
        assert abs(mine[0].total_time_s - theirs[0].total_time_s) < 1e-6

        old_ms: list[float] = []
        new_ms: list[float] = []
        for _ in range(rounds):
            start = time.perf_counter()
            run_baseline(baseline, case)
            old_ms.append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            run_current(case)
            new_ms.append((time.perf_counter() - start) * 1000.0)

        old_median = statistics.median(old_ms)
        new_median = statistics.median(new_ms)
        rows.append({
            "race_laps": race_laps,
            "candidates": candidates,
            "baseline_median_ms": round(old_median, 3),
            "current_median_ms": round(new_median, 3),
            "speedup": round(old_median / new_median, 1),
            # If the model of each cost is right, these two columns are flat.
            "baseline_ns_per_candidate_lap": round(
                old_median * 1e6 / (candidates * race_laps), 1),
            "current_ns_per_candidate": round(
                new_median * 1e6 / candidates, 1),
        })
        logger.info("  %3d laps  %5d candidates  %8.2f ms -> %6.2f ms  "
                    "%5.1fx   old %5.1f ns/cand-lap   new %5.1f ns/cand",
                    race_laps, candidates, old_median, new_median,
                    rows[-1]["speedup"],
                    rows[-1]["baseline_ns_per_candidate_lap"],
                    rows[-1]["current_ns_per_candidate"])

    return {"rows": rows, "fit": _summarise_fit(rows)}


def search_width(case: dict) -> list:
    """Candidates the search costs before pruning."""
    rates = {c: case["lookup"](c) for c in case["compounds"]}
    return current.search_strategies(
        case["race_laps"], case["compounds"], rates, case["base_lap_s"],
        case["pit_loss_s"], case["fuel"])


def _spread(values: list[float]) -> float:
    """Relative spread of a list, as (max - min) / median.

    Used as the test of a cost model: if cost really goes as
    ``candidates x race_laps``, then time divided by that is a constant and
    this number is small. A model that does not hold produces a large one.
    """
    if not values:
        return float("nan")
    median = statistics.median(values)
    return (max(values) - min(values)) / median if median else float("nan")


def _summarise_fit(rows: list[dict]) -> dict:
    """Check both cost models and the predicted growth in the ratio."""
    if len(rows) < 2:
        return {}
    old_norm = [r["baseline_ns_per_candidate_lap"] for r in rows]
    new_norm = [r["current_ns_per_candidate"] for r in rows]
    shortest, longest = rows[0], rows[-1]
    return {
        "baseline_ns_per_candidate_lap_median": round(
            statistics.median(old_norm), 1),
        "baseline_model_spread": round(_spread(old_norm), 2),
        "current_ns_per_candidate_median": round(
            statistics.median(new_norm), 1),
        "current_model_spread": round(_spread(new_norm), 2),
        "speedup_at_shortest": shortest["speedup"],
        "speedup_at_longest": longest["speedup"],
        "race_laps_ratio": round(
            longest["race_laps"] / shortest["race_laps"], 2),
        "speedup_ratio": round(
            longest["speedup"] / shortest["speedup"], 2),
    }


def main() -> int:
    """Run the interleaved comparison and write the results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=30,
                        help="alternating measurements per circuit")
    parser.add_argument("--ref", default=BASELINE_REF,
                        help="git revision to compare against")
    parser.add_argument("--scaling-lengths", default="30,40,50,60,70,80,90,100",
                        help="race lengths for the complexity check")
    args = parser.parse_args()

    logger = config.setup_logging("simulator_ab")
    import logging
    for name in ("src.simulation.strategy", "strategy_baseline"):
        logging.getLogger(name).setLevel(logging.WARNING)

    baseline = load_baseline(args.ref, logger)
    pipeline = F1Pipeline().load()
    logger.info("Comparing the current search against %s, %d alternating "
                "rounds per circuit", args.ref, args.rounds)

    per_circuit = {}
    for circuit in CIRCUITS:
        case = build_case(pipeline, circuit)

        # Correctness before speed: a faster search that answers differently
        # is not a faster search.
        mine, theirs = run_current(case), run_baseline(baseline, case)
        assert mine[0].label == theirs[0].label, circuit
        assert mine[0].pit_laps == theirs[0].pit_laps, circuit
        assert abs(mine[0].total_time_s - theirs[0].total_time_s) < 1e-6, circuit

        old_ms: list[float] = []
        new_ms: list[float] = []
        for _ in range(args.rounds):
            start = time.perf_counter()
            run_baseline(baseline, case)
            old_ms.append((time.perf_counter() - start) * 1000.0)

            start = time.perf_counter()
            run_current(case)
            new_ms.append((time.perf_counter() - start) * 1000.0)

        old_median = statistics.median(old_ms)
        new_median = statistics.median(new_ms)
        per_circuit[circuit] = {
            "race_laps": case["race_laps"],
            "candidates": len(theirs),
            "kept": len(mine),
            "baseline_median_ms": round(old_median, 3),
            "current_median_ms": round(new_median, 3),
            "speedup": round(old_median / new_median, 1),
        }
        logger.info("  %-12s %2d laps  %5d searched -> %3d kept  "
                    "%7.2f ms -> %5.2f ms  %5.1fx", circuit,
                    case["race_laps"], len(theirs), len(mine),
                    old_median, new_median, old_median / new_median)

    speedups = [c["speedup"] for c in per_circuit.values()]
    total_old = sum(c["baseline_median_ms"] for c in per_circuit.values())
    total_new = sum(c["current_median_ms"] for c in per_circuit.values())

    lengths = [int(v) for v in args.scaling_lengths.split(",") if v.strip()]
    logger.info("Scaling: is the speed-up structural, or a constant factor?")
    scaling = scaling_study(baseline, pipeline, lengths, args.rounds, logger)

    summary = {
        "baseline_ref": args.ref,
        "rounds_per_circuit": args.rounds,
        "per_circuit": per_circuit,
        "speedup_min": min(speedups),
        "speedup_max": max(speedups),
        "speedup_over_all_circuits": round(total_old / total_new, 1),
        "scaling": scaling,
        "method": ("Both implementations run alternately in one process, so "
                   "clock drift affects each equally and cancels in the "
                   "ratio. Absolute milliseconds depend on machine state."),
    }
    logger.info("Search is %.1fx faster over the ten circuits combined "
                "(per circuit %.1fx to %.1fx), same answer everywhere",
                summary["speedup_over_all_circuits"],
                summary["speedup_min"], summary["speedup_max"])

    fit = scaling.get("fit", {})
    if fit:
        logger.info("Cost models: old %.1f ns per candidate-lap (spread "
                    "%.2f), new %.1f ns per candidate (spread %.2f)",
                    fit["baseline_ns_per_candidate_lap_median"],
                    fit["baseline_model_spread"],
                    fit["current_ns_per_candidate_median"],
                    fit["current_model_spread"])
        logger.info("Race length x%.2f over the sweep gave speed-up x%.2f "
                    "(%.1fx to %.1fx): %s", fit["race_laps_ratio"],
                    fit["speedup_ratio"], fit["speedup_at_shortest"],
                    fit["speedup_at_longest"],
                    "structural" if fit["speedup_ratio"] > 1.5
                    else "closer to a constant factor")

    out = config.resolve_path("reports", create=True) / "simulator_ab.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
