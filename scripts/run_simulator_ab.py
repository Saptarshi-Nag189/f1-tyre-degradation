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


def main() -> int:
    """Run the interleaved comparison and write the results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=30,
                        help="alternating measurements per circuit")
    parser.add_argument("--ref", default=BASELINE_REF,
                        help="git revision to compare against")
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
    summary = {
        "baseline_ref": args.ref,
        "rounds_per_circuit": args.rounds,
        "per_circuit": per_circuit,
        "speedup_min": min(speedups),
        "speedup_max": max(speedups),
        "speedup_over_all_circuits": round(total_old / total_new, 1),
        "method": ("Both implementations run alternately in one process, so "
                   "clock drift affects each equally and cancels in the "
                   "ratio. Absolute milliseconds depend on machine state."),
    }
    logger.info("Search is %.1fx faster over the ten circuits combined "
                "(per circuit %.1fx to %.1fx), same answer everywhere",
                summary["speedup_over_all_circuits"],
                summary["speedup_min"], summary["speedup_max"])

    out = config.resolve_path("reports", create=True) / "simulator_ab.json"
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
