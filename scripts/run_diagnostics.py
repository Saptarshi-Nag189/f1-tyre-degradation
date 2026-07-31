"""Diagnose what a stint-level feature set can and cannot explain.

Run after Gate 4 to distinguish a weak model from an unreachable target. The
key quantities are the variance decomposition, which bounds what any
event-level feature set can explain, and the oracle ceiling, which is the MAE
achievable with perfect knowledge of each event's mean degradation.

Establishing that ceiling matters because the 15% baseline gate was chosen
before anyone measured what headroom exists.

    .venv\\Scripts\\python.exe scripts/run_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                             # noqa: E402


def main() -> int:
    """Compute and record the target diagnostics."""
    logger = config.setup_logging("diagnostics")
    processed = config.resolve_path("processed")
    path = processed / "stints_target.parquet"
    if not path.exists():
        logger.error("%s not found.", path)
        return 1
    stints = pd.read_parquet(path)

    report: dict = {"n_stints": int(len(stints))}

    # --- variance decomposition: the ceiling for event-level features ---
    event_mean = stints.groupby(["Year", "EventName"])["deg_rate"].transform("mean")
    total = float(stints["deg_rate"].var())
    between = float(event_mean.var())
    within = float((stints["deg_rate"] - event_mean).var())
    report["variance"] = {
        "total": total, "between_event": between, "within_event": within,
        "between_event_pct": 100.0 * between / total,
    }
    logger.info("Variance: %.1f%% between events, %.1f%% within events",
                100.0 * between / total, 100.0 * within / total)
    logger.info("  An event-level feature set can only reach the between share.")

    # --- oracle ceiling ---
    global_mae = float(np.abs(stints["deg_rate"] - stints["deg_rate"].mean()).mean())
    oracle = {}
    for keys in (["Year", "EventName"], ["Year", "EventName", "Compound"]):
        means = stints.groupby(keys)["deg_rate"].transform("mean")
        mae = float(np.abs(stints["deg_rate"] - means).mean())
        oracle["+".join(keys)] = {
            "mae": mae, "improvement_vs_global_mean_pct":
                100.0 * (global_mae - mae) / global_mae}
    report["global_mean_mae"] = global_mae
    report["oracle_ceiling"] = oracle
    best = max(v["improvement_vs_global_mean_pct"] for v in oracle.values())
    logger.info("Oracle ceiling: %.1f%% improvement over the global mean, "
                "and that needs the answer in advance.", best)

    # --- cross-season stability of circuit degradation ---
    pivot = stints.groupby(["EventName", "Year"])["deg_rate"].mean().unstack().dropna()
    stability = {}
    if pivot.shape[1] >= 2:
        cols = list(pivot.columns)
        for i in range(len(cols) - 1):
            a, b = cols[i], cols[i + 1]
            stability[f"{a}_vs_{b}"] = {
                "n_circuits": int(len(pivot)),
                "pearson": float(pivot[a].corr(pivot[b])),
                "spearman": float(pivot[a].corr(pivot[b], method="spearman")),
            }
            logger.info("Circuit degradation %s vs %s: pearson %.3f, spearman %.3f",
                        a, b, stability[f"{a}_vs_{b}"]["pearson"],
                        stability[f"{a}_vs_{b}"]["spearman"])
    report["cross_season_circuit_stability"] = stability

    # --- compound effect, a physical sanity check ---
    labels = {0.0: "HARD", 1.0: "MEDIUM", 2.0: "SOFT"}
    compound = {}
    for value, group in stints[stints["compound_relative"].notna()].groupby(
            "compound_relative"):
        compound[labels[value]] = {
            "n": int(len(group)),
            "mean": float(group["deg_rate"].mean()),
            "median": float(group["deg_rate"].median()),
        }
    report["compound_effect"] = compound
    logger.info("Compound medians: %s",
                {k: round(v["median"], 4) for k, v in compound.items()})

    # --- target contamination ---
    negative = float((stints["deg_rate"] < 0).mean())
    extreme = float((stints["deg_rate"].abs() > 0.3).mean())
    report["contamination"] = {
        "negative_pct": 100.0 * negative,
        "abs_gt_0.3_pct": 100.0 * extreme,
        "min": float(stints["deg_rate"].min()),
        "max": float(stints["deg_rate"].max()),
    }
    logger.info("Contamination: %.1f%% negative slopes, %.1f%% beyond "
                "+/-0.3 s/lap, min %.3f", 100.0 * negative, 100.0 * extreme,
                stints["deg_rate"].min())
    logger.info("  Negative slopes are track evolution, not tyre recovery.")

    # --- TyreLife_start informativeness ---
    report["tyrelife_start"] = {
        "unique": int(stints["TyreLife_start"].nunique()),
        "pct_equal_1": float(100.0 * (stints["TyreLife_start"] == 1).mean()),
    }

    out = config.resolve_path("reports", create=True) / "diagnostics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
