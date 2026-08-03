"""Phase 5 analysis: does the true curvature proxy beat the speed-trap one?

Gate 3 demoted the energy component of the CWI at rho ~= 0, but that verdict
rested on the telemetry-free speed-trap proxy: four trap readings per lap,
standing in for mean(v^2). This re-runs the same pre-registered test using the
curvature-derived physics aggregates, which is what the compass design
actually specifies, and additionally ablates the physics features as a Tier C
addition to the model.

Two distinct questions, deliberately kept apart:

1. Does energy belong in the TARGET? That is the Spearman decision rule.
2. Do physics aggregates help PREDICT the target? That is the tier gate, and
   it is a different question with a different answer.

Note that Tier C aggregates describe laps already driven, so they are
retrospective. A positive result here would mean physics explains degradation,
not that the simulator can use it directly; circuit-level averages would be
needed for forward prediction.

    .venv\\Scripts\\python.exe scripts/run_telemetry_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                          # noqa: E402
from src.modelling import features, splits, xgb_model, tuning    # noqa: E402

#: Per-lap physics aggregates, averaged to stint level.
PHYSICS_COLS = ["lat_accel_max", "lat_accel_mean", "brake_power_max",
                "combined_g_max", "aero_load_proxy", "jerk_rms",
                "throttle_std", "full_throttle_frac", "brake_applications"]


def load_stint_physics() -> pd.DataFrame:
    """Average per-lap physics aggregates up to stint level."""
    telemetry_dir = config.resolve_path("raw_telemetry")
    files = sorted(telemetry_dir.glob("lapagg_*.parquet"))
    if not files:
        raise FileNotFoundError(
            f"No telemetry aggregates in {telemetry_dir}. "
            "Run scripts/run_collect_telemetry.py first.")
    laps = pd.concat([pd.read_parquet(p) for p in files], ignore_index=True)
    laps = laps.dropna(subset=["Stint"])
    grouped = laps.groupby(["Year", "RoundNumber", "DriverNumber", "Stint"])
    stint_physics = grouped[PHYSICS_COLS].mean().reset_index()
    stint_physics["DriverNumber"] = stint_physics["DriverNumber"].astype(str)
    return stint_physics


def main() -> int:
    """Run both questions and record the outcome."""
    logger = config.setup_logging("telemetry_study")
    settings = config.settings()

    stints = pd.read_parquet(
        config.resolve_path("processed") / "stints_target.parquet")
    stints["DriverNumber"] = stints["DriverNumber"].astype(str)
    physics = load_stint_physics()

    merged = stints.merge(physics, on=["Year", "RoundNumber", "DriverNumber", "Stint"],
                          how="inner")
    logger.info("Matched %d stints with telemetry (of %d total, %d covered "
                "sessions)", len(merged), len(stints),
                merged.groupby(["Year", "RoundNumber"]).ngroups)
    if len(merged) < 50:
        logger.error("Too few matched stints (%d) for a reliable study", len(merged))
        return 1

    report: dict = {"n_matched_stints": int(len(merged)),
                    "n_sessions": int(merged.groupby(["Year", "RoundNumber"]).ngroups)}

    # --- Question 1: does energy belong in the target? ---
    logger.info("--- Question 1: energy in the TARGET (Spearman rule) ---")
    target_cfg = settings["target"]
    correlations = {}
    for column in ["aero_load_proxy", "lat_accel_mean", "lat_accel_max",
                   "combined_g_max", "brake_power_max"]:
        valid = merged.dropna(subset=[column, "deg_rate"])
        rho, p_value = spearmanr(valid[column], valid["deg_rate"])
        correlations[column] = {"rho": float(rho), "p": float(p_value),
                                "n": int(len(valid))}
        logger.info("  Spearman(%-18s, deg_rate) = %+.4f  (p=%.3g, n=%d)",
                    column, rho, p_value, len(valid))

    best = max(correlations.items(), key=lambda kv: abs(kv[1]["rho"]))
    rho = best[1]["rho"]
    keep = target_cfg["spearman_keep_threshold"]
    low = target_cfg["spearman_downweight_low"]
    if rho > keep:
        decision = "energy_in_target"
    elif rho >= low:
        decision = "energy_downweighted"
    else:
        decision = "energy_demoted_to_feature"
    report["target_correlations"] = correlations
    report["best_proxy"] = best[0]
    report["decision"] = decision
    logger.info("  Strongest proxy: %s at rho %+.4f -> %s",
                best[0], rho, decision)
    logger.info("  Speed-trap proxy for comparison: rho ~= 0.00 "
                "(the Stage-0 verdict this test re-examines)")

    # --- Question 2: do physics features help PREDICT? ---
    logger.info("--- Question 2: physics as a Tier C feature block ---")
    try:
        train, holdout = splits.season_holdout(
            merged, settings["modelling"]["train_seasons"],
            settings["modelling"]["holdout_season"])
        can_split = True
    except ValueError as exc:
        logger.warning("Chronological holdout unavailable on the pilot subset "
                       "(%s); using a grouped-CV comparison instead.", exc)
        can_split = False

    core = features.available(merged.columns)
    tier_c = [c for c in PHYSICS_COLS if c in merged.columns]
    results = {}

    if can_split and len(holdout) >= 40:
        from sklearn.metrics import mean_absolute_error, r2_score
        # Both arms are tuned separately. Comparing at fixed default parameters
        # measures whether physics rescues a badly configured model, not
        # whether it improves a good one, and on this data the two answers
        # differ by a factor of three (+4.63% untuned against +1.49% tuned).
        # The same trap produced an apparent gain from Driver features that
        # vanished under a fair comparison.
        for label, feature_set in (("core", core), ("core + physics", core + tier_c)):
            categories = features.training_categories(train)
            x_train, columns = features.build_matrix(train, feature_set, categories)
            x_holdout, _ = features.build_matrix(holdout, feature_set, categories)

            tuning_frame = pd.concat(
                [train.reset_index(drop=True), x_train.reset_index(drop=True)[
                    [c for c in columns if c not in train.columns]]], axis=1)
            best = tuning.tune(tuning_frame, columns, "deg_rate",
                               n_trials=settings["modelling"]["optuna_trials"],
                               n_splits=settings["modelling"]["ts_splits"])

            model = xgb_model.XGBRegressor(
                **{**xgb_model.DEFAULT_PARAMS, **best["best_params"]})
            model.fit(x_train.to_numpy(float), train["deg_rate"].to_numpy(float))
            predictions = model.predict(x_holdout.to_numpy(float))
            mae = float(mean_absolute_error(holdout["deg_rate"], predictions))
            results[label] = {"mae": mae,
                              "r2": float(r2_score(holdout["deg_rate"], predictions)),
                              "n_features": len(columns), "tuned": True}
            logger.info("  %-16s MAE %.5f  R2 %+.4f  (%d features, tuned)",
                        label, mae, results[label]["r2"], len(columns))
    else:
        n_splits = min(3, max(2, len(splits.event_order(merged)) - 1))
        for label, feature_set in (("core", core), ("core + physics", core + tier_c)):
            categories = features.training_categories(merged)
            matrix, columns = features.build_matrix(merged, feature_set, categories)
            frame = pd.concat([merged.reset_index(drop=True),
                               matrix.reset_index(drop=True)[
                                   [c for c in columns if c not in merged.columns]]],
                              axis=1)
            mae = xgb_model.cv_mae(frame, columns, "deg_rate", n_splits)
            results[label] = {"cv_mae": mae, "n_features": len(columns)}
            logger.info("  %-16s grouped-CV MAE %.5f  (%d features)",
                        label, mae, len(columns))

    key = "mae" if "mae" in results.get("core", {}) else "cv_mae"
    gain = 100.0 * (results["core"][key] - results["core + physics"][key]) \
        / results["core"][key]
    gate = settings["modelling"]["tier_gate_pct"]
    report["tier_c"] = {"results": results, "gain_pct": gain,
                        "gate_pct": gate, "pass": bool(gain >= gate)}

    logger.info("=" * 62)
    logger.info("TARGET  best physics proxy rho %+.4f -> %s", rho, decision)
    logger.info("TIER C  %+.2f%% improvement (gate %.1f%%): %s",
                gain, gate, "PASS" if gain >= gate else "FAIL")

    out = config.resolve_path("reports", create=True) / "telemetry_study.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
