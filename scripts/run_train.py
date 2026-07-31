"""Phase 4: train and evaluate the degradation model.

    .venv\\Scripts\\python.exe scripts/run_train.py

Gate 4: XGBoost must beat the LinearRegression baseline by at least
``modelling.baseline_gate_pct`` on holdout MAE. If it fails, the split-half
reliability recorded by the CWI study distinguishes a noisy target from a weak
model, and the gate should be reported against that number rather than tuned
against.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                        # noqa: E402
from src.modelling import baseline, features, splits, xgb_model   # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", default="deg_rate",
                        choices=["deg_rate", "CWI"],
                        help="deg_rate is in s/lap and is what the simulator uses")
    parser.add_argument("--tune", action="store_true",
                        help="run the Optuna search before the final fit")
    return parser.parse_args()


def main() -> int:
    """Train, evaluate and record Gate 4."""
    args = parse_args()
    logger = config.setup_logging("train")
    settings = config.settings()
    model_cfg = settings["modelling"]

    processed = config.resolve_path("processed")
    path = processed / "stints_target.parquet"
    if not path.exists():
        logger.error("%s not found. Run scripts/run_cwi_study.py first.", path)
        return 1
    stints = pd.read_parquet(path)
    logger.info("Loaded %d stints, seasons %s",
                len(stints), sorted(stints["Year"].unique().tolist()))

    # --- features ---
    feats = features.available(stints.columns)
    logger.info("Tier A features (%d): %s", len(feats), feats)
    leaked = [f for f in feats if f in features.FORBIDDEN]
    if leaked:
        logger.error("Forbidden columns present in the feature set: %s", leaked)
        return 1

    # --- chronological split ---
    train, holdout = splits.season_holdout(
        stints, model_cfg["train_seasons"], model_cfg["holdout_season"])

    n_events = len(splits.event_order(train))
    n_splits = min(model_cfg["ts_splits"], max(2, n_events - 1))
    if n_splits != model_cfg["ts_splits"]:
        logger.warning("Reduced to %d folds; only %d training events available",
                       n_splits, n_events)
    splits.assert_no_event_leakage(train, n_splits)

    target = args.target
    train = train.dropna(subset=[target])
    holdout = holdout.dropna(subset=[target])

    # --- models ---
    logger.info("--- mean predictor (floor) ---")
    mean_metrics = baseline.mean_predictor(train, holdout, target)

    logger.info("--- linear baseline ---")
    base = baseline.train_baseline(train, holdout, feats, target)

    logger.info("--- XGBoost ---")
    params = None
    if args.tune:
        from src.modelling import tuning
        tuned = tuning.tune(train, feats, target,
                            n_trials=model_cfg["optuna_trials"],
                            n_splits=n_splits)
        params = tuned["best_params"]
        logger.info("Best CV MAE %.5f with %s", tuned["best_cv_mae"], params)

    xgb_result = xgb_model.train_xgb(train, holdout, feats, target, params)

    cv = xgb_model.cv_mae(train, feats, target, n_splits, params)
    logger.info("XGBoost grouped-chronological CV MAE: %.5f s/lap", cv)

    # --- SHAP sanity check (Gate 6) ---
    ranking = xgb_model.shap_ranking(xgb_result["model"], holdout, feats)
    logger.info("--- SHAP ranking ---")
    for name, value in list(ranking.items())[:8]:
        logger.info("  %-20s %.5f", name, value)
    top8 = set(list(ranking)[:8])
    expected = {"TyreLife_start", "TrackTemp_mean"}
    expected_compound = {"compound_ordinal", "compound_relative"}
    gate6 = expected.issubset(top8) and bool(expected_compound & top8)

    # --- Gate 4 ---
    base_mae = base["metrics"]["MAE"]
    xgb_mae = xgb_result["metrics"]["MAE"]
    improvement = 100.0 * (base_mae - xgb_mae) / base_mae
    gate4 = improvement >= model_cfg["baseline_gate_pct"]

    logger.info("=" * 62)
    logger.info("Holdout season %d, %d stints", model_cfg["holdout_season"], len(holdout))
    logger.info("  mean predictor MAE %.5f s/lap", mean_metrics["MAE"])
    logger.info("  linear baseline MAE %.5f s/lap, R2 %+.3f",
                base_mae, base["metrics"]["R2"])
    logger.info("  XGBoost        MAE %.5f s/lap, R2 %+.3f",
                xgb_mae, xgb_result["metrics"]["R2"])
    logger.info("  within 0.02 s/lap: %.1f%% | within 0.05 s/lap: %.1f%%",
                xgb_result["metrics"]["within_0.02_s_per_lap"],
                xgb_result["metrics"]["within_0.05_s_per_lap"])
    logger.info("GATE 4  improvement over baseline %+.1f%% (need >= %.1f%%): %s",
                improvement, model_cfg["baseline_gate_pct"],
                "PASS" if gate4 else "FAIL")
    logger.info("GATE 6  TyreLife/TrackTemp/compound in SHAP top 8: %s",
                "PASS" if gate6 else "FAIL")

    # --- persist ---
    models_dir = config.resolve_path("models", create=True)
    reports_dir = config.resolve_path("reports", create=True)
    joblib.dump({"model": xgb_result["model"], "features": feats,
                 "target": target, "medians": base["medians"].to_dict()},
                models_dir / f"xgb_{target}.joblib")

    report = {
        "target": target,
        "features": feats,
        "n_train": len(train), "n_holdout": len(holdout),
        "train_seasons": sorted(train["Year"].unique().tolist()),
        "holdout_season": model_cfg["holdout_season"],
        "n_folds": n_splits,
        "cv_mae": cv,
        "metrics": {"mean_predictor": mean_metrics,
                    "baseline": base["metrics"],
                    "xgboost": xgb_result["metrics"]},
        "improvement_pct": improvement,
        "gate4_pass": bool(gate4),
        "gate6_pass": bool(gate6),
        "shap_ranking": ranking,
        "importance": xgb_result["importance"],
        "params": xgb_result["params"],
    }
    (reports_dir / f"train_{target}.json").write_text(
        json.dumps(report, indent=2, default=str), encoding="utf-8")
    logger.info("Wrote %s", reports_dir / f"train_{target}.json")

    return 0 if gate4 else 1


if __name__ == "__main__":
    sys.exit(main())
