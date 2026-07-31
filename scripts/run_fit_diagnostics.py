"""Evidence on whether the model is over-fitted, under-fitted, or neither.

No single number settles this, so four independent lines of evidence are
gathered, each of which fails in a different, recognisable way:

1. **The train / CV / holdout triple.** A large train-to-holdout gap means
   memorisation. CV close to holdout means the validation scheme is honest.
2. **A capacity sweep.** If the best model sits at the most complex end of the
   sweep, capacity is still being under-used; if at the simplest end, the
   chosen model is probably too complex. A proper fit sits in the interior.
3. **A learning curve.** If holdout error is still falling as training data is
   added, the model is data-limited rather than capacity-limited.
4. **Residual structure.** Residuals should not correlate with the prediction
   or drift across the holdout season.

    .venv\\Scripts\\python.exe scripts/run_fit_diagnostics.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, r2_score
from xgboost import XGBRegressor

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                     # noqa: E402
from src.modelling import features, splits, xgb_model      # noqa: E402


def load_params() -> dict:
    """Load the persisted tuned parameters, falling back to defaults."""
    path = config.CONFIG_DIR / "tuned_params.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["params"]
    return dict(xgb_model.DEFAULT_PARAMS)


def main() -> int:
    """Gather all four lines of evidence and summarise the verdict."""
    logger = config.setup_logging("fit_diagnostics")
    settings = config.settings()
    model_cfg = settings["modelling"]

    stints = pd.read_parquet(
        config.resolve_path("processed") / "stints_target.parquet")
    train, holdout = splits.season_holdout(
        stints, model_cfg["train_seasons"], model_cfg["holdout_season"])

    numeric = features.available(stints.columns)
    categories = features.training_categories(train)
    x_train, columns = features.build_matrix(train, numeric, categories)
    x_holdout, _ = features.build_matrix(holdout, numeric, categories)
    y_train = train["deg_rate"].to_numpy(float)
    y_holdout = holdout["deg_rate"].to_numpy(float)

    params = load_params()
    report: dict = {"params": params, "n_train": len(train),
                    "n_holdout": len(holdout), "n_features": len(columns)}

    # ---- 1. train / CV / holdout ----
    logger.info("--- 1. train / CV / holdout ---")
    model = XGBRegressor(**params)
    model.fit(x_train.to_numpy(float), y_train)
    train_pred = model.predict(x_train.to_numpy(float))
    holdout_pred = model.predict(x_holdout.to_numpy(float))

    train_mae = float(mean_absolute_error(y_train, train_pred))
    holdout_mae = float(mean_absolute_error(y_holdout, holdout_pred))

    frame = pd.concat([train.reset_index(drop=True),
                       x_train.reset_index(drop=True)[
                           [c for c in columns if c not in train.columns]]], axis=1)
    cv = xgb_model.cv_mae(frame, columns, "deg_rate",
                          model_cfg["ts_splits"], params)

    gap = 100.0 * (holdout_mae - train_mae) / train_mae
    report["fit"] = {"train_mae": train_mae, "cv_mae": cv,
                     "holdout_mae": holdout_mae, "train_to_holdout_gap_pct": gap,
                     "train_r2": float(r2_score(y_train, train_pred)),
                     "holdout_r2": float(r2_score(y_holdout, holdout_pred))}
    logger.info("  train   MAE %.5f  R2 %+.4f", train_mae, report["fit"]["train_r2"])
    logger.info("  CV      MAE %.5f", cv)
    logger.info("  holdout MAE %.5f  R2 %+.4f", holdout_mae, report["fit"]["holdout_r2"])
    logger.info("  train-to-holdout gap: %+.1f%%", gap)
    logger.info("  CV-to-holdout gap:    %+.1f%%",
                100.0 * (holdout_mae - cv) / cv)

    # ---- 2. capacity sweep ----
    logger.info("--- 2. capacity sweep (is the optimum interior?) ---")
    sweep = []
    for n_estimators, max_depth in [(20, 2), (50, 3), (71, 5), (150, 5),
                                    (400, 4), (400, 8), (1000, 8)]:
        candidate = {**params, "n_estimators": n_estimators, "max_depth": max_depth}
        m = XGBRegressor(**candidate)
        m.fit(x_train.to_numpy(float), y_train)
        tr = float(mean_absolute_error(y_train, m.predict(x_train.to_numpy(float))))
        ho = float(mean_absolute_error(y_holdout, m.predict(x_holdout.to_numpy(float))))
        sweep.append({"n_estimators": n_estimators, "max_depth": max_depth,
                      "train_mae": tr, "holdout_mae": ho,
                      "gap_pct": 100.0 * (ho - tr) / tr})
        logger.info("  trees=%4d depth=%d  train %.5f  holdout %.5f  gap %+6.1f%%",
                    n_estimators, max_depth, tr, ho, sweep[-1]["gap_pct"])
    best = min(sweep, key=lambda r: r["holdout_mae"])
    interior = sweep[0] is not best and sweep[-1] is not best
    report["capacity_sweep"] = {"sweep": sweep, "best": best,
                                "optimum_is_interior": bool(interior)}
    logger.info("  best holdout at trees=%d depth=%d; optimum interior: %s",
                best["n_estimators"], best["max_depth"], interior)

    # ---- 3. learning curve ----
    logger.info("--- 3. learning curve (is more data still helping?) ---")
    curve = []
    events = splits.event_order(train)
    rows = splits.event_series(train)
    for fraction in (0.25, 0.5, 0.75, 1.0):
        subset_events = set(events[:max(2, int(len(events) * fraction))])
        mask = rows.isin(subset_events).to_numpy()
        m = XGBRegressor(**params)
        m.fit(x_train.to_numpy(float)[mask], y_train[mask])
        ho = float(mean_absolute_error(
            y_holdout, m.predict(x_holdout.to_numpy(float))))
        curve.append({"fraction": fraction, "n_events": len(subset_events),
                      "n_stints": int(mask.sum()), "holdout_mae": ho})
        logger.info("  %3.0f%% of events (%3d stints): holdout MAE %.5f",
                    fraction * 100, int(mask.sum()), ho)
    still_improving = curve[-1]["holdout_mae"] < curve[-2]["holdout_mae"] * 0.99
    report["learning_curve"] = {"curve": curve, "still_improving": bool(still_improving)}
    logger.info("  still improving with more data: %s", still_improving)

    # ---- 4. residual structure ----
    logger.info("--- 4. residual structure ---")
    residuals = y_holdout - holdout_pred
    r_pred, p_pred = pearsonr(holdout_pred, residuals)
    order = holdout.sort_values(splits.EVENT_KEYS).index
    drift_r, drift_p = pearsonr(np.arange(len(order)),
                                (y_holdout - holdout_pred)[
                                    holdout.index.get_indexer(order)])
    report["residuals"] = {
        "corr_with_prediction": float(r_pred), "p_prediction": float(p_pred),
        "corr_with_time": float(drift_r), "p_time": float(drift_p),
        "mean": float(residuals.mean()), "std": float(residuals.std()),
    }
    logger.info("  corr(prediction, residual) = %+.3f (p=%.3g)", r_pred, p_pred)
    logger.info("  corr(time, residual)       = %+.3f (p=%.3g)", drift_r, drift_p)
    logger.info("  residual mean %+.5f, sd %.5f", residuals.mean(), residuals.std())

    # ---- verdict ----
    overfit = gap > 40.0
    underfit = report["capacity_sweep"]["best"] is sweep[-1]
    logger.info("=" * 62)
    logger.info("VERDICT")
    logger.info("  overfitting  (train-holdout gap > 40%%): %s  [gap %+.1f%%]",
                overfit, gap)
    logger.info("  underfitting (best at max capacity)   : %s", underfit)
    logger.info("  optimum interior to the sweep          : %s", interior)
    logger.info("  data-limited (curve still falling)     : %s", still_improving)
    report["verdict"] = {"overfitting": bool(overfit), "underfitting": bool(underfit),
                         "optimum_interior": bool(interior),
                         "data_limited": bool(still_improving)}

    out = config.resolve_path("reports", create=True) / "fit_diagnostics.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
