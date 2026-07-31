"""Phase 4: XGBoost on the per-stint tabular target.

Tree ensembles remain state of the art on medium-sized tabular data
(grinsztajn_2022), and a per-stint table of order 1,500 rows is exactly that
regime. XGBoost handles the missing ``compound_ordinal`` natively, which
matters because Pirelli nominations are confirmed for only a minority of events.

Runs on CPU deliberately. ``device='cuda'`` on a matrix of this size is slower
than ``tree_method='hist'`` on CPU, because PCIe transfer dominates the actual
computation. The GPU has no role on this critical path.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from xgboost import XGBRegressor

from src.modelling import splits
from src.modelling.baseline import evaluate

logger = logging.getLogger(__name__)

DEFAULT_PARAMS = {
    "n_estimators": 400,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
    "min_child_weight": 5,
    "tree_method": "hist",
    "random_state": 42,
}


def cv_mae(train: pd.DataFrame, features: list[str], target: str,
           n_splits: int, params: dict | None = None) -> float:
    """Mean cross-validated MAE using the grouped chronological split.

    Whole events are held out together, so no race contributes stints to both
    sides of a fold boundary.

    :param train: training stints, must carry ``Year`` and ``RoundNumber``.
    :param features: feature names.
    :param target: target column.
    :param n_splits: number of expanding-window folds.
    :param params: XGBoost parameters; defaults are used when None.
    :returns: mean MAE across folds.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    ordered = train.sort_values(splits.EVENT_KEYS).reset_index(drop=True)
    x = ordered[features].to_numpy(dtype=float)
    y = ordered[target].to_numpy(dtype=float)

    fold_maes = []
    for train_idx, val_idx in splits.grouped_time_series_split(ordered, n_splits):
        model = XGBRegressor(**params)
        model.fit(x[train_idx], y[train_idx])
        fold_maes.append(float(np.mean(np.abs(y[val_idx] - model.predict(x[val_idx])))))
    return float(np.mean(fold_maes))


def train_xgb(train: pd.DataFrame, holdout: pd.DataFrame, features: list[str],
              target: str = "deg_rate", params: dict | None = None) -> dict:
    """Fit XGBoost on the training seasons and evaluate on the holdout season.

    :param train: training stints.
    :param holdout: held-out stints.
    :param features: feature names.
    :param target: target column.
    :param params: XGBoost parameters; defaults are used when None.
    :returns: mapping with the model, metrics and gain-based importances.
    """
    params = {**DEFAULT_PARAMS, **(params or {})}
    feats = [f for f in features if f in train.columns]

    model = XGBRegressor(**params)
    model.fit(train[feats].to_numpy(dtype=float),
              train[target].to_numpy(dtype=float))

    predictions = model.predict(holdout[feats].to_numpy(dtype=float))
    metrics = evaluate(holdout[target].to_numpy(dtype=float), predictions)
    logger.info("XGBoost holdout MAE %.5f s/lap, R2 %.3f",
                metrics["MAE"], metrics["R2"])

    importance = dict(zip(feats, (float(v) for v in model.feature_importances_)))
    for name, value in sorted(importance.items(), key=lambda kv: -kv[1])[:6]:
        logger.info("  %-20s %.4f", name, value)

    return {"model": model, "metrics": metrics, "features": feats,
            "importance": importance, "params": params,
            "predictions": predictions}


def shap_ranking(model: XGBRegressor, frame: pd.DataFrame,
                 features: list[str]) -> dict[str, float]:
    """Global feature ranking by mean absolute SHAP value.

    Uses XGBoost's native exact TreeSHAP via ``pred_contribs``, which avoids
    depending on the ``shap`` package and its numba/llvmlite chain.

    :param model: a fitted XGBRegressor.
    :param frame: rows to explain.
    :param features: feature names, in model order.
    :returns: mapping of feature to mean |SHAP|, descending.
    """
    import xgboost as xgb

    matrix = xgb.DMatrix(frame[features].to_numpy(dtype=float),
                         feature_names=features)
    contribs = model.get_booster().predict(matrix, pred_contribs=True)
    # Final column is the bias term.
    mean_abs = np.abs(contribs[:, :-1]).mean(axis=0)
    ranking = dict(zip(features, (float(v) for v in mean_abs)))
    return dict(sorted(ranking.items(), key=lambda kv: -kv[1]))
