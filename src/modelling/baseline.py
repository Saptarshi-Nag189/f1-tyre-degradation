"""Phase 4: the LinearRegression baseline the tree model must beat.

The gate is deliberately demanding: XGBoost must beat this by at least 15% on
holdout MAE (``modelling.baseline_gate_pct``). A gradient-boosted model that
cannot clear a linear fit on nine features is not earning its complexity.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

logger = logging.getLogger(__name__)


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Compute the standard metric set for a degradation-rate prediction.

    :param y_true: observed degradation rates, s/lap.
    :param y_pred: predicted degradation rates, s/lap.
    :returns: mapping of metric name to value.
    """
    errors = np.abs(y_true - y_pred)
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "RMSE": float(np.sqrt(np.mean((y_true - y_pred) ** 2))),
        "MedAE": float(np.median(errors)),
        "R2": float(r2_score(y_true, y_pred)),
        "within_0.02_s_per_lap": float(100.0 * np.mean(errors <= 0.02)),
        "within_0.05_s_per_lap": float(100.0 * np.mean(errors <= 0.05)),
    }


def train_baseline(train: pd.DataFrame, holdout: pd.DataFrame,
                   features: list[str], target: str = "deg_rate") -> dict:
    """Fit a LinearRegression baseline and report holdout metrics.

    Missing values are median-imputed from the training set only, so no
    holdout information reaches the fit.

    :param train: training stints.
    :param holdout: held-out stints.
    :param features: Tier A feature names.
    :param target: target column.
    :returns: mapping with the fitted model, metrics and coefficients.
    """
    feats = [f for f in features if f in train.columns]
    medians = train[feats].median()

    x_train = train[feats].fillna(medians)
    x_holdout = holdout[feats].fillna(medians)

    model = LinearRegression()
    model.fit(x_train, train[target])

    metrics = evaluate(holdout[target].to_numpy(),
                       model.predict(x_holdout))
    logger.info("Baseline holdout MAE %.5f s/lap, R2 %.3f",
                metrics["MAE"], metrics["R2"])

    coefficients = dict(zip(feats, (float(c) for c in model.coef_)))
    for name, value in sorted(coefficients.items(),
                              key=lambda kv: -abs(kv[1]))[:5]:
        logger.info("  %-20s %+.5f", name, value)

    return {"model": model, "metrics": metrics, "features": feats,
            "coefficients": coefficients, "medians": medians}


def mean_predictor(train: pd.DataFrame, holdout: pd.DataFrame,
                   target: str = "deg_rate") -> dict[str, float]:
    """Metrics for predicting the training mean, the floor any model must clear.

    :param train: training stints.
    :param holdout: held-out stints.
    :param target: target column.
    :returns: metric mapping.
    """
    prediction = np.full(len(holdout), float(train[target].mean()))
    metrics = evaluate(holdout[target].to_numpy(), prediction)
    logger.info("Mean-predictor holdout MAE %.5f s/lap", metrics["MAE"])
    return metrics
