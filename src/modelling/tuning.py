"""Phase 6: Optuna hyperparameter search nested in the grouped chronological CV.

Search space is deliberately weighted towards heavy regularisation. The signal
here is weak (only ~32% of degradation variance lies between events, and the
oracle ceiling on MAE improvement is ~27%), and the untuned default of 400
trees at depth 4 demonstrably overfits: it scores worse on the holdout than
predicting the training mean while the linear baseline does not.

Runs on CPU. 50 trials over ~2,000 rows takes seconds, and ``device='cuda'``
on a matrix this size is slower than ``tree_method='hist'`` on CPU because
PCIe transfer dominates the computation (chen_guestrin_2016, grinsztajn_2022).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from src.modelling import splits

logger = logging.getLogger(__name__)


def tune(train: pd.DataFrame, features: list[str], target: str = "deg_rate",
         n_trials: int = 50, n_splits: int = 4, seed: int = 42) -> dict:
    """Search XGBoost hyperparameters under grouped chronological CV.

    :param train: training stints, carrying ``Year`` and ``RoundNumber``.
    :param features: feature names.
    :param target: target column.
    :param n_trials: Optuna trials.
    :param n_splits: expanding-window folds.
    :param seed: sampler seed, for reproducibility.
    :returns: mapping with the best parameters and the best CV MAE.
    """
    import optuna
    from xgboost import XGBRegressor

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    ordered = train.sort_values(splits.EVENT_KEYS).reset_index(drop=True)
    x = ordered[features].to_numpy(dtype=float)
    y = ordered[target].to_numpy(dtype=float)
    folds = list(splits.grouped_time_series_split(ordered, n_splits))

    def objective(trial: "optuna.Trial") -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 600),
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.005, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            # Regularisation ranges run high deliberately: the signal is weak
            # and the default configuration overfits it.
            "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 60),
            "gamma": trial.suggest_float("gamma", 1e-4, 1.0, log=True),
            "tree_method": "hist",
            "random_state": seed,
        }
        fold_maes = []
        for train_idx, val_idx in folds:
            model = XGBRegressor(**params)
            model.fit(x[train_idx], y[train_idx])
            fold_maes.append(
                float(np.mean(np.abs(y[val_idx] - model.predict(x[val_idx])))))
        return float(np.mean(fold_maes))

    study = optuna.create_study(
        direction="minimize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    logger.info("Optuna best CV MAE %.5f over %d trials",
                study.best_value, n_trials)
    best = {**study.best_params, "tree_method": "hist", "random_state": seed}
    return {"best_params": best, "best_cv_mae": float(study.best_value),
            "n_trials": n_trials}
