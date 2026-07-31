"""Phase 3: rolling and lag features over tyre life within a stint.

All rolling and lag operations are grouped by
``(Year, RoundNumber, DriverNumber, Stint)`` and ordered by ``TyreLife``, so no
value can leak across a stint, a driver or an event boundary. Rolling windows
are trailing only.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_GROUP = ["Year", "RoundNumber", "DriverNumber", "Stint"]


def _sorted_by_tyre_life(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy ordered by stint then tyre life, preserving the index."""
    missing = [k for k in _GROUP + ["TyreLife"] if k not in df.columns]
    if missing:
        raise KeyError(f"timeseries features require columns: {missing}")
    return df.sort_values(_GROUP + ["TyreLife"]).copy()


def add_lag_features(df: pd.DataFrame, col: str,
                     lags: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Add lagged versions of a column within each stint.

    :param df: feature frame.
    :param col: column to lag.
    :param lags: lag steps in laps.
    :returns: frame with new ``{col}_lag{n}`` columns.
    """
    out = _sorted_by_tyre_life(df)
    for lag in lags:
        out[f"{col}_lag{lag}"] = out.groupby(_GROUP, observed=True)[col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, col: str,
                         windows: tuple[int, ...] = (3, 5)) -> pd.DataFrame:
    """Add trailing rolling means and standard deviations within each stint.

    :param df: feature frame.
    :param col: column to roll.
    :param windows: rolling window sizes in laps.
    :returns: frame with new ``{col}_rollmean{n}`` and ``{col}_rollstd{n}`` columns.
    """
    out = _sorted_by_tyre_life(df)
    for win in windows:
        grouped = out.groupby(_GROUP, observed=True)[col]
        out[f"{col}_rollmean{win}"] = grouped.transform(
            lambda s: s.rolling(win, min_periods=1).mean())
        out[f"{col}_rollstd{win}"] = grouped.transform(
            lambda s: s.rolling(win, min_periods=1).std())
    return out
