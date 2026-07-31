"""Tests for src/modelling/splits.py and the feature contract.

The leakage guarantee is the point of this module: the compass reference
``_cv_mae`` sorts by ``(Year, RoundNumber)`` and hands the row array to
``TimeSeriesSplit``, which cuts at arbitrary row indices, so one race's stints
land on both sides of a fold boundary.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.modelling import features, splits


@pytest.fixture
def stints() -> pd.DataFrame:
    """Two seasons of 10 events, 8 stints each."""
    rows = []
    for year in (2023, 2024):
        for round_number in range(1, 11):
            for stint in range(8):
                rows.append({"Year": year, "RoundNumber": round_number,
                             "Stint": stint, "deg_rate": 0.05 + 0.001 * stint})
    return pd.DataFrame(rows)


def test_no_event_straddles_a_fold_boundary(stints):
    splits.assert_no_event_leakage(stints, n_splits=4)


def test_validation_is_always_chronologically_after_training(stints):
    rows = splits.event_series(stints)
    for train_idx, val_idx in splits.grouped_time_series_split(stints, 4):
        latest_train = max(rows.iloc[train_idx])
        earliest_val = min(rows.iloc[val_idx])
        assert latest_train < earliest_val, "future information leaked backwards"


def test_folds_are_disjoint_and_expanding(stints):
    previous_train = 0
    for train_idx, val_idx in splits.grouped_time_series_split(stints, 4):
        assert not set(train_idx) & set(val_idx)
        assert len(train_idx) > previous_train, "training window should expand"
        previous_train = len(train_idx)


def test_too_few_events_raises(stints):
    tiny = stints[stints["RoundNumber"] <= 2]
    with pytest.raises(ValueError, match="cannot support"):
        list(splits.grouped_time_series_split(tiny, n_splits=4))


def test_season_holdout_separates_years(stints):
    train, holdout = splits.season_holdout(stints, [2023], 2024)
    assert set(train["Year"].unique()) == {2023}
    assert set(holdout["Year"].unique()) == {2024}
    assert len(train) + len(holdout) == len(stints)


def test_missing_holdout_season_raises(stints):
    with pytest.raises(ValueError, match="Holdout season"):
        splits.season_holdout(stints, [2023], 2029)


# --- feature contract -------------------------------------------------------

def test_no_forbidden_column_is_a_tier_a_feature():
    """StintLength and n_laps are reverse-causal, and the target's own
    regression outputs must never be fed back in."""
    assert not set(features.TIER_A) & set(features.FORBIDDEN)


def test_target_columns_are_forbidden():
    for column in ("deg_rate", "CWI", "StintLength", "n_laps", "energy_proxy"):
        assert column in features.FORBIDDEN


def test_categorical_levels_come_from_training_only():
    train = pd.DataFrame({"Team": ["Ferrari", "McLaren"], "x": [1.0, 2.0]})
    holdout = pd.DataFrame({"Team": ["Ferrari", "Unseen Team"], "x": [3.0, 4.0]})
    categories = features.training_categories(train)

    _, train_cols = features.build_matrix(train, ["x"], categories)
    holdout_matrix, holdout_cols = features.build_matrix(holdout, ["x"], categories)

    assert train_cols == holdout_cols, "encoding must be identical across splits"
    assert not any("Unseen" in c for c in holdout_cols)
    # The unseen team encodes as all-zeros rather than creating a new column.
    assert holdout_matrix.iloc[1][[c for c in holdout_cols if c.startswith("Team_")]].sum() == 0
