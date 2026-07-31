"""Grouped chronological splitting for event-level time-series validation.

Standard k-fold leaks future information when observations are serially
dependent (bergmeir_2018), so validation windows must always sit temporally
behind training windows.

``TimeSeriesSplit`` alone is not sufficient here. Applied to rows, it cuts at
arbitrary row indices, so a single race's stints land on both sides of a fold
boundary and drivers who shared the same track, weather and safety-car periods
leak across the split. The compass reference code has exactly this defect: it
sorts by ``(Year, RoundNumber)`` and then hands the row array straight to
``TimeSeriesSplit``.

Splitting on the unique ordered ``(Year, RoundNumber)`` keys and mapping back
to rows guarantees that an event is wholly on one side of every boundary.
"""
from __future__ import annotations

import logging
from typing import Iterator

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

EVENT_KEYS = ["Year", "RoundNumber"]


def event_order(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Return the unique ``(Year, RoundNumber)`` keys in chronological order."""
    missing = [k for k in EVENT_KEYS if k not in df.columns]
    if missing:
        raise KeyError(f"chronological splitting requires columns: {missing}")
    pairs = df[EVENT_KEYS].drop_duplicates()
    pairs = pairs.sort_values(EVENT_KEYS)
    return [(int(y), int(r)) for y, r in pairs.itertuples(index=False)]


def event_series(df: pd.DataFrame) -> pd.Series:
    """Return a per-row ``(Year, RoundNumber)`` tuple series."""
    return pd.Series(
        list(zip(df["Year"].astype(int), df["RoundNumber"].astype(int))),
        index=df.index)


def grouped_time_series_split(df: pd.DataFrame, n_splits: int = 4
                              ) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield positional train/validation indices, split on whole events.

    Expanding-window folds: each validation block is the chronologically next
    slice of events, and training is everything before it.

    :param df: frame with ``Year`` and ``RoundNumber``.
    :param n_splits: number of folds.
    :yields: (train positions, validation positions) as integer arrays.
    :raises ValueError: if there are too few events for the requested folds.
    """
    events = event_order(df)
    if len(events) < n_splits + 1:
        raise ValueError(
            f"{len(events)} events cannot support {n_splits} folds; "
            "collect more seasons or reduce n_splits")

    rows = event_series(df)
    positions = {event: np.flatnonzero((rows == event).to_numpy())
                 for event in events}

    # Leave the first block for training, then split the rest into n_splits.
    fold_edges = np.array_split(np.arange(len(events)), n_splits + 1)
    for fold in range(1, n_splits + 1):
        train_events = [events[i] for i in np.concatenate(fold_edges[:fold])]
        val_events = [events[i] for i in fold_edges[fold]]

        train_idx = np.concatenate([positions[e] for e in train_events])
        val_idx = np.concatenate([positions[e] for e in val_events])

        overlap = set(train_events) & set(val_events)
        assert not overlap, f"event straddles a fold boundary: {overlap}"

        yield np.sort(train_idx), np.sort(val_idx)


def season_holdout(df: pd.DataFrame, train_seasons: list[int],
                   holdout_season: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split into training seasons and a held-out final season.

    :param df: stint table.
    :param train_seasons: seasons used for training.
    :param holdout_season: season held out entirely.
    :returns: (train frame, holdout frame).
    """
    available = sorted(df["Year"].unique().tolist())
    usable_train = [s for s in train_seasons if s in available]

    train = df[df["Year"].isin(usable_train)].copy()
    holdout = df[df["Year"] == holdout_season].copy()

    logger.info("Chronological split: train %s (%d stints), holdout %d (%d stints)",
                usable_train, len(train), holdout_season, len(holdout))
    if not usable_train:
        raise ValueError(
            f"None of the training seasons {train_seasons} are present in {available}")
    if holdout.empty:
        raise ValueError(
            f"Holdout season {holdout_season} is absent from {available}")
    if len(usable_train) < len(train_seasons):
        logger.warning("Training on %s only; %s not yet collected. Cross-"
                       "validation folds are thinner than intended.",
                       usable_train,
                       sorted(set(train_seasons) - set(usable_train)))
    return train, holdout


def assert_no_event_leakage(df: pd.DataFrame, n_splits: int = 4) -> None:
    """Verify no event appears on both sides of any fold boundary.

    :raises AssertionError: if any event straddles a split.
    """
    rows = event_series(df)
    for fold, (train_idx, val_idx) in enumerate(
            grouped_time_series_split(df, n_splits), start=1):
        train_events = set(rows.iloc[train_idx])
        val_events = set(rows.iloc[val_idx])
        shared = train_events & val_events
        assert not shared, f"fold {fold}: events on both sides: {shared}"
    logger.info("Verified: no event straddles any of the %d fold boundaries",
                n_splits)
