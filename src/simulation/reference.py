"""Empirical per-circuit and per-compound reference table.

The trained model predicts a degradation slope from circuit traits, compound
and temperature. For circuits the model has actually seen, the observed slope
is the stronger estimate: per-circuit degradation now correlates 0.66 to 0.71
across seasons, whereas the model captures only 19% of the achievable headroom.

This module therefore builds a blended reference:

- where enough observed stints exist, the empirical median is used and the
  model prediction is recorded alongside it for comparison;
- where they do not, the model prediction fills the gap, flagged as such.

Every entry carries its sample size and interquartile spread, so the strategy
layer can report an honest confidence rather than a decorative number.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

#: Minimum observed stints before the empirical estimate is preferred.
MIN_STINTS_FOR_EMPIRICAL = 6


def base_pace_by_event(laps: pd.DataFrame) -> pd.DataFrame:
    """Representative clean-lap pace per event.

    Uses the 5th percentile of clean fuel-corrected lap times, which
    approximates a low-fuel representative lap without being set by a single
    outlier.

    :param laps: validated laps frame with ``is_clean_lap`` and ``lap_s``.
    :returns: frame indexed by (Year, EventName) with pace columns.
    """
    clean = laps[laps["is_clean_lap"]]
    grouped = clean.groupby(["Year", "EventName"])
    out = grouped.agg(
        base_lap_s=("lap_s", lambda s: float(np.nanpercentile(s, 5))),
        median_lap_s=("lap_s", "median"),
        race_laps=("RaceLaps", "max"),
        n_clean_laps=("lap_s", "size"),
    ).reset_index()
    return out


def compound_reference(stints: pd.DataFrame,
                       min_stints: int = MIN_STINTS_FOR_EMPIRICAL) -> pd.DataFrame:
    """Observed degradation per (EventName, compound), pooled across seasons.

    Pooling across seasons is deliberate: per-circuit degradation is reasonably
    stable year to year once the target is cleaned, and pooling triples the
    sample behind each estimate.

    :param stints: stint table with ``deg_rate`` and ``Compound``.
    :param min_stints: minimum observations to treat an estimate as reliable.
    :returns: one row per (EventName, Compound).
    """
    grouped = stints.groupby(["EventName", "Compound"])
    ref = grouped.agg(
        deg_rate_median=("deg_rate", "median"),
        deg_rate_mean=("deg_rate", "mean"),
        deg_rate_q25=("deg_rate", lambda s: float(s.quantile(0.25))),
        deg_rate_q75=("deg_rate", lambda s: float(s.quantile(0.75))),
        n_stints=("deg_rate", "size"),
        mean_stint_length=("StintLength", "mean"),
        seasons=("Year", lambda s: sorted(set(int(v) for v in s))),
    ).reset_index()
    ref["reliable"] = ref["n_stints"] >= min_stints

    logger.info("Compound reference: %d (event, compound) pairs, %d reliable "
                "with >= %d stints", len(ref), int(ref["reliable"].sum()), min_stints)
    return ref


def build_reference(laps: pd.DataFrame, stints: pd.DataFrame) -> dict:
    """Assemble the full reference used by the strategy layer and the UI export.

    :param laps: validated laps frame.
    :param stints: stint table carrying the target.
    :returns: mapping with pace, compound reference and provenance metadata.
    """
    pace = base_pace_by_event(laps)
    # Latest season's pace per event, since cars get faster year on year.
    pace = pace.sort_values("Year").groupby("EventName", as_index=False).last()
    ref = compound_reference(stints)

    return {
        "pace": pace,
        "compounds": ref,
        "seasons": sorted(int(y) for y in stints["Year"].unique()),
        "n_stints": int(len(stints)),
    }
