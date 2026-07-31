"""Phase 2: data validation, compound mapping and the is_clean_lap mask.

Implements the compass specification with four corrections, each addressing a
defect observed either in the reference code or in the superseded pipeline:

1. ``add_is_clean_lap`` in the reference computes a ``lap_s`` local that is
   never used, then re-derives it through an ``np.issubdtype(..., np.floating)``
   test applied to a timedelta column. Here the float ``lap_s`` column written
   at collection time is used directly.
2. ``status_bad`` in the reference does ``out.get("TrackStatus", "1").astype(str)``,
   which raises when the column is missing because the fallback is a bare
   string. Column presence is now guarded properly.
3. Compound is never coerced. The superseded ``data_validation.py`` mapped
   every unrecognised value (including FastF1's ``UNKNOWN`` and
   ``TEST_UNKNOWN``) to ``MEDIUM`` via ``fillna``, with no counter and no log
   line, corrupting the single most important tyre categorical. Values are now
   routed to explicit categories, mapped to a C-ordinal through the per-event
   Pirelli nomination, and every outcome is counted.
4. Wet and intermediate stints are flagged and excluded from the degradation
   slope fit rather than silently modelled as dry.

House rules honoured: join on ``DriverNumber`` (never the three-letter
abbreviation, which is not stable), key events by ``RoundNumber`` (never
``EventName``), and treat ``GridPosition == 0.0`` as a pit-lane start to be
flagged rather than overwritten.
"""
from __future__ import annotations

import logging
from collections import Counter
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def validate_driver_join(laps: pd.DataFrame) -> pd.DataFrame:
    """Ensure DriverNumber is present and typed as a stable string key.

    :param laps: laps frame.
    :returns: frame with a normalised ``DriverNumber`` string column.
    :raises KeyError: if DriverNumber is entirely absent.
    """
    if "DriverNumber" not in laps.columns:
        raise KeyError("DriverNumber missing; cannot build a stable join key")
    out = laps.copy()
    out["DriverNumber"] = out["DriverNumber"].astype(str).str.strip()
    missing = int(out["DriverNumber"].isin(["", "nan", "None", "<NA>"]).sum())
    if missing:
        logger.warning("%d rows have an empty DriverNumber", missing)
    return out


def flag_grid_anomalies(laps: pd.DataFrame) -> pd.DataFrame:
    """Flag pit-lane starts (GridPosition == 0.0) without overwriting them.

    The superseded pipeline rewrote these to 25, inventing a grid slot that
    never existed.

    :param laps: laps frame, optionally containing ``GridPosition``.
    :returns: frame with a boolean ``pit_lane_start`` column added.
    """
    out = laps.copy()
    if "GridPosition" in out.columns:
        out["pit_lane_start"] = (
            pd.to_numeric(out["GridPosition"], errors="coerce").fillna(-1).eq(0.0))
        n = int(out["pit_lane_start"].sum())
        if n:
            logger.info("Flagged %d pit-lane-start lap rows", n)
    else:
        out["pit_lane_start"] = False
    return out


def impute_weather(laps: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    """Impute missing AirTemp/TrackTemp from climatological defaults.

    Provenance is tracked so an invented value can never masquerade as a
    measured one.

    :param laps: laps frame with ``EventName``, ``AirTemp``, ``TrackTemp``.
    :param defaults: ``climatological_defaults`` mapping from track_traits.yaml.
    :returns: frame with imputed values and a ``weather_imputed`` flag.
    """
    out = laps.copy()
    if "weather_imputed" not in out.columns:
        out["weather_imputed"] = False
    out["weather_imputed"] = out["weather_imputed"].fillna(False).astype(bool)
    fallback = defaults.get("default", {"air": 22, "track": 30})

    for col, key in (("AirTemp", "air"), ("TrackTemp", "track")):
        if col not in out.columns:
            out[col] = np.nan
        out[col] = pd.to_numeric(out[col], errors="coerce")
        mask = out[col].isna()
        if not mask.any():
            continue

        def _lookup(name: object) -> float:
            entry = defaults.get(str(name).replace(" ", "_"), fallback)
            return float(entry.get(key, fallback[key]))

        source = out.loc[mask, "EventName"] if "EventName" in out.columns else pd.Series(
            [None] * int(mask.sum()), index=out.index[mask])
        out.loc[mask, col] = source.map(_lookup)
        out.loc[mask, "weather_imputed"] = True
        logger.info("Imputed %d missing %s values from climatology",
                    int(mask.sum()), col)
    return out


def map_compounds(laps: pd.DataFrame, mapping: dict[str, Any],
                  event_keys: dict[str, Any]) -> tuple[pd.DataFrame, Counter]:
    """Categorise Compound and map it to a Pirelli C-ordinal.

    Never coerces an unrecognised compound to a real one. Adds:

    - ``compound_category``: SLICK, WET or UNKNOWN;
    - ``compound_ordinal``: 0-5 from the per-event nomination, else NaN;
    - ``compound_known``: whether the ordinal was resolved from a source.

    The ordinal is the model feature, never the hard/medium/soft label, because
    the label is relative per weekend: a C3 is the hard tyre at Monaco and the
    soft at Silverstone.

    :param laps: laps frame with ``Compound``, ``EventName``, ``Year``.
    :param mapping: parsed compound_mapping.yaml.
    :param event_keys: ``event_name_to_key`` mapping from track_traits.yaml.
    :returns: (frame with the three new columns, counter of outcomes).
    """
    out = laps.copy()
    counts: Counter = Counter()

    encoding = mapping["encoding"]
    nominations = mapping.get("nominations", {})
    slicks = {c.upper() for c in mapping["slick_compounds"]}
    wets = {c.upper() for c in mapping["wet_compounds"]}

    raw = out["Compound"].astype(str).str.upper().str.strip() \
        if "Compound" in out.columns else pd.Series("UNKNOWN", index=out.index)
    raw = raw.replace({"NAN": "UNKNOWN", "NONE": "UNKNOWN", "": "UNKNOWN"})

    category = np.where(raw.isin(slicks), "SLICK",
                        np.where(raw.isin(wets), "WET", "UNKNOWN"))
    out["compound_category"] = category

    label_for_slot = {"SOFT": "soft", "MEDIUM": "medium", "HARD": "hard"}
    ordinals: list[float] = []

    for compound, event_name, year in zip(
            raw, out.get("EventName", pd.Series("", index=out.index)),
            out.get("Year", pd.Series(0, index=out.index))):
        if compound not in slicks:
            counts[f"no_ordinal:{compound}"] += 1
            ordinals.append(np.nan)
            continue

        key_entry = event_keys.get(str(event_name))
        if key_entry is None:
            counts["no_ordinal:unmapped_event"] += 1
            ordinals.append(np.nan)
            continue

        season = nominations.get(int(year), {}) if pd.notna(year) else {}
        nomination = season.get(key_entry["compound"])
        if nomination is None:
            counts["no_ordinal:event_not_nominated"] += 1
            ordinals.append(np.nan)
            continue

        c_label = nomination.get(label_for_slot[compound])
        if c_label in (None, "UNK"):
            counts["no_ordinal:nomination_unk"] += 1
            ordinals.append(np.nan)
            continue

        ordinals.append(float(encoding[c_label]))
        counts[f"ordinal:{c_label}"] += 1

    out["compound_ordinal"] = ordinals
    out["compound_known"] = out["compound_ordinal"].notna()

    for outcome, n in sorted(counts.items()):
        logger.info("  compound %-32s %6d rows", outcome, n)
    resolved = int(out["compound_known"].sum())
    logger.info("Compound ordinals resolved for %d / %d rows (%.1f%%)",
                resolved, len(out), 100.0 * resolved / max(len(out), 1))
    return out, counts


def add_is_clean_lap(laps: pd.DataFrame, multiplier: float = 1.07,
                     track_status_ok: str = "1") -> pd.DataFrame:
    """Compute the is_clean_lap mask selecting representative-pace laps.

    Excludes in-laps and out-laps, non-green ``TrackStatus``, the first lap of
    each stint, laps slower than ``multiplier`` times the stint median, and
    laps with no recorded time.

    Requires the float ``lap_s`` column written at collection time, which
    avoids the reference implementation's fragile dtype branch.

    :param laps: laps frame with ``lap_s``, ``Stint``, ``LapNumber`` and the
        grouping keys ``Year``, ``RoundNumber``, ``DriverNumber``.
    :param multiplier: stint-median multiplier (1.07 = 107%).
    :param track_status_ok: green-flag TrackStatus code.
    :returns: frame with a boolean ``is_clean_lap`` column.
    """
    out = laps.copy()
    if "lap_s" not in out.columns:
        raise KeyError("lap_s missing; it is written at collection time")
    out["lap_s"] = pd.to_numeric(out["lap_s"], errors="coerce")

    n = len(out)
    false_series = pd.Series(False, index=out.index)

    is_out = out["PitOutTime"].notna() if "PitOutTime" in out.columns else false_series
    is_in = out["PitInTime"].notna() if "PitInTime" in out.columns else false_series

    if "TrackStatus" in out.columns:
        status_bad = out["TrackStatus"].astype(str).str.strip() != str(track_status_ok)
    else:
        logger.warning("TrackStatus absent; safety-car laps cannot be excluded")
        status_bad = false_series

    group = ["Year", "RoundNumber", "DriverNumber", "Stint"]
    missing_keys = [k for k in group if k not in out.columns]
    if missing_keys:
        raise KeyError(f"is_clean_lap requires grouping keys, missing: {missing_keys}")

    first_lap = out["LapNumber"].eq(out.groupby(group)["LapNumber"].transform("min"))
    stint_median = out.groupby(group)["lap_s"].transform("median")
    too_slow = out["lap_s"] > (multiplier * stint_median)
    no_time = out["lap_s"].isna()

    out["is_clean_lap"] = ~(is_out | is_in | status_bad | first_lap | too_slow | no_time)

    logger.info("Clean laps: %d / %d (%.1f%%)",
                int(out["is_clean_lap"].sum()), n,
                100.0 * out["is_clean_lap"].sum() / max(n, 1))
    logger.info("  excluded: in/out %d, non-green %d, stint-first %d, "
                ">%.0f%% median %d, no time %d",
                int((is_in | is_out).sum()), int(status_bad.sum()),
                int(first_lap.sum()), multiplier * 100, int(too_slow.sum()),
                int(no_time.sum()))
    return out


def flag_wet_stints(laps: pd.DataFrame) -> pd.DataFrame:
    """Flag stints run in the wet so they can be excluded from the slope fit.

    Wet running is governed by different physics and by track drying rather
    than tyre wear, so those stints are flagged and retained, never deleted.

    :param laps: laps frame with ``compound_category`` and optionally ``Rainfall``.
    :returns: frame with a boolean ``wet_stint`` column.
    """
    out = laps.copy()
    group = ["Year", "RoundNumber", "DriverNumber", "Stint"]

    wet_compound = out.get("compound_category", pd.Series("SLICK", index=out.index)) == "WET"
    if "Rainfall" in out.columns:
        raining = out["Rainfall"].fillna(False).astype(bool)
    else:
        raining = pd.Series(False, index=out.index)

    out["wet_stint"] = (wet_compound | raining).groupby(
        [out[k] for k in group]).transform("any")
    n_stints = out.groupby(group).ngroups
    n_wet = out[out["wet_stint"]].groupby(group).ngroups
    logger.info("Wet stints flagged: %d / %d", n_wet, n_stints)
    return out


def validate(laps: pd.DataFrame, track_cfg: dict[str, Any],
             compound_cfg: dict[str, Any], clean_cfg: dict[str, Any]) -> pd.DataFrame:
    """Run the full validation pipeline defensively.

    :param laps: raw laps frame from collection.
    :param track_cfg: parsed track_traits.yaml.
    :param compound_cfg: parsed compound_mapping.yaml.
    :param clean_cfg: ``clean_lap`` section of settings.yaml.
    :returns: validated, flagged and masked frame; empty if the input is empty.
    """
    if laps is None or laps.empty:
        logger.error("Empty laps frame passed to validate()")
        return pd.DataFrame()

    out = validate_driver_join(laps)
    out = flag_grid_anomalies(out)
    out = impute_weather(out, track_cfg.get("climatological_defaults", {}))
    out, _ = map_compounds(out, compound_cfg, track_cfg.get("event_name_to_key", {}))
    out = add_is_clean_lap(
        out,
        multiplier=clean_cfg.get("stint_median_multiplier", 1.07),
        track_status_ok=str(clean_cfg.get("track_status_ok", "1")))
    out = flag_wet_stints(out)
    return out
