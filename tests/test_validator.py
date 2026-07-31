"""Tests for src/acquisition/validator.py."""
from __future__ import annotations

import pandas as pd
import pytest

from src import config
from src.acquisition import validator


# --- is_clean_lap mask -------------------------------------------------------

def test_first_lap_of_stint_excluded(toy_laps):
    out = validator.add_is_clean_lap(toy_laps)
    assert not out.loc[out["LapNumber"] == 1, "is_clean_lap"].iloc[0]


def test_slow_lap_excluded(toy_laps):
    out = validator.add_is_clean_lap(toy_laps)
    assert not out.loc[out["LapNumber"] == 4, "is_clean_lap"].iloc[0]


def test_non_green_track_status_excluded(toy_laps):
    out = validator.add_is_clean_lap(toy_laps)
    assert not out.loc[out["LapNumber"] == 5, "is_clean_lap"].iloc[0]


def test_representative_laps_kept(toy_laps):
    out = validator.add_is_clean_lap(toy_laps)
    assert out.loc[out["LapNumber"].isin([2, 3]), "is_clean_lap"].all()


def test_missing_track_status_does_not_raise(toy_laps):
    """The reference implementation raises here: out.get(col, "1").astype(str)
    returns a bare string when the column is absent."""
    out = validator.add_is_clean_lap(toy_laps.drop(columns=["TrackStatus"]))
    assert "is_clean_lap" in out.columns


def test_missing_lap_s_raises_clearly(toy_laps):
    with pytest.raises(KeyError, match="lap_s"):
        validator.add_is_clean_lap(toy_laps.drop(columns=["lap_s"]))


# --- grid position -----------------------------------------------------------

def test_grid_zero_flagged_not_overwritten(toy_laps):
    """GridPosition 0.0 means a pit-lane start. The superseded pipeline rewrote
    it to 25, inventing a grid slot that never existed."""
    out = validator.flag_grid_anomalies(toy_laps)
    assert bool(out["pit_lane_start"].iloc[0])
    assert out["GridPosition"].iloc[0] == 0.0
    assert not out["pit_lane_start"].iloc[1:].any()


# --- driver join -------------------------------------------------------------

def test_missing_driver_number_raises():
    with pytest.raises(KeyError):
        validator.validate_driver_join(pd.DataFrame({"A": [1]}))


def test_driver_number_normalised_to_string():
    out = validator.validate_driver_join(pd.DataFrame({"DriverNumber": [44, 1]}))
    assert out["DriverNumber"].tolist() == ["44", "1"]


# --- compound mapping --------------------------------------------------------

def _compound_cfgs():
    return config.compound_mapping(), config.track_traits()["event_name_to_key"]


def test_unknown_compound_never_becomes_medium(toy_laps):
    """The single most damaging silent corruption in the superseded pipeline:
    data_validation.py mapped UNKNOWN and TEST_UNKNOWN to MEDIUM via fillna,
    with no counter and no log line."""
    mapping, event_keys = _compound_cfgs()
    laps = toy_laps.copy()
    laps["Compound"] = ["UNKNOWN", "TEST_UNKNOWN", "SOFT", "MEDIUM", "HARD"]
    laps["Year"] = 2024

    out, counts = validator.map_compounds(laps, mapping, event_keys)

    assert (out["compound_category"].iloc[:2] == "UNKNOWN").all()
    assert out["compound_ordinal"].iloc[:2].isna().all()
    assert not out["compound_known"].iloc[:2].any()
    assert counts["no_ordinal:UNKNOWN"] == 1
    assert counts["no_ordinal:TEST_UNKNOWN"] == 1


def test_compound_ordinal_uses_per_event_nomination(toy_laps):
    """Bahrain 2024 is nominated C1/C2/C3, so SOFT is C3 and HARD is C1.
    The label is relative per weekend; only the ordinal is comparable."""
    mapping, event_keys = _compound_cfgs()
    laps = toy_laps.copy()
    laps["Year"] = 2024
    laps["Compound"] = ["SOFT", "MEDIUM", "HARD", "SOFT", "HARD"]

    out, _ = validator.map_compounds(laps, mapping, event_keys)

    assert out["compound_ordinal"].tolist() == [3.0, 2.0, 1.0, 3.0, 1.0]


def test_unk_nomination_yields_no_ordinal(toy_laps):
    """Entries marked UNK must stay unresolved, never be guessed."""
    mapping, event_keys = _compound_cfgs()
    laps = toy_laps.copy()
    laps["Year"] = 2022
    laps["EventName"] = "Italian Grand Prix"   # 2022 Italy is UNK/UNK/UNK

    out, counts = validator.map_compounds(laps, mapping, event_keys)

    assert out["compound_ordinal"].isna().all()
    assert counts["no_ordinal:nomination_unk"] == len(laps)


def test_wet_compound_categorised_not_ordinalised(toy_laps):
    mapping, event_keys = _compound_cfgs()
    laps = toy_laps.copy()
    laps["Compound"] = ["WET", "INTERMEDIATE", "SOFT", "SOFT", "SOFT"]
    laps["Year"] = 2024

    out, _ = validator.map_compounds(laps, mapping, event_keys)

    assert out["compound_category"].tolist()[:2] == ["WET", "WET"]
    assert out["compound_ordinal"].iloc[:2].isna().all()


# --- weather imputation ------------------------------------------------------

def test_imputed_weather_is_flagged(toy_laps):
    defaults = config.track_traits()["climatological_defaults"]
    laps = toy_laps.copy()
    laps.loc[0, "AirTemp"] = None

    out = validator.impute_weather(laps, defaults)

    assert bool(out["weather_imputed"].iloc[0])
    assert not out["weather_imputed"].iloc[1:].any()
    assert out["AirTemp"].notna().all()


def test_measured_weather_not_flagged(toy_laps):
    defaults = config.track_traits()["climatological_defaults"]
    out = validator.impute_weather(toy_laps, defaults)
    assert not out["weather_imputed"].any()


# --- wet stint flagging ------------------------------------------------------

def test_rainfall_flags_whole_stint(toy_laps):
    laps = toy_laps.copy()
    laps["compound_category"] = "SLICK"
    laps["Rainfall"] = [False, False, True, False, False]

    out = validator.flag_wet_stints(laps)

    assert out["wet_stint"].all(), "one wet lap should flag the entire stint"
