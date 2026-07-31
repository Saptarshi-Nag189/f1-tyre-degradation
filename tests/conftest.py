"""Shared pytest fixtures and path setup."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture
def toy_laps() -> pd.DataFrame:
    """A minimal laps frame exercising every is_clean_lap branch.

    Five laps of one stint: lap 1 is the stint's first lap, lap 4 is far slower
    than the stint median, lap 5 runs under a non-green track status. Only laps
    2 and 3 should survive the mask.
    """
    return pd.DataFrame({
        "Year": [2024] * 5,
        "RoundNumber": [1] * 5,
        "DriverNumber": ["44"] * 5,
        "Stint": [1] * 5,
        "LapNumber": [1, 2, 3, 4, 5],
        "lap_s": [90.0, 91.0, 92.0, 200.0, 91.0],
        "TyreLife": [1, 2, 3, 4, 5],
        "TrackStatus": ["1", "1", "1", "1", "4"],
        "PitInTime": [pd.NaT] * 5,
        "PitOutTime": [pd.NaT] * 5,
        "GridPosition": [0.0, 3.0, 3.0, 3.0, 3.0],
        "EventName": ["Bahrain Grand Prix"] * 5,
        "Compound": ["SOFT"] * 5,
        "AirTemp": [27.0] * 5,
        "TrackTemp": [33.0] * 5,
        "RaceLaps": [57] * 5,
    })


@pytest.fixture
def degrading_stint() -> pd.DataFrame:
    """A clean 12-lap stint that degrades at a known 0.05 s per lap."""
    n = 12
    return pd.DataFrame({
        "Year": [2024] * n,
        "RoundNumber": [1] * n,
        "DriverNumber": ["44"] * n,
        "Stint": [1] * n,
        "LapNumber": range(1, n + 1),
        "TyreLife": range(1, n + 1),
        "is_clean_lap": [True] * n,
        "corrected_lap_s": [90.0 + 0.05 * i for i in range(n)],
        "SpeedI1": [200.0] * n,
        "SpeedI2": [250.0] * n,
        "SpeedFL": [270.0] * n,
        "SpeedST": [280.0] * n,
    })
