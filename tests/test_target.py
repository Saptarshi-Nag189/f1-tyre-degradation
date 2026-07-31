"""Tests for src/features/target.py."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features import target


# --- fuel correction ---------------------------------------------------------

def test_fuel_correction_formula():
    """Fuel burned by lap N is (N-1) * burn_per_lap, because the car completes
    lap N carrying the fuel it had at the start of that lap."""
    laps = pd.DataFrame({
        "LapNumber": [1, 2],
        "lap_s": [100.0, 100.0],
        "RaceLaps": [50, 50],
    })
    out = target.fuel_correct(laps, penalty=0.032, initial_fuel=110.0)

    # Lap 1 carries the full 110 kg -> correction 0.032 * 110 = 3.52 s
    assert out["corrected_lap_s"].iloc[0] == pytest.approx(100.0 - 3.52)
    # Lap 2 carries 110 - 1 * (110/50) = 107.8 kg
    assert out["corrected_lap_s"].iloc[1] == pytest.approx(100.0 - 0.032 * 107.8)


def test_fuel_correction_makes_early_laps_relatively_slower():
    """A constant raw lap time means the car was actually getting slower as
    fuel burned off, so corrected times must increase through the stint."""
    laps = pd.DataFrame({
        "LapNumber": range(1, 11),
        "lap_s": [90.0] * 10,
        "RaceLaps": [50] * 10,
    })
    out = target.fuel_correct(laps, penalty=0.032, initial_fuel=110.0)
    assert out["corrected_lap_s"].is_monotonic_increasing


def test_fuel_correction_requires_race_length():
    laps = pd.DataFrame({"LapNumber": [1], "lap_s": [90.0]})
    with pytest.raises(KeyError, match="RaceLaps"):
        target.fuel_correct(laps, penalty=0.032, initial_fuel=110.0)


# --- stint degradation -------------------------------------------------------

def test_recovers_known_degradation_rate(degrading_stint):
    result = target.stint_degradation(degrading_stint, min_r=0.3, min_laps=5)
    assert result is not None
    assert result.deg_rate == pytest.approx(0.05, abs=1e-6)
    assert result.r_value == pytest.approx(1.0, abs=1e-6)
    assert result.n_laps == 12


def test_noise_stint_rejected_by_min_r():
    n = 12
    rng = np.random.RandomState(0)
    stint = pd.DataFrame({
        "is_clean_lap": [True] * n,
        "TyreLife": np.arange(n),
        "corrected_lap_s": rng.normal(90, 5, n),
    })
    assert target.stint_degradation(stint, min_r=0.9, min_laps=5) is None


def test_short_stint_rejected(degrading_stint):
    assert target.stint_degradation(
        degrading_stint.head(3), min_r=0.3, min_laps=5) is None


# --- acceptance by precision, not by correlation strength -------------------

def _flat_stint(noise: float, n: int = 14, seed: int = 0) -> pd.DataFrame:
    """A stint with no real degradation, plus lap-to-lap noise."""
    rng = np.random.RandomState(seed)
    return pd.DataFrame({
        "is_clean_lap": [True] * n,
        "TyreLife": np.arange(1, n + 1),
        "corrected_lap_s": 90.0 + rng.normal(0, noise, n),
    })


def test_flat_but_precise_stint_is_kept():
    """A genuinely flat stint has near-zero |r| however well it is measured.
    Rejecting it discards exactly the low-degradation circuits a strategist
    most needs, which is what removed Monaco from the dataset."""
    stint = _flat_stint(noise=0.02)
    assert target.stint_degradation(stint, min_r=0.0, min_laps=5,
                                    max_stderr=0.02) is not None


def test_correlation_filter_discards_flat_stints_the_precision_one_keeps():
    """Documents the defect the precision criterion replaces.

    The claim is statistical rather than about any single stint: across many
    well-measured flat stints, an |r| threshold rejects most of them while a
    precision threshold keeps them. Measured on the real data, the |r| >= 0.3
    pass rate runs from 29.5% for near-zero slopes to 98.7% for large ones.
    """
    kept_by_r = kept_by_precision = 0
    trials = 40
    for seed in range(trials):
        stint = _flat_stint(noise=0.02, seed=seed)
        if target.stint_degradation(stint, min_r=0.3, min_laps=5) is not None:
            kept_by_r += 1
        if target.stint_degradation(stint, min_r=0.0, min_laps=5,
                                    max_stderr=0.02) is not None:
            kept_by_precision += 1

    assert kept_by_precision == trials, "precision should keep every flat stint"
    assert kept_by_r < trials * 0.6, (
        f"|r| kept {kept_by_r}/{trials} flat stints; it is expected to discard "
        "most of them, which is the bias being removed")


def test_noisy_stint_is_rejected_on_precision():
    """Large lap-to-lap scatter makes the slope unusable regardless of |r|."""
    stint = _flat_stint(noise=3.0)
    assert target.stint_degradation(stint, min_r=0.0, min_laps=5,
                                    max_stderr=0.02) is None


def test_stderr_is_reported(degrading_stint):
    result = target.stint_degradation(degrading_stint, min_r=0.0, min_laps=5)
    assert result is not None
    assert result.stderr >= 0.0


def test_constant_tyre_life_rejected():
    n = 10
    stint = pd.DataFrame({
        "is_clean_lap": [True] * n,
        "TyreLife": [5] * n,
        "corrected_lap_s": np.linspace(90, 91, n),
    })
    assert target.stint_degradation(stint, min_r=0.3, min_laps=5) is None


# --- telemetry-free energy proxy --------------------------------------------

def test_speed_trap_energy_is_mean_v_squared(degrading_stint):
    """The proxy must be mean(v^2) in m^2/s^2, matching the compass aero-load
    proxy's units, so it can stand in before telemetry is collected."""
    energy = target.speed_trap_energy(degrading_stint)
    speeds_ms = np.array([200.0, 250.0, 270.0, 280.0]) / 3.6
    assert energy == pytest.approx(float(np.mean(speeds_ms ** 2)))


def test_speed_trap_energy_absent_returns_nan():
    stint = pd.DataFrame({"TyreLife": [1, 2, 3]})
    assert np.isnan(target.speed_trap_energy(stint))


# --- CWI decision rule -------------------------------------------------------

def _stint_table(energy: np.ndarray, deg: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame({"deg_rate": deg, "energy_proxy": energy})


def test_uncorrelated_energy_is_demoted():
    rng = np.random.RandomState(1)
    stints = _stint_table(rng.normal(0, 1, 40), np.linspace(0.01, 0.1, 40))

    out, meta = target.validate_and_build_cwi(
        stints, keep_thr=0.4, downweight_low=0.2,
        energy_w_full=0.5, laptime_w_down=0.7)

    assert meta["decision"] == "energy_demoted_to_feature"
    assert meta["energy_weight"] == 0.0
    # With energy demoted, CWI is exactly the z-scored degradation slope.
    z_deg = (stints["deg_rate"] - stints["deg_rate"].mean()) / stints["deg_rate"].std(ddof=0)
    assert np.allclose(out["CWI"], z_deg)


def test_strongly_correlated_energy_is_kept():
    deg = np.linspace(0.01, 0.1, 40)
    stints = _stint_table(deg * 1000 + 5, deg)   # monotone, so rho = 1

    _, meta = target.validate_and_build_cwi(
        stints, keep_thr=0.4, downweight_low=0.2,
        energy_w_full=0.5, laptime_w_down=0.7)

    assert meta["decision"] == "energy_in_target"
    assert meta["energy_weight"] == 0.5
    assert meta["laptime_weight"] == 0.5


def test_weights_always_sum_to_one():
    for energy, deg in (
        (np.linspace(1, 40, 40), np.linspace(0.01, 0.1, 40)),
        (np.random.RandomState(2).normal(0, 1, 40), np.linspace(0.01, 0.1, 40)),
    ):
        _, meta = target.validate_and_build_cwi(
            _stint_table(energy, deg), keep_thr=0.4, downweight_low=0.2,
            energy_w_full=0.5, laptime_w_down=0.7)
        assert meta["energy_weight"] + meta["laptime_weight"] == pytest.approx(1.0)


def test_empty_table_does_not_raise():
    out, meta = target.validate_and_build_cwi(
        pd.DataFrame(columns=["deg_rate", "energy_proxy"]),
        keep_thr=0.4, downweight_low=0.2,
        energy_w_full=0.5, laptime_w_down=0.7)
    assert meta["decision"] == "no_data"
    assert out.empty
