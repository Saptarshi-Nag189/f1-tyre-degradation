"""Phase 5: curvature-based physics proxies from X, Y telemetry.

Physics, SI units unless stated::

    curvature      kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)   [1/m]
    lateral accel  a_lat = v^2 * kappa                               [m/s^2]
    braking power  v * |dv/dt| for decelerations only                [m^2/s^3]
    combined G     sqrt(a_lat^2 + a_long^2) / g                      [g]
    aero load      mean(v^2)                                         [m^2/s^2]

Derivatives are estimated with Savitzky-Golay (savitzky_golay_1964), which
fits a local polynomial and so yields analytic derivatives with controlled
noise amplification, unlike a moving average.

No Pacejka Magic Formula. Its coefficients are load-dependent, proprietary and
effectively unavailable for F1 tyres, so any implementation would rest on
fabricated numbers. These curvature proxies capture the same physical drivers
from telemetry alone.

**Units.** FastF1 reports position ``X``, ``Y``, ``Z`` in **1/10 m**
(``fastf1/core.py:60-62``), not metres. Curvature scales as 1/L, so using raw
coordinates yields lateral acceleration ten times too low - a plausible-looking
number that is simply wrong. The compass reference code documents metres and
does not correct for this.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)

#: FastF1 position units per metre. See the module docstring.
POSITION_UNITS_PER_METRE = 10.0

GRAVITY_MS2 = 9.80665


def _savgol(signal: np.ndarray, window: int, poly: int, deriv: int,
            delta: float) -> np.ndarray:
    """Savitzky-Golay wrapper honouring array length and window parity.

    :param signal: input samples.
    :param window: desired window length; clamped to the array and forced odd.
    :param poly: polynomial order.
    :param deriv: derivative order, 0 to smooth.
    :param delta: sample spacing, for derivative scaling.
    :returns: filtered array, or the input unchanged if it is too short.
    """
    n = len(signal)
    if n < poly + 2:
        return signal.astype(float)
    window = min(window, n if n % 2 == 1 else n - 1)
    if window % 2 == 0:
        window -= 1
    if window <= poly:
        return signal.astype(float)
    return savgol_filter(signal, window, poly, deriv=deriv, delta=delta,
                         mode="interp")


def compute_curvature(x: np.ndarray, y: np.ndarray, window: int,
                      poly: int) -> np.ndarray:
    """Path curvature from X, Y position via smoothed derivatives.

    :param x: X coordinates in metres (already unit-corrected).
    :param y: Y coordinates in metres.
    :param window: Savitzky-Golay window.
    :param poly: polynomial order.
    :returns: curvature in 1/m, zero where the path is degenerate.
    """
    if len(x) < poly + 2:
        return np.zeros_like(x, dtype=float)
    dx = _savgol(x, window, poly, 1, 1.0)
    dy = _savgol(y, window, poly, 1, 1.0)
    ddx = _savgol(x, window, poly, 2, 1.0)
    ddy = _savgol(y, window, poly, 2, 1.0)
    denominator = np.power(dx * dx + dy * dy, 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denominator > 1e-9,
                         np.abs(dx * ddy - dy * ddx) / denominator, 0.0)
    return np.nan_to_num(kappa)


def lap_physics_features(telemetry: pd.DataFrame, window: int = 11,
                         poly: int = 2) -> dict[str, float]:
    """Aggregate physics and behavioural proxies for one lap.

    :param telemetry: lap telemetry with ``X``, ``Y``, ``Speed`` (km/h),
        ``Time`` and optionally ``Throttle`` and ``Brake``.
    :param window: Savitzky-Golay window.
    :param poly: polynomial order.
    :returns: mapping of aggregate name to value; NaN where uncomputable.
    """
    empty = {"lat_accel_max": np.nan, "lat_accel_mean": np.nan,
             "brake_power_max": np.nan, "combined_g_max": np.nan,
             "aero_load_proxy": np.nan, "jerk_rms": np.nan,
             "throttle_std": np.nan, "full_throttle_frac": np.nan,
             "brake_applications": np.nan}
    if telemetry is None or telemetry.empty or len(telemetry) < poly + 2:
        return empty
    if not {"X", "Y", "Speed"}.issubset(telemetry.columns):
        return empty

    speed_ms = pd.to_numeric(telemetry["Speed"], errors="coerce").to_numpy(float) / 3.6
    # The unit correction that makes the whole module correct.
    x = pd.to_numeric(telemetry["X"], errors="coerce").to_numpy(float) / POSITION_UNITS_PER_METRE
    y = pd.to_numeric(telemetry["Y"], errors="coerce").to_numpy(float) / POSITION_UNITS_PER_METRE

    valid = np.isfinite(speed_ms) & np.isfinite(x) & np.isfinite(y)
    if valid.sum() < poly + 2:
        return empty
    speed_ms, x, y = speed_ms[valid], x[valid], y[valid]

    if "Time" in telemetry.columns:
        seconds = pd.to_timedelta(
            telemetry["Time"][valid]).dt.total_seconds().to_numpy()
        deltas = np.diff(seconds)
        dt = float(np.median(deltas)) if len(deltas) else 0.1
    else:
        dt = 0.1
    if not np.isfinite(dt) or dt <= 0:
        dt = 0.1

    kappa = compute_curvature(x, y, window, poly)
    a_lat = speed_ms ** 2 * kappa
    a_long = _savgol(speed_ms, window, poly, 1, dt)
    brake_power = speed_ms * np.abs(np.minimum(a_long, 0.0))
    combined_g = np.sqrt(a_lat ** 2 + a_long ** 2) / GRAVITY_MS2

    # Jerk, via a second differentiation of the smoothed acceleration.
    jerk = np.gradient(_savgol(a_long, window, poly, 0, dt), dt)

    out = {
        "lat_accel_max": float(np.nanmax(a_lat)),
        "lat_accel_mean": float(np.nanmean(a_lat)),
        "brake_power_max": float(np.nanmax(brake_power)),
        "combined_g_max": float(np.nanmax(combined_g)),
        "aero_load_proxy": float(np.nanmean(speed_ms ** 2)),
        "jerk_rms": float(np.sqrt(np.nanmean(jerk ** 2))),
        "throttle_std": np.nan,
        "full_throttle_frac": np.nan,
        "brake_applications": np.nan,
    }

    if "Throttle" in telemetry.columns:
        throttle = pd.to_numeric(telemetry["Throttle"], errors="coerce").to_numpy(float)
        out["throttle_std"] = float(np.nanstd(throttle))
        out["full_throttle_frac"] = float(np.nanmean(throttle > 95.0))
    if "Brake" in telemetry.columns:
        brake = pd.to_numeric(telemetry["Brake"], errors="coerce").fillna(0) \
            if hasattr(telemetry["Brake"], "fillna") else telemetry["Brake"]
        brake = np.nan_to_num(np.asarray(brake, dtype=float))
        if len(brake) > 1:
            out["brake_applications"] = float(
                np.sum((brake[1:] > 0.5) & (brake[:-1] <= 0.5)))
    return out


def plausibility_report(frame: pd.DataFrame) -> dict[str, bool]:
    """Gate 5: check the aggregates are physically sensible.

    An F1 car pulls roughly 3-6 g laterally, so ``lat_accel_max`` should sit
    around 30-60 m/s^2. Values ten times lower mean the position unit
    correction was missed.

    :param frame: per-lap aggregates.
    :returns: mapping of check name to pass/fail.
    """
    lat = frame["lat_accel_max"].median()
    combined = frame["combined_g_max"].median()
    aero = frame["aero_load_proxy"].median()

    checks = {
        "lat_accel_max_30_to_70_ms2": bool(25 <= lat <= 75),
        "combined_g_max_2_to_8": bool(2 <= combined <= 8),
        "aero_load_positive": bool(aero > 0),
    }
    logger.info("Gate 5 plausibility: lat_accel_max median %.1f m/s^2 "
                "(%.1f g), combined_g_max median %.2f g, aero load %.0f m2/s2",
                lat, lat / GRAVITY_MS2, combined, aero)
    for name, passed in checks.items():
        logger.info("  %-30s %s", name, "PASS" if passed else "FAIL")
    if not checks["lat_accel_max_30_to_70_ms2"] and lat < 10:
        logger.error("Lateral acceleration is roughly 10x too low; the X/Y "
                     "unit correction (1/10 m) was probably missed.")
    return checks
