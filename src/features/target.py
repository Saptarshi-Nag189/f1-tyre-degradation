"""Phase 3/4: fuel correction, per-stint degradation rate and the CWI target.

Pipeline:

1. Fuel correction. ``corrected_lap = lap_s - penalty * fuel_remaining``, where
   fuel burned by lap N is ``(N-1) * burn_per_lap`` because the car completes
   lap N carrying the fuel it had at the *start* of that lap. Without this, a
   lap-time target measures fuel burn far more than it measures tyre wear,
   which is the central defect of the superseded pipeline.
2. Per-stint degradation rate: OLS slope of corrected lap time against
   TyreLife over clean laps, after IQR outlier removal, retained only if
   ``|r| > min_r``.
3. Energy proxy. Two forms are supported. The telemetry-free "speed-trap"
   proxy uses mean squared speed-trap readings as a stand-in for ``mean(v^2)``,
   which is exactly the compass aero-load proxy; it needs no telemetry and no
   API budget, so the Spearman study can run before any telemetry is collected.
   The "curvature" form uses the true physics aggregates once available.
4. CWI construction under a pre-registered Spearman decision rule.

``deg_rate`` in seconds per lap is always carried alongside the z-scored CWI,
so the strategy simulator has a dimensionally meaningful quantity to work with
regardless of what the Spearman gate decides.

References: archard_1953 (wear proportional to frictional work),
persson_2001 (viscoelastic friction). See config/references.yaml.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

logger = logging.getLogger(__name__)

STINT_GROUP = ["Year", "RoundNumber", "DriverNumber", "Stint"]

#: Speed-trap columns used by the telemetry-free energy proxy.
SPEED_TRAP_COLS = ["SpeedI1", "SpeedI2", "SpeedFL", "SpeedST"]


def fuel_correct(laps: pd.DataFrame, penalty: float, initial_fuel: float,
                 race_laps_col: str = "RaceLaps") -> pd.DataFrame:
    """Apply the linear fuel correction to lap times.

    ``fuel_remaining(N) = initial - (N-1) * (initial / race_laps)``
    ``corrected_lap_s = lap_s - penalty * fuel_remaining``

    Early laps carry the most fuel and so receive the largest subtraction,
    which is what makes corrected times comparable across a stint.

    :param laps: laps with ``lap_s``, ``LapNumber`` and the race length column.
    :param penalty: seconds per kilogram, e.g. 0.032.
    :param initial_fuel: starting fuel in kg, e.g. 110.
    :param race_laps_col: column giving scheduled race laps per event.
    :returns: frame with ``fuel_remaining_kg`` and ``corrected_lap_s``.
    """
    out = laps.copy()
    if "lap_s" not in out.columns:
        raise KeyError("lap_s missing; it is written at collection time")

    if race_laps_col not in out.columns:
        raise KeyError(
            f"{race_laps_col} missing; fuel correction needs scheduled race "
            "distance, captured at collection time")

    race_laps = pd.to_numeric(out[race_laps_col], errors="coerce").replace(0, np.nan)
    burn_per_lap = initial_fuel / race_laps
    fuel_remaining = (initial_fuel
                      - (pd.to_numeric(out["LapNumber"], errors="coerce") - 1)
                      * burn_per_lap).clip(lower=0.0)

    out["fuel_remaining_kg"] = fuel_remaining
    out["corrected_lap_s"] = out["lap_s"] - penalty * fuel_remaining

    n_missing = int(fuel_remaining.isna().sum())
    if n_missing:
        logger.warning("Fuel correction unavailable for %d rows (missing race length)",
                       n_missing)
    logger.info("Fuel correction applied: mean shift %.3f s over %d rows",
                float((penalty * fuel_remaining).mean(skipna=True)), len(out))
    return out


def _iqr_mask(values: np.ndarray) -> np.ndarray:
    """Return a boolean mask keeping values inside 1.5*IQR fences."""
    q1, q3 = np.nanpercentile(values, [25, 75])
    iqr = q3 - q1
    return (values >= q1 - 1.5 * iqr) & (values <= q3 + 1.5 * iqr)


def speed_trap_energy(stint: pd.DataFrame) -> float:
    """Telemetry-free energy proxy: mean squared speed-trap reading.

    Speed traps are recorded in km/h, converted here to m/s so the quantity is
    dimensionally the same ``mean(v^2)`` the compass aero-load proxy computes
    from telemetry.

    :param stint: clean laps of one stint.
    :returns: mean of v^2 in m^2/s^2, or NaN if no trap data is present.
    """
    present = [c for c in SPEED_TRAP_COLS if c in stint.columns]
    if not present:
        return float("nan")
    speeds_ms = stint[present].apply(pd.to_numeric, errors="coerce") / 3.6
    return float(np.nanmean(np.square(speeds_ms.to_numpy(dtype=float))))


@dataclass
class StintResult:
    """Per-stint degradation regression result."""

    deg_rate: float          # slope, seconds per lap of tyre life
    r_value: float
    n_laps: int
    energy_proxy: float
    intercept: float         # corrected pace at zero tyre life, seconds


def stint_degradation(stint: pd.DataFrame, min_r: float, min_laps: int,
                      energy_col: str | None = None) -> StintResult | None:
    """Fit the corrected-lap-time slope against TyreLife for one stint.

    :param stint: all laps of one stint, with ``is_clean_lap`` and
        ``corrected_lap_s``.
    :param min_r: minimum ``|r|`` to accept the slope.
    :param min_laps: minimum clean laps required.
    :param energy_col: column holding a per-lap energy aggregate; when None the
        telemetry-free speed-trap proxy is used instead.
    :returns: a StintResult, or None if the stint is unusable.
    """
    clean = stint[stint["is_clean_lap"]].copy()
    if len(clean) < min_laps:
        return None

    clean = clean[clean["corrected_lap_s"].notna()]
    if len(clean) < min_laps:
        return None

    clean = clean[_iqr_mask(clean["corrected_lap_s"].to_numpy(dtype=float))]
    if len(clean) < min_laps:
        return None

    x = pd.to_numeric(clean["TyreLife"], errors="coerce").to_numpy(dtype=float)
    y = clean["corrected_lap_s"].to_numpy(dtype=float)
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if len(x) < min_laps or np.ptp(x) == 0:
        return None

    reg = linregress(x, y)
    if abs(reg.rvalue) < min_r:
        return None

    if energy_col and energy_col in clean.columns:
        energy = float(pd.to_numeric(clean[energy_col], errors="coerce").mean())
    else:
        energy = speed_trap_energy(clean)

    return StintResult(deg_rate=float(reg.slope), r_value=float(reg.rvalue),
                       n_laps=int(len(x)), energy_proxy=energy,
                       intercept=float(reg.intercept))


def build_stint_table(laps: pd.DataFrame, min_r: float, min_laps: int,
                      energy_col: str | None = None,
                      exclude_wet: bool = True) -> pd.DataFrame:
    """Build the per-stint target table.

    :param laps: fully validated and fuel-corrected laps frame.
    :param min_r: minimum ``|r|`` filter.
    :param min_laps: minimum clean laps per stint.
    :param energy_col: per-lap energy aggregate column, or None for the
        telemetry-free proxy.
    :param exclude_wet: drop stints flagged wet (different physics).
    :returns: one row per stint with deg_rate, r_value, n_laps, energy_proxy.
    """
    frame = laps
    n_all = frame.groupby(STINT_GROUP).ngroups
    if exclude_wet and "wet_stint" in frame.columns:
        frame = frame[~frame["wet_stint"]]
        logger.info("Excluded %d wet stints from the slope fit",
                    n_all - frame.groupby(STINT_GROUP).ngroups)

    rows: list[dict] = []
    rejected = {"too_few_laps": 0, "low_r": 0}

    for keys, stint in frame.groupby(STINT_GROUP, observed=True):
        result = stint_degradation(stint, min_r, min_laps, energy_col)
        if result is None:
            # Distinguish the two rejection causes for the retention diagnostic.
            n_clean = int(stint["is_clean_lap"].sum())
            rejected["too_few_laps" if n_clean < min_laps else "low_r"] += 1
            continue
        row = dict(zip(STINT_GROUP, keys))
        row.update(asdict(result))
        first = stint.iloc[0]
        for col in ("EventName", "Compound", "compound_ordinal",
                    "compound_category", "compound_known", "Team",
                    "pit_lane_start", "weather_imputed", "RaceLaps"):
            if col in stint.columns:
                row[col] = first[col]
        for col in ("AirTemp", "TrackTemp"):
            if col in stint.columns:
                row[f"{col}_mean"] = float(
                    pd.to_numeric(stint[col], errors="coerce").mean())
        row["StintLength"] = int(len(stint))
        row["TyreLife_start"] = float(
            pd.to_numeric(stint["TyreLife"], errors="coerce").min())
        rows.append(row)

    table = pd.DataFrame(rows)
    retention = 100.0 * len(table) / max(n_all, 1)
    logger.info("Stint table: %d usable of %d raw stints (%.1f%% retention)",
                len(table), n_all, retention)
    logger.info("  rejected: %d too few clean laps, %d |r| below %.2f",
                rejected["too_few_laps"], rejected["low_r"], min_r)
    if not table.empty:
        logger.info("  deg_rate: median %.4f s/lap, %.1f%% positive",
                    float(table["deg_rate"].median()),
                    100.0 * float((table["deg_rate"] > 0).mean()))
    return table


def validate_and_build_cwi(stints: pd.DataFrame, keep_thr: float,
                           downweight_low: float, energy_w_full: float,
                           laptime_w_down: float) -> tuple[pd.DataFrame, dict]:
    """Run the Spearman validation study and construct the CWI target.

    Pre-registered decision rule, fixed before seeing the data:

    - ``rho > keep_thr``            -> energy stays, weight ``energy_w_full``;
    - ``downweight_low <= rho``     -> down-weighted, lap-time weight
      ``laptime_w_down``;
    - ``rho < downweight_low``      -> energy demoted to a feature (weight 0),
      leaving the CWI equal to the z-scored degradation slope.

    Both components are z-scored before weighting, so the CWI is unitless.
    ``deg_rate`` is retained separately in seconds per lap.

    :param stints: stint table from :func:`build_stint_table`.
    :param keep_thr: Spearman threshold to keep energy at full weight.
    :param downweight_low: lower Spearman threshold before demotion.
    :param energy_w_full: energy weight when fully kept.
    :param laptime_w_down: lap-time weight in the down-weighted branch.
    :returns: (stints with a ``CWI`` column, decision metadata).
    """
    if stints.empty:
        logger.error("Empty stint table; cannot run the Spearman study")
        return stints, {"decision": "no_data", "spearman_rho": float("nan")}

    valid = stints.dropna(subset=["deg_rate", "energy_proxy"])
    if len(valid) < 10:
        logger.warning("Only %d stints with both components; Spearman unreliable",
                       len(valid))
        rho, p_value = float("nan"), float("nan")
    else:
        rho, p_value = spearmanr(valid["energy_proxy"], valid["deg_rate"])

    logger.info("Spearman(energy, deg_rate) = %.4f (p=%.3g, n=%d)",
                rho, p_value, len(valid))

    def _z(series: pd.Series) -> pd.Series:
        sd = series.std(ddof=0)
        if not sd or np.isnan(sd):
            return series * 0.0
        return (series - series.mean()) / sd

    z_deg = _z(stints["deg_rate"])
    z_energy = _z(stints["energy_proxy"])

    if not np.isnan(rho) and rho > keep_thr:
        decision = "energy_in_target"
        energy_w = energy_w_full
        laptime_w = 1.0 - energy_w
    elif not np.isnan(rho) and rho >= downweight_low:
        decision = "energy_downweighted"
        laptime_w = laptime_w_down
        energy_w = 1.0 - laptime_w
    else:
        decision = "energy_demoted_to_feature"
        energy_w, laptime_w = 0.0, 1.0

    out = stints.copy()
    out["CWI"] = laptime_w * z_deg + energy_w * z_energy.fillna(0.0)

    meta = {
        "spearman_rho": None if np.isnan(rho) else float(rho),
        "spearman_p": None if np.isnan(p_value) else float(p_value),
        "n_stints": int(len(valid)),
        "decision": decision,
        "energy_weight": float(energy_w),
        "laptime_weight": float(laptime_w),
    }
    logger.info("CWI decision: %s (energy_w=%.2f, laptime_w=%.2f)",
                decision, energy_w, laptime_w)
    if decision == "energy_demoted_to_feature":
        logger.info("  CWI is now the z-scored fuel-corrected degradation slope; "
                    "deg_rate in s/lap remains the interpretable target.")
    return out, meta


def split_half_reliability(laps: pd.DataFrame, min_r: float,
                           min_laps: int) -> dict[str, float]:
    """Estimate how much of the target is signal rather than noise.

    Fits ``deg_rate`` separately on the odd and even clean laps of each stint
    and correlates the two. A low correlation means the target itself is
    largely noise, in which case no model can clear the 15% baseline gate and
    the honest response is to report that measurement rather than tune against
    it.

    :param laps: validated, fuel-corrected laps frame.
    :param min_r: minimum ``|r|`` for an accepted slope.
    :param min_laps: minimum clean laps per half.
    :returns: mapping with the Pearson and Spearman correlations and n.
    """
    odd_rates, even_rates = [], []
    for _, stint in laps.groupby(STINT_GROUP, observed=True):
        clean = stint[stint["is_clean_lap"]]
        if len(clean) < 2 * min_laps:
            continue
        halves = []
        for offset in (0, 1):
            half = clean.iloc[offset::2]
            result = stint_degradation(half, min_r=0.0, min_laps=min_laps)
            halves.append(result.deg_rate if result else None)
        if halves[0] is not None and halves[1] is not None:
            odd_rates.append(halves[0])
            even_rates.append(halves[1])

    if len(odd_rates) < 10:
        logger.warning("Only %d stints long enough for split-half reliability",
                       len(odd_rates))
        return {"n": len(odd_rates), "pearson": float("nan"),
                "spearman": float("nan")}

    pearson = float(np.corrcoef(odd_rates, even_rates)[0, 1])
    spearman = float(spearmanr(odd_rates, even_rates).statistic)
    logger.info("Split-half reliability over %d stints: pearson %.3f, spearman %.3f",
                len(odd_rates), pearson, spearman)
    return {"n": len(odd_rates), "pearson": pearson, "spearman": spearman}
