"""Phase 2: assemble collected sessions into a modelling-ready stint table.

Loads every per-session parquet, validates it, applies the fuel correction and
reduces it to one row per ``(Year, RoundNumber, DriverNumber, Stint)``, joined
to the two-axis circuit traits.

The unit of analysis is the stint, not the lap. A per-lap sequence target made
the superseded model largely an autocorrelation exercise, because each lap's
three sector times sum exactly to its own lap time and consecutive lap deltas
are strongly serially correlated.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src import config
from src.acquisition import validator
from src.features import target as target_mod

logger = logging.getLogger(__name__)


def load_sessions(raw_dir: Path, years: list[int] | None = None) -> pd.DataFrame:
    """Load and concatenate every collected session parquet.

    :param raw_dir: directory holding ``laps_{year}_R{round}.parquet`` files.
    :param years: optional season filter.
    :returns: concatenated laps frame.
    :raises FileNotFoundError: if no session files are present.
    """
    paths = sorted(raw_dir.glob("laps_*.parquet"))
    if years:
        paths = [p for p in paths if int(p.stem.split("_")[1]) in years]
    if not paths:
        raise FileNotFoundError(
            f"No session parquet files in {raw_dir}. Run scripts/run_collect_laps.py.")

    frames = []
    for path in paths:
        try:
            frames.append(pd.read_parquet(path))
        except Exception as exc:                   # defensive: one bad file
            logger.error("Unreadable session %s: %s", path.name, exc)

    laps = pd.concat(frames, ignore_index=True)
    logger.info("Loaded %d sessions, %d laps, seasons %s",
                len(frames), len(laps), sorted(laps["Year"].unique().tolist()))
    return laps


def attach_track_traits(stints: pd.DataFrame, track_cfg: dict) -> pd.DataFrame:
    """Join the two-axis abrasion/energy traits onto the stint table.

    Abrasion and lateral energy genuinely diverge: Silverstone is maximum
    energy on a low-abrasion surface, Bahrain the reverse. Both axes are joined
    alongside the composite so a model can use whichever carries signal.

    :param stints: stint table with ``EventName``.
    :param track_cfg: parsed track_traits.yaml.
    :returns: stint table with ``abrasion``, ``energy`` and ``composite``.
    """
    out = stints.copy()
    event_keys = track_cfg.get("event_name_to_key", {})
    circuits = track_cfg.get("circuits", {})

    def _traits(event_name: object) -> tuple[float, float, float]:
        entry = event_keys.get(str(event_name))
        if entry is None:
            return (float("nan"),) * 3
        traits = circuits.get(entry["traits"])
        if traits is None:
            return (float("nan"),) * 3
        return (float(traits["abrasion"]), float(traits["energy"]),
                float(traits["composite"]))

    resolved = out["EventName"].map(_traits)
    out["abrasion"] = [t[0] for t in resolved]
    out["energy"] = [t[1] for t in resolved]
    out["composite"] = [t[2] for t in resolved]

    unmapped = out.loc[out["abrasion"].isna(), "EventName"].unique().tolist()
    if unmapped:
        logger.warning("No track traits for events: %s", unmapped)
    return out


def build_dataset(years: list[int] | None = None,
                  energy_col: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run the full offline pipeline from session parquet to stint table.

    :param years: optional season filter.
    :param energy_col: per-lap energy aggregate column; None uses the
        telemetry-free speed-trap proxy.
    :returns: (validated laps frame, stint table).
    """
    settings = config.settings()
    track_cfg = config.track_traits()

    laps = load_sessions(config.resolve_path("raw_laps"), years)

    logger.info("--- validation ---")
    laps = validator.validate(
        laps, track_cfg, config.compound_mapping(), settings["clean_lap"])

    logger.info("--- fuel correction ---")
    fuel = settings["fuel_correction"]
    laps = target_mod.fuel_correct(
        laps, penalty=fuel["penalty_s_per_kg"],
        initial_fuel=fuel["initial_fuel_kg"])

    logger.info("--- stint table ---")
    tgt = settings["target"]
    stints = target_mod.build_stint_table(
        laps, min_r=tgt["linregress_min_r"], min_laps=tgt["min_stint_laps"],
        energy_col=energy_col)

    if not stints.empty:
        stints = attach_track_traits(stints, track_cfg)
    return laps, stints
