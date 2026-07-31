"""Feature tier definitions, split by what is knowable *before* a stint runs.

This distinction is not in the compass document but is forced by the intended
use. The strategy simulator must answer "how will this compound degrade if I
fit it now", so it can only use quantities known in advance: the circuit, the
compound, the temperatures and the tyre's age at the start of the stint.

Several attractive columns are therefore excluded:

- ``StintLength`` and ``n_laps`` are reverse-causal. A stint that degraded
  quickly was pitted early, so stint length is partly an *effect* of
  degradation. Feeding it back in would inflate accuracy and destroy the
  simulator, whose entire job is to choose the stint length.
- ``intercept`` and ``r_value`` are outputs of the very regression that
  produces the target.
- ``energy_proxy`` measured per stint is only known after the stint has been
  driven. Its circuit-level average is a genuine track property and is
  knowable in advance, so energy enters through the circuit traits instead.

Tier B and Tier C, when they arrive, share the retrospective problem: physics
and behavioural aggregates describe laps already driven. They are useful for
explaining degradation but cannot be used for forward prediction unless
averaged to a circuit or driver level first.
"""
from __future__ import annotations

#: Tier A: knowable before the stint starts, so usable by the simulator.
TIER_A = [
    "TyreLife_start",     # tyre age at stint start (laps)
    "compound_ordinal",   # absolute Pirelli C-number, NaN where unconfirmed
    "compound_relative",  # hard/medium/soft as 0/1/2, always available
    "AirTemp_mean",       # deg C
    "TrackTemp_mean",     # deg C
    "abrasion",           # circuit surface abrasiveness, 1-3
    "energy",             # circuit lateral/vertical load, 1-3
    "composite",          # circuit composite severity, 1-3
    "RaceLaps",           # scheduled race distance, a proxy for circuit length
]

#: Columns that must never become features, with the reason enforced in tests.
FORBIDDEN = {
    "deg_rate": "the target",
    "CWI": "the target",
    "intercept": "output of the target regression",
    "r_value": "output of the target regression",
    "n_laps": "reverse-causal: derived from how long the stint actually ran",
    "StintLength": "reverse-causal: an effect of degradation, and the "
                   "quantity the simulator exists to choose",
    "energy_proxy": "measured from laps already driven; enters via circuit traits",
}


def available(frame_columns) -> list[str]:
    """Return the Tier A features present in a frame, preserving tier order.

    :param frame_columns: columns of the stint table.
    :returns: usable feature names.
    """
    present = set(frame_columns)
    return [f for f in TIER_A if f in present]
