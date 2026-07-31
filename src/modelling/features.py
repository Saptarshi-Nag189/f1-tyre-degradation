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

#: Tested and rejected. Held-out ablation, each set tuned separately:
#:
#:   core 9                          MAE 0.03455  R2 +0.023
#:   core + context                  MAE 0.03399  R2 +0.066
#:   core + Team                     MAE 0.03345  R2 +0.097   <- best
#:   core + context + Team           MAE 0.03380  R2 +0.084
#:   core + context + Team + Driver  MAE 0.03350  R2 +0.096
#:
#: Stint context makes the model worse, and Driver adds nothing beyond Team.
#: Neither clears the 3% tier gate. The columns are still computed on the
#: stint table because they are useful for diagnosis, but they are not
#: features. Recorded here so the experiment is not silently repeated.
REJECTED = {
    "stint_number": "no gain; degrades the model when combined with Team",
    "fuel_at_start_kg": "no gain; the fuel effect is already removed from the "
                        "target by the fuel correction",
    "race_fraction_at_start": "no gain",
    "Driver": "no gain beyond Team; 25 extra one-hots for a worse R2",
}

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


#: Categorical Tier A features, one-hot encoded against training levels only.
#: Team earns its place: within event and compound, the residual degradation
#: carries a stable car signature (Haas +0.014 s/lap, Mercedes -0.011 s/lap,
#: over 150+ stints each), it reflects real setup and aerodynamic differences,
#: and the team is of course known before the stint is run.
#: Team only. Driver was tested on the expectation that tyre management is a
#: real driver skill, and it did not survive the ablation above: Team already
#: carries the signal, and 25 further one-hots produced a worse R2.
TIER_A_CATEGORICAL = ["Team"]


def available(frame_columns) -> list[str]:
    """Return the numeric Tier A features present in a frame, in tier order.

    :param frame_columns: columns of the stint table.
    :returns: usable feature names.
    """
    present = set(frame_columns)
    return [f for f in TIER_A if f in present]


def build_matrix(frame, numeric: list[str], categories: dict[str, list[str]]):
    """Assemble a model matrix with categoricals one-hot encoded.

    Category levels must come from the training split alone, so no holdout
    information reaches the encoding.

    :param frame: stint table slice.
    :param numeric: numeric feature names.
    :param categories: mapping of column to the training levels to encode.
    :returns: (DataFrame of features, list of column names).
    """
    import pandas as pd

    matrix = frame[numeric].copy()
    for column, levels in categories.items():
        if column not in frame.columns:
            continue
        for level in levels:
            matrix[f"{column}_{level}"] = (frame[column] == level).astype(float)
    return matrix, list(matrix.columns)


def training_categories(train, columns: list[str] | None = None
                        ) -> dict[str, list[str]]:
    """Collect categorical levels observed in the training split.

    :param train: training stint table.
    :param columns: categorical columns; defaults to TIER_A_CATEGORICAL.
    :returns: mapping of column to sorted levels.
    """
    columns = columns or TIER_A_CATEGORICAL
    return {c: sorted(train[c].dropna().unique().tolist())
            for c in columns if c in train.columns}
