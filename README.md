# F1 Tyre Degradation Model

Predicts tyre degradation rate, in seconds of lap time lost per lap of tyre
age, from FastF1 race data for the 2022-2024 ground-effect regulation era.

## Status, stated plainly

The model is **real but modest**. It beats both a linear baseline and the
training mean on a held-out season it never saw, but it does not clear the
pre-registered 15% gate.

| Model | Holdout MAE (s/lap) | R² |
|---|---|---|
| Mean predictor | 0.03539 | — |
| Linear baseline | 0.03337 | +0.044 |
| **XGBoost (tuned)** | **0.03345** | **+0.097** |

Trained on 2022-2023, held out on 2024. 74.4% of predictions land within
0.05 s/lap.

**Gate 4 fails, at −0.2% against a 15% threshold** — the tuned tree is level
with the linear baseline rather than ahead of it. That is informative rather
than merely disappointing: once the target's construction bias is removed, the
relationship between circuit severity and degradation is close to linear, so
gradient boosting's extra flexibility buys almost nothing. XGBoost does hold a
better R² (+0.097 against +0.044), meaning it handles the tails better while
matching on average error.

The 15% threshold was fixed before anyone measured the available headroom. The
oracle ceiling — predicting each stint's own event-and-compound mean, which
requires knowing the answer in advance — is 38.0%. The model captures 14.4% of
that. Just under half of degradation variance lies between events; the rest
separates stints at the same race, where no event-level feature can reach.

The per-circuit degradation estimates underneath the model are stronger than
the model itself: they correlate 0.66-0.71 season to season. The strategy layer
therefore prefers observed data where it exists and uses the model only to fill
gaps, recording which is which.

## What is not modelled

- Wet and intermediate running. Those stints are excluded from the target fit.
- The non-linear degradation cliff. Degradation is linear in tyre age here, so
  projections far beyond observed stint lengths are extrapolation and are
  flagged as such.
- Anything outside 2022-2024. Requests beyond it are flagged, not extrapolated.
- Within-event variation. Roughly half the variance separates stints at the
  same race — traffic, fuel saving, driver management — and none of it is
  reachable from circuit-level features.

## A bias worth knowing about

Stints are accepted by the **precision** of the fitted slope, not by the
strength of its correlation. The compass design filtered on `|r| >= 0.3`,
which measures against this data as a slope-magnitude filter in disguise:

```
corr(|r|, |slope|) = +0.53      corr(|r|, stderr) = -0.20
```

Its pass rate ran from 29.5% for near-zero slopes to 98.7% for large ones, so
it systematically deleted the low-degradation circuits. Monaco was the clearest
casualty: its stints are measured *more* precisely than average (slope standard
error 0.0062 against 0.0152) and were discarded for being flat. Replacing the
criterion raised between-event variance from 32.3% to 49.7% and fixed every
remaining directional error in the simulator.

## Pipeline

```
scripts/probe_api_cost.py      measure API cost before spending budget
scripts/run_collect_laps.py    collect races, one parquet per session
scripts/run_build_dataset.py   validate, fuel-correct, build the stint table
scripts/run_cwi_study.py       the pre-registered Spearman target study
scripts/run_diagnostics.py     variance decomposition and the oracle ceiling
scripts/run_train.py           train, evaluate, report the gates
scripts/run_export_ui.py       export model parameters for the front end
```

Collection is resumable: re-run the same command. 68 sessions across three
seasons collect in roughly 10 minutes of runtime against a 500 requests/hour
limit, and every subsequent run is free because cache hits bypass the limiter.

## Results of the pre-registered decisions

- **CWI Spearman study**: ρ = −0.089, below the 0.2 threshold, so the energy
  proxy was demoted to a feature and the target reduced to the fuel-corrected
  degradation slope. Telemetry left the critical path as a result.
- **Target reliability**: split-half 0.92 Pearson / 0.79 Spearman. The target
  is measured reliably; the difficulty is transferring across events, not
  measuring within them.
- **Cross-season circuit stability**: 0.74 for 2022 vs 2023, 0.64 for 2023 vs
  2024. Per-circuit degradation persists well enough that the strategy layer
  prefers observed values over model predictions wherever they exist.
- **LSTM**: not built. On a ~2,000-row per-stint tabular target there is no
  sequence axis left, and the gate would not be met.

## Setup

Python 3.12 in `.venv`. See `requirements.lock.txt` for the working versions
and `CLAUDE.md` for the rules on not disturbing them.

Historical documents in `docs/` and `archive/legacy_2025/` describe the
superseded design. They are kept for reference and are not accurate
descriptions of this code.
