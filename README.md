# F1 Tyre Degradation Model

Predicts tyre degradation rate, in seconds of lap time lost per lap of tyre
age, from FastF1 race data for the 2022-2024 ground-effect regulation era.

## Status, stated plainly

The model is **real but modest**. It beats both a linear baseline and the
training mean on a held-out season it never saw, but it does not clear the
pre-registered 15% gate.

| Model | Holdout MAE (s/lap) | R² |
|---|---|---|
| Mean predictor | 0.04020 | −0.000 |
| Linear baseline | 0.03994 | −0.053 |
| **XGBoost (tuned)** | **0.03810** | **+0.047** |

Trained on 2022-2023 (1,367 stints), held out on 2024 (734 stints).
76.8% of predictions land within 0.05 s/lap.

**Gate 4 fails at +4.6% against a 15% threshold.** That threshold was fixed
before anyone measured the available headroom. The oracle ceiling — predicting
each stint's own event-and-compound mean, which requires knowing the answer in
advance — is 27.4%. The model captures 19% of that. Only 32.3% of degradation
variance lies between events at all; the remaining 67.7% separates stints at
the same race, where no event-level feature can reach.

The per-circuit degradation estimates underneath the model are stronger than
the model itself: they correlate 0.66-0.71 season to season. The strategy layer
therefore prefers observed data where it exists and uses the model only to fill
gaps, recording which is which.

## What is not modelled

- Wet and intermediate running. Those stints are excluded from the target fit.
- The non-linear degradation cliff. Degradation is linear in tyre age here.
- Anything outside 2022-2024. Requests beyond it are flagged, not extrapolated.
- **Low-degradation circuits are under-represented.** Monaco degrades so little
  that its slope fits fall below the `|r| >= 0.3` quality filter and are
  discarded, so its predictions come from the model and are flagged low
  confidence. The simulator currently returns a two-stop for Monaco, which is
  wrong.

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
- **LSTM**: not built. On a ~2,000-row per-stint tabular target there is no
  sequence axis left, and the gate would not be met.

## Setup

Python 3.12 in `.venv`. See `requirements.lock.txt` for the working versions
and `CLAUDE.md` for the rules on not disturbing them.

Historical documents in `docs/` and `archive/legacy_2025/` describe the
superseded design. They are kept for reference and are not accurate
descriptions of this code.
