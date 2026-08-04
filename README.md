# F1 Tyre Degradation Model

Predicts how fast a Formula 1 tyre loses lap time, in seconds per lap of tyre
age, and uses that to simulate pit strategies. Built on public FastF1 data for
the 2022-2024 ground-effect regulation era.

## What it does

Given a circuit, a compound, track and air temperature, and the tyre's age at
the start of a stint, the model estimates a degradation rate. A simulator then
projects lap times across a race distance, enumerates one- and two-stop plans
under the mandatory two-compound rule, and returns the fastest with its pit
window.

```python
from src.core.pipeline import F1Pipeline

pipeline = F1Pipeline().load()
result = pipeline.predict_strategy({
    "season": 2024, "circuit": "silverstone", "driver": "NOR",
    "laps": 52, "air_temp": 20, "track_temp": 32,
    "conditions": "dry", "compounds": ["SOFT", "MEDIUM", "HARD"],
})

result["best_strategy"]["label"]      # '1-stop: SOFT-MEDIUM'
result["best_strategy"]["pit_laps"]   # [22]
result["confidence"]["level"]         # 'high'
```

Every prediction carries its provenance. Where enough stints have been observed
at that circuit and compound, the empirical value is used; where they have not,
the model fills the gap and the response says so.

## Results

Trained on 2022-2023, evaluated on a held-out 2024 season.

| Model | Holdout MAE (s/lap) | R2 |
|---|---|---|
| Mean predictor | 0.03539 | - |
| Linear baseline | 0.03337 | +0.044 |
| XGBoost (tuned) | 0.03345 | +0.097 |

The tuned model beats the mean predictor and matches the linear baseline on
absolute error while roughly doubling its R2. **It does not clear the
pre-registered acceptance gate of a 15% MAE improvement over the baseline; the
measured figure is -0.2%.**

That gate was set before anyone established how much headroom the problem has.
Predicting each stint's own event-and-compound mean, which requires knowing the
answer in advance, improves on the global mean by only 38%. Just under half of
degradation variance sits between events; the rest separates stints within a
single race, where traffic, fuel saving and driver management dominate and no
circuit-level feature can reach.

Directional checks against known race strategy pass: Monaco, Monza, Spa,
Singapore, Jeddah and Silverstone come out one-stop; Bahrain, Suzuka, Barcelona
and Hungaroring two-stop.

## How stints are selected

The most transferable finding here is a defect in the standard method.

Per-stint degradation is normally taken as the slope of fuel-corrected lap time
against tyre age, keeping only stints whose fit clears a correlation threshold
such as `|r| >= 0.3`. That threshold is described as a fit-quality filter. It is
not. Measured across 2,815 dry stints:

```
corr(|r|, |slope|)  = +0.53      corr(|r|, stderr) = -0.20
```

It tracks how *large* the degradation is roughly three times more strongly than
how *well* it was measured, because `r` tends to zero as the slope does,
independently of the residual scatter. A genuinely flat stint cannot clear the
threshold however precisely it is measured.

| Slope magnitude (s/lap) | Stints | Accepted by \|r\| >= 0.3 |
|---|---|---|
| 0.00-0.02 | 505 | **29.5%** |
| 0.02-0.05 | 653 | 86.2% |
| 0.05-0.10 | 935 | 98.7% |
| 0.10-0.20 | 541 | 99.8% |

Monaco is the clearest casualty: its stints are measured *more* precisely than
average (median slope standard error 0.0062 against 0.0152 elsewhere) and were
being discarded for being flat.

This model instead accepts a stint when the standard error of its fitted slope
is at most 0.02 s/lap, small against a typical degradation of 0.065 s/lap. The
lower plausibility bound is -0.05 s/lap rather than zero, because a circuit with
no measurable degradation scatters either side of zero and truncating at zero
keeps only the positive half of that noise.

Per-circuit degradation then correlates 0.64 to 0.74 between seasons, against
0.18 under the correlation filter.

## Pipeline

```
scripts/probe_api_cost.py         measure API cost before spending budget
scripts/run_collect_laps.py       collect races, one parquet per session
scripts/run_build_dataset.py      validate, fuel-correct, build the stint table
scripts/run_cwi_study.py          pre-registered Spearman study on the target
scripts/run_diagnostics.py        variance decomposition and the oracle ceiling
scripts/run_collect_telemetry.py  per-lap physics aggregates from telemetry
scripts/run_telemetry_study.py    curvature proxy against the speed-trap proxy
scripts/run_train.py              train, evaluate, report the gates
scripts/run_fit_diagnostics.py    over/under-fitting evidence
scripts/run_export_ui.py          export model parameters for the front end
```

Collection is resumable: re-run the same command. Each session is written to its
own parquet through a temporary file and an atomic rename, so an interruption
costs one session rather than the corpus. Rate limiting is detected by exception
type and stops the run cleanly.

68 race sessions take about 10 minutes of network time against a 500
requests/hour limit. Cache hits bypass the limiter entirely, so every subsequent
run is free.

Tuned hyperparameters persist to `config/tuned_params.json` and are reloaded by
default, guarded against a stale feature set.

## Design decisions

**Fuel correction is applied before any slope is fitted.** Fuel remaining on lap
N of an L-lap race is `110 - (N-1) * 110/L` kilograms, at 0.032 s/kg. Without
it, the target measures fuel burn rather than tyre wear.

**Features must be knowable before the stint runs**, since the simulator's job is
to choose stint length. Stint length and clean-lap count are excluded as
reverse-causal.

**Validation splits on whole events.** Splitting rows lets one race contribute
stints to both sides of a fold boundary, leaking shared track, weather and
safety-car conditions.

**Runs on CPU.** The stint table is ~1,700 rows; `device='cuda'` is measurably
slower than `tree_method='hist'` on CPU at this size (27.5 ms against 139.6 ms),
because transfer overhead dominates. GPU only wins above roughly 200k rows.

## Scope

- Dry running only. Wet and intermediate stints are excluded from the target fit
  and requests return a `dry_model_only` flag.
- Degradation is linear in tyre age. The non-linear cliff beyond a tyre's
  working range is not modelled, and projections past observed stint lengths are
  flagged as extrapolation.
- 2022-2024 only. Requests outside that range are flagged, not extrapolated.
- Absolute Pirelli C-numbers are confirmed for a minority of rounds;
  `compound_relative` carries the load elsewhere.
- Residuals drift -0.26 across the holdout season, so the model is well fitted
  but not uniformly calibrated within a year.

## Setup

Python 3.12. See `requirements.lock.txt` for the working versions and
`CLAUDE.md` for the conventions this repository follows.

```bash
python -m venv .venv && .venv/Scripts/activate
pip install -r requirements.txt
python scripts/probe_api_cost.py
python scripts/run_collect_laps.py --year 2023
python scripts/run_build_dataset.py && python scripts/run_train.py
```

Full measurements and the reasoning behind each decision are in
[docs/FINDINGS.md](docs/FINDINGS.md). The write-up is in
[docs/report/](docs/report/).
