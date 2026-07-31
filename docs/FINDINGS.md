# Findings

A complete record of what was measured, what was decided, and why. Written
because the project's original documentation described code that did not
exist, and that gap is what made the rebuild necessary.

Every number here is reproducible from the scripts named beside it.

---

## 1. The state that prompted the rebuild

Verified by reading the code and artefacts, not inferred:

| Finding | Evidence |
|---|---|
| Both data pickles truncated | Neither ends in the pickle STOP opcode; `pickle.load` raises `EOFError` on both (3.43 GB and 1.70 GB) |
| FastF1 cache deleted | `data/cache/fastf1_cache` absent; every re-collection started cold against a 500/hour limit |
| Three collection runs lost | Logs from Dec 2025, 1 Mar 2026 and 4 Mar 2026 all end in rate-limit failures |
| Target measured fuel burn, not wear | `lap_time_delta` with **no fuel correction** |
| Near-persistence leakage | The three sector times in each input window sum exactly to that lap's lap time |
| Split was by driver, not time | Sequences built by a `groupby` pandas sorts alphabetically, then sliced 70/15/15, despite a comment claiming temporal ordering |
| ~570 lines unreachable | Including the only callers of `gpu_processing.py` |
| Documented features absent | No physics-informed loss, no thermal/mechanical heads; the loss was plain `SmoothL1Loss` |
| Config/artefact mismatch | Reported metrics came from `sequence_length=5`; config said 3 |

Recorded baseline from the superseded model, for reference only:
RMSE 2.223 s, MAE 1.315 s, R² 0.454.

---

## 2. Measurements that reshaped the plan

### API cost (`scripts/probe_api_cost.py`)

| Measurement | Value |
|---|---|
| Cold load, laps + weather | **11 requests/session** |
| Identical repeat load | **0 requests** |
| `telemetry=True` increment | **+2 requests** |

The zero-request repeat is the important one. `_CachedSessionWithRateLimiting`
is `(CacheMixin, _SessionWithRateLimiting)`, and on a cache hit `CacheMixin.send`
returns without calling `super().send()`, so a cached request never reaches the
limiter. **Preserving the cache is the single highest-leverage rule in the
project.** Deleting it is what made the earlier failures unrecoverable.

This also inverted the assumption that telemetry was expensive. At +2 requests
per session it was never a budget problem; the 245 MB/event cost came from
*persisting raw frames*, not from fetching them.

### Collection outcome

68 sessions across 2022-2024, **zero failures**, ~10 minutes of runtime.
Against three previous attempts that produced nothing usable.

---

## 3. Gate results

| Gate | Criterion | Result |
|---|---|---|
| 1 — collection | ≥42 of 46 sessions | **PASS** — 68/68 |
| 2 — stint table | ≥600 stints, ≥25% retention | **PASS** — 1,726 stints, 44.3% |
| 3 — CWI Spearman | pre-registered thresholds | **Energy demoted**, ρ ≈ 0 |
| 4 — beat baseline | ≥15% holdout MAE | **FAIL** — −0.2% |
| 5 — physics plausibility | 3-7 g combined | **PASS** — 4.85 g |
| 6 — SHAP sanity | TyreLife/temp/compound top 8 | **FAIL on criterion**, see §6 |
| 7 — LSTM | Ljung-Box + 5% gain | **Not built**, see §7 |

### Gate 4 in context

| Model | Holdout MAE (s/lap) | R² |
|---|---|---|
| Mean predictor | 0.03539 | — |
| Linear baseline | 0.03337 | +0.044 |
| **XGBoost (tuned)** | **0.03345** | **+0.097** |

The tuned tree is **level with** the linear baseline, not behind it. That is
itself the finding: once the target's construction bias is removed, the
circuit-severity-to-degradation relationship is close to linear, so gradient
boosting's extra flexibility buys nothing on average error. It does hold a
better R² (+0.097 against +0.044), so it handles the tails better.

**The 15% threshold was fixed before anyone measured the headroom.** The
oracle ceiling — predicting each stint's own event-and-compound mean, which
requires knowing the answer in advance — is **38.0%**. Clearing 15% would mean
capturing 40% of everything achievable. The model captures 14.4%.

Just under half of degradation variance lies between events; the rest
separates stints at the same race, where no event-level feature can reach.

---

## 4. Assumptions the data overturned

The central lesson of the project. Each of these was reasoned carefully and
was wrong.

### 4.1 The `|r| ≥ 0.3` filter was a magnitude filter in disguise

From the research-backed compass document, with citations and a pre-registered
rule. Measured against the collected data:

```
corr(|r|, |slope|) = +0.53      corr(|r|, stderr) = -0.20
```

It tracked *how large* the degradation is roughly three times more strongly
than *how well it was measured*. Pass rate by slope size:

| Slope magnitude | Pass rate |
|---|---|
| 0.00-0.02 | **29.5%** |
| 0.02-0.05 | 86.2% |
| 0.05-0.10 | 98.7% |
| 0.10-0.20 | 99.8% |

A genuinely flat stint has low correlation *by construction*, however
precisely it is measured. Monaco was the clearest casualty: its stints are
measured **more** precisely than average (slope standard error 0.0062 against
0.0152 elsewhere) and were being discarded for being flat.

**Replaced with** acceptance by slope standard error (≤ 0.02 s/lap), which
keeps a well-measured flat stint and rejects a noisy one.

### 4.2 The zero lower bound compounded it

Monaco's true median slope is **−0.008 s/lap**. Excluding all negative slopes
therefore deleted whatever survived the `|r|` filter. A circuit with no
measurable degradation scatters either side of zero, and truncating at zero
keeps only the positive half of that noise — biasing the estimate upward
exactly where a strategist most needs to know degradation is negligible.

**Replaced with** a physical bound of −0.05 s/lap, about one second across a
20-lap stint. Only 12 stints now fall below it, against 219 under a zero bound.

Effects of 4.1 and 4.2 together, all in the same direction:

| | Before | After |
|---|---|---|
| Between-event variance | 32.3% | **49.7%** |
| Oracle ceiling | 27.4% | **38.0%** |
| XGBoost holdout R² | +0.047 | **+0.097** |
| Linear baseline R² | −0.053 | **+0.044** |
| Cross-season circuit r (23 v 24) | 0.18 | **0.64** |

### 4.3 The model was overfitting, not underfitting

With R² at −0.155 the intuitive reading is too little capacity. The opposite
was true: at 400 trees and depth 4 it scored worse than predicting the
training mean, while the *linear* baseline did not — the signature of excess
capacity against a weak signal. The tuned model is **71 trees** with
`min_child_weight` 46.

### 4.4 Driver identity and stint context add nothing

Sound reasoning: half the variance is within-event, and stint number, fuel
load and driver skill are knowable in advance. Held-out ablation, each set
tuned separately:

| Feature set | MAE | R² |
|---|---|---|
| core 9 | 0.03455 | +0.023 |
| core + context | 0.03399 | +0.066 |
| **core + Team** | **0.03345** | **+0.097** |
| core + context + Team | 0.03380 | +0.084 |
| core + context + Team + Driver | 0.03350 | +0.096 |

Team alone was already correct. Recorded in `features.REJECTED`.

**A near-miss worth recording.** With the failing features added, the headline
gate appeared to *improve*, from −0.2% to +2.5%. That was not the model getting
better — XGBoost was flat at R² 0.096 against 0.097 — but the linear baseline
getting *worse* under 25 driver one-hots. **A relative gate can be improved by
weakening its reference.** The ablation caught it only because it compared
absolute holdout error as well.

### 4.5 Position units

FastF1 reports `X`, `Y`, `Z` in **1/10 m** (`fastf1/core.py:60-62`), not
metres. Curvature scales as 1/L, so raw coordinates give lateral acceleration
ten times too low — a plausible-looking number that is simply wrong. The
compass reference code documents metres and does not correct for it.

Confirmed by Gate 5: with the correction, lateral acceleration median is
**4.8 g**. Without it, 0.48 g.

---

## 5. Decisions and their justifications

| Decision | Justification, stated independently of any metric |
|---|---|
| Fuel correction at 0.032 s/kg | Without it the target measures fuel burn. Practitioner band 0.025-0.040 |
| Exclude slopes below −0.05 s/lap | A tyre does not recover with age; that is track evolution |
| Accept by slope standard error | We want the rate known accurately, *including* knowing accurately that it is near zero |
| **Not** raising the `\|r\|` threshold | Produced a better R², but is selection on the outcome. Rejected despite helping |
| Exclude `StintLength`, `n_laps` | Reverse-causal; `StintLength` is what the simulator exists to choose |
| Team in, Driver out | Ablation, not intuition |
| Prefer observed over model in the simulator | Per-circuit degradation correlates 0.64-0.74 across seasons; the model captures 14.4% of headroom |
| Floor degradation only for simulation | The measured value is reported unchanged; an unbounded negative rate would reward infinite stints |
| Drop the "GPU-accelerated" framing | Every step runs on CPU in minutes; `device='cuda'` on 2,000 rows is slower than `hist` on CPU |

The fourth row is the one that matters most. Raising `|r|` improved the score
and was rejected anyway, because the justification would have been "it helped."

---

## 6. Gate 6: the criterion was wrong, not the model

Gate 6 requires `TyreLife_start` in the SHAP top 8. That feature is correctly
uninformative here, because **80% of stints start on fresh tyres** — it is
near-constant. Circuit severity, abrasion and compound do lead the ranking,
which is the substance the gate was meant to check.

Recorded as a failure rather than quietly redefined.

---

## 7. Gate 7: the LSTM was not built

The unit of analysis is one row per stint, so there is no sequence axis left
in the target. Clearing a 5% gain with a recurrent model on ~2,000 tabular
rows against tuned gradient boosting is, by the compass document's own
citation (grinsztajn_2022), the regime where trees win.

Consequence: the RTX 4050 contributes nothing to the critical path, and the
project should not be described as GPU-accelerated.

---

## 8. What is not modelled

- **Wet and intermediate running.** Excluded from the target fit; requests
  return a `dry_model_only` flag rather than a fabricated number.
- **The non-linear degradation cliff.** Degradation is linear in tyre age
  here, so projections beyond observed stint lengths are flagged as
  extrapolation.
- **Anything outside 2022-2024.** Flagged, not extrapolated.
- **Within-event variation.** Roughly half the variance — traffic, fuel
  saving, driver management — is unreachable from circuit-level features.
- **Compound absolute hardness for most events.** Pirelli C-number
  nominations are confirmed for only a minority of rounds; `compound_ordinal`
  is NaN elsewhere and `compound_relative` carries the load.

---

## 9. Reproduction

```
scripts/probe_api_cost.py         measure API cost before spending budget
scripts/run_collect_laps.py       collect races, one parquet per session
scripts/run_build_dataset.py      validate, fuel-correct, build the stint table
scripts/run_cwi_study.py          the pre-registered Spearman target study
scripts/run_diagnostics.py        variance decomposition and the oracle ceiling
scripts/run_collect_telemetry.py  telemetry pilot, per-lap physics aggregates
scripts/run_telemetry_study.py    curvature proxy against the speed-trap one
scripts/run_train.py              train, evaluate, report the gates
scripts/run_export_ui.py          export model parameters for the front end
```

Collection is resumable: re-run the same command. Tuned hyperparameters
persist to `config/tuned_params.json` and reload by default, guarded against
a stale feature set.

Reports land in `artifacts/reports/`.
