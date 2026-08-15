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

### 4.5 The telemetry-free proxy was inadequate, and the real one is mis-named

Gate 3 demoted the energy component at ρ ≈ 0, but that used the speed-trap
stand-in for `mean(v^2)`. Re-run on 10 races of true curvature telemetry
(`run_telemetry_study.py`, 209 stints):

| Proxy | Spearman with deg_rate | p |
|---|---|---|
| **aero_load_proxy** (true) | **−0.596** | 1.9e-21 |
| lat_accel_max | −0.270 | 7.6e-05 |
| combined_g_max | −0.269 | 8.0e-05 |
| lat_accel_mean | −0.181 | 0.0088 |
| speed-trap stand-in | ≈ 0.00 | — |

Two conclusions, and the second matters more.

**The Stage-0 shortcut was wrong.** Four speed-trap readings per lap do not
approximate `mean(v^2)` well enough to test the hypothesis. The cheap proxy
found nothing where the real one finds a strong relationship.

**The relationship is strongly NEGATIVE, which falsifies the hypothesis as
stated.** The pre-registered physics (archard_1953: wear follows frictional
work, so more energy means more wear) predicts a *positive* correlation. The
data says faster circuits degrade *less*:

| Circuit | mean speed | mean deg_rate |
|---|---|---|
| Barcelona | 196 km/h | **+0.117** |
| Bahrain | 203 km/h | +0.165 |
| Miami | 214 km/h | +0.008 |
| Jeddah | 230 km/h | **+0.019** |

Circuit-level ρ = −0.717 (n=9, p=0.03).

`mean(v^2)` is not measuring contact-patch work. It is measuring **straight-line
fraction**: a high lap-average speed means a lot of the lap is spent not
cornering, and cornering is what works the tyre. The compass document names it
`aero_load_proxy` and treats it as an energy term; on this evidence that
naming is misleading.

**It is a track property, not a stint property.** 92.6% of its variance lies
between circuits. So it is knowable in advance and legitimately usable by the
simulator — but it is not new physics. Compare against the hand-coded traits:

| Feature | Spearman with deg_rate |
|---|---|
| aero_load_proxy | −0.596 |
| composite (hand-assigned 1-3) | +0.578 |
| abrasion (hand-assigned 1-3) | +0.522 |

Comparable strength, opposite sign: they measure the same underlying thing
from opposite directions. The honest reading of the Tier C gate result is that
physics aggregates are a **better-measured circuit encoding** — a continuous
quantity derived from telemetry replacing a subjective 1-3 rating — rather
than a new physical mechanism.

**Tier C passes the gate at +13.47%** (threshold 3%), but on grouped CV over
209 stints from 9 sessions of 2022 only, with no chronological holdout
available on the pilot subset. Suggestive, not established. Confirming it
requires telemetry across all 68 sessions and a proper 2022-2023 / 2024 test.

### 4.6 Position units

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

## 9. Serving

### 9.1 The simulator was doing 30 times more work than it needed to

A prediction searched a few thousand candidate strategies and projected a
full lap-by-lap time trace for every one, then read exactly one of them.

Race time splits cleanly into a part every strategy over the same distance
pays alike and a part the strategy controls. Base pace contributes
`race_laps x base_lap_s`. The fuel penalty is a function of the absolute lap
number, so summed over a whole race it is identical however the stints fall:
with `burn = initial / race_laps` it comes to `penalty x initial x
(race_laps + 1) / 2`. What remains is per stint, `n x offset + deg x n(n-1)/2`,
plus the pit loss. Quadratic in stint length, which is why stopping can be
worth twenty-five seconds.

Ranking on that is exact rather than approximate, so nothing is traded away.
Two further changes follow from it: candidates slower than
`best + pit_window_slack_s` are discarded unless they are the first of their
compound sequence, and the pace and compound reference frames are indexed
into dictionaries once at load rather than filtered per request.

| Circuit | Laps | Searched | Kept | Before | After | Speed-up |
|---|---|---|---|---|---|---|
| Silverstone | 52 | 1,830 | 32 | 149.42 ms | 4.18 ms | 35.8x |
| Monaco | 78 | 1,830 | 30 | 213.22 ms | 5.29 ms | 40.3x |
| Monza | 53 | 1,830 | 32 | 189.16 ms | 6.47 ms | 29.2x |
| Spa | 44 | 5,382 | 32 | 541.17 ms | 17.72 ms | 30.5x |
| Singapore | 62 | 2,484 | 32 | 295.27 ms | 9.30 ms | 31.7x |
| Jeddah | 50 | 1,620 | 40 | 168.28 ms | 6.07 ms | 27.7x |
| Bahrain | 57 | 2,148 | 51 | 253.29 ms | 7.93 ms | 31.9x |
| Suzuka | 53 | 1,830 | 48 | 198.03 ms | 6.41 ms | 30.9x |
| Barcelona | 66 | 2,538 | 61 | 317.44 ms | 9.61 ms | 33.0x |
| Hungaroring | 70 | 2,928 | 78 | 378.64 ms | 10.72 ms | 35.3x |

**Around 35x over the ten circuits combined**, and the same winning strategy,
the same pit laps and the same race time to within 1e-6 s everywhere. Two
runs of the same comparison gave 32.3x and 35.4x, which is the run-to-run
spread on this machine and the reason the figure is quoted loosely.

#### Is that a real improvement, or just a bad first attempt?

The honest position is that the baseline is **the obvious implementation**:
build each candidate strategy, simulate it lap by lap, keep the best. It is
what the design called for and what most people would write. But "35x faster
than my own earlier code" is not evidence of anything on its own, since the
baseline could simply have been careless.

The distinction that matters is whether this is a complexity change or a
constant-factor tidy-up, and that is testable. The old cost goes as
`candidates x race_laps`, because it projects a lap time for every lap of
every candidate. The new cost goes as `candidates`, plus one materialised
trace. If that is right, **the speed-up must grow in proportion to race
length**; a constant-factor improvement would show a flat ratio instead.

Race length swept from 30 to 100 laps, everything else held fixed:

| Race laps | Candidates | Before | After | Speed-up | Old ns per candidate-lap | New ns per candidate |
|---|---|---|---|---|---|---|
| 30 | 2,226 | 49.35 ms | 2.47 ms | 20.0x | 738.9 | 1109.2 |
| 40 | 3,726 | 106.20 ms | 3.45 ms | 30.8x | 712.6 | 925.4 |
| 50 | 1,620 | 50.31 ms | 1.40 ms | 36.1x | 621.2 | 861.4 |
| 60 | 2,226 | 76.72 ms | 1.84 ms | 41.6x | 574.4 | 828.4 |
| 70 | 2,928 | 124.12 ms | 2.58 ms | 48.2x | 605.6 | 880.4 |
| 80 | 1,614 | 61.11 ms | 1.39 ms | 43.9x | 473.3 | 861.8 |
| 90 | 2,226 | 104.96 ms | 2.47 ms | 42.5x | 523.9 | 1109.3 |
| 100 | 1,620 | 78.17 ms | 1.49 ms | 52.3x | 482.5 | 922.2 |

Read the last two columns, not the middle ones. **The new implementation's
cost per candidate does not depend on race length at all** (median 901 ns,
spread 0.31, no trend): a 100-lap race costs it what a 30-lap race costs.
The old implementation's cost per *candidate-lap* is roughly constant
(median 590 ns, spread 0.45), which is what `candidates x race_laps` means.

Race length x3.33 across the sweep produced speed-up x2.61, and the ratio
climbs monotonically apart from run-to-run noise. It falls a little short of
the full x3.33 because the old code also carried a fixed per-candidate cost
in object construction, so its normalised column drifts down at long races
rather than staying flat. That is a real effect and it is visible in the
table rather than smoothed away.

Candidate counts do not rise smoothly with race length, because the search
grid steps by `race_laps // 25`. They are measured rather than assumed for
exactly that reason.

Note that latency tracks the number of candidates, not the race length. The
search grid steps by `race_laps // 25`, so a 44-lap race is searched at
one-lap resolution and a 78-lap race at three: **Spa is the most expensive
circuit despite being the shortest race**. That is an accident of integer
division rather than a decision, and it is left alone because changing the
grid would change published pit windows.

#### Method

This is a laptop whose clock varies by a factor of three between a cold turbo
burst and sustained load - the same unchanged benchmark recorded 1.32 ms and
6.13 ms per prediction twenty minutes apart. Measuring the old code, then the
new code, then subtracting, would have measured the thermal state as much as
the change.
`scripts/run_simulator_ab.py` therefore runs both implementations
**alternately in one process**, reading the old one out of git so it cannot
drift from what was replaced. Drift then hits both arms equally and cancels
in the ratio. The absolute milliseconds still depend on machine state and
should be read as a spread, not a specification.

`tests/test_strategy.py` checks the optimised path against an exhaustive
lap-by-lap brute force at five race lengths: same winner, and every
near-optimal strategy retained, so a pit window cannot be silently clipped by
the pruning rather than by its own criterion.

### 9.2 The HTTP service

`src/service/` puts the model behind an API. The model loads once at
start-up; every request is pure computation over that shared state, writing
nothing and caching nothing, so a worker's hundred-thousandth request behaves
like its first. The one piece of shared state that is not obviously safe is
the XGBoost booster, which takes a narrow lock during inference; most
requests never reach it, because a circuit with enough observed stints uses
the empirical rate.

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness |
| `GET /ready` | Readiness, separate because loading takes seconds |
| `GET /meta` | Model card, stating the failed gate outright |
| `GET /circuits` | What `/predict` accepts, and the evidence behind each |
| `POST /predict` | Strategy prediction |

Validation separates two things that are easy to conflate. A request the
simulator cannot represent is a **400** naming the field and, where a short
list exists, the allowed values. A request outside the model's evidence but
still answerable is a **200 carrying a flag**: an out-of-scope season, a wet
race, a circuit-compound pair with too few stints. The response already
reports how much of the answer rests on observed data; refusing to answer
would hide that judgement rather than expose it.

### 9.3 Throughput and latency

`scripts/run_load_test.py`, closed-loop, requests drawn from all ten circuits
because per-request cost varies threefold between them. Load generator and
service share the machine, so these are a floor.

One replica, waitress with 8 threads:

| Concurrency | req/s | p50 | p95 | p99 | Errors |
|---|---|---|---|---|---|
| 1 | 66.3 | 13.23 ms | 24.63 ms | 49.81 ms | 0 |
| 2 | 71.3 | 25.25 ms | 46.58 ms | 113.28 ms | 0 |
| 4 | 66.2 | 54.44 ms | 124.50 ms | 183.77 ms | 0 |
| 8 | 67.5 | 125.87 ms | 259.66 ms | 331.93 ms | 0 |
| 16 | 59.8 | 254.11 ms | 442.74 ms | 544.56 ms | 0 |

**Throughput is flat from concurrency 1 to 16 while latency rises linearly.**
That is the signature of a single-process GIL ceiling, not of a resource
running out: a prediction is CPU-bound Python, so one process saturates one
core no matter how many threads waitress is given, and extra concurrency buys
queueing rather than work. Threads beyond about two are not useful here.

Scaling is therefore by process. Three replicas:

| Replicas | Peak req/s | p95 at peak |
|---|---|---|
| 1 | 71.3 | 46.58 ms |
| 3 | 138.1 | 398.30 ms |

1.9x rather than 3x, on a busy 8-core laptop that was also running the load
generator and an assortment of unrelated desktop software. The replicas do
not contend with each other - they share nothing, each loading its own copy
of the model - so the sublinearity is the host's. Two load generators driving
the three replicas together reached the same aggregate as one, which rules
out the client as the limit.

One expectation that measurement killed: an access log line per prediction
was assumed to be a throughput tax, since `setup_logging` attaches a file
handler that writes on the request thread. Measured both ways, 67.1 against
68.2 requests/s - inside the run-to-run spread. It is off by default because
a reverse proxy records the same thing, not because it is expensive.

### 9.4 Container

`Dockerfile` builds a serving-only image, roughly 5.4 MB of artefacts on
`python:3.12-slim`. Collection and training stay on the host.

`requirements-service.txt` is pinned exactly, unlike `requirements.txt` which
uses lower bounds. Resolving those bounds freshly produced pandas 3.0.5,
numpy 2.5.2 and xgboost 3.4.1 against the 2.3.3 / 2.3.5 / 3.1.2 the model was
trained and measured on. It loaded and returned an identical strategy, but
xgboost warned that a model serialised by an older version should be
re-exported first, and that warning becomes an error eventually. A joblib
pickle is only guaranteed to load under the versions that wrote it.

**The image is not built or run here.** Docker Desktop would not start on
this machine, and nothing that has not been executed is reported as working.
What was verified instead: the exact file set the Dockerfile copies was
assembled into a clean directory, a fresh virtual environment was built from
`requirements-service.txt` alone, and the service was started from that and
returned byte-identical predictions with no xgboost warning. That covers the
file list and the dependency list, which are the two things most likely to be
wrong. It does not cover the base image, the healthcheck, the compose file or
nginx, and those should be treated as unverified.

---

## 10. Reproduction

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
scripts/run_service.py            serve the model over HTTP
scripts/run_predict_bench.py      per-circuit prediction latency, in process
scripts/run_simulator_ab.py       interleaved A/B against the old simulator
scripts/run_load_test.py          throughput and tail latency under load
```

Collection is resumable: re-run the same command. Tuned hyperparameters
persist to `config/tuned_params.json` and reload by default, guarded against
a stale feature set.

Reports land in `artifacts/reports/`.
