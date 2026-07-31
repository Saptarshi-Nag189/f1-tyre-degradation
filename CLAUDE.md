# CLAUDE.md — F1 Tyre Degradation Modelling

## Golden rules

- British English in comments, docstrings and prose. No emoji. No em dashes.
- Use the `logging` module, never `print`, outside of `scripts/` entry points.
- Every magic number lives in `config/*.yaml`. No hard-coded constants in `src/`.
- Type hints and docstrings, with units, on every public function.
- Handle empty or malformed frames defensively: return empty, log, never crash a
  season run on one bad session.

## Project-specific rules, each earned the hard way

- **Never write a monolithic pickle.** One parquet per session, written to a
  temporary file then `os.replace`d. The superseded pipeline re-serialised the
  whole corpus after every event and left 5 GB unreadable.
- **Never delete `cache/fastf1`.** Cache hits bypass the rate limiter entirely,
  so a warm cache makes every re-run free. Deleting it is what made three
  previous collection attempts unrecoverable.
- **Never modify the venv's `torch`, `numpy`, `pandas` or `fastf1` versions.**
  `requirements.lock.txt` records the working CUDA build. The compass
  document's `requirements.txt` targets Python 3.11 with numpy 1.26 and would
  break it; treat that file as a bill of materials, never as an install command.
- **Never coerce an unrecognised compound.** Route it to an explicit category
  and count it. The old `fillna('MEDIUM')` silently corrupted the most
  important tyre categorical.
- **Detect rate limiting by exception type**, never by message text. FastF1
  swallows `RateLimitExceededError` internally and surfaces
  `DataNotLoadedError`, so a string match can never fire.

## Data contract

- Join on `DriverNumber` (string), never the three-letter abbreviation.
- Key events by `RoundNumber`, never `EventName`.
- `GridPosition == 0.0` means a pit-lane start: flag it, do not overwrite.
- Weather imputation must set `weather_imputed = True`.
- Representative pace uses the `is_clean_lap` mask only.
- fastf1 position `X`, `Y`, `Z` are in **1/10 m**, not metres. Curvature scales
  as 1/L, so raw coordinates give lateral acceleration 10x too low.

## Modelling order

1. Collect → validate → assemble → target → model. Never reorder.
2. Fuel correction: `fuel_remaining(N) = 110 - (N-1) * burn_per_lap`, penalty
   0.032 s/kg. Without it the target measures fuel burn, not tyre wear.
3. Features must be knowable **before** the stint runs. `StintLength` and
   `n_laps` are reverse-causal and are forbidden.
4. Split chronologically on whole events. An event must never straddle a fold
   boundary.
5. Train 2022-2023, hold out 2024. Never shuffle across seasons.
6. Report gates against the measured oracle ceiling, not in isolation.

## Honesty rules

The project's original documentation described a physics-informed loss and
thermal/mechanical decomposition heads that did not exist in the code, and
reported metrics from a configuration that had since changed. That gap is the
reason for this rebuild.

- Do not describe the project as GPU-accelerated. Every step runs on CPU in
  minutes, and `device='cuda'` on a 2,000-row matrix is slower than
  `tree_method='hist'` on CPU.
- Do not describe Tier A as physics-informed. It is physics-*motivated*
  through the fuel correction and the circuit traits.
- Report gate failures as failures. Do not adopt a filter because it moved a
  metric; adopt it only if you can state why it is right without reference to
  the score.
