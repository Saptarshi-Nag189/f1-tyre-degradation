# F1 Tyre Performance Degradation Modelling
## A Research-Backed, Physics-Informed Technical Document with Complete Code Implementation

**Author context:** AI/ML engineering team lead, C-DAC. **Data era:** 2022-2024 (single ground-effect regulation era). **Language:** British English throughout. **Stack:** Python 3.11, fastf1, pandas, numpy, pyarrow, scikit-learn, xgboost, optuna, shap, scipy, matplotlib, pytest.

---

## TL;DR

- **The model is buildable and scientifically defensible from public fastf1 data.** The Composite Wear Index (CWI) target combines a fuel-corrected lap-time degradation slope with a curvature-based energy-dissipation proxy, gated by an empirical Spearman validation study. This is grounded in Persson's multiscale rubber-friction theory and the Archard/Schallamach result that wear rate is proportional to frictional work dissipated at the contact patch, so an energy proxy is a physically justified target component.
- **Excluding the Pacejka Magic Formula from the critical path is the correct decision.** Its coefficients are load-dependent, proprietary and scarce in public literature, and it has documented low-speed instability. Curvature-based proxies (lateral acceleration from track X,Y, braking power from speed derivative, combined G, aero load as mean speed squared) capture the same physics from telemetry without unavailable parameters.
- **XGBoost over deep learning, with strict chronological validation, is the right modelling spine.** Tree-based models remain state of the art on medium-sized tabular data (Grinsztajn et al., NeurIPS 2022); TimeSeriesSplit and a 2022-2023 train / 2024 hold-out prevent temporal leakage (Bergmeir et al., 2018). The LSTM-XGBoost ensemble stays conditional on a Ljung-Box test for residual autocorrelation (p < 0.05) and a 5% MAE improvement gate.

---

## Key Findings

1. **Grip is viscoelastic, wear is energetic.** Rubber friction splits into adhesion and hysteresis components. Persson's theory (J. Chem. Phys. 115(8):3840-3861, 2001) links friction to sliding velocity by integrating viscoelastic energy dissipation over the full roughness spectrum; the hysteresis contribution peaks in the transition region between rubbery (low frequency) and glassy (high frequency) regimes. Wear literature (Archard 1953; Schallamach; recent reviews in Tribology Letters 2025) establishes wear rate is proportional to frictional power dissipated, which depends on local sliding speed, contact pressure and dynamic friction coefficient. This is the physical warrant for the energy-dissipation component of the CWI.

2. **Fuel correction of ~0.03 s/kg is well supported.** Per F1Briefing ("Fuel Load vs. Lap Time"): "A full fuel load of 110 kg (242 lbs) significantly slows lap times, with each additional kilogram adding about 0.03 seconds per lap." Industry practitioners cite 0.025-0.040 s/kg depending on circuit braking demand. The agreed 0.032 s/kg sits squarely inside this band. Fuel burned on lap N is (N-1) times burn-per-lap because the car completes lap N carrying the fuel it had at the *start* of that lap.

3. **Pirelli compound windows and allocations are partially recoverable.** Public sources (Red Bull, thef1db) give approximate operating windows; note a source discrepancy (Red Bull states soft C5 ~85-115 C and hard C1 ~110-140 C, while thef1db states soft ~90-110 C and hard ~100-120 C with a 15-20 C window width). Weekend-by-weekend C-compound nominations for 2022-2024 are available from Pirelli press releases and are embedded in config below, with unknowns marked UNK.

4. **Track severity has two distinct axes: abrasion and energy.** These diverge. Bahrain scores 5/5 abrasiveness ("a high percentage of granite within the asphalt") but only medium lateral load. Silverstone and Suzuka are maximum-energy (5/5 lateral) but low-abrasion surfaces. This two-axis reality is encoded in the track-traits config.

5. **The 2022 regulation baseline is fixed and citable.** FIA 2022 Technical Regulations Article 4.1: "The mass of the car, without fuel, must not be less than 798kg, at all times during the competition" (a +3 kg increase from 795 kg confirmed in issue 10, RaceFans, 17 March 2022). Fuel capacity 110 kg; 18-inch wheels from 2022; pit-lane time loss typically 20-25 s (Imola an outlier above 30 s; COTA the lowest).

---

## Details

### 1. Physics Foundations (with citations)

**1.1 Grip generation.** The contact patch generates force through two mechanisms. **Adhesion** arises from molecular bonding at the real area of contact and dominates at low sliding speed; **hysteresis** arises from viscoelastic energy dissipation as tread rubber deforms around road asperities and dominates at the sliding speeds relevant to tyre dynamics. Persson (2001) provides the multiscale framework: friction is obtained by integrating viscoelastic losses over the surface roughness power spectral density. Grosch's master curves first separated the two components experimentally. The practical consequence for F1: peak grip requires the compound to be in a temperature window where the loss modulus is high enough for hysteresis but the rubber is not so hot that it reverts.

**1.2 Degradation mechanisms.**
- **Abrasion:** normal wear as rubber is worn away, leaving uniform ridges. Predictable and roughly linear, which is what strategists model.
- **Graining:** rubber shears from the surface and rolls into grains (like an eraser dragged across paper), typically linked to low-temperature sliding and understeer; can "clean up" if the driver backs off and the tyre returns to its window.
- **Blistering:** overheating causes the rubber to boil internally, forming bubbles that burst and leave craters. Caused by exceeding the thermal limit.
- **Thermal degradation:** molecular breakdown of the compound above its optimal range, producing permanent grip loss and highly non-linear "cliff" behaviour.

**1.3 Thermal windows.** Public sources disagree on exact numbers. Red Bull ("Everything you need to know about F1 tyres") states soft (C5) 85-115 C and hardest (C1) 110-140 C. thef1db states soft ~90-110 C and hard ~100-120 C, with the working window "often just 15 to 20 degrees Celsius." The report and config treat these as approximate and flag the discrepancy; the model uses measured `TrackTemp`/`AirTemp` from fastf1 rather than hard-coding windows.

**1.4 Why data-driven physics-informed rather than first-principles.** The energy dissipation at the contact patch is proportional to force times slip velocity, both of which require the very tyre-force model (Pacejka) whose coefficients are unavailable. We therefore compute *proxies* for the physical drivers (lateral acceleration, braking power, combined G, aero load) from telemetry and let a gradient-boosted model learn the mapping to observed degradation, respecting the physics through feature construction rather than through an unidentifiable closed-form model. This mirrors published F1 work: Sulsky/Mercedes-AMG PETRONAS ("Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy," arXiv:2501.04067) forecast tyre energy with XGBoost and LSTM and note the industry standard is simple linear degradation models.

### 2. Literature Review Justifying Each Design Decision

**2.1 Why no Pacejka on the critical path.** The Magic Formula (Bakker, Nyborg, Pacejka 1986; Pacejka, *Tire and Vehicle Dynamics*, Elsevier) is semi-empirical: its coefficients are fitted to rig data and are load-dependent (the dynamic/normalised-load form df_z = (F_z - F_z0)/F_z0 scales the shape/peak coefficients). Public coefficient sets are scarce and heavily protected IP (the Racer reference notes "the data is a bit lacking; some coefficients for load sensitivity are missing"). The formula also has a documented low-speed singularity requiring a VXLOW smoothing patch. For F1 tyres, with proprietary construction and no public rig data, identifying these coefficients is infeasible, so any Pacejka path would rest on fabricated numbers. Excluding it and using curvature proxies is both honest and reproducible.

**2.2 Why XGBoost.** Chen & Guestrin, "XGBoost: A Scalable Tree Boosting System," KDD 2016, pp. 785-794 (arXiv:1603.02754), provides sparsity-aware split finding and regularisation that resist overfitting. Grinsztajn, Oyallon & Varoquaux, "Why do tree-based models still outperform deep learning on typical tabular data?", NeurIPS 2022 (Datasets and Benchmarks track), vol. 35, pp. 507-520, show tree ensembles remain state of the art on medium-sized (~10k sample) tabular data due to inductive biases robust to uninformative features and irregular target functions. Our per-stint dataset is exactly this regime.

**2.3 Why Savitzky-Golay.** Savitzky & Golay, "Smoothing and Differentiation of Data by Simplified Least Squares Procedures," *Analytical Chemistry* 36(8):1627-1639, 1964 (DOI 10.1021/ac60214a047). The method fits a local polynomial and preserves peaks, minima and derivatives that moving averages flatten, and yields analytic derivative estimators. This is ideal for differentiating noisy 0.1 s telemetry twice (speed to acceleration to jerk) with controlled noise amplification. Window lengths {7, 11, 15, 21} are tested empirically; note the original paper's tables contained typographical errors corrected by Steinier, Termonia & Deltour (*Anal. Chem.* 44(11):1906-1909), so we rely on scipy's implementation.

**2.4 Why the CWI validation gate.** Because the physical claim (energy proportional to wear) must hold *in the data*, we test it before trusting it. A per-stint Spearman correlation between the energy proxy and the lap-time degradation slope decides the target: correlation > 0.4 keeps energy at weight 0.5; 0.2-0.4 down-weights it (lap-time weight 0.7); < 0.2 demotes energy to a feature only. This is a falsifiable, pre-registered decision rule rather than an assumption.

**2.5 Why chronological splits.** Bergmeir, Hyndman & Koo, "A note on the validity of cross-validation for evaluating autoregressive time series prediction," *Computational Statistics & Data Analysis* 120:70-83 (2018), and Bergmeir & Benitez (2012) establish that standard k-fold leaks future information when observations are serially dependent. We train on 2022-2023 and hold out 2024, with Optuna nested inside TimeSeriesSplit(n_splits=4), so validation windows are always temporally behind training windows.

### 3. Phase-by-Phase Plan and Complete Code (Phases 1-4), Skeletons (Phases 5-7)

The consolidated document (Deliverable A) and the standalone files (Deliverable B) follow. Redundancy between them is intended.

---

## DELIVERABLE A + B: FULL PROJECT ARTIFACTS

### `requirements.txt`
```
python-version == 3.11
fastf1==3.4.*
pandas==2.2.*
numpy==1.26.*
pyarrow==16.*
scikit-learn==1.5.*
xgboost==2.1.*
optuna==3.6.*
shap==0.46.*
scipy==1.13.*
matplotlib==3.9.*
pyyaml==6.*
pytest==8.*
statsmodels==0.14.*   # Ljung-Box test (Phase 6)
tensorflow==2.16.*    # conditional Phase 6 LSTM only
```

### `config/settings.yaml`
```yaml
# Global configuration. All magic numbers live here.
# Sources cited inline; see references section of the main document.

project:
  seasons: [2022, 2023, 2024]
  session_type: "R"          # race sessions only
  british_english: true

paths:
  fastf1_cache: "./cache/fastf1"
  raw_parquet: "./data/raw"
  processed_parquet: "./data/processed"
  models: "./artifacts/models"
  logs: "./logs"

fuel_correction:
  # F1Briefing: "each additional kilogram adding about 0.03 seconds per lap".
  # Agreed value 0.032 s/kg sits inside the practitioner band 0.025-0.040.
  penalty_s_per_kg: 0.032
  initial_fuel_kg: 110.0      # FIA max fuel load
  # Fuel burned on lap N = (N-1) * burn_per_lap (car runs lap N on start-of-lap fuel).
  # burn_per_lap derived at runtime: initial_fuel_kg / scheduled_race_laps.

car:
  # FIA 2022 Technical Regulations Art. 4.1: minimum mass without fuel 798 kg.
  min_mass_kg: 798.0
  wheel_diameter_in: 18       # 18-inch wheels from 2022
  # Tyre dimensions 305/720-18 front, 405/720-18 rear (Pirelli 18-inch era).

physics:
  # Savitzky-Golay derivative estimation (Savitzky & Golay 1964).
  savgol_polyorder: 2
  savgol_windows: [7, 11, 15, 21]   # tested empirically; odd lengths required
  gravity_ms2: 9.80665
  # aero load proxy uses mean(speed^2); speed in m/s.

strategy:
  pit_loss_s: 25.0            # typical pit-lane time loss 20-25 s; 25 s agreed default

clean_lap:
  # is_clean_lap mask parameters
  stint_median_multiplier: 1.07   # exclude laps > 107% of stint median
  track_status_ok: "1"            # TrackStatus == '1' means green/all-clear

target:
  # CWI validation decision rule (Spearman energy vs degradation slope)
  spearman_keep_threshold: 0.4    # energy stays, weight 0.5
  spearman_downweight_low: 0.2    # 0.2-0.4 -> laptime weight 0.7
  energy_weight_full: 0.5
  laptime_weight_downweight: 0.7
  linregress_min_r: 0.3           # keep stint slope only if |r| > 0.3
  min_stint_laps: 5               # need enough clean laps to fit a slope

modelling:
  optuna_trials: 50
  ts_splits: 4
  train_seasons: [2022, 2023]
  holdout_season: 2024
  baseline_gate_pct: 15.0         # XGBoost must beat baseline MAE by >=15%
  tier_gate_pct: 3.0              # each feature tier kept if CV MAE improves >=3%
  ensemble_gate_pct: 5.0          # LSTM ensemble kept if MAE improves >=5%
  ljung_box_alpha: 0.05           # residual autocorrelation gate
```

### `config/compound_mapping.yaml`
```yaml
# Pirelli C-compound nominations per event (Hard / Medium / Soft).
# Sources: Pirelli press releases, formula1.com, racingnews365, f1technical.
# Compounds are RELATIVE per weekend: "hard" is the hardest of the nominated trio.
# Ordinal encoding used downstream: C0=0, C1=1, C2=2, C3=3, C4=4, C5=5.
# Entries not confirmed from a primary source are marked UNK.

encoding: {C0: 0, C1: 1, C2: 2, C3: 3, C4: 4, C5: 5}

2022:
  # Keyed by RoundNumber where known, else by event name.
  Bahrain:        {hard: C1, medium: C2, soft: C3}   # three hardest
  Saudi_Arabia:   {hard: UNK, medium: UNK, soft: UNK}
  Australia:      {hard: UNK, medium: UNK, soft: UNK}
  Emilia_Romagna: {hard: C2, medium: C3, soft: C4}   # confirmed 18-inch precedent
  Miami:          {hard: C2, medium: C3, soft: C4}
  Spain:          {hard: C1, medium: C2, soft: C3}   # harsh surface, hardest trio
  Monaco:         {hard: C3, medium: C4, soft: C5}   # softest trio
  France:         {hard: C2, medium: C3, soft: C4}
  Hungary:        {hard: C2, medium: C3, soft: C4}
  Belgium:        {hard: C2, medium: C3, soft: C4}
  Netherlands:    {hard: C1, medium: C2, soft: C3}   # banked corners, hardest trio
  Italy:          {hard: UNK, medium: UNK, soft: UNK}
  # Remaining 2022 rounds: UNK unless later confirmed from Pirelli releases.

2023:
  Bahrain:      {hard: C1, medium: C2, soft: C3}     # new-for-2023 C1
  Saudi_Arabia: {hard: C2, medium: C3, soft: C4}
  Australia:    {hard: C2, medium: C3, soft: C4}
  Monaco:       {hard: C3, medium: C4, soft: C5}
  Spain:        {hard: C1, medium: C2, soft: C3}
  Canada:       {hard: C3, medium: C4, soft: C5}
  Austria:      {hard: C3, medium: C4, soft: C5}
  Great_Britain:{hard: C1, medium: C2, soft: C3}     # high lateral energy
  Belgium:      {hard: C1, medium: C3, soft: C4}     # deliberate gap (skip C2)
  # Remaining 2023 rounds: UNK unless confirmed.

2024:
  Bahrain:      {hard: C1, medium: C2, soft: C3}
  Saudi_Arabia: {hard: C2, medium: C3, soft: C4}
  Australia:    {hard: C3, medium: C4, soft: C5}     # step softer vs 2023
  Japan:        {hard: C1, medium: C2, soft: C3}     # Suzuka, max lateral
  Emilia_Romagna:{hard: C3, medium: C4, soft: C5}    # step softer than 2022
  Monaco:       {hard: C3, medium: C4, soft: C5}
  Canada:       {hard: C3, medium: C4, soft: C5}
  Spain:        {hard: C1, medium: C2, soft: C3}
  # Remaining 2024 rounds: UNK unless confirmed.
```

### `config/track_traits.yaml`
```yaml
# Track abrasiveness/severity classifications, 1 (low) to 3 (high).
# Two DISTINCT axes: 'abrasion' (surface roughness/wear) and
# 'energy' (lateral/vertical load). They diverge; see per-track notes.
# Sources: Pirelli previews (press.pirelli.com), formula1.com, motorsport.com,
# f1technical.net, pitpass.com. Composite is a modelling convenience 1-3.
# Pirelli's public 5/5 lateral/severity group: Bahrain(abrasion), Silverstone,
# Suzuka, Barcelona, Spa, Zandvoort, Losail.

circuits:
  Bahrain:      {abrasion: 3, energy: 2, composite: 3, note: "5/5 abrasiveness, granite in asphalt; traction/braking led"}
  Jeddah:       {abrasion: 1, energy: 2, composite: 2, note: "not very abrasive, average roughness; considerable lateral but below Suzuka/Barcelona"}
  Australia:    {abrasion: 2, energy: 2, composite: 2, note: "2022 resurface, low grip 2/5, contained abrasion; overall stress 3/5"}
  Emilia_Romagna:{abrasion: 1, energy: 2, composite: 2, note: "low-side severity, not as low as a street circuit"}
  Miami:        {abrasion: 1, energy: 2, composite: 2, note: "very smooth asphalt; degradation primarily thermal"}
  Spain:        {abrasion: 2, energy: 3, composite: 3, note: "toughest of year; fast long corners; thermal deg; 5/5 group"}
  Monaco:       {abrasion: 1, energy: 1, composite: 1, note: "slippery street asphalt, almost no degradation, lowest energy"}
  Azerbaijan:   {abrasion: 1, energy: 2, composite: 2, note: "low abrasion, contained lateral loads; high straight-line vertical load"}
  Canada:       {abrasion: 1, energy: 2, composite: 2, note: "not particularly abrasive; heavy braking, high longitudinal forces"}
  Great_Britain:{abrasion: 1, energy: 3, composite: 3, note: "HIGH energy 5/5 lateral, >5g; surface medium-low abrasion"}
  Austria:      {abrasion: 3, energy: 2, composite: 2, note: "old, high abrasiveness; low lateral; thermal deg dominates"}
  France:       {abrasion: 2, energy: 2, composite: 2, note: "2022 calendar only; historically gritty/abrasive; low confidence"}
  Hungary:      {abrasion: 2, energy: 1, composite: 2, note: "not particularly severe; overheating/traction limited"}
  Belgium:      {abrasion: 2, energy: 3, composite: 3, note: "highest lateral energy loads; resurfaced -> lower abrasion"}
  Netherlands:  {abrasion: 2, energy: 3, composite: 3, note: "banked T3/T14 high vertical+lateral load; hardest trio 2022-24; 5/5 group"}
  Italy:        {abrasion: 1, energy: 2, composite: 2, note: "Monza; degradation usually low; braking/traction stability key"}
  Singapore:    {abrasion: 1, energy: 2, composite: 2, note: "street; traction/braking dominant, low corner energy"}
  Japan:        {abrasion: 2, energy: 3, composite: 3, note: "Suzuka; max lateral 5/5; low-med abrasion surface"}
  United_States:{abrasion: 2, energy: 2, composite: 2, note: "COTA; partial resurface 2023; lateral>longitudinal; thermal deg"}
  Mexico:       {abrasion: 2, energy: 2, composite: 2, note: "altitude -> low downforce/energy; softest trio 2024"}
  Brazil:       {abrasion: 2, energy: 2, composite: 2, note: "Interlagos; med-low forces both axles; 2024 resurface smoother"}
  Las_Vegas:    {abrasion: 1, energy: 1, composite: 1, note: "low roughness, low corner severity; graining from cold not abrasion"}
  Qatar:        {abrasion: 3, energy: 3, composite: 3, note: "Losail; abrasive surface + energy comparable to Suzuka/Silverstone; hardest trio; 5/5 group"}
  Abu_Dhabi:    {abrasion: 2, energy: 2, composite: 2, note: "Yas Marina; low-med severity; softest trio"}
  China:        {abrasion: 2, energy: 2, composite: 2, note: "Shanghai; 2024 resurface smoother/less abrasive but graining-prone"}

climatological_defaults:
  # Fallback AirTemp/TrackTemp (deg C) when weather data missing; provenance-flagged.
  Bahrain:      {air: 27, track: 33}
  Qatar:        {air: 29, track: 34}
  Singapore:    {air: 30, track: 34}
  Las_Vegas:    {air: 16, track: 20}
  Great_Britain:{air: 18, track: 28}
  default:      {air: 22, track: 30}
```

### `config/references.yaml` (machine-readable citation index)
```yaml
persson_2001: "Persson, B.N.J. (2001). Theory of rubber friction and contact mechanics. J. Chem. Phys. 115(8):3840-3861. DOI 10.1063/1.1388626"
savitzky_golay_1964: "Savitzky, A. & Golay, M.J.E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures. Anal. Chem. 36(8):1627-1639. DOI 10.1021/ac60214a047"
chen_guestrin_2016: "Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16, pp. 785-794. arXiv:1603.02754"
grinsztajn_2022: "Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? NeurIPS 35:507-520"
lundberg_lee_2017: "Lundberg, S.M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS 30:4765-4774"
bergmeir_2018: "Bergmeir, C., Hyndman, R.J. & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. CSDA 120:70-83"
hochreiter_1997: "Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. Neural Computation 9(8):1735-1780. DOI 10.1162/neco.1997.9.8.1735"
ljung_box_1978: "Ljung, G.M. & Box, G.E.P. (1978). On a Measure of Lack of Fit in Time Series Models. Biometrika 65(2):297-303. DOI 10.1093/biomet/65.2.297"
archard_1953: "Archard, J.F. (1953). Contact and Rubbing of Flat Surfaces. J. Appl. Phys. 24(8):981-988"
```

---

### PHASE 1: DATA ACQUISITION

#### `src/acquisition/collector.py`
```python
"""Phase 1: Season data collection from fastf1 with parquet persistence.

Collects race-session lap and telemetry data for 2022-2024, persists to
parquet, and handles caching and error logging defensively.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import fastf1
import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_settings(path: str | Path = "config/settings.yaml") -> dict[str, Any]:
    """Load the global settings YAML.

    :param path: path to settings.yaml.
    :returns: parsed settings dictionary.
    """
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def init_cache(cache_dir: str | Path) -> None:
    """Enable the fastf1 on-disk cache (mandatory for reproducible runs).

    :param cache_dir: directory for the fastf1 cache.
    """
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    fastf1.Cache.enable_cache(str(cache_dir))
    logger.info("fastf1 cache enabled at %s", cache_dir)


def collect_session(year: int, round_number: int, session_type: str) -> pd.DataFrame | None:
    """Load a single race session and return its enriched laps frame.

    Laps are joined with weather and identified by DriverNumber (stable key).
    Returns None on any load failure so a season run can continue.

    :param year: season year.
    :param round_number: FIA round number (stable key, not event name).
    :param session_type: session code, e.g. 'R' for race.
    :returns: laps DataFrame with metadata columns, or None on failure.
    """
    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=True, telemetry=False, weather=True, messages=False)
    except Exception as exc:  # defensive: fastf1/network/data errors
        logger.error("Failed to load %s R%s %s: %s", year, round_number, session_type, exc)
        return None

    laps = session.laps
    if laps is None or laps.empty:
        logger.warning("No laps for %s R%s", year, round_number)
        return None

    laps = laps.copy()
    laps["Year"] = year
    laps["RoundNumber"] = round_number
    try:
        laps["EventName"] = session.event["EventName"]
    except Exception:
        laps["EventName"] = "UNKNOWN"

    # Attach weather via merge_asof on session time if available.
    weather = getattr(session, "weather_data", None)
    if weather is not None and not weather.empty and "Time" in laps.columns:
        try:
            laps = pd.merge_asof(
                laps.sort_values("Time"),
                weather.sort_values("Time")[["Time", "AirTemp", "TrackTemp",
                                             "Humidity", "Rainfall", "WindSpeed"]],
                on="Time", direction="nearest",
            )
            laps["weather_imputed"] = False
        except Exception as exc:
            logger.warning("Weather merge failed %s R%s: %s", year, round_number, exc)
            laps["weather_imputed"] = True
    else:
        laps["weather_imputed"] = True

    logger.info("Collected %s R%s: %d laps", year, round_number, len(laps))
    return laps


def collect_season(year: int, session_type: str, out_dir: str | Path,
                   max_rounds: int = 24) -> Path | None:
    """Collect all race sessions for a season and persist to parquet.

    :param year: season year.
    :param session_type: session code.
    :param out_dir: output directory for parquet files.
    :param max_rounds: upper bound on rounds to probe.
    :returns: path to the written parquet file, or None if nothing collected.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []

    for rnd in range(1, max_rounds + 1):
        frame = collect_session(year, rnd, session_type)
        if frame is not None:
            frames.append(frame)

    if not frames:
        logger.error("No data collected for season %s", year)
        return None

    season = pd.concat(frames, ignore_index=True)
    out_path = out_dir / f"laps_{year}.parquet"
    season.to_parquet(out_path, engine="pyarrow", index=False)
    logger.info("Wrote %s (%d laps)", out_path, len(season))
    return out_path


def run(settings_path: str | Path = "config/settings.yaml") -> None:
    """Entry point: collect every configured season."""
    settings = load_settings(settings_path)
    logging.basicConfig(level=logging.INFO)
    init_cache(settings["paths"]["fastf1_cache"])
    for year in settings["project"]["seasons"]:
        collect_season(year, settings["project"]["session_type"],
                       settings["paths"]["raw_parquet"])


if __name__ == "__main__":
    run()
```

---

### PHASE 2: VALIDATION AND CLEANING

#### `src/acquisition/validator.py`
```python
"""Phase 2: Data validation and the is_clean_lap mask.

Fixes agreed in PLAN.md:
 - join on DriverNumber, not abbreviation;
 - key events by RoundNumber, not event name;
 - GridPosition 0.0 means pit-lane start: flag it, do NOT overwrite;
 - weather imputation with provenance flags;
 - is_clean_lap mask excludes in/out laps, TrackStatus != 1, first lap,
   and laps slower than 107% of the stint median.
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def flag_grid_anomalies(laps: pd.DataFrame) -> pd.DataFrame:
    """Flag pit-lane starts (GridPosition == 0.0) without overwriting.

    :param laps: laps frame containing 'GridPosition'.
    :returns: frame with boolean 'pit_lane_start' column added.
    """
    out = laps.copy()
    if "GridPosition" in out.columns:
        out["pit_lane_start"] = out["GridPosition"].fillna(-1).astype(float).eq(0.0)
        n = int(out["pit_lane_start"].sum())
        if n:
            logger.info("Flagged %d pit-lane-start lap rows", n)
    else:
        out["pit_lane_start"] = False
    return out


def validate_driver_join(laps: pd.DataFrame) -> pd.DataFrame:
    """Ensure DriverNumber is present and typed as a stable string key.

    :param laps: laps frame.
    :returns: frame with a normalised 'DriverNumber' string column.
    :raises KeyError: if DriverNumber is entirely absent.
    """
    out = laps.copy()
    if "DriverNumber" not in out.columns:
        raise KeyError("DriverNumber missing; cannot build a stable join key")
    out["DriverNumber"] = out["DriverNumber"].astype(str).str.strip()
    missing = out["DriverNumber"].isin(["", "nan", "None"]).sum()
    if missing:
        logger.warning("%d rows have an empty DriverNumber", int(missing))
    return out


def impute_weather(laps: pd.DataFrame, defaults: dict[str, Any]) -> pd.DataFrame:
    """Impute missing AirTemp/TrackTemp from climatological defaults.

    Provenance is tracked so imputed values never masquerade as measured.

    :param laps: laps frame with EventName, AirTemp, TrackTemp.
    :param defaults: climatological_defaults mapping from track_traits.yaml.
    :returns: frame with imputed values and 'weather_imputed' provenance flag.
    """
    out = laps.copy()
    if "weather_imputed" not in out.columns:
        out["weather_imputed"] = False
    fallback = defaults.get("default", {"air": 22, "track": 30})

    for col, key in (("AirTemp", "air"), ("TrackTemp", "track")):
        if col not in out.columns:
            out[col] = np.nan
        mask = out[col].isna()
        if mask.any():
            def _lookup(name: str) -> float:
                entry = defaults.get(str(name).replace(" ", "_"), fallback)
                return float(entry.get(key, fallback[key]))
            out.loc[mask, col] = out.loc[mask, "EventName"].map(_lookup)
            out.loc[mask, "weather_imputed"] = True
            logger.info("Imputed %d missing %s values", int(mask.sum()), col)
    return out


def add_is_clean_lap(laps: pd.DataFrame, multiplier: float = 1.07,
                     track_status_ok: str = "1") -> pd.DataFrame:
    """Compute the is_clean_lap mask for representative-pace laps.

    Excludes: in/out laps, non-green TrackStatus, first lap of a stint,
    and laps slower than `multiplier` times the stint median lap time.

    :param laps: laps frame; expects PitInTime, PitOutTime, TrackStatus,
        LapTime, Stint, LapNumber, DriverNumber, RoundNumber, Year.
    :param multiplier: stint-median multiplier (1.07 = 107%).
    :param track_status_ok: green-flag TrackStatus code.
    :returns: frame with boolean 'is_clean_lap' column.
    """
    out = laps.copy()
    lap_s = out["LapTime"].dt.total_seconds() if np.issubdtype(
        out["LapTime"].dtype, np.datetime64) or hasattr(out["LapTime"], "dt") else out["LapTime"]
    out["_lap_s"] = pd.to_timedelta(out["LapTime"]).dt.total_seconds() \
        if not np.issubdtype(out["LapTime"].dtype, np.floating) else out["LapTime"]

    is_out = out.get("PitOutTime").notna() if "PitOutTime" in out else False
    is_in = out.get("PitInTime").notna() if "PitInTime" in out else False
    status_bad = out.get("TrackStatus", track_status_ok).astype(str) != track_status_ok

    grp = ["Year", "RoundNumber", "DriverNumber", "Stint"]
    out["_stint_min_lap"] = out.groupby(grp)["LapNumber"].transform("min")
    first_lap = out["LapNumber"].eq(out["_stint_min_lap"])
    stint_median = out.groupby(grp)["_lap_s"].transform("median")
    too_slow = out["_lap_s"] > (multiplier * stint_median)
    no_time = out["_lap_s"].isna()

    out["is_clean_lap"] = ~(is_out | is_in | status_bad | first_lap | too_slow | no_time)
    out = out.drop(columns=["_stint_min_lap"], errors="ignore")
    logger.info("Clean laps: %d / %d", int(out["is_clean_lap"].sum()), len(out))
    return out


def validate(laps: pd.DataFrame, defaults: dict[str, Any],
             multiplier: float = 1.07, track_status_ok: str = "1") -> pd.DataFrame:
    """Run the full validation pipeline defensively.

    :param laps: raw laps frame.
    :param defaults: climatological defaults.
    :returns: validated, flagged, masked frame.
    """
    if laps is None or laps.empty:
        logger.error("Empty laps frame passed to validate()")
        return pd.DataFrame()
    out = validate_driver_join(laps)
    out = flag_grid_anomalies(out)
    out = impute_weather(out, defaults)
    out = add_is_clean_lap(out, multiplier, track_status_ok)
    return out
```

---

### PHASE 3: PHYSICS AND BEHAVIOURAL FEATURES

#### `src/features/physics.py`
```python
"""Phase 3: Curvature-based physics proxies from X,Y telemetry.

Physics (SI units unless stated):
 - Curvature kappa = |x' y'' - y' x''| / (x'^2 + y'^2)^(3/2)   [1/m]
 - Lateral acceleration a_lat = v^2 * kappa                    [m/s^2]
 - Braking power proxy = v * |dv/dt| (per unit mass)           [m^2/s^3]
 - Combined G = sqrt(a_lat^2 + a_long^2) / g                   [g]
 - Aero load proxy = mean(v^2)                                 [m^2/s^2]
Derivatives estimated with Savitzky-Golay (Savitzky & Golay 1964) to
control noise amplification.

No Pacejka Magic Formula is used: its load-dependent coefficients are
proprietary and unavailable publicly, so curvature proxies stand in.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


def _savgol(series: np.ndarray, window: int, poly: int, deriv: int, dx: float) -> np.ndarray:
    """Safe Savitzky-Golay wrapper honouring array length and parity.

    :param series: input signal.
    :param window: window length (odd, >= poly + 2); clamped to length.
    :param poly: polynomial order.
    :param deriv: derivative order (0 = smooth).
    :param dx: sample spacing for derivative scaling.
    :returns: filtered array (falls back to input if too short).
    """
    n = len(series)
    if n < poly + 2:
        return series.astype(float)
    win = min(window, n if n % 2 == 1 else n - 1)
    if win % 2 == 0:
        win -= 1
    if win <= poly:
        return series.astype(float)
    return savgol_filter(series, win, poly, deriv=deriv, delta=dx, mode="interp")


def compute_curvature(x: np.ndarray, y: np.ndarray, window: int, poly: int) -> np.ndarray:
    """Compute path curvature from X,Y position via smoothed derivatives.

    :param x: X coordinates (m).
    :param y: Y coordinates (m).
    :param window: Savitzky-Golay window.
    :param poly: polynomial order.
    :returns: curvature array (1/m), zero where degenerate.
    """
    if len(x) < poly + 2:
        return np.zeros_like(x, dtype=float)
    dx = _savgol(x, window, poly, 1, 1.0)
    dy = _savgol(y, window, poly, 1, 1.0)
    ddx = _savgol(x, window, poly, 2, 1.0)
    ddy = _savgol(y, window, poly, 2, 1.0)
    denom = np.power(dx * dx + dy * dy, 1.5)
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(denom > 1e-9, np.abs(dx * ddy - dy * ddx) / denom, 0.0)
    return np.nan_to_num(kappa)


def lap_physics_features(tel: pd.DataFrame, window: int, poly: int,
                         gravity: float = 9.80665) -> dict[str, float]:
    """Aggregate physics proxies for a single lap's telemetry.

    :param tel: telemetry with columns X, Y, Speed (km/h), Time.
    :param window: Savitzky-Golay window.
    :param poly: polynomial order.
    :param gravity: g in m/s^2.
    :returns: dict of aggregate physics features for the lap.
    """
    if tel is None or tel.empty or len(tel) < poly + 2:
        return {"lat_accel_max": np.nan, "lat_accel_mean": np.nan,
                "brake_power_max": np.nan, "combined_g_max": np.nan,
                "aero_load_proxy": np.nan}

    v = tel["Speed"].to_numpy(dtype=float) / 3.6  # km/h -> m/s
    x = tel["X"].to_numpy(dtype=float)
    y = tel["Y"].to_numpy(dtype=float)
    t = pd.to_timedelta(tel["Time"]).dt.total_seconds().to_numpy()
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.1
    dt = dt if dt > 0 else 0.1

    kappa = compute_curvature(x, y, window, poly)
    a_lat = v * v * kappa                              # v^2 * kappa
    a_long = _savgol(v, window, poly, 1, dt)           # dv/dt
    brake_power = v * np.abs(np.minimum(a_long, 0.0))  # only decel
    combined_g = np.sqrt(a_lat ** 2 + a_long ** 2) / gravity

    return {
        "lat_accel_max": float(np.nanmax(a_lat)),
        "lat_accel_mean": float(np.nanmean(a_lat)),
        "brake_power_max": float(np.nanmax(brake_power)),
        "combined_g_max": float(np.nanmax(combined_g)),
        "aero_load_proxy": float(np.nanmean(v ** 2)),  # mean(speed^2)
    }
```

#### `src/features/behaviour.py`
```python
"""Phase 3: Behavioural (driving-style) features.

Jerk pipeline (double Savitzky-Golay):
  1. smooth speed;
  2. differentiate for longitudinal acceleration;
  3. differentiate the smoothed acceleration for jerk.
Jerk [m/s^3] captures aggression/roughness of inputs. Window lengths
{7, 11, 15, 21} are tested empirically (Savitzky & Golay 1964).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter

logger = logging.getLogger(__name__)


def _safe_savgol(sig: np.ndarray, window: int, poly: int, deriv: int, dx: float) -> np.ndarray:
    n = len(sig)
    if n < poly + 2:
        return sig.astype(float)
    win = min(window, n if n % 2 == 1 else n - 1)
    if win % 2 == 0:
        win -= 1
    if win <= poly:
        return sig.astype(float)
    return savgol_filter(sig, win, poly, deriv=deriv, delta=dx, mode="interp")


def jerk_features(tel: pd.DataFrame, window: int, poly: int = 2) -> dict[str, float]:
    """Compute jerk aggregates via double Savitzky-Golay differentiation.

    :param tel: telemetry with Speed (km/h) and Time.
    :param window: Savitzky-Golay window length.
    :param poly: polynomial order.
    :returns: dict with jerk RMS and max magnitude.
    """
    if tel is None or tel.empty or len(tel) < poly + 2:
        return {"jerk_rms": np.nan, "jerk_abs_max": np.nan}
    v = tel["Speed"].to_numpy(dtype=float) / 3.6
    t = pd.to_timedelta(tel["Time"]).dt.total_seconds().to_numpy()
    dt = np.median(np.diff(t)) if len(t) > 1 else 0.1
    dt = dt if dt > 0 else 0.1

    v_smooth = _safe_savgol(v, window, poly, 0, dt)
    accel = _safe_savgol(v_smooth, window, poly, 1, dt)
    accel_smooth = _safe_savgol(accel, window, poly, 0, dt)
    jerk = np.gradient(accel_smooth, dt)
    return {
        "jerk_rms": float(np.sqrt(np.nanmean(jerk ** 2))),
        "jerk_abs_max": float(np.nanmax(np.abs(jerk))),
    }


def throttle_brake_features(tel: pd.DataFrame) -> dict[str, float]:
    """Throttle and brake behavioural aggregates for a lap.

    :param tel: telemetry with Throttle (0-100) and Brake (bool/0-1).
    :returns: dict of throttle std, full-throttle fraction, brake applications.
    """
    out: dict[str, float] = {"throttle_std": np.nan, "full_throttle_frac": np.nan,
                             "brake_applications": np.nan}
    if tel is None or tel.empty:
        return out
    if "Throttle" in tel.columns:
        thr = tel["Throttle"].to_numpy(dtype=float)
        out["throttle_std"] = float(np.nanstd(thr))
        out["full_throttle_frac"] = float(np.nanmean(thr > 95.0))
    if "Brake" in tel.columns:
        brake = tel["Brake"].astype(float).to_numpy()
        # count rising edges (brake application onsets)
        onsets = np.sum((brake[1:] > 0.5) & (brake[:-1] <= 0.5))
        out["brake_applications"] = float(onsets)
    return out
```

#### `src/features/timeseries.py`
```python
"""Phase 3: Rolling and lag features over tyre life within a stint.

All rolling/lag operations are grouped by (Year, RoundNumber, DriverNumber,
Stint) and ordered by TyreLife to prevent leakage across stints or drivers.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)

_GROUP = ["Year", "RoundNumber", "DriverNumber", "Stint"]


def add_lag_features(df: pd.DataFrame, col: str, lags: tuple[int, ...] = (1, 2, 3)) -> pd.DataFrame:
    """Add lagged versions of a column within each stint.

    :param df: feature frame sorted implicitly by TyreLife.
    :param col: column to lag.
    :param lags: lag steps.
    :returns: frame with new lag columns.
    """
    out = df.sort_values(_GROUP + ["TyreLife"]).copy()
    for lag in lags:
        out[f"{col}_lag{lag}"] = out.groupby(_GROUP)[col].shift(lag)
    return out


def add_rolling_features(df: pd.DataFrame, col: str, windows: tuple[int, ...] = (3, 5)) -> pd.DataFrame:
    """Add trailing rolling means/stds within each stint (no leakage).

    :param df: feature frame.
    :param col: column to roll.
    :param windows: rolling window sizes.
    :returns: frame with rolling columns.
    """
    out = df.sort_values(_GROUP + ["TyreLife"]).copy()
    for win in windows:
        g = out.groupby(_GROUP)[col]
        out[f"{col}_rollmean{win}"] = g.transform(
            lambda s: s.rolling(win, min_periods=1).mean())
        out[f"{col}_rollstd{win}"] = g.transform(
            lambda s: s.rolling(win, min_periods=1).std())
    return out
```

---

### PHASE 4: TARGET CONSTRUCTION AND MODELLING

#### `src/features/target.py`
```python
"""Phase 4: Composite Wear Index (CWI) target construction.

Steps:
 1. Fuel correction: corrected_lap = lap - penalty * fuel_remaining,
    where fuel burned on lap N is (N-1) * burn_per_lap, so
    fuel_remaining(N) = initial - (N-1) * burn_per_lap.
 2. Per-stint degradation rate: OLS slope of corrected lap time vs TyreLife
    (scipy.stats.linregress), after IQR outlier removal, kept only if |r|>0.3.
 3. Energy proxy per stint: mean per-lap contact-patch energy proxy
    (aggregate of physics proxies; wear ~ frictional work, Archard 1953).
 4. CWI construction with the empirical validation decision rule:
      Spearman(energy, deg_rate) > 0.4  -> energy weight 0.5;
      0.2-0.4 -> down-weight (laptime weight 0.7);
      < 0.2   -> energy demoted to a feature only (weight 0).
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import linregress, spearmanr

logger = logging.getLogger(__name__)


def fuel_correct(df: pd.DataFrame, penalty: float, initial_fuel: float,
                 race_laps_col: str = "RaceLaps") -> pd.DataFrame:
    """Apply linear fuel correction to lap times.

    fuel_remaining(N) = initial - (N-1) * (initial / race_laps)
    corrected = raw_lap_s - penalty * fuel_remaining

    :param df: laps with LapNumber, lap-time seconds, and race length.
    :param penalty: seconds per kg (e.g. 0.032).
    :param initial_fuel: starting fuel kg (e.g. 110).
    :param race_laps_col: column giving scheduled race laps per event.
    :returns: frame with 'lap_s' and 'corrected_lap_s'.
    """
    out = df.copy()
    if "lap_s" not in out.columns:
        out["lap_s"] = pd.to_timedelta(out["LapTime"]).dt.total_seconds()
    race_laps = out[race_laps_col].replace(0, np.nan)
    burn_per_lap = initial_fuel / race_laps
    fuel_remaining = initial_fuel - (out["LapNumber"] - 1) * burn_per_lap
    fuel_remaining = fuel_remaining.clip(lower=0.0)
    out["corrected_lap_s"] = out["lap_s"] - penalty * fuel_remaining
    return out


def _iqr_mask(values: np.ndarray) -> np.ndarray:
    """Return a boolean mask keeping values inside 1.5*IQR fences."""
    q1, q3 = np.nanpercentile(values, [25, 75])
    iqr = q3 - q1
    lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    return (values >= lo) & (values <= hi)


@dataclass
class StintResult:
    """Per-stint degradation regression result."""
    deg_rate: float          # slope (s per lap of tyre life)
    r_value: float
    n_laps: int
    energy_proxy: float


def stint_degradation(stint: pd.DataFrame, min_r: float, min_laps: int) -> StintResult | None:
    """Fit corrected-lap-time slope vs TyreLife for a single clean stint.

    :param stint: clean laps of one stint with corrected_lap_s, TyreLife.
    :param min_r: minimum |r| to accept the slope.
    :param min_laps: minimum clean laps required.
    :returns: StintResult or None if the stint is unusable.
    """
    clean = stint[stint["is_clean_lap"]].copy()
    if len(clean) < min_laps:
        return None
    mask = _iqr_mask(clean["corrected_lap_s"].to_numpy())
    clean = clean[mask]
    if len(clean) < min_laps:
        return None
    x = clean["TyreLife"].to_numpy(dtype=float)
    y = clean["corrected_lap_s"].to_numpy(dtype=float)
    if np.ptp(x) == 0:
        return None
    reg = linregress(x, y)
    if abs(reg.rvalue) < min_r:
        logger.debug("Stint rejected: |r|=%.3f < %.2f", abs(reg.rvalue), min_r)
        return None
    energy = float(clean.get("aero_load_proxy", pd.Series([np.nan])).mean())
    return StintResult(float(reg.slope), float(reg.rvalue), len(clean), energy)


def build_stint_table(laps: pd.DataFrame, min_r: float, min_laps: int) -> pd.DataFrame:
    """Build the per-stint target table with degradation rate and energy.

    :param laps: fully featured laps frame.
    :param min_r: minimum |r| filter.
    :param min_laps: minimum clean laps per stint.
    :returns: one row per (Year, Round, Driver, Stint) with deg_rate, energy.
    """
    rows: list[dict] = []
    group = ["Year", "RoundNumber", "DriverNumber", "Stint"]
    for keys, stint in laps.groupby(group):
        res = stint_degradation(stint, min_r, min_laps)
        if res is None:
            continue
        rows.append({**dict(zip(group, keys)),
                     "deg_rate": res.deg_rate, "r_value": res.r_value,
                     "n_laps": res.n_laps, "energy_proxy": res.energy_proxy,
                     "Compound": stint["Compound"].iloc[0] if "Compound" in stint else None})
    table = pd.DataFrame(rows)
    logger.info("Built stint table: %d usable stints", len(table))
    return table


def validate_and_build_cwi(stints: pd.DataFrame, keep_thr: float, downweight_low: float,
                           energy_w_full: float, laptime_w_down: float) -> tuple[pd.DataFrame, dict]:
    """Run the Spearman validation study and construct the CWI target.

    Decision rule (per PLAN.md):
      rho > keep_thr           -> energy stays, weight = energy_w_full (0.5);
      downweight_low<=rho<=keep-> down-weight, laptime weight = laptime_w_down (0.7);
      rho < downweight_low     -> energy demoted to feature only (weight 0).

    Both components are z-scored before weighting so the CWI is unitless.

    :returns: (stints with 'CWI' column, decision metadata dict).
    """
    valid = stints.dropna(subset=["deg_rate", "energy_proxy"])
    if len(valid) < 10:
        logger.warning("Too few stints (%d) for a reliable Spearman study", len(valid))
        rho, p = np.nan, np.nan
    else:
        rho, p = spearmanr(valid["energy_proxy"], valid["deg_rate"])
    logger.info("Spearman(energy, deg_rate) = %.3f (p=%.3g)", rho, p)

    def _z(s: pd.Series) -> pd.Series:
        sd = s.std(ddof=0)
        return (s - s.mean()) / sd if sd and not np.isnan(sd) else s * 0.0

    z_deg = _z(stints["deg_rate"])
    z_energy = _z(stints["energy_proxy"])

    if not np.isnan(rho) and rho > keep_thr:
        decision, e_w = "energy_in_target", energy_w_full
        l_w = 1.0 - e_w
    elif not np.isnan(rho) and rho >= downweight_low:
        decision, l_w = "energy_downweighted", laptime_w_down
        e_w = 1.0 - l_w
    else:
        decision, e_w, l_w = "energy_demoted_to_feature", 0.0, 1.0

    out = stints.copy()
    out["CWI"] = l_w * z_deg + e_w * z_energy
    meta = {"spearman_rho": rho, "spearman_p": p, "decision": decision,
            "energy_weight": e_w, "laptime_weight": l_w}
    logger.info("CWI decision: %s (energy_w=%.2f, laptime_w=%.2f)", decision, e_w, l_w)
    return out, meta
```

#### `src/modelling/baseline.py`
```python
"""Phase 4: LinearRegression baseline on Tier A core features."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

logger = logging.getLogger(__name__)

TIER_A = ["TyreLife", "compound_ordinal", "AirTemp", "TrackTemp", "abrasiveness"]


def train_baseline(train: pd.DataFrame, holdout: pd.DataFrame,
                   features: list[str] = TIER_A, target: str = "CWI") -> dict:
    """Fit a LinearRegression baseline and report hold-out MAE.

    :param train: training rows (2022-2023).
    :param holdout: hold-out rows (2024).
    :param features: Tier A feature list.
    :param target: target column (CWI).
    :returns: dict with fitted model and MAE.
    """
    feats = [f for f in features if f in train.columns]
    xtr = train[feats].fillna(train[feats].median())
    xho = holdout[feats].fillna(train[feats].median())
    model = LinearRegression()
    model.fit(xtr, train[target])
    mae = mean_absolute_error(holdout[target], model.predict(xho))
    logger.info("Baseline hold-out MAE: %.4f", mae)
    return {"model": model, "mae": float(mae), "features": feats}
```

#### `src/modelling/xgb_model.py`
```python
"""Phase 4: Tiered XGBoost runs with the >=3% per-tier gate.

Tier A core, Tier B behavioural, Tier C physics proxies are added
cumulatively; a tier is kept only if cross-validated MAE improves by
>= tier_gate_pct. XGBoost is chosen over deep learning because tree
ensembles dominate medium-sized tabular data (Grinsztajn et al. 2022;
Chen & Guestrin 2016).
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)

TIER_A = ["TyreLife", "compound_ordinal", "AirTemp", "TrackTemp", "abrasiveness"]
TIER_B = ["throttle_std", "brake_applications", "jerk_rms", "jerk_abs_max", "full_throttle_frac"]
TIER_C = ["lat_accel_max", "lat_accel_mean", "brake_power_max", "combined_g_max", "aero_load_proxy"]


def _cv_mae(df: pd.DataFrame, feats: list[str], target: str, n_splits: int, params: dict) -> float:
    """Mean CV MAE using TimeSeriesSplit (chronological, no leakage)."""
    tscv = TimeSeriesSplit(n_splits=n_splits)
    order = df.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
    x = order[feats].to_numpy()
    y = order[target].to_numpy()
    maes = []
    for tr, va in tscv.split(x):
        m = XGBRegressor(**params)
        m.fit(x[tr], y[tr])
        maes.append(mean_absolute_error(y[va], m.predict(x[va])))
    return float(np.mean(maes))


def run_tiers(train: pd.DataFrame, target: str = "CWI", n_splits: int = 4,
              gate_pct: float = 3.0, params: dict | None = None) -> dict:
    """Add feature tiers cumulatively, keeping each only on >= gate_pct gain.

    :returns: dict with selected features and per-tier CV MAEs.
    """
    params = params or {"n_estimators": 400, "max_depth": 4, "learning_rate": 0.05,
                        "subsample": 0.8, "colsample_bytree": 0.8, "random_state": 42}
    selected = [f for f in TIER_A if f in train.columns]
    mae_a = _cv_mae(train, selected, target, n_splits, params)
    history = {"A": mae_a}
    logger.info("Tier A CV MAE: %.4f", mae_a)

    for name, tier in (("B", TIER_B), ("C", TIER_C)):
        cand = selected + [f for f in tier if f in train.columns]
        mae = _cv_mae(train, cand, target, n_splits, params)
        improve = 100.0 * (history[list(history)[-1]] - mae) / history[list(history)[-1]]
        history[name] = mae
        if improve >= gate_pct:
            selected = cand
            logger.info("Tier %s kept: MAE %.4f (%.2f%% gain)", name, mae, improve)
        else:
            logger.info("Tier %s rejected: MAE %.4f (%.2f%% gain < %.1f%%)",
                        name, mae, improve, gate_pct)
    return {"features": selected, "cv_history": history, "params": params}
```

#### `src/modelling/tuning.py`
```python
"""Phase 4: Optuna hyperparameter search nested in TimeSeriesSplit.

50 trials by default. The final gate: tuned XGBoost hold-out MAE must beat
the LinearRegression baseline by >= baseline_gate_pct (15%).
"""
from __future__ import annotations

import logging

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBRegressor

logger = logging.getLogger(__name__)


def tune(train: pd.DataFrame, features: list[str], target: str = "CWI",
         n_trials: int = 50, n_splits: int = 4, seed: int = 42) -> dict:
    """Run Optuna search with chronological CV and return the best params.

    :param train: training frame ordered chronologically.
    :param features: selected feature list from tier selection.
    :returns: dict with best params and best CV MAE.
    """
    order = train.sort_values(["Year", "RoundNumber"]).reset_index(drop=True)
    x = order[features].to_numpy()
    y = order[target].to_numpy()
    tscv = TimeSeriesSplit(n_splits=n_splits)

    def objective(trial: optuna.Trial) -> float:
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 900),
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "random_state": seed,
        }
        maes = []
        for tr, va in tscv.split(x):
            m = XGBRegressor(**params)
            m.fit(x[tr], y[tr])
            maes.append(mean_absolute_error(y[va], m.predict(x[va])))
        return float(np.mean(maes))

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    logger.info("Best CV MAE: %.4f", study.best_value)
    return {"best_params": study.best_params, "best_cv_mae": float(study.best_value)}


def passes_baseline_gate(baseline_mae: float, xgb_mae: float, gate_pct: float = 15.0) -> bool:
    """Return True if XGBoost beats the baseline MAE by >= gate_pct."""
    improve = 100.0 * (baseline_mae - xgb_mae) / baseline_mae
    logger.info("XGBoost vs baseline improvement: %.2f%% (gate %.1f%%)", improve, gate_pct)
    return improve >= gate_pct
```

---

### PHASE 5-7: DETAILED SKELETONS WITH PSEUDOCODE

#### `src/modelling/interpret.py` (Phase 5: SHAP)
```python
"""Phase 5: SHAP interpretation (Lundberg & Lee 2017, NeurIPS 30).

SHAP TreeExplainer gives exact Shapley values for tree ensembles. We compute:
 - global mean(|SHAP|) feature ranking;
 - local explanations for representative stints;
 - a sanity check that TyreLife, temperature and compound rank in the top 8.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def global_importance(model, x_frame):
    """Return features ranked by mean(|SHAP value|).

    PSEUDOCODE:
      explainer = shap.TreeExplainer(model)
      shap_values = explainer.shap_values(x_frame)
      importance = mean(abs(shap_values), axis=0)
      return sorted(zip(x_frame.columns, importance), key=..., desc)
    """
    raise NotImplementedError


def sanity_check_top_features(ranking, must_include=("TyreLife", "TrackTemp", "compound_ordinal"),
                              top_n: int = 8) -> bool:
    """Assert physics-critical features appear in the top-N ranking.

    PSEUDOCODE:
      top = {name for name, _ in ranking[:top_n]}
      missing = [f for f in must_include if f not in top]
      if missing: logger.warning("Expected features outside top %d: %s", top_n, missing)
      return not missing
    """
    raise NotImplementedError


def local_explanation(model, row):
    """Waterfall explanation for a single stint prediction.

    PSEUDOCODE:
      explainer = shap.TreeExplainer(model)
      sv = explainer(row)         # single-row Explanation
      return sv                   # feed to shap.plots.waterfall
    """
    raise NotImplementedError
```

#### `src/simulation/strategy.py` (Phase 6: stint simulator)
```python
"""Phase 6: Iterative stint simulator and pit-window crossover finder.

Uses the trained CWI/degradation model to project cumulative stint time and
finds the pit lap at which a two-stop crosses a one-stop, given pit_loss_s
(25 s default). Fuel correction is re-applied forward.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def simulate_stint(model, compound_ordinal, start_life, n_laps, context):
    """Project lap-by-lap corrected times for a stint on one compound.

    PSEUDOCODE:
      total = 0
      for lap in range(n_laps):
          life = start_life + lap
          feat = build_feature_row(compound_ordinal, life, context)
          deg_rate = model.predict(feat)        # or CWI -> deg mapping
          lap_time = base_pace(compound_ordinal, context) + deg_rate * life
          lap_time += fuel_term(lap, context)   # forward fuel correction
          total += lap_time
      return total, per_lap_times
    """
    raise NotImplementedError


def find_pit_crossover(model, strategies, race_laps, pit_loss_s, context):
    """Find the pit lap where a 2-stop total beats a 1-stop total.

    PSEUDOCODE:
      one_stop = min over pit_lap of (
          simulate_stint(stintA) + pit_loss_s + simulate_stint(stintB))
      two_stop = min over (p1,p2) of (
          sum of three stints + 2 * pit_loss_s)
      crossover_lap = smallest pit_lap where two_stop < one_stop
      return {one_stop, two_stop, crossover_lap}
    NOTES:
      - enforce mandatory two-compound rule;
      - clamp stint lengths to plausible tyre life;
      - handle empty/degenerate inputs by returning None.
    """
    raise NotImplementedError
```

#### `src/modelling/lstm_ensemble.py` (Phase 6: conditional)
```python
"""Phase 6 (CONDITIONAL): LSTM-XGBoost stacked ensemble.

Gate 1: build only if a Ljung-Box test (Ljung & Box 1978, Biometrika 65)
on XGBoost residuals shows temporal autocorrelation (p < ljung_box_alpha).
Gate 2: keep only if ensemble hold-out MAE improves by >= ensemble_gate_pct
(5%) over standalone XGBoost.

LSTM reference: Hochreiter & Schmidhuber 1997, Neural Computation 9(8).
Stacked LSTM + gradient boosting: e.g. stacked snapshot LSTM ensembles and
LSTM+boosting residual-correction hybrids in the forecasting literature.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def ljung_box_gate(residuals, lags: int = 10, alpha: float = 0.05) -> bool:
    """Return True if residuals show significant autocorrelation.

    PSEUDOCODE:
      from statsmodels.stats.diagnostic import acorr_ljungbox
      result = acorr_ljungbox(residuals, lags=[lags], return_df=True)
      p = result['lb_pvalue'].iloc[0]
      logger.info("Ljung-Box p=%.4g", p)
      return p < alpha            # autocorrelation present -> build LSTM
    """
    raise NotImplementedError


def build_ensemble(train_seq, xgb_model):
    """Train an LSTM on stint sequences and stack with XGBoost.

    PSEUDOCODE:
      # sequences ordered by TyreLife within (year,round,driver,stint)
      lstm = Sequential([LSTM(units), Dense(1)])
      lstm.compile(loss='mae', optimizer='adam')
      lstm.fit(X_seq_train, y_train, validation_split=..., shuffle=False)
      # meta-features: [xgb_pred, lstm_pred] -> ridge/linear meta-learner
      meta = LinearRegression().fit([xgb_pred, lstm_pred], y_train)
      return {lstm, meta}
    """
    raise NotImplementedError


def passes_ensemble_gate(xgb_mae, ensemble_mae, gate_pct: float = 5.0) -> bool:
    """Keep the ensemble only if it improves MAE by >= gate_pct.

    PSEUDOCODE:
      improve = 100 * (xgb_mae - ensemble_mae) / xgb_mae
      return improve >= gate_pct
    """
    raise NotImplementedError
```

---

### TESTS (skeletons)

#### `tests/test_validator.py`
```python
"""Pytest skeletons for src/acquisition/validator.py."""
import pandas as pd
import pytest

from src.acquisition import validator


def _toy_laps():
    """Minimal laps frame exercising the mask branches."""
    return pd.DataFrame({
        "Year": [2024] * 5, "RoundNumber": [1] * 5, "DriverNumber": ["44"] * 5,
        "Stint": [1] * 5, "LapNumber": [1, 2, 3, 4, 5],
        "LapTime": pd.to_timedelta([90, 91, 92, 200, 91], unit="s"),
        "TrackStatus": ["1", "1", "1", "1", "4"],
        "PitInTime": [pd.NaT] * 5, "PitOutTime": [pd.NaT] * 5,
        "GridPosition": [0.0, 3.0, 3.0, 3.0, 3.0], "EventName": ["Bahrain"] * 5,
    })


def test_grid_zero_flagged_not_overwritten():
    out = validator.flag_grid_anomalies(_toy_laps())
    assert out["pit_lane_start"].iloc[0] is True or out["pit_lane_start"].iloc[0]
    assert out["GridPosition"].iloc[0] == 0.0  # not overwritten


def test_first_lap_excluded():
    out = validator.add_is_clean_lap(_toy_laps())
    assert not out.loc[out["LapNumber"] == 1, "is_clean_lap"].iloc[0]


def test_slow_lap_excluded():
    out = validator.add_is_clean_lap(_toy_laps())
    assert not out.loc[out["LapNumber"] == 4, "is_clean_lap"].iloc[0]  # 200s > 107% median


def test_track_status_excluded():
    out = validator.add_is_clean_lap(_toy_laps())
    assert not out.loc[out["LapNumber"] == 5, "is_clean_lap"].iloc[0]


def test_missing_driver_number_raises():
    with pytest.raises(KeyError):
        validator.validate_driver_join(pd.DataFrame({"A": [1]}))
```

#### `tests/test_target.py`
```python
"""Pytest skeletons for src/features/target.py."""
import numpy as np
import pandas as pd

from src.features import target


def test_fuel_correction_formula():
    df = pd.DataFrame({"LapNumber": [1, 2], "lap_s": [100.0, 100.0], "RaceLaps": [50, 50]})
    out = target.fuel_correct(df, penalty=0.032, initial_fuel=110.0)
    # Lap 1 fuel_remaining = 110; correction = 0.032*110 = 3.52
    assert abs(out["corrected_lap_s"].iloc[0] - (100.0 - 3.52)) < 1e-6
    # Lap 2 fuel_remaining = 110 - 1*(110/50) = 107.8
    assert abs(out["corrected_lap_s"].iloc[1] - (100.0 - 0.032 * 107.8)) < 1e-6


def test_stint_rejected_below_min_r():
    # random noise -> low |r| -> rejected
    n = 12
    stint = pd.DataFrame({
        "is_clean_lap": [True] * n, "TyreLife": np.arange(n),
        "corrected_lap_s": np.random.RandomState(0).normal(90, 5, n),
        "aero_load_proxy": np.full(n, 1000.0),
    })
    res = target.stint_degradation(stint, min_r=0.9, min_laps=5)
    assert res is None


def test_cwi_decision_rule_demotes_when_low_rho():
    stints = pd.DataFrame({
        "deg_rate": np.linspace(0.01, 0.1, 20),
        "energy_proxy": np.random.RandomState(1).normal(0, 1, 20),  # uncorrelated
    })
    _, meta = target.validate_and_build_cwi(
        stints, keep_thr=0.4, downweight_low=0.2, energy_w_full=0.5, laptime_w_down=0.7)
    assert meta["decision"] in {"energy_demoted_to_feature", "energy_downweighted", "energy_in_target"}
```

---

### `CLAUDE.md` (project instructions for Claude Code)
```markdown
# CLAUDE.md - F1 Tyre Degradation Modelling

## Golden rules
- British English in all comments, docstrings and prose. No emoji. No em dashes.
- Use the `logging` module, never `print`.
- Every magic number comes from `config/*.yaml`. No hard-coded constants in `src/`.
- Type hints and docstrings (with physics units) on every public function.
- Defensive handling of empty/malformed frames: return empty/None, log, never crash a season run.

## Environment
- Python 3.11 in an isolated venv. Install from `requirements.txt`.
- Enable the fastf1 cache before any collection (`init_cache`).

## Data contract
- Join laps on `DriverNumber` (string), never the three-letter abbreviation.
- Key events by `RoundNumber`, never `EventName`.
- `GridPosition == 0.0` means a pit-lane start: set `pit_lane_start`, do NOT overwrite.
- Weather imputation must set `weather_imputed = True` (provenance).
- Representative pace uses the `is_clean_lap` mask only.

## Modelling order (do not reorder)
1. Collect (Phase 1) -> validate (Phase 2) -> features (Phase 3) -> target+models (Phase 4).
2. Fuel correction: fuel_remaining(N) = 110 - (N-1)*burn_per_lap; penalty 0.032 s/kg.
3. CWI Spearman gate decides energy weighting (0.5 / down-weight 0.7 / demote).
4. Baseline LinearRegression; XGBoost must beat it by >= 15% hold-out MAE.
5. Feature tiers kept only on >= 3% CV MAE gain. Optuna 50 trials in TimeSeriesSplit(4).
6. Train 2022-2023, hold out 2024. Never shuffle across seasons.
7. SHAP sanity check: TyreLife, temperature, compound in top 8.
8. LSTM ensemble ONLY if Ljung-Box p < 0.05 AND it gains >= 5% MAE.

## No Pacejka
Do not implement the Magic Formula on the critical path. Coefficients are
load-dependent and unavailable publicly. Use curvature-based physics proxies.
```

---

## Recommendations

**Stage 1 (Weeks 1-2): Build the spine and prove the target.** Implement Phases 1-2, then Phase 4 target construction first. Run the Spearman validation study immediately on the raw stint table. **Decision threshold:** if rho > 0.4, proceed with energy at weight 0.5; if 0.2-0.4, down-weight to laptime 0.7; if < 0.2, demote energy to a feature and treat the CWI as essentially the fuel-corrected degradation slope. This single number determines the entire target design, so run it before investing in physics features.

**Stage 2 (Weeks 3-4): Feature tiers and the baseline gate.** Add Tier B and Tier C only if each clears the 3% CV MAE gate. **Threshold that changes the plan:** if Tier C physics proxies fail the 3% gate, that is evidence the curvature proxies are not adding signal beyond TyreLife/temperature/compound; in that case, simplify to Tier A+B and document it rather than forcing physics features in. The XGBoost-vs-baseline 15% gate is the go/no-go for the whole modelling approach: if XGBoost cannot beat LinearRegression by 15%, the problem is likely target noise, and you should revisit the clean-lap mask and IQR filtering before adding model complexity.

**Stage 3 (Weeks 5-6): Interpret, simulate, and only conditionally ensemble.** Run SHAP and enforce the top-8 sanity check; if TyreLife/temperature/compound are not in the top 8, treat it as a data-quality red flag (likely a leakage or encoding bug), not a modelling triumph. Build the stint simulator with the 25 s pit loss. **Do not build the LSTM ensemble unless the Ljung-Box test on residuals returns p < 0.05** and it then clears the 5% MAE gate; on a per-stint tabular target, residual autocorrelation is often absent, in which case the ensemble is unjustified complexity and should be skipped.

**Cross-cutting:** treat the compound and abrasiveness configs as living documents. The high-value, high-confidence entries (Bahrain 5/5 abrasion; Silverstone/Suzuka/Barcelona/Spa/Zandvoort/Losail 5/5 lateral energy; Monaco lowest severity) are well sourced and should anchor the model; expand the UNK compound entries from Pirelli press releases before final training.

## Caveats

- **Fuel correction is an approximation.** The 0.032 s/kg penalty and linear-burn assumption ignore lift-and-coast, safety-car laps, per-driver and per-lap burn variation, and non-linear weight-to-laptime effects. Even F1.com's public fuel-corrected data uses an educated fuel-load guess. Treat corrected times as comparable within a stint, not as absolute truth.
- **Thermal window numbers conflict across public sources.** Red Bull cites soft ~85-115 C and hard ~110-140 C; thef1db cites soft ~90-110 C and hard ~100-120 C. Neither is an official Pirelli spec sheet. The model should rely on measured `TrackTemp`/`AirTemp`, not hard-coded windows.
- **Compound allocations are incomplete.** Only weekend nominations confirmable from Pirelli press releases are embedded; the rest are marked UNK and must not be guessed. Compound labels are relative per weekend (a C3 can be the hard tyre at Monaco and the soft at Silverstone), so always use the ordinal C-number, not the hard/medium/soft label, as the model feature.
- **Track abrasiveness is a two-axis simplification.** Abrasion and lateral energy diverge (Silverstone is high energy but low abrasion; Bahrain is high abrasion but medium energy; Austria is high abrasion but low lateral). The single composite 1-3 is a modelling convenience; where possible use the separate abrasion and energy fields. Surfaces also changed within 2022-2024 (Spa, Shanghai, Interlagos, COTA, Melbourne resurfaced), so a circuit's rating is era-dependent.
- **Curvature proxies are not tyre forces.** They approximate the physical drivers of energy dissipation but omit slip angle, actual vertical load and the true friction coefficient, all of which require the excluded Pacejka model. The Spearman gate exists precisely to test whether these proxies carry real signal before trusting them.
- **Some retrieved sources are secondary or forward-dated.** Several compound and severity confirmations come from 2025-2026 previews describing stable circuit characteristics; the physics and regulation facts (798 kg, 110 kg, 18-inch, 0.03 s/kg) are corroborated by multiple independent sources and are safe to rely on. The Savitzky-Golay 1964 paper's original coefficient tables contained known typographical errors, so use scipy's implementation rather than transcribing coefficients.

---

## References

1. Persson, B.N.J. (2001). Theory of rubber friction and contact mechanics. *The Journal of Chemical Physics* 115(8):3840-3861. DOI 10.1063/1.1388626.
2. Archard, J.F. (1953). Contact and Rubbing of Flat Surfaces. *Journal of Applied Physics* 24(8):981-988.
3. Rubber Wear: History, Mechanisms, and Perspectives (2025). *Tribology Letters*, Springer. DOI 10.1007/s11249-025-02025-9 (reviews Archard and Schallamach frictional-work wear laws).
4. Savitzky, A. & Golay, M.J.E. (1964). Smoothing and Differentiation of Data by Simplified Least Squares Procedures. *Analytical Chemistry* 36(8):1627-1639. DOI 10.1021/ac60214a047.
5. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. *Proceedings of the 22nd ACM SIGKDD*, pp. 785-794. arXiv:1603.02754.
6. Grinsztajn, L., Oyallon, E. & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? *Advances in Neural Information Processing Systems* 35:507-520.
7. Lundberg, S.M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *Advances in Neural Information Processing Systems* 30:4765-4774.
8. Bergmeir, C., Hyndman, R.J. & Koo, B. (2018). A note on the validity of cross-validation for evaluating autoregressive time series prediction. *Computational Statistics & Data Analysis* 120:70-83. (See also Bergmeir & Benitez 2012, *Journal of Intelligent Information Systems* 38.)
9. Hochreiter, S. & Schmidhuber, J. (1997). Long Short-Term Memory. *Neural Computation* 9(8):1735-1780. DOI 10.1162/neco.1997.9.8.1735.
10. Ljung, G.M. & Box, G.E.P. (1978). On a Measure of Lack of Fit in Time Series Models. *Biometrika* 65(2):297-303. DOI 10.1093/biomet/65.2.297.
11. Pacejka, H.B. (2005/2012). *Tire and Vehicle Dynamics*. Elsevier. (Magic Formula; load-dependent coefficients and VXLOW low-speed handling.) See also Bakker, Nyborg & Pacejka (1986).
12. Milliken, W.F. & Milliken, D.L. (1995). *Race Car Vehicle Dynamics*. SAE International. (Load transfer, slip angle, lateral force vs vertical load.)
13. "Explainable Time Series Prediction of Tyre Energy in Formula One Race Strategy" (2025). arXiv:2501.04067. (Mercedes-AMG PETRONAS; XGBoost and deep learning for tyre energy; frictional-energy tyre model.)
14. "A State-Space Approach to Modeling Tire Degradation in Formula 1 Racing" (2025). arXiv:2512.00640. (fastf1-based; fuel mass plus latent tyre pace; pit stops as state resets.)
15. FIA 2022 Formula 1 Technical Regulations, Article 4.1 (minimum mass 798 kg); RaceFans, 17 March 2022 (798 kg confirmation). Motorsport.com and Wikipedia (18-inch wheels, 110 kg fuel, two-compound rule).
16. Pirelli Motorsport F1 press releases (press.pirelli.com) and previews via formula1.com, autosport.com, racingnews365.com, f1technical.net, pitpass.com (compound nominations 2022-2024; abrasiveness and lateral-energy ratings; Bahrain 5/5 abrasiveness; Silverstone/Suzuka/Barcelona/Spa/Zandvoort/Losail 5/5 lateral energy; Monaco lowest severity).
17. Red Bull ("Everything you need to know about F1 tyres") and thef1db (compound operating temperature windows; note inter-source discrepancy).
18. F1Briefing, "Fuel Load vs. Lap Time" ("each additional kilogram adding about 0.03 seconds per lap"; 110 kg full load) and practitioner sources citing 0.025-0.040 s/kg.
19. Race Sundays / F1Briefing / Formula1.com (pit-stop stationary time 2.0-2.5 s; pit-lane time loss 20-25 s; Imola outlier >30 s).
20. Raceteq, "The science behind tyre degradation in Formula 1"; Catapult; Motorsport-Metrics (abrasion, graining, blistering, thermal degradation mechanisms).