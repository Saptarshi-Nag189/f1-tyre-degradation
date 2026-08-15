"""Modular F1 tyre degradation pipeline, and the UI-facing prediction surface.

The class interface is preserved from the superseded implementation because
``user_ui/app.js`` is already shaped around it: the stage list, the
``callback(stage, progress, message)`` signature, and above all the
``predict_strategy`` return contract of ``best_strategy``, ``alternatives``,
``confidence``, ``wear_curves``, ``lap_times`` and ``pit_windows``.

The body is entirely new. The previous version inserted paths onto ``sys.path``
at call time and imported a training script that no longer exists.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from src import config
from src.simulation import reference, strategy as strat

logger = logging.getLogger(__name__)


class F1Pipeline:
    """Orchestrates dataset assembly, training, and strategy prediction.

    Usage (script)::

        pipeline = F1Pipeline(project_root=config.PROJECT_ROOT)
        results = pipeline.run_full_pipeline()

    Usage (UI)::

        pipeline = F1Pipeline(project_root=..., config_overrides={...})
        pipeline.load()
        result = pipeline.predict_strategy({"circuit": "silverstone", ...})
    """

    STAGES = ["setup", "load_stints", "build_features", "build_target",
              "train", "evaluate", "save"]

    def __init__(self, project_root: Path | None = None,
                 config_overrides: dict | None = None) -> None:
        self.project_root = Path(project_root or config.PROJECT_ROOT)
        self.config_overrides = config_overrides or {}
        self.config = config.settings()
        self._apply_overrides()
        self.model = None
        self.features: list[str] = []
        self.reference: dict | None = None
        self.metrics: dict | None = None
        self._callback: Optional[Callable] = None
        # Guards the booster during inference. See _predict_deg_rate.
        self._predict_lock = threading.Lock()

    # --------------------------------------------------------------- training

    def run_full_pipeline(self, callback: Optional[Callable] = None) -> dict:
        """Assemble the dataset, train, and evaluate, end to end.

        :param callback: optional ``fn(stage, progress, message)`` for UI
            progress; ``progress`` runs 0.0 to 1.0.
        :returns: mapping with metrics, the model path and the feature list.
        """
        import joblib

        from src.features import assemble
        from src.features import target as target_mod
        from src.modelling import baseline, features as feat, splits, xgb_model

        self._callback = callback
        model_cfg = self.config["modelling"]

        self._report("setup", 0.0, "Loading configuration")

        self._report("load_stints", 0.10, "Assembling sessions into stints")
        laps, stints = assemble.build_dataset()
        if stints.empty:
            raise RuntimeError("No usable stints; check collection and Gate 2.")

        self._report("build_target", 0.35, "Building the CWI target")
        target_cfg = self.config["target"]
        stints, cwi_meta = target_mod.validate_and_build_cwi(
            stints,
            keep_thr=target_cfg["spearman_keep_threshold"],
            downweight_low=target_cfg["spearman_downweight_low"],
            energy_w_full=target_cfg["energy_weight_full"],
            laptime_w_down=target_cfg["laptime_weight_downweight"])

        processed = config.resolve_path("processed", create=True)
        laps.to_parquet(processed / "laps_clean.parquet", index=False)
        stints.to_parquet(processed / "stints_target.parquet", index=False)

        self._report("build_features", 0.50, "Selecting features")
        train, holdout = splits.season_holdout(
            stints, model_cfg["train_seasons"], model_cfg["holdout_season"])
        numeric = feat.available(stints.columns)
        categories = feat.training_categories(train)
        x_train, columns = feat.build_matrix(train, numeric, categories)
        x_holdout, _ = feat.build_matrix(holdout, numeric, categories)

        self._report("train", 0.65, "Fitting the model")
        params = None
        tuned_path = config.CONFIG_DIR / "tuned_params.json"
        if tuned_path.exists():
            import json
            saved = json.loads(tuned_path.read_text(encoding="utf-8"))
            if saved.get("features") == columns:
                params = saved["params"]

        train_matrix = pd.concat(
            [train.reset_index(drop=True), x_train.reset_index(drop=True)[
                [c for c in columns if c not in train.columns]]], axis=1)
        holdout_matrix = pd.concat(
            [holdout.reset_index(drop=True), x_holdout.reset_index(drop=True)[
                [c for c in columns if c not in holdout.columns]]], axis=1)

        result = xgb_model.train_xgb(
            train_matrix, holdout_matrix, columns, "deg_rate", params)

        self._report("evaluate", 0.85, "Evaluating against the baseline")
        base = baseline.train_baseline(
            train_matrix, holdout_matrix, columns, "deg_rate")
        improvement = 100.0 * (base["metrics"]["MAE"]
                               - result["metrics"]["MAE"]) / base["metrics"]["MAE"]

        self._report("save", 0.95, "Saving artefacts")
        models_dir = config.resolve_path("models", create=True)
        model_path = models_dir / "xgb_deg_rate.joblib"
        joblib.dump({"model": result["model"], "features": columns,
                     "target": "deg_rate",
                     "medians": base["medians"].to_dict()}, model_path)

        self.model = result["model"]
        self.features = columns
        self.metrics = {"metrics": {"xgboost": result["metrics"],
                                    "baseline": base["metrics"]},
                        "improvement_pct": improvement,
                        "train_seasons": sorted(train["Year"].unique().tolist()),
                        "holdout_season": model_cfg["holdout_season"]}

        self._report("save", 1.0, "Pipeline complete")
        return {"metrics": result["metrics"], "baseline": base["metrics"],
                "improvement_pct": improvement, "cwi": cwi_meta,
                "model_path": str(model_path), "features": columns,
                "n_train": len(train), "n_holdout": len(holdout)}

    # ---------------------------------------------------------------- loading

    def load(self) -> "F1Pipeline":
        """Load the trained model and the empirical reference tables.

        :returns: self, for chaining.
        :raises FileNotFoundError: if the artefacts have not been built.
        """
        import joblib

        model_path = config.resolve_path("models") / "xgb_deg_rate.joblib"
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run scripts/run_train.py first.")
        bundle = joblib.load(model_path)
        self.model = bundle["model"]
        self.features = bundle["features"]

        processed = config.resolve_path("processed")
        laps = pd.read_parquet(processed / "laps_clean.parquet")
        stints = pd.read_parquet(processed / "stints_target.parquet")
        self.reference = reference.build_reference(laps, stints)

        report = config.resolve_path("reports") / "train_deg_rate.json"
        if report.exists():
            import json
            self.metrics = json.loads(report.read_text(encoding="utf-8"))

        logger.info("Loaded model (%d features) and reference over seasons %s",
                    len(self.features), self.reference["seasons"])
        return self

    # ------------------------------------------------------------- prediction

    def predict_strategy(self, params: dict) -> dict:
        """Return an optimal tyre strategy for the requested race parameters.

        :param params: ``{season, circuit, driver, laps, air_temp, track_temp,
            conditions, compounds}``. ``circuit`` is a front-end slug.
        :returns: mapping with ``best_strategy``, ``alternatives``,
            ``confidence``, ``wear_curves``, ``lap_times``, ``pit_windows``
            and ``flags``.
        :raises RuntimeError: if :meth:`load` has not been called.
        :raises ValueError: if the circuit or season is unsupported.
        """
        if self.reference is None:
            raise RuntimeError("Call load() before predict_strategy().")

        flags: list[str] = []
        ui_map = config.ui_circuit_map()

        # --- resolve the circuit ---
        slug = str(params.get("circuit", "")).lower()
        entry = ui_map["circuits"].get(slug)
        if entry is None:
            raise ValueError(
                f"Unsupported circuit '{slug}'. Known: {sorted(ui_map['circuits'])}")
        event_name = entry["event_name"]

        # --- season scope: this is a single-regulation-era model ---
        season = params.get("season")
        if season is not None and int(season) not in ui_map["supported_seasons"]:
            flags.append(
                f"season_out_of_scope: trained on {ui_map['supported_seasons']} "
                "(ground-effect era); this prediction is an extrapolation")

        # --- conditions: the model is dry-only ---
        conditions = str(params.get("conditions", "dry")).lower()
        if conditions != "dry":
            flags.append("dry_model_only: wet and intermediate stints are "
                         "excluded from the target fit, so no wet prediction "
                         "is offered")

        # --- circuit pace and race length ---
        pace = self.reference["pace_index"].get(event_name)
        if pace is None:
            raise ValueError(f"No collected data for {event_name}")
        base_lap_s = pace["base_lap_s"]
        race_laps = int(params.get("laps") or pace["race_laps"])

        # --- degradation per compound ---
        requested = [c.upper() for c in params.get("compounds", strat.DRY_COMPOUNDS)]
        compounds = [c for c in requested if c in strat.DRY_COMPOUNDS]
        if len(compounds) < 2:
            compounds = list(strat.DRY_COMPOUNDS)
            flags.append("compound_set_widened: a dry race must use at least "
                         "two compounds")

        deg_table, max_observed = self._degradation_table(
            event_name, compounds, params, flags)

        def lookup(compound: str):
            entry = deg_table[compound]
            # The measured rate is reported unchanged in the response; only the
            # simulated one is floored, because a tyre does not get faster with
            # age and an unbounded negative rate would reward infinite stints.
            return (max(entry["deg_rate"], strat.MIN_SIMULATED_DEG_RATE),
                    entry["base_offset"], entry["source"])

        # --- simulate ---
        settings = self.config
        slack = settings["strategy"]["pit_window_slack_s"]
        results = strat.enumerate_strategies(
            race_laps, compounds, lookup, base_lap_s,
            settings["strategy"]["pit_loss_s"], settings["fuel_correction"],
            max_observed_stint=max_observed, near_optimal_slack_s=slack)
        if not results:
            raise ValueError("No feasible strategy for the given parameters")

        best = results[0]
        alternatives = strat.dedupe_by_shape(results, limit=4)[1:]

        return {
            "best_strategy": self._describe(best, base_lap_s),
            "alternatives": [self._describe(r, base_lap_s) for r in alternatives],
            "confidence": self._confidence(deg_table, event_name, flags),
            "wear_curves": {
                s.compound: [round(s.deg_rate * age, 4) for age in range(s.n_laps)]
                for s in best.stints},
            "lap_times": [round(t, 3) for t in best.lap_times],
            "pit_windows": self._pit_windows(results, best, slack),
            "degradation": {c: round(v["deg_rate"], 5) for c, v in deg_table.items()},
            "flags": flags,
        }

    # ---------------------------------------------------------------- helpers

    def _degradation_table(self, event_name: str, compounds: list[str],
                           params: dict, flags: list[str]) -> tuple[dict, dict]:
        """Build per-compound degradation rates, preferring observed data.

        Where enough stints have been observed at this circuit on this
        compound, the empirical median is used: per-circuit degradation
        correlates 0.66-0.71 across seasons, whereas the model captures only
        19% of the achievable headroom. The model fills genuine gaps.
        """
        index = self.reference["compound_index"]

        table: dict[str, dict] = {}
        max_observed: dict[str, float] = {}

        for compound in compounds:
            match = index.get((event_name, compound))
            if match is not None and match["reliable"]:
                table[compound] = {
                    "deg_rate": match["deg_rate_median"],
                    "q25": match["deg_rate_q25"],
                    "q75": match["deg_rate_q75"],
                    "n_stints": match["n_stints"],
                    "source": "observed",
                }
                max_observed[compound] = match["mean_stint_length"] * 1.5
            else:
                predicted = self._predict_deg_rate(event_name, compound, params)
                n = match["n_stints"] if match is not None else 0
                table[compound] = {
                    "deg_rate": predicted, "q25": predicted * 0.6,
                    "q75": predicted * 1.4, "n_stints": n, "source": "model",
                }
                flags.append(f"model_estimate: {compound} at {event_name} has "
                             f"only {n} observed stints")

        # Pace offsets, anchored to the softest requested compound.
        softest = min(compounds, key=lambda c: strat.DEFAULT_PACE_OFFSET[c])
        for compound in compounds:
            table[compound]["base_offset"] = (
                strat.DEFAULT_PACE_OFFSET[compound]
                - strat.DEFAULT_PACE_OFFSET[softest])
        return table, max_observed

    def _predict_deg_rate(self, event_name: str, compound: str,
                          params: dict) -> float:
        """Predict a degradation rate from the trained model.

        Serialised on ``_predict_lock``. Everything else in ``predict_strategy``
        reads immutable state and is safe to run concurrently, but an XGBoost
        booster keeps internal prediction buffers and is not documented as
        thread-safe. The lock is narrow on purpose: most requests never reach
        it, because a circuit with enough observed stints uses the empirical
        rate and the model is only consulted where the evidence runs out.
        """
        track = config.track_traits()
        keys = track["event_name_to_key"].get(event_name)
        traits = track["circuits"].get(keys["traits"]) if keys else None

        relative = {"HARD": 0.0, "MEDIUM": 1.0, "SOFT": 2.0}[compound]
        values = {
            "TyreLife_start": 1.0,
            "compound_ordinal": np.nan,
            "compound_relative": relative,
            "AirTemp_mean": float(params.get("air_temp") or 25.0),
            "TrackTemp_mean": float(params.get("track_temp") or 35.0),
            "abrasion": float(traits["abrasion"]) if traits else np.nan,
            "energy": float(traits["energy"]) if traits else np.nan,
            "composite": float(traits["composite"]) if traits else np.nan,
            "RaceLaps": float(params.get("laps") or 57),
        }
        row = [values.get(f, 0.0) for f in self.features]
        with self._predict_lock:
            prediction = float(
                self.model.predict(np.array([row], dtype=float))[0])
        return max(prediction, 0.005)

    def _describe(self, result: "strat.StrategyResult", base_lap_s: float) -> dict:
        """Convert a simulated strategy into the UI's expected shape."""
        return {
            "label": result.label,
            "stops": result.pit_stops,
            "pit_laps": result.pit_laps,
            "total_time_s": round(result.total_time_s, 2),
            "delta_to_best_s": None,
            "stints": [{"compound": s.compound, "start_lap": s.start_lap,
                        "laps": s.n_laps, "deg_rate": round(s.deg_rate, 5),
                        "extrapolated": s.extrapolated}
                       for s in result.stints],
        }

    def _pit_windows(self, results: list, best, slack_s: float) -> list[dict]:
        """Pit-lap ranges that stay within ``slack_s`` of the best strategy.

        :param results: ranked strategies from ``enumerate_strategies``.
        :param best: the chosen strategy.
        :param slack_s: acceptable loss against the optimum, in seconds. Must
            match the slack the search was pruned with.
        :returns: one window per stop.
        """
        shape = tuple(s.compound for s in best.stints)
        same = [r for r in results
                if tuple(s.compound for s in r.stints) == shape
                and r.total_time_s <= best.total_time_s + slack_s]
        windows = []
        for position in range(len(best.pit_laps)):
            laps = sorted({r.pit_laps[position] for r in same
                           if len(r.pit_laps) > position})
            if laps:
                windows.append({"stop": position + 1, "earliest": laps[0],
                                "latest": laps[-1], "optimal": best.pit_laps[position]})
        return windows

    def _confidence(self, deg_table: dict, event_name: str,
                    flags: list[str]) -> dict:
        """Report an honest confidence, derived from evidence not decoration.

        Combines how much of the estimate rests on observed data rather than
        the model, the spread of those observations, and the model's own
        holdout error.
        """
        observed = [v for v in deg_table.values() if v["source"] == "observed"]
        share = len(observed) / max(len(deg_table), 1)
        total_stints = sum(v["n_stints"] for v in deg_table.values())
        spread = float(np.mean([v["q75"] - v["q25"] for v in deg_table.values()]))

        if share == 1.0 and total_stints >= 30:
            level = "high"
        elif share >= 0.5:
            level = "moderate"
        else:
            level = "low"
        if flags and any(f.startswith(("season_out_of_scope", "dry_model_only"))
                         for f in flags):
            level = "low"

        return {
            "level": level,
            "observed_compound_share": round(share, 2),
            "supporting_stints": total_stints,
            "deg_rate_iqr_s_per_lap": round(spread, 4),
            "model_holdout_mae_s_per_lap": (
                round(self.metrics["metrics"]["xgboost"]["MAE"], 5)
                if self.metrics else None),
            "note": ("Degradation is modelled as linear in tyre age. Real tyres "
                     "fall off non-linearly beyond their working range, which "
                     "this simulator does not represent."),
        }

    # ------------------------------------------------------------------ misc

    @staticmethod
    def get_default_config() -> dict:
        """Return the full settings mapping, for a UI to auto-populate from."""
        return config.settings()

    def _apply_overrides(self) -> None:
        """Deep-merge ``config_overrides`` into the loaded settings."""
        def merge(base: dict, override: dict) -> None:
            for key, value in override.items():
                if isinstance(value, dict) and isinstance(base.get(key), dict):
                    merge(base[key], value)
                else:
                    base[key] = value
        if self.config:
            merge(self.config, self.config_overrides)

    def _report(self, stage: str, progress: float, message: str) -> None:
        """Report progress to the callback if one is set."""
        logger.info("[%s] %s", stage, message)
        if self._callback:
            self._callback(stage, progress, message)
