"""Flask application serving the trained model over HTTP.

Shape of the thing:

- the model and the reference tables load **once**, at start-up, not per
  request. Loading costs about two seconds and reads two parquet files; doing
  it per request would make the service slower than the pipeline it wraps;
- ``/predict`` is pure computation over that shared state. Nothing is written,
  nothing is cached, nothing accumulates, so a worker's behaviour on its
  hundred-thousandth request is the same as on its first;
- a request that cannot be served is a 400 with the reason and, where a short
  list exists, the permitted values. An unhandled exception is a 500 with an
  identifier and nothing else, since the caller cannot act on a stack trace
  and should not be shown one.

Flask rather than FastAPI deliberately. FastAPI pulls in pydantic v2, which
ships compiled wheels; this venv holds a pinned scientific stack that is
tedious to reconstruct and the validation here is a hundred lines of plain
Python. See ``schema.py``.

Run it with ``scripts/run_service.py``, which serves this app through
waitress. Flask's built-in server is for development only and is not used.
"""
from __future__ import annotations

import logging
import time
import uuid

from flask import Flask, g, jsonify, request

from src import config
from src.core.pipeline import F1Pipeline
from src.service.schema import ValidationError, validate
from src.simulation import strategy as strat

logger = logging.getLogger(__name__)


def create_app(pipeline: F1Pipeline | None = None,
               access_log: bool | None = None) -> Flask:
    """Build the application.

    :param pipeline: a loaded pipeline. When omitted one is loaded here, which
        is what the production entry point does; tests pass their own.
    :param access_log: log a line per prediction. Defaults to the
        ``service.access_log`` setting; see the note in ``_register_timing``.
    :returns: the configured Flask application.
    :raises FileNotFoundError: if the model artefacts have not been built.
    """
    settings = config.settings()["service"]

    app = Flask(__name__)
    app.config["MAX_CONTENT_LENGTH"] = settings["max_content_length_bytes"]
    app.config["ACCESS_LOG"] = (settings["access_log"] if access_log is None
                                else access_log)
    # Flask 2.3 moved this onto the JSON provider; the old config key is
    # ignored. Insertion order is more readable than alphabetical here, since
    # best_strategy should not sort below alternatives.
    app.json.sort_keys = False

    if pipeline is None:
        started = time.perf_counter()
        pipeline = F1Pipeline().load()
        logger.info("Model and reference loaded in %.2f s",
                    time.perf_counter() - started)

    app.config["PIPELINE"] = pipeline
    app.config["MODEL_CARD"] = _model_card(pipeline)

    _register_routes(app)
    _register_error_handlers(app)
    _register_timing(app)
    return app


# --------------------------------------------------------------------- routes


def _register_routes(app: Flask) -> None:
    """Attach the endpoints."""

    @app.get("/health")
    def health():
        """Liveness. Answers as long as the process is up."""
        return jsonify({"status": "ok"})

    @app.get("/ready")
    def ready():
        """Readiness. Distinguished from liveness because loading the model
        takes seconds, and a load balancer should not send traffic during it."""
        pipeline = app.config.get("PIPELINE")
        if pipeline is None or pipeline.reference is None:
            return jsonify({"status": "loading"}), 503
        return jsonify({"status": "ready",
                        "stints": pipeline.reference["n_stints"]})

    @app.get("/meta")
    def meta():
        """What the model is, what it was measured at, and where it stops.

        Served rather than documented so a caller reading the API cannot miss
        that the headline gate failed.
        """
        return jsonify(app.config["MODEL_CARD"])

    @app.get("/circuits")
    def circuits():
        """Circuits the service accepts, with race distance and evidence."""
        pipeline = app.config["PIPELINE"]
        pace = pipeline.reference["pace_index"]
        compound_index = pipeline.reference["compound_index"]

        out = []
        for slug, entry in sorted(config.ui_circuit_map()["circuits"].items()):
            event = entry["event_name"]
            if event not in pace:
                continue
            observed = [c for c in strat.DRY_COMPOUNDS
                        if (event, c) in compound_index
                        and compound_index[(event, c)]["reliable"]]
            out.append({
                "circuit": slug,
                "name": entry["ui_label"],
                "event": event,
                "race_laps": pace[event]["race_laps"],
                "compounds_from_observed_stints": observed,
            })
        return jsonify({"circuits": out, "count": len(out)})

    @app.post("/predict")
    def predict():
        """Return an optimal tyre strategy for the requested race.

        400 if the request cannot be served, 200 otherwise. A 200 may still
        carry ``flags`` saying the answer rests on thin evidence; read them.
        """
        payload = request.get_json(silent=True)
        if payload is None:
            raise ValidationError(
                "request body must be JSON with Content-Type: application/json")

        params = validate(payload)
        pipeline = app.config["PIPELINE"]
        try:
            result = pipeline.predict_strategy(params)
        except ValueError as exc:
            # predict_strategy rejects circuits and compound sets that got
            # past validation, which means the reference tables and the
            # circuit map disagree. That is a 400 for the caller, but it is
            # also a deployment fault, so it is logged as one.
            logger.warning("Pipeline rejected a validated request %s: %s",
                           params, exc)
            raise ValidationError(str(exc)) from exc

        result["request"] = params
        return jsonify(result)


def _model_card(pipeline: F1Pipeline) -> dict:
    """Assemble the static metadata served at ``/meta``.

    Built once, since none of it changes between requests.
    """
    metrics = pipeline.metrics or {}
    scores = metrics.get("metrics", {})
    return {
        "model": "xgboost",
        "target": "deg_rate",
        "units": "seconds of lap time lost per lap of tyre age",
        "training": {
            "seasons": metrics.get("train_seasons"),
            "holdout_season": metrics.get("holdout_season"),
            "n_train_stints": metrics.get("n_train"),
            "n_holdout_stints": metrics.get("n_holdout"),
            "n_reference_stints": pipeline.reference["n_stints"],
            "n_features": len(pipeline.features),
        },
        "holdout": {
            "mean_predictor_mae": scores.get("mean_predictor", {}).get("MAE"),
            "baseline_mae": scores.get("baseline", {}).get("MAE"),
            "xgboost_mae": scores.get("xgboost", {}).get("MAE"),
            "xgboost_r2": scores.get("xgboost", {}).get("R2"),
            "improvement_vs_baseline_pct": metrics.get("improvement_pct"),
            "oracle_ceiling_pct": metrics.get("oracle_ceiling_pct"),
            "share_of_headroom_pct": metrics.get("share_of_headroom_pct"),
        },
        "acceptance": {
            "gate": "XGBoost must beat the linear baseline MAE by 15%",
            "passed": metrics.get("gate4_pass"),
            "note": ("The gate fails. The model beats the mean predictor and "
                     "roughly doubles the baseline R2, but not its absolute "
                     "error. Predicting each stint's own event-and-compound "
                     "mean, which requires knowing the answer, improves on "
                     "the mean predictor by only 38%, so most of the variance "
                     "is within events rather than between them."),
        },
        "scope": {
            "seasons": config.ui_circuit_map()["supported_seasons"],
            "conditions": "dry running only",
            "degradation": ("linear in tyre age; the non-linear cliff beyond "
                            "a tyre's working range is not modelled"),
            "compounds": list(strat.DRY_COMPOUNDS),
        },
    }


# ------------------------------------------------------------ cross-cutting


def _register_timing(app: Flask) -> None:
    """Time every request and report it on the response.

    The header is there so a load generator can separate the prediction from
    the queueing and transport around it without the two being conflated in
    one client-side number.

    A line per request is written only when ``ACCESS_LOG`` is set, and it is
    off by default, because a reverse proxy in front of the service records
    the same thing. It is not off for cost: ``setup_logging`` attaches a file
    handler and ``logging`` writes through it on the request's own thread, so
    it looked like an obvious throughput tax, but measuring the sweep both
    ways gave 67.1 against 68.2 requests/s, which is inside the run-to-run
    spread. Failures are logged either way.
    """

    @app.before_request
    def _start_timer():
        g.started = time.perf_counter()

    @app.after_request
    def _record(response):
        started = getattr(g, "started", None)
        if started is not None:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            response.headers["X-Server-Duration-Ms"] = f"{elapsed_ms:.2f}"
            if app.config["ACCESS_LOG"] and request.path == "/predict":
                logger.info("%s %s -> %d in %.2f ms", request.method,
                            request.path, response.status_code, elapsed_ms)
        return response


def _register_error_handlers(app: Flask) -> None:
    """Return JSON for every failure, never HTML and never a stack trace."""

    @app.errorhandler(ValidationError)
    def _invalid(exc: ValidationError):
        return jsonify(exc.as_dict()), 400

    @app.errorhandler(404)
    def _not_found(_exc):
        return jsonify({"error": "not_found",
                        "message": f"no route for {request.path}",
                        "routes": ["/health", "/ready", "/meta", "/circuits",
                                   "POST /predict"]}), 404

    @app.errorhandler(405)
    def _wrong_method(_exc):
        return jsonify({"error": "method_not_allowed",
                        "message": f"{request.method} is not allowed on "
                                   f"{request.path}"}), 405

    @app.errorhandler(413)
    def _too_large(_exc):
        limit = app.config["MAX_CONTENT_LENGTH"]
        return jsonify({"error": "request_too_large",
                        "message": f"request body exceeds {limit} bytes"}), 413

    @app.errorhandler(Exception)
    def _unhandled(exc: Exception):
        # The identifier is the only thing tying the caller's failed request
        # to the traceback in the log, so it goes in both.
        incident = uuid.uuid4().hex[:12]
        logger.exception("Unhandled error [%s] on %s %s", incident,
                         request.method, request.path)
        return jsonify({"error": "internal_error", "incident": incident,
                        "message": "the request could not be completed"}), 500
