"""Contract tests for the HTTP service.

Exercised through Flask's test client against the real loaded model, because
the things most likely to break are the joins between layers: a validated
request the pipeline still rejects, a response that will not serialise, an
error path that returns HTML.

Skipped rather than failed when the artefacts are absent, so a checkout that
has not been trained yet still runs a green suite.
"""
from __future__ import annotations

import json

import pytest

from src import config

flask = pytest.importorskip("flask", reason="service extras not installed")


@pytest.fixture(scope="module")
def client():
    """A test client over the real model, loaded once for the module."""
    from src.core.pipeline import F1Pipeline
    from src.service.app import create_app

    try:
        pipeline = F1Pipeline().load()
    except FileNotFoundError as exc:
        pytest.skip(f"model artefacts not built: {exc}")

    app = create_app(pipeline=pipeline)
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def post(client, payload):
    """POST a prediction request as JSON."""
    return client.post("/predict", data=json.dumps(payload),
                       content_type="application/json")


# ------------------------------------------------------------------- probes


def test_health_needs_nothing(client) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.get_json()["status"] == "ok"


def test_ready_reports_the_loaded_corpus(client) -> None:
    response = client.get("/ready")
    assert response.status_code == 200
    body = response.get_json()
    assert body["status"] == "ready"
    assert body["stints"] > 0


def test_meta_states_the_gate_failed(client) -> None:
    """The headline result is a failed acceptance gate. A caller reading the
    API must not have to go looking for that."""
    body = client.get("/meta").get_json()
    assert body["acceptance"]["passed"] is False
    assert body["holdout"]["oracle_ceiling_pct"] > 0
    assert body["scope"]["conditions"] == "dry running only"


def test_circuits_lists_what_predict_accepts(client) -> None:
    """Every advertised circuit must actually serve a prediction."""
    body = client.get("/circuits").get_json()
    assert body["count"] >= 10
    for entry in body["circuits"]:
        assert post(client, {"circuit": entry["circuit"]}).status_code == 200


# -------------------------------------------------------------- predictions


def test_predict_returns_the_documented_contract(client) -> None:
    response = post(client, {
        "season": 2024, "circuit": "silverstone", "laps": 52,
        "air_temp": 20, "track_temp": 32, "conditions": "dry",
        "compounds": ["SOFT", "MEDIUM", "HARD"]})
    assert response.status_code == 200

    body = response.get_json()
    for key in ("best_strategy", "alternatives", "confidence", "wear_curves",
                "lap_times", "pit_windows", "degradation", "flags"):
        assert key in body, key

    best = body["best_strategy"]
    assert best["stops"] in (1, 2)
    assert len(best["pit_laps"]) == best["stops"]
    assert len(body["lap_times"]) == 52
    assert sum(s["laps"] for s in best["stints"]) == 52
    assert body["confidence"]["level"] in ("low", "moderate", "high")


def test_circuit_is_the_only_required_field(client) -> None:
    """Defaults must produce an answer, not a 400."""
    response = post(client, {"circuit": "monza"})
    assert response.status_code == 200
    assert response.get_json()["best_strategy"]["stops"] >= 1


def test_response_carries_the_server_timing_header(client) -> None:
    response = post(client, {"circuit": "monza"})
    assert float(response.headers["X-Server-Duration-Ms"]) > 0


def test_out_of_scope_season_is_a_flag_not_an_error(client) -> None:
    """The model covers 2022-2024. A 2019 request is answerable and wrong to
    silently pretend otherwise, so it answers and says so."""
    response = post(client, {"circuit": "monza", "season": 2019})
    assert response.status_code == 200
    flags = response.get_json()["flags"]
    assert any(f.startswith("season_out_of_scope") for f in flags)


def test_wet_conditions_are_flagged(client) -> None:
    response = post(client, {"circuit": "spa", "conditions": "wet"})
    assert response.status_code == 200
    body = response.get_json()
    assert any(f.startswith("dry_model_only") for f in body["flags"])
    assert body["confidence"]["level"] == "low"


def test_a_single_compound_is_widened_to_meet_the_rule(client) -> None:
    """A dry race must use two compounds; asking for one is a request the
    regulations forbid, not one the service cannot represent."""
    response = post(client, {"circuit": "monza", "compounds": ["SOFT"]})
    assert response.status_code == 200
    assert any(f.startswith("compound_set_widened")
               for f in response.get_json()["flags"])


# ------------------------------------------------------------- bad requests


@pytest.mark.parametrize("payload,field", [
    ({}, "circuit"),
    ({"circuit": ""}, "circuit"),
    ({"circuit": "nurburgring"}, "circuit"),
    ({"circuit": 42}, "circuit"),
    ({"circuit": "monza", "laps": 0}, "laps"),
    ({"circuit": "monza", "laps": 5000}, "laps"),
    ({"circuit": "monza", "laps": 52.5}, "laps"),
    ({"circuit": "monza", "laps": "many"}, "laps"),
    ({"circuit": "monza", "air_temp": 500}, "air_temp"),
    ({"circuit": "monza", "track_temp": -300}, "track_temp"),
    ({"circuit": "monza", "conditions": "damp"}, "conditions"),
    ({"circuit": "monza", "compounds": []}, "compounds"),
    ({"circuit": "monza", "compounds": ["WET"]}, "compounds"),
    ({"circuit": "monza", "compounds": "SOFT"}, "compounds"),
    ({"circuit": "monza", "compounds": [1, 2]}, "compounds"),
])
def test_malformed_requests_are_400_with_the_field(client, payload, field
                                                   ) -> None:
    """A bad request must never reach the model, and must say what was bad."""
    response = post(client, payload)
    assert response.status_code == 400, response.get_data(as_text=True)
    body = response.get_json()
    assert body["error"] == "invalid_request"
    assert body.get("field") == field


def test_unknown_circuit_lists_the_known_ones(client) -> None:
    body = post(client, {"circuit": "nurburgring"}).get_json()
    assert "monza" in body["allowed"]


def test_unrecognised_field_is_rejected(client) -> None:
    """Silently ignoring an unknown field lets a caller believe a setting took
    effect when it did not."""
    response = post(client, {"circuit": "monza", "tyre_pressure": 23.0})
    assert response.status_code == 400
    assert "tyre_pressure" in response.get_json()["message"]


def test_boolean_is_not_a_temperature(client) -> None:
    """``True`` is an ``int`` in Python and would otherwise pass as 1 degree."""
    response = post(client, {"circuit": "monza", "air_temp": True})
    assert response.status_code == 400


def test_non_json_body_is_400(client) -> None:
    response = client.post("/predict", data="circuit=monza",
                           content_type="application/x-www-form-urlencoded")
    assert response.status_code == 400
    assert response.get_json()["error"] == "invalid_request"


def test_malformed_json_is_400_not_500(client) -> None:
    response = client.post("/predict", data="{not json",
                           content_type="application/json")
    assert response.status_code == 400


def test_oversized_body_is_rejected(client) -> None:
    limit = config.settings()["service"]["max_content_length_bytes"]
    response = client.post("/predict", data="x" * (limit + 1024),
                           content_type="application/json")
    assert response.status_code == 413


def test_unknown_route_returns_json(client) -> None:
    response = client.get("/predict/best")
    assert response.status_code == 404
    assert response.get_json()["error"] == "not_found"


def test_wrong_method_returns_json(client) -> None:
    response = client.get("/predict")
    assert response.status_code == 405
    assert response.get_json()["error"] == "method_not_allowed"


# ------------------------------------------------------------ concurrency


def test_concurrent_predictions_agree_with_serial_ones() -> None:
    """The service shares one loaded pipeline across waitress worker threads.
    If any part of the prediction path held mutable state, this is where it
    would show.

    Driven against the pipeline rather than the test client: Flask's test
    client keeps its request context in a ``ContextVar`` and is not safe to
    call from several threads at once, so a failure here would be the client's
    and not the model's. Concurrency over real HTTP is measured instead by
    ``scripts/run_load_test.py``.
    """
    from concurrent.futures import ThreadPoolExecutor

    from src.core.pipeline import F1Pipeline

    try:
        pipeline = F1Pipeline().load()
    except FileNotFoundError as exc:
        pytest.skip(f"model artefacts not built: {exc}")

    circuits = ["silverstone", "monaco", "monza", "spa", "bahrain",
                "suzuka", "barcelona", "hungaroring"]
    params = {c: {"circuit": c, "conditions": "dry"} for c in circuits}
    expected = {c: pipeline.predict_strategy(params[c])["best_strategy"]
                for c in circuits}

    def predict(circuit: str):
        return circuit, pipeline.predict_strategy(params[circuit])

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(predict, circuits * 8))

    assert len(results) == len(circuits) * 8
    for circuit, body in results:
        assert body["best_strategy"] == expected[circuit], circuit
