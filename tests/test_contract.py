"""Golden-schema tests tying predict_strategy and the UI export to app.js.

These guard the seam between Python and the front end. ``user_ui/app.js``
reads specific key names; renaming one on the Python side would break the page
silently, since JavaScript yields ``undefined`` rather than raising.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
UI_DIR = PROJECT_ROOT.parent / "user_ui"

#: Keys user_ui/app.js consumes from a predict_strategy result.
PREDICT_STRATEGY_KEYS = {
    "best_strategy", "alternatives", "confidence",
    "wear_curves", "lap_times", "pit_windows",
}

#: Keys the export must provide per circuit.
CIRCUIT_KEYS = {"name", "event", "laps", "baseLap", "compounds"}
COMPOUND_KEYS = {"degRate", "q25", "q75", "nStints", "source"}


def test_pipeline_declares_the_documented_contract():
    """The docstring contract must match what the UI actually reads."""
    source = (PROJECT_ROOT / "src" / "core" / "pipeline.py").read_text(encoding="utf-8")
    for key in PREDICT_STRATEGY_KEYS:
        assert f'"{key}"' in source, f"predict_strategy must return {key!r}"


@pytest.mark.skipif(not (UI_DIR / "data" / "model_params.js").exists(),
                    reason="run scripts/run_export_ui.py first")
def test_export_is_a_js_assignment_not_json():
    """user_ui opens from file://, where fetch() of a .json file is blocked by
    CORS and fails silently. The export must be a script assignment."""
    text = (UI_DIR / "data" / "model_params.js").read_text(encoding="utf-8")
    assert "window.F1_MODEL" in text
    assert not (UI_DIR / "data" / "model_params.json").exists(), \
        "a .json export would be silently unreachable from file://"


@pytest.mark.skipif(not (UI_DIR / "data" / "model_params.js").exists(),
                    reason="run scripts/run_export_ui.py first")
def test_export_payload_shape():
    text = (UI_DIR / "data" / "model_params.js").read_text(encoding="utf-8")
    payload = json.loads(re.search(r"window\.F1_MODEL = (\{.*\});\s*$",
                                   text, re.S).group(1))

    assert payload["circuits"], "export contains no circuits"
    for slug, circuit in payload["circuits"].items():
        assert CIRCUIT_KEYS <= set(circuit), f"{slug} missing keys"
        assert circuit["laps"] > 0 and circuit["baseLap"] > 0
        for compound, values in circuit["compounds"].items():
            assert COMPOUND_KEYS <= set(values), f"{slug}/{compound} missing keys"
            assert values["source"] in {"observed", "model"}
            # Not "> 0". A circuit with no measurable degradation, Monaco
            # being the standing example, produces slopes scattered either
            # side of zero, and the export reports the measured value rather
            # than truncating it upward. The physical lower bound is enforced
            # upstream and the simulator floors the value it acts on.
            assert values["degRate"] >= -0.05, "beyond the physical bound"
            assert values["q25"] <= values["q75"], "spread must be ordered"

    meta = payload["meta"]
    for key in ("trainSeasons", "holdoutSeason", "holdoutMAE", "units", "caveat"):
        assert key in meta, f"meta must carry {key} so the page can be honest"


@pytest.mark.skipif(not (UI_DIR / "index.html").exists(), reason="no UI present")
def test_index_loads_the_export_before_app():
    """model_params.js must precede app.js, which reads window.F1_MODEL at
    load time."""
    html = (UI_DIR / "index.html").read_text(encoding="utf-8")
    assert "data/model_params.js" in html
    assert html.index("data/model_params.js") < html.index('src="app.js"')


@pytest.mark.skipif(not (UI_DIR / "app.js").exists(), reason="no UI present")
def test_app_falls_back_loudly_not_silently():
    """If the export is missing the page must warn, not quietly render
    synthetic numbers as though they were model output."""
    app = (UI_DIR / "app.js").read_text(encoding="utf-8")
    assert "window.F1_MODEL" in app
    assert "console.warn" in app
