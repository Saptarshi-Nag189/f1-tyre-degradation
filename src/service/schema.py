"""Request validation for the prediction endpoint.

Kept separate from the Flask layer so it can be tested without a client, and
so the rules are readable in one place rather than scattered through a view.

The distinction this module draws, and the reason it exists:

- A request the simulator cannot represent is a **400**. Zero laps, a
  temperature of 500 C, a circuit that was never collected.
- A request outside the model's evidence but still well formed is a **200
  carrying a flag**. A 2021 season, a wet race, a circuit-compound pair with
  too few observed stints.

Turning the second kind into an error would be dishonest in the other
direction: the response already says how much of the answer rests on observed
data, and refusing to answer would hide that judgement rather than expose it.
"""
from __future__ import annotations

from typing import Any

from src import config
from src.simulation import strategy as strat


class ValidationError(ValueError):
    """A request that cannot be served, with the field that caused it.

    :param message: what is wrong, in terms the caller can act on.
    :param field: the offending request field, if a single one is to blame.
    :param allowed: the permitted values, where a short list exists.
    """

    def __init__(self, message: str, field: str | None = None,
                 allowed: list | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.field = field
        self.allowed = allowed

    def as_dict(self) -> dict[str, Any]:
        """Render as the JSON body of a 400 response."""
        body: dict[str, Any] = {"error": "invalid_request",
                                "message": self.message}
        if self.field:
            body["field"] = self.field
        if self.allowed is not None:
            body["allowed"] = self.allowed
        return body


def _as_number(value: Any, field: str) -> float:
    """Coerce to float, rejecting booleans and non-numeric strings.

    ``bool`` is excluded explicitly because it is a subclass of ``int`` and
    ``True`` would otherwise be accepted as a temperature of 1 C.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise ValidationError(f"{field} must be a number", field=field)
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise ValidationError(f"{field} must be a number, got {value!r}",
                              field=field) from None
    if number != number or number in (float("inf"), float("-inf")):
        raise ValidationError(f"{field} must be finite", field=field)
    return number


def _as_integer(value: Any, field: str) -> int:
    """Coerce to int, rejecting fractional values."""
    number = _as_number(value, field)
    if number != int(number):
        raise ValidationError(f"{field} must be a whole number", field=field)
    return int(number)


def validate(payload: Any) -> dict[str, Any]:
    """Validate a prediction request and normalise it for the pipeline.

    :param payload: the decoded JSON request body.
    :returns: parameters in the shape ``F1Pipeline.predict_strategy`` expects.
    :raises ValidationError: if the request cannot be served.
    """
    if not isinstance(payload, dict):
        raise ValidationError("request body must be a JSON object")

    settings = config.settings()["service"]["limits"]
    known = config.ui_circuit_map()["circuits"]

    unknown_fields = set(payload) - {
        "circuit", "season", "laps", "air_temp", "track_temp", "conditions",
        "compounds", "driver"}
    if unknown_fields:
        raise ValidationError(
            f"unrecognised field(s): {', '.join(sorted(unknown_fields))}",
            allowed=sorted({"circuit", "season", "laps", "air_temp",
                            "track_temp", "conditions", "compounds", "driver"}))

    # --- circuit: the one required field ---
    circuit = payload.get("circuit")
    if circuit is None or not isinstance(circuit, str) or not circuit.strip():
        raise ValidationError("circuit is required", field="circuit",
                              allowed=sorted(known))
    circuit = circuit.strip().lower()
    if circuit not in known:
        raise ValidationError(f"unknown circuit '{circuit}'", field="circuit",
                              allowed=sorted(known))

    params: dict[str, Any] = {"circuit": circuit}

    # --- season: out of scope is a flag on the response, not an error ---
    if payload.get("season") is not None:
        params["season"] = _as_integer(payload["season"], "season")

    # --- race distance ---
    if payload.get("laps") is not None:
        laps = _as_integer(payload["laps"], "laps")
        if not settings["min_race_laps"] <= laps <= settings["max_race_laps"]:
            raise ValidationError(
                f"laps must be between {settings['min_race_laps']} and "
                f"{settings['max_race_laps']}", field="laps")
        params["laps"] = laps

    # --- temperatures ---
    for field in ("air_temp", "track_temp"):
        if payload.get(field) is None:
            continue
        temp = _as_number(payload[field], field)
        if not settings["min_temp_c"] <= temp <= settings["max_temp_c"]:
            raise ValidationError(
                f"{field} must be between {settings['min_temp_c']} and "
                f"{settings['max_temp_c']} degrees Celsius", field=field)
        params[field] = temp

    # --- conditions ---
    if payload.get("conditions") is not None:
        conditions = payload["conditions"]
        if not isinstance(conditions, str):
            raise ValidationError("conditions must be a string",
                                  field="conditions")
        conditions = conditions.strip().lower()
        if conditions not in ("dry", "wet", "intermediate"):
            raise ValidationError(
                f"unknown conditions '{conditions}'", field="conditions",
                allowed=["dry", "wet", "intermediate"])
        params["conditions"] = conditions

    # --- compounds ---
    if payload.get("compounds") is not None:
        compounds = payload["compounds"]
        if not isinstance(compounds, list) or not compounds:
            raise ValidationError("compounds must be a non-empty list",
                                  field="compounds",
                                  allowed=list(strat.DRY_COMPOUNDS))
        if len(compounds) > settings["max_compounds"]:
            raise ValidationError(
                f"at most {settings['max_compounds']} compounds",
                field="compounds", allowed=list(strat.DRY_COMPOUNDS))
        normalised = []
        for compound in compounds:
            if not isinstance(compound, str):
                raise ValidationError("compounds must be strings",
                                      field="compounds",
                                      allowed=list(strat.DRY_COMPOUNDS))
            compound = compound.strip().upper()
            if compound not in strat.DRY_COMPOUNDS:
                raise ValidationError(
                    f"unknown compound '{compound}'", field="compounds",
                    allowed=list(strat.DRY_COMPOUNDS))
            if compound not in normalised:
                normalised.append(compound)
        params["compounds"] = normalised

    # --- driver: carried through, not used by the dry model ---
    if payload.get("driver") is not None:
        driver = payload["driver"]
        if not isinstance(driver, str):
            raise ValidationError("driver must be a string", field="driver")
        params["driver"] = driver.strip().upper()

    return params
