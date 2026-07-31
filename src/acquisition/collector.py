"""Phase 1: race lap and weather collection, one parquet per session.

Replaces the superseded ``scripts/download_data.py`` and
``src/data_acquisition/collector.py``, whose design lost three collection runs:

- it re-serialised the entire accumulated dictionary after every event, so late
  events rewrote a multi-gigabyte pickle and a kill mid-write truncated the
  whole corpus (both files on disk were unreadable);
- it caught rate limiting by string-matching a message that never arrives,
  because FastF1 swallows ``RateLimitExceededError`` internally and surfaces
  ``DataNotLoadedError`` instead, so the run ploughed on issuing doomed
  requests until the calendar ran out;
- it loaded three sessions per event with full telemetry, roughly five times
  the requests needed, and retained raw telemetry frames at ~245 MB per event.

Here each session is written to its own parquet through a temporary file and an
atomic replace, so a kill costs one session rather than the corpus, and resume
is simply re-running the same command.
"""
from __future__ import annotations

import logging
import os
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

#: Columns retained from session.laps. Everything needed for the clean-lap
#: mask, the stint table, fuel correction and the speed-trap energy proxy.
LAP_COLUMNS = [
    "Driver", "DriverNumber", "Team", "LapNumber", "Stint", "TyreLife",
    "Compound", "FreshTyre", "LapTime", "Sector1Time", "Sector2Time",
    "Sector3Time", "SpeedI1", "SpeedI2", "SpeedFL", "SpeedST",
    "PitInTime", "PitOutTime", "TrackStatus", "IsAccurate", "Position",
    "LapStartTime", "Time",
]

WEATHER_COLUMNS = ["Time", "AirTemp", "TrackTemp", "Humidity", "Pressure",
                   "Rainfall", "WindSpeed", "WindDirection"]


class RateLimitHit(RuntimeError):
    """Raised when the API rate limit is reached, to stop the run cleanly."""


def _is_rate_limit(exc: BaseException) -> bool:
    """Identify rate limiting by exception type, never by message text.

    FastF1 catches ``RateLimitExceededError`` inside ``session.load()`` and the
    caller subsequently sees ``DataNotLoadedError``, so both are treated as
    evidence, along with a bare HTTP 429.
    """
    from fastf1.req import RateLimitExceededError

    if isinstance(exc, RateLimitExceededError):
        return True
    try:
        from fastf1.core import DataNotLoadedError
        if isinstance(exc, DataNotLoadedError):
            return True
    except ImportError:                            # pragma: no cover
        pass
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) == 429


def session_path(out_dir: Path, year: int, round_number: int) -> Path:
    """Return the parquet path for one session."""
    return out_dir / f"laps_{year}_R{round_number:02d}.parquet"


def load_manifest(out_dir: Path) -> dict[str, Any]:
    """Load the collection manifest, tolerating absence or corruption."""
    path = out_dir / "_manifest.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        logger.warning("Manifest unreadable; starting a fresh one")
        return {}


def write_manifest(out_dir: Path, manifest: dict[str, Any]) -> None:
    """Write the manifest atomically."""
    path = out_dir / "_manifest.json"
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(tmp, path)


def _attach_weather(laps: pd.DataFrame, weather: pd.DataFrame) -> pd.DataFrame:
    """Attach weather to laps by nearest session time, preserving row order.

    ``merge_asof`` requires its left frame sorted by the join key, which
    scrambles lap order and silently drops rows whose key is NaT. The reference
    implementation does exactly that. Here the original order is restored and
    any row-count change is logged.

    :param laps: laps frame with a ``Time`` column.
    :param weather: session weather frame.
    :returns: laps with weather columns and a ``weather_imputed`` flag.
    """
    out = laps.copy()
    out["weather_imputed"] = True

    if weather is None or weather.empty or "Time" not in out.columns:
        logger.warning("No weather data available for this session")
        for col in ("AirTemp", "TrackTemp", "Humidity", "Rainfall", "WindSpeed"):
            out[col] = pd.NA
        return out

    weather_cols = [c for c in WEATHER_COLUMNS if c in weather.columns]
    n_before = len(out)

    out["_row_order"] = range(n_before)
    mergeable = out[out["Time"].notna()].sort_values("Time")
    unmergeable = out[out["Time"].isna()]

    merged = pd.merge_asof(
        mergeable,
        weather[weather_cols].sort_values("Time"),
        on="Time", direction="nearest")
    merged["weather_imputed"] = False

    if not unmergeable.empty:
        logger.warning("%d laps have no session Time; weather left unmerged",
                       len(unmergeable))
        merged = pd.concat([merged, unmergeable], ignore_index=True)

    merged = merged.sort_values("_row_order").drop(columns="_row_order")
    merged = merged.reset_index(drop=True)

    if len(merged) != n_before:
        logger.error("Weather merge changed row count: %d -> %d",
                     n_before, len(merged))
    return merged


def collect_session(year: int, round_number: int, session_type: str = "R") -> pd.DataFrame:
    """Load one race session and return its enriched laps frame.

    :param year: season year.
    :param round_number: FIA round number (stable key, never the event name).
    :param session_type: session code, ``"R"`` for race.
    :returns: laps frame with metadata, weather and derived time columns.
    :raises RateLimitHit: if the API rate limit is reached.
    """
    import fastf1

    try:
        session = fastf1.get_session(year, round_number, session_type)
        session.load(laps=True, telemetry=False, weather=True, messages=False)
        laps = session.laps
    except Exception as exc:
        if _is_rate_limit(exc):
            raise RateLimitHit(
                f"Rate limit reached at {year} R{round_number}") from exc
        raise

    if laps is None or laps.empty:
        raise ValueError(f"No laps returned for {year} R{round_number}")

    keep = [c for c in LAP_COLUMNS if c in laps.columns]
    out = laps[keep].copy()

    # --- identity ---
    out["Year"] = int(year)
    out["RoundNumber"] = int(round_number)
    out["SessionType"] = session_type
    try:
        out["EventName"] = str(session.event["EventName"])
    except Exception:                              # pragma: no cover
        out["EventName"] = "UNKNOWN"
        logger.warning("Event name unavailable for %s R%s", year, round_number)

    # --- scheduled race distance, needed later for fuel correction ---
    # Captured now because it is not on the laps frame and re-deriving it after
    # collection would mean reopening every session.
    out["RaceLaps"] = int(pd.to_numeric(laps["LapNumber"], errors="coerce").max())

    # --- time columns: keep the timedelta, add a float mirror ---
    # Downstream code then never has to guess the dtype, which is the source of
    # the reference add_is_clean_lap's fragile double conversion.
    out["lap_s"] = laps["LapTime"].dt.total_seconds()
    for i in (1, 2, 3):
        col = f"Sector{i}Time"
        if col in laps.columns:
            out[f"sector{i}_s"] = laps[col].dt.total_seconds()

    out = _attach_weather(out, session.weather_data)

    # --- dtype hygiene for a stable parquet schema ---
    for col in ("DriverNumber", "TrackStatus", "Driver", "Team", "Compound"):
        if col in out.columns:
            out[col] = out[col].astype(str)
    for col in ("Stint", "TyreLife", "LapNumber", "Position"):
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")

    logger.info("  %s R%02d %-28s %4d laps, %2d drivers, race length %d",
                year, round_number, out["EventName"].iloc[0], len(out),
                out["DriverNumber"].nunique(), out["RaceLaps"].iloc[0])
    return out


def write_session(frame: pd.DataFrame, path: Path) -> None:
    """Write one session's parquet atomically.

    A temporary file plus ``os.replace`` means an interrupted write can never
    leave a truncated parquet, which is how the superseded pipeline destroyed
    5 GB of collected data.

    :param frame: the session laps frame.
    :param path: destination parquet path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".parquet.tmp")
    frame.to_parquet(tmp, engine="pyarrow", index=False)
    os.replace(tmp, path)


def manifest_entry(status: str, *, n_laps: int = 0, n_requests: int = 0,
                   event_name: str = "", detail: str = "") -> dict[str, Any]:
    """Build a manifest record for one session."""
    import fastf1

    return {
        "status": status,
        "n_laps": int(n_laps),
        "n_requests": int(n_requests),
        "event_name": event_name,
        "detail": detail,
        "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fastf1_version": fastf1.__version__,
    }
