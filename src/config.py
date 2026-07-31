"""Configuration loading, with every path anchored to the project root.

All magic numbers live in ``config/*.yaml``; nothing in ``src/`` hard-codes a
constant. Relative paths in ``settings.yaml`` are resolved against the project
root rather than the current working directory, so scripts behave identically
regardless of where they are invoked from.
"""
from __future__ import annotations

import functools
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


@functools.lru_cache(maxsize=None)
def load_yaml(name: str) -> dict[str, Any]:
    """Load and cache a YAML file from the config directory.

    :param name: file name, e.g. ``"settings.yaml"``.
    :returns: parsed mapping.
    :raises FileNotFoundError: if the file is absent.
    """
    path = CONFIG_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def settings() -> dict[str, Any]:
    """Return the global settings mapping."""
    return load_yaml("settings.yaml")


def compound_mapping() -> dict[str, Any]:
    """Return the Pirelli C-compound nomination mapping."""
    return load_yaml("compound_mapping.yaml")


def track_traits() -> dict[str, Any]:
    """Return circuit abrasion/energy traits and climatological defaults."""
    return load_yaml("track_traits.yaml")


def ui_circuit_map() -> dict[str, Any]:
    """Return the front-end slug to event-name/track-key mapping."""
    return load_yaml("ui_circuit_map.yaml")


def resolve_path(key: str, *, create: bool = False) -> Path:
    """Resolve a ``paths`` entry from settings.yaml against the project root.

    :param key: key under ``paths``, e.g. ``"raw_laps"``.
    :param create: create the directory if it does not exist.
    :returns: absolute path.
    :raises KeyError: if the key is not defined.
    """
    paths = settings()["paths"]
    if key not in paths:
        raise KeyError(f"paths.{key} is not defined in settings.yaml")
    path = (PROJECT_ROOT / paths[key]).resolve()
    if create:
        path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logging(name: str, *, level: int = logging.INFO,
                  to_file: bool = True) -> logging.Logger:
    """Configure console (and optionally file) logging for a script.

    FastF1 attaches its own noisy handlers, so its logger is capped at WARNING
    to keep run output readable.

    :param name: logger name, also used for the log file stem.
    :param level: level for this project's loggers.
    :param to_file: also write to ``logs/<name>.log``.
    :returns: the configured logger.
    """
    root = logging.getLogger()
    if not root.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s",
            datefmt="%H:%M:%S"))
        root.addHandler(handler)
    root.setLevel(level)

    if to_file:
        log_dir = resolve_path("logs", create=True)
        file_handler = logging.FileHandler(log_dir / f"{name}.log", encoding="utf-8")
        file_handler.setFormatter(logging.Formatter(
            "%(asctime)s %(levelname)-7s %(name)s: %(message)s"))
        root.addHandler(file_handler)

    # FastF1 attaches its own console handler and also propagates to the root,
    # so without this every one of its lines is printed twice.
    fastf1_logger = logging.getLogger("fastf1")
    fastf1_logger.setLevel(logging.WARNING)
    fastf1_logger.propagate = False
    return logging.getLogger(name)
