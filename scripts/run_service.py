"""Serve the trained model over HTTP.

    .venv\\Scripts\\python.exe scripts/run_service.py
    .venv\\Scripts\\python.exe scripts/run_service.py --port 8080 --threads 4

Settings resolve config file, then ``F1_SERVICE_HOST`` / ``_PORT`` /
``_THREADS`` / ``_ACCESS_LOG``, then the command line. The environment layer
is for containers, where the config file is baked in and ``CMD`` is fixed.

Runs through waitress, a production WSGI server that works identically on
Windows and Linux. Flask's development server is single-threaded, reloads on
file changes and says so in a warning on every start; it is not used here.

The model loads before the socket is bound, so a successful start means the
service is ready. ``/ready`` exists for the case where a supervisor starts
polling before this process gets that far.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                  # noqa: E402


def _env(name: str, default, cast=str):
    """Read a ``F1_SERVICE_*`` override.

    Settings resolve in the order config file, environment, command line, each
    overriding the last. The environment layer exists for containers, where
    the config file is baked into the image and the command line is fixed by
    ``CMD``, so an orchestrator has nothing else to turn.

    :param name: suffix after ``F1_SERVICE_``, e.g. ``"THREADS"``.
    :param default: value from the config file.
    :param cast: conversion applied to the raw string.
    :returns: the resolved value.
    :raises SystemExit: if the variable is set but unusable, rather than
        silently falling back and serving a configuration nobody asked for.
    """
    raw = os.environ.get(f"F1_SERVICE_{name}")
    if raw is None:
        return default
    if cast is bool:
        return raw.strip().lower() in ("1", "true", "yes", "on")
    try:
        return cast(raw)
    except ValueError:
        raise SystemExit(
            f"F1_SERVICE_{name}={raw!r} is not a valid "
            f"{cast.__name__}") from None


def main() -> int:
    """Load the model and serve it."""
    settings = config.settings()["service"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=_env("HOST", settings["host"]))
    parser.add_argument("--port", type=int,
                        default=_env("PORT", settings["port"], int))
    parser.add_argument("--threads", type=int,
                        default=_env("THREADS", settings["threads"], int),
                        help="waitress worker threads")
    parser.add_argument("--access-log", action="store_true",
                        default=_env("ACCESS_LOG", settings["access_log"],
                                     bool),
                        help="log a line per prediction; off by default "
                             "because a reverse proxy records the same thing")
    args = parser.parse_args()

    logger = config.setup_logging("service")

    from waitress import serve                          # noqa: E402
    from src.service.app import create_app              # noqa: E402

    started = time.perf_counter()
    try:
        app = create_app(access_log=args.access_log)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        logger.error("Run scripts/run_train.py before serving.")
        return 2

    logger.info("Ready in %.2f s; serving on http://%s:%d with %d threads",
                time.perf_counter() - started, args.host, args.port,
                args.threads)
    logger.info("  GET  /health    liveness")
    logger.info("  GET  /ready     readiness")
    logger.info("  GET  /meta      model card, including the failed gate")
    logger.info("  GET  /circuits  supported circuits")
    logger.info("  POST /predict   strategy prediction")

    serve(app, host=args.host, port=args.port, threads=args.threads,
          ident="f1-tyre-degradation")
    return 0


if __name__ == "__main__":
    sys.exit(main())
