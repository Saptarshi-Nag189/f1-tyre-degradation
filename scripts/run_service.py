"""Serve the trained model over HTTP.

    .venv\\Scripts\\python.exe scripts/run_service.py
    .venv\\Scripts\\python.exe scripts/run_service.py --port 8080 --threads 4

Runs through waitress, a production WSGI server that works identically on
Windows and Linux. Flask's development server is single-threaded, reloads on
file changes and says so in a warning on every start; it is not used here.

The model loads before the socket is bound, so a successful start means the
service is ready. ``/ready`` exists for the case where a supervisor starts
polling before this process gets that far.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                  # noqa: E402


def main() -> int:
    """Load the model and serve it."""
    settings = config.settings()["service"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default=settings["host"])
    parser.add_argument("--port", type=int, default=settings["port"])
    parser.add_argument("--threads", type=int, default=settings["threads"],
                        help="waitress worker threads")
    parser.add_argument("--access-log", action="store_true",
                        default=settings["access_log"],
                        help="log a line per prediction; costs about a third "
                             "of throughput, see config/settings.yaml")
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
