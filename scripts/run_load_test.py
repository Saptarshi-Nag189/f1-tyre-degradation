"""Measure the service's throughput and tail latency under load.

    .venv\\Scripts\\python.exe scripts/run_service.py                 # terminal 1
    .venv\\Scripts\\python.exe scripts/run_load_test.py                # terminal 2
    .venv\\Scripts\\python.exe scripts/run_load_test.py --concurrency 1,4,16

A closed-loop generator: each worker sends a request, waits for the response,
and immediately sends the next. Offered load therefore rises with concurrency
rather than being fixed, which is the right shape for a service whose callers
are other programs waiting on an answer.

Three things this reports that a single average would hide.

**Client latency against server latency.** Every response carries an
``X-Server-Duration-Ms`` header. Client latency includes connection reuse,
scheduling and the queue inside waitress; the server figure is the work
itself. When the two diverge, the service is queueing rather than computing,
and that is the point at which adding threads stops helping.

**Requests are drawn from ten circuits, not one.** Per-request cost varies
roughly three-fold across circuits because the strategy search grid is finer
for shorter races, so hammering one circuit measures that circuit.

**The generator shares a machine with the service.** Both are Python
processes on the same host, competing for the same cores, so these numbers
are a floor rather than what dedicated hardware would give. They are reported
as measured; no headroom is extrapolated from them.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                                  # noqa: E402

#: Circuits spanning the range of race lengths, and therefore of search width.
CIRCUITS = ["silverstone", "monaco", "monza", "spa", "singapore",
            "jeddah", "bahrain", "suzuka", "barcelona", "hungaroring"]


def payloads() -> list[dict]:
    """Representative prediction requests, one per circuit."""
    return [{"season": 2024, "circuit": circuit, "air_temp": 22,
             "track_temp": 38, "conditions": "dry",
             "compounds": ["SOFT", "MEDIUM", "HARD"]}
            for circuit in CIRCUITS]


def percentile(sorted_values: list[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted list."""
    if not sorted_values:
        return float("nan")
    index = min(len(sorted_values) - 1, int(fraction * len(sorted_values)))
    return sorted_values[index]


class Worker:
    """One closed-loop client, holding a session so connections are reused."""

    def __init__(self, url: str | list[str], requests_list: list[dict],
                 deadline: float, offset: int) -> None:
        self.url = url
        self.payloads = requests_list
        self.deadline = deadline
        self.offset = offset
        self.client_ms: list[float] = []
        self.server_ms: list[float] = []
        self.errors = 0
        self.status_counts: dict[int, int] = {}

    def run(self) -> None:
        """Send requests until the deadline.

        With several URLs the worker rotates between them, standing in for the
        round-robin a load balancer would do. Each replica is a separate
        process with its own copy of the model and no shared state, which is
        what makes that legitimate.
        """
        urls = self.url if isinstance(self.url, list) else [self.url]
        session = requests.Session()
        index = self.offset
        try:
            while time.perf_counter() < self.deadline:
                payload = self.payloads[index % len(self.payloads)]
                url = urls[index % len(urls)]
                index += 1
                start = time.perf_counter()
                try:
                    response = session.post(url, json=payload, timeout=30)
                except requests.RequestException:
                    self.errors += 1
                    continue
                elapsed_ms = (time.perf_counter() - start) * 1000.0

                self.status_counts[response.status_code] = (
                    self.status_counts.get(response.status_code, 0) + 1)
                if response.status_code != 200:
                    self.errors += 1
                    continue

                self.client_ms.append(elapsed_ms)
                header = response.headers.get("X-Server-Duration-Ms")
                if header:
                    self.server_ms.append(float(header))
        finally:
            session.close()


def run_level(url: str | list[str], concurrency: int, duration_s: float,
              logger) -> dict:
    """Drive the service at one concurrency level.

    :param url: the predict endpoint, or a list of them across replicas.
    :param concurrency: number of closed-loop workers.
    :param duration_s: measurement window in seconds.
    :param logger: where to report progress.
    :returns: measurements for this level.
    """
    requests_list = payloads()
    barrier = threading.Barrier(concurrency + 1)

    def start(worker: Worker) -> Worker:
        barrier.wait()
        worker.run()
        return worker

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        # The deadline is set after the barrier releases, so start-up cost is
        # not charged to the measurement window.
        workers = [Worker(url, requests_list, 0.0, offset=i)
                   for i in range(concurrency)]
        futures = [pool.submit(start, w) for w in workers]
        barrier.wait()
        wall_start = time.perf_counter()
        deadline = wall_start + duration_s
        for worker in workers:
            worker.deadline = deadline
        for future in futures:
            future.result()
        wall_s = time.perf_counter() - wall_start

    client = sorted(m for w in workers for m in w.client_ms)
    server = sorted(m for w in workers for m in w.server_ms)
    errors = sum(w.errors for w in workers)
    statuses: dict[int, int] = {}
    for worker in workers:
        for code, count in worker.status_counts.items():
            statuses[code] = statuses.get(code, 0) + count

    result = {
        "concurrency": concurrency,
        "wall_s": round(wall_s, 2),
        "requests": len(client),
        "errors": errors,
        "status_counts": {str(k): v for k, v in sorted(statuses.items())},
        "rps": round(len(client) / wall_s, 1) if wall_s else 0.0,
        "client_p50_ms": round(percentile(client, 0.50), 2),
        "client_p95_ms": round(percentile(client, 0.95), 2),
        "client_p99_ms": round(percentile(client, 0.99), 2),
        "client_max_ms": round(client[-1], 2) if client else float("nan"),
        "server_p50_ms": round(percentile(server, 0.50), 2),
        "server_p95_ms": round(percentile(server, 0.95), 2),
        "queueing_p95_ms": round(
            percentile(client, 0.95) - percentile(server, 0.95), 2),
    }
    logger.info("  c=%-3d %7.1f req/s   p50 %6.2f ms   p95 %7.2f ms   "
                "p99 %7.2f ms   server p95 %6.2f ms   errors %d",
                concurrency, result["rps"], result["client_p50_ms"],
                result["client_p95_ms"], result["client_p99_ms"],
                result["server_p95_ms"], errors)
    return result


def check_service(base_url: str, logger) -> dict:
    """Confirm the service is up and ready before measuring it.

    :raises SystemExit: if it is not, since an unreachable service produces a
        table of zeroes that looks like a result.
    """
    try:
        ready = requests.get(f"{base_url}/ready", timeout=5)
    except requests.RequestException as exc:
        logger.error("No service at %s (%s).", base_url, exc)
        logger.error("Start one first: python scripts/run_service.py")
        raise SystemExit(2) from exc
    if ready.status_code != 200:
        logger.error("Service at %s is not ready: %d %s", base_url,
                     ready.status_code, ready.text[:200])
        raise SystemExit(2)
    return ready.json()


def main() -> int:
    """Run the sweep and write the results as JSON."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000",
                        help="service base URL, or several comma-separated to "
                             "measure replicas in aggregate")
    parser.add_argument("--concurrency", default="1,2,4,8,16,32",
                        help="comma-separated levels to sweep")
    parser.add_argument("--duration", type=float, default=10.0,
                        help="seconds of measurement per level")
    parser.add_argument("--warmup", type=float, default=3.0,
                        help="seconds of untimed load before measuring")
    args = parser.parse_args()

    logger = config.setup_logging("load_test")
    levels = [int(c) for c in args.concurrency.split(",") if c.strip()]
    bases = [u.strip().rstrip("/") for u in args.url.split(",") if u.strip()]

    for base in bases:
        ready = check_service(base, logger)
    logger.info("%d replica(s) ready over %d stints; sweeping %s for %.0f s "
                "each", len(bases), ready.get("stints", 0), levels,
                args.duration)

    predict_urls = [f"{base}/predict" for base in bases]
    logger.info("Warming up for %.0f s", args.warmup)
    run_level(predict_urls, 4, args.warmup, logging_noop(logger))

    results = [run_level(predict_urls, c, args.duration, logger)
               for c in levels]

    peak = max(results, key=lambda r: r["rps"])
    total_errors = sum(r["errors"] for r in results)
    summary = {
        "urls": bases,
        "replicas": len(bases),
        "duration_s_per_level": args.duration,
        "circuits": CIRCUITS,
        "levels": results,
        "peak_rps": peak["rps"],
        "peak_rps_concurrency": peak["concurrency"],
        "peak_rps_p95_ms": peak["client_p95_ms"],
        "total_errors": total_errors,
        "note": ("Load generator and service share one machine, so these are "
                 "a floor rather than what dedicated hardware would give."),
    }

    logger.info("Peak %.1f requests/s at concurrency %d, p95 %.2f ms; "
                "%d errors across the whole sweep",
                peak["rps"], peak["concurrency"], peak["client_p95_ms"],
                total_errors)

    suffix = "" if len(bases) == 1 else f"_x{len(bases)}"
    out = (config.resolve_path("reports", create=True)
           / f"load_test{suffix}.json")
    out.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %s", out)
    return 0


def logging_noop(logger):
    """A logger whose ``info`` is discarded, for the untimed warm-up."""
    class _Quiet:
        def info(self, *_args, **_kwargs) -> None:
            pass

        def error(self, *args, **kwargs) -> None:
            logger.error(*args, **kwargs)
    return _Quiet()


if __name__ == "__main__":
    sys.exit(main())
