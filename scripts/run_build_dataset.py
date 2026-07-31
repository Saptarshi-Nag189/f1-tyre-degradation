"""Build the validated laps frame and the per-stint modelling table.

    .venv\\Scripts\\python.exe scripts/run_build_dataset.py

Writes:
    data/processed/laps_clean.parquet
    data/processed/stints_core.parquet

Gate 2: requires at least ``--min-stints`` usable stints and ``--min-retention``
percent retention. Low retention points at the |r| filter, which selects on the
outcome by keeping only stints that degrade tidily and linearly.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                             # noqa: E402
from src.features import assemble                  # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--year", type=int, action="append",
                        help="restrict to these seasons; repeatable")
    parser.add_argument("--min-stints", type=int, default=600,
                        help="Gate 2 minimum usable stints")
    parser.add_argument("--min-retention", type=float, default=25.0,
                        help="Gate 2 minimum retention percent")
    return parser.parse_args()


def main() -> int:
    """Assemble the dataset and evaluate Gate 2."""
    args = parse_args()
    logger = config.setup_logging("build_dataset")

    laps, stints = assemble.build_dataset(years=args.year)

    if stints.empty:
        logger.error("No usable stints produced.")
        return 1

    out_dir = config.resolve_path("processed", create=True)
    laps.to_parquet(out_dir / "laps_clean.parquet", engine="pyarrow", index=False)
    stints.to_parquet(out_dir / "stints_core.parquet", engine="pyarrow", index=False)
    logger.info("Wrote %s and %s", out_dir / "laps_clean.parquet",
                out_dir / "stints_core.parquet")

    # --- Gate 2 ---
    n_raw = laps.groupby(assemble.target_mod.STINT_GROUP, observed=True).ngroups
    retention = 100.0 * len(stints) / max(n_raw, 1)
    median_deg = float(stints["deg_rate"].median())
    pct_positive = 100.0 * float((stints["deg_rate"] > 0).mean())

    logger.info("=" * 62)
    logger.info("GATE 2  usable stints %d (need >= %d)", len(stints), args.min_stints)
    logger.info("        retention %.1f%% (need >= %.1f%%)", retention, args.min_retention)
    logger.info("        deg_rate median %.4f s/lap, %.1f%% positive",
                median_deg, pct_positive)
    logger.info("        seasons %s, events %d",
                sorted(stints["Year"].unique().tolist()), stints["EventName"].nunique())
    logger.info("        compound ordinal resolved for %.1f%% of stints",
                100.0 * float(stints["compound_ordinal"].notna().mean()))

    # Sanity check 3 from the plan: tyres get slower, so a fuel-corrected
    # degradation slope should be positive for most dry stints. A negative
    # median means the fuel correction sign is inverted.
    if median_deg <= 0:
        logger.error("Median deg_rate is %.4f s/lap, not positive. The fuel "
                     "correction sign is likely inverted.", median_deg)
        return 1

    passed = len(stints) >= args.min_stints and retention >= args.min_retention
    logger.info("GATE 2: %s", "PASS" if passed else "FAIL")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
