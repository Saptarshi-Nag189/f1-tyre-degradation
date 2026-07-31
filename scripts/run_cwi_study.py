"""Phase 3: the pre-registered Spearman study that decides the CWI target.

The physical claim behind the Composite Wear Index is that wear rate is
proportional to the frictional work dissipated at the contact patch
(archard_1953). That claim must hold *in the data* before it is trusted, so the
decision rule is fixed in settings.yaml before the correlation is computed:

    rho > 0.4          -> energy stays in the target at weight 0.5
    0.2 <= rho <= 0.4  -> energy down-weighted (lap-time weight 0.7)
    rho < 0.2          -> energy demoted to a feature; the CWI reduces to the
                          z-scored fuel-corrected degradation slope

This is Stage 0: the energy proxy is the telemetry-free mean squared
speed-trap reading, dimensionally the same mean(v^2) the compass computes from
telemetry. It therefore costs no API budget, so the gate that decides whether
telemetry is worth collecting runs before any telemetry is collected.

    .venv\\Scripts\\python.exe scripts/run_cwi_study.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config                             # noqa: E402
from src.features import target as target_mod      # noqa: E402


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", default="0", choices=["0", "telemetry"],
                        help="'0' uses the speed-trap proxy; 'telemetry' uses "
                             "the curvature aggregates once collected")
    parser.add_argument("--energy-col", default=None,
                        help="energy column for the telemetry stage")
    return parser.parse_args()


def write_plot(stints: pd.DataFrame, path: Path, rho: float) -> bool:
    """Write the energy-versus-degradation scatter plot.

    :returns: True if the plot was written.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False

    valid = stints.dropna(subset=["deg_rate", "energy_proxy"])
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(valid["energy_proxy"], valid["deg_rate"], s=12, alpha=0.45,
               edgecolor="none")
    ax.set_xlabel("Energy proxy, mean $v^2$ (m$^2$/s$^2$)")
    ax.set_ylabel("Degradation rate (s per lap of tyre life)")
    ax.set_title(f"Stint energy versus fuel-corrected degradation\n"
                 f"Spearman rho = {rho:.3f}, n = {len(valid)}")
    ax.axhline(0, color="grey", linewidth=0.8, linestyle="--")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return True


def write_report(path: Path, meta: dict, reliability: dict,
                 stints: pd.DataFrame, stage: str) -> None:
    """Write the human-readable study report."""
    decision_text = {
        "energy_in_target": (
            "Energy correlates with degradation strongly enough to enter the "
            "target at weight 0.5. Collecting telemetry for the true curvature "
            "proxy is well motivated."),
        "energy_downweighted": (
            "Energy correlates weakly. It stays in the target at reduced "
            "weight; telemetry is justified but secondary."),
        "energy_demoted_to_feature": (
            "Energy does not correlate with degradation in this data, so the "
            "pre-registered rule demotes it to a feature. The CWI reduces to "
            "the z-scored fuel-corrected degradation slope, and deg_rate in "
            "seconds per lap remains the interpretable target. This is a "
            "cleaner outcome than it sounds: the target now means something "
            "physical and needs no z-score inversion for the simulator. "
            "Telemetry moves off the critical path to an optional feature "
            "tier behind the 3% gate."),
        "no_data": "No usable stints; the study could not run.",
    }[meta["decision"]]

    rho = meta["spearman_rho"]
    lines = [
        "# CWI Validation Study",
        "",
        f"Stage: **{stage}**  ",
        f"Stints analysed: **{meta['n_stints']}**  ",
        f"Seasons: {sorted(stints['Year'].unique().tolist())}",
        "",
        "## Pre-registered decision rule",
        "",
        "Fixed in `config/settings.yaml` before the correlation was computed.",
        "",
        "| Spearman rho | Outcome |",
        "|---|---|",
        "| > 0.4 | energy in target, weight 0.5 |",
        "| 0.2 to 0.4 | energy down-weighted, lap-time weight 0.7 |",
        "| < 0.2 | energy demoted to a feature |",
        "",
        "## Result",
        "",
        f"- Spearman rho = **{rho:.4f}**" if rho is not None else "- Spearman rho = n/a",
        f"- p = {meta['spearman_p']:.3g}" if meta.get("spearman_p") is not None else "- p = n/a",
        f"- Decision: **{meta['decision']}**",
        f"- Weights: energy {meta['energy_weight']:.2f}, lap-time {meta['laptime_weight']:.2f}",
        "",
        decision_text,
        "",
        "## Target reliability",
        "",
        "Degradation slopes fitted separately on the odd and even clean laps of "
        "each stint, then correlated. This separates a noisy target from a weak "
        "model: if reliability is low, no model can clear the 15% baseline gate "
        "and the gate should be reported against this number rather than tuned "
        "against.",
        "",
        f"- Stints long enough: {reliability['n']}",
        f"- Pearson: {reliability['pearson']:.3f}",
        f"- Spearman: {reliability['spearman']:.3f}",
        "",
        "## Degradation rate distribution",
        "",
        f"- Median: {stints['deg_rate'].median():.4f} s/lap",
        f"- Positive: {100.0 * (stints['deg_rate'] > 0).mean():.1f}% "
        "(tyres get slower, so this should dominate)",
        f"- Interquartile range: {stints['deg_rate'].quantile(0.25):.4f} to "
        f"{stints['deg_rate'].quantile(0.75):.4f} s/lap",
        "",
        "![Energy versus degradation](cwi_study.png)",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    """Run the study and record the decision."""
    args = parse_args()
    logger = config.setup_logging("cwi_study")
    settings = config.settings()

    processed = config.resolve_path("processed")
    stints_path = processed / "stints_core.parquet"
    if not stints_path.exists():
        logger.error("%s not found. Run scripts/run_build_dataset.py first.",
                     stints_path)
        return 1
    stints = pd.read_parquet(stints_path)
    logger.info("Loaded %d stints", len(stints))

    tgt = settings["target"]
    stints, meta = target_mod.validate_and_build_cwi(
        stints,
        keep_thr=tgt["spearman_keep_threshold"],
        downweight_low=tgt["spearman_downweight_low"],
        energy_w_full=tgt["energy_weight_full"],
        laptime_w_down=tgt["laptime_weight_downweight"])

    logger.info("--- target reliability (split-half) ---")
    laps = pd.read_parquet(processed / "laps_clean.parquet")
    reliability = target_mod.split_half_reliability(
        laps, min_r=tgt["linregress_min_r"], min_laps=tgt["min_stint_laps"])

    reports = config.resolve_path("reports", create=True)
    meta["stage"] = args.stage
    meta["reliability"] = reliability
    meta["deg_rate_median"] = float(stints["deg_rate"].median())
    meta["deg_rate_positive_pct"] = float(100.0 * (stints["deg_rate"] > 0).mean())
    (reports / "cwi_study.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8")

    rho = meta["spearman_rho"] if meta["spearman_rho"] is not None else float("nan")
    if write_plot(stints, reports / "cwi_study.png", rho):
        logger.info("Wrote %s", reports / "cwi_study.png")
    write_report(reports / "cwi_study.md", meta, reliability, stints, args.stage)

    stints.to_parquet(processed / "stints_target.parquet",
                      engine="pyarrow", index=False)

    logger.info("=" * 62)
    logger.info("GATE 3  rho = %.4f  ->  %s", rho, meta["decision"])
    logger.info("        energy weight %.2f, lap-time weight %.2f",
                meta["energy_weight"], meta["laptime_weight"])
    logger.info("        split-half reliability: pearson %.3f",
                reliability["pearson"])
    logger.info("Reports written to %s", reports)
    return 0


if __name__ == "__main__":
    sys.exit(main())
