#!/usr/bin/env python3
"""Trend-level overlay: Track A prior ensemble bands vs measured low-frequency Z ratios.

This figure is TREND correspondence only (direction, relative dose ordering,
stage-level evolution). The prior ensemble is not calibrated to the measured
impedance, and a magnitude gap is expected because the measurement includes
interface/electrode contributions that the bone-matrix model does not
represent. No kinetic parameters are inverted from this comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cranial_model.plotting import DOSE_COLORS, configure_style


ROOT = Path(__file__).resolve().parent
DISCLAIMER = (
    "Trend-level comparison only: simulation from literature priors, not "
    "calibrated to measured impedance; magnitudes include interface/electrode "
    "contributions absent from the bone-matrix model."
)


def _measurement_time_min(stage: str, time_s: float) -> float:
    return (5.0 if stage == "edta" else 0.0) + time_s / 60.0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prior-results", type=Path, default=ROOT / "results" / "prior_mechanism")
    parser.add_argument("--targets", type=Path, default=ROOT / "results" / "equivalent_calibration_v2" / "z_ratio_targets.csv")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "trend_comparison")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    quantiles = pd.read_csv(args.prior_results / "mechanism_timeseries_quantiles.csv")
    whole = quantiles[quantiles["metric"] == "whole_resistance_ratio"]
    targets = pd.read_csv(args.targets)

    doses = sorted(targets["edta_concentration_mol_per_l"].unique())
    observed = targets.copy()
    observed["time_min"] = [
        _measurement_time_min(stage, time_s)
        for stage, time_s in zip(observed["stage"], observed["time_s"])
    ]
    condition_stats = (
        observed.groupby(["edta_concentration_mol_per_l", "time_min"])["z_ratio"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )

    # Trend correspondence per dose: sim median sampled at measurement times.
    rows = []
    for dose in doses:
        sim = quantiles[
            (quantiles["metric"] == "whole_resistance_ratio")
            & (quantiles["edta_concentration_mol_per_l"] == dose)
        ]
        stats = condition_stats[condition_stats["edta_concentration_mol_per_l"] == dose]
        sim_at_points = np.interp(stats["time_min"] * 60.0, sim["global_time_s"], sim["q50"])
        rho, pvalue = spearmanr(sim_at_points, stats["mean"])
        rows.append(
            {
                "edta_concentration_mol_per_l": dose,
                "n_conditions": int(len(stats)),
                "measured_monotone_decreasing": bool(np.all(np.diff(stats.sort_values("time_min")["mean"]) <= 0)),
                "sim_median_monotone_decreasing": bool(np.all(np.diff(sim_at_points) <= 0)),
                "trend_spearman_rho": float(rho),
                "trend_spearman_p": float(pvalue),
                "measured_total_drop": float(stats["mean"].iloc[0] - stats["mean"].iloc[-1]),
                "sim_median_total_drop": float(sim_at_points[0] - sim_at_points[-1]),
            }
        )
    correspondence = pd.DataFrame(rows)
    correspondence.to_csv(args.output / "trend_correspondence.csv", index=False)

    configure_style()
    figure, axes = plt.subplots(1, len(doses), figsize=(3.1 * len(doses), 3.4), sharey=True)
    for axis, dose, color in zip(np.atleast_1d(axes), doses, DOSE_COLORS):
        sim = whole[whole["edta_concentration_mol_per_l"] == dose].sort_values("global_time_s")
        time_min = sim["global_time_s"] / 60.0
        axis.fill_between(time_min, sim["q05"], sim["q95"], color=color, alpha=0.18, lw=0)
        axis.plot(time_min, sim["q50"], color=color, lw=1.6, label="Prior ensemble median (5-95%)")
        stats = condition_stats[condition_stats["edta_concentration_mol_per_l"] == dose]
        axis.errorbar(
            stats["time_min"], stats["mean"], yerr=stats["std"],
            fmt="o", color="black", ms=4, capsize=3, lw=1.0,
            label="Measured Z ratio (mean ± SD, n rats)",
        )
        axis.axvline(5.0, color="0.6", lw=0.8, ls=":")
        axis.set_title(f"EDTA {dose:.2f} mol/L", fontsize=9)
        axis.set_xlabel("Process time (min)")
        axis.grid(alpha=0.2)
    np.atleast_1d(axes)[0].set_ylabel("Resistance ratio to baseline/native")
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, 0.075))
    figure.suptitle("Trend correspondence: prior mechanism ensemble vs measured low-frequency Z ratio", fontsize=10)
    figure.text(0.99, 0.005, DISCLAIMER, ha="right", fontsize=6, color="0.35")
    figure.tight_layout(rect=(0, 0.12, 1, 0.94))
    figure.savefig(args.output / "FigC1_trend_correspondence.png", facecolor="white")
    plt.close(figure)

    summary = {
        "status": "TREND_LEVEL_COMPARISON_NOT_CALIBRATION",
        "disclaimer": DISCLAIMER,
        "correspondence": rows,
    }
    (args.output / "trend_correspondence.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
