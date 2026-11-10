#!/usr/bin/env python3
"""Frozen LHS -> calibration-feasible domain -> one independent validation workflow."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cranial_model.core import load_parameters
from cranial_model.diagnostics import thiele_biot_rows
from cranial_model.ensemble import (
    PARAMETER_PATHS,
    apply_candidate,
    monotonic_trend_fraction,
    predict_conditions,
    prediction_interval,
    sample_lhs_log10,
    screen_candidates,
)
from cranial_model.experimental import (
    apply_magnitude_calibration,
    build_r0_ratio_observations,
    condition_statistics,
    rat_condition_means,
    read_experimental_spectra,
    split_rats_stratified,
)
from cranial_model.impedance import fit_rcpe_magnitude


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/workspace/.uploads/a32f8259-e715-40b5-a800-8c39875ea94b_体外EDTA数据记录汇总 (1).xlsx")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row: dict) -> tuple:
    return (row["stage"], row["edta_concentration_mol_per_l"], row["time_s"])


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def instrument_range_summary(spectra: pd.DataFrame) -> dict:
    by_frequency = {}
    for frequency, group in spectra.groupby("frequency_hz"):
        outside = group["magnitude_mohm"] > 10.0
        by_frequency[str(int(frequency))] = {
            "count": int(len(group)),
            "outside_count": int(outside.sum()),
            "outside_fraction": float(outside.mean()),
        }
    spectrum_outside = spectra.assign(
        outside=spectra["magnitude_mohm"] > 10.0
    ).groupby("spectrum_id")["outside"].any()
    baseline_ids = set(spectra.loc[spectra["stage"] == "baseline", "spectrum_id"])
    return {
        "maximum_resistance_mohm": 10.0,
        "source": "User-supplied ZT-DQ01 product specifications; resistance range 0-10 Mohm",
        "by_frequency_hz": by_frequency,
        "spectra_with_any_outside_reading": int(spectrum_outside.sum()),
        "baseline_spectra_with_any_outside_reading": int(
            spectrum_outside[spectrum_outside.index.isin(baseline_ids)].sum()
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--parameters", type=Path, default=ROOT / "parameters" / "theoretical_prior.json")
    parser.add_argument(
        "--calibration",
        type=Path,
        help="Independent standard-load CSV: frequency_hz,measured_over_true",
    )
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "ensemble_validation")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--split-seed", type=int, default=20261110)
    parser.add_argument("--lhs-seed", type=int, default=20261111)
    parser.add_argument("--cells", nargs=3, type=int, default=[12, 6, 6])
    parser.add_argument("--max-step", type=float, default=30.0)
    args = parser.parse_args()
    if args.samples < 500:
        raise ValueError("The frozen protocol requires at least 500 LHS candidates")
    args.output.mkdir(parents=True, exist_ok=True)

    document = load_parameters(args.parameters)
    base_values = document["values"]
    base_values["stages"]["edta"]["duration_s"] = 5400.0
    base_values["numerics"]["output_times_per_stage"] = 31

    spectra = read_experimental_spectra(args.data)
    range_summary = instrument_range_summary(spectra)
    if args.calibration:
        spectra = apply_magnitude_calibration(spectra, pd.read_csv(args.calibration))
        spectra.to_csv(args.output / "measurement_corrected_spectra.csv", index=False)
    calibration_protocol = {
        "applied": args.calibration is not None,
        "file": str(args.calibration) if args.calibration else None,
        "sha256": sha256(args.calibration) if args.calibration else None,
        "factor_definition": "measured_over_true from independent standard loads",
        "interpolation": "log-log interpolation; no extrapolation",
    }
    fits, observations = build_r0_ratio_observations(
        spectra, fit_rcpe_magnitude, instrument_max_mohm=10.0
    )
    fits.to_csv(args.output / "rcpe_spectrum_fits.csv", index=False)
    observations.to_csv(args.output / "r0_ratio_observations.csv", index=False)
    rat_means = rat_condition_means(observations)
    if rat_means.empty:
        pivot = spectra.pivot(index="spectrum_id", columns="frequency_hz", values="magnitude_mohm")
        apparent_slope = -np.log10(pivot[10000] / pivot[1000])
        source_names = [
            "cranial_model/core.py",
            "cranial_model/ensemble.py",
            "cranial_model/experimental.py",
            "cranial_model/impedance.py",
            "parameters/theoretical_prior.json",
        ]
        protocol = {
            "status": "FROZEN_PREPROCESSING_FAILED_BEFORE_DATA_SPLIT_OR_LHS",
            "data_file": str(args.data),
            "data_sha256": sha256(args.data),
            "preprocessing": {
                "model": "Rinf+(R0-Rinf)/(1+(j*omega*tau)^beta), fitted to magnitude only",
                "measurement_calibration": calibration_protocol,
                "instrument_range": range_summary,
                "relative_rmse_tolerance": 0.05,
                "jacobian_condition_limit": 1e8,
                "accepted_r0_remains_exploratory_without_phase": True,
            },
            "lhs": {
                "sample_count": args.samples,
                "seed": args.lhs_seed,
                "space": "log10",
                "sampled_paths": list(PARAMETER_PATHS),
                "prior_ranges": {path: document["uncertainty"][path]["range"] for path in PARAMETER_PATHS},
                "executed": False,
            },
            "source_sha256": {name: sha256(ROOT / name) for name in source_names},
        }
        write_json(args.output / "frozen_protocol.json", protocol)
        result = {
            "status": "PREPROCESSING_FAILED_NO_VALID_R0_RATIOS",
            "spectrum_count": int(len(fits)),
            "quality_counts": {str(k): int(v) for k, v in fits["quality_flag"].value_counts().items()},
            "instrument_range": range_summary,
            "valid_ratio_count": int(observations["valid"].sum()),
            "total_ratio_count": int(len(observations)),
            "apparent_slope_1k_to_10k": {
                "min": float(apparent_slope.min()),
                "median": float(apparent_slope.median()),
                "max": float(apparent_slope.max()),
                "fraction_greater_than_single_rcpe_limit": float((apparent_slope > 1.0).mean()),
                "single_rcpe_limit": "For 0<beta<=1, the high-frequency -dlog10|Z|/dlog10(f) slope cannot exceed 1",
            },
            "candidate_screening_performed": False,
            "validation_performed": False,
            "reason": "No accepted baseline/post R0 pairs; using original impedance ratios instead is forbidden by the frozen protocol",
            "required_next_step": "Acquire phase/complex impedance or revise the equivalent-circuit structure and refreeze the preprocessing protocol before LHS screening",
        }
        write_json(args.output / "validation_metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return
    split = split_rats_stratified(rat_means, args.split_seed, test_fraction=0.3)
    rat_means["split"] = rat_means["rat"].map(split)
    rat_means.to_csv(args.output / "rat_condition_means.csv", index=False)
    pd.DataFrame([
        {"rat": rat, "split": name, "split_seed": args.split_seed}
        for rat, name in sorted(split.items())
    ]).to_csv(args.output / "data_split.csv", index=False)

    calibration_conditions, calibration_rats = condition_statistics(rat_means, "calibration")
    if len(calibration_conditions) != 17:
        raise RuntimeError(f"Expected 17 calibration conditions, got {len(calibration_conditions)}")
    observed_mean = np.array([row["observed_mean"] for row in calibration_conditions])
    observed_sd = np.array([row["observed_sd"] for row in calibration_conditions])
    tolerance = float(max(np.sqrt(np.mean(observed_sd**2)), 1e-6))

    protocol = {
        "status": "FROZEN_BEFORE_CANDIDATE_SCREENING_AND_INDEPENDENT_VALIDATION",
        "data_file": str(args.data),
        "data_sha256": sha256(args.data),
        "preprocessing": {
            "model": "Rinf+(R0-Rinf)/(1+(j*omega*tau)^beta), fitted to magnitude only",
            "limitation": "Four magnitudes and four free parameters; accepted R0 values are exploratory without phase data",
            "baseline_and_post_must_be_accepted": True,
            "discarded_original_frequency_ratio_target": True,
            "target": "R0_post/R0_base matched by rat and site",
        },
        "split": {
            "method": "rat-level EDTA-dose-stratified random split; no site or time leakage",
            "seed": args.split_seed,
            "calibration_fraction": 0.7,
            "validation_fraction": 0.3,
            "calibration_rat_count": sum(v == "calibration" for v in split.values()),
            "validation_rat_count": sum(v == "validation" for v in split.values()),
            "assignment": split,
        },
        "lhs": {
            "sample_count": args.samples,
            "seed": args.lhs_seed,
            "space": "log10",
            "sampled_paths": list(PARAMETER_PATHS),
            "fixed_skull_properties": True,
            "prior_ranges": {path: document["uncertainty"][path]["range"] for path in PARAMETER_PATHS},
        },
        "screening": {
            "metric": "unweighted RMSE over calibration condition means",
            "tolerance_rule": "sqrt(mean(calibration condition between-rat SD^2))",
            "tolerance": tolerance,
            "accepted_rule": "finite RMSE <= tolerance",
            "no_zero_tolerance": True,
        },
        "validation": {
            "runs_after_posterior_freeze_only": True,
            "no_secondary_candidate_screening": True,
            "band": "min/max over every feasible candidate",
            "coverage_pass_rule": "at least 50% of validation condition means inside full band",
            "trend_pass_rule": "at least 90% of feasible candidates monotone in both dose and time",
        },
        "numerics": {
            "cells_per_layer": args.cells,
            "max_step_s": args.max_step,
            "output_times_per_stage": 31,
            "edta_duration_s": 5400.0,
        },
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in [
                "cranial_model/core.py",
                "cranial_model/ensemble.py",
                "cranial_model/experimental.py",
                "cranial_model/impedance.py",
                "parameters/theoretical_prior.json",
            ]
        },
    }
    write_json(args.output / "frozen_protocol.json", protocol)
    pd.DataFrame([
        {**row, "observations": ";".join(str(v) for v in row["observations"])}
        for row in calibration_conditions
    ]).to_csv(args.output / "calibration_conditions.csv", index=False)

    candidates = sample_lhs_log10(document, args.samples, args.lhs_seed)

    def progress(done: int, total: int) -> None:
        print(f"LHS screening {done}/{total}", flush=True)

    rmse, accepted, calibration_predictions = screen_candidates(
        base_values,
        candidates,
        calibration_conditions,
        observed_mean,
        tolerance,
        args.cells,
        args.max_step,
        progress=progress,
    )
    candidate_frame = pd.DataFrame(candidates, columns=PARAMETER_PATHS)
    candidate_frame["candidate_id"] = np.arange(args.samples)
    candidate_frame["calibration_rmse"] = rmse
    candidate_frame["accepted"] = accepted
    candidate_frame.to_csv(args.output / "calibration_candidates.csv", index=False)
    accepted_indices = np.flatnonzero(accepted)
    candidate_frame[accepted].to_csv(args.output / "posterior_feasible_parameters.csv", index=False)
    if len(accepted_indices) == 0:
        result = {
            "status": "NO_FEASIBLE_CANDIDATE",
            "tolerance": tolerance,
            "minimum_calibration_rmse": float(np.nanmin(rmse)),
            "valid_spectra": int(fits["quality_flag"].eq("accepted").sum()),
            "total_spectra": int(len(fits)),
            "validation_performed": False,
            "next_check": ["R-CPE identifiability", "R0 preprocessing", "prior intervals", "SCM/porosity structure"],
        }
        write_json(args.output / "validation_metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # The feasible domain is now fixed. Validation outcomes are first accessed below.
    validation_conditions, validation_rats = condition_statistics(rat_means, "validation")
    calibration_by_key = {key(row): row for row in calibration_conditions}
    validation_by_key = {key(row): row for row in validation_conditions}
    if set(validation_by_key) != set(calibration_by_key):
        raise RuntimeError("Calibration and validation condition sets differ")

    full_conditions = calibration_conditions
    full_predictions = np.empty((len(accepted_indices), len(full_conditions)))
    diagnostics = []
    for local_index, candidate_index in enumerate(accepted_indices):
        values = apply_candidate(base_values, PARAMETER_PATHS, candidates[candidate_index])
        prediction, scenarios = predict_conditions(
            values, full_conditions, args.cells, args.max_step, return_scenarios=True
        )
        full_predictions[local_index] = prediction
        for dose, scenario in scenarios.items():
            for row in thiele_biot_rows(scenario):
                diagnostics.append({"candidate_id": int(candidate_index), **row})
        print(f"Posterior ensemble {local_index + 1}/{len(accepted_indices)}", flush=True)

    lower, upper = prediction_interval(full_predictions)
    band_rows = []
    for index, condition in enumerate(full_conditions):
        condition_key = key(condition)
        calibration = calibration_by_key[condition_key]
        validation = validation_by_key[condition_key]
        band_rows.append({
            **condition,
            "simulation_min": lower[index],
            "simulation_max": upper[index],
            "simulation_median": float(np.median(full_predictions[:, index])),
            "calibration_observed_mean": calibration["observed_mean"],
            "calibration_observed_sd": calibration["observed_sd"],
            "calibration_rat_count": calibration["rat_count"],
            "validation_observed_mean": validation["observed_mean"],
            "validation_observed_sd": validation["observed_sd"],
            "validation_rat_count": validation["rat_count"],
            "validation_covered": bool(lower[index] <= validation["observed_mean"] <= upper[index]),
        })
    bands = pd.DataFrame(band_rows)
    bands.to_csv(args.output / "prediction_bands.csv", index=False)

    long_predictions = []
    for local_index, candidate_index in enumerate(accepted_indices):
        for condition_index, condition in enumerate(full_conditions):
            long_predictions.append({
                "candidate_id": int(candidate_index),
                **condition,
                "predicted_ratio": full_predictions[local_index, condition_index],
            })
    pd.DataFrame(long_predictions).to_csv(args.output / "posterior_condition_predictions.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(args.output / "posterior_mechanism_diagnostics_long.csv", index=False)
    if diagnostics:
        diagnostic_frame = pd.DataFrame(diagnostics)
        diagnostic_summary = diagnostic_frame.groupby(
            ["edta_concentration_mol_per_l", "stage", "layer"]
        ).agg(
            thiele_min=("thiele_modulus", "min"),
            thiele_median=("thiele_modulus", "median"),
            thiele_max=("thiele_modulus", "max"),
            damkohler_min=("damkohler_number", "min"),
            damkohler_median=("damkohler_number", "median"),
            damkohler_max=("damkohler_number", "max"),
            biot_min=("biot_number", "min"),
            biot_median=("biot_number", "median"),
            biot_max=("biot_number", "max"),
        ).reset_index()
        diagnostic_summary.to_csv(args.output / "posterior_mechanism_diagnostics.csv", index=False)

    edta_bands = bands[bands["stage"] == "edta"]
    coverage = float(edta_bands["validation_covered"].mean())
    trends = monotonic_trend_fraction(full_conditions, full_predictions)
    posterior_summary = {}
    for column, name in enumerate(PARAMETER_PATHS):
        values = candidates[accepted, column]
        posterior_summary[name] = {
            "min": float(np.min(values)),
            "q25": float(np.quantile(values, 0.25)),
            "median": float(np.median(values)),
            "q75": float(np.quantile(values, 0.75)),
            "max": float(np.max(values)),
        }
    metrics = {
        "status": "INDEPENDENT_VALIDATION_COMPLETED_ONCE_AFTER_POSTERIOR_FREEZE",
        "preprocessing": {
            "spectrum_count": int(len(fits)),
            "accepted_spectrum_count": int(fits["quality_flag"].eq("accepted").sum()),
            "valid_ratio_count": int(observations["valid"].sum()),
            "total_ratio_count": int(len(observations)),
            "magnitude_only_r0_is_exploratory": True,
        },
        "split": protocol["split"],
        "screening": {
            "candidate_count": int(args.samples),
            "accepted_count": int(len(accepted_indices)),
            "acceptance_fraction": float(np.mean(accepted)),
            "tolerance": tolerance,
            "accepted_calibration_rmse_min": float(np.min(rmse[accepted])),
            "accepted_calibration_rmse_median": float(np.median(rmse[accepted])),
            "accepted_calibration_rmse_max": float(np.max(rmse[accepted])),
        },
        "independent_validation": {
            "edta_condition_count": int(len(edta_bands)),
            "covered_condition_count": int(edta_bands["validation_covered"].sum()),
            "condition_mean_coverage": coverage,
            "coverage_pass": bool(coverage >= 0.5),
            "mean_band_width": float(np.mean(edta_bands["simulation_max"] - edta_bands["simulation_min"])),
            "trends": trends,
            "trend_pass": bool(trends["both_monotonic_fraction"] >= 0.9),
        },
        "posterior_parameter_domain": posterior_summary,
        "caution": "Feasible-domain calibration and band coverage are not unique-parameter identification or proof of mechanism.",
    }
    write_json(args.output / "validation_metrics.json", metrics)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9), sharex=True, sharey=True)
    for axis, dose in zip(axes.flat, sorted(edta_bands["edta_concentration_mol_per_l"].unique())):
        subset = edta_bands[edta_bands["edta_concentration_mol_per_l"] == dose].sort_values("time_s")
        time_min = subset["time_s"] / 60.0
        axis.fill_between(time_min, subset["simulation_min"], subset["simulation_max"], alpha=0.25, label="Feasible-domain band")
        axis.errorbar(time_min, subset["calibration_observed_mean"], yerr=subset["calibration_observed_sd"], fmt="o", capsize=3, label="Calibration mean ± SD")
        axis.errorbar(time_min, subset["validation_observed_mean"], yerr=subset["validation_observed_sd"], fmt="s", capsize=3, label="Independent validation mean ± SD")
        axis.set_title(f"EDTA {dose:.2f} mol/L")
        axis.set_xlabel("EDTA time (min)")
        axis.set_ylabel("R0 post / R0 baseline")
        axis.grid(alpha=0.2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3)
    fig.suptitle(f"Posterior-feasible ensemble: validation coverage {coverage:.1%}")
    fig.tight_layout(rect=(0, 0.06, 1, 0.96))
    fig.savefig(args.output / "ensemble_validation.png", dpi=180)
    plt.close(fig)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
