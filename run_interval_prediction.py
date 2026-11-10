#!/usr/bin/env python3
"""Nested leave-one-rat-out validation of calibrated R-Z or Delta-Z intervals."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from cranial_model.delta_z_calibration import (
    build_delta_z_targets, delta_candidate_metrics, delta_frequency_tolerances,
)
from cranial_model.experimental import read_experimental_spectra
from cranial_model.frequency_surface_polarization import COMBINED_PARAMETER_BOUNDS
from cranial_model.interval_prediction import (
    ensemble_intervals, expand_intervals, frequency_expansions, nonconformity_scores,
)
from cranial_model.multifrequency_calibration import (
    FREQUENCIES_HZ, build_multifrequency_targets,
)
from run_frequency_surface_polarization import aligned_arrays, candidate_metrics


ADAPTIVE_SCALES = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10)
NEAREST_ENSEMBLE_SIZE = 256


def fit_training_ensemble(
    predictions: pd.DataFrame,
    train: pd.DataFrame,
    structure: str,
    seed: int,
    maxiter: int,
    popsize: int,
    adaptive_draws: int,
    target: str = "ratio",
):
    """Fit using training rats only, then explore the local compatible region."""
    arrays = aligned_arrays(predictions, train, structure)
    if target == "delta":
        tolerances = delta_frequency_tolerances(train)
        arrays["delta_tolerances"] = tolerances
        arrays["delta_tolerance_vector"] = np.array([
            tolerances[f] for f in FREQUENCIES_HZ
        ])
    elif target != "ratio":
        raise ValueError("Unknown interval target")
    if len(arrays["mechanism_ids"]) != 1:
        raise ValueError("The interval runner requires one fixed mechanism trajectory")

    def objective(parameters):
        if target == "delta":
            return delta_candidate_metrics(parameters, arrays, train, 0, False)[0]
        return candidate_metrics(parameters, arrays, 0, True, False)[0]

    result = differential_evolution(
        objective, COMBINED_PARAMETER_BOUNDS, seed=seed, maxiter=maxiter,
        popsize=popsize, tol=1e-8, polish=True, vectorized=True,
        updating="deferred",
    )
    lower = np.array([bound[0] for bound in COMBINED_PARAMETER_BOUNDS])
    upper = np.array([bound[1] for bound in COMBINED_PARAMETER_BOUNDS])
    width = upper - lower
    center = (result.x - lower) / width
    rng = np.random.default_rng(seed + 100000)
    adaptive = []
    for scale in ADAPTIVE_SCALES:
        unit = center[None, :] + scale * rng.uniform(
            -1.0, 1.0, size=(adaptive_draws, len(center))
        )
        adaptive.append(lower[None, :] + np.clip(unit, 0.0, 1.0) * width[None, :])
    candidates = np.concatenate(
        [result.population, result.x[None, :], *adaptive], axis=0
    )
    if target == "delta":
        minimax, normalized, raw, _ = delta_candidate_metrics(
            candidates, arrays, train, 0, False
        )
    else:
        minimax, normalized, raw = candidate_metrics(candidates, arrays, 0, True, False)
    compatible = minimax <= 1.0
    compatible_count = int(compatible.sum())
    if compatible_count:
        selected = compatible
        role = "TRAINING_COMPATIBLE_ENSEMBLE"
    else:
        order = np.argsort(minimax, kind="stable")[:NEAREST_ENSEMBLE_SIZE]
        selected = np.zeros(len(candidates), dtype=bool)
        selected[order] = True
        role = "NEAREST_TRAINING_ENSEMBLE_NOT_ORIGINAL_COMPATIBLE_SET"
    chosen = candidates[selected]
    parameters = pd.DataFrame(chosen, columns=[f"x{i}" for i in range(17)])
    parameters["training_minimax"] = minimax[selected]
    parameters["training_normalized_rmse"] = normalized[selected]
    best_index = int(np.argmin(minimax))
    best = candidates[best_index]
    boundary = np.flatnonzero(
        (np.abs(best - lower) / np.maximum(width, 1.0) < 1e-5)
        | (np.abs(best - upper) / np.maximum(width, 1.0) < 1e-5)
    )
    summary = {
        "best_training_minimax": float(minimax[best_index]),
        "best_training_normalized_rmse": float(normalized[best_index]),
        "compatible_count": compatible_count,
        "ensemble_count": len(parameters),
        "ensemble_role": role,
        "boundary_parameter_indices": ";".join(map(str, boundary)),
        "tolerances": arrays["delta_tolerances"] if target == "delta" else arrays["tolerances"],
        **{
            f"best_training_raw_rmse_{f:g}hz": float(raw[best_index, index])
            for index, f in enumerate(FREQUENCIES_HZ)
        },
    }
    return parameters, summary


def predict_ensemble(
    parameters, predictions, rows, structure, fit_summary, target="ratio"
):
    arrays = aligned_arrays(predictions, rows, structure, estimate_tolerances=False)
    intervals = ensemble_intervals(parameters, arrays)
    if target == "delta":
        baseline = rows["baseline_mean_mohm"].to_numpy(dtype=float)
        for column in ("sim_q05", "sim_q50", "sim_q95"):
            intervals[column] = baseline * (intervals[column] - 1.0)
    result = pd.concat([rows.reset_index(drop=True), intervals], axis=1)
    result["ensemble_count"] = len(parameters)
    result["compatible_count"] = fit_summary["compatible_count"]
    result["ensemble_role"] = fit_summary["ensemble_role"]
    result["training_minimax"] = fit_summary["best_training_minimax"]
    result["original_compatible_prediction_available"] = bool(
        fit_summary["compatible_count"] > 0
    )
    return result


def inner_folds(rat_ids, fold_count, seed):
    rng = np.random.default_rng(seed)
    shuffled = np.array(sorted(rat_ids), dtype=object)
    rng.shuffle(shuffled)
    return [list(group) for group in np.array_split(shuffled, fold_count) if len(group)]


def run_outer_fold(
    predictions, targets, held_out, outer_index, args,
):
    train = targets[targets["rat"] != held_out].reset_index(drop=True)
    test = targets[targets["rat"] == held_out].reset_index(drop=True)
    outer_parameters, outer_fit = fit_training_ensemble(
        predictions, train, args.structure, 20268000 + outer_index,
        args.maxiter, args.popsize, args.adaptive_draws, args.target,
    )
    calibration_frames = []
    groups = inner_folds(train["rat"].unique(), args.inner_folds, 20267000 + outer_index)
    for inner_index, inner_held in enumerate(groups):
        inner_train = train[~train["rat"].isin(inner_held)].reset_index(drop=True)
        inner_test = train[train["rat"].isin(inner_held)].reset_index(drop=True)
        inner_parameters, inner_fit = fit_training_ensemble(
            predictions, inner_train, args.structure,
            20269000 + outer_index * 10 + inner_index,
            args.inner_maxiter, args.inner_popsize, args.adaptive_draws, args.target,
        )
        inner_intervals = predict_ensemble(
            inner_parameters, predictions, inner_test, args.structure, inner_fit, args.target
        )
        observed_column = "delta_z_mohm" if args.target == "delta" else "z_ratio"
        inner_intervals["nonconformity"] = nonconformity_scores(
            inner_intervals, inner_fit["tolerances"], observed_column
        )
        inner_intervals["inner_fold"] = inner_index
        inner_intervals["inner_training_minimax"] = inner_fit["best_training_minimax"]
        inner_intervals["inner_compatible_count"] = inner_fit["compatible_count"]
        calibration_frames.append(inner_intervals)
    calibration = pd.concat(calibration_frames, ignore_index=True)
    expansions = frequency_expansions(
        calibration["nonconformity"].to_numpy(dtype=float),
        calibration["frequency_hz"].to_numpy(dtype=float), args.level,
    )
    outer_intervals = predict_ensemble(
        outer_parameters, predictions, test, args.structure, outer_fit, args.target
    )
    observed_column = "delta_z_mohm" if args.target == "delta" else "z_ratio"
    outer_intervals = expand_intervals(
        outer_intervals, outer_fit["tolerances"], expansions, observed_column,
        physical_lower_bound=None if args.target == "delta" else 0.0,
    )
    outer_intervals.insert(0, "outer_fold", outer_index)
    calibration.insert(0, "outer_fold", outer_index)
    outer_parameters.insert(0, "outer_fold", outer_index)
    summary = {
        "outer_fold": outer_index, "rat": held_out, "n_points": len(test),
        "coverage": float(outer_intervals["calibrated_covered"].mean()),
        "median_width": float(outer_intervals["calibrated_width"].median()),
        "original_compatible_coverage": (
            float(outer_intervals["original_compatible_covered"].mean())
            if outer_fit["compatible_count"] else np.nan
        ),
        "inner_fold_count": len(groups),
        "inner_empty_compatible_fold_count": int(sum(
            frame["inner_compatible_count"].iloc[0] == 0 for frame in calibration_frames
        )),
        "calibration_point_count": len(calibration),
        **{key: value for key, value in outer_fit.items() if key != "tolerances"},
        **{
            f"tolerance_{f:g}hz": float(outer_fit["tolerances"][f])
            for f in FREQUENCIES_HZ
        },
        **{f"expansion_{f:g}hz": expansions[f] for f in FREQUENCIES_HZ},
    }
    return summary, outer_intervals, calibration, outer_parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structure", default="two_sided_50um")
    parser.add_argument("--target", choices=("ratio", "delta"), default="ratio")
    parser.add_argument("--mechanism-id", type=int, default=157)
    parser.add_argument("--rat", action="append")
    parser.add_argument("--level", type=float, default=0.90)
    parser.add_argument("--inner-folds", type=int, default=5)
    parser.add_argument("--maxiter", type=int, default=450)
    parser.add_argument("--inner-maxiter", type=int, default=280)
    parser.add_argument("--popsize", type=int, default=12)
    parser.add_argument("--inner-popsize", type=int, default=10)
    parser.add_argument("--adaptive-draws", type=int, default=768)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.inner_folds < 2:
        raise ValueError("At least two inner grouped folds are required")
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    predictions = predictions[predictions["mechanism_id"] == args.mechanism_id].copy()
    if predictions.empty:
        raise ValueError("Selected mechanism has no predictions")
    spectra = read_experimental_spectra(args.data)
    targets = (
        build_delta_z_targets(spectra) if args.target == "delta"
        else build_multifrequency_targets(spectra)
    )
    rat_ids = sorted(targets["rat"].unique())
    selected = args.rat if args.rat else rat_ids
    unknown = set(selected).difference(rat_ids)
    if unknown:
        raise ValueError(f"Unknown rats: {sorted(unknown)}")
    jobs = [(index, rat) for index, rat in enumerate(rat_ids) if rat in selected]

    summaries, outer_frames, calibration_frames, parameter_frames = [], [], [], []
    context = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else None
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                run_outer_fold, predictions, targets, rat, index, args
            ): rat
            for index, rat in jobs
        }
        for future in as_completed(futures):
            rat = futures[future]
            summary, outer, calibration, parameters = future.result()
            summaries.append(summary)
            outer_frames.append(outer)
            calibration_frames.append(calibration)
            parameter_frames.append(parameters)
            print(
                f"interval held-out {rat}: coverage={summary['coverage']:.4f} "
                f"width={summary['median_width']:.5f} "
                f"train_minimax={summary['best_training_minimax']:.9f} "
                f"compatible={summary['compatible_count']}", flush=True,
            )
            pd.DataFrame(summaries).sort_values("outer_fold").to_csv(
                args.output / "loo_interval_rat_summary.running.csv", index=False
            )
    summary = pd.DataFrame(summaries).sort_values("outer_fold").reset_index(drop=True)
    outer = pd.concat(outer_frames, ignore_index=True).sort_values(
        ["outer_fold", "frequency_hz", "stage", "edta_concentration_mol_per_l", "time_s"]
    )
    calibration = pd.concat(calibration_frames, ignore_index=True)
    parameters = pd.concat(parameter_frames, ignore_index=True)
    summary.to_csv(args.output / "loo_interval_rat_summary.csv", index=False)
    outer.to_csv(args.output / "loo_calibrated_point_intervals.csv", index=False)
    calibration.to_csv(args.output / "inner_calibration_intervals.csv", index=False)
    parameters.to_csv(args.output / "outer_training_ensemble_parameters.csv", index=False)
    frequency = outer.groupby("frequency_hz", sort=True).agg(
        point_count=("calibrated_covered", "size"),
        coverage=("calibrated_covered", "mean"),
        median_width=("calibrated_width", "median"),
        max_width=("calibrated_width", "max"),
    ).reset_index()
    frequency["coverage_gap_to_nominal"] = frequency["coverage"] - args.level
    target_name = "delta_z" if args.target == "delta" else "r_z"
    units = "MOhm" if args.target == "delta" else "dimensionless_ratio"
    frequency.insert(0, "target", target_name)
    frequency.insert(1, "units", units)
    frequency.to_csv(args.output / "loo_interval_frequency_summary.csv", index=False)
    original = outer[outer["original_compatible_prediction_available"]]
    lower_clip_count = int(
        (outer["calibrated_lower"] > outer["unclipped_calibrated_lower"]).sum()
    )
    overall = {
        "role": "NESTED_GROUPED_CROSS_CONFORMAL_ENSEMBLE_INTERVALS",
        "target": target_name,
        "observed_column": "delta_z_mohm" if args.target == "delta" else "z_ratio",
        "units": units,
        "center_model": "frequency_surface_polarization_17_parameter",
        "mechanism_id": args.mechanism_id,
        "structure": args.structure,
        "nominal_point_coverage": args.level,
        "outer_fold_count": len(summary),
        "inner_folds_per_outer_fold": args.inner_folds,
        "calibrated_point_coverage": float(outer["calibrated_covered"].mean()),
        "coverage_gap_to_nominal": float(outer["calibrated_covered"].mean() - args.level),
        "median_interval_width": float(outer["calibrated_width"].median()),
        "max_interval_width": float(outer["calibrated_width"].max()),
        "physical_lower_bound": None if args.target == "delta" else 0.0,
        "physical_lower_clip_count": lower_clip_count,
        "physical_nonnegative_lower_clip_count": (
            lower_clip_count if args.target == "ratio" else 0
        ),
        "nominal_overall_coverage_met": bool(
            outer["calibrated_covered"].mean() >= args.level
        ),
        "all_rats_meet_nominal_coverage": bool(
            (summary["coverage"] >= args.level).all()
        ),
        "per_rat_coverage_min": float(summary["coverage"].min()),
        "per_rat_coverage_median": float(summary["coverage"].median()),
        "original_empty_compatible_outer_folds": int((summary["compatible_count"] == 0).sum()),
        "original_compatible_point_coverage": (
            float(original["original_compatible_covered"].mean()) if len(original) else None
        ),
        "original_compatible_coverage_empty_fold_rule": "NA_NOT_ZERO",
        "outer_test_use": "FINAL_EVALUATION_ONLY_NOT_FIT_OR_CALIBRATION",
        "interpretation": (
            "Delta-Z intervals are a parallel sensitivity analysis in MOhm; "
            "negative lower bounds are physically valid. This is not original "
            "R-Z Track B point compatibility."
            if args.target == "delta" else
            "Interval validation is distinct from original Track B point compatibility."
        ),
    }
    (args.output / "loo_interval_overall.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
