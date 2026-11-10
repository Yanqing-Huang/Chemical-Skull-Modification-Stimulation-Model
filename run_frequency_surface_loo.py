#!/usr/bin/env python3
"""Fold-independent DE leave-one-rat-out validation for the 19-parameter model."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from cranial_model.experimental import read_experimental_spectra
from cranial_model.frequency_surface_polarization import (
    TIME_COMBINED_PARAMETER_BOUNDS, frequency_surface_time_polarization_forward,
)
from cranial_model.multifrequency_calibration import (
    FREQUENCIES_HZ, build_multifrequency_targets,
)
from run_frequency_surface_polarization import aligned_arrays, candidate_metrics


def optimize_fold(predictions, train, held_out, fold_index, structure, maxiter, popsize):
    arrays = aligned_arrays(predictions, train, structure)
    if len(arrays["mechanism_ids"]) != 1:
        raise ValueError("This fold runner requires exactly one fixed mechanism trajectory")
    mechanism_id = int(arrays["mechanism_ids"][0])

    def objective(parameters):
        return candidate_metrics(parameters, arrays, 0, True, True)[0]

    result = differential_evolution(
        objective, TIME_COMBINED_PARAMETER_BOUNDS,
        seed=20265000 + fold_index, maxiter=maxiter, popsize=popsize,
        tol=1e-9, polish=True, vectorized=True, updating="deferred",
    )
    population = np.concatenate([result.population, result.x[None, :]], axis=0)
    minimax, normalized, raw = candidate_metrics(population, arrays, 0, True, True)
    compatible = minimax <= 1.0
    best_index = int(np.argmin(minimax))
    best = population[best_index]
    lower = np.array([bound[0] for bound in TIME_COMBINED_PARAMETER_BOUNDS])
    upper = np.array([bound[1] for bound in TIME_COMBINED_PARAMETER_BOUNDS])
    scale = np.maximum(upper - lower, 1.0)
    boundary = np.flatnonzero(
        (np.abs(best - lower) / scale < 1e-5)
        | (np.abs(best - upper) / scale < 1e-5)
    )
    summary = {
        "held_out": fold_index, "mechanism_id": mechanism_id,
        "compatible_count": int(compatible.sum()),
        "best_training_minimax": float(minimax[best_index]),
        "best_training_normalized_rmse": float(normalized[best_index]),
        "degenerate_interval": bool(compatible.sum() == 1),
        "boundary_parameter_indices": ";".join(map(str, boundary)),
        **{
            f"best_training_raw_rmse_{f:g}hz": float(raw[best_index, index])
            for index, f in enumerate(FREQUENCIES_HZ)
        },
        **{
            f"tolerance_{f:g}hz": float(arrays["tolerances"][f])
            for f in FREQUENCIES_HZ
        },
    }
    selected = population[compatible]
    if not len(selected):
        selected = best[None, :]
    parameters = pd.DataFrame(selected, columns=[f"x{i}" for i in range(19)])
    parameters.insert(0, "held_out", fold_index)
    parameters.insert(1, "mechanism_id", mechanism_id)
    parameters["training_compatible"] = bool(compatible.sum()) if len(selected) == 1 and not compatible.any() else True
    return summary, parameters


def predict_test(parameters, predictions, test, structure):
    arrays = aligned_arrays(predictions, test, structure, estimate_tolerances=False)
    x = parameters[[f"x{i}" for i in range(19)]].to_numpy(dtype=float)
    prediction = frequency_surface_time_polarization_forward(
        x, arrays["matrix"][0], arrays["interface"][0], arrays["exposure"][0],
        arrays["stage"], arrays["dose"], arrays["time_s"],
    )
    values = prediction[:, arrays["frequency_index"], arrays["condition_index"]]
    q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95], axis=0)
    result = test.copy().reset_index(drop=True)
    result["sim_q05"], result["sim_q50"], result["sim_q95"] = q05, q50, q95
    result["compatible_count"] = len(parameters)
    result["degenerate_interval"] = len(parameters) == 1
    result["covered"] = (
        (result["z_ratio"] >= result["sim_q05"])
        & (result["z_ratio"] <= result["sim_q95"])
    ).astype("boolean")
    result["width"] = result["sim_q95"] - result["sim_q05"]
    result["prediction_available"] = True
    return result


def empty_test(test, compatible_count=0):
    result = test.copy().reset_index(drop=True)
    for column in ("sim_q05", "sim_q50", "sim_q95", "width"):
        result[column] = np.nan
    result["compatible_count"] = compatible_count
    result["degenerate_interval"] = False
    result["covered"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["prediction_available"] = False
    return result


def run_fold(predictions, targets, held_out, fold_index, structure, maxiter, popsize):
    train = targets[targets["rat"] != held_out].reset_index(drop=True)
    test = targets[targets["rat"] == held_out].reset_index(drop=True)
    summary, parameters = optimize_fold(
        predictions, train, held_out, fold_index, structure, maxiter, popsize
    )
    summary["rat"] = held_out
    summary["n_points"] = len(test)
    if summary["compatible_count"]:
        points = predict_test(parameters, predictions, test, structure)
        summary["coverage"] = float(points["covered"].mean())
        summary["median_width"] = float(points["width"].median())
    else:
        points = empty_test(test)
        summary["coverage"] = np.nan
        summary["median_width"] = np.nan
    points.insert(0, "held_out", fold_index)
    return summary, points, parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structure", default="two_sided_50um")
    parser.add_argument("--mechanism-id", type=int, default=157)
    parser.add_argument("--rat", action="append")
    parser.add_argument("--maxiter", type=int, default=600)
    parser.add_argument("--popsize", type=int, default=14)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    predictions = predictions[predictions["mechanism_id"] == args.mechanism_id].copy()
    if predictions.empty:
        raise ValueError("Selected mechanism has no predictions")
    targets = build_multifrequency_targets(read_experimental_spectra(args.data))
    rat_ids = sorted(targets["rat"].unique())
    selected_rats = args.rat if args.rat else rat_ids
    unknown = set(selected_rats).difference(rat_ids)
    if unknown:
        raise ValueError(f"Unknown rats: {sorted(unknown)}")
    jobs = list(enumerate(selected_rats))

    summaries, point_frames, parameter_frames = [], [], []
    context = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else None
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                run_fold, predictions, targets, rat, fold_index, args.structure,
                args.maxiter, args.popsize,
            ): rat
            for fold_index, rat in jobs
        }
        for future in as_completed(futures):
            rat = futures[future]
            summary, points, parameters = future.result()
            summaries.append(summary)
            point_frames.append(points)
            parameter_frames.append(parameters)
            print(
                f"held-out {rat}: best={summary['best_training_minimax']:.12f} "
                f"compatible={summary['compatible_count']}", flush=True,
            )
            pd.DataFrame(summaries).sort_values("held_out").to_csv(
                args.output / "loo_rat_summary.running.csv", index=False
            )
    summary = pd.DataFrame(summaries).sort_values("held_out").reset_index(drop=True)
    points = pd.concat(point_frames, ignore_index=True).sort_values(
        ["held_out", "frequency_hz", "stage", "edta_concentration_mol_per_l", "time_s"]
    )
    parameters = pd.concat(parameter_frames, ignore_index=True).sort_values("held_out")
    summary.to_csv(args.output / "loo_rat_summary.csv", index=False)
    points.to_csv(args.output / "loo_point_intervals.csv", index=False)
    parameters.to_csv(args.output / "loo_fold_optimized_parameters.csv", index=False)
    available = points[points["prediction_available"]]
    overall = {
        "role": "FOLD_INDEPENDENT_DE_COMPATIBLE_SETS_NO_FULL_DATA_CANDIDATES",
        "model": "frequency_surface_time_polarization_19_parameter",
        "mechanism_id": args.mechanism_id,
        "structure": args.structure,
        "fold_count": len(summary),
        "empty_fold_count": int((summary["compatible_count"] == 0).sum()),
        "degenerate_fold_count": int((summary["compatible_count"] == 1).sum()),
        "all_folds_nonempty": bool((summary["compatible_count"] > 0).all()),
        "all_folds_nondegenerate": bool((summary["compatible_count"] > 1).all()),
        "point_coverage": float(available["covered"].mean()) if len(available) else None,
        "median_interval_width": float(available["width"].median()) if len(available) else None,
        "available_point_count": len(available),
        "total_point_count": len(points),
        "coverage_empty_fold_rule": "NA_NOT_ZERO",
        "note": "Conditional on the fixed mechanism trajectory; not external validation.",
    }
    (args.output / "loo_overall.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
