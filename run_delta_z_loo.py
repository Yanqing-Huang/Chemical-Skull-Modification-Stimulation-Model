#!/usr/bin/env python3
"""Fold-independent leave-one-rat-out validation for Delta-Z minimax targets."""

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
    ratio_prediction_to_delta,
)
from cranial_model.experimental import read_experimental_spectra
from cranial_model.frequency_surface_polarization import (
    COMBINED_PARAMETER_BOUNDS, frequency_surface_polarization_forward,
)
from cranial_model.multifrequency_calibration import FREQUENCIES_HZ
from run_frequency_surface_polarization import aligned_arrays


def training_arrays(predictions, train, structure):
    arrays = aligned_arrays(predictions, train, structure)
    tolerances = delta_frequency_tolerances(train)
    arrays["delta_tolerances"] = tolerances
    arrays["delta_tolerance_vector"] = np.array([tolerances[f] for f in FREQUENCIES_HZ])
    return arrays


def optimize_fold(predictions, train, held_out, fold_index, structure, maxiter, popsize):
    arrays = training_arrays(predictions, train, structure)
    if len(arrays["mechanism_ids"]) != 1:
        raise ValueError("The Delta-Z fold runner requires one fixed mechanism")

    def objective(parameters):
        return delta_candidate_metrics(parameters, arrays, train, 0, False)[0]

    result = differential_evolution(
        objective, COMBINED_PARAMETER_BOUNDS, seed=20272000 + fold_index,
        maxiter=maxiter, popsize=popsize, tol=1e-9, polish=True,
        vectorized=True, updating="deferred",
    )
    population = np.concatenate([result.population, result.x[None, :]], axis=0)
    minimax, normalized, raw, tolerances = delta_candidate_metrics(
        population, arrays, train, 0, False
    )
    compatible = minimax <= 1.0
    best_index = int(np.argmin(minimax))
    best = population[best_index]
    lower = np.array([bound[0] for bound in COMBINED_PARAMETER_BOUNDS])
    upper = np.array([bound[1] for bound in COMBINED_PARAMETER_BOUNDS])
    scale = np.maximum(upper - lower, 1.0)
    boundary = np.flatnonzero(
        (np.abs(best - lower) / scale < 1e-5)
        | (np.abs(best - upper) / scale < 1e-5)
    )
    summary = {
        "held_out": fold_index, "rat": held_out,
        "mechanism_id": int(arrays["mechanism_ids"][0]),
        "compatible_count": int(compatible.sum()),
        "best_training_minimax": float(minimax[best_index]),
        "best_training_normalized_rmse": float(normalized[best_index]),
        "degenerate_interval": bool(compatible.sum() == 1),
        "boundary_parameter_indices": ";".join(map(str, boundary)),
        **{
            f"best_training_raw_rmse_{f:g}hz_mohm": float(raw[best_index, index])
            for index, f in enumerate(FREQUENCIES_HZ)
        },
        **{
            f"tolerance_{f:g}hz_mohm": float(tolerances[f]) for f in FREQUENCIES_HZ
        },
    }
    selected = population[compatible]
    if not len(selected):
        selected = best[None, :]
    parameters = pd.DataFrame(selected, columns=[f"x{i}" for i in range(17)])
    parameters.insert(0, "held_out", fold_index)
    parameters["training_compatible"] = bool(compatible.sum()) if not compatible.any() else True
    return summary, parameters


def predict_test(parameters, predictions, test, structure, compatible_count):
    arrays = aligned_arrays(predictions, test, structure, estimate_tolerances=False)
    x = parameters[[f"x{i}" for i in range(17)]].to_numpy(dtype=float)
    ratio = frequency_surface_polarization_forward(
        x, arrays["matrix"][0], arrays["interface"][0], arrays["exposure"][0],
        arrays["stage"], arrays["dose"], arrays["time_s"],
    )
    delta = ratio_prediction_to_delta(
        ratio, arrays["frequency_index"], arrays["condition_index"],
        test["baseline_mean_mohm"].to_numpy(dtype=float),
    )
    q05, q50, q95 = np.quantile(delta, [0.05, 0.5, 0.95], axis=0)
    result = test.copy().reset_index(drop=True)
    result["sim_delta_q05_mohm"] = q05
    result["sim_delta_q50_mohm"] = q50
    result["sim_delta_q95_mohm"] = q95
    result["compatible_count"] = compatible_count
    result["degenerate_interval"] = compatible_count == 1
    result["covered"] = (
        (result["delta_z_mohm"] >= result["sim_delta_q05_mohm"])
        & (result["delta_z_mohm"] <= result["sim_delta_q95_mohm"])
    ).astype("boolean")
    result["width_mohm"] = result["sim_delta_q95_mohm"] - result["sim_delta_q05_mohm"]
    result["prediction_available"] = True
    return result


def empty_test(test):
    result = test.copy().reset_index(drop=True)
    for column in (
        "sim_delta_q05_mohm", "sim_delta_q50_mohm", "sim_delta_q95_mohm", "width_mohm"
    ):
        result[column] = np.nan
    result["compatible_count"] = 0
    result["degenerate_interval"] = False
    result["covered"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    result["prediction_available"] = False
    return result


def run_fold(predictions, targets, held_out, fold_index, args):
    train = targets[targets["rat"] != held_out].reset_index(drop=True)
    test = targets[targets["rat"] == held_out].reset_index(drop=True)
    summary, parameters = optimize_fold(
        predictions, train, held_out, fold_index, args.structure,
        args.maxiter, args.popsize,
    )
    summary["n_points"] = len(test)
    if summary["compatible_count"]:
        points = predict_test(
            parameters, predictions, test, args.structure, summary["compatible_count"]
        )
        summary["coverage"] = float(points["covered"].mean())
        summary["median_width_mohm"] = float(points["width_mohm"].median())
    else:
        points = empty_test(test)
        summary["coverage"] = np.nan
        summary["median_width_mohm"] = np.nan
    points.insert(0, "held_out", fold_index)
    return summary, points, parameters


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structure", default="two_sided_50um")
    parser.add_argument("--mechanism-id", type=int, default=19)
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
    targets = build_delta_z_targets(read_experimental_spectra(args.data))
    rat_ids = sorted(targets["rat"].unique())
    selected = args.rat if args.rat else rat_ids
    unknown = set(selected).difference(rat_ids)
    if unknown:
        raise ValueError(f"Unknown rats: {sorted(unknown)}")
    jobs = [(index, rat) for index, rat in enumerate(rat_ids) if rat in selected]
    summaries, point_frames, parameter_frames = [], [], []
    context = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else None
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                run_fold, predictions, targets, rat, index, args
            ): rat
            for index, rat in jobs
        }
        for future in as_completed(futures):
            rat = futures[future]
            summary, points, parameters = future.result()
            summaries.append(summary)
            point_frames.append(points)
            parameter_frames.append(parameters)
            print(
                f"Delta-Z held-out {rat}: best={summary['best_training_minimax']:.9f} "
                f"compatible={summary['compatible_count']} coverage={summary['coverage']}",
                flush=True,
            )
            pd.DataFrame(summaries).sort_values("held_out").to_csv(
                args.output / "loo_delta_z_rat_summary.running.csv", index=False
            )
    summary = pd.DataFrame(summaries).sort_values("held_out").reset_index(drop=True)
    points = pd.concat(point_frames, ignore_index=True).sort_values(
        ["held_out", "frequency_hz", "stage", "edta_concentration_mol_per_l", "time_s"]
    )
    parameters = pd.concat(parameter_frames, ignore_index=True)
    summary.to_csv(args.output / "loo_delta_z_rat_summary.csv", index=False)
    points.to_csv(args.output / "loo_delta_z_point_intervals.csv", index=False)
    parameters.to_csv(args.output / "loo_delta_z_fold_parameters.csv", index=False)
    available = points[points["prediction_available"]]
    frequency = available.groupby("frequency_hz", sort=True).agg(
        available_points=("covered", "size"), coverage=("covered", "mean"),
        median_width_mohm=("width_mohm", "median"),
    ).reset_index()
    frequency.to_csv(args.output / "loo_delta_z_frequency_summary.csv", index=False)
    overall = {
        "role": "FOLD_INDEPENDENT_DELTA_Z_COMPATIBLE_SETS",
        "target": "Delta-Z in MOhm; own-rat/site/frequency baseline subtracted before site averaging",
        "center_model": "frequency_surface_polarization_17_parameter",
        "mechanism_id": args.mechanism_id,
        "structure": args.structure,
        "fold_count": len(summary),
        "empty_fold_count": int((summary["compatible_count"] == 0).sum()),
        "degenerate_fold_count": int((summary["compatible_count"] == 1).sum()),
        "all_folds_nonempty": bool((summary["compatible_count"] > 0).all()),
        "all_folds_nondegenerate": bool((summary["compatible_count"] > 1).all()),
        "point_coverage": float(available["covered"].mean()) if len(available) else None,
        "median_interval_width_mohm": float(available["width_mohm"].median()) if len(available) else None,
        "per_rat_coverage_min": float(summary["coverage"].min()),
        "per_rat_coverage_median": float(summary["coverage"].median()),
        "available_point_count": len(available),
        "total_point_count": len(points),
        "coverage_empty_fold_rule": "NA_NOT_ZERO",
        "interpretation": "Delta-Z sensitivity target; does not rewrite original R-Z Track B results.",
    }
    (args.output / "loo_delta_z_overall.json").write_text(
        json.dumps(overall, indent=2), encoding="utf-8"
    )
    print(json.dumps(overall, indent=2), flush=True)


if __name__ == "__main__":
    main()
