#!/usr/bin/env python3
"""Full-data Delta-Z reachability diagnostic without changing original Track B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from cranial_model.delta_z_calibration import (
    build_delta_z_targets, delta_candidate_metrics, delta_frequency_tolerances,
)
from cranial_model.experimental import read_experimental_spectra
from cranial_model.frequency_surface_polarization import (
    COMBINED_PARAMETER_BOUNDS, TIME_COMBINED_PARAMETER_BOUNDS,
)
from cranial_model.multifrequency_calibration import FREQUENCIES_HZ
from run_frequency_surface_polarization import aligned_arrays


def run_diagnostic(predictions, targets, structure, output, time_response, maxiter, popsize):
    arrays = aligned_arrays(predictions, targets, structure)
    tolerances = delta_frequency_tolerances(targets)
    arrays["delta_tolerances"] = tolerances
    arrays["delta_tolerance_vector"] = np.array([tolerances[f] for f in FREQUENCIES_HZ])
    bounds = TIME_COMBINED_PARAMETER_BOUNDS if time_response else COMBINED_PARAMETER_BOUNDS
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    rows = []
    for mechanism_index, mechanism_id in enumerate(arrays["mechanism_ids"]):
        def objective(parameters):
            return delta_candidate_metrics(
                parameters, arrays, targets, mechanism_index, time_response
            )[0]

        result = differential_evolution(
            objective, bounds,
            seed=20271000 + int(mechanism_id) + 200 * int(time_response),
            maxiter=maxiter, popsize=popsize, tol=1e-9, polish=True,
            vectorized=True, updating="deferred",
        )
        minimax, normalized, raw, _ = delta_candidate_metrics(
            result.x, arrays, targets, mechanism_index, time_response
        )
        scale = np.maximum(upper - lower, 1.0)
        boundary = np.flatnonzero(
            (np.abs(result.x - lower) / scale < 1e-5)
            | (np.abs(result.x - upper) / scale < 1e-5)
        )
        rows.append({
            "mechanism_id": int(mechanism_id), "structure": structure,
            "model": "time_combined_19" if time_response else "combined_17",
            "target": "DELTA_Z_POST_MINUS_OWN_BASELINE_MOHM",
            "minimax": float(minimax[0]), "normalized_rmse": float(normalized[0]),
            "compatible_reachability_only": bool(minimax[0] <= 1.0),
            "boundary_parameter_indices": ";".join(map(str, boundary)),
            **{
                f"raw_rmse_{f:g}hz_mohm": float(raw[0, index])
                for index, f in enumerate(FREQUENCIES_HZ)
            },
            **{f"x{i}": float(value) for i, value in enumerate(result.x)},
        })
        print(
            f"Delta-Z {structure} mechanism {mechanism_id} "
            f"{'19p' if time_response else '17p'}: minimax={minimax[0]:.12f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    stem = "delta_z_time19" if time_response else "delta_z_combined17"
    frame.to_csv(output / f"{stem}_{structure}_de_reachability.csv", index=False)
    best = frame.loc[frame["minimax"].idxmin()]
    summary = {
        "role": "FULL_DATA_DELTA_Z_REACHABILITY_NOT_ORIGINAL_TRACK_B",
        "target": "post-treatment Z minus same-rat/site/frequency baseline, site-averaged in MOhm",
        "prediction_conversion": "mean_site_baseline * (common_predicted_R_Z - 1)",
        "structure": structure,
        "model": stem,
        "compatible_count": int(frame["compatible_reachability_only"].sum()),
        "best_mechanism_id": int(best["mechanism_id"]),
        "best_minimax": float(best["minimax"]),
        "best_normalized_rmse": float(best["normalized_rmse"]),
        "best_raw_rmse_mohm": {
            f"{f:g}": float(best[f"raw_rmse_{f:g}hz_mohm"]) for f in FREQUENCIES_HZ
        },
        "tolerances_mohm": {f"{f:g}": tolerances[f] for f in FREQUENCIES_HZ},
        "best_boundary_parameter_indices": best["boundary_parameter_indices"],
    }
    (output / f"{stem}_{structure}_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--structure", default="two_sided_50um")
    parser.add_argument("--mechanism-id", type=int, action="append")
    parser.add_argument("--time-response", action="store_true")
    parser.add_argument("--maxiter", type=int, default=650)
    parser.add_argument("--popsize", type=int, default=14)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    if args.mechanism_id:
        predictions = predictions[predictions["mechanism_id"].isin(args.mechanism_id)]
    if predictions.empty:
        raise ValueError("No selected mechanism predictions")
    targets = build_delta_z_targets(read_experimental_spectra(args.data))
    run_diagnostic(
        predictions, targets, args.structure, args.output,
        args.time_response, args.maxiter, args.popsize,
    )


if __name__ == "__main__":
    main()
