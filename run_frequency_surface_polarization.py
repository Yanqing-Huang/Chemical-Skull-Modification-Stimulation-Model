#!/usr/bin/env python3
"""Full-data DE reachability for the reproduced best observation model plus V5b."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from cranial_model.experimental import read_experimental_spectra
from cranial_model.frequency_surface_polarization import (
    COMBINED_PARAMETER_BOUNDS, OBSERVATION_PARAMETER_BOUNDS,
    TIME_COMBINED_PARAMETER_BOUNDS, frequency_surface_forward,
    frequency_surface_polarization_forward,
    frequency_surface_time_polarization_forward,
)
from cranial_model.multifrequency_calibration import (
    FREQUENCIES_HZ, build_multifrequency_targets, frequency_tolerances,
)
from cranial_model.near_surface_calibration import _condition_alignment


def aligned_arrays(
    predictions: pd.DataFrame, targets: pd.DataFrame, structure: str,
    estimate_tolerances: bool = True,
):
    condition_index, matrix, interface, exposure, mechanism_ids = _condition_alignment(
        predictions, targets, structure
    )
    conditions = targets[["stage", "edta_concentration_mol_per_l", "time_s"]].drop_duplicates()
    conditions = conditions.reset_index(drop=True)
    observed = targets["z_ratio"].to_numpy(dtype=float)
    frequency_values = targets["frequency_hz"].to_numpy(dtype=float)
    frequency_index = targets["frequency_hz"].map(
        {frequency: index for index, frequency in enumerate(FREQUENCIES_HZ)}
    ).to_numpy(dtype=int)
    tolerances = frequency_tolerances(targets) if estimate_tolerances else {}
    tolerance_vector = (
        np.array([tolerances[f] for f in FREQUENCIES_HZ])
        if estimate_tolerances else np.full(len(FREQUENCIES_HZ), np.nan)
    )
    counts = np.array([(frequency_values == f).sum() for f in FREQUENCIES_HZ])
    return {
        "condition_index": condition_index,
        "matrix": matrix,
        "interface": interface,
        "exposure": exposure,
        "mechanism_ids": mechanism_ids,
        "stage": conditions["stage"].to_numpy(),
        "dose": conditions["edta_concentration_mol_per_l"].to_numpy(dtype=float),
        "time_s": conditions["time_s"].to_numpy(dtype=float),
        "observed": observed,
        "frequency_values": frequency_values,
        "frequency_index": frequency_index,
        "tolerances": tolerances,
        "tolerance_vector": tolerance_vector,
        "counts": counts,
    }


def candidate_metrics(
    parameters, arrays, mechanism_index: int, combined: bool,
    time_response: bool = False,
):
    values = np.asarray(parameters, dtype=float)
    dimensions = 19 if time_response else (17 if combined else 13)
    if values.ndim == 1:
        values = values[None, :]
    elif values.shape[0] == dimensions and values.shape[1] != dimensions:
        values = values.T
    if values.ndim != 2 or values.shape[1] != dimensions:
        raise ValueError(f"Expected candidate matrices with {dimensions} columns")
    if time_response:
        forward = frequency_surface_time_polarization_forward
    elif combined:
        forward = frequency_surface_polarization_forward
    else:
        forward = frequency_surface_forward
    prediction = forward(
        values, arrays["matrix"][mechanism_index], arrays["interface"][mechanism_index],
        arrays["exposure"][mechanism_index], arrays["stage"], arrays["dose"],
        arrays["time_s"],
    )
    residual = (
        prediction[:, arrays["frequency_index"], arrays["condition_index"]]
        - arrays["observed"][None, :]
    )
    squared = np.stack([
        np.sum(residual[:, arrays["frequency_values"] == f] ** 2, axis=1)
        for f in FREQUENCIES_HZ
    ], axis=1)
    raw = np.sqrt(squared / arrays["counts"][None, :])
    normalized = np.sqrt(
        np.sum(squared / arrays["tolerance_vector"][None, :] ** 2, axis=1)
        / residual.shape[1]
    )
    minimax = np.maximum(
        normalized, np.max(raw / arrays["tolerance_vector"][None, :], axis=1)
    )
    return minimax, normalized, raw


def run_diagnostic(
    predictions, targets, structure, output, combined, maxiter, popsize,
    time_response=False,
):
    arrays = aligned_arrays(predictions, targets, structure)
    if time_response:
        bounds = TIME_COMBINED_PARAMETER_BOUNDS
    elif combined:
        bounds = COMBINED_PARAMETER_BOUNDS
    else:
        bounds = OBSERVATION_PARAMETER_BOUNDS
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    rows = []
    for mechanism_index, mechanism_id in enumerate(arrays["mechanism_ids"]):
        def objective(parameters):
            return candidate_metrics(
                parameters, arrays, mechanism_index, combined, time_response
            )[0]

        result = differential_evolution(
            objective, bounds,
            seed=20263000 + int(mechanism_id) + 100 * int(combined) + 200 * int(time_response),
            maxiter=maxiter, popsize=popsize, tol=1e-9, polish=True,
            vectorized=True, updating="deferred",
        )
        minimax, normalized, raw = candidate_metrics(
            result.x, arrays, mechanism_index, combined, time_response
        )
        scale = np.maximum(np.abs(upper - lower), 1.0)
        at_boundary = np.flatnonzero(
            (np.abs(result.x - lower) / scale < 1e-5)
            | (np.abs(result.x - upper) / scale < 1e-5)
        )
        rows.append({
            "mechanism_id": int(mechanism_id), "structure": structure,
            "model": (
                "time_combined" if time_response
                else ("combined" if combined else "observation_baseline")
            ),
            "minimax": float(minimax[0]), "normalized_rmse": float(normalized[0]),
            "compatible_reachability_only": bool(minimax[0] <= 1.0),
            "boundary_parameter_indices": ";".join(map(str, at_boundary)),
            **{
                f"raw_rmse_{f:g}hz": float(raw[0, index])
                for index, f in enumerate(FREQUENCIES_HZ)
            },
            **{f"x{index}": float(value) for index, value in enumerate(result.x)},
        })
        label = "V5b-time-combined" if time_response else (
            "V5b-combined" if combined else "best-observation"
        )
        print(
            f"{label} {structure} mechanism {mechanism_id}: "
            f"minimax={minimax[0]:.12f}", flush=True,
        )
    frame = pd.DataFrame(rows)
    stem = "v5b_time_best_observation" if time_response else (
        "v5b_best_observation" if combined else "best_observation"
    )
    frame.to_csv(output / f"{stem}_{structure}_de_reachability.csv", index=False)
    best = frame.loc[frame["minimax"].idxmin()]
    summary = {
        "role": "FULL_DATA_DE_REACHABILITY_NOT_COMPATIBLE_SET",
        "model": stem,
        "structure": structure,
        "compatible_count": int(frame["compatible_reachability_only"].sum()),
        "best_mechanism_id": int(best["mechanism_id"]),
        "best_minimax": float(best["minimax"]),
        "best_normalized_rmse": float(best["normalized_rmse"]),
        "best_raw_rmse": {
            f"{f:g}": float(best[f"raw_rmse_{f:g}hz"]) for f in FREQUENCIES_HZ
        },
        "best_boundary_parameter_indices": best["boundary_parameter_indices"],
        "tolerances": {f"{f:g}": arrays["tolerances"][f] for f in FREQUENCIES_HZ},
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
    parser.add_argument("--combined", action="store_true")
    parser.add_argument("--time-response", action="store_true")
    parser.add_argument("--maxiter", type=int, default=500)
    parser.add_argument("--popsize", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    if args.mechanism_id:
        predictions = predictions[predictions["mechanism_id"].isin(args.mechanism_id)]
    if predictions.empty:
        raise ValueError("No selected mechanism predictions")
    targets = build_multifrequency_targets(read_experimental_spectra(args.data))
    if args.time_response and not args.combined:
        raise ValueError("--time-response requires --combined")
    run_diagnostic(
        predictions, targets, args.structure, args.output,
        args.combined, args.maxiter, args.popsize, args.time_response,
    )


if __name__ == "__main__":
    main()
