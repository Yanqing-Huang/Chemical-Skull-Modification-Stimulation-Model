#!/usr/bin/env python3
"""Full-data V5b diploe-polarization reachability diagnostic, not a compatible set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from cranial_model.diploe_polarization import (
    POLARIZATION_PARAMETER_BOUNDS, diploe_polarization_forward,
)
from cranial_model.experimental import read_experimental_spectra
from cranial_model.multifrequency_calibration import (
    FREQUENCIES_HZ, build_multifrequency_targets, frequency_tolerances,
)
from cranial_model.site_film_calibration import (
    CONDITION_KEYS, PARAMETER_BOUNDS, condition_site_alignment, rat_condition_mean,
)


def run_diagnostic(predictions, targets, depth_um, side, output, maxiter=420):
    condition_index, matrix, po, pi, xo, xi, mechanism_ids = condition_site_alignment(
        predictions, targets, depth_um
    )
    conditions = targets[CONDITION_KEYS].drop_duplicates().reset_index(drop=True)
    stage = conditions["stage"].to_numpy()
    time_s = conditions["time_s"].to_numpy(dtype=float)
    observed = targets["z_ratio"].to_numpy(dtype=float)
    frequency_values = targets["frequency_hz"].to_numpy(dtype=float)
    frequency_index = targets["frequency_hz"].map(
        {frequency: index for index, frequency in enumerate(FREQUENCIES_HZ)}
    ).to_numpy(dtype=int)
    tolerances = frequency_tolerances(targets)
    tolerance_vector = np.array([tolerances[f] for f in FREQUENCIES_HZ])
    counts = np.array([(frequency_values == f).sum() for f in FREQUENCIES_HZ])
    bounds = [*PARAMETER_BOUNDS, *POLARIZATION_PARAMETER_BOUNDS]
    lower = np.array([bound[0] for bound in bounds])
    upper = np.array([bound[1] for bound in bounds])
    rows = []
    for mechanism_index, mechanism_id in enumerate(mechanism_ids):
        def metrics(parameters):
            values = np.asarray(parameters, dtype=float)
            if values.ndim == 1:
                values = values[None, :]
            elif values.shape[0] == 16 and values.shape[1] != 16:
                values = values.T
            site_prediction = diploe_polarization_forward(
                values[:, :12], values[:, 12:], matrix[mechanism_index],
                po[mechanism_index], pi[mechanism_index], xo[mechanism_index],
                xi[mechanism_index], stage, time_s, side,
            )
            prediction = rat_condition_mean(site_prediction)
            residual = prediction[:, frequency_index, condition_index] - observed[None, :]
            squared = np.stack([
                np.sum(residual[:, frequency_values == f] ** 2, axis=1)
                for f in FREQUENCIES_HZ
            ], axis=1)
            raw = np.sqrt(squared / counts[None, :])
            normalized = np.sqrt(
                np.sum(squared / tolerance_vector[None, :] ** 2, axis=1)
                / residual.shape[1]
            )
            return normalized, raw

        def objective(parameters):
            normalized, raw = metrics(parameters)
            return np.maximum(normalized, np.max(raw / tolerance_vector[None, :], axis=1))

        result = differential_evolution(
            objective, bounds, seed=20262000 + int(mechanism_id) + depth_um,
            maxiter=maxiter, popsize=12, tol=1e-8, polish=True,
            vectorized=True, updating="deferred",
        )
        normalized, raw = metrics(result.x)
        scale = np.maximum(np.abs(upper - lower), 1.0)
        at_boundary = np.flatnonzero(
            (np.abs(result.x - lower) / scale < 1e-5)
            | (np.abs(result.x - upper) / scale < 1e-5)
        )
        rows.append({
            "mechanism_id": int(mechanism_id), "side": side, "depth_um": depth_um,
            "minimax": float(result.fun), "normalized_rmse": float(normalized[0]),
            "compatible_reachability_only": bool(result.fun <= 1.0),
            "boundary_parameter_indices": ";".join(map(str, at_boundary)),
            **{
                f"raw_rmse_{f:g}hz": float(raw[0, index])
                for index, f in enumerate(FREQUENCIES_HZ)
            },
            **{f"x{index}": float(value) for index, value in enumerate(result.x)},
        })
        print(
            f"V5b DE {side} {depth_um}um mechanism {mechanism_id}: "
            f"minimax={result.fun:.9f}", flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"v5b_{side}_{depth_um}um_de_reachability.csv", index=False)
    best = frame.loc[frame["minimax"].idxmin()]
    summary = {
        "role": "FULL_DATA_DE_REACHABILITY_NOT_COMPATIBLE_SET",
        "model": "V5b treatment-induced series diploe polarization; V5a film remains parallel",
        "side": side, "depth_um": depth_um,
        "compatible_count": int(frame["compatible_reachability_only"].sum()),
        "best_mechanism_id": int(best["mechanism_id"]),
        "best_minimax": float(best["minimax"]),
        "best_normalized_rmse": float(best["normalized_rmse"]),
        "best_raw_rmse": {
            f"{f:g}": float(best[f"raw_rmse_{f:g}hz"]) for f in FREQUENCIES_HZ
        },
        "best_boundary_parameter_indices": best["boundary_parameter_indices"],
        "tolerances": {f"{f:g}": tolerances[f] for f in FREQUENCIES_HZ},
    }
    (output / f"v5b_{side}_{depth_um}um_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--depth", type=int, default=50)
    parser.add_argument("--side", choices=("two_sided", "outer_only"), default="two_sided")
    parser.add_argument("--mechanism-id", type=int, action="append")
    parser.add_argument("--maxiter", type=int, default=420)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    predictions = pd.read_csv(args.predictions)
    if args.mechanism_id:
        predictions = predictions[predictions["mechanism_id"].isin(args.mechanism_id)]
    if predictions.empty:
        raise ValueError("No selected V5a predictions")
    targets = build_multifrequency_targets(read_experimental_spectra(args.data))
    run_diagnostic(predictions, targets, args.depth, args.side, args.output, args.maxiter)


if __name__ == "__main__":
    main()
