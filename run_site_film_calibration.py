#!/usr/bin/env python3
"""V5a three-site local-PDE and dynamic-film diagnostic.

DE results are explicitly full-data reachability diagnostics, not a compatible
set or leave-one-rat-out evidence.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from cranial_model.core import load_parameters, simulate_scenario
from cranial_model.experimental import read_experimental_spectra
from cranial_model.multifrequency_calibration import (
    FREQUENCIES_HZ, build_multifrequency_targets, frequency_tolerances,
)
from cranial_model.near_surface import condition_near_surface_states
from cranial_model.site_film_calibration import (
    PARAMETER_BOUNDS, condition_site_alignment, rat_condition_mean, site_film_forward,
)
from cranial_model.site_geometry import apply_site_geometry, sample_site_geometries
from run_equivalent_calibration import DEFAULT_DATA
from run_near_surface_calibration import reconstruct_v2_mechanisms
from run_prior_mechanism import MECHANISM_PATHS, STAGES


ROOT = Path(__file__).resolve().parent


def run_site_candidate(base_values, candidate, geometry, doses, cells, max_step):
    from cranial_model.ensemble import apply_candidate

    shared = apply_candidate(base_values, MECHANISM_PATHS, candidate)
    rows = []
    for _, site_geometry in geometry.iterrows():
        values = apply_site_geometry(shared, site_geometry)
        for dose in doses:
            scenario = simulate_scenario(
                values, dose, cells_per_layer=cells, max_step_s=max_step,
                stage_names=STAGES,
            )
            for row in condition_near_surface_states(scenario, dose):
                rows.append({"site": site_geometry["site"], **row})
    return rows


def generate_predictions(args, mechanism, geometry):
    document = load_parameters(args.parameters)
    base_values = document["values"]
    doses = sorted(float(d) for d in base_values["scenarios"]["edta_concentrations_mol_per_l"])
    rows, failures = [], []
    context = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else None
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                run_site_candidate, base_values,
                row[list(MECHANISM_PATHS)].to_numpy(dtype=float),
                geometry[geometry["mechanism_id"] == int(row["mechanism_id"])],
                doses, args.cells, args.max_step,
            ): int(row["mechanism_id"])
            for _, row in mechanism.iterrows()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            mechanism_id = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001
                failures.append({"mechanism_id": mechanism_id, "error": str(error)})
                print(f"mechanism {mechanism_id} failed: {error}", flush=True)
                continue
            rows.extend({"mechanism_id": mechanism_id, **row} for row in result)
            print(f"V5a local-PDE sweep {completed}/{len(futures)}", flush=True)
            pd.DataFrame(rows).to_csv(args.output / "v5a_condition_predictions.partial.csv", index=False)
    predictions = pd.DataFrame(rows)
    predictions.to_csv(args.output / "v5a_condition_predictions.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output / "v5a_failures.csv", index=False)
    if failures:
        raise RuntimeError(f"{len(failures)} V5a mechanism candidates failed")
    return predictions


def reachability_diagnostic(predictions, targets, depth_um, side, output):
    condition_index, matrix, po, pi, xo, xi, mechanism_ids = condition_site_alignment(
        predictions, targets, depth_um
    )
    observed = targets["z_ratio"].to_numpy(dtype=float)
    frequency_values = targets["frequency_hz"].to_numpy(dtype=float)
    frequency_index = targets["frequency_hz"].map(
        {frequency: index for index, frequency in enumerate(FREQUENCIES_HZ)}
    ).to_numpy(dtype=int)
    tolerances = frequency_tolerances(targets)
    tolerance_vector = np.array([tolerances[f] for f in FREQUENCIES_HZ])
    counts = np.array([(frequency_values == f).sum() for f in FREQUENCIES_HZ])
    rows = []
    for mechanism_index, mechanism_id in enumerate(mechanism_ids):
        def metrics(parameters):
            site_prediction = site_film_forward(
                parameters, matrix[mechanism_index], po[mechanism_index],
                pi[mechanism_index], xo[mechanism_index], xi[mechanism_index], side,
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
            objective, PARAMETER_BOUNDS, seed=20261410 + int(mechanism_id),
            maxiter=360, popsize=12, tol=1e-8, polish=True,
            vectorized=True, updating="deferred",
        )
        normalized, raw = metrics(result.x)
        rows.append({
            "mechanism_id": int(mechanism_id),
            "side": side,
            "depth_um": int(depth_um),
            "minimax": float(result.fun),
            "normalized_rmse": float(normalized[0]),
            "compatible_reachability_only": bool(result.fun <= 1.0),
            **{
                f"raw_rmse_{f:g}hz": float(raw[0, index])
                for index, f in enumerate(FREQUENCIES_HZ)
            },
            **{f"x{index}": float(value) for index, value in enumerate(result.x)},
        })
        print(
            f"V5a DE {side} mechanism {mechanism_id}: minimax={result.fun:.9f}",
            flush=True,
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"v5a_{side}_{depth_um}um_de_reachability.csv", index=False)
    return frame


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--parameters", type=Path, default=ROOT / "parameters" / "theoretical_prior.json")
    parser.add_argument("--v2-results", type=Path, default=ROOT / "results" / "equivalent_calibration_v2")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=25)
    parser.add_argument("--cells", nargs=3, type=int, default=[12, 6, 6])
    parser.add_argument("--max-step", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--geometry-seed", type=int, default=20261401)
    parser.add_argument("--reuse-sweep", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    document = load_parameters(args.parameters)
    mechanism = reconstruct_v2_mechanisms(args.samples, args.v2_results, 20261113)
    geometry = sample_site_geometries(
        mechanism["mechanism_id"].to_numpy(), document, args.geometry_seed
    )
    mechanism.to_csv(args.output / "v5a_shared_mechanisms.csv", index=False, float_format="%.17g")
    geometry.to_csv(args.output / "v5a_site_geometry.csv", index=False, float_format="%.17g")
    if args.reuse_sweep:
        predictions = pd.read_csv(args.output / "v5a_condition_predictions.csv")
    else:
        predictions = generate_predictions(args, mechanism, geometry)
    targets = build_multifrequency_targets(read_experimental_spectra(args.data))
    # Exact native sanity check before any optimization.
    condition_index, matrix, po, pi, xo, xi, _ = condition_site_alignment(predictions, targets, 100)
    native = site_film_forward(
        np.zeros(12), np.ones((3, 20)), np.ones((3, 20)), np.ones((3, 20)),
        np.zeros((3, 20)), np.zeros((3, 20)),
    )
    if not np.allclose(native, 1.0, rtol=0.0, atol=1e-12):
        raise RuntimeError("V5a native baseline is not exactly one")
    summaries = {}
    for side in ("two_sided", "outer_only"):
        frame = reachability_diagnostic(predictions, targets, 100, side, args.output)
        best = frame.loc[frame["minimax"].idxmin()]
        summaries[side] = {
            "role": "FULL_DATA_DE_REACHABILITY_NOT_COMPATIBLE_SET",
            "compatible_count": int(frame["compatible_reachability_only"].sum()),
            "best_minimax": float(best["minimax"]),
            "best_mechanism_id": int(best["mechanism_id"]),
            "best_normalized_rmse": float(best["normalized_rmse"]),
        }
    (args.output / "v5a_reachability_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
