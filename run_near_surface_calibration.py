#!/usr/bin/env python3
"""Track B v4a/v4b near-surface calibration with unchanged v3 acceptance rules."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from cranial_model.core import load_parameters, simulate_scenario
from cranial_model.ensemble import sample_lhs_log10
from cranial_model.experimental import read_experimental_spectra
from cranial_model.near_surface import STRUCTURES, condition_near_surface_states
from cranial_model.near_surface_calibration import (
    EXPOSURE_LOGIT_SLOPE_RANGE, WEIGHT_COLUMNS, leave_one_rat_out,
    prediction_quantiles, sample_near_surface_lhs, score_near_surface_candidates,
)
from cranial_model.multifrequency_calibration import (
    FREQUENCIES_HZ, build_multifrequency_targets, frequency_tolerances,
    residual_drift_by_frequency,
)
from run_equivalent_calibration import DEFAULT_DATA, EQUIVALENT_SPACE
from run_prior_mechanism import MECHANISM_PATHS, STAGES


ROOT = Path(__file__).resolve().parent
CONDITION_KEYS = ["stage", "edta_concentration_mol_per_l", "time_s"]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json_safe(value):
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def reconstruct_v2_mechanisms(sample_count: int, v2_results: Path, seed: int) -> pd.DataFrame:
    """Regenerate full-precision v2 inputs instead of round-tripping its display CSV."""
    saved = pd.read_csv(v2_results / "equivalent_lhs_samples.csv")
    saved = saved[["mechanism_id", *MECHANISM_PATHS]].drop_duplicates("mechanism_id")
    saved = saved.sort_values("mechanism_id").reset_index(drop=True)
    total_count = len(saved)
    if sample_count < 1 or sample_count > total_count:
        raise ValueError("Requested mechanism count is outside the saved v2 sweep")
    if not np.array_equal(saved["mechanism_id"].to_numpy(), np.arange(total_count)):
        raise ValueError("Saved v2 mechanism IDs are not contiguous from zero")

    pseudo_document = {
        "uncertainty": {path: {"range": bounds} for path, bounds in EQUIVALENT_SPACE.items()}
    }
    all_samples = sample_lhs_log10(pseudo_document, total_count, seed, paths=MECHANISM_PATHS)
    samples = all_samples[:sample_count]
    saved = saved.iloc[:sample_count].reset_index(drop=True)
    if not np.allclose(
        samples, saved[list(MECHANISM_PATHS)].to_numpy(dtype=float), rtol=1e-12, atol=0.0
    ):
        raise ValueError("Saved v2 LHS values do not match regeneration from the recorded seed")
    mechanism = pd.DataFrame(samples, columns=MECHANISM_PATHS)
    mechanism.insert(0, "mechanism_id", np.arange(sample_count))
    return mechanism


def run_near_surface_candidate(base_values, candidate, doses, cells, max_step):
    values = load_parameters(ROOT / "parameters" / "theoretical_prior.json")["values"] if base_values is None else base_values
    from cranial_model.ensemble import apply_candidate
    values = apply_candidate(values, MECHANISM_PATHS, candidate)
    rows = []
    for dose in doses:
        scenario = simulate_scenario(
            values, dose, cells_per_layer=cells, max_step_s=max_step, stage_names=STAGES
        )
        rows.extend(condition_near_surface_states(scenario, dose))
    return rows


def generate_predictions(args, mechanism: pd.DataFrame) -> pd.DataFrame:
    document = load_parameters(args.parameters)
    base_values = document["values"]
    doses = sorted(float(d) for d in base_values["scenarios"]["edta_concentrations_mol_per_l"])
    rows, failures = [], []
    context = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else None
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                run_near_surface_candidate, base_values,
                row[list(MECHANISM_PATHS)].to_numpy(dtype=float), doses, args.cells, args.max_step,
            ): int(row["mechanism_id"])
            for _, row in mechanism.iterrows()
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            mechanism_id = futures[future]
            try:
                result = future.result()
            except Exception as error:  # noqa: BLE001
                failures.append({"mechanism_id": mechanism_id, "error": str(error)})
                continue
            for row in result:
                rows.append({"mechanism_id": mechanism_id, **row})
            if completed % 25 == 0 or completed == len(futures):
                pd.DataFrame(rows).to_csv(args.output / "near_surface_condition_predictions.partial.csv", index=False)
                print(f"near-surface sweep {completed}/{len(futures)}", flush=True)
    predictions = pd.DataFrame(rows)
    predictions.to_csv(args.output / "near_surface_condition_predictions.csv", index=False)
    pd.DataFrame(failures).to_csv(args.output / "near_surface_failures.csv", index=False)
    if failures:
        raise RuntimeError(f"{len(failures)} mechanism candidates failed")
    return predictions


def verify_fixed_mechanism(predictions: pd.DataFrame, v2_results: Path) -> dict:
    legacy = pd.read_csv(v2_results / "equivalent_condition_predictions.csv").rename(
        columns={"candidate_id": "mechanism_id"}
    )
    legacy = legacy[legacy["mechanism_id"].isin(predictions["mechanism_id"].unique())]
    keys = ["mechanism_id", *CONDITION_KEYS]
    merged = predictions.merge(legacy, on=keys, how="inner", suffixes=("_v4", "_v2"), validate="one_to_one")
    if len(merged) != len(predictions) or len(merged) != len(legacy):
        raise ValueError("v4/v2 fixed-mechanism rows do not align exactly")
    differences = {
        "sim_whole_resistance_ratio": np.max(np.abs(
            merged["sim_whole_resistance_ratio_v4"] - merged["sim_whole_resistance_ratio_v2"]
        )),
        "sim_outer_porosity_ratio_100um": np.max(np.abs(
            merged["sim_outer_porosity_ratio_100um"] - merged["sim_outer_surface_porosity_ratio"]
        )),
    }
    if any(value > 1e-10 for value in differences.values()):
        raise ValueError(f"v4 changed the fixed mechanism output: {differences}")
    return {key: float(value) for key, value in differences.items()}


def build_candidates(mechanism_ids: np.ndarray, draws: int, seed: int, dynamic: bool) -> pd.DataFrame:
    frames = []
    for mechanism_id in mechanism_ids:
        sampled = sample_near_surface_lhs(draws, seed + 1 + int(mechanism_id), dynamic=dynamic)
        sampled.insert(0, "mechanism_id", int(mechanism_id))
        frames.append(sampled)
    result = pd.concat(frames, ignore_index=True)
    result.insert(0, "candidate_id", np.arange(len(result)))
    return result


def run_analysis(output: Path, targets: pd.DataFrame, predictions: pd.DataFrame,
                 candidates: pd.DataFrame, dynamic: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    tolerances = frequency_tolerances(targets)
    variants = {}
    for structure in STRUCTURES:
        scores = score_near_surface_candidates(
            predictions, candidates, targets, tolerances, structure, dynamic
        )
        metrics = scores.metrics
        metrics.to_csv(output / f"{structure}_candidate_metrics.csv", index=False)
        compatible = metrics[metrics["compatible"]]
        compatible.to_csv(output / f"{structure}_compatible_parameters.csv", index=False)
        rat, points, frequency, overall = leave_one_rat_out(
            predictions, candidates, targets, scores, frequency_tolerances, structure, dynamic
        )
        rat.to_csv(output / f"{structure}_loo_rat_summary.csv", index=False)
        points.to_csv(output / f"{structure}_loo_point_intervals.csv", index=False)
        frequency.to_csv(output / f"{structure}_loo_frequency_summary.csv", index=False)
        if len(compatible):
            interval_ids = compatible["candidate_id"].to_numpy()
            label = "COMPATIBLE_SET_EXPLORATORY_PLATFORM_BOUND"
        else:
            count = max(1, int(round(0.05 * len(metrics))))
            interval_ids = metrics.sort_values("normalized_rmse")["candidate_id"].iloc[:count].to_numpy()
            label = "EXPLORATORY_NEAREST_APPROACH_NOT_A_COMPATIBLE_SET"
        intervals = prediction_quantiles(
            predictions, candidates, interval_ids, targets, structure, dynamic
        )
        intervals["interval_set"] = label
        intervals.to_csv(output / f"{structure}_full_data_prediction_intervals.csv", index=False)
        drift, stage = residual_drift_by_frequency(targets, intervals)
        drift.to_csv(output / f"{structure}_residual_drift.csv", index=False)
        stage.to_csv(output / f"{structure}_stage_residuals.csv", index=False)
        best = metrics.loc[metrics["normalized_rmse"].idxmin()]
        best_raw = {}
        for f in FREQUENCIES_HZ:
            column = f"raw_rmse_{f:g}hz"
            best_raw[f"{f:g}"] = float(metrics[column].min())
        variants[structure] = {
            "candidate_count": int(len(metrics)),
            "compatible_count": int(len(compatible)),
            "best_candidate_id": int(best["candidate_id"]),
            "best_mechanism_id": int(best["mechanism_id"]),
            "best_normalized_rmse": float(best["normalized_rmse"]),
            "best_raw_rmse_by_frequency": {
                f"{f:g}": float(best[f"raw_rmse_{f:g}hz"]) for f in FREQUENCIES_HZ
            },
            "best_raw_rmse_full_space_by_frequency": best_raw,
            "pass_count_by_frequency": {
                f"{f:g}": int(metrics[f"pass_{f:g}hz"].sum()) for f in FREQUENCIES_HZ
            },
            "loo": {"overall": _json_safe(overall), "by_frequency": _json_safe(frequency.to_dict("records"))},
            "residual_drift": _json_safe(drift.to_dict("records")),
        }
        print(json.dumps({"structure": structure, **variants[structure]}, ensure_ascii=False), flush=True)
    return {"tolerances": tolerances, "variants": variants}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--parameters", type=Path, default=ROOT / "parameters" / "theoretical_prior.json")
    parser.add_argument("--v2-results", type=Path, default=ROOT / "results" / "equivalent_calibration_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "near_surface_calibration_v4a")
    parser.add_argument("--sensitivity-output", type=Path, default=ROOT / "results" / "near_surface_calibration_v4a_all24_sensitivity")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--observation-draws", type=int, default=32)
    parser.add_argument("--lhs-seed", type=int, default=20261115)
    parser.add_argument("--v2-lhs-seed", type=int, default=20261113)
    parser.add_argument("--cells", nargs=3, type=int, default=[12, 6, 6])
    parser.add_argument("--max-step", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--dynamic", action="store_true", help="Enable v4b shared exposure-dependent logit weight.")
    parser.add_argument("--reuse-sweep", action="store_true")
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    mechanism = reconstruct_v2_mechanisms(args.samples, args.v2_results, args.v2_lhs_seed)
    mechanism.to_csv(
        args.output / "near_surface_mechanism_lhs.csv", index=False, float_format="%.17g"
    )
    prediction_path = args.output / "near_surface_condition_predictions.csv"
    if args.reuse_sweep:
        predictions = pd.read_csv(prediction_path)
        manifest_path = args.output / "near_surface_sweep_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["source_sha256"]["predictions"] != sha256(prediction_path):
            raise ValueError("Reuse manifest does not match the prediction CSV")
    else:
        predictions = generate_predictions(args, mechanism)
    fixed_difference = verify_fixed_mechanism(predictions, args.v2_results)
    candidates = build_candidates(
        mechanism["mechanism_id"].to_numpy(), args.observation_draws, args.lhs_seed, args.dynamic
    )
    candidates.to_csv(args.output / "near_surface_observation_lhs.csv", index=False)
    manifest = {
        "model_version": "v4b_exposure_dynamic" if args.dynamic else "v4a_static_near_surface",
        "structures": list(STRUCTURES), "mechanism_count": len(mechanism),
        "prediction_rows": len(predictions), "cells_per_layer": args.cells,
        "max_step_s": args.max_step, "mechanism_output_points_per_stage": 31,
        "exposure_quadrature": "post-solve trapezoid on the fixed 31-point mechanism grid",
        "fixed_mechanism_max_absolute_difference": fixed_difference,
        "observation_draws_per_mechanism": args.observation_draws,
        "lhs_seed": args.lhs_seed, "v2_lhs_seed": args.v2_lhs_seed, "dynamic": args.dynamic,
        "exposure_logit_slope_range": list(EXPOSURE_LOGIT_SLOPE_RANGE) if args.dynamic else None,
        "source_sha256": {
            "data": sha256(args.data), "parameters": sha256(args.parameters),
            "predictions": sha256(prediction_path),
            "v2_predictions": sha256(args.v2_results / "equivalent_condition_predictions.csv"),
            "v2_lhs": sha256(args.v2_results / "equivalent_lhs_samples.csv"),
            "core": sha256(ROOT / "cranial_model" / "core.py"),
            "near_surface": sha256(ROOT / "cranial_model" / "near_surface.py"),
            "calibration": sha256(ROOT / "cranial_model" / "near_surface_calibration.py"),
            "runner": sha256(Path(__file__).resolve()),
        },
    }
    (args.output / "near_surface_sweep_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    spectra = read_experimental_spectra(args.data)
    primary_targets = build_multifrequency_targets(spectra)
    primary_targets.to_csv(args.output / "near_surface_z_ratio_targets.csv", index=False)
    primary = run_analysis(args.output, primary_targets, predictions, candidates, args.dynamic)
    result = {**manifest, "analysis_role": "PRIMARY_CONFOUNDER_FILTERED_22_RATS", **primary}
    (args.output / "near_surface_calibration_metrics.json").write_text(
        json.dumps(_json_safe(result), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if not args.skip_sensitivity:
        args.sensitivity_output.mkdir(parents=True, exist_ok=True)
        sensitivity_targets = build_multifrequency_targets(spectra, drop_confounded=False)
        sensitivity_targets.to_csv(args.sensitivity_output / "near_surface_z_ratio_targets.csv", index=False)
        sensitivity = run_analysis(
            args.sensitivity_output, sensitivity_targets, predictions, candidates, args.dynamic
        )
        sensitivity_result = {
            **manifest, "analysis_role": "SENSITIVITY_ALL_24_RATS_ALL_FREEZE_RECORDS_NOT_PRIMARY",
            **sensitivity,
        }
        args.sensitivity_output.mkdir(parents=True, exist_ok=True)
        (args.sensitivity_output / "near_surface_calibration_metrics.json").write_text(
            json.dumps(_json_safe(sensitivity_result), ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
