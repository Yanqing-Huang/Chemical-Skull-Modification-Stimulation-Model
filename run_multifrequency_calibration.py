#!/usr/bin/env python3
"""Track B v3: reuse v2 trajectories for four-frequency magnitude calibration.

Runs the 22-rat primary analysis and the 24-rat freeze-thaw sensitivity analysis.
No PDE is solved and no reaction/boundary parameter is changed by frequency.
"""

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

from cranial_model.equivalent_calibration import CONDITION_KEYS, PLATFORM_DISCLAIMER
from cranial_model.experimental import read_experimental_spectra
from cranial_model.multifrequency_calibration import (
    BETA_INTERPOLATION,
    BETA_LOW_RANGE,
    FREQUENCIES_HZ,
    OBSERVATION_OPERATOR,
    build_multifrequency_targets,
    condition_sd_by_frequency,
    frequency_tolerances,
    leave_one_rat_out,
    matrix_only_parameters,
    prediction_quantiles,
    residual_drift_by_frequency,
    sample_observation_lhs,
    score_multifrequency_candidates,
)
from cranial_model.equivalent_calibration import INTERFACE_EXPONENT_RANGE
from cranial_model.plotting import DOSE_COLORS, configure_style
from run_equivalent_calibration import DEFAULT_DATA
from run_prior_mechanism import MECHANISM_PATHS


ROOT = Path(__file__).resolve().parent


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


def _records(frame: pd.DataFrame) -> list[dict]:
    return _json_safe(frame.to_dict(orient="records"))


def build_observation_candidates(
    v2_results: Path, samples: int, draws: int, seed: int
) -> pd.DataFrame:
    """Expand each saved v2 mechanism with an independent three-parameter LHS."""
    v2_lhs = pd.read_csv(v2_results / "equivalent_lhs_samples.csv")
    mechanism = v2_lhs[["mechanism_id", *MECHANISM_PATHS]].drop_duplicates("mechanism_id")
    mechanism = mechanism.sort_values("mechanism_id").reset_index(drop=True)
    if len(mechanism) != samples:
        raise ValueError(f"Expected {samples} saved mechanisms, found {len(mechanism)}")
    mechanism_ids = np.repeat(mechanism["mechanism_id"].to_numpy(), draws)
    frame = pd.DataFrame(
        mechanism.iloc[np.repeat(np.arange(samples), draws)].reset_index(drop=True)
    )
    frame.insert(0, "candidate_id", np.arange(samples * draws))
    samples_array = np.vstack([
        sample_observation_lhs(draws, seed + 1 + int(mechanism_id))
        for mechanism_id in mechanism["mechanism_id"]
    ])
    if not np.array_equal(mechanism_ids, frame["mechanism_id"].to_numpy()):
        raise RuntimeError("Mechanism expansion is not aligned")
    frame["beta_low"] = samples_array[:, 0]
    frame["beta_ratio"] = samples_array[:, 1]
    frame["interface_exponent"] = samples_array[:, 2]
    return frame


def _best_candidate(metrics: pd.DataFrame) -> dict:
    row = metrics.loc[metrics["normalized_rmse"].idxmin()]
    return {
        "candidate_id": int(row["candidate_id"]),
        "mechanism_id": int(row["mechanism_id"]),
        "normalized_rmse": float(row["normalized_rmse"]),
        "raw_rmse_by_frequency": {
            f"{frequency:g}": float(row[f"raw_rmse_{frequency:g}hz"])
            for frequency in FREQUENCIES_HZ
        },
        "passes_every_frequency": bool(row["compatible"]),
    }


def _best_frequency_metrics(metrics: pd.DataFrame) -> dict:
    result = {}
    for frequency in FREQUENCIES_HZ:
        column = f"raw_rmse_{frequency:g}hz"
        row = metrics.loc[metrics[column].idxmin()]
        result[f"{frequency:g}"] = {
            "candidate_id": int(row["candidate_id"]),
            "raw_rmse": float(row[column]),
        }
    return result


def _loo_metrics(summary: dict, frequency: pd.DataFrame) -> dict:
    return {"overall": _json_safe(summary), "by_frequency": _records(frequency)}


def run_analysis(
    *,
    output: Path,
    spectra: pd.DataFrame,
    predictions: pd.DataFrame,
    candidates: pd.DataFrame,
    data_path: Path,
    v2_results: Path,
    parameters_path: Path,
    retained_confounded_rats: tuple[str, ...],
    restore_all_confounded: bool,
    analysis_role: str,
    seed: int,
    draws: int,
    chunk_size: int,
) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    unfiltered = build_multifrequency_targets(spectra, drop_confounded=False)
    targets = (
        unfiltered.copy() if restore_all_confounded else build_multifrequency_targets(
            spectra, retain_confounded_rats=retained_confounded_rats
        )
    )
    targets.to_csv(output / "multifrequency_z_ratio_targets.csv", index=False)
    condition_sd = condition_sd_by_frequency(targets)
    condition_sd.to_csv(output / "condition_sd_by_frequency.csv", index=False)
    tolerances = frequency_tolerances(targets)

    scores = score_multifrequency_candidates(
        predictions, candidates, targets, tolerances, chunk_size=chunk_size
    )
    scores.metrics.to_csv(output / "multifrequency_candidate_metrics.csv", index=False)
    compatible_metrics = scores.metrics[scores.metrics["compatible"]].copy()
    compatible_metrics.to_csv(output / "compatible_multifrequency_parameters.csv", index=False)

    mechanism_ids = np.array(sorted(predictions["mechanism_id"].unique()))
    matrix_parameters = matrix_only_parameters(mechanism_ids)
    matrix_scores = score_multifrequency_candidates(
        predictions, matrix_parameters, targets, tolerances,
        chunk_size=chunk_size, matrix_only=True,
    )
    matrix_scores.metrics.to_csv(output / "matrix_only_candidate_metrics.csv", index=False)

    rat_summary, points, loo_frequency, loo_overall = leave_one_rat_out(
        predictions, candidates, targets, scores=scores
    )
    rat_summary.to_csv(output / "loo_rat_summary.csv", index=False)
    points.to_csv(output / "loo_point_intervals.csv", index=False)
    loo_frequency.to_csv(output / "loo_frequency_summary.csv", index=False)
    matrix_rat, matrix_points, matrix_frequency, matrix_overall = leave_one_rat_out(
        predictions, matrix_parameters, targets,
        scores=matrix_scores, matrix_only=True,
    )
    matrix_rat.to_csv(output / "matrix_only_loo_rat_summary.csv", index=False)
    matrix_points.to_csv(output / "matrix_only_loo_point_intervals.csv", index=False)
    matrix_frequency.to_csv(output / "matrix_only_loo_frequency_summary.csv", index=False)

    compatible_ids = scores.candidate_ids[scores.metrics["compatible"].to_numpy()]
    exploratory = None
    if len(compatible_ids):
        interval_ids = compatible_ids
        interval_label = "COMPATIBLE_SET_EXPLORATORY_PLATFORM_BOUND"
    else:
        count = max(1, int(round(0.05 * len(scores.metrics))))
        interval_ids = scores.metrics.sort_values("normalized_rmse")["candidate_id"].iloc[:count].to_numpy()
        interval_label = "EXPLORATORY_NEAREST_APPROACH_NOT_A_COMPATIBLE_SET"
        exploratory = {
            "label": interval_label,
            "candidate_count": int(count),
            "selection_rule": "Lowest 5% total normalized RMSE; diagnostic only",
            "interpretation": (
                "No candidate satisfies the preregistered joint rule. The nearest-approach "
                "envelope and residual diagnostics are not validation evidence."
            ),
        }
    intervals = prediction_quantiles(
        predictions, candidates, interval_ids, targets
    )
    intervals["interval_set"] = interval_label
    intervals.to_csv(output / "full_data_prediction_intervals.csv", index=False)
    drift, stage_residual = residual_drift_by_frequency(targets, intervals)
    drift.to_csv(output / "residual_drift_by_frequency.csv", index=False)
    stage_residual.to_csv(output / "stage_residuals_by_frequency.csv", index=False)

    condition_counts = targets.groupby(["frequency_hz", *CONDITION_KEYS])["rat"].nunique()
    single_rat_conditions = [
        {
            "frequency_hz": float(key[0]),
            "stage": key[1],
            "edta_concentration_mol_per_l": float(key[2]),
            "time_s": float(key[3]),
        }
        for key, count in condition_counts.items() if count == 1
    ]
    best = _best_candidate(scores.metrics)
    if exploratory is not None:
        exploratory["best_candidate"] = best
        exploratory["point_coverage_of_targets"] = float(np.mean(
            (intervals["z_ratio"] >= intervals["sim_q05"])
            & (intervals["z_ratio"] <= intervals["sim_q95"])
        ))
    metrics = {
        "status": "TRACK_B_V3_FOUR_FREQUENCY_EQUIVALENT_CALIBRATION_PLATFORM_BOUND",
        "model_version": "v3_four_frequency_magnitude_observation_layer",
        "analysis_role": analysis_role,
        "platform_disclaimer": PLATFORM_DISCLAIMER,
        "frequencies_hz": list(FREQUENCIES_HZ),
        "observation_operator": OBSERVATION_OPERATOR,
        "beta_interpolation": BETA_INTERPOLATION,
        "beta_low_range": list(BETA_LOW_RANGE),
        "beta_ratio_open_range": [0.0, 1.0],
        "interface_exponent_range": list(INTERFACE_EXPONENT_RANGE),
        "mechanism_reuse": (
            "The same 200 v2 trajectories and mechanism parameters are reused; no PDE "
            "rerun and no frequency-dependent decalcification kinetics."
        ),
        "lhs": {
            "seed": int(seed),
            "per_mechanism_seed_rule": "seed + 1 + mechanism_id",
            "draws_per_mechanism": int(draws),
            "mechanism_count": int(len(mechanism_ids)),
            "candidate_count": int(len(candidates)),
            "dimensions": ["beta_low:linear", "beta_ratio:linear", "interface_exponent:log10"],
        },
        "target": (
            "Z_post/Z_baseline magnitude ratio independently at 100/120/1000/10000 Hz; "
            "three sites averaged within each rat/condition/frequency, never across frequency"
        ),
        "retained_confounded_rats": list(retained_confounded_rats),
        "all_confounded_records_restored": bool(restore_all_confounded),
        "data_audit": {
            "source_frequency_rows": int(len(spectra)),
            "source_spectrum_count": int(spectra["spectrum_id"].nunique()),
            "source_rat_count": int(spectra["rat"].nunique()),
            "unfiltered_target_count": int(len(unfiltered)),
            "excluded_confounded_target_count": int(len(unfiltered) - len(targets)),
            "included_target_count": int(len(targets)),
            "included_rat_count": int(targets["rat"].nunique()),
            "target_count_by_frequency": {
                f"{frequency:g}": int((targets["frequency_hz"] == frequency).sum())
                for frequency in FREQUENCIES_HZ
            },
            "all_targets_have_three_sites_and_three_values": bool(
                (targets["site_count"] == 3).all() and (targets["n_values"] == 3).all()
            ),
            "duplicate_target_key_count": int(targets.duplicated(
                ["rat", "frequency_hz", *CONDITION_KEYS]
            ).sum()),
            "single_rat_conditions": single_rat_conditions,
        },
        "tolerance_rule": (
            "Per-frequency RMS of between-rat condition SD; acetic-acid SD pools future "
            "EDTA doses only for noise estimation; single-rat conditions do not contribute SD"
        ),
        "tolerances": {f"{frequency:g}": tolerances[frequency] for frequency in FREQUENCIES_HZ},
        "compatibility_rule": (
            "Total normalized RMSE<=1 AND raw RMSE<=the corresponding tolerance at every frequency"
        ),
        "matrix_only_baseline": {
            "candidate_count": int(len(matrix_scores.metrics)),
            "compatible_count": int(matrix_scores.metrics["compatible"].sum()),
            "best_candidate": _best_candidate(matrix_scores.metrics),
            "best_raw_rmse_by_frequency": _best_frequency_metrics(matrix_scores.metrics),
            "loo": _loo_metrics(matrix_overall, matrix_frequency),
        },
        "multifrequency_screening": {
            "candidate_count": int(len(scores.metrics)),
            "compatible_count": int(len(compatible_ids)),
            "compatible_fraction": float(len(compatible_ids) / len(scores.metrics)),
            "best_candidate": best,
            "best_raw_rmse_by_frequency": _best_frequency_metrics(scores.metrics),
            "pass_count_by_frequency": {
                f"{frequency:g}": int(scores.metrics[f"pass_{frequency:g}hz"].sum())
                for frequency in FREQUENCIES_HZ
            },
        },
        "loo": _loo_metrics(loo_overall, loo_frequency),
        "residual_drift_by_frequency": _records(drift),
        "stage_residuals_by_frequency": _records(stage_residual),
        "exploratory_nearest_approach": exploratory,
        "source_sha256": {
            "data": sha256(data_path),
            "experimental_parser": sha256(ROOT / "cranial_model" / "experimental.py"),
            "module": sha256(ROOT / "cranial_model" / "multifrequency_calibration.py"),
            "runner": sha256(Path(__file__).resolve()),
            "v2_module": sha256(ROOT / "cranial_model" / "equivalent_calibration.py"),
            "v2_condition_predictions": sha256(v2_results / "equivalent_condition_predictions.csv"),
            "v2_lhs_samples": sha256(v2_results / "equivalent_lhs_samples.csv"),
            "parameters": sha256(parameters_path),
            "core": sha256(ROOT / "cranial_model" / "core.py"),
        },
    }
    (output / "multifrequency_calibration_metrics.json").write_text(
        json.dumps(_json_safe(metrics), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    _plot_prediction_bands(intervals, targets, output, exploratory is not None)
    _plot_residuals(targets, intervals, drift, output, exploratory is not None)
    return metrics


def _stage_time_min(frame: pd.DataFrame) -> pd.Series:
    return frame["time_s"] / 60.0 + np.where(frame["stage"] == "edta", 5.0, 0.0)


def _plot_prediction_bands(
    intervals: pd.DataFrame, targets: pd.DataFrame, output: Path, exploratory: bool
) -> None:
    configure_style()
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.4), sharex=True, sharey=True)
    for axis, frequency in zip(axes.flat, FREQUENCIES_HZ):
        predicted = intervals[intervals["frequency_hz"] == frequency].copy()
        observed = targets[targets["frequency_hz"] == frequency].copy()
        predicted["time_min"] = _stage_time_min(predicted)
        observed["time_min"] = _stage_time_min(observed)
        for dose, color in zip(sorted(targets["edta_concentration_mol_per_l"].unique()), DOSE_COLORS):
            pred_dose = predicted[predicted["edta_concentration_mol_per_l"] == dose]
            obs_dose = observed[observed["edta_concentration_mol_per_l"] == dose]
            band = pred_dose.groupby("time_min").agg(
                q05=("sim_q05", "median"), q50=("sim_q50", "median"), q95=("sim_q95", "median")
            ).sort_index()
            points = obs_dose.groupby("time_min")["z_ratio"].mean().sort_index()
            axis.fill_between(band.index, band["q05"], band["q95"], color=color, alpha=0.12, lw=0)
            axis.plot(band.index, band["q50"], "--", color=color, lw=1.0)
            axis.plot(points.index, points, "o", color=color, ms=2.8, label=f"{dose:.2f} mol/L")
        axis.set_title(f"{frequency:g} Hz", fontsize=10)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Process time (min)")
        axis.set_ylabel("Z post / Z baseline")
    axes.flat[0].legend(fontsize=7, ncol=2)
    label = "nearest approach (not compatible)" if exploratory else "compatible set"
    figure.suptitle(f"Track B v3 four-frequency {label}: lines=model, points=observed condition mean", fontsize=10)
    figure.text(0.99, 0.01, PLATFORM_DISCLAIMER, ha="right", fontsize=6, color="0.35")
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(output / "FigB3_multifrequency_prediction_bands.png", facecolor="white")
    plt.close(figure)


def _plot_residuals(
    targets: pd.DataFrame, intervals: pd.DataFrame, drift: pd.DataFrame,
    output: Path, exploratory: bool,
) -> None:
    configure_style()
    keys = ["rat", "frequency_hz", *CONDITION_KEYS]
    merged = targets.merge(intervals[[*keys, "sim_q50"]], on=keys, validate="one_to_one")
    merged["residual"] = merged["z_ratio"] - merged["sim_q50"]
    merged["time_min"] = _stage_time_min(merged)
    figure, axes = plt.subplots(2, 2, figsize=(9.0, 6.2), sharex=True)
    for axis, frequency in zip(axes.flat, FREQUENCIES_HZ):
        subset = merged[merged["frequency_hz"] == frequency]
        for stage, color, marker in (("acetic_acid", "#762A83", "s"), ("edta", "#2166AC", "o")):
            group = subset[subset["stage"] == stage]
            axis.scatter(group["time_min"], group["residual"], s=9, alpha=0.5,
                         color=color, marker=marker, label=stage.replace("_", " "))
        row = drift[drift["frequency_hz"] == frequency].iloc[0]
        axis.axhline(0.0, color="0.4", lw=0.8)
        axis.set_title(
            f"{frequency:g} Hz: EDTA rho={row['residual_vs_edta_time_spearman_rho']:.2f}, "
            f"p={row['residual_vs_edta_time_spearman_p']:.2g}", fontsize=9
        )
        axis.set_xlabel("Process time (min)")
        axis.set_ylabel("Observed - predicted median")
        axis.grid(alpha=0.2)
    axes.flat[0].legend(fontsize=7)
    label = "exploratory nearest approach" if exploratory else "compatible set"
    figure.suptitle(f"Track B v3 residual drift, {label}", fontsize=10)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(output / "FigB4_multifrequency_residual_drift.png", facecolor="white")
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--parameters", type=Path, default=ROOT / "parameters" / "theoretical_prior.json")
    parser.add_argument("--v2-results", type=Path, default=ROOT / "results" / "equivalent_calibration_v2")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "multifrequency_calibration_v3")
    parser.add_argument(
        "--sensitivity-output", type=Path,
        default=ROOT / "results" / "multifrequency_calibration_v3_all24_sensitivity",
    )
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--observation-draws", type=int, default=32)
    parser.add_argument("--lhs-seed", type=int, default=20261114)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--skip-sensitivity", action="store_true")
    args = parser.parse_args()

    spectra = read_experimental_spectra(args.data)
    predictions = pd.read_csv(args.v2_results / "equivalent_condition_predictions.csv")
    predictions = predictions.rename(columns={"candidate_id": "mechanism_id"})
    candidates = build_observation_candidates(
        args.v2_results, args.samples, args.observation_draws, args.lhs_seed
    )
    common = dict(
        spectra=spectra,
        predictions=predictions,
        candidates=candidates,
        data_path=args.data,
        v2_results=args.v2_results,
        parameters_path=args.parameters,
        seed=args.lhs_seed,
        draws=args.observation_draws,
        chunk_size=args.chunk_size,
    )
    primary = run_analysis(
        output=args.output,
        retained_confounded_rats=(),
        restore_all_confounded=False,
        analysis_role="PRIMARY_CONFOUNDER_FILTERED_22_RATS",
        **common,
    )
    print(json.dumps({
        "analysis": "primary",
        "targets": primary["data_audit"]["included_target_count"],
        "tolerances": primary["tolerances"],
        "matrix_only": primary["matrix_only_baseline"]["best_candidate"],
        "v3": primary["multifrequency_screening"],
        "loo": primary["loo"]["overall"],
    }, ensure_ascii=False, indent=2))
    if not args.skip_sensitivity:
        sensitivity = run_analysis(
            output=args.sensitivity_output,
            retained_confounded_rats=(),
            restore_all_confounded=True,
            analysis_role="SENSITIVITY_ALL_24_RATS_ALL_FREEZE_RECORDS_NOT_PRIMARY",
            **common,
        )
        print(json.dumps({
            "analysis": "all24_sensitivity",
            "targets": sensitivity["data_audit"]["included_target_count"],
            "tolerances": sensitivity["tolerances"],
            "matrix_only": sensitivity["matrix_only_baseline"]["best_candidate"],
            "v3": sensitivity["multifrequency_screening"],
            "loo": sensitivity["loo"]["overall"],
        }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
