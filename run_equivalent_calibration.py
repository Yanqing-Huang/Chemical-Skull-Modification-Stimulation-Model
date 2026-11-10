#!/usr/bin/env python3
"""Track B v2: dual-branch equivalent calibration + leave-one-rat-out validation.

The mechanism model supplies both whole-skull matrix resistance and outer-
surface porosity. A series interface branch converts them to measured Z ratios
using two additional platform parameters (baseline fraction beta, exponent m).
The matrix-only v1 result remains in results/equivalent_calibration/.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cranial_model.core import load_parameters
from cranial_model.ensemble import sample_lhs_log10
from cranial_model.equivalent_calibration import (
    INTERFACE_EXPONENT_RANGE,
    INTERFACE_FRACTION_RANGE,
    PLATFORM_DISCLAIMER,
    apply_dual_branch_forward_operator,
    build_z_ratio_targets,
    compatible_candidates,
    condition_tolerance,
    interface_contribution_summary,
    leave_one_rat_out,
    prediction_quantiles,
    residual_drift,
    sample_interface_lhs,
)
from cranial_model.experimental import read_experimental_spectra
from cranial_model.plotting import DOSE_COLORS, configure_style
from run_prior_mechanism import MECHANISM_PATHS, run_candidate


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/workspace/.uploads/a32f8259-e715-40b5-a800-8c39875ea94b_体外EDTA数据记录汇总 (1).xlsx")

# Track B sampling space: COMPOSITE EQUIVALENT ranges, deliberately wider than
# the literature priors. These are curve-family degrees of freedom that bundle
# interface/electrode effects; they are NOT intrinsic hydroxyapatite kinetics.
EQUIVALENT_SPACE = {
    "stages.acetic_acid.k_r_m3_per_mol_s.outer_table": (1e-9, 1e-3),
    "stages.acetic_acid.k_r_m3_per_mol_s.diploe": (5e-9, 3e-3),
    "stages.acetic_acid.k_r_m3_per_mol_s.inner_table": (1e-9, 1e-3),
    "stages.edta.k_r_m3_per_mol_s.outer_table": (1e-8, 1e-2),
    "stages.edta.k_r_m3_per_mol_s.diploe": (5e-8, 3e-2),
    "stages.edta.k_r_m3_per_mol_s.inner_table": (1e-8, 1e-2),
    "stages.edta.surface_site_capacity_mol_per_m3.outer_table": (50.0, 20000.0),
    "stages.edta.surface_site_capacity_mol_per_m3.diploe": (50.0, 12000.0),
    "stages.edta.surface_site_capacity_mol_per_m3.inner_table": (50.0, 20000.0),
    "transport.external_mass_transfer_m_per_s": (1e-6, 1e-3),
    "transport.porosity_per_released_calcium_m3_per_mol": (1e-5, 3e-4),
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--parameters", type=Path, default=ROOT / "parameters" / "theoretical_prior.json")
    parser.add_argument("--prior-results", type=Path, default=ROOT / "results" / "prior_mechanism")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "equivalent_calibration_v2")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--lhs-seed", type=int, default=20261113)
    parser.add_argument(
        "--interface-draws", type=int, default=16,
        help="Conditional beta/m LHS draws per saved mechanism trajectory.",
    )
    parser.add_argument("--cells", nargs=3, type=int, default=[12, 6, 6])
    parser.add_argument("--max-step", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--reuse-sweep",
        action="store_true",
        help="Skip forward simulations; reuse equivalent_condition_predictions.csv.",
    )
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    spectra = read_experimental_spectra(args.data)
    unfiltered_targets = build_z_ratio_targets(spectra, drop_confounded=False)
    targets = build_z_ratio_targets(spectra)
    targets.to_csv(args.output / "z_ratio_targets.csv", index=False)
    tolerance = condition_tolerance(targets)
    condition_keys = ["stage", "edta_concentration_mol_per_l", "time_s"]
    condition_rat_counts = targets.groupby(condition_keys)["rat"].nunique()
    single_rat_conditions = [
        {"stage": stage, "edta_concentration_mol_per_l": float(dose), "time_s": float(time_s)}
        for (stage, dose, time_s), count in condition_rat_counts.items()
        if count == 1
    ]

    # --- Part 1: literature-prior candidates screened against data ----------
    prior_predictions = pd.read_csv(args.prior_results / "candidate_condition_predictions.csv")
    prior_compatible, prior_all, prior_rmse = compatible_candidates(
        prior_predictions, targets, tolerance
    )
    prior_screening = {
        "candidates_total": int(len(prior_all)),
        "compatible_count": int(len(prior_compatible)),
        "best_rmse": float(np.min(prior_rmse)),
        "tolerance": tolerance,
        "interpretation": (
            "No tested literature-prior matrix trajectory meets the measured "
            "Z-ratio compatibility rule. Track A remains trend-level only; this "
            "mismatch motivates testing non-matrix observation contributions but "
            "does not identify their causal share."
            if len(prior_compatible) == 0 else
            "Some literature-prior matrix trajectories meet the measured Z-ratio "
            "compatibility rule. Track A remains uncalibrated; compatibility alone "
            "is not independent validation and does not identify intrinsic kinetics."
        ),
    }

    # --- Part 2: equivalent-space sweep (composite platform parameters) -----
    document = load_parameters(args.parameters)
    base_values = document["values"]
    doses = sorted(float(d) for d in base_values["scenarios"]["edta_concentrations_mol_per_l"])
    pseudo_document = {"uncertainty": {p: {"range": r} for p, r in EQUIVALENT_SPACE.items()}}
    candidates = sample_lhs_log10(
        pseudo_document, args.samples, args.lhs_seed, paths=MECHANISM_PATHS
    )
    if args.reuse_sweep:
        predictions = pd.read_csv(args.output / "equivalent_condition_predictions.csv")
        failures = 0
    else:
        condition_rows: list[dict] = []
        failures = 0
        completed = 0
        context = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else None
        with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
            futures = {
                executor.submit(
                    run_candidate, base_values, candidate, doses,
                    args.cells, args.max_step, MECHANISM_PATHS,
                ): index
                for index, candidate in enumerate(candidates)
            }
            for future in as_completed(futures):
                index = futures[future]
                completed += 1
                try:
                    per_dose = future.result()
                except Exception as error:  # noqa: BLE001 - record and continue ensemble
                    failures += 1
                    print(f"equivalent candidate {index} failed: {error}", flush=True)
                    continue
                for dose in doses:
                    for key, value in per_dose[dose].items():
                        if isinstance(value, float) and key.startswith("cond__"):
                            _, stage, time_s = key.split("__")
                            condition_rows.append(
                                {"candidate_id": index, "stage": stage, "time_s": float(time_s),
                                 "edta_concentration_mol_per_l": dose,
                                 "sim_whole_resistance_ratio": value,
                                 "sim_outer_surface_porosity_ratio": per_dose[dose][
                                     key.replace("cond__", "cond_surface__", 1)
                                 ]}
                            )
                if completed % 25 == 0:
                    pd.DataFrame(condition_rows).to_csv(
                        args.output / "equivalent_condition_predictions.partial.csv", index=False
                    )
                    print(f"equivalent sweep {completed}/{args.samples}", flush=True)
        predictions = pd.DataFrame(condition_rows)
        predictions.to_csv(args.output / "equivalent_condition_predictions.csv", index=False)
    composite_count = args.samples * args.interface_draws
    mechanism_ids = np.repeat(np.arange(args.samples), args.interface_draws)
    candidate_frame = pd.DataFrame(candidates[mechanism_ids], columns=MECHANISM_PATHS)
    candidate_frame.insert(0, "mechanism_id", mechanism_ids)
    candidate_frame.insert(0, "candidate_id", np.arange(composite_count))
    interface_samples = np.vstack(
        [
            sample_interface_lhs(
                args.interface_draws, args.lhs_seed + 1 + mechanism_id
            )
            for mechanism_id in range(args.samples)
        ]
    )
    candidate_frame["interface_fraction"] = interface_samples[:, 0]
    candidate_frame["interface_exponent"] = interface_samples[:, 1]
    candidate_frame.to_csv(args.output / "equivalent_lhs_samples.csv", index=False)

    expanded_predictions = (
        predictions.rename(columns={"candidate_id": "mechanism_id"})
        .merge(
            candidate_frame[["candidate_id", "mechanism_id"]],
            on="mechanism_id", how="inner", validate="many_to_many",
        )
    )
    composite_predictions = apply_dual_branch_forward_operator(
        expanded_predictions, candidate_frame
    )
    composite_predictions.to_csv(
        args.output / "composite_condition_predictions.csv", index=False
    )
    compatible, all_ids, rmse = compatible_candidates(composite_predictions, targets, tolerance)
    matrix_compatible, matrix_all_ids, matrix_rmse = compatible_candidates(
        predictions, targets, tolerance
    )
    mechanism_frame = pd.DataFrame(candidates, columns=MECHANISM_PATHS)
    mechanism_frame.insert(0, "candidate_id", np.arange(args.samples))
    matrix_frame = mechanism_frame[
        mechanism_frame["candidate_id"].isin(matrix_compatible)
    ].copy()
    matrix_frame["calibration_rmse"] = matrix_rmse[
        np.isin(matrix_all_ids, matrix_compatible)
    ]
    matrix_frame.to_csv(args.output / "matrix_only_compatible_candidates.csv", index=False)
    compatible_frame = candidate_frame[candidate_frame["candidate_id"].isin(compatible)].copy()
    compatible_frame["calibration_rmse"] = rmse[np.isin(all_ids, compatible)]
    compatible_frame.to_csv(args.output / "compatible_equivalent_parameters.csv", index=False)

    rat_summary, points = leave_one_rat_out(composite_predictions, targets)
    rat_summary.to_csv(args.output / "loo_rat_summary.csv", index=False)
    points.to_csv(args.output / "loo_point_intervals.csv", index=False)
    matrix_rat_summary, matrix_points = leave_one_rat_out(predictions, targets)
    matrix_rat_summary.to_csv(args.output / "matrix_only_loo_rat_summary.csv", index=False)
    matrix_points.to_csv(args.output / "matrix_only_loo_point_intervals.csv", index=False)
    if len(matrix_compatible):
        matrix_intervals = prediction_quantiles(predictions, matrix_compatible, targets)
        matrix_drift = residual_drift(targets, matrix_intervals)
    else:
        matrix_drift = None

    if len(compatible):
        full_intervals = prediction_quantiles(composite_predictions, compatible, targets)
        drift = residual_drift(targets, full_intervals)
        interface_summary = interface_contribution_summary(composite_predictions, compatible)
        exploratory = None
    else:
        # Pre-registered threshold yields an empty set. Report a clearly
        # labelled nearest-approach envelope (top 5% RMSE) as an exploratory
        # diagnostic: it quantifies the gap and exercises the drift check.
        k = max(1, int(round(0.05 * len(all_ids))))
        exploratory_ids = all_ids[np.argsort(rmse)[:k]]
        full_intervals = prediction_quantiles(composite_predictions, exploratory_ids, targets)
        drift = residual_drift(targets, full_intervals)
        covered = (full_intervals["z_ratio"] >= full_intervals["sim_q05"]) & (
            full_intervals["z_ratio"] <= full_intervals["sim_q95"]
        )
        exploratory = {
            "label": "EXPLORATORY_NEAREST_APPROACH_NOT_A_COMPATIBLE_SET",
            "candidate_count": int(k),
            "best_rmse": float(np.min(rmse)),
            "point_coverage_of_targets": float(np.mean(covered)),
            "interpretation": "Even the best 5% of the dual-branch composite "
            "space fails the compatibility rule; structural mismatch remains "
            "and the model must not be used as a prediction tool.",
        }
        interface_summary = None
    full_intervals.to_csv(args.output / "full_data_prediction_intervals.csv", index=False)

    metrics = {
        "status": "TRACK_B_V2_DUAL_BRANCH_EQUIVALENT_CALIBRATION_PLATFORM_BOUND",
        "model_version": "v2_dual_branch",
        "boundary_modes": {
            stage_name: stage.get(
                "boundary_mode",
                base_values["transport"].get("boundary_mode", "outer_only"),
            )
            for stage_name, stage in base_values["stages"].items()
        },
        "platform_disclaimer": PLATFORM_DISCLAIMER,
        "equivalent_space": {p: list(r) for p, r in EQUIVALENT_SPACE.items()},
        "equivalent_space_label": "COMPOSITE_EQUIVALENT_RANGES_NOT_LITERATURE_PRIORS",
        "interface_fraction_range": list(INTERFACE_FRACTION_RANGE),
        "interface_exponent_range": list(INTERFACE_EXPONENT_RANGE),
        "observation_operator": "Z/Z0 = (1-beta)*R_matrix/R_matrix0 + beta*(porosity_outer_surface0/porosity_outer_surface(t))**m",
        "interface_branch_scope": "Electrode-contact outer surface only; acid contacts the outer table alone, while the immersed EDTA inner-surface reaction is represented in the matrix mechanism.",
        "target": "Z_post/Z_baseline magnitude ratio at 100/120 Hz, sites averaged within rat",
        "confounded_points_dropped": True,
        "data_audit": {
            "source_frequency_rows": int(len(spectra)),
            "source_spectrum_count": int(spectra["spectrum_id"].nunique()),
            "source_rat_count": int(spectra["rat"].nunique()),
            "unfiltered_low_frequency_target_count": int(len(unfiltered_targets)),
            "excluded_confounded_target_count": int(len(unfiltered_targets) - len(targets)),
            "included_target_count": int(len(targets)),
            "included_rat_count": int(targets["rat"].nunique()),
            "all_targets_have_three_sites_and_six_values": bool(
                (targets["site_count"] == 3).all() and (targets["n_values"] == 6).all()
            ),
            "single_rat_conditions": single_rat_conditions,
        },
        "tolerance_rule": "RMS of per-condition between-rat SD of Z ratio",
        "tolerance": tolerance,
        "prior_space_screening": prior_screening,
        "matrix_only_baseline": {
            "description": "Same stage-specific-boundary mechanism candidates without the interface branch; isolates the effect of the v2 observation operator.",
            "compatible_count": int(len(matrix_compatible)),
            "best_rmse": float(np.min(matrix_rmse)),
            "loo_point_coverage": float(matrix_points["covered"].mean()) if len(matrix_points) else None,
            "loo_median_interval_width": float(matrix_points["width"].median()) if len(matrix_points) else None,
            "residual_drift": matrix_drift,
        },
        "equivalent_space_screening": {
            "mechanism_candidates_total": int(args.samples),
            "interface_draws_per_mechanism": int(args.interface_draws),
            "candidates_total": int(len(all_ids)),
            "compatible_count": int(len(compatible)),
            "compatible_fraction": float(len(compatible) / max(len(all_ids), 1)),
            "best_rmse": float(np.min(rmse)),
            "sweep_failures": int(failures),
        },
        "loo": {
            "point_coverage": float(points["covered"].mean()) if len(points) else None,
            "median_interval_width": float(points["width"].median()) if len(points) else None,
            "median_relative_width": float((points["width"] / points["z_ratio"]).median()) if len(points) else None,
            "rats_with_empty_compatible_set": rat_summary.loc[rat_summary["compatible_count"] == 0, "rat"].tolist(),
            "reading_rule": "Coverage and interval width are dual pre-registered metrics; interval width is itself a conclusion, with no narrowness threshold.",
        },
        "residual_drift": drift,
        "interface_contribution": interface_summary,
        "exploratory_nearest_approach": exploratory,
        "diffusion_timescale_s": _diffusion_timescales(base_values),
        "source_sha256": {
            "data": sha256(args.data),
            "experimental_parser": sha256(ROOT / "cranial_model" / "experimental.py"),
            "module": sha256(ROOT / "cranial_model" / "equivalent_calibration.py"),
            "runner": sha256(Path(__file__).resolve()),
            "parameters": sha256(args.parameters),
            "core": sha256(ROOT / "cranial_model" / "core.py"),
        },
    }
    (args.output / "equivalent_calibration_metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if len(points):
        _plot_loo_bands(points, targets, args.output, exploratory=None)
    elif len(full_intervals):
        _plot_loo_bands(full_intervals, targets, args.output, exploratory=exploratory)
    if len(full_intervals) and np.isfinite(drift.get("residual_vs_edta_time_spearman_rho", np.nan)):
        _plot_residual_drift(targets, full_intervals, drift, args.output, exploratory=exploratory)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


def _diffusion_timescales(values: dict) -> dict[str, dict[str, float]]:
    """Characteristic through-layer diffusion times L^2/D_eff at native porosity."""
    result: dict[str, dict[str, float]] = {}
    for stage_name in ("acetic_acid", "edta"):
        bulk = values["stages"][stage_name]["bulk_diffusivity_m2_per_s"]
        per_layer = {}
        for layer in values["geometry"]["layers"]:
            d_eff = bulk * layer["porosity"] / layer["tortuosity"]
            per_layer[layer["name"]] = float((layer["thickness_mm"] * 1e-3) ** 2 / d_eff)
        result[stage_name] = per_layer
    return result


def _stage_time_min(row) -> float:
    offset = 5.0 if row["stage"] == "edta" else 0.0  # acetic ends at 5 min
    return offset + row["time_s"] / 60.0


def _plot_loo_bands(points: pd.DataFrame, targets: pd.DataFrame, output: Path, exploratory: dict | None) -> None:
    configure_style()
    doses = sorted(targets["edta_concentration_mol_per_l"].unique())
    figure, axes = plt.subplots(1, len(doses), figsize=(3.1 * len(doses), 3.4), sharey=True)
    for axis, dose, color in zip(np.atleast_1d(axes), doses, DOSE_COLORS):
        subset = points[points["edta_concentration_mol_per_l"] == dose].copy()
        subset["time_min"] = subset.apply(_stage_time_min, axis=1)
        condition = subset.groupby("time_min").agg(
            q05=("sim_q05", "median"), q95=("sim_q95", "median")
        ).sort_index()
        axis.fill_between(condition.index, condition["q05"], condition["q95"], color=color, alpha=0.2, lw=0)
        observed = targets[targets["edta_concentration_mol_per_l"] == dose].copy()
        observed["time_min"] = observed.apply(_stage_time_min, axis=1)
        for rat, group in observed.groupby("rat"):
            axis.plot(group["time_min"], group["z_ratio"], ".-", color="0.45", lw=0.7, ms=3, alpha=0.75)
        axis.set_title(f"EDTA {dose:.2f} mol/L", fontsize=9)
        axis.set_xlabel("Process time (min)")
        axis.grid(alpha=0.2)
    np.atleast_1d(axes)[0].set_ylabel("Z post / Z baseline (100-120 Hz)")
    if exploratory is None:
        title = "Track B v2 leave-one-rat-out 90% prediction bands (grey lines: held-out rats)"
        name = "FigB1_loo_prediction_bands.png"
    else:
        title = ("Track B v2 EXPLORATORY nearest-approach envelope (top 5% RMSE; "
                 "not a compatible set - structural mismatch remains)")
        name = "FigB1_exploratory_nearest_approach.png"
    figure.suptitle(title, fontsize=9)
    figure.text(0.99, 0.01, PLATFORM_DISCLAIMER, ha="right", fontsize=6, color="0.35", wrap=True)
    figure.tight_layout(rect=(0, 0.05, 1, 0.94))
    figure.savefig(output / name, facecolor="white")
    plt.close(figure)


def _plot_residual_drift(targets: pd.DataFrame, intervals: pd.DataFrame, drift: dict, output: Path, exploratory: dict | None) -> None:
    from cranial_model.equivalent_calibration import CONDITION_KEYS

    configure_style()
    merged = targets.merge(
        intervals[["rat", *CONDITION_KEYS, "sim_q50"]], on=["rat", *CONDITION_KEYS], how="inner"
    )
    merged["residual"] = merged["z_ratio"] - merged["sim_q50"]
    merged["time_min"] = merged.apply(_stage_time_min, axis=1)
    figure, axis = plt.subplots(figsize=(6.6, 3.6))
    for stage, color, marker in (("acetic_acid", "#762A83", "s"), ("edta", "#2166AC", "o")):
        subset = merged[merged["stage"] == stage]
        axis.scatter(subset["time_min"], subset["residual"], s=14, alpha=0.65,
                     color=color, marker=marker, label=stage.replace("_", " ").title())
    axis.axhline(0.0, color="0.4", lw=0.9)
    axis.set_xlabel("Process time (min)")
    axis.set_ylabel("Observed - envelope-median Z ratio")
    rho = drift["residual_vs_edta_time_spearman_rho"]
    p = drift["residual_vs_edta_time_spearman_p"]
    label = "exploratory nearest-approach" if exploratory is not None else "compatible-set"
    axis.set_title(f"Residual drift diagnostic, {label} (EDTA stage Spearman rho={rho:.2f}, p={p:.3f})", fontsize=10)
    axis.legend(fontsize=8)
    axis.grid(alpha=0.2)
    figure.text(0.99, 0.01, PLATFORM_DISCLAIMER, ha="right", fontsize=6, color="0.35")
    figure.tight_layout()
    figure.savefig(output / "FigB2_residual_drift.png", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
