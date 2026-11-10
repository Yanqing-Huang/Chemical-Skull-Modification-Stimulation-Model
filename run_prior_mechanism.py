#!/usr/bin/env python3
"""Track A (main line): literature-prior LHS mechanism ensemble for the three-layer skull.

Samples only literature-declared prior ranges (9 per-layer kinetic keys plus 2
transport keys), never touches measured impedance, and reports ensemble
quantile bands for per-layer fronts, porosity, calcium and whole-skull
equivalent resistance. All outputs are mechanism predictions from literature
priors and are NOT calibrated to in-house impedance data.
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
from scipy.stats import spearmanr

from cranial_model.core import (
    LAYER_NAMES,
    load_parameters,
    porosity_from_calcium,
    simulate_scenario,
)
from cranial_model.diagnostics import thiele_biot_rows
from cranial_model.ensemble import apply_candidate, sample_lhs_log10
from cranial_model.plotting import COLORS, DOSE_COLORS, configure_style


ROOT = Path(__file__).resolve().parent

MECHANISM_PATHS = (
    "stages.acetic_acid.k_r_m3_per_mol_s.outer_table",
    "stages.acetic_acid.k_r_m3_per_mol_s.diploe",
    "stages.acetic_acid.k_r_m3_per_mol_s.inner_table",
    "stages.edta.k_r_m3_per_mol_s.outer_table",
    "stages.edta.k_r_m3_per_mol_s.diploe",
    "stages.edta.k_r_m3_per_mol_s.inner_table",
    "stages.edta.surface_site_capacity_mol_per_m3.outer_table",
    "stages.edta.surface_site_capacity_mol_per_m3.diploe",
    "stages.edta.surface_site_capacity_mol_per_m3.inner_table",
    "transport.external_mass_transfer_m_per_s",
    "transport.porosity_per_released_calcium_m3_per_mol",
)
STAGES = ("acetic_acid", "edta")
FRONT_THRESHOLD = 0.99
# Fixed superficial region driving the platform-specific interface branch.
# Cell-left-edge selection gives ~0.1 mm at both A and B grid resolutions.
OUTER_SURFACE_DEPTH_M = 1e-4
# Impedance measurement times within each stage (seconds after stage start).
MEASURED_STAGE_TIMES_S = {
    "acetic_acid": (300.0,),
    "edta": (900.0, 1800.0, 3600.0, 5400.0),
}
DISCLAIMER = (
    "Mechanism prediction from literature priors; "
    "not calibrated to in-house impedance data."
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _conductivity(values: dict, porosity: np.ndarray) -> np.ndarray:
    electrical = values["electrical"]
    return (
        electrical["mineral_matrix_conductivity_s_per_m"]
        + electrical["measurement_fluid_conductivity_s_per_m"]
        * porosity ** electrical["archie_exponent"]
    )


def _layer_top_edges_m(grid) -> np.ndarray:
    edges = []
    for layer_slice in grid.layer_slices:
        edges.append(float(grid.edges[layer_slice.start]))
    return np.asarray(edges)


def extract_scenario_series(scenario) -> dict[str, np.ndarray | float]:
    """Per-stage time series for layers/fronts/whole-skull resistance."""
    grid = scenario.grid
    native = scenario.native_area_resistance_ohm_m2
    layer_tops = _layer_top_edges_m(grid)
    outer = grid.layer_slices[0]
    surface_local = np.flatnonzero(
        grid.edges[outer.start : outer.stop] < OUTER_SURFACE_DEPTH_M
    )
    if surface_local.size == 0:
        surface_local = np.array([0])
    surface_indices = outer.start + surface_local
    surface_widths = grid.widths[surface_indices]
    series: dict[str, np.ndarray | float] = {
        "native_outer_surface_porosity": float(
            np.average(grid.initial_porosity[surface_indices], weights=surface_widths)
        )
    }
    for stage_name, result in scenario.stages.items():
        n_times = result.time_s.shape[0]
        prefix = f"{stage_name}__"
        boundary_mode = scenario.values["stages"][stage_name].get(
            "boundary_mode",
            scenario.values["transport"].get("boundary_mode", "outer_only"),
        )
        series[prefix + "time_s"] = result.time_s
        series[prefix + "global_time_s"] = result.global_time_s
        whole = np.empty(n_times)
        layer_ca = np.empty((n_times, 3))
        layer_porosity = np.empty((n_times, 3))
        layer_resistance = np.empty((n_times, 3))
        surface_porosity = np.empty(n_times)
        fronts = np.full((n_times, 3), np.nan)
        for time_index in range(n_times):
            calcium = result.calcium[time_index]
            porosity = result.porosity[time_index]
            surface_porosity[time_index] = float(
                np.average(porosity[surface_indices], weights=surface_widths)
            )
            conductivity = _conductivity(scenario.values, porosity)
            resistance_cells = grid.widths / conductivity
            whole[time_index] = float(np.sum(resistance_cells)) / native
            for layer_index, layer_slice in enumerate(grid.layer_slices):
                widths = grid.widths[layer_slice]
                layer_ca[time_index, layer_index] = float(
                    np.sum(calcium[layer_slice] * widths)
                    / np.sum(grid.q_reference[layer_slice] * widths)
                )
                layer_porosity[time_index, layer_index] = float(
                    np.average(porosity[layer_slice], weights=widths)
                )
                layer_resistance[time_index, layer_index] = float(
                    np.sum(resistance_cells[layer_slice])
                )
                if layer_index in (0, 2):
                    fraction = calcium[layer_slice] / grid.q_reference[layer_slice]
                    reacted = np.flatnonzero(fraction < FRONT_THRESHOLD)
                    if reacted.size:
                        if layer_index == 2 and boundary_mode == "immersed":
                            # Inner table is wetted from the skull interior:
                            # track the front inward from its inner surface.
                            layer_bottom = grid.edges[layer_slice.stop]
                            shallowest = grid.centers[layer_slice][reacted[0]]
                            fronts[time_index, layer_index] = (
                                layer_bottom - shallowest
                            ) * 1000.0
                        else:
                            deepest = grid.centers[layer_slice][reacted[-1]]
                            fronts[time_index, layer_index] = (
                                deepest - layer_tops[layer_index]
                            ) * 1000.0
                    else:
                        fronts[time_index, layer_index] = 0.0
        series[prefix + "whole_resistance_ratio"] = whole
        series[prefix + "outer_surface_porosity"] = surface_porosity
        for layer_index, layer_name in enumerate(LAYER_NAMES):
            series[prefix + f"ca_remaining__{layer_name}"] = layer_ca[:, layer_index]
            series[prefix + f"porosity__{layer_name}"] = layer_porosity[:, layer_index]
            series[prefix + f"layer_resistance__{layer_name}"] = layer_resistance[:, layer_index]
            if layer_index in (0, 2):
                series[prefix + f"front_mm__{layer_name}"] = fronts[:, layer_index]
    # Whole-skull resistance ratio at the bench measurement times; shared with
    # the Track B equivalent-calibration module (same LHS candidates).
    for stage_name, result in scenario.stages.items():
        ratio = series[f"{stage_name}__whole_resistance_ratio"]
        for time_s in MEASURED_STAGE_TIMES_S.get(stage_name, ()):  # type: ignore[attr-defined]
            index = int(np.argmin(np.abs(result.time_s - time_s)))
            if abs(result.time_s[index] - time_s) > 1e-6:
                raise RuntimeError(f"{stage_name} grid lacks measurement time {time_s}")
            series[f"cond__{stage_name}__{time_s:.0f}"] = float(ratio[index])
            series[f"cond_surface__{stage_name}__{time_s:.0f}"] = float(
                series[f"{stage_name}__outer_surface_porosity"][index]  # type: ignore[index]
                / series["native_outer_surface_porosity"]
            )
    return series


def run_candidate(
    base_values: dict,
    candidate: np.ndarray,
    doses: list[float],
    cells_per_layer: list[int],
    max_step_s: float,
    paths: tuple[str, ...],
) -> dict[float, dict[str, np.ndarray | float]]:
    values = apply_candidate(base_values, paths, candidate)
    results = {}
    for dose in doses:
        scenario = simulate_scenario(
            values,
            dose,
            cells_per_layer=cells_per_layer,
            max_step_s=max_step_s,
            stage_names=STAGES,
        )
        results[dose] = extract_scenario_series(scenario)
    return results


def quantile_frame(
    stacked: dict[str, np.ndarray],
    doses: list[float],
) -> pd.DataFrame:
    """stacked: metric key -> array (n_candidates, n_doses, n_times)."""
    rows = []
    for key, values in stacked.items():
        stage, metric = key.split("__", 1)
        for dose_index, dose in enumerate(doses):
            times = stacked[f"{stage}__global_time_s"][0, dose_index]
            quantiles = np.nanquantile(values[:, dose_index, :], [0.05, 0.5, 0.95], axis=0)
            for column, value in enumerate(quantiles.T):
                rows.append(
                    {
                        "stage": stage,
                        "metric": metric,
                        "edta_concentration_mol_per_l": dose,
                        "global_time_s": float(times[column]),
                        "q05": float(value[0]),
                        "q50": float(value[1]),
                        "q95": float(value[2]),
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=ROOT / "parameters" / "theoretical_prior.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "prior_mechanism")
    parser.add_argument("--samples", type=int, default=200)
    parser.add_argument("--lhs-seed", type=int, default=20261112)
    parser.add_argument("--cells", nargs=3, type=int, default=[24, 12, 12])
    parser.add_argument("--max-step", type=float, default=10.0)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    document = load_parameters(args.parameters)
    base_values = document["values"]
    doses = sorted(float(d) for d in base_values["scenarios"]["edta_concentrations_mol_per_l"])

    manifest = {
        "status": "MECHANISM_PRIOR_ENSEMBLE_NOT_CALIBRATED_TO_INHOUSE_IMPEDANCE",
        "disclaimer": DISCLAIMER,
        "lhs": {
            "sample_count": args.samples,
            "seed": args.lhs_seed,
            "space": "log10",
            "sampled_paths": list(MECHANISM_PATHS),
            "prior_ranges": {path: document["uncertainty"][path]["range"] for path in MECHANISM_PATHS},
        },
        "numerics": {
            "cells_per_layer": args.cells,
            "max_step_s": args.max_step,
            "output_times_per_stage": base_values["numerics"]["output_times_per_stage"],
        },
        "doses_mol_per_l": doses,
        "boundary_modes": {
            stage_name: stage.get(
                "boundary_mode",
                base_values["transport"].get("boundary_mode", "outer_only"),
            )
            for stage_name, stage in base_values["stages"].items()
        },
        "outer_surface_depth_m": OUTER_SURFACE_DEPTH_M,
        "front_threshold_ca_fraction": FRONT_THRESHOLD,
        "front_definition": "Deepest 1%-decalcified position from the stage-specific wetted surface; inner table is measured inward from the inner surface only in immersed stages (EDTA/urea), not during outer-only acetic acid.",
        "source_sha256": {
            name: sha256(ROOT / name)
            for name in [
                "run_prior_mechanism.py",
                "cranial_model/core.py",
                "cranial_model/ensemble.py",
                "cranial_model/diagnostics.py",
                "parameters/theoretical_prior.json",
            ]
        },
    }

    candidates = sample_lhs_log10(document, args.samples, args.lhs_seed, paths=MECHANISM_PATHS)
    candidate_frame = pd.DataFrame(candidates, columns=MECHANISM_PATHS)
    candidate_frame.insert(0, "candidate_id", np.arange(args.samples))
    candidate_frame.to_csv(args.output / "lhs_samples.csv", index=False)

    # --- ensemble sweep -----------------------------------------------------
    stacked: dict[str, list] = {}
    condition_rows: list[dict] = []
    failure_count = 0
    context = multiprocessing.get_context("fork") if "fork" in multiprocessing.get_all_start_methods() else None
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as executor:
        futures = {
            executor.submit(
                run_candidate,
                base_values,
                candidate,
                doses,
                args.cells,
                args.max_step,
                MECHANISM_PATHS,
            ): index
            for index, candidate in enumerate(candidates)
        }
        completed = 0
        for future in as_completed(futures):
            index = futures[future]
            try:
                per_dose = future.result()
            except Exception as error:  # noqa: BLE001 - record and continue ensemble
                failure_count += 1
                print(f"candidate {index} failed: {error}", flush=True)
                continue
            for dose_index, dose in enumerate(doses):
                for key, value in per_dose[dose].items():
                    if isinstance(value, np.ndarray):
                        stacked.setdefault(key, []).append((index, dose_index, value))
                    elif key.startswith("cond__"):
                        _, stage, time_s = key.split("__")
                        condition_rows.append(
                            {
                                "candidate_id": index,
                                "stage": stage,
                                "time_s": float(time_s),
                                "edta_concentration_mol_per_l": dose,
                                "sim_whole_resistance_ratio": value,
                                "sim_outer_surface_porosity_ratio": per_dose[dose][
                                    key.replace("cond__", "cond_surface__", 1)
                                ],
                            }
                        )
            completed += 1
            if completed % 10 == 0 or completed == args.samples:
                print(f"mechanism ensemble {completed}/{args.samples}", flush=True)

    metric_arrays: dict[str, np.ndarray] = {}
    for key, entries in stacked.items():
        first = entries[0][2]
        array = np.full((args.samples, len(doses), first.shape[0]), np.nan)
        for index, dose_index, value in entries:
            array[index, dose_index, :] = value
        metric_arrays[key] = array

    quantiles = quantile_frame(metric_arrays, doses)
    quantiles.to_csv(args.output / "mechanism_timeseries_quantiles.csv", index=False)
    conditions = pd.DataFrame(condition_rows)
    conditions.to_csv(args.output / "candidate_condition_predictions.csv", index=False)

    # --- scalar targets for sensitivity ------------------------------------
    final_ratio = metric_arrays["edta__whole_resistance_ratio"][:, :, -1]
    sensitivity_rows = []
    valid = np.all(np.isfinite(final_ratio), axis=1)
    for path_index, path in enumerate(MECHANISM_PATHS):
        for dose_index, dose in enumerate(doses):
            rho, pvalue = spearmanr(candidates[valid, path_index], final_ratio[valid, dose_index])
            sensitivity_rows.append(
                {
                    "parameter": path,
                    "target": f"whole_resistance_ratio@edta_end dose={dose:.2f}",
                    "spearman_rho": float(rho),
                    "p_value": float(pvalue),
                }
            )
    for layer_index, layer_name in enumerate(LAYER_NAMES):
        layer_final = metric_arrays[f"edta__ca_remaining__{layer_name}"][:, :, -1]
        reference_dose_index = doses.index(0.1) if 0.1 in doses else len(doses) - 1
        layer_valid = np.isfinite(layer_final[:, reference_dose_index])
        for path_index, path in enumerate(MECHANISM_PATHS):
            rho, pvalue = spearmanr(
                candidates[layer_valid, path_index],
                layer_final[layer_valid, reference_dose_index],
            )
            sensitivity_rows.append(
                {
                    "parameter": path,
                    "target": f"ca_remaining@edta_end layer={layer_name} dose=0.10",
                    "spearman_rho": float(rho),
                    "p_value": float(pvalue),
                }
            )
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(args.output / "sensitivity_spearman.csv", index=False)

    # --- Thiele/Biot table at central values --------------------------------
    diagnostic_rows = []
    for dose in doses:
        scenario = simulate_scenario(
            base_values,
            dose,
            cells_per_layer=args.cells,
            max_step_s=args.max_step,
            stage_names=STAGES,
        )
        diagnostic_rows.extend(thiele_biot_rows(scenario))
    pd.DataFrame(diagnostic_rows).to_csv(args.output / "thiele_biot_central_values.csv", index=False)

    # --- figures -------------------------------------------------------------
    configure_style()
    _plot_whole_skull_bands(metric_arrays, doses, base_values, args.output)
    _plot_layer_bands(metric_arrays, doses, base_values, args.output)
    _plot_fronts(metric_arrays, doses, base_values, args.output)
    _plot_sensitivity(sensitivity, args.output)

    manifest["failures"] = int(failure_count)
    manifest["completed_candidates"] = int(args.samples - failure_count)
    (args.output / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({"status": manifest["status"], "failures": failure_count}, ensure_ascii=False))


def _dose_panels(doses):
    figure, axes = plt.subplots(1, len(doses), figsize=(3.2 * len(doses), 3.4), sharey=True)
    return figure, np.atleast_1d(axes)


def _plot_whole_skull_bands(metric_arrays, doses, values, output: Path) -> None:
    figure, axes = _dose_panels(doses)
    acid_end = values["stages"]["acetic_acid"]["duration_s"]
    for dose_index, (axis, dose) in enumerate(zip(axes, doses)):
        time_acid = metric_arrays["acetic_acid__global_time_s"][0, dose_index] / 60.0
        time_edta = metric_arrays["edta__global_time_s"][0, dose_index] / 60.0
        for stage, time in (("acetic_acid", time_acid), ("edta", time_edta)):
            data = metric_arrays[f"{stage}__whole_resistance_ratio"][:, dose_index, :]
            q05, q50, q95 = np.nanquantile(data, [0.05, 0.5, 0.95], axis=0)
            axis.fill_between(time, q05, q95, color="#2166AC", alpha=0.25, lw=0)
            axis.plot(time, q50, color="#2166AC", lw=1.6)
        axis.axvline(acid_end / 60.0, color="0.6", lw=0.8, ls=":")
        axis.set_title(f"EDTA {dose:.2f} mol/L", fontsize=9)
        axis.set_xlabel("Process time (min)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Whole-skull area resistance / native")
    figure.suptitle("Track A prior ensemble: whole-skull equivalent resistance (5-95% band)", fontsize=10)
    figure.text(0.99, 0.01, DISCLAIMER, ha="right", fontsize=7, color="0.35")
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(output / "FigA1_whole_skull_resistance_bands.png", facecolor="white")
    plt.close(figure)


def _plot_layer_bands(metric_arrays, doses, values, output: Path) -> None:
    reference_dose = 0.1 if 0.1 in doses else doses[-1]
    dose_index = doses.index(reference_dose)
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.4), sharey=True)
    for layer_index, (axis, layer_name) in enumerate(zip(axes, LAYER_NAMES)):
        for stage in ("acetic_acid", "edta"):
            time = metric_arrays[f"{stage}__global_time_s"][0, dose_index] / 60.0
            data = metric_arrays[f"{stage}__ca_remaining__{layer_name}"][:, dose_index, :]
            q05, q50, q95 = np.nanquantile(data, [0.05, 0.5, 0.95], axis=0)
            axis.fill_between(time, 100 * q05, 100 * q95, color=COLORS[layer_name], alpha=0.22, lw=0)
            axis.plot(time, 100 * q50, color=COLORS[layer_name], lw=1.6)
        axis.axvline(values["stages"]["acetic_acid"]["duration_s"] / 60.0, color="0.6", lw=0.8, ls=":")
        axis.set_title(layer_name.replace("_", " ").title(), fontsize=9)
        axis.set_xlabel("Process time (min)")
        axis.grid(alpha=0.2)
    axes[0].set_ylabel("Layer calcium remaining (%)")
    figure.suptitle(f"Track A prior ensemble: layer decalcification at EDTA {reference_dose:.2f} mol/L", fontsize=10)
    figure.text(0.99, 0.01, DISCLAIMER, ha="right", fontsize=7, color="0.35")
    figure.tight_layout(rect=(0, 0.04, 1, 0.94))
    figure.savefig(output / "FigA2_layer_calcium_bands.png", facecolor="white")
    plt.close(figure)


def _plot_fronts(metric_arrays, doses, values, output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(7.4, 3.4))
    thickness = {
        name: layer["thickness_mm"]
        for name, layer in zip(LAYER_NAMES, values["geometry"]["layers"])
    }
    for axis, layer_name in zip(axes, ("outer_table", "inner_table")):
        for dose_index, dose in enumerate(doses):
            time = metric_arrays["edta__global_time_s"][0, dose_index] / 60.0
            data = metric_arrays[f"edta__front_mm__{layer_name}"][:, dose_index, :]
            q05, q50, q95 = np.nanquantile(data, [0.05, 0.5, 0.95], axis=0)
            color = DOSE_COLORS[dose_index]
            axis.plot(time, q50, color=color, lw=1.6, label=f"{doses[dose_index]:.2f} mol/L")
            axis.fill_between(time, q05, q95, color=color, alpha=0.15, lw=0)
        axis.axhline(thickness[layer_name], color="0.5", lw=0.9, ls="--")
        axis.set_title(f"{layer_name.replace('_', ' ').title()} reaction front (EDTA stage)", fontsize=9)
        axis.set_xlabel("EDTA stage time (min)")
        axis.set_ylabel("Front depth from wetted surface (mm)")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=7)
    figure.text(0.99, 0.01, DISCLAIMER, ha="right", fontsize=7, color="0.35")
    figure.tight_layout(rect=(0, 0.04, 1, 1.0))
    figure.savefig(output / "FigA3_reaction_fronts.png", facecolor="white")
    plt.close(figure)


def _plot_sensitivity(sensitivity: pd.DataFrame, output: Path) -> None:
    pivot = sensitivity.pivot(index="parameter", columns="target", values="spearman_rho")
    short = {
        path: path.replace("stages.", "").replace("k_r_m3_per_mol_s", "k_r")
        .replace("surface_site_capacity_mol_per_m3", "C_site")
        .replace("transport.", "")
        for path in pivot.index
    }
    pivot = pivot.rename(index=short)
    figure, axis = plt.subplots(figsize=(8.6, 4.6))
    image = axis.imshow(pivot.to_numpy(), cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    axis.set_xticks(range(len(pivot.columns)))
    axis.set_xticklabels([c.replace("@edta_end ", "\n") for c in pivot.columns], fontsize=7, rotation=35, ha="right")
    axis.set_yticks(range(len(pivot.index)))
    axis.set_yticklabels(pivot.index, fontsize=8)
    for row in range(pivot.shape[0]):
        for column in range(pivot.shape[1]):
            value = pivot.to_numpy()[row, column]
            axis.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                      color="white" if abs(value) > 0.55 else "black")
    figure.colorbar(image, label="Spearman ρ", shrink=0.85)
    axis.set_title("Track A prior-ensemble parameter sensitivity (LHS, no experimental fitting)", fontsize=10)
    figure.text(0.99, 0.01, DISCLAIMER, ha="right", fontsize=7, color="0.35")
    figure.tight_layout()
    figure.savefig(output / "FigA4_sensitivity_heatmap.png", facecolor="white")
    plt.close(figure)


if __name__ == "__main__":
    main()
