#!/usr/bin/env python3
"""Run the uncalibrated SD-rat skull prior and export reproducible results."""

from __future__ import annotations

import argparse
import csv
import json
import platform
from pathlib import Path

import numpy as np
import scipy
from scipy.integrate import solve_ivp

from cranial_model import LAYER_NAMES, STAGE_NAMES, load_parameters, simulate_scenario
from cranial_model.core import (
    calcium_release_per_area,
    layer_calcium_remaining_fraction,
    layer_mean_porosity,
    layer_parameter_values,
)
from cranial_model.diagnostics import (
    inner_scm_flux_mol_per_m2_s,
    inner_scm_time_constant_driver_s,
    outer_scm_flux_mol_per_m2_s,
    outer_scm_time_to_front_s,
    thiele_biot_rows,
)
from cranial_model.plotting import (
    plot_dose_and_resistance,
    plot_model_schematic,
    plot_three_stage_results,
)


ROOT = Path(__file__).resolve().parent


def write_dict_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def export_profiles(scenarios, path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "edta_concentration_mol_per_l",
                "stage",
                "local_time_s",
                "global_process_time_s",
                "depth_mm",
                "layer",
                "concentration_mol_per_l_pore_liquid",
                "calcium_mol_per_m3_bulk_bone",
                "porosity",
            ]
        )
        for scenario in scenarios:
            x_mm = scenario.grid.centers * 1000.0
            for stage_name in STAGE_NAMES:
                result = scenario.stages[stage_name]
                for time_index in range(len(result.time_s)):
                    for cell in range(scenario.grid.centers.size):
                        writer.writerow(
                            [
                                scenario.edta_concentration_mol_per_l,
                                stage_name,
                                result.time_s[time_index],
                                result.global_time_s[time_index],
                                x_mm[cell],
                                LAYER_NAMES[scenario.grid.layer_index[cell]],
                                result.concentration[time_index, cell] / 1000.0,
                                result.calcium[time_index, cell],
                                result.porosity[time_index, cell],
                            ]
                        )


def export_parameter_table(document: dict, path: Path) -> None:
    values = document["values"]
    uncertainty = document["uncertainty"]
    rows = []

    def add_row(display_path, value, uncertainty_path=None):
        metadata = uncertainty.get(uncertainty_path or display_path, {})
        prior_range = metadata.get("range", ["", ""])
        rows.append(
            {
                "parameter_path": display_path,
                "value": value,
                "prior_range_lower": prior_range[0],
                "prior_range_upper": prior_range[1],
                "status": "THEORETICAL_PRIOR_PENDING_CALIBRATION",
                "source_or_assumption": metadata.get(
                    "source", "Editable model input; no direct SD-rat measurement is claimed."
                ),
            }
        )

    add_row("geometry.total_thickness_mm", values["geometry"]["total_thickness_mm"])
    for layer in values["geometry"]["layers"]:
        for field in ("thickness_mm", "porosity", "tortuosity", "initial_calcium_mol_per_m3"):
            display = f"layers.{layer['name']}.{field}"
            add_row(display, layer[field])
    for section in ("transport", "stages", "electrical"):
        section_values = values[section]
        if section == "stages":
            for stage_name, stage_values in section_values.items():
                for field, value in stage_values.items():
                    add_row(f"stages.{stage_name}.{field}", value)
        else:
            for field, value in section_values.items():
                add_row(f"{section}.{field}", value)
    add_row(
        "scenarios.edta_concentrations_mol_per_l",
        ";".join(str(value) for value in values["scenarios"]["edta_concentrations_mol_per_l"]),
    )
    write_dict_csv(path, rows)


def export_summary(scenarios, output: Path) -> list[dict]:
    summary_rows = []
    time_rows = []
    for scenario in scenarios:
        grid = scenario.grid
        native_calcium = np.sum(grid.q_reference * grid.widths)
        for stage_name in STAGE_NAMES:
            result = scenario.stages[stage_name]
            resistance = scenario.area_resistance(result.final_calcium)
            targets: list[str | None] = [None, *LAYER_NAMES]
            for layer_name in targets:
                label = layer_name or "whole_skull"
                selected = (
                    slice(None)
                    if layer_name is None
                    else grid.layer_slices[LAYER_NAMES.index(layer_name)]
                )
                mean_porosity = (
                    float(np.average(result.porosity[-1], weights=grid.widths))
                    if layer_name is None
                    else layer_mean_porosity(scenario, stage_name, layer_name)
                )
                summary_rows.append(
                    {
                        "edta_concentration_mol_per_l": scenario.edta_concentration_mol_per_l,
                        "stage": stage_name,
                        "layer": label,
                        "calcium_remaining_fraction_of_native": layer_calcium_remaining_fraction(
                            scenario, stage_name, layer_name
                        ),
                        "calcium_release_during_stage_mol_per_m2": calcium_release_per_area(
                            scenario, stage_name, layer_name
                        ),
                        "mean_porosity": mean_porosity,
                        "area_resistance_ohm_m2": resistance,
                        "area_resistance_ratio_to_native": resistance
                        / scenario.native_area_resistance_ohm_m2,
                        "native_area_resistance_ohm_m2": scenario.native_area_resistance_ohm_m2,
                    }
                )
            for time_index in range(len(result.time_s)):
                calcium = result.calcium[time_index]
                resistance_t = scenario.area_resistance(calcium)
                time_rows.append(
                    {
                        "edta_concentration_mol_per_l": scenario.edta_concentration_mol_per_l,
                        "stage": stage_name,
                        "local_time_s": result.time_s[time_index],
                        "global_process_time_s": result.global_time_s[time_index],
                        "whole_skull_calcium_remaining_fraction": float(
                            np.sum(calcium * grid.widths) / native_calcium
                        ),
                        "mean_porosity": float(
                            np.average(result.porosity[time_index], weights=grid.widths)
                        ),
                        "area_resistance_ohm_m2": resistance_t,
                        "area_resistance_ratio_to_native": resistance_t
                        / scenario.native_area_resistance_ohm_m2,
                        "cumulative_inflow_mol_per_m2": result.cumulative_inflow_mol_per_m2[
                            time_index
                        ],
                        "cumulative_reagent_reaction_mol_per_m2": result.cumulative_reaction_mol_per_m2[
                            time_index
                        ],
                    }
                )
    write_dict_csv(output / "scenario_summary.csv", summary_rows)
    write_dict_csv(output / "whole_skull_timeseries.csv", time_rows)
    return summary_rows


def scm_analytical_validation(scenario) -> dict:
    values = scenario.values
    grid = scenario.grid
    layer_slice = grid.layer_slices[0]
    layer = values["geometry"]["layers"][0]
    q = float(np.average(grid.q_reference[layer_slice], weights=grid.widths[layer_slice]))
    thickness = layer["thickness_mm"] * 1e-3
    porosity = layer["porosity"]
    stage = values["stages"]["acetic_acid"]
    d_e = stage["bulk_diffusivity_m2_per_s"] * porosity / layer["tortuosity"]
    k_r_cells = layer_parameter_values(grid, stage["k_r_m3_per_mol_s"])
    k_r_mean = float(np.average(k_r_cells[layer_slice], weights=grid.widths[layer_slice]))
    k_s = k_r_mean * q * values["transport"]["scm_reaction_zone_thickness_m"]
    k_f = values["transport"]["external_mass_transfer_m_per_s"]
    stoichiometry = stage["reagent_per_calcium_stoichiometry"]
    concentration = stage["bulk_concentration_mol_per_l"] * 1000.0
    target = 0.1 * thickness
    outer_time = float(
        outer_scm_time_to_front_s(
            target, concentration, k_f, d_e, k_s, stoichiometry, q
        )
    )

    def outer_rhs(_time, state):
        flux = outer_scm_flux_mol_per_m2_s(concentration, k_f, state[0], d_e, k_s)
        return [flux / (stoichiometry * q)]

    outer_solution = solve_ivp(
        outer_rhs, (0.0, outer_time), [0.0], t_eval=[outer_time], rtol=1e-11, atol=1e-15
    )
    outer_numerical = float(outer_solution.y[0, -1])

    interface_concentration = 0.25 * concentration
    inner_time = float(
        inner_scm_time_constant_driver_s(
            target, interface_concentration, d_e, k_s, stoichiometry, q
        )
    )

    def inner_rhs(_time, state):
        flux = inner_scm_flux_mol_per_m2_s(interface_concentration, state[0], d_e, k_s)
        return [flux / (stoichiometry * q)]

    inner_solution = solve_ivp(
        inner_rhs, (0.0, inner_time), [0.0], t_eval=[inner_time], rtol=1e-11, atol=1e-15
    )
    inner_numerical = float(inner_solution.y[0, -1])
    return {
        "target_front_m": target,
        "outer_table_analytic_time_s": outer_time,
        "outer_table_front_relative_error": abs(outer_numerical - target) / target,
        "inner_table_constant_interface_analytic_time_s": inner_time,
        "inner_table_front_relative_error": abs(inner_numerical - target) / target,
        "inner_driver_mol_per_m3": interface_concentration,
        "external_bath_mol_per_m3": concentration,
        "inner_driver_is_interface_not_external_bath": True,
    }


def convergence_validation(values: dict, baseline) -> tuple[list[dict], list[dict]]:
    grid_rows = []
    grid_cases = [([24, 12, 12], "coarse"), ([48, 24, 24], "baseline"), ([96, 48, 48], "fine")]
    grid_results = {}
    counts_by_label = {label: counts for counts, label in grid_cases}
    for counts, label in grid_cases:
        result = (
            baseline
            if label == "baseline"
            else simulate_scenario(values, 0.1, cells_per_layer=counts)
        )
        grid_results[label] = result
    fine = grid_results["fine"]
    fine_q = fine.stages["edta"].final_calcium / fine.grid.q_reference
    fine_resistance = fine.area_resistance(fine.stages["edta"].final_calcium)
    fine_total_q = np.sum(fine.stages["edta"].final_calcium * fine.grid.widths)
    for label, result in grid_results.items():
        q_ratio = result.stages["edta"].final_calcium / result.grid.q_reference
        interpolated = np.interp(fine.grid.centers, result.grid.centers, q_ratio)
        resistance = result.area_resistance(result.stages["edta"].final_calcium)
        total_q = np.sum(result.stages["edta"].final_calcium * result.grid.widths)
        grid_rows.append(
            {
                "case": label,
                "cells_per_layer": "-".join(str(value) for value in counts_by_label[label]),
                "total_cells": result.grid.centers.size,
                "edta_end_total_calcium_mol_per_m2": total_q,
                "total_calcium_relative_difference_from_fine": (total_q - fine_total_q) / fine_total_q,
                "edta_end_area_resistance_ohm_m2": resistance,
                "resistance_relative_difference_from_fine": (resistance - fine_resistance) / fine_resistance,
                "interpolated_calcium_profile_rms_difference_from_fine": float(np.sqrt(np.mean((interpolated - fine_q) ** 2))),
            }
        )

    time_rows = []
    reference = baseline
    reference_q = np.sum(reference.stages["edta"].final_calcium * reference.grid.widths)
    reference_resistance = reference.area_resistance(reference.stages["edta"].final_calcium)
    for max_step in (20.0, 10.0, 5.0):
        result = baseline if max_step == 5.0 else simulate_scenario(values, 0.1, max_step_s=max_step)
        total_q = np.sum(result.stages["edta"].final_calcium * result.grid.widths)
        resistance = result.area_resistance(result.stages["edta"].final_calcium)
        time_rows.append(
            {
                "max_step_s": max_step,
                "edta_end_total_calcium_mol_per_m2": total_q,
                "total_calcium_relative_difference_from_5s": (total_q - reference_q) / reference_q,
                "edta_end_area_resistance_ohm_m2": resistance,
                "resistance_relative_difference_from_5s": (resistance - reference_resistance) / reference_resistance,
            }
        )
    return grid_rows, time_rows


def build_validation(scenarios, baseline, values, output: Path) -> dict:
    mass_balance = {}
    minima = {"concentration_mol_per_m3": np.inf, "calcium_mol_per_m3": np.inf, "porosity": np.inf}
    continuity = {}
    for scenario in scenarios:
        label = f"EDTA_{scenario.edta_concentration_mol_per_l:.3f}_mol_per_l"
        mass_balance[label] = {
            stage: scenario.stage_mass_residuals(stage) for stage in STAGE_NAMES
        }
        for stage in STAGE_NAMES:
            result = scenario.stages[stage]
            minima["concentration_mol_per_m3"] = min(
                minima["concentration_mol_per_m3"], float(result.concentration.min())
            )
            minima["calcium_mol_per_m3"] = min(
                minima["calcium_mol_per_m3"], float(result.calcium.min())
            )
            minima["porosity"] = min(minima["porosity"], float(result.porosity.min()))
        continuity[label] = {
            "acid_to_edta_max_calcium_jump": float(
                np.max(
                    np.abs(
                        scenario.stages["edta"].initial_calcium
                        - scenario.stages["acetic_acid"].final_calcium
                    )
                )
            ),
            "edta_to_urea_max_calcium_jump": float(
                np.max(
                    np.abs(
                        scenario.stages["urea"].initial_calcium
                        - scenario.stages["edta"].final_calcium
                    )
                )
            ),
            "max_stage_initial_liquid_amount": max(
                float(np.max(np.abs(scenario.stages[stage].initial_liquid_amount)))
                for stage in STAGE_NAMES
            ),
            "urea_max_calcium_change": float(
                np.max(
                    np.abs(
                        scenario.stages["urea"].final_calcium
                        - scenario.stages["urea"].initial_calcium
                    )
                )
            ),
        }
    grid_rows, time_rows = convergence_validation(values, baseline)
    write_dict_csv(output / "grid_convergence.csv", grid_rows)
    write_dict_csv(output / "time_step_convergence.csv", time_rows)
    scm = scm_analytical_validation(baseline)
    max_mass_residual = max(
        item["liquid_residual_relative_to_inflow"]
        for scenario_values in mass_balance.values()
        for item in scenario_values.values()
    )
    baseline_grid_error = abs(
        next(row for row in grid_rows if row["case"] == "baseline")[
            "resistance_relative_difference_from_fine"
        ]
    )
    checks = {
        "all_solvers_succeeded": all(
            scenario.stages[stage].solver_success
            for scenario in scenarios
            for stage in STAGE_NAMES
        ),
        "liquid_mass_balance_relative_error_below_1e-5": max_mass_residual < 1e-5,
        "stage_calcium_continuity_and_rinse_reset": all(
            item["acid_to_edta_max_calcium_jump"] < 1e-8
            and item["edta_to_urea_max_calcium_jump"] < 1e-8
            and item["max_stage_initial_liquid_amount"] == 0.0
            for item in continuity.values()
        ),
        "urea_does_not_consume_calcium": all(
            item["urea_max_calcium_change"] < 1e-8 for item in continuity.values()
        ),
        "physical_nonnegative_state_within_solver_tolerance": minima["concentration_mol_per_m3"] > -1e-5
        and minima["calcium_mol_per_m3"] > -1e-6
        and minima["porosity"] > 0,
        "baseline_grid_resistance_difference_from_fine_below_2_percent": baseline_grid_error < 0.02,
        "outer_scm_analytical_error_below_1e-6": scm["outer_table_front_relative_error"] < 1e-6,
        "inner_scm_analytical_error_below_1e-6": scm["inner_table_front_relative_error"] < 1e-6,
    }
    return {
        "model_status": "THEORETICAL_PRIOR_NOT_EXPERIMENTALLY_CALIBRATED",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "checks": checks,
        "all_numerical_checks_passed": all(checks.values()),
        "state_minima": minima,
        "mass_balance": mass_balance,
        "stage_transition_checks": continuity,
        "scm_analytical_validation": scm,
        "baseline_grid_resistance_relative_difference_from_fine": baseline_grid_error,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parameters", type=Path, default=ROOT / "parameters" / "theoretical_prior.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    parser.add_argument("--skip-convergence", action="store_true", help="Skip extra coarse/fine convergence runs.")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    parameter_document = load_parameters(args.parameters)
    values = parameter_document["values"]
    doses = values["scenarios"]["edta_concentrations_mol_per_l"]
    scenarios = [simulate_scenario(values, dose) for dose in doses]
    baseline = min(scenarios, key=lambda item: abs(item.edta_concentration_mol_per_l - 0.1))

    export_parameter_table(parameter_document, args.output / "parameter_prior_table.csv")
    export_profiles(scenarios, args.output / "spatial_temporal_profiles.csv")
    summary = export_summary(scenarios, args.output)
    diagnostics = [row for scenario in scenarios for row in thiele_biot_rows(scenario)]
    write_dict_csv(args.output / "thiele_biot_scm_diagnostics.csv", diagnostics)

    plot_model_schematic(values, args.output / "Fig1_model_and_sequential_protocol.png")
    plot_three_stage_results(scenarios, args.output / "Fig2_three_stage_theoretical_results.png")
    plot_dose_and_resistance(scenarios, args.output / "Fig3_EDTA_dose_and_resistance.png")

    if args.skip_convergence:
        validation = {
            "model_status": "THEORETICAL_PRIOR_NOT_EXPERIMENTALLY_CALIBRATED",
            "checks": {"convergence_skipped_by_user": True},
            "all_numerical_checks_passed": False,
        }
    else:
        validation = build_validation(scenarios, baseline, values, args.output)
    with (args.output / "validation.json").open("w", encoding="utf-8") as handle:
        json.dump(validation, handle, indent=2, ensure_ascii=False)

    concise = {
        "warning": "All SD-rat values are theoretical priors pending in-vitro calibration; these are not measured experimental outcomes.",
        "native_area_resistance_ohm_m2": baseline.native_area_resistance_ohm_m2,
        "scenarios": [
            {
                "edta_concentration_mol_per_l": scenario.edta_concentration_mol_per_l,
                "whole_skull_calcium_remaining_after_edta": layer_calcium_remaining_fraction(scenario, "edta"),
                "area_resistance_ratio_after_edta": scenario.area_resistance(scenario.stages["edta"].final_calcium) / scenario.native_area_resistance_ohm_m2,
                "area_resistance_ratio_after_urea": scenario.area_resistance(scenario.stages["urea"].final_calcium) / scenario.native_area_resistance_ohm_m2,
            }
            for scenario in scenarios
        ],
        "output_summary_rows": len(summary),
    }
    with (args.output / "theoretical_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(concise, handle, indent=2, ensure_ascii=False)
    print(json.dumps(concise, indent=2, ensure_ascii=False))
    print(f"Results written to {args.output}")
    if not args.skip_convergence:
        print(f"All numerical checks passed: {validation['all_numerical_checks_passed']}")


if __name__ == "__main__":
    main()
