"""Import in-vitro observations and calibrate declared theoretical-prior parameters."""

from __future__ import annotations

import copy
import csv
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import least_squares

from .core import LAYER_NAMES, ScenarioResult, load_parameters, simulate_scenario


OBSERVABLES = {
    "ca_remaining_fraction",
    "ca_release_mol_per_m2",
    "porosity",
    "area_resistance_ohm_m2",
    "area_resistance_ratio_to_native",
}


def read_experimental_data(path: str | Path) -> list[dict]:
    with Path(path).open("r", newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("Experimental CSV contains no observations.")
    required = {
        "experiment_id",
        "edta_concentration_mol_per_l",
        "stage",
        "time_s",
        "layer",
        "observable",
        "value",
        "sd",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"Missing experimental columns: {sorted(missing)}")
    parsed = []
    for row_number, row in enumerate(rows, start=2):
        try:
            observable = row["observable"].strip()
            if observable not in OBSERVABLES:
                raise ValueError(f"observable must be one of {sorted(OBSERVABLES)}")
            stage = row["stage"].strip()
            if stage not in ("acetic_acid", "edta", "urea"):
                raise ValueError("stage must be acetic_acid, edta or urea")
            layer = row["layer"].strip()
            if layer != "whole_skull" and layer not in LAYER_NAMES:
                raise ValueError("layer must be whole_skull or a valid layer name")
            sd = float(row["sd"]) if row["sd"].strip() else 1.0
            if sd <= 0:
                raise ValueError("sd must be positive or blank")
            parsed.append(
                {
                    **row,
                    "edta_concentration_mol_per_l": float(row["edta_concentration_mol_per_l"]),
                    "stage": stage,
                    "time_s": float(row["time_s"]),
                    "layer": layer,
                    "observable": observable,
                    "value": float(row["value"]),
                    "sd": sd,
                }
            )
        except (TypeError, ValueError) as error:
            raise ValueError(f"Invalid experimental row {row_number}: {error}") from error
    return parsed


def get_value_path(values: dict, path: str) -> float:
    target = values
    for component in path.split("."):
        target = target[component]
    return float(target)


def set_value_path(values: dict, path: str, value: float) -> None:
    target = values
    components = path.split(".")
    for component in components[:-1]:
        target = target[component]
    target[components[-1]] = float(value)


def _selected_slice(scenario: ScenarioResult, layer: str) -> slice:
    if layer == "whole_skull":
        return slice(None)
    return scenario.grid.layer_slices[LAYER_NAMES.index(layer)]


def predict_observation(scenario: ScenarioResult, observation: dict) -> float:
    result = scenario.stages[observation["stage"]]
    time_s = observation["time_s"]
    if time_s < 0 or time_s > result.time_s[-1] + 1e-9:
        raise ValueError(
            f"Observation time {time_s} s is outside {observation['stage']} duration {result.time_s[-1]} s."
        )
    grid = scenario.grid
    selected = _selected_slice(scenario, observation["layer"])
    widths = grid.widths[selected]
    calcium = np.array(
        [np.interp(time_s, result.time_s, result.calcium[:, cell]) for cell in range(grid.centers.size)]
    )
    observable = observation["observable"]
    if observable == "ca_remaining_fraction":
        return float(
            np.sum(calcium[selected] * widths)
            / np.sum(grid.q_reference[selected] * widths)
        )
    if observable == "ca_release_mol_per_m2":
        return float(np.sum((grid.q_reference[selected] - calcium[selected]) * widths))
    if observable == "porosity":
        porosity = np.array(
            [np.interp(time_s, result.time_s, result.porosity[:, cell]) for cell in range(grid.centers.size)]
        )
        return float(np.average(porosity[selected], weights=widths))
    resistance = scenario.area_resistance(calcium)
    if observable == "area_resistance_ohm_m2":
        return resistance
    if observable == "area_resistance_ratio_to_native":
        return resistance / scenario.native_area_resistance_ohm_m2
    raise ValueError(f"Unsupported observable: {observable}")


def predict_observations(values: dict, observations: list[dict]) -> np.ndarray:
    doses = sorted({row["edta_concentration_mol_per_l"] for row in observations})
    scenarios = {dose: simulate_scenario(values, dose) for dose in doses}
    return np.array(
        [
            predict_observation(scenarios[row["edta_concentration_mol_per_l"]], row)
            for row in observations
        ]
    )


def fit_parameters(
    parameter_document: dict,
    observations: list[dict],
    fit_paths: Iterable[str],
    cells_per_layer: list[int] | None = None,
    max_step_s: float | None = None,
    max_nfev: int = 100,
) -> tuple[dict, dict, np.ndarray, np.ndarray]:
    """Fit positive parameters in log10 space and return prior/calibrated predictions."""
    fit_paths = list(fit_paths)
    if not fit_paths:
        raise ValueError("At least one parameter path must be supplied for calibration.")
    base_values = copy.deepcopy(parameter_document["values"])
    if cells_per_layer is not None:
        base_values["numerics"]["cells_per_layer"] = cells_per_layer
    if max_step_s is not None:
        base_values["numerics"]["max_step_s"] = max_step_s
    initial = np.array([get_value_path(base_values, path) for path in fit_paths])
    if np.any(initial <= 0):
        raise ValueError("This calibration interface supports positive parameters only.")
    lower = []
    upper = []
    for path in fit_paths:
        if path not in parameter_document["uncertainty"]:
            raise ValueError(f"No declared calibration range for {path}")
        bounds = parameter_document["uncertainty"][path]["range"]
        lower.append(np.log10(bounds[0]))
        upper.append(np.log10(bounds[1]))
    observed = np.array([row["value"] for row in observations])
    sigma = np.array([row["sd"] for row in observations])
    prior_prediction = predict_observations(base_values, observations)

    def residual(log_parameters: np.ndarray) -> np.ndarray:
        values = copy.deepcopy(base_values)
        for path, log_value in zip(fit_paths, log_parameters):
            set_value_path(values, path, 10.0**log_value)
        prediction = predict_observations(values, observations)
        return (prediction - observed) / sigma

    fit = least_squares(
        residual,
        np.log10(initial),
        bounds=(np.array(lower), np.array(upper)),
        diff_step=1e-3,
        max_nfev=max_nfev,
    )
    calibrated_document = copy.deepcopy(parameter_document)
    calibrated_document["model_status"] = "CALIBRATED_TO_SUPPLIED_IN_VITRO_DATA_NOT_INDEPENDENTLY_VALIDATED"
    calibrated_document["values"] = copy.deepcopy(base_values)
    for path, log_value in zip(fit_paths, fit.x):
        set_value_path(calibrated_document["values"], path, 10.0**log_value)
    calibrated_prediction = predict_observations(calibrated_document["values"], observations)
    metadata = {
        "status": "CALIBRATED_TO_SUPPLIED_IN_VITRO_DATA",
        "fit_paths": fit_paths,
        "fitted_values": {
            path: get_value_path(calibrated_document["values"], path) for path in fit_paths
        },
        "prior_weighted_rmse": float(np.sqrt(np.mean(((prior_prediction - observed) / sigma) ** 2))),
        "calibrated_weighted_rmse": float(np.sqrt(np.mean(((calibrated_prediction - observed) / sigma) ** 2))),
        "prior_unweighted_rmse": float(np.sqrt(np.mean((prior_prediction - observed) ** 2))),
        "calibrated_unweighted_rmse": float(np.sqrt(np.mean((calibrated_prediction - observed) ** 2))),
        "least_squares_cost": float(fit.cost),
        "least_squares_optimality": float(fit.optimality),
        "function_evaluations": int(fit.nfev),
        "solver_message": fit.message,
        "cells_per_layer": calibrated_document["values"]["numerics"]["cells_per_layer"],
        "max_step_s": calibrated_document["values"]["numerics"]["max_step_s"],
        "caution": "Calibration is not independent validation; retain a holdout experiment for prospective checking.",
    }
    calibrated_document["calibration_metadata"] = metadata
    return calibrated_document, metadata, prior_prediction, calibrated_prediction


def export_calibration_predictions(
    path: str | Path,
    observations: list[dict],
    prior: np.ndarray,
    calibrated: np.ndarray,
) -> None:
    fields = [
        "experiment_id",
        "edta_concentration_mol_per_l",
        "stage",
        "time_s",
        "layer",
        "observable",
        "observed",
        "sd",
        "prior_predicted",
        "calibrated_predicted",
        "prior_residual",
        "calibrated_residual",
    ]
    with Path(path).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row, prior_value, calibrated_value in zip(observations, prior, calibrated):
            writer.writerow(
                {
                    "experiment_id": row["experiment_id"],
                    "edta_concentration_mol_per_l": row["edta_concentration_mol_per_l"],
                    "stage": row["stage"],
                    "time_s": row["time_s"],
                    "layer": row["layer"],
                    "observable": row["observable"],
                    "observed": row["value"],
                    "sd": row["sd"],
                    "prior_predicted": prior_value,
                    "calibrated_predicted": calibrated_value,
                    "prior_residual": prior_value - row["value"],
                    "calibrated_residual": calibrated_value - row["value"],
                }
            )
