"""Prior-interval sampling and ensemble prediction with fixed skull properties."""

from __future__ import annotations

import copy
import multiprocessing
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from .calibration import predict_observation
from .core import simulate_scenario


PARAMETER_PATHS = (
    "transport.external_mass_transfer_m_per_s",
    "stages.acetic_acid.k_r_m3_per_mol_s",
    "stages.edta.k_r_m3_per_mol_s",
    "transport.porosity_per_released_calcium_m3_per_mol",
)


def set_parameter_path(values: dict, path: str, value: float) -> None:
    target = values
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[part]
    target[parts[-1]] = float(value)


def sample_lhs_log10(
    parameter_document: dict,
    sample_count: int,
    seed: int,
    paths: tuple[str, ...] = PARAMETER_PATHS,
) -> np.ndarray:
    """Latin hypercube samples in log10 space; only kinetic parameters are sampled."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    samples = np.empty((sample_count, len(paths)))
    for column, path in enumerate(paths):
        lower, upper = parameter_document["uncertainty"][path]["range"]
        if lower <= 0 or upper < lower:
            raise ValueError(f"Invalid positive prior range for {path}")
        strata = (np.arange(sample_count) + rng.random(sample_count)) / sample_count
        rng.shuffle(strata)
        samples[:, column] = 10.0 ** (
            np.log10(lower) + strata * (np.log10(upper) - np.log10(lower))
        )
    return samples


def apply_candidate(values: dict, paths: tuple[str, ...], candidate: np.ndarray) -> dict:
    updated = copy.deepcopy(values)
    for path, value in zip(paths, candidate):
        set_parameter_path(updated, path, value)
    return updated


def predict_conditions(
    values: dict,
    conditions: list[dict],
    cells_per_layer: list[int],
    max_step_s: float,
    return_scenarios: bool = False,
):
    """Run each EDTA dose once and interpolate all requested stage/time outputs."""
    predictions = np.empty(len(conditions))
    scenarios = {}
    for dose in sorted({float(row["edta_concentration_mol_per_l"]) for row in conditions}):
        scenario = simulate_scenario(
            values,
            dose,
            cells_per_layer=cells_per_layer,
            max_step_s=max_step_s,
            stage_names=("acetic_acid", "edta"),
        )
        scenarios[dose] = scenario
    for index, row in enumerate(conditions):
        scenario = scenarios[float(row["edta_concentration_mol_per_l"])]
        predictions[index] = predict_observation(
            scenario,
            {
                "stage": row["stage"],
                "time_s": float(row["time_s"]),
                "layer": "whole_skull",
                "observable": "area_resistance_ratio_to_native",
            },
        )
    if return_scenarios:
        return predictions, scenarios
    return predictions


def _evaluate_candidate(
    base_values: dict,
    candidate: np.ndarray,
    calibration_conditions: list[dict],
    observed_mean: np.ndarray,
    cells_per_layer: list[int],
    max_step_s: float,
    paths: tuple[str, ...],
) -> tuple[float, np.ndarray]:
    values = apply_candidate(base_values, paths, candidate)
    prediction = np.full(len(calibration_conditions), np.nan)
    try:
        calculated = predict_conditions(values, calibration_conditions, cells_per_layer, max_step_s)
        if np.all(np.isfinite(calculated)):
            prediction = calculated
            rmse = float(np.sqrt(np.mean((prediction - observed_mean) ** 2)))
            return rmse, prediction
    except (RuntimeError, ValueError, FloatingPointError):
        pass
    return float("nan"), prediction


def screen_candidates(
    base_values: dict,
    candidates: np.ndarray,
    calibration_conditions: list[dict],
    observed_mean: np.ndarray,
    tolerance: float,
    cells_per_layer: list[int],
    max_step_s: float,
    paths: tuple[str, ...] = PARAMETER_PATHS,
    progress: Callable[[int, int], None] | None = None,
    workers: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return calibration RMSE, accept flags and all calibration predictions."""
    observed_mean = np.asarray(observed_mean, dtype=float)
    rmse = np.full(len(candidates), np.nan)
    predictions = np.full((len(candidates), len(calibration_conditions)), np.nan)
    if workers < 1:
        raise ValueError("workers must be positive")
    if workers == 1:
        for index, candidate in enumerate(candidates):
            rmse[index], predictions[index] = _evaluate_candidate(
                base_values, candidate, calibration_conditions, observed_mean,
                cells_per_layer, max_step_s, paths,
            )
            if progress and ((index + 1) % 25 == 0 or index + 1 == len(candidates)):
                progress(index + 1, len(candidates))
    else:
        methods = multiprocessing.get_all_start_methods()
        context = multiprocessing.get_context("fork") if "fork" in methods else None
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
            futures = {
                executor.submit(
                    _evaluate_candidate, base_values, candidate, calibration_conditions,
                    observed_mean, cells_per_layer, max_step_s, paths,
                ): index
                for index, candidate in enumerate(candidates)
            }
            completed = 0
            for future in as_completed(futures):
                index = futures[future]
                rmse[index], predictions[index] = future.result()
                completed += 1
                if progress and (completed % 25 == 0 or completed == len(candidates)):
                    progress(completed, len(candidates))
    accepted = np.isfinite(rmse) & (rmse <= tolerance)
    return rmse, accepted, predictions


def prediction_interval(predictions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Full posterior-feasible envelope, not a confidence interval."""
    values = np.asarray(predictions, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("Expected a non-empty candidate-by-condition prediction array")
    return np.min(values, axis=0), np.max(values, axis=0)


def monotonic_trend_fraction(conditions: list[dict], predictions: np.ndarray) -> dict[str, float]:
    """Fraction of candidates with non-increasing dose and time responses."""
    values = np.asarray(predictions, dtype=float)
    edta_indices = [i for i, row in enumerate(conditions) if row["stage"] == "edta"]
    time_ok = np.ones(values.shape[0], dtype=bool)
    dose_ok = np.ones(values.shape[0], dtype=bool)
    doses = sorted({conditions[i]["edta_concentration_mol_per_l"] for i in edta_indices})
    times = sorted({conditions[i]["time_s"] for i in edta_indices})
    for dose in doses:
        indices = [i for i in edta_indices if conditions[i]["edta_concentration_mol_per_l"] == dose]
        indices.sort(key=lambda i: conditions[i]["time_s"])
        if len(indices) > 1:
            time_ok &= np.all(np.diff(values[:, indices], axis=1) <= 1e-10, axis=1)
    for time_s in times:
        indices = [i for i in edta_indices if conditions[i]["time_s"] == time_s]
        indices.sort(key=lambda i: conditions[i]["edta_concentration_mol_per_l"])
        if len(indices) > 1:
            dose_ok &= np.all(np.diff(values[:, indices], axis=1) <= 1e-10, axis=1)
    return {
        "time_monotonic_fraction": float(np.mean(time_ok)),
        "dose_monotonic_fraction": float(np.mean(dose_ok)),
        "both_monotonic_fraction": float(np.mean(time_ok & dose_ok)),
    }
