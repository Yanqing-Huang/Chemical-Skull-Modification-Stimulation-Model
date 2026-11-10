"""Track B v4 near-surface and optional exposure-dynamic observation models."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import expit

from cranial_model.equivalent_calibration import CONDITION_KEYS, INTERFACE_EXPONENT_RANGE
from cranial_model.multifrequency_calibration import (
    CandidateScores,
    FREQUENCIES_HZ,
    _scores_for_rats,
    _tolerance_vector,
    loo_summary,
    _loo_frequency_summary,
)
from cranial_model.near_surface import combine_structure_state


WEIGHT_COLUMNS = tuple(f"weight_{f:g}hz" for f in FREQUENCIES_HZ)
EXPOSURE_LOGIT_SLOPE_RANGE = (-6.0, 6.0)


def sample_near_surface_lhs(sample_count: int, seed: int, dynamic: bool = False) -> pd.DataFrame:
    """LHS for four independent weights, shared m, and optional shared slope."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    dimensions = 6 if dynamic else 5
    strata = []
    for _ in range(dimensions):
        values = (np.arange(sample_count) + rng.random(sample_count)) / sample_count
        rng.shuffle(values)
        strata.append(values)
    result = pd.DataFrame({column: strata[index] for index, column in enumerate(WEIGHT_COLUMNS)})
    lower, upper = np.log10(INTERFACE_EXPONENT_RANGE)
    result["interface_exponent"] = 10.0 ** (lower + strata[4] * (upper - lower))
    if dynamic:
        low, high = EXPOSURE_LOGIT_SLOPE_RANGE
        result["exposure_logit_slope"] = low + strata[5] * (high - low)
    return result


def near_surface_forward(
    matrix_ratio: np.ndarray,
    interface_ratio: np.ndarray,
    relative_exposure: np.ndarray,
    weights: np.ndarray,
    exponent: np.ndarray,
    exposure_logit_slope: np.ndarray | None = None,
) -> np.ndarray:
    """Return candidate-by-frequency-by-condition magnitude-ratio predictions."""
    matrix = np.asarray(matrix_ratio, dtype=float).reshape(-1)
    interface = np.asarray(interface_ratio, dtype=float).reshape(-1)
    exposure = np.asarray(relative_exposure, dtype=float).reshape(-1)
    weight = np.asarray(weights, dtype=float)
    power = np.asarray(exponent, dtype=float).reshape(-1)
    if not (matrix.shape == interface.shape == exposure.shape) or matrix.size == 0:
        raise ValueError("Mechanism state arrays must be nonempty and aligned")
    if weight.ndim != 2 or weight.shape[1] != len(FREQUENCIES_HZ) or len(weight) != len(power):
        raise ValueError("weights must be candidate-by-four-frequency and align with exponent")
    if np.any(interface <= 0) or np.any(power <= 0) or np.any(exposure < 0):
        raise ValueError("Invalid interface, exponent, or exposure state")
    if exposure_logit_slope is None:
        effective_weight = weight[:, :, None]
    else:
        slope = np.asarray(exposure_logit_slope, dtype=float).reshape(-1)
        if len(slope) != len(power):
            raise ValueError("Exposure slope must align with candidates")
        logits = np.log(weight / (1.0 - weight))
        effective_weight = expit(logits[:, :, None] + slope[:, None, None] * exposure[None, None, :])
    interface_response = interface[None, None, :] ** (-power[:, None, None])
    return (1.0 - effective_weight) * matrix[None, None, :] + effective_weight * interface_response


def _condition_alignment(predictions: pd.DataFrame, targets: pd.DataFrame, structure: str):
    required = {"mechanism_id", "sim_whole_resistance_ratio", *CONDITION_KEYS}
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing prediction columns: {sorted(missing)}")
    conditions = targets[CONDITION_KEYS].drop_duplicates().reset_index(drop=True)
    conditions["_condition"] = np.arange(len(conditions))
    target_keys = targets[CONDITION_KEYS].copy()
    target_keys["_target_row"] = np.arange(len(target_keys))
    target_map = target_keys.merge(
        conditions, on=CONDITION_KEYS, how="left", validate="many_to_one", sort=False
    ).sort_values("_target_row")
    condition_index = target_map["_condition"].to_numpy(dtype=int)
    merged = predictions.merge(
        conditions, on=CONDITION_KEYS, how="right", validate="many_to_one", sort=False
    )
    merged["sim_interface"] = combine_structure_state(merged, structure, "porosity_ratio")
    merged["sim_exposure"] = combine_structure_state(merged, structure, "exposure")
    matrices = []
    for column in ("sim_whole_resistance_ratio", "sim_interface", "sim_exposure"):
        matrix = merged.pivot(index="mechanism_id", columns="_condition", values=column)
        matrix = matrix.reindex(columns=np.arange(len(conditions)))
        if matrix.isna().any().any():
            raise ValueError(f"Missing {column} for a mechanism/target condition")
        matrices.append(matrix)
    ids = matrices[0].index.to_numpy()
    if not all(np.array_equal(ids, matrix.index.to_numpy()) for matrix in matrices[1:]):
        raise ValueError("Mechanism IDs are not aligned")
    return condition_index, *(matrix.to_numpy(dtype=float) for matrix in matrices), ids


def _validate_parameters(parameters: pd.DataFrame, dynamic: bool) -> None:
    required = {"candidate_id", "mechanism_id", *WEIGHT_COLUMNS, "interface_exponent"}
    if dynamic:
        required.add("exposure_logit_slope")
    missing = required.difference(parameters.columns)
    if missing:
        raise ValueError(f"Missing observation parameters: {sorted(missing)}")
    if parameters["candidate_id"].duplicated().any():
        raise ValueError("candidate_id must be unique")


def _forward_group(group: pd.DataFrame, matrix: np.ndarray, interface: np.ndarray,
                   exposure: np.ndarray, dynamic: bool) -> np.ndarray:
    return near_surface_forward(
        matrix, interface, exposure,
        group.loc[:, list(WEIGHT_COLUMNS)].to_numpy(dtype=float),
        group["interface_exponent"].to_numpy(dtype=float),
        group["exposure_logit_slope"].to_numpy(dtype=float) if dynamic else None,
    )


def score_near_surface_candidates(
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
    targets: pd.DataFrame,
    tolerances: dict[float, float],
    structure: str,
    dynamic: bool = False,
) -> CandidateScores:
    """Score v4 candidates with the unchanged v3 rat/frequency sufficient statistics."""
    _validate_parameters(parameters, dynamic)
    parameters = parameters.reset_index(drop=True).copy()
    condition_index, matrix, interface, exposure, prediction_ids = _condition_alignment(
        predictions, targets, structure
    )
    mechanism_ids = parameters["mechanism_id"].unique()
    missing = set(mechanism_ids).difference(prediction_ids)
    if missing:
        raise ValueError(f"Missing mechanism trajectories: {sorted(missing)}")
    rat_ids = np.array(sorted(targets["rat"].unique()))
    frequencies = np.array(sorted(targets["frequency_hz"].unique()), dtype=float)
    _tolerance_vector(tolerances, frequencies)
    rat_lookup = {rat: index for index, rat in enumerate(rat_ids)}
    frequency_lookup = {float(f): index for index, f in enumerate(frequencies)}
    rat_index = targets["rat"].map(rat_lookup).to_numpy(dtype=int)
    frequency_index = targets["frequency_hz"].astype(float).map(frequency_lookup).to_numpy(dtype=int)
    group_index = rat_index * len(frequencies) + frequency_index
    counts = np.bincount(
        group_index, minlength=len(rat_ids) * len(frequencies)
    ).reshape(len(rat_ids), len(frequencies))
    order = np.argsort(group_index, kind="stable")
    sorted_group = group_index[order]
    observed = targets["z_ratio"].to_numpy(dtype=float)[order]
    condition_index = condition_index[order]
    frequency_index = frequency_index[order]
    nonzero_groups = np.flatnonzero(counts.reshape(-1))
    starts = np.searchsorted(sorted_group, nonzero_groups)
    squared_error = np.zeros((len(parameters), len(rat_ids), len(frequencies)), dtype=float)
    mechanism_values = parameters["mechanism_id"].to_numpy()
    prediction_lookup = {mechanism: index for index, mechanism in enumerate(prediction_ids)}
    for mechanism in mechanism_ids:
        positions = np.flatnonzero(mechanism_values == mechanism)
        mechanism_index = prediction_lookup[mechanism]
        values = _forward_group(
            parameters.iloc[positions], matrix[mechanism_index], interface[mechanism_index],
            exposure[mechanism_index], dynamic,
        )
        predicted = values[:, frequency_index, condition_index]
        residual_squared = (predicted - observed[None, :]) ** 2
        reduced = np.add.reduceat(residual_squared, starts, axis=1)
        flat_error = squared_error.reshape(len(parameters), -1)
        flat_error[np.ix_(positions, nonzero_groups)] = reduced
    result = CandidateScores(
        metrics=parameters, squared_error=squared_error, rat_ids=rat_ids,
        frequencies=frequencies, counts=counts,
    )
    normalized, raw, compatible = _scores_for_rats(
        result, np.ones(len(rat_ids), dtype=bool), tolerances
    )
    result.metrics["normalized_rmse"] = normalized
    for index, frequency in enumerate(frequencies):
        result.metrics[f"raw_rmse_{frequency:g}hz"] = raw[:, index]
        result.metrics[f"pass_{frequency:g}hz"] = raw[:, index] <= tolerances[float(frequency)]
    result.metrics["compatible"] = compatible
    return result


def prediction_quantiles(
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
    candidate_ids: np.ndarray,
    rows: pd.DataFrame,
    structure: str,
    dynamic: bool = False,
) -> pd.DataFrame:
    """Exact compatible-set 5/50/95% parameter-envelope quantiles."""
    _validate_parameters(parameters, dynamic)
    candidate_ids = np.asarray(candidate_ids)
    selected = parameters[parameters["candidate_id"].isin(candidate_ids)]
    if selected.empty:
        raise ValueError("Empty compatible set: no prediction interval available")
    if len(selected) != len(np.unique(candidate_ids)):
        raise ValueError("Unknown candidate ID requested")
    condition_index, matrix, interface, exposure, prediction_ids = _condition_alignment(
        predictions, rows, structure
    )
    prediction_lookup = {mechanism: index for index, mechanism in enumerate(prediction_ids)}
    frequencies = np.array(sorted(rows["frequency_hz"].unique()), dtype=float)
    frequency_lookup = {float(f): index for index, f in enumerate(frequencies)}
    frequency_index = rows["frequency_hz"].astype(float).map(frequency_lookup).to_numpy(dtype=int)
    quantiles = np.empty((3, len(rows)))
    for start in range(0, len(rows), 32):
        stop = min(start + 32, len(rows))
        block_condition = condition_index[start:stop]
        block_frequency = frequency_index[start:stop]
        parts = []
        for mechanism, group in selected.groupby("mechanism_id", sort=True):
            mechanism_index = prediction_lookup[mechanism]
            values = _forward_group(
                group, matrix[mechanism_index], interface[mechanism_index],
                exposure[mechanism_index], dynamic,
            )
            parts.append(values[:, block_frequency, block_condition])
        chosen = np.concatenate(parts, axis=0)
        quantiles[:, start:stop] = np.quantile(chosen, [0.05, 0.5, 0.95], axis=0)
    result = rows.copy().reset_index(drop=True)
    result["sim_q05"], result["sim_q50"], result["sim_q95"] = quantiles
    result["compatible_count"] = int(len(selected))
    result["degenerate_interval"] = len(selected) == 1
    return result


def leave_one_rat_out(
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
    targets: pd.DataFrame,
    scores: CandidateScores,
    tolerances_function,
    structure: str,
    dynamic: bool = False,
):
    """Refit tolerances and the compatible set independently in every training fold."""
    rat_rows, point_rows = [], []
    for held_out in scores.rat_ids:
        train = targets[targets["rat"] != held_out]
        test = targets[targets["rat"] == held_out].copy().reset_index(drop=True)
        tolerances = tolerances_function(train)
        rat_mask = scores.rat_ids != held_out
        normalized, raw, compatible = _scores_for_rats(scores, rat_mask, tolerances)
        selected = scores.candidate_ids[compatible]
        rat_row = {
            "rat": held_out, "compatible_count": int(len(selected)), "n_points": int(len(test)),
            "best_training_normalized_rmse": float(np.min(normalized)),
            "coverage": np.nan, "median_width": np.nan,
            "degenerate_interval": len(selected) == 1,
        }
        for index, frequency in enumerate(scores.frequencies):
            rat_row[f"tolerance_{frequency:g}hz"] = tolerances[float(frequency)]
            rat_row[f"best_training_raw_rmse_{frequency:g}hz"] = float(raw[np.argmin(normalized), index])
        if len(selected):
            intervals = prediction_quantiles(
                predictions, parameters, selected, test, structure, dynamic
            )
            covered = (
                (intervals["z_ratio"] >= intervals["sim_q05"])
                & (intervals["z_ratio"] <= intervals["sim_q95"])
            )
            intervals["covered"] = covered.astype("boolean")
            intervals["width"] = intervals["sim_q95"] - intervals["sim_q05"]
            intervals["prediction_available"] = True
            rat_row["coverage"] = float(covered.mean())
            rat_row["median_width"] = float(intervals["width"].median())
            point_rows.append(intervals)
        else:
            intervals = test
            intervals["sim_q05"] = np.nan
            intervals["sim_q50"] = np.nan
            intervals["sim_q95"] = np.nan
            intervals["compatible_count"] = 0
            intervals["degenerate_interval"] = False
            intervals["covered"] = pd.Series(pd.NA, index=intervals.index, dtype="boolean")
            intervals["width"] = np.nan
            intervals["prediction_available"] = False
            point_rows.append(intervals)
        rat_rows.append(rat_row)
    rat_summary = pd.DataFrame(rat_rows)
    points = pd.concat(point_rows, ignore_index=True)
    frequency_summary = _loo_frequency_summary(rat_summary, points)
    return rat_summary, points, frequency_summary, loo_summary(rat_summary, points)
