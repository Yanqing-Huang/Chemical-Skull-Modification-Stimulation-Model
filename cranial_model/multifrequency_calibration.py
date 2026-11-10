"""Track B v3: exploratory four-frequency magnitude-ratio calibration.

Frequency changes only the observation operator. The 200 mechanism trajectories,
including their reaction kinetics and stage-specific boundaries, are reused
unchanged from Track B v2. All fitted quantities are platform-bound composite
parameters, not intrinsic interfacial shares or hydroxyapatite kinetics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from cranial_model.equivalent_calibration import (
    CONDITION_KEYS,
    INTERFACE_EXPONENT_RANGE,
    INTERFACE_FRACTION_RANGE,
    PLATFORM_DISCLAIMER,
    build_z_ratio_targets,
)


FREQUENCIES_HZ = (100.0, 120.0, 1000.0, 10000.0)
BETA_LOW_RANGE = INTERFACE_FRACTION_RANGE
BETA_RATIO_RANGE = (0.0, 1.0)  # sampled LHS values are strictly inside this range
OBSERVATION_OPERATOR = (
    "Z_f/Z_f0 = (1-beta_f)*R_matrix/R_matrix0 + "
    "beta_f*(phi_surface0/phi_surface(t))**m"
)
BETA_INTERPOLATION = (
    "logit(beta_f) is linear in log10(f) from beta_low at 100 Hz to "
    "beta_high=beta_low*q at 10000 Hz; 0<q<1"
)


@dataclass
class CandidateScores:
    """Candidate metrics and rat/frequency squared-error sufficient statistics."""

    metrics: pd.DataFrame
    squared_error: np.ndarray
    rat_ids: np.ndarray
    frequencies: np.ndarray
    counts: np.ndarray

    @property
    def candidate_ids(self) -> np.ndarray:
        return self.metrics["candidate_id"].to_numpy()


def sample_observation_lhs(sample_count: int, seed: int) -> np.ndarray:
    """Three LHS dimensions: beta_low (linear), q (linear), and m (log)."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    strata = []
    for _ in range(3):
        values = (np.arange(sample_count) + rng.random(sample_count)) / sample_count
        rng.shuffle(values)
        strata.append(values)
    beta_low = BETA_LOW_RANGE[0] + strata[0] * (
        BETA_LOW_RANGE[1] - BETA_LOW_RANGE[0]
    )
    beta_ratio = strata[1]
    log_lower, log_upper = np.log10(INTERFACE_EXPONENT_RANGE)
    exponent = 10.0 ** (log_lower + strata[2] * (log_upper - log_lower))
    return np.column_stack([beta_low, beta_ratio, exponent])


def frequency_interface_fraction(
    beta_low: np.ndarray | float,
    beta_ratio: np.ndarray | float,
    frequencies: np.ndarray | list[float] | tuple[float, ...],
) -> np.ndarray:
    """Return candidate-by-frequency equivalent weights, monotone in frequency."""
    low = np.atleast_1d(np.asarray(beta_low, dtype=float))
    ratio = np.atleast_1d(np.asarray(beta_ratio, dtype=float))
    try:
        low, ratio = np.broadcast_arrays(low, ratio)
    except ValueError as error:
        raise ValueError("beta_low and beta_ratio must have compatible shapes") from error
    low = low.reshape(-1, 1)
    ratio = ratio.reshape(-1, 1)
    frequency = np.atleast_1d(np.asarray(frequencies, dtype=float)).reshape(1, -1)
    if (
        np.any(~np.isfinite(low))
        or np.any(~np.isfinite(ratio))
        or np.any(low <= 0)
        or np.any(low >= 1)
        or np.any(ratio <= 0)
        or np.any(ratio >= 1)
    ):
        raise ValueError("Require 0<beta_low<1 and 0<beta_ratio<1")
    if np.any(frequency < FREQUENCIES_HZ[0]) or np.any(frequency > FREQUENCIES_HZ[-1]):
        raise ValueError("Frequency interpolation is defined only from 100 to 10000 Hz")
    high = low * ratio
    logit_low = np.log(low / (1.0 - low))
    logit_high = np.log(high / (1.0 - high))
    fraction = (
        np.log10(frequency) - np.log10(FREQUENCIES_HZ[0])
    ) / (np.log10(FREQUENCIES_HZ[-1]) - np.log10(FREQUENCIES_HZ[0]))
    logits = (1.0 - fraction) * logit_low + fraction * logit_high
    return 1.0 / (1.0 + np.exp(-logits))


def multifrequency_forward(
    matrix_ratio: np.ndarray,
    surface_porosity_ratio: np.ndarray,
    beta_low: np.ndarray,
    beta_ratio: np.ndarray,
    exponent: np.ndarray,
    frequencies: np.ndarray | list[float] | tuple[float, ...] = FREQUENCIES_HZ,
) -> np.ndarray:
    """Return predictions with shape (candidate, frequency, condition)."""
    matrix = np.asarray(matrix_ratio, dtype=float).reshape(-1)
    surface = np.asarray(surface_porosity_ratio, dtype=float).reshape(-1)
    if matrix.shape != surface.shape or matrix.size == 0:
        raise ValueError("matrix_ratio and surface_porosity_ratio must be nonempty and aligned")
    if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(surface)) or np.any(surface <= 0):
        raise ValueError("Mechanism ratios must be finite and surface porosity must be positive")
    low = np.asarray(beta_low, dtype=float).reshape(-1)
    ratio = np.asarray(beta_ratio, dtype=float).reshape(-1)
    power = np.asarray(exponent, dtype=float).reshape(-1)
    if not (len(low) == len(ratio) == len(power)):
        raise ValueError("Observation parameter arrays must have equal lengths")
    if np.any(~np.isfinite(power)) or np.any(power <= 0):
        raise ValueError("Interface exponents must be positive and finite")
    beta = frequency_interface_fraction(low, ratio, frequencies)
    interface = surface[None, None, :] ** (-power[:, None, None])
    return (
        (1.0 - beta[:, :, None]) * matrix[None, None, :]
        + beta[:, :, None] * interface
    )


def build_multifrequency_targets(
    spectra: pd.DataFrame,
    frequencies: tuple[float, ...] = FREQUENCIES_HZ,
    drop_confounded: bool = True,
    retain_confounded_rats: tuple[str, ...] = (),
) -> pd.DataFrame:
    """Build rat/condition/frequency targets, averaging sites but never frequencies."""
    pieces = []
    for frequency in frequencies:
        part = build_z_ratio_targets(
            spectra,
            frequencies=(frequency,),
            drop_confounded=drop_confounded,
            retain_confounded_rats=retain_confounded_rats,
        )
        part["frequency_hz"] = float(frequency)
        pieces.append(part)
    targets = pd.concat(pieces, ignore_index=True)
    keys = ["rat", "frequency_hz", *CONDITION_KEYS]
    if targets.duplicated(keys).any():
        raise ValueError("Duplicate rat/condition/frequency target")
    return targets.sort_values(["frequency_hz", *CONDITION_KEYS, "rat"]).reset_index(drop=True)


def condition_sd_by_frequency(train: pd.DataFrame) -> pd.DataFrame:
    """Condition SDs; acetic-acid noise pools future EDTA dose groups only."""
    rows = []
    for frequency, frequency_frame in train.groupby("frequency_hz", sort=True):
        acetic = frequency_frame[frequency_frame["stage"] == "acetic_acid"]
        for (stage, time_s), group in acetic.groupby(["stage", "time_s"], sort=True):
            rows.append({
                "frequency_hz": float(frequency),
                "stage": stage,
                "edta_concentration_mol_per_l": np.nan,
                "time_s": float(time_s),
                "rat_count": int(group["rat"].nunique()),
                "sd": float(group["z_ratio"].std(ddof=1)) if len(group) > 1 else np.nan,
                "noise_pool": "acetic_acid_across_future_edta_doses",
            })
        edta = frequency_frame[frequency_frame["stage"] == "edta"]
        for key, group in edta.groupby(CONDITION_KEYS, sort=True):
            stage, dose, time_s = key
            rows.append({
                "frequency_hz": float(frequency),
                "stage": stage,
                "edta_concentration_mol_per_l": float(dose),
                "time_s": float(time_s),
                "rat_count": int(group["rat"].nunique()),
                "sd": float(group["z_ratio"].std(ddof=1)) if len(group) > 1 else np.nan,
                "noise_pool": "edta_within_condition",
            })
    return pd.DataFrame(rows)


def frequency_tolerances(train: pd.DataFrame) -> dict[float, float]:
    """RMS of available between-rat condition SDs independently at each frequency."""
    condition_sd = condition_sd_by_frequency(train)
    tolerances = {}
    for frequency, group in condition_sd.groupby("frequency_hz", sort=True):
        sd = group["sd"].dropna().to_numpy(dtype=float)
        if sd.size == 0:
            raise ValueError(f"No multi-rat condition at {frequency:g} Hz")
        tolerance = float(np.sqrt(np.mean(sd**2)))
        if not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError(f"Tolerance at {frequency:g} Hz must be positive and finite")
        tolerances[float(frequency)] = tolerance
    return tolerances


def _condition_alignment(
    predictions: pd.DataFrame, targets: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {
        "mechanism_id", "sim_whole_resistance_ratio",
        "sim_outer_surface_porosity_ratio", *CONDITION_KEYS,
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing mechanism prediction columns: {sorted(missing)}")
    if predictions.duplicated(["mechanism_id", *CONDITION_KEYS]).any():
        raise ValueError("Mechanism predictions must be unique per condition")
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
    matrices = []
    for column in ("sim_whole_resistance_ratio", "sim_outer_surface_porosity_ratio"):
        matrix = merged.pivot(index="mechanism_id", columns="_condition", values=column)
        matrix = matrix.reindex(columns=np.arange(len(conditions)))
        if matrix.isna().any().any():
            raise ValueError(f"Missing {column} for a mechanism/target condition")
        matrices.append(matrix)
    mechanism_ids = matrices[0].index.to_numpy()
    if not np.array_equal(mechanism_ids, matrices[1].index.to_numpy()):
        raise ValueError("Matrix and surface mechanism IDs are not aligned")
    return (
        condition_index,
        matrices[0].to_numpy(dtype=float),
        matrices[1].to_numpy(dtype=float),
        mechanism_ids,
    )


def _validate_parameters(parameters: pd.DataFrame) -> None:
    required = {"candidate_id", "mechanism_id", "beta_low", "beta_ratio", "interface_exponent"}
    missing = required.difference(parameters.columns)
    if missing:
        raise ValueError(f"Missing observation parameter columns: {sorted(missing)}")
    if parameters["candidate_id"].duplicated().any():
        raise ValueError("candidate_id must be unique")


def _tolerance_vector(tolerances: dict[float, float], frequencies: np.ndarray) -> np.ndarray:
    try:
        vector = np.array([tolerances[float(f)] for f in frequencies], dtype=float)
    except KeyError as error:
        raise ValueError("A tolerance is required for every target frequency") from error
    if np.any(~np.isfinite(vector)) or np.any(vector <= 0):
        raise ValueError("Tolerances must be positive and finite")
    return vector


def _scores_for_rats(
    scores: CandidateScores,
    rat_mask: np.ndarray,
    tolerances: dict[float, float],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    tolerance = _tolerance_vector(tolerances, scores.frequencies)
    squared = scores.squared_error[:, rat_mask, :]
    counts = scores.counts[rat_mask, :]
    count_by_frequency = counts.sum(axis=0)
    raw_rmse = np.sqrt(squared.sum(axis=1) / count_by_frequency[None, :])
    total_count = int(counts.sum())
    normalized = np.sqrt(
        (squared / tolerance[None, None, :] ** 2).sum(axis=(1, 2)) / total_count
    )
    compatible = (normalized <= 1.0) & np.all(raw_rmse <= tolerance[None, :], axis=1)
    return normalized, raw_rmse, compatible


def score_multifrequency_candidates(
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
    targets: pd.DataFrame,
    tolerances: dict[float, float],
    chunk_size: int = 16,
    matrix_only: bool = False,
) -> CandidateScores:
    """Score in mechanism blocks without storing a candidate-by-target array."""
    _validate_parameters(parameters)
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    condition_index, matrix, surface, prediction_ids = _condition_alignment(predictions, targets)
    parameters = parameters.reset_index(drop=True).copy()
    mechanism_ids = parameters["mechanism_id"].unique()
    missing_mechanisms = set(mechanism_ids).difference(prediction_ids)
    if missing_mechanisms:
        raise ValueError(f"Missing mechanism trajectories: {sorted(missing_mechanisms)}")
    rat_ids = np.array(sorted(targets["rat"].unique()))
    frequencies = np.array(sorted(targets["frequency_hz"].unique()), dtype=float)
    tolerance = _tolerance_vector(tolerances, frequencies)
    rat_lookup = {rat: index for index, rat in enumerate(rat_ids)}
    frequency_lookup = {float(f): index for index, f in enumerate(frequencies)}
    rat_index = targets["rat"].map(rat_lookup).to_numpy(dtype=int)
    frequency_index = targets["frequency_hz"].astype(float).map(frequency_lookup).to_numpy(dtype=int)
    if np.any(pd.isna(frequency_index)):
        raise ValueError("Unknown target frequency")
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
    squared_error = np.zeros(
        (len(parameters), len(rat_ids), len(frequencies)), dtype=float
    )
    mechanism_values = parameters["mechanism_id"].to_numpy()
    prediction_lookup = {mechanism: index for index, mechanism in enumerate(prediction_ids)}
    for block_start in range(0, len(mechanism_ids), chunk_size):
        block_mechanisms = mechanism_ids[block_start:block_start + chunk_size]
        for mechanism in block_mechanisms:
            positions = np.flatnonzero(mechanism_values == mechanism)
            mechanism_index = prediction_lookup[mechanism]
            matrix_row = matrix[mechanism_index, condition_index]
            if matrix_only:
                predicted = np.broadcast_to(matrix_row, (len(positions), len(matrix_row)))
            else:
                group = parameters.iloc[positions]
                surface_conditions = surface[mechanism_index]
                matrix_conditions = matrix[mechanism_index]
                values = multifrequency_forward(
                    matrix_conditions,
                    surface_conditions,
                    group["beta_low"].to_numpy(dtype=float),
                    group["beta_ratio"].to_numpy(dtype=float),
                    group["interface_exponent"].to_numpy(dtype=float),
                    frequencies,
                )
                predicted = values[:, frequency_index, condition_index]
            residual_squared = (predicted - observed[None, :]) ** 2
            reduced = np.add.reduceat(residual_squared, starts, axis=1)
            flat_error = squared_error.reshape(len(parameters), -1)
            flat_error[np.ix_(positions, nonzero_groups)] = reduced
    result = CandidateScores(
        metrics=parameters,
        squared_error=squared_error,
        rat_ids=rat_ids,
        frequencies=frequencies,
        counts=counts,
    )
    normalized, raw, compatible = _scores_for_rats(
        result, np.ones(len(rat_ids), dtype=bool), tolerances
    )
    result.metrics["normalized_rmse"] = normalized
    for index, frequency in enumerate(frequencies):
        result.metrics[f"raw_rmse_{frequency:g}hz"] = raw[:, index]
        result.metrics[f"pass_{frequency:g}hz"] = raw[:, index] <= tolerance[index]
    result.metrics["compatible"] = compatible
    return result


def matrix_only_parameters(mechanism_ids: np.ndarray) -> pd.DataFrame:
    """One matrix-only candidate per mechanism, scored under the same v3 rule."""
    mechanism_ids = np.asarray(mechanism_ids)
    return pd.DataFrame({
        "candidate_id": np.arange(len(mechanism_ids)),
        "mechanism_id": mechanism_ids,
        "beta_low": 0.5,
        "beta_ratio": 0.5,
        "interface_exponent": 1.0,
    })


def prediction_quantiles(
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
    candidate_ids: np.ndarray,
    rows: pd.DataFrame,
    matrix_only: bool = False,
    row_block_size: int = 32,
) -> pd.DataFrame:
    """Exact compatible-set quantiles, evaluated in small target-row blocks."""
    _validate_parameters(parameters)
    candidate_ids = np.asarray(candidate_ids)
    selected = parameters[parameters["candidate_id"].isin(candidate_ids)]
    if selected.empty:
        raise ValueError("Empty compatible set: no prediction interval available")
    if len(selected) != len(np.unique(candidate_ids)):
        raise ValueError("Unknown candidate ID requested")
    condition_index, matrix, surface, prediction_ids = _condition_alignment(predictions, rows)
    prediction_lookup = {mechanism: index for index, mechanism in enumerate(prediction_ids)}
    frequencies = np.array(sorted(rows["frequency_hz"].unique()), dtype=float)
    frequency_lookup = {float(f): index for index, f in enumerate(frequencies)}
    frequency_index = rows["frequency_hz"].astype(float).map(frequency_lookup).to_numpy(dtype=int)
    quantiles = np.empty((3, len(rows)), dtype=float)
    for start in range(0, len(rows), row_block_size):
        stop = min(start + row_block_size, len(rows))
        block_condition = condition_index[start:stop]
        block_frequency = frequency_index[start:stop]
        parts = []
        for mechanism, group in selected.groupby("mechanism_id", sort=True):
            mechanism_index = prediction_lookup[mechanism]
            if matrix_only:
                part = np.broadcast_to(
                    matrix[mechanism_index, block_condition],
                    (len(group), len(block_condition)),
                )
            else:
                values = multifrequency_forward(
                    matrix[mechanism_index],
                    surface[mechanism_index],
                    group["beta_low"].to_numpy(dtype=float),
                    group["beta_ratio"].to_numpy(dtype=float),
                    group["interface_exponent"].to_numpy(dtype=float),
                    frequencies,
                )
                part = values[:, block_frequency, block_condition]
            parts.append(part)
        chosen = np.concatenate(parts, axis=0)
        quantiles[:, start:stop] = np.quantile(chosen, [0.05, 0.5, 0.95], axis=0)
    result = rows.copy().reset_index(drop=True)
    result["sim_q05"] = quantiles[0]
    result["sim_q50"] = quantiles[1]
    result["sim_q95"] = quantiles[2]
    result["compatible_count"] = int(len(selected))
    result["degenerate_interval"] = len(selected) == 1
    return result


def loo_summary(rat_summary: pd.DataFrame, points: pd.DataFrame) -> dict:
    available = points[points["prediction_available"]]
    return {
        "fold_count": int(len(rat_summary)),
        "available_fold_count": int((rat_summary["compatible_count"] > 0).sum()),
        "empty_fold_count": int((rat_summary["compatible_count"] == 0).sum()),
        "degenerate_fold_count": int(rat_summary["degenerate_interval"].sum()),
        "total_point_count": int(len(points)),
        "available_point_count": int(len(available)),
        "point_coverage": float(available["covered"].mean()) if len(available) else None,
        "median_interval_width": float(available["width"].median()) if len(available) else None,
        "median_relative_width": (
            float((available["width"] / available["z_ratio"]).median()) if len(available) else None
        ),
        "reading_rule": (
            "Coverage uses only nonempty folds and is reported with prediction availability; "
            "empty-fold coverage is NA and single-candidate zero-width intervals are degenerate."
        ),
    }


def _loo_frequency_summary(rat_summary: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for frequency, group in points.groupby("frequency_hz", sort=True):
        available = group[group["prediction_available"]]
        fold_counts = group.groupby("rat")["prediction_available"].any()
        rows.append({
            "frequency_hz": float(frequency),
            "fold_count": int(len(fold_counts)),
            "available_fold_count": int(fold_counts.sum()),
            "empty_fold_count": int((~fold_counts).sum()),
            "degenerate_fold_count": int(
                available.groupby("rat")["degenerate_interval"].any().sum()
            ) if len(available) else 0,
            "total_point_count": int(len(group)),
            "available_point_count": int(len(available)),
            "point_coverage": float(available["covered"].mean()) if len(available) else np.nan,
            "median_interval_width": float(available["width"].median()) if len(available) else np.nan,
            "median_relative_width": (
                float((available["width"] / available["z_ratio"]).median())
                if len(available) else np.nan
            ),
        })
    return pd.DataFrame(rows)


def leave_one_rat_out(
    predictions: pd.DataFrame,
    parameters: pd.DataFrame,
    targets: pd.DataFrame,
    scores: CandidateScores | None = None,
    matrix_only: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    """Refit four tolerances and the compatible set within every training fold."""
    if scores is None:
        scores = score_multifrequency_candidates(
            predictions, parameters, targets, frequency_tolerances(targets),
            matrix_only=matrix_only,
        )
    rat_rows = []
    point_rows = []
    for held_out in scores.rat_ids:
        train = targets[targets["rat"] != held_out]
        test = targets[targets["rat"] == held_out].copy().reset_index(drop=True)
        tolerances = frequency_tolerances(train)
        rat_mask = scores.rat_ids != held_out
        normalized, raw, compatible = _scores_for_rats(scores, rat_mask, tolerances)
        selected = scores.candidate_ids[compatible]
        rat_row = {
            "rat": held_out,
            "compatible_count": int(len(selected)),
            "n_points": int(len(test)),
            "best_training_normalized_rmse": float(np.min(normalized)),
            "coverage": np.nan,
            "median_width": np.nan,
            "degenerate_interval": len(selected) == 1,
        }
        for index, frequency in enumerate(scores.frequencies):
            rat_row[f"tolerance_{frequency:g}hz"] = tolerances[float(frequency)]
            rat_row[f"best_training_raw_rmse_{frequency:g}hz"] = float(
                raw[np.argmin(normalized), index]
            )
        if len(selected):
            intervals = prediction_quantiles(
                predictions, parameters, selected, test, matrix_only=matrix_only
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


def residual_drift_by_frequency(
    targets: pd.DataFrame, intervals: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """EDTA time drift and stage residuals independently at every frequency."""
    keys = ["rat", "frequency_hz", *CONDITION_KEYS]
    merged = targets.merge(
        intervals[[*keys, "sim_q50"]], on=keys, how="inner", validate="one_to_one"
    )
    merged["residual"] = merged["z_ratio"] - merged["sim_q50"]
    drift_rows = []
    for frequency, group in merged.groupby("frequency_hz", sort=True):
        edta = group[group["stage"] == "edta"]
        if len(edta) >= 8:
            rho, p_value = spearmanr(edta["time_s"], edta["residual"])
        else:
            rho, p_value = np.nan, np.nan
        drift_rows.append({
            "frequency_hz": float(frequency),
            "residual_vs_edta_time_spearman_rho": float(rho),
            "residual_vs_edta_time_spearman_p": float(p_value),
            "drift_flag_abs_rho_gt_0.4_and_p_lt_0.05": bool(
                np.isfinite(rho) and abs(rho) > 0.4 and p_value < 0.05
            ),
        })
    stage = merged.groupby(["frequency_hz", "stage"], as_index=False)["residual"].agg(
        residual_mean="mean", residual_sd="std", residual_count="count"
    )
    return pd.DataFrame(drift_rows), stage
