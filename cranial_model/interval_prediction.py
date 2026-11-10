"""Training-only ensemble intervals and grouped cross-conformal calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cranial_model.frequency_surface_polarization import (
    frequency_surface_polarization_forward,
)
from cranial_model.multifrequency_calibration import FREQUENCIES_HZ


def ensemble_intervals(
    parameters: pd.DataFrame,
    arrays: dict,
) -> pd.DataFrame:
    """Return exact ensemble 5/50/95% predictions in target-row order."""
    x = parameters[[f"x{i}" for i in range(17)]].to_numpy(dtype=float)
    prediction = frequency_surface_polarization_forward(
        x, arrays["matrix"][0], arrays["interface"][0], arrays["exposure"][0],
        arrays["stage"], arrays["dose"], arrays["time_s"],
    )
    values = prediction[:, arrays["frequency_index"], arrays["condition_index"]]
    q05, q50, q95 = np.quantile(values, [0.05, 0.5, 0.95], axis=0)
    return pd.DataFrame({"sim_q05": q05, "sim_q50": q50, "sim_q95": q95})


def nonconformity_scores(
    intervals: pd.DataFrame,
    tolerances: dict[float, float],
    observed_column: str = "z_ratio",
) -> np.ndarray:
    """Normalized signed expansion needed to cover each observed point.

    Negative values mean the unexpanded ensemble interval already covers the point.
    """
    tolerance = intervals["frequency_hz"].map(tolerances).to_numpy(dtype=float)
    if np.any(~np.isfinite(tolerance)) or np.any(tolerance <= 0):
        raise ValueError("Every calibration row needs a positive training tolerance")
    observed = intervals[observed_column].to_numpy(dtype=float)
    lower = intervals["sim_q05"].to_numpy(dtype=float)
    upper = intervals["sim_q95"].to_numpy(dtype=float)
    score = np.maximum(lower - observed, observed - upper) / tolerance
    return np.maximum(score, 0.0)


def frequency_expansions(
    scores: np.ndarray,
    frequencies: np.ndarray,
    level: float = 0.90,
) -> dict[float, float]:
    """Frequency-specific finite-sample higher quantiles of calibration scores."""
    if not 0.0 < level < 1.0:
        raise ValueError("Calibration level must lie between zero and one")
    scores = np.asarray(scores, dtype=float).reshape(-1)
    frequencies = np.asarray(frequencies, dtype=float).reshape(-1)
    if scores.shape != frequencies.shape or scores.size == 0:
        raise ValueError("Calibration scores and frequencies must be nonempty and aligned")
    if np.any(~np.isfinite(scores)) or np.any(scores < 0):
        raise ValueError("Calibration scores must be finite and nonnegative")
    result = {}
    for frequency in FREQUENCIES_HZ:
        selected = scores[frequencies == frequency]
        if not len(selected):
            raise ValueError(f"No inner calibration scores at {frequency:g} Hz")
        # np.quantile(method='higher') gives the conservative empirical order statistic.
        result[float(frequency)] = float(np.quantile(selected, level, method="higher"))
    return result


def expand_intervals(
    intervals: pd.DataFrame,
    tolerances: dict[float, float],
    expansions: dict[float, float],
    observed_column: str = "z_ratio",
    physical_lower_bound: float | None = 0.0,
) -> pd.DataFrame:
    """Expand both ensemble tails without using any outer-held-out observations.

    Impedance ratios retain a nonnegative physical lower bound. Delta-Z callers
    must pass ``physical_lower_bound=None`` because impedance decreases are negative.
    """
    result = intervals.copy()
    tolerance = result["frequency_hz"].map(tolerances).to_numpy(dtype=float)
    expansion = result["frequency_hz"].map(expansions).to_numpy(dtype=float)
    if np.any(~np.isfinite(tolerance)) or np.any(~np.isfinite(expansion)):
        raise ValueError("Missing tolerance or calibrated expansion")
    padding = tolerance * expansion
    result["unclipped_calibrated_lower"] = result["sim_q05"] - padding
    if physical_lower_bound is None:
        result["calibrated_lower"] = result["unclipped_calibrated_lower"]
    else:
        result["calibrated_lower"] = np.maximum(
            physical_lower_bound, result["unclipped_calibrated_lower"]
        )
    result["calibrated_upper"] = result["sim_q95"] + padding
    result["interval_padding"] = padding
    result["calibrated_width"] = result["calibrated_upper"] - result["calibrated_lower"]
    result["calibrated_covered"] = (
        (result[observed_column] >= result["calibrated_lower"])
        & (result[observed_column] <= result["calibrated_upper"])
    ).astype("boolean")
    result["original_compatible_covered"] = pd.Series(pd.NA, index=result.index, dtype="boolean")
    compatible = result["original_compatible_prediction_available"].to_numpy(dtype=bool)
    if compatible.any():
        result.loc[compatible, "original_compatible_covered"] = (
            (result.loc[compatible, observed_column] >= result.loc[compatible, "sim_q05"])
            & (result.loc[compatible, observed_column] <= result.loc[compatible, "sim_q95"])
        ).to_numpy(dtype=bool)
    return result
