"""Absolute post-minus-baseline impedance-change targets and minimax scoring."""

from __future__ import annotations

import numpy as np
import pandas as pd

from cranial_model.equivalent_calibration import CONDITION_KEYS, build_z_ratio_targets
from cranial_model.frequency_surface_polarization import (
    frequency_surface_polarization_forward,
    frequency_surface_time_polarization_forward,
)
from cranial_model.multifrequency_calibration import (
    FREQUENCIES_HZ, frequency_tolerances,
)


def build_delta_z_targets(
    spectra: pd.DataFrame,
    frequencies: tuple[float, ...] = FREQUENCIES_HZ,
    drop_confounded: bool = True,
) -> pd.DataFrame:
    """Build rat-level Delta-Z targets after per-site baseline subtraction.

    Delta-Z is averaged over sites in MΩ. The matching R-Z targets are generated
    by the existing audited builder and merged one-to-one for alignment checks.
    """
    pieces = []
    for frequency in frequencies:
        ratio = build_z_ratio_targets(
            spectra, frequencies=(frequency,), drop_confounded=drop_confounded
        )
        ratio["frequency_hz"] = float(frequency)
        frame = spectra[spectra["frequency_hz"] == frequency].copy()
        if drop_confounded:
            frame = frame[frame["confounder"] == ""]
        rat_dose = (
            ratio[ratio["stage"] == "edta"]
            [["rat", "edta_concentration_mol_per_l"]]
            .drop_duplicates().set_index("rat")["edta_concentration_mol_per_l"]
        )
        if not rat_dose.index.is_unique:
            raise ValueError("A rat maps to multiple EDTA doses")
        frame.loc[frame["stage"] != "edta", "edta_concentration_mol_per_l"] = np.nan
        frame["edta_concentration_mol_per_l"] = frame[
            "edta_concentration_mol_per_l"
        ].fillna(frame["rat"].map(rat_dose))
        baseline = (
            frame[frame["stage"] == "baseline"]
            .set_index(["rat", "site"])["magnitude_mohm"]
        )
        post = frame[frame["stage"].isin(("acetic_acid", "edta"))].copy()
        post = post.join(baseline.rename("baseline_mohm"), on=["rat", "site"])
        post = post[np.isfinite(post["baseline_mohm"])]
        post["delta_z_mohm"] = post["magnitude_mohm"] - post["baseline_mohm"]
        delta = post.groupby(["rat", *CONDITION_KEYS], as_index=False).agg(
            delta_z_mohm=("delta_z_mohm", "mean"),
            baseline_mean_mohm=("baseline_mohm", "mean"),
            delta_n_values=("delta_z_mohm", "count"),
            delta_site_count=("site", "nunique"),
        )
        delta["frequency_hz"] = float(frequency)
        keys = ["rat", "frequency_hz", *CONDITION_KEYS]
        merged = ratio.merge(
            delta, on=keys, how="inner", validate="one_to_one"
        )
        if len(merged) != len(ratio) or len(merged) != len(delta):
            raise ValueError("Delta-Z and R-Z target rows do not align")
        if not np.array_equal(
            merged["n_values"].to_numpy(), merged["delta_n_values"].to_numpy()
        ) or not np.array_equal(
            merged["site_count"].to_numpy(), merged["delta_site_count"].to_numpy()
        ):
            raise ValueError("Delta-Z and R-Z site audits differ")
        pieces.append(merged)
    result = pd.concat(pieces, ignore_index=True)
    result = result.drop(columns=["delta_n_values", "delta_site_count"])
    if result[["delta_z_mohm", "baseline_mean_mohm"]].isna().any().any():
        raise ValueError("Missing Delta-Z target or baseline")
    if (result["baseline_mean_mohm"] <= 0).any():
        raise ValueError("Baseline impedance must be positive")
    return result.sort_values(
        ["frequency_hz", *CONDITION_KEYS, "rat"]
    ).reset_index(drop=True)


def delta_frequency_tolerances(train: pd.DataFrame) -> dict[float, float]:
    """Apply the unchanged RMS condition-SD rule to Delta-Z in MΩ."""
    renamed = train.copy()
    renamed["z_ratio"] = renamed["delta_z_mohm"]
    return frequency_tolerances(renamed)


def ratio_prediction_to_delta(
    ratio_prediction: np.ndarray,
    frequency_index: np.ndarray,
    condition_index: np.ndarray,
    baseline_mean_mohm: np.ndarray,
) -> np.ndarray:
    """Convert a common site-averaged R-Z prediction to Delta-Z per target row."""
    baseline = np.asarray(baseline_mean_mohm, dtype=float).reshape(-1)
    if baseline.shape != np.asarray(frequency_index).shape or baseline.shape != np.asarray(condition_index).shape:
        raise ValueError("Baseline and target indices must align")
    if np.any(baseline <= 0):
        raise ValueError("Baseline impedance must be positive")
    ratio = np.asarray(ratio_prediction, dtype=float)
    selected = ratio[:, frequency_index, condition_index]
    return baseline[None, :] * (selected - 1.0)


def delta_candidate_metrics(
    parameters,
    arrays: dict,
    targets: pd.DataFrame,
    mechanism_index: int,
    time_response: bool = False,
):
    """Return original minimax form evaluated against Delta-Z targets."""
    values = np.asarray(parameters, dtype=float)
    dimensions = 19 if time_response else 17
    if values.ndim == 1:
        values = values[None, :]
    elif values.shape[0] == dimensions and values.shape[1] != dimensions:
        values = values.T
    if values.ndim != 2 or values.shape[1] != dimensions:
        raise ValueError(f"Expected candidate matrices with {dimensions} columns")
    forward = (
        frequency_surface_time_polarization_forward if time_response
        else frequency_surface_polarization_forward
    )
    ratio = forward(
        values, arrays["matrix"][mechanism_index], arrays["interface"][mechanism_index],
        arrays["exposure"][mechanism_index], arrays["stage"], arrays["dose"],
        arrays["time_s"],
    )
    predicted = ratio_prediction_to_delta(
        ratio, arrays["frequency_index"], arrays["condition_index"],
        targets["baseline_mean_mohm"].to_numpy(dtype=float),
    )
    observed = targets["delta_z_mohm"].to_numpy(dtype=float)
    residual = predicted - observed[None, :]
    if "delta_tolerance_vector" in arrays:
        tolerance_vector = arrays["delta_tolerance_vector"]
        tolerances = arrays["delta_tolerances"]
    else:
        tolerances = delta_frequency_tolerances(targets)
        tolerance_vector = np.array([tolerances[f] for f in FREQUENCIES_HZ])
    frequency_values = targets["frequency_hz"].to_numpy(dtype=float)
    counts = arrays["counts"]
    squared = np.stack([
        np.sum(residual[:, frequency_values == f] ** 2, axis=1)
        for f in FREQUENCIES_HZ
    ], axis=1)
    raw = np.sqrt(squared / counts[None, :])
    normalized = np.sqrt(
        np.sum(squared / tolerance_vector[None, :] ** 2, axis=1)
        / residual.shape[1]
    )
    minimax = np.maximum(
        normalized, np.max(raw / tolerance_vector[None, :], axis=1)
    )
    return minimax, normalized, raw, tolerances
