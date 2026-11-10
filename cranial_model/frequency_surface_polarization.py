"""Best reproduced frequency-surface observation model with optional V5b polarization."""

from __future__ import annotations

import numpy as np
from scipy.special import expit

from cranial_model.diploe_polarization import (
    POLARIZATION_PARAMETER_BOUNDS, polarization_state,
)
from cranial_model.multifrequency_calibration import FREQUENCIES_HZ


# Four weights, four interface exponents, shared exposure slope,
# EDTA log-dose linear/quadratic, acid offset, exposure quadratic.
OBSERVATION_PARAMETER_BOUNDS = (
    [(1e-6, 1.0 - 1e-6)] * 4
    + [(0.5, 3.0)] * 4
    + [(-6.0, 6.0), (-3.0, 3.0), (-3.0, 3.0), (-3.0, 3.0), (-6.0, 6.0)]
)
COMBINED_PARAMETER_BOUNDS = [
    *OBSERVATION_PARAMETER_BOUNDS, *POLARIZATION_PARAMETER_BOUNDS
]
# Optional surface-relaxation response in normalized global soaking time.
TIME_RESPONSE_PARAMETER_BOUNDS = [(-6.0, 6.0), (-3.0, 3.0)]
TIME_COMBINED_PARAMETER_BOUNDS = [
    *COMBINED_PARAMETER_BOUNDS, *TIME_RESPONSE_PARAMETER_BOUNDS
]


def _candidates(parameters: np.ndarray, dimensions: int) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    elif values.shape[0] == dimensions and values.shape[1] != dimensions:
        values = values.T
    if values.ndim != 2 or values.shape[1] != dimensions:
        raise ValueError(f"Expected candidate matrices with {dimensions} columns")
    return values


def _condition_features(stage, edta_concentration_mol_per_l, time_s):
    stage = np.asarray(stage)
    dose = np.asarray(edta_concentration_mol_per_l, dtype=float).reshape(-1)
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    if stage.shape != dose.shape or stage.shape != time_s.shape:
        raise ValueError("Condition arrays must align")
    if not set(stage).issubset({"acetic_acid", "edta"}):
        raise ValueError("Only acid and EDTA conditions are supported")
    acid = (stage == "acetic_acid").astype(float)
    log_dose = np.log(dose / 0.1)
    return acid, log_dose, time_s


def frequency_surface_forward(
    parameters: np.ndarray,
    matrix_ratio: np.ndarray,
    interface_ratio: np.ndarray,
    relative_exposure: np.ndarray,
    stage: np.ndarray,
    edta_concentration_mol_per_l: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    """Return candidate-by-frequency-by-condition observation predictions."""
    a = _candidates(parameters, 13)
    matrix = np.asarray(matrix_ratio, dtype=float).reshape(-1)
    interface = np.asarray(interface_ratio, dtype=float).reshape(-1)
    exposure = np.asarray(relative_exposure, dtype=float).reshape(-1)
    acid, log_dose, _ = _condition_features(
        stage, edta_concentration_mol_per_l, time_s
    )
    if not matrix.shape == interface.shape == exposure.shape == acid.shape:
        raise ValueError("Mechanism states and conditions must align")
    if np.any(interface <= 0) or np.any(exposure < 0):
        raise ValueError("Invalid interface or exposure state")

    weights = np.clip(a[:, :4], 1e-12, 1.0 - 1e-12)
    logits = np.log(weights / (1.0 - weights))
    edta = 1.0 - acid
    adjustment = (
        a[:, 8, None] * exposure[None, :]
        + edta[None, :] * (
            a[:, 9, None] * log_dose[None, :]
            + a[:, 10, None] * log_dose[None, :] ** 2
        )
        + a[:, 11, None] * acid[None, :]
        + a[:, 12, None] * exposure[None, :] ** 2
    )
    effective_weight = expit(logits[:, :, None] + adjustment[:, None, :])
    interface_response = interface[None, None, :] ** (-a[:, 4:8, None])
    return (
        (1.0 - effective_weight) * matrix[None, None, :]
        + effective_weight * interface_response
    )


def frequency_surface_polarization_forward(
    parameters: np.ndarray,
    matrix_ratio: np.ndarray,
    interface_ratio: np.ndarray,
    relative_exposure: np.ndarray,
    stage: np.ndarray,
    edta_concentration_mol_per_l: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    """Add a series diploe-polarization magnitude to the best observation model.

    Zero polarization amplitude exactly delegates to the 13-parameter baseline.
    """
    a = _candidates(parameters, 17)
    amplitude = 10.0 ** a[:, 15]
    if np.all(amplitude == 0.0):
        return frequency_surface_forward(
            a[:, :13], matrix_ratio, interface_ratio, relative_exposure,
            stage, edta_concentration_mol_per_l, time_s,
        )
    base = frequency_surface_forward(
        a[:, :13], matrix_ratio, interface_ratio, relative_exposure,
        stage, edta_concentration_mol_per_l, time_s,
    )
    # Recompute only the matrix term; the calibrated surface term is unchanged.
    matrix = np.asarray(matrix_ratio, dtype=float).reshape(-1)
    exposure = np.asarray(relative_exposure, dtype=float).reshape(-1)
    interface = np.asarray(interface_ratio, dtype=float).reshape(-1)
    acid, log_dose, time_s = _condition_features(
        stage, edta_concentration_mol_per_l, time_s
    )
    weights = np.clip(a[:, :4], 1e-12, 1.0 - 1e-12)
    logits = np.log(weights / (1.0 - weights))
    edta = 1.0 - acid
    adjustment = (
        a[:, 8, None] * exposure[None, :]
        + edta[None, :] * (
            a[:, 9, None] * log_dose[None, :]
            + a[:, 10, None] * log_dose[None, :] ** 2
        )
        + a[:, 11, None] * acid[None, :]
        + a[:, 12, None] * exposure[None, :] ** 2
    )
    effective_weight = expit(logits[:, :, None] + adjustment[:, None, :])
    interface_response = interface[None, None, :] ** (-a[:, 4:8, None])

    m = polarization_state(a[:, 13:17], stage, time_s)
    alpha = a[:, 16]
    omega = 2.0 * np.pi * np.asarray(FREQUENCIES_HZ, dtype=float)
    jw_minus_alpha = np.exp(-alpha[:, None] * (np.log(omega)[None, :] + 1j * np.pi / 2))
    z_polarization = amplitude[:, None, None] * m[:, None, :] * jw_minus_alpha[:, :, None]
    matrix_response = np.abs(matrix[None, None, :] + z_polarization)
    return (
        (1.0 - effective_weight) * matrix_response
        + effective_weight * interface_response
    )


def frequency_surface_time_polarization_forward(
    parameters: np.ndarray,
    matrix_ratio: np.ndarray,
    interface_ratio: np.ndarray,
    relative_exposure: np.ndarray,
    stage: np.ndarray,
    edta_concentration_mol_per_l: np.ndarray,
    time_s: np.ndarray,
) -> np.ndarray:
    """Add quadratic soaking-time surface relaxation to the combined model.

    The two final coefficients multiply t and t^2, with t=time_s/5400.
    Setting both to zero exactly delegates to the 17-parameter model.
    """
    a = _candidates(parameters, 19)
    if np.all(a[:, 17:19] == 0.0):
        return frequency_surface_polarization_forward(
            a[:, :17], matrix_ratio, interface_ratio, relative_exposure,
            stage, edta_concentration_mol_per_l, time_s,
        )
    matrix = np.asarray(matrix_ratio, dtype=float).reshape(-1)
    exposure = np.asarray(relative_exposure, dtype=float).reshape(-1)
    interface = np.asarray(interface_ratio, dtype=float).reshape(-1)
    acid, log_dose, time_s = _condition_features(
        stage, edta_concentration_mol_per_l, time_s
    )
    if not matrix.shape == interface.shape == exposure.shape == acid.shape:
        raise ValueError("Mechanism states and conditions must align")
    if np.any(interface <= 0) or np.any(exposure < 0):
        raise ValueError("Invalid interface or exposure state")
    weights = np.clip(a[:, :4], 1e-12, 1.0 - 1e-12)
    logits = np.log(weights / (1.0 - weights))
    edta = 1.0 - acid
    normalized_time = time_s / 5400.0
    adjustment = (
        a[:, 8, None] * exposure[None, :]
        + edta[None, :] * (
            a[:, 9, None] * log_dose[None, :]
            + a[:, 10, None] * log_dose[None, :] ** 2
        )
        + a[:, 11, None] * acid[None, :]
        + a[:, 12, None] * exposure[None, :] ** 2
        + a[:, 17, None] * normalized_time[None, :]
        + a[:, 18, None] * normalized_time[None, :] ** 2
    )
    effective_weight = expit(logits[:, :, None] + adjustment[:, None, :])
    interface_response = interface[None, None, :] ** (-a[:, 4:8, None])
    m = polarization_state(a[:, 13:17], stage, time_s)
    alpha = a[:, 16]
    omega = 2.0 * np.pi * np.asarray(FREQUENCIES_HZ, dtype=float)
    jw_minus_alpha = np.exp(
        -alpha[:, None] * (np.log(omega)[None, :] + 1j * np.pi / 2)
    )
    amplitude = 10.0 ** a[:, 15]
    z_polarization = (
        amplitude[:, None, None] * m[:, None, :] * jw_minus_alpha[:, :, None]
    )
    matrix_response = np.abs(matrix[None, None, :] + z_polarization)
    return (
        (1.0 - effective_weight) * matrix_response
        + effective_weight * interface_response
    )
