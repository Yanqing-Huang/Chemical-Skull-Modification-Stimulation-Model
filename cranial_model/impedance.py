"""Exploratory R-CPE fitting for impedance magnitude spectra.

Four magnitudes cannot uniquely confirm four circuit parameters. The quality
checks below reject failed, poorly fitted, rank-deficient, ill-conditioned and
boundary solutions; an accepted low-frequency limit remains exploratory until
phase or complex-impedance measurements are available.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares


def complex_rcpe_impedance(
    frequency_hz: np.ndarray | float,
    r0: float,
    rinf: float,
    tau: float,
    beta: float,
) -> np.ndarray:
    """Return Z = Rinf + (R0-Rinf)/(1+(j*omega*tau)**beta)."""
    frequency = np.asarray(frequency_hz, dtype=float)
    if np.any(~np.isfinite(frequency)) or np.any(frequency < 0):
        raise ValueError("Frequencies must be finite and non-negative")
    if not (r0 > 0 and rinf >= 0 and tau > 0 and 0 < beta <= 1):
        raise ValueError("Require R0>0, Rinf>=0, tau>0 and 0<beta<=1")
    omega = 2.0 * np.pi * frequency
    return rinf + (r0 - rinf) / (1.0 + (1j * omega * tau) ** beta)


def fit_rcpe_magnitude(
    frequency_hz: np.ndarray,
    magnitude_mohm: np.ndarray,
    relative_rmse_tolerance: float = 0.05,
    condition_limit: float = 1e8,
) -> dict:
    """Fit four R-CPE parameters with deterministic multi-start optimization.

    Parameters are log(Rinf), log(R0-Rinf), log(tau), and beta. This enforces
    R0>Rinf>0, positive tau, and 0<beta<=1 throughout optimization.
    """
    frequency = np.asarray(frequency_hz, dtype=float).reshape(-1)
    magnitude = np.asarray(magnitude_mohm, dtype=float).reshape(-1)
    if frequency.size != magnitude.size or frequency.size < 4:
        raise ValueError("At least four paired frequency/magnitude values are required")
    if (
        np.any(~np.isfinite(frequency))
        or np.any(~np.isfinite(magnitude))
        or np.any(frequency <= 0)
        or np.any(magnitude <= 0)
        or np.unique(frequency).size < 4
    ):
        raise ValueError("Require positive finite magnitudes and at least four positive unique frequencies")
    order = np.argsort(frequency)
    frequency = frequency[order]
    magnitude = magnitude[order]

    log_lower = np.log([1e-9, 1e-9, 1e-12])
    log_upper = np.log([1e6, 1e6, 1e6])
    lower = np.array([*log_lower, 0.05])
    upper = np.array([*log_upper, 1.0])

    def unpack(parameters: np.ndarray) -> tuple[float, float, float, float]:
        rinf, difference, tau = np.exp(parameters[:3])
        return float(rinf + difference), float(rinf), float(tau), float(parameters[3])

    def residual(parameters: np.ndarray) -> np.ndarray:
        r0, rinf, tau, beta = unpack(parameters)
        predicted = np.abs(complex_rcpe_impedance(frequency, r0, rinf, tau, beta))
        return (predicted - magnitude) / magnitude

    y_min = float(np.min(magnitude))
    y_max = float(np.max(magnitude))
    starts = []
    for rinf_fraction in (0.05, 0.5, 0.9):
        rinf_start = y_min * rinf_fraction
        difference_start = max(y_max - rinf_start, float(np.mean(magnitude)) * 1e-3, 1e-8)
        for characteristic_frequency in (100.0, 1000.0, 10000.0):
            tau_start = 1.0 / (2.0 * np.pi * characteristic_frequency)
            for beta_start in (0.5, 0.8):
                starts.append([rinf_start, difference_start, tau_start, beta_start])

    best = None
    best_cost = np.inf
    for start in starts:
        initial = np.array([*np.log(start[:3]), start[3]], dtype=float)
        initial = np.clip(initial, lower + 1e-10, upper - 1e-10)
        try:
            fit = least_squares(
                residual,
                initial,
                bounds=(lower, upper),
                method="trf",
                ftol=1e-12,
                xtol=1e-12,
                gtol=1e-12,
                max_nfev=1500,
            )
        except (ValueError, FloatingPointError):
            continue
        if np.isfinite(fit.cost) and fit.cost < best_cost:
            best = fit
            best_cost = float(fit.cost)

    if best is None:
        return {
            "r0_mohm": np.nan,
            "rinf_mohm": np.nan,
            "tau_s": np.nan,
            "beta": np.nan,
            "predicted_magnitude_mohm": np.full(frequency.size, np.nan),
            "relative_rmse": np.nan,
            "success": False,
            "identifiable": False,
            "jacobian_rank": 0,
            "jacobian_condition": np.inf,
            "boundary_solution": False,
            "quality_flag": "fit_failed",
            "message": "All deterministic optimization starts failed",
        }

    r0, rinf, tau, beta = unpack(best.x)
    predicted = np.abs(complex_rcpe_impedance(frequency, r0, rinf, tau, beta))
    relative_rmse = float(np.sqrt(np.mean(((predicted - magnitude) / magnitude) ** 2)))
    singular_values = np.linalg.svd(best.jac, compute_uv=False)
    rank_tolerance = np.finfo(float).eps * max(best.jac.shape) * singular_values[0]
    rank = int(np.sum(singular_values > rank_tolerance))
    condition = float(singular_values[0] / singular_values[-1]) if singular_values[-1] > 0 else np.inf
    boundary_tolerance = 1e-4
    at_boundary = bool(
        np.any(best.x[:3] <= lower[:3] + boundary_tolerance)
        or np.any(best.x[:3] >= upper[:3] - boundary_tolerance)
        or beta <= lower[3] + boundary_tolerance
        or beta >= upper[3] - boundary_tolerance
    )
    success = bool(best.success and np.all(np.isfinite(predicted)))
    identifiable = bool(rank == 4 and condition <= condition_limit and not at_boundary)

    if not success:
        quality = "fit_failed"
        message = str(best.message)
    elif relative_rmse > relative_rmse_tolerance:
        quality = "poor_fit"
        message = f"Relative RMSE {relative_rmse:.4g} exceeds {relative_rmse_tolerance:.4g}"
    elif rank < 4 or condition > condition_limit:
        quality = "non_identifiable"
        message = f"Jacobian rank {rank}/4; condition number {condition:.4g}"
    elif at_boundary:
        quality = "boundary_solution"
        message = "At least one fitted parameter is at a physical/numerical bound"
    else:
        quality = "accepted"
        message = "Exploratory magnitude-only fit; phase data are required for confirmation"

    return {
        "r0_mohm": float(r0),
        "rinf_mohm": float(rinf),
        "tau_s": float(tau),
        "beta": float(beta),
        "predicted_magnitude_mohm": predicted,
        "relative_rmse": relative_rmse,
        "success": success,
        "identifiable": identifiable,
        "jacobian_rank": rank,
        "jacobian_condition": condition,
        "boundary_solution": at_boundary,
        "quality_flag": quality,
        "message": message,
    }
