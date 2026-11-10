"""V5b treatment-induced diploe polarization on the V5a observation layer.

This is a phenomenological, globally shared interface-state diagnostic, not a
identified protein-concentration PDE. The state grows during acetic acid and
desorbs with zero source during EDTA. A series polarization impedance modifies
only the bone path; the V5a surface-film admittance remains in parallel.
"""

from __future__ import annotations

import numpy as np

from cranial_model.multifrequency_calibration import FREQUENCIES_HZ
from cranial_model.site_film_calibration import _candidate_matrix, site_film_forward


# log10(k_on / s^-1), log10(k_off / s^-1),
# log10(A_pol / native series resistance), alpha_pol
POLARIZATION_PARAMETER_BOUNDS = [(-6.0, 0.0), (-7.0, 0.0), (-8.0, 2.0), (0.5, 1.0)]


def _polarization_candidates(parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    elif values.shape[0] == 4 and values.shape[1] != 4:
        values = values.T
    if values.ndim != 2 or values.shape[1] != 4:
        raise ValueError("Diploe polarization parameters must have four columns")
    return values


def polarization_state(parameters: np.ndarray, stage: np.ndarray, time_s: np.ndarray) -> np.ndarray:
    """Return candidate-by-condition treatment-induced dimensionless state.

    The acid source is one and the EDTA source is zero. The state is retained
    across liquid rinses; it is not recomputed from binary reagent supply.
    """
    a = _polarization_candidates(parameters)
    stage = np.asarray(stage)
    time_s = np.asarray(time_s, dtype=float).reshape(-1)
    if stage.shape != time_s.shape or not set(stage).issubset({"acetic_acid", "edta"}):
        raise ValueError("Polarization conditions must be aligned acid/EDTA arrays")
    k_on, k_off = 10.0 ** a[:, 0], 10.0 ** a[:, 1]
    total_rate = k_on + k_off
    m_acid = (k_on / total_rate)[:, None] * (
        1.0 - np.exp(-total_rate[:, None] * 300.0)
    )
    acid_m = (k_on / total_rate)[:, None] * (
        1.0 - np.exp(-total_rate[:, None] * time_s[None, :])
    )
    edta_m = m_acid * np.exp(-k_off[:, None] * time_s[None, :])
    return np.where(stage[None, :] == "acetic_acid", acid_m, edta_m)


def diploe_polarization_forward(
    base_parameters: np.ndarray,
    polarization_parameters: np.ndarray,
    matrix_ratio: np.ndarray,
    outer_porosity: np.ndarray,
    inner_porosity: np.ndarray,
    outer_exposure: np.ndarray,
    inner_exposure: np.ndarray,
    stage: np.ndarray,
    time_s: np.ndarray,
    side: str = "two_sided",
) -> np.ndarray:
    """Return candidate-by-site-by-frequency-by-condition magnitude ratios.

    ``A_pol=0`` exactly delegates to V5a. Otherwise a series impedance ratio
    ``A_pol*m*(j*omega)^(-alpha)`` is added to the normalized three-layer bone
    resistance before combining it with the unchanged parallel film.
    """
    base = _candidate_matrix(base_parameters)
    polarization = _polarization_candidates(polarization_parameters)
    if len(base) != len(polarization):
        raise ValueError("V5a and polarization candidate counts must align")
    amplitude = 10.0 ** polarization[:, 2]
    if np.all(amplitude == 0.0):
        return site_film_forward(
            base, matrix_ratio, outer_porosity, inner_porosity,
            outer_exposure, inner_exposure, side,
        )

    matrix = np.asarray(matrix_ratio, dtype=float)
    outer_phi = np.asarray(outer_porosity, dtype=float)
    inner_phi = np.asarray(inner_porosity, dtype=float)
    outer_xi = np.asarray(outer_exposure, dtype=float)
    inner_xi = np.asarray(inner_exposure, dtype=float)
    if not (
        matrix.shape == outer_phi.shape == inner_phi.shape
        == outer_xi.shape == inner_xi.shape
        and matrix.ndim == 2 and matrix.shape[0] == 3
    ):
        raise ValueError("Expected aligned site-by-condition mechanism states")
    if side not in ("two_sided", "outer_only"):
        raise ValueError("Unknown film side configuration")

    m = polarization_state(polarization, stage, time_s)
    alpha = polarization[:, 3]
    omega = 2.0 * np.pi * np.asarray(FREQUENCIES_HZ, dtype=float)
    jw_minus_alpha = np.exp(-alpha[:, None] * (np.log(omega)[None, :] + 1j * np.pi / 2))
    z_polarization = amplitude[:, None, None] * m[:, None, :] * jw_minus_alpha[:, :, None]

    film_alpha = base[:, 6]
    jw_alpha = np.exp(film_alpha[:, None] * (np.log(omega)[None, :] + 1j * np.pi / 2))
    pg, pq = base[:, 7], base[:, 8]
    ag, aq = 10.0 ** base[:, 9] - 1.0, 10.0 ** base[:, 10] - 1.0
    exposure_rate = 10.0 ** base[:, 11]
    h_outer = 1.0 - np.exp(-exposure_rate[:, None, None] * outer_xi[None, ...])
    h_inner = 1.0 - np.exp(-exposure_rate[:, None, None] * inner_xi[None, ...])

    site_predictions = []
    for site in range(3):
        g0 = 10.0 ** base[:, site]
        q0 = 10.0 ** base[:, 3 + site]
        g_outer = (
            (1.0 + ag[:, None] * h_outer[:, site, :])
            * outer_phi[None, site, :] ** pg[:, None]
        )
        q_outer = (
            (1.0 + aq[:, None] * h_outer[:, site, :])
            * outer_phi[None, site, :] ** pq[:, None]
        )
        if side == "two_sided":
            g_inner = (
                (1.0 + ag[:, None] * h_inner[:, site, :])
                * inner_phi[None, site, :] ** pg[:, None]
            )
            q_inner = (
                (1.0 + aq[:, None] * h_inner[:, site, :])
                * inner_phi[None, site, :] ** pq[:, None]
            )
        else:
            g_inner = np.ones_like(g_outer)
            q_inner = np.ones_like(q_outer)
        conductance = g0[:, None] * (g_outer + g_inner)
        susceptance = q0[:, None] * (q_outer + q_inner)
        bone_admittance = 1.0 / (
            matrix[None, None, site, :] + z_polarization
        )
        admittance = (
            bone_admittance
            + conductance[:, None, :]
            + susceptance[:, None, :] * jw_alpha[:, :, None]
        )
        native_admittance = 1.0 + 2.0 * g0[:, None] + 2.0 * q0[:, None] * jw_alpha
        ratio = np.abs(native_admittance)[:, :, None] / np.abs(admittance)
        site_predictions.append(ratio)
    return np.stack(site_predictions, axis=1)
