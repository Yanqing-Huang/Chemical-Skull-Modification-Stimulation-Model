"""V5a site-resolved series-bone / parallel dynamic-film observation model."""

from __future__ import annotations

import numpy as np

from cranial_model.multifrequency_calibration import FREQUENCIES_HZ
from cranial_model.site_geometry import SITE_NAMES


CONDITION_KEYS = ["stage", "edta_concentration_mol_per_l", "time_s"]
# log10(g0_B/M/L), log10(q0_B/M/L), alpha, p_g, p_q,
# log10(1+a_g), log10(1+a_q), log10(k_exposure)
PARAMETER_BOUNDS = (
    [(-6.0, 2.0)] * 3
    + [(-10.0, 0.0)] * 3
    + [(0.5, 1.0), (0.0, 3.0), (0.0, 3.0),
       (0.0, np.log10(11.0)), (0.0, np.log10(11.0)), (-1.0, 3.0)]
)


def _candidate_matrix(parameters: np.ndarray) -> np.ndarray:
    values = np.asarray(parameters, dtype=float)
    if values.ndim == 1:
        values = values[None, :]
    elif values.shape[0] == 12 and values.shape[1] != 12:
        values = values.T
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError("V5a parameters must have twelve columns")
    return values


def site_film_forward(
    parameters: np.ndarray,
    matrix_ratio: np.ndarray,
    outer_porosity: np.ndarray,
    inner_porosity: np.ndarray,
    outer_exposure: np.ndarray,
    inner_exposure: np.ndarray,
    side: str = "two_sided",
) -> np.ndarray:
    """Return candidate-by-site-by-frequency-by-condition magnitude ratios.

    Each state input is one mechanism's site-by-condition array. The through-skull
    resistance already is the outer/diploe/inner series sum. Film conductance and
    CPE admittance are parallel to that through-path.
    """
    a = _candidate_matrix(parameters)
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

    alpha = a[:, 6]
    omega = 2.0 * np.pi * np.asarray(FREQUENCIES_HZ, dtype=float)
    jw_alpha = np.exp(alpha[:, None] * (np.log(omega)[None, :] + 1j * np.pi / 2))
    pg, pq = a[:, 7], a[:, 8]
    ag, aq = 10.0 ** a[:, 9] - 1.0, 10.0 ** a[:, 10] - 1.0
    exposure_rate = 10.0 ** a[:, 11]
    h_outer = 1.0 - np.exp(-exposure_rate[:, None, None] * outer_xi[None, ...])
    h_inner = 1.0 - np.exp(-exposure_rate[:, None, None] * inner_xi[None, ...])

    site_predictions = []
    for site in range(3):
        g0 = 10.0 ** a[:, site]
        q0 = 10.0 ** a[:, 3 + site]
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
        admittance = (
            1.0 / matrix[None, None, site, :]
            + conductance[:, None, :]
            + susceptance[:, None, :] * jw_alpha[:, :, None]
        )
        native_admittance = 1.0 + 2.0 * g0[:, None] + 2.0 * q0[:, None] * jw_alpha
        ratio = np.abs(native_admittance)[:, :, None] / np.abs(admittance)
        site_predictions.append(ratio)
    return np.stack(site_predictions, axis=1)


def rat_condition_mean(site_predictions: np.ndarray) -> np.ndarray:
    """Average site-normalized ratios, matching the locked experimental target."""
    values = np.asarray(site_predictions)
    if values.ndim != 4 or values.shape[1] != 3:
        raise ValueError("Expected candidate-by-site-by-frequency-by-condition predictions")
    return values.mean(axis=1)


def condition_site_alignment(predictions, targets, depth_um: int):
    """Align local mechanism states to target condition order and three sites."""
    conditions = targets[CONDITION_KEYS].drop_duplicates().reset_index(drop=True)
    conditions["_condition"] = np.arange(len(conditions))
    target_map = targets[CONDITION_KEYS].copy()
    target_map["_target_row"] = np.arange(len(target_map))
    target_map = target_map.merge(
        conditions, on=CONDITION_KEYS, how="left", validate="many_to_one", sort=False
    ).sort_values("_target_row")
    merged = predictions.merge(
        conditions, on=CONDITION_KEYS, how="right", validate="many_to_one", sort=False
    )
    columns = [
        "sim_whole_resistance_ratio",
        f"sim_outer_porosity_ratio_{depth_um}um",
        f"sim_inner_porosity_ratio_{depth_um}um",
        f"sim_outer_exposure_{depth_um}um",
        f"sim_inner_exposure_{depth_um}um",
    ]
    mechanism_ids = np.array(sorted(predictions["mechanism_id"].unique()))
    matrices = []
    for column in columns:
        by_site = []
        for site in SITE_NAMES:
            site_frame = merged[merged["site"] == site]
            matrix = site_frame.pivot(
                index="mechanism_id", columns="_condition", values=column
            ).reindex(index=mechanism_ids, columns=np.arange(len(conditions)))
            if matrix.isna().any().any():
                raise ValueError(f"Missing {column} for {site}")
            by_site.append(matrix.to_numpy(dtype=float))
        matrices.append(np.stack(by_site, axis=1))
    return target_map["_condition"].to_numpy(dtype=int), *matrices, mechanism_ids
