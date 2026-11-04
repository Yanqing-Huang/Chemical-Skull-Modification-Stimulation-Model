"""Shrinking-core, Thiele-modulus, effectiveness-factor and Biot diagnostics."""

from __future__ import annotations

import numpy as np

from .core import LAYER_NAMES, ScenarioResult, effective_diffusivity


def outer_scm_flux_mol_per_m2_s(
    bulk_concentration_mol_per_m3: float,
    film_coefficient_m_per_s: float,
    front_position_m: float,
    effective_diffusivity_m2_per_s: float,
    surface_reaction_m_per_s: float,
) -> float:
    resistance = (
        1.0 / film_coefficient_m_per_s
        + front_position_m / effective_diffusivity_m2_per_s
        + 1.0 / surface_reaction_m_per_s
    )
    return float(bulk_concentration_mol_per_m3 / resistance)


def outer_scm_time_to_front_s(
    front_position_m: np.ndarray | float,
    bulk_concentration_mol_per_m3: float,
    film_coefficient_m_per_s: float,
    effective_diffusivity_m2_per_s: float,
    surface_reaction_m_per_s: float,
    reagent_per_calcium_stoichiometry: float,
    calcium_mol_per_m3: float,
) -> np.ndarray:
    """Analytical outer-table SCM time, including film, product layer and reaction."""
    position = np.asarray(front_position_m, dtype=float)
    prefactor = reagent_per_calcium_stoichiometry * calcium_mol_per_m3
    prefactor /= bulk_concentration_mol_per_m3
    resistance_time = (
        position / film_coefficient_m_per_s
        + position**2 / (2.0 * effective_diffusivity_m2_per_s)
        + position / surface_reaction_m_per_s
    )
    return prefactor * resistance_time


def inner_scm_flux_mol_per_m2_s(
    interface_concentration_mol_per_m3: float,
    front_position_m: float,
    effective_diffusivity_m2_per_s: float,
    surface_reaction_m_per_s: float,
) -> float:
    """Inner-table flux driven by diploe/interface concentration, never by external bath."""
    resistance = (
        front_position_m / effective_diffusivity_m2_per_s
        + 1.0 / surface_reaction_m_per_s
    )
    return float(interface_concentration_mol_per_m3 / resistance)


def inner_scm_time_constant_driver_s(
    front_position_m: np.ndarray | float,
    interface_concentration_mol_per_m3: float,
    effective_diffusivity_m2_per_s: float,
    surface_reaction_m_per_s: float,
    reagent_per_calcium_stoichiometry: float,
    calcium_mol_per_m3: float,
) -> np.ndarray:
    """Analytical inner-table SCM limit for a constant interface concentration."""
    position = np.asarray(front_position_m, dtype=float)
    prefactor = reagent_per_calcium_stoichiometry * calcium_mol_per_m3
    prefactor /= interface_concentration_mol_per_m3
    return prefactor * (
        position**2 / (2.0 * effective_diffusivity_m2_per_s)
        + position / surface_reaction_m_per_s
    )


def thiele_biot_rows(scenario: ScenarioResult) -> list[dict[str, float | str | int]]:
    rows: list[dict[str, float | str | int]] = []
    values = scenario.values
    grid = scenario.grid
    for stage_name in ("acetic_acid", "edta", "urea"):
        stage = values["stages"][stage_name]
        initial_calcium = scenario.stages[stage_name].initial_calcium
        diffusivity = effective_diffusivity(grid, values, stage_name, initial_calcium)
        for layer_index, (layer_name, layer_slice) in enumerate(
            zip(LAYER_NAMES, grid.layer_slices)
        ):
            layer = values["geometry"]["layers"][layer_index]
            thickness = layer["thickness_mm"] * 1e-3
            weights = grid.widths[layer_slice]
            q_mean = float(np.average(initial_calcium[layer_slice], weights=weights))
            d_e = float(np.average(diffusivity[layer_slice], weights=weights))
            k_r = stage["k_r_m3_per_mol_s"]
            stoichiometry = stage["reagent_per_calcium_stoichiometry"]
            phi = thickness * np.sqrt(stoichiometry * k_r * q_mean / d_e) if k_r > 0 else 0.0
            effectiveness = float(np.tanh(phi) / phi) if phi > 0 else 1.0
            biot = values["transport"]["external_mass_transfer_m_per_s"] * thickness / d_e
            reaction_zone = values["transport"]["scm_reaction_zone_thickness_m"]
            k_s = k_r * q_mean * reaction_zone
            bulk_concentration = stage["bulk_concentration_mol_per_l"] * 1000.0
            if k_r > 0 and bulk_concentration > 0:
                t50 = outer_scm_time_to_front_s(
                    0.5 * thickness,
                    bulk_concentration,
                    values["transport"]["external_mass_transfer_m_per_s"],
                    d_e,
                    k_s,
                    stoichiometry,
                    q_mean,
                )
                t50_value: float | str = float(t50)
            else:
                t50_value = ""
            rows.append(
                {
                    "edta_concentration_mol_per_l": scenario.edta_concentration_mol_per_l,
                    "stage": stage_name,
                    "layer": layer_name,
                    "stage_initial_calcium_mol_per_m3": q_mean,
                    "effective_diffusivity_m2_per_s": d_e,
                    "thiele_modulus": float(phi),
                    "internal_effectiveness_factor": effectiveness,
                    "biot_number": float(biot),
                    "equivalent_scm_surface_reaction_m_per_s": float(k_s),
                    "outer_table_constant_bath_scm_t50_s": t50_value,
                }
            )
    return rows
