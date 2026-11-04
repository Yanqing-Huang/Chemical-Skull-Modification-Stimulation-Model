"""Three-layer, sequential diffusion-reaction model for an isolated SD-rat skull.

Liquid concentration c is per pore-liquid volume. Calcium q is per total bone
volume. The conserved liquid variable is a = porosity * c, per total bone volume.
Effective diffusivity D_e = D_bulk * porosity / tortuosity already contains
porosity and is therefore not multiplied by porosity again in the flux.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp
from scipy.sparse import lil_matrix


LAYER_NAMES = ("outer_table", "diploe", "inner_table")
STAGE_NAMES = ("acetic_acid", "edta", "urea")


def load_parameters(path: str | Path) -> dict[str, Any]:
    import json

    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def layer_by_name(values: dict[str, Any], name: str) -> dict[str, Any]:
    return next(layer for layer in values["geometry"]["layers"] if layer["name"] == name)


@dataclass
class Grid:
    edges: np.ndarray
    centers: np.ndarray
    widths: np.ndarray
    layer_index: np.ndarray
    q_reference: np.ndarray
    initial_porosity: np.ndarray
    tortuosity: np.ndarray
    layer_slices: tuple[slice, slice, slice]

    @classmethod
    def from_parameters(
        cls, values: dict[str, Any], cells_per_layer: list[int] | None = None
    ) -> "Grid":
        layers = values["geometry"]["layers"]
        counts = cells_per_layer or values["numerics"]["cells_per_layer"]
        if len(layers) != 3 or len(counts) != 3:
            raise ValueError("Exactly three layers and three cell counts are required.")
        declared_total = values["geometry"]["total_thickness_mm"]
        actual_total = sum(layer["thickness_mm"] for layer in layers)
        if abs(declared_total - actual_total) > 1e-9:
            raise ValueError("Layer thicknesses must sum to total_thickness_mm.")

        edge_parts = [np.array([0.0])]
        start = 0.0
        slices = []
        cell_offset = 0
        for layer, count in zip(layers, counts):
            if count < 2:
                raise ValueError("Each layer needs at least two cells.")
            end = start + layer["thickness_mm"] * 1e-3
            edge_parts.append(np.linspace(start, end, count + 1)[1:])
            slices.append(slice(cell_offset, cell_offset + count))
            cell_offset += count
            start = end
        edges = np.concatenate(edge_parts)
        widths = np.diff(edges)
        centers = 0.5 * (edges[:-1] + edges[1:])
        layer_index = np.empty(widths.size, dtype=int)
        q_reference = np.empty(widths.size)
        initial_porosity = np.empty(widths.size)
        tortuosity = np.empty(widths.size)
        for index, (layer, layer_slice) in enumerate(zip(layers, slices)):
            layer_index[layer_slice] = index
            q_reference[layer_slice] = layer["initial_calcium_mol_per_m3"]
            initial_porosity[layer_slice] = layer["porosity"]
            tortuosity[layer_slice] = layer["tortuosity"]
        return cls(
            edges=edges,
            centers=centers,
            widths=widths,
            layer_index=layer_index,
            q_reference=q_reference,
            initial_porosity=initial_porosity,
            tortuosity=tortuosity,
            layer_slices=tuple(slices),
        )


@dataclass
class StageResult:
    stage: str
    time_s: np.ndarray
    global_time_s: np.ndarray
    liquid_amount: np.ndarray
    calcium: np.ndarray
    concentration: np.ndarray
    porosity: np.ndarray
    cumulative_inflow_mol_per_m2: np.ndarray
    cumulative_reaction_mol_per_m2: np.ndarray
    initial_calcium: np.ndarray
    initial_liquid_amount: np.ndarray
    solver_steps: int
    solver_success: bool
    solver_message: str

    @property
    def final_calcium(self) -> np.ndarray:
        return self.calcium[-1]



@dataclass
class ScenarioResult:
    edta_concentration_mol_per_l: float
    grid: Grid
    values: dict[str, Any]
    stages: dict[str, StageResult]
    native_area_resistance_ohm_m2: float

    def area_resistance(self, calcium: np.ndarray) -> float:
        return area_resistance_ohm_m2(self.grid, self.values, calcium)

    def stage_mass_residuals(self, stage_name: str) -> dict[str, float]:
        result = self.stages[stage_name]
        liquid = np.sum(result.liquid_amount * self.grid.widths[None, :], axis=1)
        liquid_residual = liquid - (
            result.cumulative_inflow_mol_per_m2 - result.cumulative_reaction_mol_per_m2
        )
        calcium_loss = np.sum(
            (result.initial_calcium[None, :] - result.calcium)
            * self.grid.widths[None, :],
            axis=1,
        )
        stage_values = self.values["stages"][stage_name]
        stoichiometry = stage_values["reagent_per_calcium_stoichiometry"]
        if stoichiometry > 0:
            calcium_residual = calcium_loss - result.cumulative_reaction_mol_per_m2 / stoichiometry
        else:
            calcium_residual = calcium_loss
        scale = max(1.0, float(np.max(result.cumulative_inflow_mol_per_m2)))
        return {
            "max_absolute_liquid_residual_mol_per_m2": float(np.max(np.abs(liquid_residual))),
            "max_absolute_calcium_residual_mol_per_m2": float(np.max(np.abs(calcium_residual))),
            "liquid_residual_relative_to_inflow": float(
                np.max(np.abs(liquid_residual)) / scale
            ),
        }


def porosity_from_calcium(grid: Grid, values: dict[str, Any], calcium: np.ndarray) -> np.ndarray:
    alpha = values["transport"]["porosity_per_released_calcium_m3_per_mol"]
    return grid.initial_porosity + alpha * (grid.q_reference - calcium)


def effective_diffusivity(
    grid: Grid, values: dict[str, Any], stage_name: str, calcium: np.ndarray
) -> np.ndarray:
    porosity = porosity_from_calcium(grid, values, calcium)
    bulk = values["stages"][stage_name]["bulk_diffusivity_m2_per_s"]
    return bulk * porosity / grid.tortuosity


def area_resistance_ohm_m2(
    grid: Grid, values: dict[str, Any], calcium: np.ndarray
) -> float:
    """Post-rinse layer-series resistance per unit area using an Archie prior."""
    electrical = values["electrical"]
    porosity = porosity_from_calcium(grid, values, calcium)
    resistance = 0.0
    for layer_slice in grid.layer_slices:
        thickness = float(np.sum(grid.widths[layer_slice]))
        mean_porosity = float(
            np.average(porosity[layer_slice], weights=grid.widths[layer_slice])
        )
        conductivity = (
            electrical["mineral_matrix_conductivity_s_per_m"]
            + electrical["measurement_fluid_conductivity_s_per_m"]
            * mean_porosity ** electrical["archie_exponent"]
        )
        resistance += thickness / conductivity
    return float(resistance)


def _jacobian_sparsity(n: int):
    sparse = lil_matrix((2 * n + 2, 2 * n + 2), dtype=int)
    for i in range(n):
        neighbors = {i}
        if i > 0:
            neighbors.add(i - 1)
        if i < n - 1:
            neighbors.add(i + 1)
        for j in neighbors:
            sparse[i, j] = 1
            sparse[i, n + j] = 1
        sparse[n + i, i] = 1
        sparse[n + i, n + i] = 1
    sparse[2 * n, 0] = 1
    sparse[2 * n, n] = 1
    for j in range(2 * n):
        sparse[2 * n + 1, j] = 1
    return sparse.tocsr()


def simulate_stage(
    grid: Grid,
    values: dict[str, Any],
    stage_name: str,
    initial_calcium: np.ndarray,
    initial_liquid_amount: np.ndarray | None = None,
    time_offset_s: float = 0.0,
    max_step_s: float | None = None,
    output_times_per_stage: int | None = None,
) -> StageResult:
    """Solve one stage with backward-differentiation adaptive time integration."""
    stage = values["stages"][stage_name]
    duration = float(stage["duration_s"])
    output_count = output_times_per_stage or values["numerics"]["output_times_per_stage"]
    times = np.linspace(0.0, duration, output_count)
    n = grid.centers.size
    a0 = (
        np.zeros(n)
        if initial_liquid_amount is None
        else np.asarray(initial_liquid_amount, dtype=float).copy()
    )
    q0 = np.asarray(initial_calcium, dtype=float).copy()
    y0 = np.concatenate([a0, q0, [0.0, 0.0]])
    alpha = values["transport"]["porosity_per_released_calcium_m3_per_mol"]
    k_f = values["transport"]["external_mass_transfer_m_per_s"]
    bulk_diffusivity = stage["bulk_diffusivity_m2_per_s"]
    bulk_concentration = stage["bulk_concentration_mol_per_l"] * 1000.0
    k_r = stage["k_r_m3_per_mol_s"]
    stoichiometry = stage["reagent_per_calcium_stoichiometry"]
    center_distances = np.diff(grid.centers)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        amount = state[:n]
        calcium = state[n : 2 * n]
        calcium_nonnegative = np.maximum(calcium, 0.0)
        porosity = grid.initial_porosity + alpha * (grid.q_reference - calcium)
        porosity = np.maximum(porosity, 1e-8)
        concentration = amount / porosity
        diffusivity = bulk_diffusivity * porosity / grid.tortuosity

        flux = np.zeros(n + 1)
        film_plus_half_cell = 1.0 / k_f + grid.widths[0] / (2.0 * diffusivity[0])
        flux[0] = (bulk_concentration - concentration[0]) / film_plus_half_cell
        if n > 1:
            face_diffusivity = (
                2.0
                * diffusivity[:-1]
                * diffusivity[1:]
                / np.maximum(diffusivity[:-1] + diffusivity[1:], 1e-30)
            )
            flux[1:n] = (
                -face_diffusivity
                * np.diff(concentration)
                / center_distances
            )
        diffusion = -(flux[1:] - flux[:-1]) / grid.widths

        reaction = k_r * np.maximum(concentration, 0.0) * calcium_nonnegative
        d_amount = diffusion - stoichiometry * reaction
        d_calcium = -reaction
        d_inflow = flux[0]
        d_reaction = stoichiometry * float(np.sum(reaction * grid.widths))
        return np.concatenate([d_amount, d_calcium, [d_inflow, d_reaction]])

    solution = solve_ivp(
        rhs,
        (0.0, duration),
        y0,
        method="BDF",
        t_eval=times,
        rtol=values["numerics"]["relative_tolerance"],
        atol=values["numerics"]["absolute_tolerance"],
        max_step=max_step_s or values["numerics"]["max_step_s"],
        jac_sparsity=_jacobian_sparsity(n),
    )
    if not solution.success:
        raise RuntimeError(f"{stage_name} solver failed: {solution.message}")

    amount = solution.y[:n].T
    calcium = solution.y[n : 2 * n].T
    porosity = np.maximum(
        grid.initial_porosity[None, :]
        + alpha * (grid.q_reference[None, :] - calcium),
        1e-8,
    )
    concentration = amount / porosity
    return StageResult(
        stage=stage_name,
        time_s=times,
        global_time_s=times + time_offset_s,
        liquid_amount=amount,
        calcium=calcium,
        concentration=concentration,
        porosity=porosity,
        cumulative_inflow_mol_per_m2=solution.y[2 * n],
        cumulative_reaction_mol_per_m2=solution.y[2 * n + 1],
        initial_calcium=q0,
        initial_liquid_amount=a0,
        solver_steps=len(solution.t),
        solver_success=bool(solution.success),
        solver_message=solution.message,
    )


def simulate_scenario(
    values: dict[str, Any],
    edta_concentration_mol_per_l: float,
    cells_per_layer: list[int] | None = None,
    max_step_s: float | None = None,
) -> ScenarioResult:
    """Run acid -> EDTA -> urea, with complete liquid clearing between stages."""
    import copy

    scenario_values = copy.deepcopy(values)
    scenario_values["stages"]["edta"]["bulk_concentration_mol_per_l"] = float(
        edta_concentration_mol_per_l
    )
    grid = Grid.from_parameters(scenario_values, cells_per_layer)
    calcium = grid.q_reference.copy()
    native_resistance = area_resistance_ohm_m2(grid, scenario_values, calcium)
    results: dict[str, StageResult] = {}
    offset = 0.0
    for stage_name in STAGE_NAMES:
        # Rinsing is represented by initial_liquid_amount=None, while q is retained.
        result = simulate_stage(
            grid,
            scenario_values,
            stage_name,
            calcium,
            initial_liquid_amount=None,
            time_offset_s=offset,
            max_step_s=max_step_s,
        )
        results[stage_name] = result
        calcium = result.final_calcium.copy()
        offset += scenario_values["stages"][stage_name]["duration_s"]
    return ScenarioResult(
        edta_concentration_mol_per_l=float(edta_concentration_mol_per_l),
        grid=grid,
        values=scenario_values,
        stages=results,
        native_area_resistance_ohm_m2=native_resistance,
    )


def layer_calcium_remaining_fraction(
    scenario: ScenarioResult, stage_name: str, layer_name: str | None = None
) -> float:
    result = scenario.stages[stage_name]
    if layer_name is None:
        selected = slice(None)
    else:
        selected = scenario.grid.layer_slices[LAYER_NAMES.index(layer_name)]
    widths = scenario.grid.widths[selected]
    initial = scenario.grid.q_reference[selected]
    final = result.final_calcium[selected]
    return float(np.sum(final * widths) / np.sum(initial * widths))


def layer_mean_porosity(
    scenario: ScenarioResult, stage_name: str, layer_name: str
) -> float:
    result = scenario.stages[stage_name]
    selected = scenario.grid.layer_slices[LAYER_NAMES.index(layer_name)]
    return float(
        np.average(result.porosity[-1, selected], weights=scenario.grid.widths[selected])
    )


def calcium_release_per_area(
    scenario: ScenarioResult, stage_name: str, layer_name: str | None = None
) -> float:
    result = scenario.stages[stage_name]
    selected = (
        slice(None)
        if layer_name is None
        else scenario.grid.layer_slices[LAYER_NAMES.index(layer_name)]
    )
    widths = scenario.grid.widths[selected]
    return float(
        np.sum(
            (result.initial_calcium[selected] - result.final_calcium[selected]) * widths
        )
    )
