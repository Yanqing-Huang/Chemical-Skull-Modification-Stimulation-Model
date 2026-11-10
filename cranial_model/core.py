"""Three-layer, sequential diffusion-reaction model for an isolated SD-rat skull.

Liquid concentration c is per pore-liquid volume. Calcium q is per total bone
volume. The conserved liquid variable is a = porosity * c, per total bone volume.
Effective diffusivity D_e = D_bulk * porosity / tortuosity already contains
porosity and is therefore not multiplied by porosity again in the flux.
Stage-specific stages.<name>.boundary_mode selects immersed bath contact at
both skull surfaces or outer_only application with zero inner reagent flux;
transport.boundary_mode remains a legacy default when a stage has no override.
"""

from __future__ import annotations

import copy
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
    # Per-outer-area cell volumes and local area ratios. Both default to the
    # planar grid; a finite outer curvature radius defines a convex shell.
    cell_volumes: np.ndarray
    cell_areas: np.ndarray
    face_areas: np.ndarray

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
        radius = values["geometry"].get("outer_curvature_radius_m")
        if radius is None:
            face_areas = np.ones_like(edges)
            cell_areas = np.ones_like(centers)
            cell_volumes = widths.copy()
        else:
            radius = float(radius)
            if not np.isfinite(radius) or radius <= float(edges[-1]):
                raise ValueError("Outer curvature radius must exceed total skull thickness")
            # Convex shell coordinate x points inward from the outer surface.
            # A(x)/A_outer = (1-x/R)^2; integrate it exactly over each cell.
            face_areas = (1.0 - edges / radius) ** 2
            cell_areas = (1.0 - centers / radius) ** 2
            cell_volumes = widths * (
                1.0 - (edges[:-1] + edges[1:]) / radius
                + (edges[:-1] ** 2 + edges[:-1] * edges[1:] + edges[1:] ** 2)
                / (3.0 * radius**2)
            )
        return cls(
            edges=edges,
            centers=centers,
            widths=widths,
            layer_index=layer_index,
            q_reference=q_reference,
            initial_porosity=initial_porosity,
            tortuosity=tortuosity,
            layer_slices=tuple(slices),
            cell_volumes=cell_volumes,
            cell_areas=cell_areas,
            face_areas=face_areas,
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
    cumulative_liquid_cleared_mol_per_m2: np.ndarray
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
        liquid = np.sum(result.liquid_amount * self.grid.cell_volumes[None, :], axis=1)
        liquid_residual = liquid + result.cumulative_liquid_cleared_mol_per_m2 - (
            result.cumulative_inflow_mol_per_m2 - result.cumulative_reaction_mol_per_m2
        )
        calcium_loss = np.sum(
            (result.initial_calcium[None, :] - result.calcium)
            * self.grid.cell_volumes[None, :],
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


def layer_parameter_values(grid: Grid, spec: Any) -> np.ndarray:
    """Broadcast a scalar or a {layer_name: value} mapping onto a per-cell array."""
    if isinstance(spec, dict):
        missing = [name for name in LAYER_NAMES if name not in spec]
        if missing:
            raise ValueError(f"Per-layer parameter missing layers: {missing}")
        values = np.empty(grid.centers.size)
        for index, layer_slice in enumerate(grid.layer_slices):
            values[layer_slice] = float(spec[LAYER_NAMES[index]])
        return values
    return np.full(grid.centers.size, float(spec))


def porosity_from_calcium(grid: Grid, values: dict[str, Any], calcium: np.ndarray) -> np.ndarray:
    alpha = values["transport"]["porosity_per_released_calcium_m3_per_mol"]
    maximum = values["transport"].get("maximum_porosity", 0.95)
    porosity = grid.initial_porosity + alpha * (grid.q_reference - calcium)
    return np.clip(porosity, 1e-8, maximum)


def reaction_rate_mol_per_m3_s(
    grid: Grid,
    values: dict[str, Any],
    stage_name: str,
    concentration: np.ndarray,
    calcium: np.ndarray,
    diffusivity: np.ndarray,
) -> np.ndarray:
    """Fick-reaction in diploe; optional grain-scale SCM in the compact tables.

    ``k_r_m3_per_mol_s`` may be a scalar or a per-layer mapping. If the stage
    defines ``surface_site_capacity_mol_per_m3`` (Tiselius-type finite
    exchangeable surface-calcium pool, used for neutral EDTA chelation), the
    rate is multiplied by the remaining surface-site fraction
    ``clip(1 - (q_ref - q) / C_site_max, 0, 1)`` so chelation decelerates as
    the accessible site pool is exhausted.
    """
    stage = values["stages"][stage_name]
    k_r_cells = layer_parameter_values(grid, stage["k_r_m3_per_mol_s"])
    concentration = np.maximum(concentration, 0.0)
    calcium = np.maximum(calcium, 0.0)
    capacity_spec = stage.get("surface_site_capacity_mol_per_m3")
    if capacity_spec is not None:
        capacity = np.maximum(
            layer_parameter_values(grid, capacity_spec), 1e-30
        )
        site_fraction = np.clip(
            1.0 - (grid.q_reference - calcium) / capacity, 0.0, 1.0
        )
    else:
        site_fraction = np.ones(grid.centers.size)
    reaction = k_r_cells * concentration * calcium * site_fraction
    if not np.any(k_r_cells):
        return reaction
    if values["transport"].get("compact_layer_model") != "shrinking_core_grain":
        return reaction

    radius = values["transport"]["scm_grain_radius_m"]
    for layer_index in (0, 2):
        selected = grid.layer_slices[layer_index]
        q_reference = grid.q_reference[selected]
        initial_porosity = grid.initial_porosity[selected]
        solid_calcium = q_reference / (1.0 - initial_porosity)
        surface_reaction = k_r_cells[selected] * solid_calcium * radius / 3.0
        core_radius = radius * np.cbrt(np.clip(calcium[selected] / q_reference, 0.0, 1.0))
        safe_radius = np.maximum(core_radius, radius * 1e-12)
        grains_per_bulk_volume = 3.0 * (1.0 - initial_porosity) / (4.0 * np.pi * radius**3)
        reaction_resistance = 1.0 / (safe_radius**2 * surface_reaction)
        if values["transport"].get("scm_include_product_layer_resistance", True):
            product_resistance = (radius - safe_radius) / (
                np.maximum(diffusivity[selected], 1e-30) * radius * safe_radius
            )
        else:
            product_resistance = 0.0
        grain_rate = 4.0 * np.pi * concentration[selected] / (
            reaction_resistance + product_resistance
        )
        scm_rate = grains_per_bulk_volume * grain_rate
        scm_rate = scm_rate * site_fraction[selected]
        reaction[selected] = np.where(calcium[selected] > 0.0, scm_rate, 0.0)
    return reaction


def effective_diffusivity(
    grid: Grid, values: dict[str, Any], stage_name: str, calcium: np.ndarray
) -> np.ndarray:
    porosity = porosity_from_calcium(grid, values, calcium)
    bulk = values["stages"][stage_name]["bulk_diffusivity_m2_per_s"]
    return bulk * porosity / grid.tortuosity


def area_resistance_ohm_m2(
    grid: Grid, values: dict[str, Any], calcium: np.ndarray
) -> float:
    """Post-rinse cell-series resistance per unit area using an Archie prior."""
    electrical = values["electrical"]
    porosity = porosity_from_calcium(grid, values, calcium)
    conductivity = (
        electrical["mineral_matrix_conductivity_s_per_m"]
        + electrical["measurement_fluid_conductivity_s_per_m"]
        * porosity ** electrical["archie_exponent"]
    )
    # Cells are series resistances; retain the spatial decalcification gradient.
    # In a convex shell the inward area decreases, so dR/A_outer = dx/(sigma*A/A_outer).
    return float(np.sum(grid.widths / (conductivity * grid.cell_areas)))


def _jacobian_sparsity(n: int, immersed: bool = False):
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
    if immersed:
        # Net inflow also depends on the inner-surface amount and calcium
        # (porosity-dependent diffusivity enters the inner boundary flux).
        sparse[2 * n, n - 1] = 1
        sparse[2 * n, 2 * n - 1] = 1
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
    k_f = values["transport"]["external_mass_transfer_m_per_s"]
    immersed = stage.get(
        "boundary_mode", values["transport"].get("boundary_mode", "outer_only")
    ) == "immersed"
    bulk_diffusivity = stage["bulk_diffusivity_m2_per_s"]
    bulk_concentration = stage["bulk_concentration_mol_per_l"] * 1000.0
    stoichiometry = stage["reagent_per_calcium_stoichiometry"]
    center_distances = np.diff(grid.centers)

    def rhs(_time: float, state: np.ndarray) -> np.ndarray:
        amount = state[:n]
        calcium = state[n : 2 * n]
        calcium_nonnegative = np.maximum(calcium, 0.0)
        porosity = porosity_from_calcium(grid, values, calcium)
        concentration = amount / porosity
        diffusivity = bulk_diffusivity * porosity / grid.tortuosity

        flux = np.zeros(n + 1)
        film_plus_half_cell = 1.0 / k_f + grid.widths[0] / (2.0 * diffusivity[0])
        flux[0] = (bulk_concentration - concentration[0]) / film_plus_half_cell
        if immersed:
            # Fully immersed in-vitro configuration: the inner table surface
            # contacts the same bath through an identical film + half-cell
            # resistance. Positive flux[n] carries solute out of the last cell.
            inner_resistance = 1.0 / k_f + grid.widths[-1] / (2.0 * diffusivity[-1])
            flux[n] = (concentration[-1] - bulk_concentration) / inner_resistance
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
        area_flux = grid.face_areas * flux
        diffusion = -(area_flux[1:] - area_flux[:-1]) / grid.cell_volumes

        reaction = reaction_rate_mol_per_m3_s(
            grid, values, stage_name, concentration, calcium_nonnegative, diffusivity
        )
        d_amount = diffusion - stoichiometry * reaction
        d_calcium = -reaction
        d_inflow = area_flux[0] - area_flux[n]
        d_reaction = stoichiometry * float(np.sum(reaction * grid.cell_volumes))
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
        jac_sparsity=_jacobian_sparsity(n, immersed),
    )
    if not solution.success:
        raise RuntimeError(f"{stage_name} solver failed: {solution.message}")

    amount = solution.y[:n].T
    calcium = solution.y[n : 2 * n].T
    porosity = porosity_from_calcium(grid, values, calcium)
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
        cumulative_liquid_cleared_mol_per_m2=np.zeros_like(times),
        initial_calcium=q0,
        initial_liquid_amount=a0,
        solver_steps=len(solution.t),
        solver_success=bool(solution.success),
        solver_message=solution.message,
    )


def _simulate_segmented_stage(
    grid: Grid,
    scenario_values: dict[str, Any],
    stage_name: str,
    initial_calcium: np.ndarray,
    time_offset_s: float,
    max_step_s: float | None,
) -> StageResult:
    """Simulate fresh-bath segments with PBS-clearing boundaries.

    Liquid reagent is cleared at every boundary; calcium and porosity continue.
    Duplicate time points retain both sides of each liquid discontinuity.
    """
    stage = scenario_values["stages"][stage_name]
    segments = stage["segments"]
    duration = sum(float(segment["duration_s"]) for segment in segments)
    if abs(duration - stage["duration_s"]) > 1e-9:
        raise ValueError(f"{stage_name} segment durations must sum to duration_s")

    results: list[StageResult] = []
    calcium = initial_calcium.copy()
    for segment in segments:
        segment_values = copy.deepcopy(scenario_values)
        segment_values["stages"][stage_name]["duration_s"] = float(segment["duration_s"])
        result = simulate_stage(
            grid,
            segment_values,
            stage_name,
            calcium,
            initial_liquid_amount=None,
            time_offset_s=0.0,
            max_step_s=max_step_s,
        )
        results.append(result)
        calcium = result.final_calcium

    time_parts = []
    inflow_parts = []
    reaction_parts = []
    cleared_parts = []
    elapsed = 0.0
    inflow_offset = 0.0
    reaction_offset = 0.0
    cleared_offset = 0.0
    previous: StageResult | None = None
    for result in results:
        if previous is not None:
            cleared_offset += float(np.sum(previous.liquid_amount[-1] * grid.cell_volumes))
        time_parts.append(result.time_s + elapsed)
        inflow_parts.append(result.cumulative_inflow_mol_per_m2 + inflow_offset)
        reaction_parts.append(result.cumulative_reaction_mol_per_m2 + reaction_offset)
        cleared_parts.append(np.full(result.time_s.shape, cleared_offset))
        elapsed += float(result.time_s[-1])
        inflow_offset += float(result.cumulative_inflow_mol_per_m2[-1])
        reaction_offset += float(result.cumulative_reaction_mol_per_m2[-1])
        previous = result

    combined_time = np.concatenate(time_parts)
    return StageResult(
        stage=stage_name,
        time_s=combined_time,
        global_time_s=combined_time + time_offset_s,
        liquid_amount=np.concatenate([result.liquid_amount for result in results], axis=0),
        calcium=np.concatenate([result.calcium for result in results], axis=0),
        concentration=np.concatenate([result.concentration for result in results], axis=0),
        porosity=np.concatenate([result.porosity for result in results], axis=0),
        cumulative_inflow_mol_per_m2=np.concatenate(inflow_parts),
        cumulative_reaction_mol_per_m2=np.concatenate(reaction_parts),
        cumulative_liquid_cleared_mol_per_m2=np.concatenate(cleared_parts),
        initial_calcium=initial_calcium.copy(),
        initial_liquid_amount=np.zeros_like(initial_calcium),
        solver_steps=sum(result.solver_steps for result in results),
        solver_success=all(result.solver_success for result in results),
        solver_message="; ".join(result.solver_message for result in results),
    )


def simulate_scenario(
    values: dict[str, Any],
    edta_concentration_mol_per_l: float,
    cells_per_layer: list[int] | None = None,
    max_step_s: float | None = None,
    stage_names: tuple[str, ...] = STAGE_NAMES,
) -> ScenarioResult:
    """Run selected sequential stages, with complete liquid clearing between stages."""
    scenario_values = copy.deepcopy(values)
    scenario_values["stages"]["edta"]["bulk_concentration_mol_per_l"] = float(
        edta_concentration_mol_per_l
    )
    grid = Grid.from_parameters(scenario_values, cells_per_layer)
    calcium = grid.q_reference.copy()
    native_resistance = area_resistance_ohm_m2(grid, scenario_values, calcium)
    results: dict[str, StageResult] = {}
    offset = 0.0
    for stage_name in stage_names:
        stage = scenario_values["stages"][stage_name]
        # Rinsing is represented by initial_liquid_amount=None, while q is retained.
        if stage.get("segments"):
            result = _simulate_segmented_stage(
                grid,
                scenario_values,
                stage_name,
                calcium,
                time_offset_s=offset,
                max_step_s=max_step_s,
            )
        else:
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
