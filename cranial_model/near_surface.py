"""Near-surface state extraction for the Track B v4 observation layer.

All quantities are extracted after the legacy mechanism solve. No auxiliary
output time or exposure quadrature is allowed to alter the mechanism trajectory.
"""

from __future__ import annotations

import numpy as np

from run_prior_mechanism import MEASURED_STAGE_TIMES_S


DEPTHS_UM = (50, 100, 200)
STRUCTURES = tuple(
    f"{side}_{depth}um"
    for side in ("two_sided", "outer_only")
    for depth in DEPTHS_UM
)
EXPOSURE_TIME_SCALE_S = 5700.0


def surface_indices(grid, layer_index: int, depth_um: int) -> np.ndarray:
    """Cell indices within a fixed depth of the outer or inner free surface."""
    layer = grid.layer_slices[layer_index]
    depth_m = depth_um * 1e-6
    if layer_index == 0:
        local = np.flatnonzero(grid.edges[layer.start:layer.stop] < depth_m)
    elif layer_index == 2:
        total = float(grid.edges[-1])
        local = np.flatnonzero(grid.edges[layer.start + 1:layer.stop + 1] > total - depth_m)
    else:
        raise ValueError("Near-surface states are defined only for outer/inner tables")
    if local.size == 0:
        local = np.array([0 if layer_index == 0 else layer.stop - layer.start - 1])
    return layer.start + local


def _weighted_surface(values: np.ndarray, grid, indices: np.ndarray) -> float:
    return float(np.average(values[indices], weights=grid.cell_volumes[indices]))


def _measurement_index(result, time_s: float) -> int:
    index = int(np.argmin(np.abs(result.time_s - time_s)))
    if abs(float(result.time_s[index]) - time_s) > 1e-6:
        raise RuntimeError(f"Missing mechanism measurement time {time_s}")
    return index


def surface_porosity_ratio(scenario, stage_name: str, time_s: float, depth_um: int) -> tuple[float, float]:
    """Outer and inner near-surface porosity ratios at a bench measurement."""
    result = scenario.stages[stage_name]
    index = _measurement_index(result, time_s)
    porosity = result.porosity[index]
    ratios = []
    for layer_index in (0, 2):
        indices = surface_indices(scenario.grid, layer_index, depth_um)
        current = _weighted_surface(porosity, scenario.grid, indices)
        native = _weighted_surface(scenario.grid.initial_porosity, scenario.grid, indices)
        ratios.append(current / native)
    return float(ratios[0]), float(ratios[1])


def _integrated_relative_exposure(scenario, stage_name: str, stop_index: int, indices: np.ndarray) -> float:
    """Trapezoidal integral of c/C_bulk on the existing solver output grid."""
    total = 0.0
    for name, result in scenario.stages.items():
        if name not in ("acetic_acid", "edta"):
            continue
        stop = stop_index if name == stage_name else len(result.time_s) - 1
        times = result.time_s[:stop + 1]
        concentration = result.concentration[:stop + 1]
        surface_c = np.array([
            _weighted_surface(row, scenario.grid, indices) for row in concentration
        ])
        bulk = float(scenario.values["stages"][name]["bulk_concentration_mol_per_l"]) * 1000.0
        total += float(np.trapezoid(surface_c / bulk, times))
        if name == stage_name:
            break
    return max(total, 0.0)


def cumulative_relative_exposure(scenario, stage_name: str, time_s: float, depth_um: int) -> tuple[float, float]:
    """Outer/inner cumulative exposure, normalized to the acid+EDTA duration."""
    result = scenario.stages[stage_name]
    index = _measurement_index(result, time_s)
    values = []
    for layer_index in (0, 2):
        indices = surface_indices(scenario.grid, layer_index, depth_um)
        integral = _integrated_relative_exposure(scenario, stage_name, index, indices)
        values.append(integral / EXPOSURE_TIME_SCALE_S)
    return float(values[0]), float(values[1])


def condition_near_surface_states(scenario, dose: float) -> list[dict]:
    """Extract fixed mechanism and all v4 observation states after solving."""
    from run_prior_mechanism import extract_scenario_series

    legacy = extract_scenario_series(scenario)
    rows = []
    for stage_name in ("acetic_acid", "edta"):
        for time_s in MEASURED_STAGE_TIMES_S[stage_name]:
            row = {
                "stage": stage_name,
                "time_s": float(time_s),
                "edta_concentration_mol_per_l": float(dose),
                "sim_whole_resistance_ratio": float(legacy[f"cond__{stage_name}__{time_s:.0f}"]),
            }
            for depth in DEPTHS_UM:
                outer, inner = surface_porosity_ratio(scenario, stage_name, time_s, depth)
                row[f"sim_outer_porosity_ratio_{depth}um"] = outer
                row[f"sim_inner_porosity_ratio_{depth}um"] = inner
                outer_xi, inner_xi = cumulative_relative_exposure(
                    scenario, stage_name, time_s, depth
                )
                row[f"sim_outer_exposure_{depth}um"] = outer_xi
                row[f"sim_inner_exposure_{depth}um"] = inner_xi
            rows.append(row)
    return rows


def state_columns(structure: str, state: str) -> tuple[str, ...]:
    """Return column names for porosity_ratio or exposure state."""
    if state not in ("porosity_ratio", "exposure"):
        raise ValueError("state must be porosity_ratio or exposure")
    for side in ("two_sided", "outer_only"):
        for depth in DEPTHS_UM:
            if structure == f"{side}_{depth}um":
                columns = [f"sim_outer_{state}_{depth}um"]
                if side == "two_sided":
                    columns.append(f"sim_inner_{state}_{depth}um")
                return tuple(columns)
    raise ValueError(f"Unknown structure: {structure}")


def combine_structure_state(frame, structure: str, state: str) -> np.ndarray:
    """Outer-only state or the equal outer/inner mean for two-sided structures."""
    columns = state_columns(structure, state)
    return frame.loc[:, list(columns)].to_numpy(dtype=float).mean(axis=1)
