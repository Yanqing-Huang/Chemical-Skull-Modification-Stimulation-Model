from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from cranial_model import LAYER_NAMES, STAGE_NAMES, load_parameters, simulate_scenario
from cranial_model.calibration import predict_observation
from cranial_model.core import (
    effective_diffusivity,
    layer_calcium_remaining_fraction,
    layer_parameter_values,
    porosity_from_calcium,
    reaction_rate_mol_per_m3_s,
)
from cranial_model.diagnostics import (
    inner_scm_flux_mol_per_m2_s,
    inner_scm_time_constant_driver_s,
    outer_scm_flux_mol_per_m2_s,
    outer_scm_time_to_front_s,
)
from run_prior_mechanism import extract_scenario_series


ROOT = Path(__file__).resolve().parents[1]


class CranialModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.document = load_parameters(ROOT / "parameters" / "theoretical_prior.json")
        cls.values = copy.deepcopy(cls.document["values"])
        cls.values["numerics"]["cells_per_layer"] = [8, 5, 5]
        cls.values["numerics"]["max_step_s"] = 5.0
        cls.values["numerics"]["output_times_per_stage"] = 7
        for stage in cls.values["stages"].values():
            stage["duration_s"] = 30.0
        for segment in cls.values["stages"]["edta"]["segments"]:
            segment["duration_s"] = 7.5
        cls.scenario = simulate_scenario(cls.values, 0.1)

    def test_mass_conservation_and_stage_transfer(self) -> None:
        for stage in STAGE_NAMES:
            residual = self.scenario.stage_mass_residuals(stage)
            self.assertLess(residual["liquid_residual_relative_to_inflow"], 1e-6)
            self.assertLess(residual["max_absolute_calcium_residual_mol_per_m2"], 1e-8)
        np.testing.assert_array_equal(
            self.scenario.stages["edta"].initial_calcium,
            self.scenario.stages["acetic_acid"].final_calcium,
        )
        np.testing.assert_array_equal(
            self.scenario.stages["urea"].initial_calcium,
            self.scenario.stages["edta"].final_calcium,
        )
        for stage in STAGE_NAMES:
            self.assertEqual(
                float(np.max(np.abs(self.scenario.stages[stage].initial_liquid_amount))),
                0.0,
            )

    def test_edta_fresh_bath_boundaries_clear_liquid_and_retain_calcium(self) -> None:
        edta = self.scenario.stages["edta"]
        for boundary in (7.5, 15.0, 22.5):
            indices = np.flatnonzero(np.isclose(edta.time_s, boundary))
            self.assertEqual(indices.size, 2)
            before, after = indices
            np.testing.assert_array_equal(edta.calcium[before], edta.calcium[after])
            np.testing.assert_array_equal(edta.porosity[before], edta.porosity[after])
            np.testing.assert_array_equal(edta.liquid_amount[after], np.zeros(self.scenario.grid.centers.size))
            self.assertGreater(edta.cumulative_liquid_cleared_mol_per_m2[after], 0.0)
        self.assertEqual(float(edta.time_s[-1]), 30.0)

    def test_urea_is_nonreactive_and_states_remain_physical(self) -> None:
        urea = self.scenario.stages["urea"]
        np.testing.assert_array_equal(urea.initial_calcium, urea.final_calcium)
        self.assertGreater(float(urea.concentration[-1].max()), 0.0)
        for stage in STAGE_NAMES:
            result = self.scenario.stages[stage]
            self.assertGreaterEqual(float(result.concentration.min()), -1e-6)
            self.assertGreaterEqual(float(result.calcium.min()), -1e-8)
            self.assertGreater(float(result.porosity.min()), 0.0)
            self.assertLessEqual(float(result.porosity.max()), 0.95)

    def test_protocol_boundaries_acid_outer_only_edta_immersed(self) -> None:
        acid = simulate_scenario(self.values, 0.1, stage_names=("acetic_acid",))
        inner_after_acid = layer_calcium_remaining_fraction(
            acid, "acetic_acid", "inner_table"
        )
        self.assertGreater(inner_after_acid, 0.999999)

        immersed_acid_values = copy.deepcopy(self.values)
        immersed_acid_values["stages"]["acetic_acid"]["boundary_mode"] = "immersed"
        immersed_acid = simulate_scenario(
            immersed_acid_values, 0.1, stage_names=("acetic_acid",)
        )
        self.assertLess(
            layer_calcium_remaining_fraction(
                immersed_acid, "acetic_acid", "inner_table"
            ),
            inner_after_acid,
        )

        edta = self.scenario.stages["edta"]
        self.assertLess(edta.final_calcium[-1], edta.initial_calcium[-1])
        urea = self.scenario.stages["urea"]
        self.assertGreater(urea.concentration[-1, -1], 0.0)
        for stage_name in ("acetic_acid", "edta", "urea"):
            residual = self.scenario.stage_mass_residuals(stage_name)
            self.assertLess(residual["liquid_residual_relative_to_inflow"], 1e-6)
            self.assertLess(residual["max_absolute_calcium_residual_mol_per_m2"], 1e-8)

    def test_immersed_inner_front_is_measured_from_inner_surface(self) -> None:
        scenario = copy.deepcopy(self.scenario)
        scenario.values["stages"]["acetic_acid"]["boundary_mode"] = "immersed"
        scenario.stages = {"acetic_acid": scenario.stages["acetic_acid"]}
        result = scenario.stages["acetic_acid"]
        result.time_s[-1] = 300.0  # satisfy the runner's bench-time extraction
        result.calcium[:] = scenario.grid.q_reference[None, :]
        inner = scenario.grid.layer_slices[2]
        result.calcium[-1, inner.start : inner.stop] = scenario.grid.q_reference[inner]
        result.calcium[-1, inner.stop - 1] = 0.9 * scenario.grid.q_reference[inner][-1]
        series = extract_scenario_series(scenario)
        expected_mm = (
            scenario.grid.edges[-1] - scenario.grid.centers[-1]
        ) * 1000.0
        self.assertAlmostEqual(
            series["acetic_acid__front_mm__inner_table"][-1], expected_mm, places=12
        )

    def test_effective_diffusivity_contains_porosity_once(self) -> None:
        grid = self.scenario.grid
        calcium = grid.q_reference
        calculated = effective_diffusivity(grid, self.values, "edta", calcium)
        expected = (
            self.values["stages"]["edta"]["bulk_diffusivity_m2_per_s"]
            * grid.initial_porosity
            / grid.tortuosity
        )
        np.testing.assert_allclose(calculated, expected, rtol=1e-14)

    def test_area_resistance_uses_local_series_cells(self) -> None:
        grid = self.scenario.grid
        calcium = grid.q_reference.copy()
        calcium[0] *= 0.5
        porosity = porosity_from_calcium(grid, self.values, calcium)
        electrical = self.values["electrical"]
        conductivity = (
            electrical["mineral_matrix_conductivity_s_per_m"]
            + electrical["measurement_fluid_conductivity_s_per_m"]
            * porosity ** electrical["archie_exponent"]
        )
        expected = float(np.sum(grid.widths / conductivity))
        self.assertAlmostEqual(self.scenario.area_resistance(calcium), expected, places=12)
        layer_mean_estimate = 0.0
        for layer_slice in grid.layer_slices:
            mean_porosity = float(np.average(porosity[layer_slice], weights=grid.widths[layer_slice]))
            mean_conductivity = (
                electrical["mineral_matrix_conductivity_s_per_m"]
                + electrical["measurement_fluid_conductivity_s_per_m"]
                * mean_porosity ** electrical["archie_exponent"]
            )
            layer_mean_estimate += float(np.sum(grid.widths[layer_slice])) / mean_conductivity
        self.assertGreater(abs(expected - layer_mean_estimate) / expected, 1e-4)

    def test_decalcification_increases_porosity(self) -> None:
        grid = self.scenario.grid
        final = self.scenario.stages["edta"].final_calcium
        initial_porosity = porosity_from_calcium(grid, self.values, grid.q_reference)
        final_porosity = porosity_from_calcium(grid, self.values, final)
        self.assertTrue(np.all(final_porosity >= initial_porosity))
        self.assertLess(
            self.scenario.area_resistance(final),
            self.scenario.native_area_resistance_ohm_m2,
        )

    def test_grain_scm_starts_at_second_order_rate_and_caps_porosity(self) -> None:
        grid = self.scenario.grid
        concentration = np.full(grid.centers.size, 100.0)
        diffusivity = effective_diffusivity(grid, self.values, "edta", grid.q_reference)
        initial = reaction_rate_mol_per_m3_s(
            grid, self.values, "edta", concentration, grid.q_reference, diffusivity
        )
        expected = (
            layer_parameter_values(grid, self.values["stages"]["edta"]["k_r_m3_per_mol_s"])
            * concentration
            * grid.q_reference
        )
        np.testing.assert_allclose(initial, expected, rtol=1e-12)
        converted = grid.q_reference * 0.5
        later = reaction_rate_mol_per_m3_s(
            grid, self.values, "edta", concentration, converted, diffusivity
        )
        self.assertTrue(np.all(later >= 0.0))
        self.assertTrue(np.all(later <= expected))
        without_product = copy.deepcopy(self.values)
        without_product["transport"]["scm_include_product_layer_resistance"] = False
        surface_only = reaction_rate_mol_per_m3_s(
            grid, without_product, "edta", concentration, converted, diffusivity
        )
        self.assertTrue(np.all(surface_only >= later))
        self.assertTrue(np.all(surface_only <= expected))
        depleted = np.zeros_like(grid.q_reference)
        porosity = porosity_from_calcium(grid, self.values, depleted)
        self.assertLessEqual(float(porosity.max()), self.values["transport"]["maximum_porosity"])

    def test_per_layer_rate_constant_broadcast(self) -> None:
        grid = self.scenario.grid
        values = copy.deepcopy(self.values)
        values["transport"]["compact_layer_model"] = "none"
        values["stages"]["edta"].pop("surface_site_capacity_mol_per_m3", None)
        per_layer = {"outer_table": 1e-6, "diploe": 2e-6, "inner_table": 3e-6}
        values["stages"]["edta"]["k_r_m3_per_mol_s"] = per_layer
        concentration = np.full(grid.centers.size, 50.0)
        diffusivity = effective_diffusivity(grid, values, "edta", grid.q_reference)
        rate = reaction_rate_mol_per_m3_s(
            grid, values, "edta", concentration, grid.q_reference, diffusivity
        )
        for name, layer_slice in zip(LAYER_NAMES, grid.layer_slices):
            np.testing.assert_allclose(
                rate[layer_slice],
                per_layer[name] * 50.0 * grid.q_reference[layer_slice],
                rtol=1e-12,
            )

    def test_edta_site_capacity_decelerates_and_caps(self) -> None:
        grid = self.scenario.grid
        values = copy.deepcopy(self.values)
        values["transport"]["compact_layer_model"] = "none"
        capacity_spec = {"outer_table": 1000.0, "diploe": 500.0, "inner_table": 800.0}
        values["stages"]["edta"]["surface_site_capacity_mol_per_m3"] = capacity_spec
        capacity = layer_parameter_values(grid, capacity_spec)
        concentration = np.full(grid.centers.size, 100.0)
        diffusivity = effective_diffusivity(grid, values, "edta", grid.q_reference)
        full = reaction_rate_mol_per_m3_s(
            grid, values, "edta", concentration, grid.q_reference, diffusivity
        )
        k_r_cells = layer_parameter_values(grid, values["stages"]["edta"]["k_r_m3_per_mol_s"])
        q_half = grid.q_reference - 0.5 * capacity
        half = reaction_rate_mol_per_m3_s(
            grid, values, "edta", concentration, q_half, diffusivity
        )
        np.testing.assert_allclose(half, 0.5 * k_r_cells * concentration * q_half, rtol=1e-12)
        self.assertTrue(np.all(half < 0.5 * full))
        exhausted = reaction_rate_mol_per_m3_s(
            grid, values, "edta", concentration, grid.q_reference - 2.0 * capacity, diffusivity
        )
        np.testing.assert_allclose(exhausted, 0.0, atol=1e-30)

    def test_outer_scm_against_ode(self) -> None:
        concentration, k_f, d_e, k_s = 1000.0, 1e-5, 1e-10, 1e-7
        stoichiometry, calcium, target = 2.0, 10000.0, 1e-5
        end_time = float(
            outer_scm_time_to_front_s(
                target, concentration, k_f, d_e, k_s, stoichiometry, calcium
            )
        )

        def rhs(_time, state):
            flux = outer_scm_flux_mol_per_m2_s(concentration, k_f, state[0], d_e, k_s)
            return [flux / (stoichiometry * calcium)]

        solution = solve_ivp(
            rhs,
            (0.0, end_time),
            [0.0],
            t_eval=[end_time],
            rtol=1e-11,
            atol=1e-15,
            max_step=end_time / 100.0,
        )
        self.assertAlmostEqual(float(solution.y[0, -1]), target, delta=target * 1e-7)

    def test_inner_scm_uses_interface_driver(self) -> None:
        concentration, d_e, k_s = 250.0, 1e-10, 1e-7
        stoichiometry, calcium, target = 1.0, 10000.0, 1e-5
        self.assertEqual(inner_scm_flux_mol_per_m2_s(0.0, target, d_e, k_s), 0.0)
        end_time = float(
            inner_scm_time_constant_driver_s(
                target, concentration, d_e, k_s, stoichiometry, calcium
            )
        )

        def rhs(_time, state):
            flux = inner_scm_flux_mol_per_m2_s(concentration, state[0], d_e, k_s)
            return [flux / (stoichiometry * calcium)]

        solution = solve_ivp(
            rhs,
            (0.0, end_time),
            [0.0],
            t_eval=[end_time],
            rtol=1e-11,
            atol=1e-15,
            max_step=end_time / 100.0,
        )
        self.assertAlmostEqual(float(solution.y[0, -1]), target, delta=target * 1e-7)

    def test_experimental_observation_prediction_interface(self) -> None:
        observation = {
            "stage": "edta",
            "time_s": 30.0,
            "layer": "outer_table",
            "observable": "ca_remaining_fraction",
        }
        predicted = predict_observation(self.scenario, observation)
        expected = layer_calcium_remaining_fraction(self.scenario, "edta", "outer_table")
        self.assertAlmostEqual(predicted, expected, places=12)


if __name__ == "__main__":
    unittest.main()
