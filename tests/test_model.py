from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

from cranial_model import STAGE_NAMES, load_parameters, simulate_scenario
from cranial_model.calibration import predict_observation
from cranial_model.core import (
    effective_diffusivity,
    layer_calcium_remaining_fraction,
    porosity_from_calcium,
)
from cranial_model.diagnostics import (
    inner_scm_flux_mol_per_m2_s,
    inner_scm_time_constant_driver_s,
    outer_scm_flux_mol_per_m2_s,
    outer_scm_time_to_front_s,
)


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

    def test_urea_is_nonreactive_and_states_remain_physical(self) -> None:
        urea = self.scenario.stages["urea"]
        np.testing.assert_array_equal(urea.initial_calcium, urea.final_calcium)
        self.assertGreater(float(urea.concentration[-1].max()), 0.0)
        for stage in STAGE_NAMES:
            result = self.scenario.stages[stage]
            self.assertGreaterEqual(float(result.concentration.min()), -1e-6)
            self.assertGreaterEqual(float(result.calcium.min()), -1e-8)
            self.assertGreater(float(result.porosity.min()), 0.0)
            self.assertLess(float(result.porosity.max()), 0.95)

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
