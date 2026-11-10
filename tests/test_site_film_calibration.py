"""Numerical and observation-layer tests for the V5a site-film diagnostic."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np

from cranial_model import STAGE_NAMES, load_parameters, simulate_scenario
from cranial_model.core import Grid
from cranial_model.site_film_calibration import rat_condition_mean, site_film_forward
from cranial_model.site_geometry import apply_site_geometry, sample_site_geometries


ROOT = Path(__file__).resolve().parents[1]


class CurvedShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = copy.deepcopy(
            load_parameters(ROOT / "parameters" / "theoretical_prior.json")["values"]
        )
        cls.values["numerics"]["cells_per_layer"] = [8, 5, 5]
        cls.values["numerics"]["max_step_s"] = 5.0
        cls.values["numerics"]["output_times_per_stage"] = 7
        for stage in cls.values["stages"].values():
            stage["duration_s"] = 30.0
        for segment in cls.values["stages"]["edta"]["segments"]:
            segment["duration_s"] = 7.5

    def test_planar_defaults_and_exact_curved_cell_volumes(self) -> None:
        planar = Grid.from_parameters(self.values)
        np.testing.assert_array_equal(planar.cell_volumes, planar.widths)
        np.testing.assert_array_equal(planar.cell_areas, np.ones_like(planar.widths))
        np.testing.assert_array_equal(planar.face_areas, np.ones_like(planar.edges))

        curved_values = copy.deepcopy(self.values)
        curved_values["geometry"]["outer_curvature_radius_m"] = 0.01
        curved = Grid.from_parameters(curved_values)
        radius = 0.01
        left, right = curved.edges[:-1], curved.edges[1:]
        exact = ((radius - left) ** 3 - (radius - right) ** 3) / (3.0 * radius**2)
        np.testing.assert_allclose(curved.cell_volumes, exact, rtol=5e-14, atol=1e-18)
        self.assertAlmostEqual(curved.face_areas[0], 1.0, places=15)
        self.assertLess(curved.face_areas[-1], 1.0)

        planar_resistance = simulate_scenario(
            self.values, 0.1, stage_names=("acetic_acid",)
        ).native_area_resistance_ohm_m2
        curved_resistance = simulate_scenario(
            curved_values, 0.1, stage_names=("acetic_acid",)
        ).native_area_resistance_ohm_m2
        self.assertGreater(curved_resistance, planar_resistance)

    def test_curved_shell_conserves_liquid_and_calcium(self) -> None:
        curved_values = copy.deepcopy(self.values)
        curved_values["geometry"]["outer_curvature_radius_m"] = 0.01
        scenario = simulate_scenario(curved_values, 0.1)
        for stage in STAGE_NAMES:
            residual = scenario.stage_mass_residuals(stage)
            self.assertLess(residual["liquid_residual_relative_to_inflow"], 1e-6)
            self.assertLess(residual["max_absolute_calcium_residual_mol_per_m2"], 1e-8)


class SiteFilmTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.full((3, 2), 0.8)
        self.porosity = np.ones((3, 2))
        self.exposure = np.zeros((3, 2))

    def test_native_and_zero_film_limits(self) -> None:
        native = site_film_forward(
            np.zeros(12), np.ones((3, 2)), self.porosity,
            self.porosity, self.exposure, self.exposure,
        )
        np.testing.assert_allclose(native, 1.0, rtol=0.0, atol=1e-12)

        parameters = np.full(12, -300.0)
        parameters[6:12] = 0.0
        no_film = site_film_forward(
            parameters, self.matrix, self.porosity, self.porosity,
            self.exposure, self.exposure,
        )
        expected = np.broadcast_to(self.matrix[None, :, None, :], no_film.shape)
        np.testing.assert_allclose(no_film, expected, atol=1e-12)

    def test_exposure_changes_film_without_changing_matrix_path(self) -> None:
        parameters = np.array([
            -3.0, -3.0, -3.0, -6.0, -6.0, -6.0,
            0.8, 1.0, 1.0, 1.0, 1.0, 0.0,
        ])
        exposed = self.exposure.copy()
        exposed[:, 1] = 0.5
        prediction = site_film_forward(
            parameters, self.matrix, self.porosity, self.porosity,
            exposed, exposed,
        )
        self.assertTrue(np.all(prediction[0, :, :, 1] < prediction[0, :, :, 0]))
        averaged = rat_condition_mean(prediction)
        np.testing.assert_allclose(averaged, prediction.mean(axis=1))

    def test_local_geometry_does_not_change_shared_chemistry(self) -> None:
        document = load_parameters(ROOT / "parameters" / "theoretical_prior.json")
        geometry = sample_site_geometries(np.array([0]), document, seed=17)
        self.assertEqual(len(geometry), 3)
        local = apply_site_geometry(document["values"], geometry.iloc[0])
        self.assertEqual(local["stages"], document["values"]["stages"])
        self.assertEqual(local["transport"], document["values"]["transport"])
        total = sum(layer["thickness_mm"] for layer in local["geometry"]["layers"])
        self.assertAlmostEqual(total, local["geometry"]["total_thickness_mm"], places=12)


if __name__ == "__main__":
    unittest.main()
