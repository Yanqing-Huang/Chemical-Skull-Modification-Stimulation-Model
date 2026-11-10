"""Plugin and physical-limit tests for V5b diploe polarization."""

from __future__ import annotations

import unittest

import numpy as np

from cranial_model.diploe_polarization import (
    diploe_polarization_forward, polarization_state,
)
from cranial_model.site_film_calibration import site_film_forward


class DiploePolarizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.full((3, 2), 0.8)
        self.porosity = np.ones((3, 2))
        self.exposure = np.zeros((3, 2))
        self.stage = np.array(["acetic_acid", "edta"])
        self.time = np.array([300.0, 900.0])
        self.base = np.array([
            -3.0, -3.0, -3.0, -6.0, -6.0, -6.0,
            0.8, 1.0, 1.0, 1.0, 1.0, 0.0,
        ])
        self.polarization = np.array([-2.0, -3.0, 1.0, 1.0])

    def forward(self, polarization):
        return diploe_polarization_forward(
            self.base, polarization, self.matrix, self.porosity, self.porosity,
            self.exposure, self.exposure, self.stage, self.time,
        )

    def test_disabled_plugin_is_exactly_v5a(self) -> None:
        off = self.polarization.copy()
        off[2] = -np.inf
        expected = site_film_forward(
            self.base, self.matrix, self.porosity, self.porosity,
            self.exposure, self.exposure,
        )
        np.testing.assert_array_equal(self.forward(off), expected)

    def test_zero_state_is_the_v5a_limit(self) -> None:
        time = np.zeros(2)
        stage = np.array(["acetic_acid", "acetic_acid"])
        actual = diploe_polarization_forward(
            self.base, self.polarization, self.matrix, self.porosity, self.porosity,
            self.exposure, self.exposure, stage, time,
        )
        expected = site_film_forward(
            self.base, self.matrix, self.porosity, self.porosity,
            self.exposure, self.exposure,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)

    def test_polarization_is_stronger_at_low_frequency(self) -> None:
        baseline = self.forward(np.array([-2.0, -3.0, -np.inf, 1.0]))
        active = self.forward(self.polarization)
        low = np.max(np.abs(active[:, :, 0, :] - baseline[:, :, 0, :]))
        high = np.max(np.abs(active[:, :, 3, :] - baseline[:, :, 3, :]))
        self.assertGreater(low, 1e-4)
        self.assertLess(high, low * 0.02)

    def test_acid_growth_and_edta_desorption_retain_history(self) -> None:
        stage = np.array(["acetic_acid", "acetic_acid", "edta", "edta"])
        time = np.array([0.0, 300.0, 0.0, 5400.0])
        m = polarization_state(self.polarization, stage, time)[0]
        self.assertEqual(m[0], 0.0)
        self.assertGreater(m[1], 0.0)
        self.assertAlmostEqual(m[2], m[1], places=14)
        self.assertLess(m[3], m[2])

    def test_vectorized_candidates_align_with_single_evaluations(self) -> None:
        base = np.stack([self.base, self.base])
        polarization = np.stack([self.polarization, self.polarization])
        actual = diploe_polarization_forward(
            base, polarization, self.matrix, self.porosity, self.porosity,
            self.exposure, self.exposure, self.stage, self.time,
        )
        expected = self.forward(self.polarization)
        np.testing.assert_allclose(actual, np.concatenate([expected, expected], axis=0))


if __name__ == "__main__":
    unittest.main()
