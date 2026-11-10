"""Regression tests for the reproduced best observation model and V5b plugin."""

from __future__ import annotations

import unittest

import numpy as np
from scipy.special import expit

from cranial_model.frequency_surface_polarization import (
    frequency_surface_forward, frequency_surface_polarization_forward,
    frequency_surface_time_polarization_forward,
)


class FrequencySurfacePolarizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.matrix = np.array([0.82, 0.77, 0.74])
        self.interface = np.array([1.8, 2.1, 1.6])
        self.exposure = np.array([0.01, 0.04, 0.20])
        self.stage = np.array(["acetic_acid", "acetic_acid", "edta"])
        self.dose = np.array([0.05, 0.05, 0.10])
        self.time = np.array([0.0, 300.0, 900.0])
        self.base = np.array([
            0.20, 0.40, 0.60, 0.80,
            1.20, 1.50, 1.80, 2.10,
            0.50, 0.10, -0.20, 0.30, -0.10,
        ])
        self.polarization = np.array([-2.0, -3.0, -1.0, 0.90])

    def baseline(self, parameters=None):
        return frequency_surface_forward(
            self.base if parameters is None else parameters,
            self.matrix, self.interface, self.exposure,
            self.stage, self.dose, self.time,
        )

    def combined(self, polarization=None, parameters=None):
        base = self.base if parameters is None else parameters
        pol = self.polarization if polarization is None else polarization
        return frequency_surface_polarization_forward(
            np.concatenate([base, pol]), self.matrix, self.interface,
            self.exposure, self.stage, self.dose, self.time,
        )

    def test_disabled_plugin_is_exactly_the_observation_baseline(self) -> None:
        off = self.polarization.copy()
        off[2] = -np.inf
        np.testing.assert_array_equal(self.combined(off), self.baseline())

    def test_zero_state_is_the_observation_baseline(self) -> None:
        stage = np.array(["acetic_acid", "acetic_acid"])
        time = np.zeros(2)
        actual = frequency_surface_polarization_forward(
            np.concatenate([self.base, self.polarization]),
            self.matrix[:2], self.interface[:2], self.exposure[:2],
            stage, self.dose[:2], time,
        )
        expected = frequency_surface_forward(
            self.base, self.matrix[:2], self.interface[:2], self.exposure[:2],
            stage, self.dose[:2], time,
        )
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=1e-14)

    def test_common_exponent_reproduces_full_shared_formula(self) -> None:
        # Saved v4c order: four weights, shared m, exposure slope, EDTA d/d^2,
        # acid offset, exposure squared.
        old = np.array([
            0.20, 0.40, 0.60, 0.80, 1.70,
            0.50, 0.10, -0.20, 0.30, -0.10,
        ])
        base = self.base.copy()
        base[4:8] = old[4]
        base[8:] = old[5:]
        weight = old[:4]
        logit = np.log(weight / (1.0 - weight))[:, None]
        acid = (self.stage == "acetic_acid").astype(float)
        edta = 1.0 - acid
        dose = np.log(self.dose / 0.1)
        effective = expit(
            logit + old[5] * self.exposure[None, :]
            + edta[None, :] * (old[6] * dose[None, :] + old[7] * dose[None, :] ** 2)
            + old[8] * acid[None, :] + old[9] * self.exposure[None, :] ** 2
        )
        surface = self.interface[None, None, :] ** (-old[4])
        expected = (
            (1.0 - effective) * self.matrix[None, None, :]
            + effective * surface
        )
        np.testing.assert_allclose(self.baseline(base), expected, rtol=2e-15, atol=0.0)

    def test_independent_frequency_exponents_are_applied_by_frequency(self) -> None:
        neutral = self.base.copy()
        neutral[8:] = 0.0
        actual = self.baseline(neutral)
        weights = neutral[:4]
        expected = (
            (1.0 - weights[:, None]) * self.matrix[None, :]
            + weights[:, None] * self.interface[None, :] ** (-neutral[4:8, None])
        )
        np.testing.assert_allclose(actual, expected[None, :, :], rtol=2e-15, atol=0.0)

    def test_polarization_is_stronger_at_low_frequency(self) -> None:
        off = self.polarization.copy()
        off[2] = -np.inf
        baseline = self.combined(off)
        active = self.combined()
        low = np.max(np.abs(active[:, 0, :] - baseline[:, 0, :]))
        high = np.max(np.abs(active[:, 3, :] - baseline[:, 3, :]))
        self.assertGreater(low, 1e-5)
        self.assertLess(high, low * 0.02)

    def test_zero_time_response_exactly_delegates_to_combined_model(self) -> None:
        parameters = np.concatenate([self.base, self.polarization, [0.0, 0.0]])
        actual = frequency_surface_time_polarization_forward(
            parameters, self.matrix, self.interface, self.exposure,
            self.stage, self.dose, self.time,
        )
        np.testing.assert_array_equal(actual, self.combined())

    def test_time_response_vectorization_and_finiteness(self) -> None:
        first = np.concatenate([self.base, self.polarization, [0.3, -0.2]])
        second = first.copy()
        second[17:19] = [-0.1, 0.4]
        actual = frequency_surface_time_polarization_forward(
            np.stack([first, second]).T, self.matrix, self.interface,
            self.exposure, self.stage, self.dose, self.time,
        )
        expected = np.concatenate([
            frequency_surface_time_polarization_forward(
                candidate, self.matrix, self.interface, self.exposure,
                self.stage, self.dose, self.time,
            )
            for candidate in (first, second)
        ], axis=0)
        np.testing.assert_array_equal(actual, expected)
        self.assertTrue(np.isfinite(actual).all())
        self.assertGreater(np.max(np.abs(actual - self.combined())), 1e-6)

    def test_vectorized_candidates_align_with_single_evaluations(self) -> None:
        candidates = np.stack([
            np.concatenate([self.base, self.polarization]),
            np.concatenate([self.base * 0.9 + 0.01, self.polarization + 0.1]),
        ])
        actual = frequency_surface_polarization_forward(
            candidates.T, self.matrix, self.interface, self.exposure,
            self.stage, self.dose, self.time,
        )
        expected = np.concatenate([
            frequency_surface_polarization_forward(
                candidate, self.matrix, self.interface, self.exposure,
                self.stage, self.dose, self.time,
            )
            for candidate in candidates
        ], axis=0)
        np.testing.assert_allclose(actual, expected, rtol=0.0, atol=0.0)
        self.assertTrue(np.isfinite(actual).all())


if __name__ == "__main__":
    unittest.main()
