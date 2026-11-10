from __future__ import annotations

import unittest

import numpy as np

from cranial_model.impedance import complex_rcpe_impedance, fit_rcpe_magnitude


class ImpedanceTests(unittest.TestCase):
    def test_complex_formula_limits_and_debye_form(self) -> None:
        r0, rinf, tau = 12.0, 2.0, 0.01
        self.assertAlmostEqual(complex_rcpe_impedance(0.0, r0, rinf, tau, 0.8).real, r0)
        high = complex_rcpe_impedance(1e14, r0, rinf, tau, 0.8)
        self.assertAlmostEqual(high.real, rinf, places=6)
        frequency = 100.0
        calculated = complex_rcpe_impedance(frequency, r0, rinf, tau, 1.0)
        omega_tau = 2.0 * np.pi * frequency * tau
        expected_real = rinf + (r0 - rinf) / (1.0 + omega_tau**2)
        self.assertAlmostEqual(calculated.real, expected_real, places=10)

    def test_synthetic_spectrum_recovers_parameters(self) -> None:
        frequency = np.array([50.0, 100.0, 120.0, 500.0, 1000.0, 10000.0, 50000.0])
        expected = (8.0, 1.2, 0.0008, 0.72)
        magnitude = np.abs(complex_rcpe_impedance(frequency, *expected))
        first = fit_rcpe_magnitude(frequency, magnitude)
        second = fit_rcpe_magnitude(frequency, magnitude)
        self.assertEqual(first["quality_flag"], "accepted", first["message"])
        self.assertAlmostEqual(first["r0_mohm"], expected[0], delta=0.05)
        self.assertAlmostEqual(first["rinf_mohm"], expected[1], delta=0.05)
        self.assertAlmostEqual(first["tau_s"], expected[2], delta=0.00005)
        self.assertAlmostEqual(first["beta"], expected[3], delta=0.02)
        self.assertEqual(first["r0_mohm"], second["r0_mohm"])

    def test_flat_spectrum_is_not_identifiable(self) -> None:
        result = fit_rcpe_magnitude(
            np.array([100.0, 120.0, 1000.0, 10000.0]),
            np.full(4, 3.0),
        )
        self.assertNotEqual(result["quality_flag"], "accepted")
        self.assertFalse(result["identifiable"])

    def test_steep_measured_shape_is_rejected_not_forced_to_fit(self) -> None:
        result = fit_rcpe_magnitude(
            np.array([100.0, 120.0, 1000.0, 10000.0]),
            np.array([51.0, 47.0, 16.0, 0.01]),
        )
        self.assertEqual(result["quality_flag"], "poor_fit")
        self.assertGreater(result["relative_rmse"], 0.05)

    def test_invalid_input_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            fit_rcpe_magnitude(np.array([100.0, 120.0]), np.array([3.0, 2.0]))
        with self.assertRaises(ValueError):
            complex_rcpe_impedance(100.0, 2.0, 1.0, 0.1, 1.2)


if __name__ == "__main__":
    unittest.main()
