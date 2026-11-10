"""Tests for calibrated ensemble interval arithmetic and compatibility labels."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cranial_model.interval_prediction import (
    expand_intervals, frequency_expansions, nonconformity_scores,
)
from cranial_model.multifrequency_calibration import FREQUENCIES_HZ


class IntervalPredictionTests(unittest.TestCase):
    def test_nonconformity_is_zero_inside_and_signed_distance_outside(self) -> None:
        frame = pd.DataFrame({
            "frequency_hz": [100.0, 100.0, 100.0],
            "z_ratio": [1.05, 0.80, 1.30],
            "sim_q05": [1.0, 1.0, 1.0],
            "sim_q95": [1.1, 1.1, 1.1],
        })
        actual = nonconformity_scores(frame, {100.0: 0.1})
        np.testing.assert_allclose(actual, [0.0, 2.0, 2.0])

    def test_frequency_expansion_uses_higher_finite_sample_quantile(self) -> None:
        scores = np.tile(np.arange(10, dtype=float), 4)
        frequencies = np.repeat(FREQUENCIES_HZ, 10)
        actual = frequency_expansions(scores, frequencies, level=0.90)
        for frequency in FREQUENCIES_HZ:
            self.assertEqual(actual[frequency], 9.0)

    def test_expansion_preserves_na_for_original_empty_compatible_set(self) -> None:
        frame = pd.DataFrame({
            "frequency_hz": [100.0, 100.0],
            "z_ratio": [1.0, 1.3],
            "sim_q05": [0.9, 0.9],
            "sim_q50": [1.0, 1.0],
            "sim_q95": [1.1, 1.1],
            "original_compatible_prediction_available": [False, True],
        })
        actual = expand_intervals(frame, {100.0: 0.1}, {100.0: 1.0})
        np.testing.assert_allclose(actual["calibrated_lower"], [0.8, 0.8])
        np.testing.assert_allclose(actual["calibrated_upper"], [1.2, 1.2])
        self.assertTrue(pd.isna(actual.loc[0, "original_compatible_covered"]))
        self.assertFalse(bool(actual.loc[1, "original_compatible_covered"]))
        self.assertEqual(actual["calibrated_covered"].tolist(), [True, False])

    def test_impedance_ratio_lower_bound_is_physically_nonnegative(self) -> None:
        frame = pd.DataFrame({
            "frequency_hz": [100.0], "z_ratio": [0.2],
            "sim_q05": [0.1], "sim_q50": [0.3], "sim_q95": [0.5],
            "original_compatible_prediction_available": [False],
        })
        actual = expand_intervals(frame, {100.0: 0.2}, {100.0: 2.0})
        self.assertAlmostEqual(actual.loc[0, "unclipped_calibrated_lower"], -0.3)
        self.assertEqual(actual.loc[0, "calibrated_lower"], 0.0)
        self.assertTrue(bool(actual.loc[0, "calibrated_covered"]))

    def test_delta_z_uses_mohm_observations_and_allows_negative_lower(self) -> None:
        frame = pd.DataFrame({
            "frequency_hz": [100.0], "z_ratio": [999.0],
            "delta_z_mohm": [-2.0], "sim_q05": [-1.0], "sim_q50": [-0.5],
            "sim_q95": [0.0], "original_compatible_prediction_available": [True],
        })
        scores = nonconformity_scores(
            frame, {100.0: 2.0}, observed_column="delta_z_mohm"
        )
        np.testing.assert_allclose(scores, [0.5])
        actual = expand_intervals(
            frame, {100.0: 2.0}, {100.0: 2.0},
            observed_column="delta_z_mohm", physical_lower_bound=None,
        )
        self.assertEqual(actual.loc[0, "unclipped_calibrated_lower"], -5.0)
        self.assertEqual(actual.loc[0, "calibrated_lower"], -5.0)
        self.assertEqual(actual.loc[0, "calibrated_upper"], 4.0)
        self.assertTrue(bool(actual.loc[0, "calibrated_covered"]))
        self.assertFalse(bool(actual.loc[0, "original_compatible_covered"]))


if __name__ == "__main__":
    unittest.main()
