"""Tests for same-baseline Delta-Z target construction and unit conversion."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cranial_model.delta_z_calibration import (
    build_delta_z_targets, delta_frequency_tolerances, ratio_prediction_to_delta,
)


class DeltaZCalibrationTests(unittest.TestCase):
    def synthetic_spectra(self):
        rows = []
        for rat, baseline in (("WT1", 2.0), ("WT2", 4.0)):
            for stage, multiplier, time in (
                ("baseline", 1.0, 0.0),
                ("acetic_acid", 0.5, 300.0),
                ("edta", 0.25, 900.0),
                ("edta", 0.25, 5400.0),
            ):
                rows.append({
                    "rat": rat, "site": "Bregma", "frequency_hz": 100.0,
                    "stage": stage, "time_s": time,
                    "edta_concentration_mol_per_l": 0.1,
                    "magnitude_mohm": baseline * multiplier, "confounder": "",
                })
        return pd.DataFrame(rows)

    def test_delta_is_computed_before_rat_level_site_average(self) -> None:
        actual = build_delta_z_targets(self.synthetic_spectra(), frequencies=(100.0,))
        acid = actual[actual["stage"] == "acetic_acid"].set_index("rat")
        self.assertEqual(acid.loc["WT1", "delta_z_mohm"], -1.0)
        self.assertEqual(acid.loc["WT2", "delta_z_mohm"], -2.0)
        self.assertEqual(acid.loc["WT1", "z_ratio"], 0.5)
        self.assertEqual(acid.loc["WT2", "z_ratio"], 0.5)
        self.assertEqual(acid.loc["WT2", "baseline_mean_mohm"], 4.0)

    def test_delta_tolerance_has_mohm_scale_and_differs_from_ratio(self) -> None:
        targets = build_delta_z_targets(self.synthetic_spectra(), frequencies=(100.0,))
        tolerance = delta_frequency_tolerances(targets)[100.0]
        expected = np.sqrt((0.5 + 1.125 + 1.125) / 3.0)
        self.assertAlmostEqual(tolerance, expected)

    def test_ratio_prediction_conversion_uses_each_target_baseline(self) -> None:
        ratio = np.array([[[0.5, 0.25]]])
        actual = ratio_prediction_to_delta(
            ratio, np.array([0, 0]), np.array([0, 1]), np.array([2.0, 4.0])
        )
        np.testing.assert_allclose(actual, [[-1.0, -3.0]])


if __name__ == "__main__":
    unittest.main()
