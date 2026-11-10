from __future__ import annotations

import copy
import unittest

import numpy as np

from cranial_model.ensemble import (
    PARAMETER_PATHS,
    apply_candidate,
    monotonic_trend_fraction,
    prediction_interval,
    sample_lhs_log10,
)


class EnsembleTests(unittest.TestCase):
    def test_log_lhs_covers_each_stratum_and_changes_only_kinetics(self) -> None:
        document = {
            "values": {
                "geometry": {"total_thickness_mm": 0.8},
                "transport": {
                    "external_mass_transfer_m_per_s": 1e-5,
                    "porosity_per_released_calcium_m3_per_mol": 3e-5,
                },
                "stages": {
                    "acetic_acid": {"k_r_m3_per_mol_s": 1e-7},
                    "edta": {"k_r_m3_per_mol_s": 1e-6},
                },
            },
            "uncertainty": {
                PARAMETER_PATHS[0]: {"range": [1e-6, 1e-4]},
                PARAMETER_PATHS[1]: {"range": [1e-9, 1e-5]},
                PARAMETER_PATHS[2]: {"range": [1e-8, 5e-5]},
                PARAMETER_PATHS[3]: {"range": [1e-5, 6e-5]},
            },
        }
        count = 20
        samples = sample_lhs_log10(document, count, 123)
        for column, path in enumerate(PARAMETER_PATHS):
            low, high = document["uncertainty"][path]["range"]
            scaled = (np.log10(samples[:, column]) - np.log10(low)) / (np.log10(high) - np.log10(low))
            self.assertEqual(len(set(np.floor(scaled * count).astype(int))), count)
        updated = apply_candidate(document["values"], PARAMETER_PATHS, samples[0])
        self.assertEqual(updated["geometry"], document["values"]["geometry"])
        self.assertNotEqual(updated, document["values"])
        self.assertEqual(document["values"]["transport"]["external_mass_transfer_m_per_s"], 1e-5)

    def test_prediction_envelope_and_monotonic_fractions(self) -> None:
        conditions = [
            {"stage": "edta", "edta_concentration_mol_per_l": dose, "time_s": time}
            for dose in (0.05, 0.1)
            for time in (900.0, 1800.0)
        ]
        predictions = np.array([
            [0.95, 0.90, 0.85, 0.80],
            [0.90, 0.91, 0.80, 0.79],
        ])
        lower, upper = prediction_interval(predictions)
        np.testing.assert_allclose(lower, [0.90, 0.90, 0.80, 0.79])
        np.testing.assert_allclose(upper, [0.95, 0.91, 0.85, 0.80])
        trends = monotonic_trend_fraction(conditions, predictions)
        self.assertEqual(trends["time_monotonic_fraction"], 0.5)
        self.assertEqual(trends["dose_monotonic_fraction"], 1.0)
        self.assertEqual(trends["both_monotonic_fraction"], 0.5)


if __name__ == "__main__":
    unittest.main()
