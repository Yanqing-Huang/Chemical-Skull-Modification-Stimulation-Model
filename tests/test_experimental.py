from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cranial_model.experimental import (
    apply_magnitude_calibration,
    build_r0_ratio_observations,
    condition_statistics,
    split_rats_stratified,
)


class ExperimentalTests(unittest.TestCase):
    def test_standard_load_calibration_corrects_magnitude_without_extrapolation(self) -> None:
        spectra = pd.DataFrame({
            "frequency_hz": [100.0, 1000.0, 10000.0],
            "magnitude_mohm": [20.0, 40.0, 80.0],
        })
        calibration = pd.DataFrame({
            "frequency_hz": [100.0, 1000.0, 10000.0],
            "measured_over_true": [2.0, 4.0, 8.0],
        })
        corrected = apply_magnitude_calibration(spectra, calibration)
        np.testing.assert_allclose(corrected["magnitude_mohm"], [10.0, 10.0, 10.0])
        np.testing.assert_allclose(corrected["uncorrected_magnitude_mohm"], [20.0, 40.0, 80.0])
        with self.assertRaises(ValueError):
            apply_magnitude_calibration(
                pd.DataFrame({"frequency_hz": [50.0], "magnitude_mohm": [1.0]}),
                calibration,
            )

    def test_r0_ratio_requires_accepted_baseline_and_post(self) -> None:
        rows = []
        for stage, r0, quality in (
            ("baseline", 10.0, "accepted"),
            ("edta", 5.0, "accepted"),
            ("acetic_acid", 8.0, "poor_fit"),
        ):
            for frequency in (100, 120, 1000, 10000):
                rows.append({
                    "spectrum_id": f"R1_Bregma_{stage}",
                    "rat": "R1",
                    "site": "Bregma",
                    "date": "test",
                    "stage": stage,
                    "edta_concentration_mol_per_l": 0.1,
                    "time_s": 900.0 if stage == "edta" else 300.0,
                    "frequency_hz": frequency,
                    "magnitude_mohm": r0,
                    "source_cell": "A1",
                    "confounder": "",
                })
        def fake_fitter(frequency, magnitude):
            return {
                "r0_mohm": float(magnitude[0]),
                "quality_flag": "poor_fit" if magnitude[0] == 8.0 else "accepted",
                "relative_rmse": 0.0,
                "identifiable": magnitude[0] != 8.0,
            }
        fits, observations = build_r0_ratio_observations(pd.DataFrame(rows), fake_fitter)
        self.assertEqual(len(fits), 3)
        edta = observations[observations.stage == "edta"].iloc[0]
        acid = observations[observations.stage == "acetic_acid"].iloc[0]
        self.assertTrue(edta.valid)
        self.assertEqual(edta.r0_ratio, 0.5)
        self.assertFalse(acid.valid)
        self.assertIn("post_poor_fit", acid.rejection_reason)
        _, limited = build_r0_ratio_observations(
            pd.DataFrame(rows), fake_fitter, instrument_max_mohm=9.0
        )
        self.assertFalse(limited.valid.any())
        self.assertTrue(
            limited.rejection_reason.str.contains("baseline_outside_instrument_range").all()
        )

    def test_split_has_no_rat_leakage_and_statistics_use_requested_split(self) -> None:
        rows = []
        for dose in (0.05, 0.2):
            for index in range(4):
                rat = f"{dose}_{index}"
                for stage, time_s in (("acetic_acid", 300.0), ("edta", 900.0)):
                    rows.append({
                        "rat": rat,
                        "stage": stage,
                        "edta_concentration_mol_per_l": dose,
                        "time_s": time_s,
                        "r0_ratio": float(index + 1),
                    })
        frame = pd.DataFrame(rows)
        split = split_rats_stratified(frame, 42, test_fraction=0.25)
        self.assertEqual(list(split.values()).count("validation"), 2)
        frame["split"] = frame.rat.map(split)
        for rat, group in frame.groupby("rat"):
            self.assertEqual(group.split.nunique(), 1)
        calibration, _ = condition_statistics(frame, "calibration")
        validation, _ = condition_statistics(frame, "validation")
        self.assertEqual(len(calibration), 4)
        self.assertEqual(len(validation), 4)
        self.assertTrue(all(row["rat_count"] == 3 for row in calibration))
        self.assertTrue(all(row["rat_count"] == 1 for row in validation))


if __name__ == "__main__":
    unittest.main()
