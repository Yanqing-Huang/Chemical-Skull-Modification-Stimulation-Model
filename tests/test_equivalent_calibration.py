"""Unit tests for the Track B equivalent-calibration module (synthetic data)."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cranial_model.equivalent_calibration import (
    apply_dual_branch_forward_operator,
    build_z_ratio_targets,
    compatible_candidates,
    interface_contribution_summary,
    leave_one_rat_out,
    residual_drift,
    sample_interface_lhs,
)


def _spectra_frame() -> pd.DataFrame:
    rows = []
    for rat, dose in (("WT1", 0.1), ("WT2", 0.2)):
        for site in ("Bregma", "Lambda"):
            for frequency in (100, 120, 1000):
                rows.append(
                    {"rat": rat, "site": site, "stage": "baseline",
                     "edta_concentration_mol_per_l": 0.1, "time_s": 0.0,
                     "frequency_hz": frequency, "magnitude_mohm": 100.0,
                     "confounder": "", "date": "260101", "source_cell": "A1"}
                )
                rows.append(
                    {"rat": rat, "site": site, "stage": "acetic_acid",
                     "edta_concentration_mol_per_l": 0.1, "time_s": 300.0,
                     "frequency_hz": frequency, "magnitude_mohm": 80.0,
                     "confounder": "", "date": "260101", "source_cell": "A2"}
                )
                for minutes in (15, 90):
                    rows.append(
                        {"rat": rat, "site": site, "stage": "edta",
                         "edta_concentration_mol_per_l": dose, "time_s": minutes * 60.0,
                         "frequency_hz": frequency, "magnitude_mohm": 50.0,
                         "confounder": "", "date": "260101", "source_cell": "A3"}
                    )
    return pd.DataFrame(rows)


def _predictions_frame(
    values: dict[int, tuple[float, float]] | None = None
) -> pd.DataFrame:
    rows = []
    values = values or {1: (0.8, 0.5), 2: (5.0, 5.0)}
    for candidate, (acetic_value, edta_value) in values.items():
        for stage, times, value in (
            ("acetic_acid", (300.0,), acetic_value),
            ("edta", (900.0, 5400.0), edta_value),
        ):
            for dose in (0.1, 0.2):
                for time_s in times:
                    rows.append(
                        {"candidate_id": candidate, "stage": stage, "time_s": time_s,
                         "edta_concentration_mol_per_l": dose,
                         "sim_whole_resistance_ratio": value}
                    )
    return pd.DataFrame(rows)


class TestZRatioTargets(unittest.TestCase):
    def test_low_frequency_ratio_and_dose_remap(self) -> None:
        targets = build_z_ratio_targets(_spectra_frame())
        self.assertEqual(set(targets["stage"]), {"acetic_acid", "edta"})
        # 1000 Hz must be excluded; edta ratio = 50/100, acetic ratio = 80/100.
        edta = targets[targets["stage"] == "edta"]
        acetic = targets[targets["stage"] == "acetic_acid"]
        self.assertTrue(np.allclose(edta["z_ratio"], 0.5))
        self.assertTrue(np.allclose(acetic["z_ratio"], 0.8))
        self.assertEqual(
            sorted(acetic["edta_concentration_mol_per_l"].unique()), [0.1, 0.2]
        )
        # 2 sites x 2 low frequencies averaged within rat.
        self.assertTrue((targets["n_values"] == 4).all())

    def test_confounded_rows_are_dropped(self) -> None:
        frame = _spectra_frame()
        frame.loc[
            (frame["stage"] == "edta") & (frame["time_s"] == 5400.0), "confounder"
        ] = "flagged"
        targets = build_z_ratio_targets(frame)
        edta_times = sorted(targets[targets["stage"] == "edta"]["time_s"].unique())
        self.assertEqual(edta_times, [900.0])

    def test_selected_confounded_rat_is_retained_for_sensitivity(self) -> None:
        frame = _spectra_frame()
        frame.loc[
            (frame["rat"] == "WT1") & (frame["stage"] == "edta"), "confounder"
        ] = "freeze_thaw"
        primary = build_z_ratio_targets(frame)
        sensitivity = build_z_ratio_targets(
            frame, retain_confounded_rats=("WT1",)
        )
        self.assertEqual(set(primary["rat"]), {"WT2"})
        self.assertEqual(set(sensitivity["rat"]), {"WT1", "WT2"})
        wt1_times = sorted(
            sensitivity[(sensitivity["rat"] == "WT1") & (sensitivity["stage"] == "edta")]["time_s"].unique()
        )
        self.assertEqual(wt1_times, [900.0, 5400.0])


class TestCompatibleSet(unittest.TestCase):
    def test_threshold_selects_exact_candidate(self) -> None:
        targets = build_z_ratio_targets(_spectra_frame())
        predictions = _predictions_frame()
        compatible, _, rmse = compatible_candidates(predictions, targets, 0.6)
        self.assertEqual(compatible.tolist(), [1])
        self.assertAlmostEqual(float(rmse[0]), 0.0, places=12)
        self.assertGreater(float(rmse[1]), 0.6)

    def test_leave_one_rat_out_runs_and_reports(self) -> None:
        rats = []
        for rat_index in range(4):
            frame = _spectra_frame()
            frame["rat"] = frame["rat"] + f"_{rat_index}"
            frame.loc[frame["stage"] != "baseline", "magnitude_mohm"] *= (
                1.0 + 0.02 * (rat_index - 1.5)
            )
            rats.append(frame)
        targets = build_z_ratio_targets(pd.concat(rats, ignore_index=True))
        predictions = _predictions_frame({1: (0.8, 0.5), 2: (5.0, 5.0), 3: (0.85, 0.55)})
        rat_summary, points = leave_one_rat_out(predictions, targets)
        self.assertEqual(len(rat_summary), 8)
        self.assertTrue((rat_summary["compatible_count"] > 0).all())
        self.assertTrue(points["covered"].notna().all())


class TestDualBranchForwardOperator(unittest.TestCase):
    def test_series_sum_and_interface_drop_share(self) -> None:
        predictions = pd.DataFrame(
            {
                "candidate_id": [0, 0],
                "sim_whole_resistance_ratio": [1.0, 0.5],
                "sim_outer_surface_porosity_ratio": [1.0, 2.0],
            }
        )
        parameters = pd.DataFrame(
            {"candidate_id": [0], "interface_fraction": [0.8],
             "interface_exponent": [2.0]}
        )
        composite = apply_dual_branch_forward_operator(predictions, parameters)
        np.testing.assert_allclose(composite["sim_whole_resistance_ratio"], [1.0, 0.3])
        np.testing.assert_allclose(composite["sim_interface_impedance_ratio"], [1.0, 0.25])
        summary = interface_contribution_summary(composite, np.array([0]))
        self.assertAlmostEqual(summary["baseline_fraction_q50"], 0.8)
        # Total drop = 0.7; interface drop = 0.8 * (1 - 0.25) = 0.6.
        self.assertAlmostEqual(summary["drop_share_q50"], 0.6 / 0.7)

    def test_interface_lhs_is_reproducible_and_stratified(self) -> None:
        first = sample_interface_lhs(20, 123)
        second = sample_interface_lhs(20, 123)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all((first[:, 0] >= 0.3) & (first[:, 0] <= 0.95)))
        self.assertTrue(np.all((first[:, 1] >= 0.5) & (first[:, 1] <= 3.0)))
        beta_strata = np.floor((first[:, 0] - 0.3) / 0.65 * 20).astype(int)
        self.assertEqual(len(np.unique(beta_strata)), 20)


class TestResidualDrift(unittest.TestCase):
    def test_monotone_drift_is_flagged(self) -> None:
        times = np.tile([900.0, 1800.0, 3600.0, 5400.0], 6)
        targets = pd.DataFrame(
            {
                "rat": np.repeat([f"WT{i}" for i in range(6)], 4),
                "stage": "edta",
                "edta_concentration_mol_per_l": 0.1,
                "time_s": times,
                "z_ratio": 0.5 + 0.0002 * times,
            }
        )
        intervals = targets[["rat", "stage", "edta_concentration_mol_per_l", "time_s"]].copy()
        intervals["sim_q50"] = 0.5
        drift = residual_drift(targets, intervals)
        self.assertGreater(drift["residual_vs_edta_time_spearman_rho"], 0.9)
        self.assertTrue(drift["drift_flag_|rho|>0.4_and_p<0.05"])


if __name__ == "__main__":
    unittest.main()
