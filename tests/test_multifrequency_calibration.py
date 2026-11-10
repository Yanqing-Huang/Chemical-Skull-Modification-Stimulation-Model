"""Synthetic tests for Track B v3 four-frequency magnitude calibration."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from cranial_model.equivalent_calibration import CONDITION_KEYS
from cranial_model.multifrequency_calibration import (
    FREQUENCIES_HZ,
    build_multifrequency_targets,
    condition_sd_by_frequency,
    frequency_interface_fraction,
    frequency_tolerances,
    leave_one_rat_out,
    matrix_only_parameters,
    multifrequency_forward,
    prediction_quantiles,
    sample_observation_lhs,
    score_multifrequency_candidates,
)


def _spectra_frame() -> pd.DataFrame:
    rows = []
    ratios = {100.0: 0.8, 120.0: 0.7, 1000.0: 0.6, 10000.0: 0.5}
    for rat, dose in (("WT1", 0.1), ("WT2", 0.2), ("WT3", 0.1)):
        for site in ("Bregma", "Lambda"):
            for frequency in FREQUENCIES_HZ:
                base = 100.0 + frequency / 1000.0
                rows.append({
                    "rat": rat, "site": site, "stage": "baseline",
                    "edta_concentration_mol_per_l": 0.1, "time_s": 0.0,
                    "frequency_hz": frequency, "magnitude_mohm": base,
                    "confounder": "", "date": "260101", "source_cell": "A1",
                })
                for stage, time_s, factor in (
                    ("acetic_acid", 300.0, 1.0), ("edta", 900.0, 0.75)
                ):
                    rows.append({
                        "rat": rat, "site": site, "stage": stage,
                        "edta_concentration_mol_per_l": dose, "time_s": time_s,
                        "frequency_hz": frequency,
                        "magnitude_mohm": base * ratios[frequency] * factor,
                        "confounder": "", "date": "260101", "source_cell": "A2",
                    })
    return pd.DataFrame(rows)


def _mechanism_predictions() -> pd.DataFrame:
    return pd.DataFrame([
        {"mechanism_id": 0, "stage": "acetic_acid", "time_s": 300.0,
         "edta_concentration_mol_per_l": 0.1,
         "sim_whole_resistance_ratio": 0.8, "sim_outer_surface_porosity_ratio": 1.4},
        {"mechanism_id": 0, "stage": "edta", "time_s": 900.0,
         "edta_concentration_mol_per_l": 0.1,
         "sim_whole_resistance_ratio": 0.65, "sim_outer_surface_porosity_ratio": 1.8},
        {"mechanism_id": 1, "stage": "acetic_acid", "time_s": 300.0,
         "edta_concentration_mol_per_l": 0.1,
         "sim_whole_resistance_ratio": 1.1, "sim_outer_surface_porosity_ratio": 0.9},
        {"mechanism_id": 1, "stage": "edta", "time_s": 900.0,
         "edta_concentration_mol_per_l": 0.1,
         "sim_whole_resistance_ratio": 1.2, "sim_outer_surface_porosity_ratio": 0.8},
    ])


def _observation_parameters() -> pd.DataFrame:
    return pd.DataFrame([
        {"candidate_id": 0, "mechanism_id": 0, "beta_low": 0.6,
         "beta_ratio": 0.5, "interface_exponent": 1.2},
        {"candidate_id": 1, "mechanism_id": 0, "beta_low": 0.9,
         "beta_ratio": 0.8, "interface_exponent": 2.5},
        {"candidate_id": 2, "mechanism_id": 1, "beta_low": 0.7,
         "beta_ratio": 0.4, "interface_exponent": 0.8},
    ])


def _synthetic_targets(rats: tuple[str, ...] = ("WT1", "WT2", "WT3", "WT4")) -> pd.DataFrame:
    predictions = _mechanism_predictions()
    parameters = _observation_parameters().iloc[[0]]
    matrix = predictions.loc[
        predictions["mechanism_id"] == 0, "sim_whole_resistance_ratio"
    ].to_numpy()
    surface = predictions.loc[
        predictions["mechanism_id"] == 0, "sim_outer_surface_porosity_ratio"
    ].to_numpy()
    values = multifrequency_forward(
        matrix, surface, parameters["beta_low"].to_numpy(),
        parameters["beta_ratio"].to_numpy(), parameters["interface_exponent"].to_numpy(),
    )[0]
    offsets = np.linspace(-0.012, 0.012, len(rats))
    if len(rats) % 2:
        offsets -= offsets.mean()
    rows = []
    conditions = predictions[predictions["mechanism_id"] == 0][CONDITION_KEYS].reset_index(drop=True)
    for rat_index, rat in enumerate(rats):
        for frequency_index, frequency in enumerate(FREQUENCIES_HZ):
            for condition_index, condition in conditions.iterrows():
                rows.append({
                    "rat": rat, "frequency_hz": frequency, **condition.to_dict(),
                    "z_ratio": values[frequency_index, condition_index] + offsets[rat_index],
                    "n_values": 3, "site_count": 3,
                })
    return pd.DataFrame(rows)


class TestObservationOperator(unittest.TestCase):
    def test_lhs_is_reproducible_and_stratified(self) -> None:
        first = sample_observation_lhs(24, 123)
        second = sample_observation_lhs(24, 123)
        np.testing.assert_array_equal(first, second)
        self.assertTrue(np.all((first[:, 0] >= 0.3) & (first[:, 0] <= 0.95)))
        self.assertTrue(np.all((first[:, 1] > 0) & (first[:, 1] < 1)))
        self.assertTrue(np.all((first[:, 2] >= 0.5) & (first[:, 2] <= 3.0)))
        for column in range(3):
            if column == 2:
                values = (np.log10(first[:, column]) - np.log10(0.5)) / (np.log10(3.0) - np.log10(0.5))
            elif column == 0:
                values = (first[:, column] - 0.3) / 0.65
            else:
                values = first[:, column]
            strata = np.floor(values * 24).astype(int)
            self.assertEqual(len(np.unique(strata)), 24)

    def test_baseline_is_one_endpoints_match_and_weight_decreases(self) -> None:
        beta = frequency_interface_fraction(np.array([0.8]), np.array([0.5]), FREQUENCIES_HZ)
        self.assertAlmostEqual(beta[0, 0], 0.8)
        self.assertAlmostEqual(beta[0, -1], 0.4)
        self.assertTrue(np.all(np.diff(beta[0]) < 0))
        predicted = multifrequency_forward(
            np.ones(2), np.ones(2), np.array([0.8]), np.array([0.5]),
            np.array([2.0]),
        )
        np.testing.assert_allclose(predicted, np.ones((1, 4, 2)))


class TestTargetsAndTolerances(unittest.TestCase):
    def test_sites_are_averaged_within_each_frequency(self) -> None:
        targets = build_multifrequency_targets(_spectra_frame())
        self.assertEqual(set(targets["frequency_hz"]), set(FREQUENCIES_HZ))
        self.assertTrue((targets["n_values"] == 2).all())
        self.assertTrue((targets["site_count"] == 2).all())
        acetic = targets[targets["stage"] == "acetic_acid"]
        observed = acetic.groupby("frequency_hz")["z_ratio"].mean()
        np.testing.assert_allclose(
            observed.loc[list(FREQUENCIES_HZ)], [0.8, 0.7, 0.6, 0.5]
        )
        self.assertEqual(len(acetic), 3 * 4)

    def test_acetic_noise_pools_doses_and_single_rat_conditions_do_not_contribute(self) -> None:
        rows = []
        values = {"WT1": 0.80, "WT2": 0.82, "WT3": 0.90, "WT4": 0.92}
        for frequency in (100.0, 120.0):
            for index, (rat, value) in enumerate(values.items()):
                dose = 0.1 if index < 2 else 0.2
                rows.append({"rat": rat, "frequency_hz": frequency, "stage": "acetic_acid",
                             "edta_concentration_mol_per_l": dose, "time_s": 300.0,
                             "z_ratio": value * (1.1 if frequency == 120 else 1.0)})
                rows.append({"rat": rat, "frequency_hz": frequency, "stage": "edta",
                             "edta_concentration_mol_per_l": dose, "time_s": 900.0,
                             "z_ratio": value - 0.2})
            rows.append({"rat": "WT1", "frequency_hz": frequency, "stage": "edta",
                         "edta_concentration_mol_per_l": 0.1, "time_s": 5400.0,
                         "z_ratio": 0.1})
        frame = pd.DataFrame(rows)
        condition_sd = condition_sd_by_frequency(frame)
        acetic = condition_sd[condition_sd["stage"] == "acetic_acid"]
        self.assertTrue((acetic["rat_count"] == 4).all())
        single = condition_sd[condition_sd["time_s"] == 5400.0]
        self.assertTrue(single["sd"].isna().all())
        tolerances = frequency_tolerances(frame)
        expected_sd = [
            np.std(list(values.values()), ddof=1),
            np.std([0.60, 0.62], ddof=1),
            np.std([0.70, 0.72], ddof=1),
        ]
        self.assertAlmostEqual(tolerances[100.0], np.sqrt(np.mean(np.square(expected_sd))))


class TestScoring(unittest.TestCase):
    def test_true_candidate_is_recovered_and_chunks_match(self) -> None:
        targets = _synthetic_targets()
        tolerances = frequency_tolerances(targets)
        whole = score_multifrequency_candidates(
            _mechanism_predictions(), _observation_parameters(), targets, tolerances,
            chunk_size=20,
        )
        chunked = score_multifrequency_candidates(
            _mechanism_predictions(), _observation_parameters(), targets, tolerances,
            chunk_size=1,
        )
        self.assertEqual(int(whole.metrics.loc[whole.metrics["normalized_rmse"].idxmin(), "candidate_id"]), 0)
        self.assertTrue(bool(whole.metrics.loc[0, "compatible"]))
        pd.testing.assert_frame_equal(whole.metrics, chunked.metrics)
        np.testing.assert_allclose(whole.squared_error, chunked.squared_error, atol=1e-15)

    def test_large_high_frequency_tolerance_cannot_hide_low_frequency_failure(self) -> None:
        rats = ("WT1", "WT2", "WT3", "WT4")
        targets = pd.DataFrame([
            {"rat": rat, "frequency_hz": frequency, "stage": "edta",
             "edta_concentration_mol_per_l": 0.1, "time_s": 900.0, "z_ratio": 1.0}
            for rat in rats for frequency in FREQUENCIES_HZ
        ])
        predictions = pd.DataFrame([
            {"mechanism_id": mechanism, "stage": "edta",
             "edta_concentration_mol_per_l": 0.1, "time_s": 900.0,
             "sim_whole_resistance_ratio": resistance,
             "sim_outer_surface_porosity_ratio": 1.0}
            for mechanism, resistance in ((0, 1.0), (1, 1.0 - 0.015 / 0.95))
        ])
        parameters = pd.DataFrame([
            {"candidate_id": 0, "mechanism_id": 0, "beta_low": 0.5,
             "beta_ratio": 0.5, "interface_exponent": 1.0},
            {"candidate_id": 1, "mechanism_id": 1, "beta_low": 0.05,
             "beta_ratio": 0.5, "interface_exponent": 1.0},
        ])
        tolerances = {100.0: 0.01, 120.0: 100.0, 1000.0: 100.0, 10000.0: 100.0}
        scores = score_multifrequency_candidates(predictions, parameters, targets, tolerances)
        bad = scores.metrics.iloc[1]
        self.assertLess(bad["normalized_rmse"], 1.0)
        self.assertGreater(bad["raw_rmse_100hz"], tolerances[100.0])
        self.assertFalse(bool(bad["compatible"]))

    def test_matrix_only_baseline_ignores_observation_parameters(self) -> None:
        targets = _synthetic_targets()
        parameters = matrix_only_parameters(np.array([0, 1]))
        first = score_multifrequency_candidates(
            _mechanism_predictions(), parameters, targets, frequency_tolerances(targets),
            matrix_only=True,
        )
        changed = parameters.copy()
        changed[["beta_low", "beta_ratio", "interface_exponent"]] = [0.9, 0.1, 3.0]
        second = score_multifrequency_candidates(
            _mechanism_predictions(), changed, targets, frequency_tolerances(targets),
            matrix_only=True,
        )
        np.testing.assert_allclose(first.squared_error, second.squared_error)


class TestLeaveOneRatOut(unittest.TestCase):
    def test_training_fold_is_isolated_from_held_out_rat(self) -> None:
        targets = _synthetic_targets()
        predictions = _mechanism_predictions()
        parameters = _observation_parameters()
        first = leave_one_rat_out(predictions, parameters, targets)
        changed = targets.copy()
        changed.loc[changed["rat"] == "WT1", "z_ratio"] *= 100.0
        second = leave_one_rat_out(predictions, parameters, changed)
        columns = ["compatible_count", "best_training_normalized_rmse", *[
            f"tolerance_{frequency:g}hz" for frequency in FREQUENCIES_HZ
        ]]
        first_fold = first[0].set_index("rat").loc["WT1", columns]
        second_fold = second[0].set_index("rat").loc["WT1", columns]
        pd.testing.assert_series_equal(first_fold, second_fold)
        train = targets[targets["rat"] != "WT1"]
        tolerances = frequency_tolerances(train)
        for frequency in FREQUENCIES_HZ:
            self.assertAlmostEqual(first_fold[f"tolerance_{frequency:g}hz"], tolerances[frequency])

    def test_empty_set_is_na_not_zero_coverage(self) -> None:
        targets = _synthetic_targets()
        bad = _observation_parameters().iloc[[2]].reset_index(drop=True)
        rat_summary, points, frequency_summary, overall = leave_one_rat_out(
            _mechanism_predictions(), bad, targets
        )
        self.assertTrue((rat_summary["compatible_count"] == 0).all())
        self.assertTrue(rat_summary["coverage"].isna().all())
        self.assertTrue(points["covered"].isna().all())
        self.assertFalse(points["prediction_available"].any())
        self.assertTrue(frequency_summary["point_coverage"].isna().all())
        self.assertIsNone(overall["point_coverage"])
        self.assertEqual(overall["available_point_count"], 0)

    def test_single_candidate_interval_is_marked_degenerate(self) -> None:
        targets = _synthetic_targets()
        true = _observation_parameters().iloc[[0]].reset_index(drop=True)
        rat_summary, points, _, overall = leave_one_rat_out(
            _mechanism_predictions(), true, targets
        )
        self.assertTrue((rat_summary["compatible_count"] == 1).all())
        self.assertTrue(rat_summary["degenerate_interval"].all())
        self.assertTrue(points["degenerate_interval"].all())
        np.testing.assert_allclose(points["width"], 0.0)
        self.assertEqual(overall["degenerate_fold_count"], len(targets["rat"].unique()))

    def test_prediction_quantiles_block_layout_and_empty_set(self) -> None:
        targets = _synthetic_targets()
        intervals = prediction_quantiles(
            _mechanism_predictions(), _observation_parameters(), np.array([0]), targets,
            row_block_size=3,
        )
        self.assertEqual(len(intervals), len(targets))
        self.assertTrue(intervals["degenerate_interval"].all())
        np.testing.assert_allclose(intervals["sim_q05"], intervals["sim_q95"])
        with self.assertRaisesRegex(ValueError, "Empty compatible set"):
            prediction_quantiles(
                _mechanism_predictions(), _observation_parameters(), np.array([]), targets
            )


if __name__ == "__main__":
    unittest.main()
