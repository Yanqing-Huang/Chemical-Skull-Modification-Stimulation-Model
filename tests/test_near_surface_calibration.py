"""Synthetic and state-extraction tests for Track B v4 near-surface calibration."""

from __future__ import annotations

import copy
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from cranial_model import load_parameters, simulate_scenario
from cranial_model.equivalent_calibration import CONDITION_KEYS
from cranial_model.multifrequency_calibration import FREQUENCIES_HZ, frequency_tolerances
from cranial_model.near_surface import (
    STRUCTURES, combine_structure_state, cumulative_relative_exposure, state_columns,
    surface_indices, surface_porosity_ratio,
)
from cranial_model.near_surface_calibration import (
    WEIGHT_COLUMNS, leave_one_rat_out, near_surface_forward, prediction_quantiles,
    sample_near_surface_lhs, score_near_surface_candidates,
)
from run_near_surface_calibration import reconstruct_v2_mechanisms


ROOT = Path(__file__).resolve().parents[1]


def _predictions() -> pd.DataFrame:
    rows = []
    for mechanism in (0, 1):
        for stage, time_s, matrix, porosity, exposure in (
            ("acetic_acid", 300.0, 0.9, 1.2, 0.04),
            ("edta", 900.0, 0.8, 1.5, 0.15),
        ):
            rows.append({
                "mechanism_id": mechanism, "stage": stage, "time_s": time_s,
                "edta_concentration_mol_per_l": 0.1,
                "sim_whole_resistance_ratio": matrix + 0.2 * mechanism,
                "sim_outer_porosity_ratio_100um": porosity,
                "sim_inner_porosity_ratio_100um": porosity,
                "sim_outer_exposure_100um": exposure,
                "sim_inner_exposure_100um": exposure,
            })
    return pd.DataFrame(rows)


def _parameters() -> pd.DataFrame:
    return pd.DataFrame([{
        "candidate_id": 0, "mechanism_id": 0,
        **{column: value for column, value in zip(WEIGHT_COLUMNS, (0.8, 0.7, 0.6, 0.5))},
        "interface_exponent": 1.5,
    }])


def _targets(rats=("WT1", "WT2", "WT3", "WT4")) -> pd.DataFrame:
    predictions = _predictions()
    predictions = predictions[predictions.mechanism_id == 0]
    values = near_surface_forward(
        predictions.sim_whole_resistance_ratio.to_numpy(),
        predictions.sim_outer_porosity_ratio_100um.to_numpy(),
        predictions.sim_outer_exposure_100um.to_numpy(),
        _parameters().loc[:, list(WEIGHT_COLUMNS)].to_numpy(),
        _parameters().interface_exponent.to_numpy(),
    )[0]
    conditions = predictions[CONDITION_KEYS].reset_index(drop=True)
    rows = []
    for rat_index, rat in enumerate(rats):
        offset = (rat_index - (len(rats) - 1) / 2) * 0.005
        for frequency_index, frequency in enumerate(FREQUENCIES_HZ):
            for condition_index, condition in conditions.iterrows():
                rows.append({
                    "rat": rat, "frequency_hz": frequency, **condition.to_dict(),
                    "z_ratio": values[frequency_index, condition_index] + offset,
                    "n_values": 3, "site_count": 3,
                })
    return pd.DataFrame(rows)


class NearSurfaceStateTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.values = copy.deepcopy(load_parameters(ROOT / "parameters" / "theoretical_prior.json")["values"])
        cls.values["numerics"]["cells_per_layer"] = [8, 5, 5]
        cls.values["numerics"]["max_step_s"] = 5.0
        cls.values["numerics"]["output_times_per_stage"] = 7
        for stage in cls.values["stages"].values():
            stage["duration_s"] = 30.0
        for segment in cls.values["stages"]["edta"]["segments"]:
            segment["duration_s"] = 7.5
        cls.scenario = simulate_scenario(cls.values, 0.1)

    def test_surface_indices_select_opposite_boundaries(self) -> None:
        grid = self.scenario.grid
        outer = surface_indices(grid, 0, 50)
        inner = surface_indices(grid, 2, 50)
        self.assertTrue(np.all(grid.centers[outer] < grid.centers[outer[-1]] + 50e-6))
        self.assertTrue(np.all(grid.centers[inner] > grid.centers[inner[0]] - 50e-6))
        self.assertLess(float(grid.centers[outer].mean()), float(grid.centers[inner].mean()))

    def test_two_sided_states_are_equal_outer_inner_mean(self) -> None:
        frame = pd.DataFrame([{
            "sim_outer_porosity_ratio_100um": 1.2, "sim_inner_porosity_ratio_100um": 1.6,
            "sim_outer_exposure_100um": 0.1, "sim_inner_exposure_100um": 0.3,
        }])
        self.assertEqual(state_columns("outer_only_100um", "exposure"), ("sim_outer_exposure_100um",))
        self.assertAlmostEqual(combine_structure_state(frame, "two_sided_100um", "porosity_ratio")[0], 1.4)
        self.assertAlmostEqual(combine_structure_state(frame, "two_sided_100um", "exposure")[0], 0.2)

    def test_exposure_clears_no_history_and_inner_edta_is_positive(self) -> None:
        acid_outer, acid_inner = cumulative_relative_exposure(self.scenario, "acetic_acid", 30.0, 100)
        edta_outer, edta_inner = cumulative_relative_exposure(self.scenario, "edta", 7.5, 100)
        self.assertGreater(acid_outer, 0.0)
        self.assertAlmostEqual(acid_inner, 0.0)
        self.assertGreater(edta_outer, acid_outer)
        self.assertGreater(edta_inner, 0.0)
        outer_ratio, inner_ratio = surface_porosity_ratio(self.scenario, "edta", 7.5, 100)
        self.assertGreater(outer_ratio, 0.0)
        self.assertGreater(inner_ratio, 0.0)


class NearSurfaceObservationTests(unittest.TestCase):
    def test_v2_mechanism_inputs_regenerate_beyond_csv_display_precision(self) -> None:
        mechanism = reconstruct_v2_mechanisms(
            200, ROOT / "results" / "equivalent_calibration_v2", 20261113
        )
        saved = pd.read_csv(ROOT / "results" / "equivalent_calibration_v2" / "equivalent_lhs_samples.csv")
        saved = saved.drop_duplicates("mechanism_id").sort_values("mechanism_id")
        columns = [column for column in mechanism.columns if column != "mechanism_id"]
        generated = mechanism[columns].to_numpy(dtype=float)
        round_tripped = saved[columns].to_numpy(dtype=float)
        self.assertFalse(np.array_equal(generated, round_tripped))
        np.testing.assert_allclose(generated, round_tripped, rtol=1e-12, atol=0.0)
        prefix = reconstruct_v2_mechanisms(
            17, ROOT / "results" / "equivalent_calibration_v2", 20261113
        )
        pd.testing.assert_frame_equal(prefix, mechanism.iloc[:17].reset_index(drop=True))

    def test_lhs_dynamic_pairs_static_dimensions(self) -> None:
        static = sample_near_surface_lhs(16, 123, dynamic=False)
        dynamic = sample_near_surface_lhs(16, 123, dynamic=True)
        pd.testing.assert_frame_equal(static, dynamic[static.columns])
        self.assertTrue(((dynamic.exposure_logit_slope >= -6) & (dynamic.exposure_logit_slope <= 6)).all())
        self.assertEqual(len(STRUCTURES), 6)

    def test_dynamic_weight_is_state_dependent_and_shared_across_frequency(self) -> None:
        weights = np.array([[0.4, 0.5, 0.6, 0.7]])
        static = near_surface_forward(np.ones(2), np.ones(2), np.array([0.0, 0.2]), weights, np.array([2.0]))
        dynamic = near_surface_forward(
            np.array([0.8, 0.8]), np.array([2.0, 2.0]), np.array([0.0, 0.2]),
            weights, np.array([1.0]), np.array([2.0]),
        )
        np.testing.assert_allclose(static, np.ones((1, 4, 2)))
        np.testing.assert_allclose(dynamic[:, :, 0], (1 - weights) * 0.8 + weights * 0.5)
        self.assertTrue(np.all(dynamic[0, :, 1] < dynamic[0, :, 0]))

    def test_true_candidate_scores_and_loo_isolated(self) -> None:
        targets = _targets()
        tolerances = frequency_tolerances(targets)
        scores = score_near_surface_candidates(
            _predictions(), _parameters(), targets, tolerances, "outer_only_100um"
        )
        self.assertTrue(bool(scores.metrics.loc[0, "compatible"]))
        rat, points, _, overall = leave_one_rat_out(
            _predictions(), _parameters(), targets, scores, frequency_tolerances,
            "outer_only_100um",
        )
        self.assertTrue((rat.compatible_count == 1).all())
        self.assertTrue(rat.degenerate_interval.all())
        self.assertEqual(overall["available_point_count"], len(targets))
        intervals = prediction_quantiles(
            _predictions(), _parameters(), np.array([0]), targets, "outer_only_100um"
        )
        self.assertTrue(np.allclose(intervals.sim_q05, intervals.sim_q95))

    def test_empty_fold_is_na_not_zero(self) -> None:
        targets = _targets()
        bad = _parameters()
        bad.loc[0, "interface_exponent"] = 2.9
        scores = score_near_surface_candidates(
            _predictions(), bad, targets, frequency_tolerances(targets), "outer_only_100um"
        )
        rat, points, _, overall = leave_one_rat_out(
            _predictions(), bad, targets, scores, frequency_tolerances, "outer_only_100um"
        )
        self.assertTrue((rat.compatible_count == 0).all())
        self.assertTrue(points.covered.isna().all())
        self.assertIsNone(overall["point_coverage"])


if __name__ == "__main__":
    unittest.main()
