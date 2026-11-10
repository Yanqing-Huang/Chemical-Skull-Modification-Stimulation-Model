"""Track B (auxiliary): equivalent calibration of composite parameters on Z ratios.

Uses the identical three-layer equation frame as Track A, but selects a
compatible parameter subset against the measured low-frequency impedance
magnitude ratio Z_post/Z_baseline (100/120 Hz). The selected parameters are
COMPOSITE EQUIVALENT parameters: they bundle three-layer bone reaction,
bone-electrolyte interface polarization and electrode response, are bound to
this measurement platform, and must not be interpreted as intrinsic
hydroxyapatite kinetics or transferred to other electrode setups.

The v2 observation operator represents measured impedance as a series sum of
bone-matrix and electrode–bone interface contributions. The interface term
decreases with the mechanism-predicted outer-surface porosity. Residual-drift
diagnostics check the remaining structural mismatch and are reported whether
or not they pass.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


LOW_FREQUENCIES_HZ = (100.0, 120.0)
CONDITION_KEYS = ["stage", "edta_concentration_mol_per_l", "time_s"]

PLATFORM_DISCLAIMER = (
    "Composite equivalent parameters bundling bone reaction, interface "
    "polarization and electrode response; valid only for this measurement "
    "platform and not transferable."
)
INTERFACE_FRACTION_RANGE = (0.3, 0.95)
INTERFACE_EXPONENT_RANGE = (0.5, 3.0)


def sample_interface_lhs(sample_count: int, seed: int) -> np.ndarray:
    """Two LHS dimensions: baseline interface fraction (linear), exponent (log)."""
    if sample_count < 1:
        raise ValueError("sample_count must be positive")
    rng = np.random.default_rng(seed)
    strata = []
    for _ in range(2):
        values = (np.arange(sample_count) + rng.random(sample_count)) / sample_count
        rng.shuffle(values)
        strata.append(values)
    beta = INTERFACE_FRACTION_RANGE[0] + strata[0] * (
        INTERFACE_FRACTION_RANGE[1] - INTERFACE_FRACTION_RANGE[0]
    )
    log_lower, log_upper = np.log10(INTERFACE_EXPONENT_RANGE)
    exponent = 10.0 ** (log_lower + strata[1] * (log_upper - log_lower))
    return np.column_stack([beta, exponent])


def apply_dual_branch_forward_operator(
    predictions: pd.DataFrame, interface_parameters: pd.DataFrame
) -> pd.DataFrame:
    """Map mechanism trajectories to measured-Z-ratio predictions.

    Z/Z0 = (1-beta) * R_matrix/R_matrix0
           + beta * (porosity_surface0/porosity_surface(t))**m.

    The legacy prediction column is overwritten with this composite measured-Z
    prediction so the same screening/LOO machinery applies unchanged; the
    matrix-only and interface-only trajectories are retained in new columns.
    """
    required = {
        "candidate_id", "sim_whole_resistance_ratio",
        "sim_outer_surface_porosity_ratio",
    }
    missing = required.difference(predictions.columns)
    if missing:
        raise ValueError(f"Missing dual-branch prediction columns: {sorted(missing)}")
    parameter_columns = ["candidate_id", "interface_fraction", "interface_exponent"]
    if interface_parameters["candidate_id"].duplicated().any():
        raise ValueError("Interface parameters must have one row per candidate")
    merged = predictions.merge(
        interface_parameters[parameter_columns], on="candidate_id",
        how="left", validate="many_to_one",
    )
    if merged[parameter_columns[1:]].isna().any().any():
        raise ValueError("Missing interface parameters for one or more candidates")
    matrix = merged["sim_whole_resistance_ratio"].to_numpy(dtype=float)
    surface = merged["sim_outer_surface_porosity_ratio"].to_numpy(dtype=float)
    if np.any(surface <= 0):
        raise ValueError("Surface-porosity ratios must be positive")
    beta = merged["interface_fraction"].to_numpy(dtype=float)
    exponent = merged["interface_exponent"].to_numpy(dtype=float)
    interface = surface ** (-exponent)
    merged["sim_matrix_resistance_ratio"] = matrix
    merged["sim_interface_impedance_ratio"] = interface
    merged["sim_whole_resistance_ratio"] = (1.0 - beta) * matrix + beta * interface
    return merged


def interface_contribution_summary(
    composite_predictions: pd.DataFrame, candidate_ids: np.ndarray
) -> dict[str, float | int | str]:
    """Compatible-set interface share at baseline and of the predicted drop."""
    selected = composite_predictions[
        composite_predictions["candidate_id"].isin(candidate_ids)
    ]
    if selected.empty:
        raise ValueError("Cannot summarize an empty compatible set")
    baseline = selected.drop_duplicates("candidate_id")["interface_fraction"].to_numpy()
    total_drop = 1.0 - selected["sim_whole_resistance_ratio"].to_numpy(dtype=float)
    interface_drop = selected["interface_fraction"].to_numpy(dtype=float) * (
        1.0 - selected["sim_interface_impedance_ratio"].to_numpy(dtype=float)
    )
    valid = (total_drop > 1e-8) & (interface_drop >= 0.0)
    shares = interface_drop[valid] / total_drop[valid]
    share_quantiles = (
        np.quantile(shares, [0.05, 0.50, 0.95])
        if shares.size else np.full(3, np.nan)
    )
    return {
        "baseline_fraction_q05": float(np.quantile(baseline, 0.05)),
        "baseline_fraction_q50": float(np.quantile(baseline, 0.50)),
        "baseline_fraction_q95": float(np.quantile(baseline, 0.95)),
        "drop_share_q05": float(share_quantiles[0]),
        "drop_share_q50": float(share_quantiles[1]),
        "drop_share_q95": float(share_quantiles[2]),
        "drop_share_point_count": int(shares.size),
        "definition": "Full-data, unweighted distribution over compatible candidates and simulated conditions; descriptive platform decomposition, not a posterior probability or independently validated causal share.",
    }


def build_z_ratio_targets(
    spectra: pd.DataFrame,
    frequencies: tuple[float, ...] = LOW_FREQUENCIES_HZ,
    drop_confounded: bool = True,
) -> pd.DataFrame:
    """Rat-level Z_post/Z_baseline targets from raw magnitude ratios.

    Sites (and the two low frequencies) are averaged within each rat before
    any between-rat statistics. The acetic-acid rows inherit each rat's EDTA
    dose from its EDTA block (the workbook stores a placeholder dose there).
    """
    frame = spectra[spectra["frequency_hz"].isin(frequencies)].copy()
    if drop_confounded:
        frame = frame[frame["confounder"] == ""]
    dose_map = (
        frame[frame["stage"] == "edta"]
        .groupby("rat")["edta_concentration_mol_per_l"]
        .agg(lambda values: sorted(set(values)))
    )
    if dose_map.map(len).max() > 1:
        raise ValueError("A rat maps to multiple EDTA doses; check the workbook")
    rat_dose = dose_map.map(lambda values: float(values[0]))
    frame.loc[frame["stage"] != "edta", "edta_concentration_mol_per_l"] = np.nan
    frame["edta_concentration_mol_per_l"] = frame["edta_concentration_mol_per_l"].fillna(
        frame["rat"].map(rat_dose)
    )

    baseline = (
        frame[frame["stage"] == "baseline"]
        .set_index(["rat", "site", "frequency_hz"])["magnitude_mohm"]
    )
    post = frame[frame["stage"].isin(("acetic_acid", "edta"))].copy()
    post = post.join(baseline.rename("baseline_mohm"), on=["rat", "site", "frequency_hz"])
    post = post[np.isfinite(post["baseline_mohm"])]
    post["z_ratio"] = post["magnitude_mohm"] / post["baseline_mohm"]

    targets = post.groupby(["rat", *CONDITION_KEYS], as_index=False).agg(
        z_ratio=("z_ratio", "mean"),
        n_values=("z_ratio", "count"),
        site_count=("site", "nunique"),
    )
    return targets.sort_values(CONDITION_KEYS + ["rat"]).reset_index(drop=True)


def _pivot_predictions(
    predictions: pd.DataFrame, rows: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Align candidate predictions to target rows; return (sim_matrix, candidate_ids).

    Target rows may repeat a condition (several rats per condition), so
    alignment is by row position, not by the condition key alone.
    """
    keyed = rows[CONDITION_KEYS].reset_index(drop=True)
    keyed["_row"] = np.arange(len(keyed))
    merged = keyed.merge(predictions, on=CONDITION_KEYS, how="left", validate="many_to_many")
    if merged["sim_whole_resistance_ratio"].isna().any():
        missing = merged[merged["sim_whole_resistance_ratio"].isna()][CONDITION_KEYS].drop_duplicates()
        raise ValueError(f"Missing simulated predictions for conditions: {missing.to_dict('records')}")
    matrix = merged.pivot(index="candidate_id", columns="_row", values="sim_whole_resistance_ratio")
    matrix = matrix[np.arange(len(keyed))]
    return matrix.to_numpy(dtype=float), matrix.index.to_numpy()


def condition_tolerance(train: pd.DataFrame) -> float:
    """RMS of per-condition between-rat SD; the compatible-set loss threshold."""
    sd = train.groupby(CONDITION_KEYS)["z_ratio"].std(ddof=1).dropna()
    if sd.empty:
        raise ValueError("No condition has at least two rats; cannot derive tolerance")
    return float(np.sqrt(np.mean(sd.to_numpy() ** 2)))


def compatible_candidates(
    predictions: pd.DataFrame, train: pd.DataFrame, tolerance: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (compatible candidate_ids, all candidate_ids, RMSE per candidate)."""
    sim_matrix, candidate_ids = _pivot_predictions(predictions, train)
    observed = train["z_ratio"].to_numpy(dtype=float)
    rmse = np.sqrt(np.mean((sim_matrix - observed[None, :]) ** 2, axis=1))
    compatible = candidate_ids[rmse <= tolerance]
    return compatible, candidate_ids, rmse


def prediction_quantiles(
    predictions: pd.DataFrame, candidate_ids: np.ndarray, rows: pd.DataFrame
) -> pd.DataFrame:
    """5/50/95% quantiles of the compatible-set predictions for each target row."""
    sim_matrix, all_ids = _pivot_predictions(predictions, rows)
    selected = np.isin(all_ids, candidate_ids)
    if not np.any(selected):
        raise ValueError("Empty compatible set: no prediction interval available")
    chosen = sim_matrix[selected]
    quantiles = np.quantile(chosen, [0.05, 0.5, 0.95], axis=0)
    result = rows.copy().reset_index(drop=True)
    result["sim_q05"] = quantiles[0]
    result["sim_q50"] = quantiles[1]
    result["sim_q95"] = quantiles[2]
    result["compatible_count"] = int(np.sum(selected))
    return result


def leave_one_rat_out(
    predictions: pd.DataFrame, targets: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Leave-one-rat-out validation; returns per-point and per-rat summaries."""
    point_rows = []
    rat_rows = []
    for held_out in sorted(targets["rat"].unique()):
        train = targets[targets["rat"] != held_out]
        test = targets[targets["rat"] == held_out]
        tolerance = condition_tolerance(train)
        compatible, _, _ = compatible_candidates(predictions, train, tolerance)
        if compatible.size == 0:
            rat_rows.append(
                {"rat": held_out, "compatible_count": 0, "tolerance": tolerance,
                 "n_points": int(len(test)), "coverage": np.nan, "median_width": np.nan}
            )
            continue
        intervals = prediction_quantiles(predictions, compatible, test)
        covered = (intervals["z_ratio"] >= intervals["sim_q05"]) & (
            intervals["z_ratio"] <= intervals["sim_q95"]
        )
        width = intervals["sim_q95"] - intervals["sim_q05"]
        point_rows.append(
            intervals.assign(rat=held_out, covered=covered.to_numpy(), width=width.to_numpy())
        )
        rat_rows.append(
            {
                "rat": held_out,
                "compatible_count": int(compatible.size),
                "tolerance": tolerance,
                "n_points": int(len(test)),
                "coverage": float(np.mean(covered)),
                "median_width": float(np.median(width)),
            }
        )
    points = pd.concat(point_rows, ignore_index=True) if point_rows else pd.DataFrame()
    return pd.DataFrame(rat_rows), points


def residual_drift(
    targets: pd.DataFrame, intervals: pd.DataFrame
) -> dict[str, float | str]:
    """Systematic residual structure vs time/stage; reported pass or fail.

    residual = observed - compatible-set median. Strong monotone drift in
    time indicates structural mismatch in the composite forward operator.
    """
    merged = targets.merge(
        intervals[["rat", *CONDITION_KEYS, "sim_q50"]],
        on=["rat", *CONDITION_KEYS],
        how="inner",
    )
    merged = merged.assign(residual=merged["z_ratio"] - merged["sim_q50"])
    edta = merged[merged["stage"] == "edta"]
    if len(edta) >= 8:
        rho_time, p_time = spearmanr(edta["time_s"], edta["residual"])
    else:
        rho_time, p_time = np.nan, np.nan
    by_stage = merged.groupby("stage")["residual"].agg(["mean", "std", "count"])
    return {
        "residual_vs_edta_time_spearman_rho": float(rho_time),
        "residual_vs_edta_time_spearman_p": float(p_time),
        "drift_flag_|rho|>0.4_and_p<0.05": bool(
            np.isfinite(rho_time) and abs(rho_time) > 0.4 and p_time < 0.05
        ),
        "stage_residual_mean": {
            stage: float(row["mean"]) for stage, row in by_stage.iterrows()
        },
        "note": "Residual structure is reported as a diagnostic either way; a "
        "drift flag weakens (not voids) the equivalent-prediction interpretation.",
    }
