"""Exploratory 1-kHz magnitude-ratio ensemble fit; not an intrinsic-R0 validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from cranial_model import load_parameters
from cranial_model.ensemble import (
    PARAMETER_PATHS,
    apply_candidate,
    monotonic_trend_fraction,
    prediction_interval,
    sample_lhs_log10,
    screen_candidates,
)
from cranial_model.experimental import read_experimental_spectra, split_rats_stratified
from validate_ensemble import instrument_range_summary, sha256, write_json

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = Path("/workspace/.uploads/a32f8259-e715-40b5-a800-8c39875ea94b_体外EDTA数据记录汇总 (1).xlsx")


def build_1khz_observations(spectra: pd.DataFrame, maximum_mohm: float = 10.0) -> pd.DataFrame:
    frame = spectra[spectra["frequency_hz"] == 1000].copy()
    baseline = frame[frame["stage"] == "baseline"].set_index(["rat", "site"])
    rows = []
    for row in frame[frame["stage"] != "baseline"].itertuples(index=False):
        base = baseline.loc[(row.rat, row.site)]
        ratio = float(row.magnitude_mohm / base.magnitude_mohm)
        reasons = []
        if base.magnitude_mohm > maximum_mohm:
            reasons.append("baseline_above_nominal_resistance_range")
        if row.magnitude_mohm > maximum_mohm:
            reasons.append("post_above_nominal_resistance_range")
        if not np.isfinite(ratio) or ratio <= 0:
            reasons.append("invalid_ratio")
        rows.append({
            "experiment_id": row.spectrum_id,
            "rat": row.rat,
            "site": row.site,
            "date": row.date,
            "stage": row.stage,
            "edta_concentration_mol_per_l": row.edta_concentration_mol_per_l,
            "time_s": row.time_s,
            "baseline_magnitude_mohm": float(base.magnitude_mohm),
            "post_magnitude_mohm": float(row.magnitude_mohm),
            "magnitude_ratio": ratio,
            "valid": not reasons,
            "rejection_reason": ";".join(reasons),
            "confounder": row.confounder,
        })
    return pd.DataFrame(rows)


def within_rat_means(observations: pd.DataFrame) -> pd.DataFrame:
    valid = observations[observations["valid"]]
    return valid.groupby(
        ["rat", "stage", "edta_concentration_mol_per_l", "time_s"], as_index=False
    ).agg(
        observed_ratio=("magnitude_ratio", "mean"),
        site_count=("magnitude_ratio", "count"),
        confounder=("confounder", lambda values: "；".join(sorted({v for v in values if v}))),
    )


def condition_statistics(rat_means: pd.DataFrame, split_name: str) -> tuple[list[dict], pd.DataFrame]:
    frame = rat_means[rat_means["split"] == split_name]
    keys = ["stage", "edta_concentration_mol_per_l", "time_s"]
    conditions = []
    for key, group in frame.groupby(keys):
        count = len(group)
        conditions.append({
            "stage": key[0],
            "edta_concentration_mol_per_l": float(key[1]),
            "time_s": float(key[2]),
            "observed_mean": float(group["observed_ratio"].mean()),
            "observed_sd": float(group["observed_ratio"].std(ddof=1)) if count > 1 else 0.0,
            "rat_count": int(count),
            "observations": group["observed_ratio"].tolist(),
        })
    conditions.sort(key=lambda row: (row["stage"] != "acetic_acid", row["edta_concentration_mol_per_l"], row["time_s"]))
    return conditions, frame


def condition_key(row: dict) -> tuple:
    return (row["stage"], float(row["edta_concentration_mol_per_l"]), float(row["time_s"]))


def band_rows(
    conditions: list[dict], predictions: np.ndarray, split_name: str
) -> list[dict]:
    lower, upper = prediction_interval(predictions)
    median = np.median(predictions, axis=0)
    rows = []
    for index, condition in enumerate(conditions):
        rows.append({
            **{key: condition[key] for key in ("stage", "edta_concentration_mol_per_l", "time_s", "observed_mean", "observed_sd", "rat_count")},
            "split": split_name,
            "sim_min": float(lower[index]),
            "sim_median": float(median[index]),
            "sim_max": float(upper[index]),
            "mean_inside_band": bool(lower[index] <= condition["observed_mean"] <= upper[index]),
        })
    return rows


def source_hashes(data_path: Path, parameter_path: Path) -> dict[str, str]:
    files = [
        data_path,
        parameter_path,
        ROOT / "cranial_model" / "core.py",
        ROOT / "cranial_model" / "ensemble.py",
        ROOT / "cranial_model" / "experimental.py",
        Path(__file__),
    ]
    return {str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else path.name: sha256(path) for path in files}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--parameters", type=Path, default=ROOT / "parameters" / "theoretical_prior.json")
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "exploratory_1khz_fit")
    parser.add_argument("--samples", type=int, default=500)
    parser.add_argument("--split-seed", type=int, default=20261110)
    parser.add_argument("--lhs-seed", type=int, default=20261111)
    parser.add_argument("--cells", type=int, nargs=3, default=[12, 6, 6])
    parser.add_argument("--max-step", type=float, default=30.0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.samples < 500:
        raise ValueError("The frozen exploratory analysis requires at least 500 LHS candidates")
    args.output.mkdir(parents=True, exist_ok=True)

    document = load_parameters(args.parameters)
    base_values = document["values"]
    base_values["numerics"]["output_times_per_stage"] = 31
    spectra = read_experimental_spectra(args.data)
    observations = build_1khz_observations(spectra)
    observations.to_csv(args.output / "magnitude_ratio_observations.csv", index=False)
    rat_means = within_rat_means(observations)
    split = split_rats_stratified(rat_means, args.split_seed, test_fraction=0.3)
    rat_means["split"] = rat_means["rat"].map(split)
    rat_means.to_csv(args.output / "rat_condition_means.csv", index=False)
    pd.DataFrame([
        {"rat": rat, "split": name, "split_seed": args.split_seed}
        for rat, name in sorted(split.items())
    ]).to_csv(args.output / "data_split.csv", index=False)

    calibration, calibration_frame = condition_statistics(rat_means, "calibration")
    tolerance = float(np.sqrt(np.mean(np.square([row["observed_sd"] for row in calibration]))))
    if not np.isfinite(tolerance) or tolerance <= 0:
        raise RuntimeError("Calibration-derived tolerance must be finite and positive")
    pd.DataFrame([
        {**{key: row[key] for key in ("stage", "edta_concentration_mol_per_l", "time_s", "observed_mean", "observed_sd", "rat_count")}, "observations": ";".join(str(v) for v in row["observations"])}
        for row in calibration
    ]).to_csv(args.output / "calibration_conditions.csv", index=False)

    protocol = {
        "status": "FROZEN_BEFORE_LHS_AND_INDEPENDENT_VALIDATION",
        "analysis_type": "EXPLORATORY_PROXY_FIT_NOT_INTRINSIC_R0_VALIDATION",
        "observable": "Intrinsic area-resistance ratio used as a proxy for measured 1-kHz magnitude ratio",
        "observable_mismatch": "The 1-kHz magnitude still contains interface/CPE contributions; this analysis cannot replace the frozen R-CPE workflow",
        "instrument": instrument_range_summary(spectra),
        "instrument_evidence": "User-supplied ZT-DQ01 images: nominal resistance range 0-10 Mohm; X/D/Q/theta/ESR supported but not saved in original Excel",
        "frequency_hz": 1000,
        "range_rule": "Both baseline and post 1-kHz readings must be <=10 Mohm",
        "valid_rats": int(rat_means["rat"].nunique()),
        "split": split,
        "split_seed": args.split_seed,
        "lhs_seed": args.lhs_seed,
        "sample_count": args.samples,
        "parameter_paths": list(PARAMETER_PATHS),
        "priors": {path: document["uncertainty"][path]["range"] for path in PARAMETER_PATHS},
        "calibration_conditions": calibration,
        "tolerance_formula": "sqrt(mean(calibration condition between-rat SD^2))",
        "tolerance": tolerance,
        "validation_rules": {"minimum_condition_coverage": 0.5, "minimum_both_monotonic_fraction": 0.9},
        "numerics": {"cells_per_layer": args.cells, "max_step_s": args.max_step, "output_times_per_stage_segment": 31, "workers": args.workers},
        "source_sha256": source_hashes(args.data, args.parameters),
    }
    write_json(args.output / "frozen_protocol.json", protocol)

    candidates = sample_lhs_log10(document, args.samples, args.lhs_seed)
    observed = np.array([row["observed_mean"] for row in calibration])
    rmse, accepted, calibration_predictions = screen_candidates(
        base_values,
        candidates,
        calibration,
        observed,
        tolerance,
        args.cells,
        args.max_step,
        progress=lambda done, total: print(f"LHS {done}/{total}", flush=True),
        workers=args.workers,
    )
    table = pd.DataFrame(candidates, columns=PARAMETER_PATHS)
    table.insert(0, "candidate_id", np.arange(len(candidates)))
    table["calibration_rmse"] = rmse
    table["accepted"] = accepted
    table.to_csv(args.output / "calibration_candidates.csv", index=False)
    posterior = table[accepted].copy()
    posterior.to_csv(args.output / "posterior_feasible_parameters.csv", index=False)

    if not np.any(accepted):
        result = {
            "status": "NO_FEASIBLE_CANDIDATE",
            "analysis_type": "EXPLORATORY_1KHZ_PROXY",
            "sample_count": args.samples,
            "tolerance": tolerance,
            "minimum_calibration_rmse": float(np.nanmin(rmse)),
            "validation_performed": False,
        }
        write_json(args.output / "validation_metrics.json", result)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    # The 17 simulated condition predictions are split-independent. Map the frozen
    # validation conditions to those predictions without rerunning or rescreening.
    validation, validation_frame = condition_statistics(rat_means, "validation")
    calibration_index = {condition_key(row): index for index, row in enumerate(calibration)}
    missing = [condition_key(row) for row in validation if condition_key(row) not in calibration_index]
    if missing:
        raise RuntimeError(f"Validation conditions absent from frozen calibration simulation grid: {missing}")
    validation_indices = [calibration_index[condition_key(row)] for row in validation]
    feasible_predictions = calibration_predictions[accepted]
    validation_predictions = feasible_predictions[:, validation_indices]
    trends = monotonic_trend_fraction(calibration, feasible_predictions)
    rows = band_rows(calibration, feasible_predictions, "calibration")
    rows.extend(band_rows(validation, validation_predictions, "validation"))
    bands = pd.DataFrame(rows)
    bands.to_csv(args.output / "prediction_bands.csv", index=False)
    np.savez(
        args.output / "posterior_condition_predictions.npz",
        candidate_id=posterior["candidate_id"].to_numpy(),
        calibration_predictions=feasible_predictions,
        validation_predictions=validation_predictions,
    )

    calibration_coverage = float(bands[bands.split == "calibration"]["mean_inside_band"].mean())
    validation_coverage = float(bands[bands.split == "validation"]["mean_inside_band"].mean())
    result = {
        "status": "EXPLORATORY_1KHZ_VALIDATION_COMPLETED",
        "analysis_type": "EXPLORATORY_PROXY_NOT_INTRINSIC_R0_VALIDATION",
        "sample_count": args.samples,
        "accepted_count": int(accepted.sum()),
        "acceptance_fraction": float(accepted.mean()),
        "tolerance": tolerance,
        "calibration_rmse_accepted": {
            "min": float(np.min(rmse[accepted])),
            "median": float(np.median(rmse[accepted])),
            "max": float(np.max(rmse[accepted])),
        },
        "calibration_condition_coverage": calibration_coverage,
        "validation_condition_coverage": validation_coverage,
        "validation_rat_count": int(validation_frame["rat"].nunique()),
        "validation_condition_count": len(validation),
        "validation_mean_band_width": float((bands[bands.split == "validation"]["sim_max"] - bands[bands.split == "validation"]["sim_min"]).mean()),
        "trends": trends,
        "coverage_criterion_pass": validation_coverage >= 0.5,
        "trend_criterion_pass": trends["both_monotonic_fraction"] >= 0.9,
        "interpretation": "Exploratory fit to in-range 1-kHz magnitude ratios only; not evidence that intrinsic R0 or the R-CPE protocol has been validated",
    }
    write_json(args.output / "validation_metrics.json", result)

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex=True, sharey=True)
    for ax, dose in zip(axes.ravel(), [0.05, 0.1, 0.15, 0.2]):
        dose_bands = bands[(bands.edta_concentration_mol_per_l == dose) & (bands.stage == "edta")]
        first = dose_bands[dose_bands.split == "calibration"].sort_values("time_s")
        x = first["time_s"].to_numpy() / 60.0
        ax.fill_between(x, first["sim_min"], first["sim_max"], color="0.8", alpha=0.7, label="Feasible envelope")
        for split_name, color in (("calibration", "tab:blue"), ("validation", "tab:red")):
            part = dose_bands[dose_bands.split == split_name].sort_values("time_s")
            ax.errorbar(part["time_s"] / 60.0, part["observed_mean"], yerr=part["observed_sd"], fmt="o", capsize=3, color=color, label=split_name)
        ax.set_title(f"EDTA {dose:.2f} M")
        ax.set_xlabel("Cumulative EDTA time (min)")
        ax.set_ylabel("1-kHz magnitude ratio / simulated proxy")
        ax.grid(alpha=0.2)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("Exploratory 1-kHz fit — not intrinsic R0 validation")
    fig.tight_layout()
    fig.savefig(args.output / "exploratory_1khz_fit.png", dpi=180)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
