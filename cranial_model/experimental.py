"""Read the supplied in-vitro workbook and build quality-controlled R0 ratios."""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


FREQUENCY_COLUMNS = {100: 5, 120: 9, 1000: 13, 10000: 17}
EDTA_TIMES_MIN = (15, 30, 60, 90)


def normalize_rat(value) -> str:
    return re.sub(r"\s+", "", str(value)).upper()


def normalize_site(value) -> str:
    text = str(value).strip().lower()
    if text.startswith("bregma"):
        return "Bregma"
    if "中间" in text:
        return "中间值"
    if text.startswith(("lamda", "lambda")):
        return "Lambda"
    raise ValueError(f"Unknown skull site: {value!r}")


def _number(ws, row: int, column: int) -> float:
    value = ws.cell(row, column).value
    if not isinstance(value, (int, float)) or isinstance(value, bool) or not np.isfinite(value):
        raise ValueError(f"Invalid numeric cell {ws.cell(row, column).coordinate}: {value!r}")
    return float(value)


def read_experimental_spectra(path: str | Path) -> pd.DataFrame:
    """Return one row per frequency measurement, with 10 kHz converted kΩ→MΩ."""
    ws = load_workbook(path, data_only=True).active
    rows = []
    for stage, excel_rows in (("baseline", range(5, 77)), ("acetic_acid", range(77, 149))):
        current = {}
        for excel_row in excel_rows:
            for column, key in ((2, "date"), (3, "rat"), (4, "site")):
                value = ws.cell(excel_row, column).value
                if value is not None:
                    current[key] = value
            rat = normalize_rat(current["rat"])
            site = normalize_site(current["site"])
            for frequency, column in FREQUENCY_COLUMNS.items():
                scale = 0.001 if frequency == 10000 else 1.0
                rows.append({
                    "rat": rat,
                    "site": site,
                    "date": str(current["date"]),
                    "stage": stage,
                    "edta_concentration_mol_per_l": 0.1,
                    "time_s": 300.0 if stage == "acetic_acid" else 0.0,
                    "frequency_hz": frequency,
                    "magnitude_mohm": _number(ws, excel_row, column) * scale,
                    "source_cell": ws.cell(excel_row, column).coordinate,
                    "confounder": "",
                })
    current = {}
    for excel_row in range(151, 223):
        for column, key in ((1, "dose"), (2, "date"), (3, "rat")):
            value = ws.cell(excel_row, column).value
            if value is not None:
                current[key] = value
        rat = normalize_rat(current["rat"])
        site = normalize_site(ws.cell(excel_row, 4).value)
        dose = float(current["dose"])
        date = str(current["date"])
        for frequency, start_column in FREQUENCY_COLUMNS.items():
            scale = 0.001 if frequency == 10000 else 1.0
            for offset, minutes in enumerate(EDTA_TIMES_MIN):
                column = start_column + offset
                flags = []
                if "260523" in date:
                    flags.append("冻存4天；备注含乙醇+尿素处理")
                if "260719" in date and minutes >= 90:
                    flags.append("60 min后冻存约4天，再测90 min")
                rows.append({
                    "rat": rat,
                    "site": site,
                    "date": date,
                    "stage": "edta",
                    "edta_concentration_mol_per_l": dose,
                    "time_s": float(minutes * 60),
                    "frequency_hz": frequency,
                    "magnitude_mohm": _number(ws, excel_row, column) * scale,
                    "source_cell": ws.cell(excel_row, column).coordinate,
                    "confounder": "；".join(flags),
                })
    frame = pd.DataFrame(rows)
    frame["spectrum_id"] = frame.apply(
        lambda row: f"{row['rat']}_{row['site']}_{row['stage']}_{row['edta_concentration_mol_per_l']:.2f}_{row['time_s']:.0f}",
        axis=1,
    )
    return frame


def apply_magnitude_calibration(spectra: pd.DataFrame, calibration: pd.DataFrame) -> pd.DataFrame:
    """Correct magnitudes using independently measured standard-load response.

    calibration must contain frequency_hz and measured_over_true. The factor is
    interpolated in log-log space; extrapolation beyond the calibrated range is
    forbidden. Skull measurements must never be used to derive this factor.
    """
    required = {"frequency_hz", "measured_over_true"}
    missing = required.difference(calibration.columns)
    if missing:
        raise ValueError(f"Missing calibration columns: {sorted(missing)}")
    reference = calibration[["frequency_hz", "measured_over_true"]].copy()
    reference["frequency_hz"] = pd.to_numeric(reference["frequency_hz"], errors="raise")
    reference["measured_over_true"] = pd.to_numeric(reference["measured_over_true"], errors="raise")
    reference = reference.sort_values("frequency_hz")
    frequency = reference["frequency_hz"].to_numpy(dtype=float)
    factor = reference["measured_over_true"].to_numpy(dtype=float)
    if (
        len(reference) < 2
        or np.any(~np.isfinite(frequency))
        or np.any(~np.isfinite(factor))
        or np.any(frequency <= 0)
        or np.any(factor <= 0)
        or np.unique(frequency).size != frequency.size
    ):
        raise ValueError("Calibration requires at least two unique positive frequencies and positive finite factors")
    measured_frequency = spectra["frequency_hz"].to_numpy(dtype=float)
    if measured_frequency.min() < frequency.min() or measured_frequency.max() > frequency.max():
        raise ValueError("Experimental frequencies exceed the independently calibrated range; extrapolation is forbidden")
    interpolated = np.exp(
        np.interp(np.log(measured_frequency), np.log(frequency), np.log(factor))
    )
    corrected = spectra.copy()
    corrected["uncorrected_magnitude_mohm"] = corrected["magnitude_mohm"]
    corrected["magnitude_calibration_measured_over_true"] = interpolated
    corrected["magnitude_mohm"] = corrected["magnitude_mohm"] / interpolated
    return corrected


def build_r0_ratio_observations(
    spectra: pd.DataFrame,
    fitter: Callable[[np.ndarray, np.ndarray], dict],
    instrument_max_mohm: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit every four-frequency spectrum and match post-treatment R0 to its baseline."""
    fit_rows = []
    keys = ["spectrum_id", "rat", "site", "date", "stage", "edta_concentration_mol_per_l", "time_s", "confounder"]
    for spectrum_id, group in spectra.groupby("spectrum_id", sort=False):
        group = group.sort_values("frequency_hz")
        result = fitter(
            group["frequency_hz"].to_numpy(dtype=float),
            group["magnitude_mohm"].to_numpy(dtype=float),
        )
        first = group.iloc[0]
        outside = group["magnitude_mohm"] > instrument_max_mohm if instrument_max_mohm else np.zeros(len(group), dtype=bool)
        fit_rows.append({
            **{key: first[key] for key in keys},
            "frequencies_hz": ";".join(str(int(value)) for value in group["frequency_hz"]),
            "source_cells": ";".join(group["source_cell"]),
            "outside_instrument_range": bool(np.any(outside)),
            "outside_instrument_range_frequencies_hz": ";".join(
                str(int(value)) for value in group.loc[outside, "frequency_hz"]
            ),
            **result,
        })
    fits = pd.DataFrame(fit_rows)
    baseline = fits[fits["stage"] == "baseline"].set_index(["rat", "site"])
    observations = []
    for row in fits[fits["stage"] != "baseline"].itertuples(index=False):
        base = baseline.loc[(row.rat, row.site)]
        ratio = row.r0_mohm / base.r0_mohm if row.r0_mohm > 0 and base.r0_mohm > 0 else np.nan
        reasons = []
        if instrument_max_mohm is not None:
            if base.outside_instrument_range:
                reasons.append("baseline_outside_instrument_range")
            if row.outside_instrument_range:
                reasons.append("post_outside_instrument_range")
        if base.quality_flag != "accepted":
            reasons.append(f"baseline_{base.quality_flag}")
        if row.quality_flag != "accepted":
            reasons.append(f"post_{row.quality_flag}")
        if not np.isfinite(ratio) or ratio <= 0:
            reasons.append("invalid_r0_ratio")
        observations.append({
            "experiment_id": row.spectrum_id,
            "rat": row.rat,
            "site": row.site,
            "date": row.date,
            "stage": row.stage,
            "edta_concentration_mol_per_l": row.edta_concentration_mol_per_l,
            "time_s": row.time_s,
            "r0_base_mohm": base.r0_mohm,
            "r0_post_mohm": row.r0_mohm,
            "r0_ratio": ratio,
            "valid": not reasons,
            "rejection_reason": ";".join(reasons),
            "baseline_relative_rmse": base.relative_rmse,
            "post_relative_rmse": row.relative_rmse,
            "baseline_identifiable": bool(base.identifiable),
            "post_identifiable": bool(row.identifiable),
            "baseline_outside_instrument_range": bool(base.outside_instrument_range),
            "post_outside_instrument_range": bool(row.outside_instrument_range),
            "confounder": row.confounder,
        })
    return fits, pd.DataFrame(observations)


def rat_condition_means(observations: pd.DataFrame, valid_only: bool = True) -> pd.DataFrame:
    """Average available sites within each rat before any between-rat statistics."""
    frame = observations[observations["valid"]].copy() if valid_only else observations.copy()
    return frame.groupby(
        ["rat", "stage", "edta_concentration_mol_per_l", "time_s"], as_index=False
    ).agg(
        r0_ratio=("r0_ratio", "mean"),
        site_count=("r0_ratio", "count"),
        confounder=("confounder", lambda values: "；".join(sorted({v for v in values if v}))),
    )


def split_rats_stratified(rat_means: pd.DataFrame, seed: int, test_fraction: float = 0.3) -> dict[str, str]:
    """Freeze a rat-level, EDTA-dose-stratified split without site/time leakage."""
    rng = np.random.default_rng(seed)
    edta = rat_means[rat_means["stage"] == "edta"]
    rat_dose = edta[["rat", "edta_concentration_mol_per_l"]].drop_duplicates()
    groups = [(dose, sorted(group["rat"].unique())) for dose, group in rat_dose.groupby("edta_concentration_mol_per_l")]
    total = sum(len(rats) for _, rats in groups)
    target_test = int(round(total * test_fraction))
    quotas = []
    for dose, rats in groups:
        exact = len(rats) * test_fraction
        quotas.append([dose, rats, max(1, int(np.floor(exact))), exact - np.floor(exact)])
    assigned = sum(item[2] for item in quotas)
    order = rng.permutation(len(quotas))
    for index in sorted(order, key=lambda i: quotas[i][3], reverse=True):
        if assigned >= target_test:
            break
        if quotas[index][2] < len(quotas[index][1]) - 1:
            quotas[index][2] += 1
            assigned += 1
    split = {}
    for _, rats, test_count, _ in quotas:
        shuffled = rng.permutation(rats)
        test = set(shuffled[:test_count])
        for rat in rats:
            split[rat] = "validation" if rat in test else "calibration"
    if sum(value == "validation" for value in split.values()) != target_test:
        raise RuntimeError("Failed to allocate requested independent-validation count")
    return split


def condition_statistics(rat_means: pd.DataFrame, split_name: str) -> tuple[list[dict], pd.DataFrame]:
    """Condition means/SD are computed only from the requested frozen split."""
    frame = rat_means[rat_means["split"] == split_name]
    conditions = []
    keys = ["stage", "edta_concentration_mol_per_l", "time_s"]
    for key, group in frame.groupby(keys):
        count = len(group)
        conditions.append({
            **{name: float(value) if name != "stage" else value for name, value in zip(keys, key)},
            "observed_mean": float(group["r0_ratio"].mean()),
            "observed_sd": float(group["r0_ratio"].std(ddof=1)) if count > 1 else 0.0,
            "rat_count": int(count),
            "observations": group["r0_ratio"].tolist(),
        })
    conditions.sort(key=lambda row: (row["stage"] != "acetic_acid", row["edta_concentration_mol_per_l"], row["time_s"]))
    return conditions, frame
