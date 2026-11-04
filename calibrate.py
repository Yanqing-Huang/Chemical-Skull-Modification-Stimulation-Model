#!/usr/bin/env python3
"""Calibrate selected prior parameters against an in-vitro observation CSV.

Observable names: ca_remaining_fraction, ca_release_mol_per_m2, porosity,
area_resistance_ohm_m2, area_resistance_ratio_to_native. Times are local to
each named stage. The output includes prior and calibrated predictions side by side.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cranial_model.calibration import (
    export_calibration_predictions,
    fit_parameters,
    read_experimental_data,
)
from cranial_model.core import load_parameters


ROOT = Path(__file__).resolve().parent


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="Completed experimental CSV.")
    parser.add_argument("--parameters", type=Path, default=ROOT / "parameters" / "theoretical_prior.json")
    parser.add_argument(
        "--fit",
        nargs="+",
        default=["stages.edta.k_r_m3_per_mol_s"],
        help="Dotted positive-parameter paths with declared uncertainty ranges.",
    )
    parser.add_argument("--cells", nargs=3, type=int, default=[24, 12, 12], help="Grid used during fitting.")
    parser.add_argument("--max-step", type=float, default=10.0, help="Maximum BDF step during fitting (s).")
    parser.add_argument("--max-nfev", type=int, default=100)
    parser.add_argument("--output", type=Path, default=ROOT / "results" / "calibration")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    document = load_parameters(args.parameters)
    observations = read_experimental_data(args.data)
    calibrated, metadata, prior, fitted = fit_parameters(
        document,
        observations,
        args.fit,
        cells_per_layer=args.cells,
        max_step_s=args.max_step,
        max_nfev=args.max_nfev,
    )
    metadata["experimental_data_file"] = str(args.data.resolve())
    calibrated["calibration_metadata"] = metadata
    with (args.output / "calibrated_parameters.json").open("w", encoding="utf-8") as handle:
        json.dump(calibrated, handle, indent=2, ensure_ascii=False)
    with (args.output / "calibration_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, ensure_ascii=False)
    export_calibration_predictions(
        args.output / "prior_vs_calibrated_predictions.csv", observations, prior, fitted
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"Calibration outputs written to {args.output}")


if __name__ == "__main__":
    main()
