"""Data-independent local geometry sampling for three SD-rat skull sites.

Geometry is sampled independently of all impedance outcomes. Photographs provide
only qualitative motivation; they are not used to infer numerical dimensions.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd


SITE_NAMES = ("Bregma", "MidPoint", "Lambda")
# Exploratory equivalent shell radii, not a photographic measurement.
CURVATURE_RADIUS_RANGE_M = (0.003, 0.030)


def sample_site_geometries(
    mechanism_ids: np.ndarray,
    parameter_document: dict,
    seed: int = 20261401,
) -> pd.DataFrame:
    """Sample independent local geometry for every mechanism and anatomical site."""
    rng = np.random.default_rng(seed)
    uncertainty = parameter_document["uncertainty"]
    rows = []
    for mechanism_id in mechanism_ids:
        for site in SITE_NAMES:
            while True:
                thickness = np.array([
                    rng.uniform(*uncertainty[f"layers.{layer}.thickness_mm"]["range"])
                    for layer in ("outer_table", "diploe", "inner_table")
                ])
                total = float(thickness.sum())
                lower, upper = uncertainty["geometry.total_thickness_mm"]["range"]
                if lower <= total <= upper:
                    break
            row = {
                "mechanism_id": int(mechanism_id),
                "site": site,
                "total_thickness_mm": total,
                "outer_curvature_radius_m": float(np.exp(rng.uniform(
                    *np.log(CURVATURE_RADIUS_RANGE_M)
                ))),
            }
            for index, layer in enumerate(("outer_table", "diploe", "inner_table")):
                row[f"{layer}_thickness_mm"] = float(thickness[index])
                row[f"{layer}_porosity"] = float(rng.uniform(
                    *uncertainty[f"layers.{layer}.porosity"]["range"]
                ))
                row[f"{layer}_initial_calcium_mol_per_m3"] = float(rng.uniform(
                    *uncertainty[f"layers.{layer}.initial_calcium_mol_per_m3"]["range"]
                ))
            rows.append(row)
    return pd.DataFrame(rows)


def apply_site_geometry(base_values: dict, geometry: pd.Series) -> dict:
    """Apply one sampled local geometry without changing shared chemistry."""
    values = copy.deepcopy(base_values)
    values["geometry"]["total_thickness_mm"] = float(geometry["total_thickness_mm"])
    values["geometry"]["outer_curvature_radius_m"] = float(
        geometry["outer_curvature_radius_m"]
    )
    for layer in values["geometry"]["layers"]:
        name = layer["name"]
        layer["thickness_mm"] = float(geometry[f"{name}_thickness_mm"])
        layer["porosity"] = float(geometry[f"{name}_porosity"])
        layer["initial_calcium_mol_per_m3"] = float(
            geometry[f"{name}_initial_calcium_mol_per_m3"]
        )
    return values
