#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""TrackA1 / A1.3 — 纯经验：H1 混合（机理改性量 delta 作形状骨架）

R_Z = 1 - kappa*(f/100)^(-s)*(1-R_Z_raw) - eps*t
delta = 1-R_Z_raw 来自 ../../data/mechanism_timeseries_quantiles.csv 的 q50。
kappa/s/eps 为经验拟合系数；t=0 时 delta=0 故天然等于 1。
"""
import os
import numpy as np
import pandas as pd

P = np.array([3.30389396, 0.11515978, 0.00291183], float)     # [kappa, s, eps]
F0 = 100.0
_HERE = os.path.dirname(os.path.abspath(__file__))
_PRIOR = os.path.join(_HERE, "..", "..", "data", "mechanism_timeseries_quantiles.csv")
_D = None


def _load():
    q = pd.read_csv(_PRIOR)
    return (q[q.metric == "whole_resistance_ratio"]
            .drop_duplicates(["edta_concentration_mol_per_l", "global_time_s"])
            .sort_values(["edta_concentration_mol_per_l", "global_time_s"]))


def raw_prior(dose, t_min):
    global _D
    if _D is None:
        _D = _load()
    g = _D[_D.edta_concentration_mol_per_l == dose]
    ts = g.global_time_s.to_numpy(float)
    vs = g.q50.to_numpy(float)
    t_s = t_min * 60.0
    return float(vs[-1]) if t_s > ts.max() else float(np.interp(t_s, ts, vs))


def predict(dose, t_min, freq_hz):
    delta = 1.0 - raw_prior(dose, t_min)
    lam = (freq_hz / F0) ** (-P[1])
    return float(1.0 - P[0] * lam * delta - P[2] * t_min)


def info():
    return {"model_id": "TrackA1.3", "route": "TrackA1 纯经验模型",
             "label": "H1 混合式", "formula": "R_Z = 1 - kappa*(f/100)^(-s)*(1-raw) - eps*t",
             "params": P.tolist(), "k": 3, "baseline": "automatic (delta(0)=0)",
             "mechanism_input": "机理改性量 delta=1-R_Z_raw 作形状骨架（仍以经验系数拟合）",
             "label_honest": "mechanism-informed SHAPE, but kappa/s/eps are fitted, not derived"}
