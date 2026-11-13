#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""TrackA / A3 — 机理融合混合式（频率维度写回机理层，校准层只留一个时间余项）"""
import os, numpy as np, pandas as pd

KAPPA, S, EPS = [3.30389396, 0.11515978, 0.00291183]   # kappa, s, eps
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
    if _D is None: _D = _load()
    g = _D[_D.edta_concentration_mol_per_l == dose]
    ts = g.global_time_s.to_numpy(float); vs = g.q50.to_numpy(float)
    t_s = t_min * 60.0
    return float(vs[-1]) if t_s > ts.max() else float(np.interp(t_s, ts, vs))

def predict(dose, t_min, freq_hz):
    """
    delta(t,d) = 1 - R_Z_raw(t,d)                (机理预测的改性量)
    lambda(f)  = (f/f0)^(-s)                     (界面极化 / CPE 频散可见度)
    R_Z = 1 - kappa*lambda(f)*delta - eps*t
    基线 t=0 时 delta=0，故 R_Z 恒等于 1，无需人为锚定。
    """
    delta = 1.0 - raw_prior(dose, t_min)
    lam = (freq_hz / F0) ** (-S)
    return float(1.0 - KAPPA * lam * delta - EPS * t_min)

def info():
    return {"model_id":"TrackA_A3",
             "formula":"R_Z = 1 - kappa*(f/f0)^(-s)*(1-R_Z_raw) - eps*t",
             "kappa":KAPPA,"s":S,"eps_per_min":EPS,"f0_hz":F0,
             "baseline_anchor":"automatic (=1 by construction)",
             "label":"mechanism-informed parameterization; s is NOT an identified CPE exponent"}
