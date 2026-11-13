#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""TrackA / A2 — 协变量扩展校正（补时间 t 与频率 ln(f/100) 两个机理先验缺失的维度）"""
import os, numpy as np, pandas as pd

P = np.array([-1.94636172, 2.87513586, -0.00284766, 0.04886211], float)   # [a, b, c_t, d_lnf]
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

def predict(dose, t_min, freq_hz, anchor_baseline=True):
    """R_Z = a + b*raw + c*t + d*ln(f/100)；t=0 按物理定义锚定。"""
    if anchor_baseline and t_min == 0: return 1.0
    raw = raw_prior(dose, t_min)
    return float(P[0] + P[1]*raw + P[2]*t_min + P[3]*np.log(freq_hz/F0))

def info():
    return {"model_id":"TrackA_A2","formula":"R_Z = a + b*raw + c*t + d*ln(f/100)",
             "params":P.tolist(),"f0_hz":F0,"baseline_anchor":1.0,
             "label":"empirical covariates added to the calibration layer; c,d have no physical derivation"}
