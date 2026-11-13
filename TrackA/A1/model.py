#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""TrackA / A1 — 经验线性校正（矫正后的点预测模型，原 TrackA1 已统一命名为 TrackA）"""
import os, numpy as np, pandas as pd

A, B = -4.729053505298141, 6.083224181420316
STAGE_BY_TIME = {0:"baseline",5:"acetic_acid",20:"edta_15",35:"edta_30",
                  65:"edta_60",95:"edta_90",105:"urea_10"}
_HERE = os.path.dirname(os.path.abspath(__file__))
_PRIOR = os.path.join(_HERE, "..", "..", "data", "mechanism_timeseries_quantiles.csv")
_P = None

def _load():
    q = pd.read_csv(_PRIOR)
    return (q[q.metric == "whole_resistance_ratio"]
            .drop_duplicates(["edta_concentration_mol_per_l", "global_time_s"])
            .sort_values(["edta_concentration_mol_per_l", "global_time_s"]))

def raw_prior(dose, t_min, stat="q50"):
    global _P
    if _P is None: _P = _load()
    g = _P[_P.edta_concentration_mol_per_l == dose]
    ts = g.global_time_s.to_numpy(float); vs = g[stat].to_numpy(float)
    t_s = t_min * 60.0
    return float(vs[-1]) if t_s > ts.max() else float(np.interp(t_s, ts, vs))

def predict(dose, t_min, a=A, b=B, anchor_baseline=True):
    """R_Z_corr = a + b * R_Z_raw；t=0 按物理定义锚定为 1.0。"""
    if anchor_baseline and t_min == 0: return 1.0
    return a + b * raw_prior(dose, t_min)

def info():
    return {"model_id":"TrackA_A1","formula":"R_Z = a + b*R_Z_raw",
             "a":A,"b":B,"baseline_anchor":1.0,
             "label":"EMPIRICAL correction, NOT a mechanistic improvement",
             "original_mechanism_unchanged":"sigma = 1e-4 + 1.5*phi^m ; R = sum dx/(sigma*A)"}
