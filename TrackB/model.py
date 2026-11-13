#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""TrackB — 校准后的 LOO 预测区间模型（只报 coverage/width/Winkler，不报 R2/RMSE）"""
import os, numpy as np, pandas as pd

TARGET_COVERAGE = 0.90
ALPHA_MIS = 0.10
_HERE = os.path.dirname(os.path.abspath(__file__))
_ALL = os.path.join(_HERE, "calibrated_intervals_allfreq_392.csv")
_LF = os.path.join(_HERE, "calibrated_intervals_lowfreq_196.csv")

def load_intervals(scope="allfreq"):
    """allfreq: 392 点 (4 频率)；lowfreq: 196 点 (100/120 Hz，交付版)。"""
    return pd.read_csv(_ALL if scope == "allfreq" else _LF)

def slice_metrics(df, doses=None, freqs=None):
    s = df.copy()
    if doses is not None: s = s[s.edta_concentration_mol_per_l.isin(doses)]
    if freqs is not None: s = s[s.frequency_hz.isin(freqs)]
    return dict(n_points=int(len(s)), n_rats=int(s.rat.nunique()),
                coverage=float(s.covered.mean()), mean_width=float(s.width.mean()),
                median_width=float(s.width.median()), max_width=float(s.width.max()),
                winkler=float(s["winkler_alpha_mis_0.10"].mean()))

def calibrated_from_raw(sim_q05, sim_q95, padding):
    """由机理 ensemble 分位数 + conformal padding 重建区间（与上游公式一致）。"""
    return np.clip(np.asarray(sim_q05, float) - padding, 0, 1), \
           np.clip(np.asarray(sim_q95, float) + padding, 0, 1)

def info():
    return {"model_id":"TrackB_calibrated", "target_coverage":0.90,
            "padding_formula":"p_(fold,f)=tolerance_f*q_f ; q_f=Quantile_0.90(inner cross-fit scores)",
            "diagnosis":"~97% of final width comes from conformal padding, NOT LHS parameter uncertainty",
            "caveats":["pointwise coverage != joint curve coverage",
                       "validation set 22/24 rats (WT5 0.10M, WT6 0.15M missing)",
                       "role x dose confounding for 0.05 M",
                       "no R2/RMSE reported"]}
