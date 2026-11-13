#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""TrackA1 / A1.1 — 纯经验：拉伸指数衰减（完全脱离机理链路）

R_Z = C + (1-C)*exp(-(b*t)^n),  b = b0 + b1*dose + b2*ln(f/100)
t 为全局时间(min)。t=0 天然等于 1（exp(0)=1）。
参数由 C1 全部条件(96 节点)拟合；参数无任何物理量纲。
"""
import numpy as np

P = np.array([0.07552382, 0.03197206, 0.06196823, -0.00588744, 0.59674663], float)     # [C, b0, b1, b2, n]
F0 = 100.0


def predict(dose, t_min, freq_hz, anchor_baseline=True):
    """dose: mol/L ; t_min: 全局时间 min ; freq_hz: Hz"""
    if anchor_baseline and t_min == 0:
        return 1.0
    lf = np.log(freq_hz / F0)
    b = np.clip(P[1] + P[2] * dose + P[3] * lf, 1e-6, 5.0)
    return float(P[0] + (1 - P[0]) * np.exp(-np.power(b * t_min, P[4])))


def info():
    return {"model_id": "TrackA1.1", "route": "TrackA1 纯经验模型",
             "label": "拉伸指数衰减", "formula": "R_Z = C+(1-C)exp(-(bt)^n)",
             "params": P.tolist(), "k": 5, "baseline": "automatic (=1)",
             "mechanism_input": "无（完全脱离机理链路）",
             "label_honest": "pure empirical curve fit; no physical parameters"}
