#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""TrackA3 — 机理驱动经验模型（本轮新建：机理给骨架 + 经验做受约束保形校准）

R_mech(f,t,d) = TrackA2 的纯机理预测（前 4 个物理参数，stagewise 固定）
R_Z = 1 - eta*(1 - R_mech) - eps*(t/105)

机理决定函数形状（基线=1、单调、频率分层、剂量/时间幂律结构）；
经验层只有 2 个受约束系数：eta 缩放机理幅度，eps 补机理未覆盖的时间漂移。
eta=1, eps=0 时严格退化为 A2（嵌套）；t=0 天然等于 1。
仅依赖 numpy，不读取任何外部数据文件（完全自包含）。
"""
import numpy as np

P = np.array([4.66292129, 0.3138528, 0.79632681, 0.23181099, 0.96261776, 0.04100661], float)     # [a0, beta, p, s, eta, eps]
F0, D0, T0 = 100.0, 0.10, 100.0


def r_mech(dose, t_min, freq_hz):
    d = dose / D0
    f = freq_hz / F0
    t = t_min / T0
    delta = P[0] * np.power(d, P[1]) * np.power(t, P[2]) * np.power(f, -P[3])
    return 1.0 / (1.0 + delta)


def predict(dose, t_min, freq_hz):
    rm = r_mech(dose, t_min, freq_hz)
    return float(1.0 - P[4] * (1.0 - rm) - P[5] * (t_min / 105.0))


def info():
    return {"model_id": "TrackA3", "route": "TrackA3 机理驱动经验模型",
             "label": "机理骨架 + 受约束保形校准",
             "formula": "R_Z = 1 - eta*(1-R_mech) - eps*(t/105); R_mech = A2 prediction",
             "params": P.tolist(), "k": 6, "k_empirical": 2, "k_mechanistic": 4,
             "baseline": "automatic (=1 by construction)",
             "mechanism_input": "机理骨架 100%（前 4 参数）+ 2 个经验校准系数",
             "self_contained": True, "reads_no_external_file": True,
             "nesting": "eta=1, eps=0 -> exactly TrackA2",
             "label_honest": "mechanism-driven structure + empirical calibration; "
                             "eta/eps are fitted correction coefficients, not physical constants"}
