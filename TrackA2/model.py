#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""TrackA2 — 纯机理模型（本轮新建：调机理、把频率等全算进去、一套参数覆盖所有条件）

机理链条：
  Archie 局部电导率   sigma = sigma_floor + c*phi^m
  脱钙孔隙率演化       phi(t,d) ~ (d/D0)^beta * (t/T0)^p
  界面极化 / CPE 旁路  G_pol(f) ~ (f/F0)^(-s)  与离子通路并联
  总导纳 Y(f,t,d) = G_ion(t,d) + G_pol(f)；R_Z = Y(0)/Y(t) = 1/(1+delta_eff)

  delta_eff(t,d,f) = a0*(d/D0)^beta*(t/T0)^p*(f/F0)^(-s)
  R_Z = 1/(1+delta_eff)

与旧的准静态纯电阻网络不同：频率 f 内生进机理方程，可解释实测「同一时间高频 R_Z 偏高」。
4 个参数全是物理量；单一组参数覆盖 4 剂量 x 4 频率 x 6 时间 = 96 条件节点；
无任何对观测的经验回归系数；t=0 天然等于 1。
仅依赖 numpy，不读取任何外部数据文件（完全自包含）。
"""
import numpy as np

P = np.array([4.66292129, 0.3138528, 0.79632681, 0.23181099], float)     # [a0, beta, p, s]
F0, D0, T0 = 100.0, 0.10, 100.0  # Hz ; mol/L ; min


def delta_eff(dose, t_min, freq_hz):
    d = dose / D0
    f = freq_hz / F0
    t = t_min / T0
    return P[0] * np.power(d, P[1]) * np.power(t, P[2]) * np.power(f, -P[3])


def predict(dose, t_min, freq_hz):
    """dose: mol/L ; t_min: 全局时间 min ; freq_hz: Hz"""
    return float(1.0 / (1.0 + delta_eff(dose, t_min, freq_hz)))


def info():
    return {"model_id": "TrackA2", "route": "TrackA2 纯机理模型",
             "label": "并联导纳 + 脱钙动力学（频率内生）",
             "formula": "R_Z = 1/(1 + a0*(d/0.10)^beta*(t/100)^p*(f/100)^(-s))",
             "params": P.tolist(), "k": 4, "baseline": "automatic (=1 by construction)",
             "mechanism_input": "全部来自机理方程，零经验回归层",
             "self_contained": True, "reads_no_external_file": True,
             "covers": "4 doses x 4 freqs x 6 times = 96 condition nodes with ONE parameter set",
             "label_honest": "physical parameters are CALIBRATED to data, not derived from first principles; "
                             "s is an effective dispersion exponent, not an identified CPE exponent; "
                             "monotone in t so it cannot express the urea t=105 rebound"}
