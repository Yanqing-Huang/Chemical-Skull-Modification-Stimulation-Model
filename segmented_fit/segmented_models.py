#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""按试剂处理分段的模型定义（本轮新建，允许每段改变公式）

三段（与 Excel 官方标签 / 冻结数据 stage 列一一对应）：
  S1 50% 乙酸处理段      t = 5            单点横截面
  S2 EDTA 处理段         t = 20/35/65/95  四时间点动力学
  S3 饱和尿素处理段      t = 105          单点，实测 R_Z 相对 t=95 回升

两条路线：
  A1seg  TrackA1 纯经验：每段从若干候选公式里按 LORO 择优，允许换公式。
  A3seg  TrackA3 机理+经验：共用 A2 机理骨架 delta_eff，按段引入
          乙酸瞬时表面酸蚀项 / EDTA 脱钙演化 / 尿素界面重构再封闭项，
          再用 eta, eps 受约束保形校准。

严格边界：本文件不修改已冻结并推送的 ThreeRoutes 包内任何模型公式或参数；
          冻结基线模型在本文件里以「逐字复刻」方式提供，仅用于同口径对照。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

F0, D0, T0 = 100.0, 0.10, 100.0
T_UREA = 105.0

SEGS = {
    "S1_acetic": dict(t=[5], label="50% 乙酸处理段", short="乙酸段", stage="acetic_acid"),
    "S2_EDTA":   dict(t=[20, 35, 65, 95], label="EDTA 处理段", short="EDTA 段", stage="edta_15/30/60/90"),
    "S3_urea":   dict(t=[105], label="饱和尿素（乙醇+尿素）处理段", short="尿素段", stage="urea_10"),
}
SEG_ORDER = ["S1_acetic", "S2_EDTA", "S3_urea"]

# ---------------------------------------------------------------- 通用派生列
def attach(d):
    """只依赖 dose / freq / t 三列，t 可被覆盖（用于构造 t=95 的反事实节点）。"""
    d = d.copy()
    d["lf"] = np.log(d["freq"].to_numpy(float) / F0)
    d["dr"] = d["dose"].to_numpy(float) / D0
    return d


def _cols(X):
    return (X["t"].to_numpy(float), X["lf"].to_numpy(float), X["dr"].to_numpy(float))


# =========================================================================
# 一、冻结基线（逐字复刻 run_three_routes.py 的 predict / fit，用于同口径对照）
# =========================================================================
FITCFG = {
    "A1.1": dict(p0=[0.10, 0.02, 0.01, -0.002, 0.8], lo=[0.0, 1e-5, -1, -1, 0.2], hi=[0.9, 1.0, 1, 1, 3.0]),
    "A1.2": dict(p0=[-4.7, 6.1, 0.0, 0.05], lo=[-50, -50, -1, -5], hi=[50, 50, 1, 5]),
    "A1.3": dict(p0=[3.3, 0.11, 0.003], lo=[0.1, -1.0, -0.05], hi=[30.0, 1.5, 0.05]),
    "A2":   dict(p0=[5.0, 0.3, 0.8, 0.2], lo=[0.01, -2, 0.05, 0.0], hi=[100, 3, 3, 1.0]),
    "A3":   dict(p0=[1.0, 0.0], lo=[0.0, -0.5], hi=[3.0, 0.5]),
}
BASE_K = {"A1.1": 5, "A1.2": 4, "A1.3": 3, "A2": 4, "A3": 6}

_W = None


def init_prior(path):
    global _W
    q = pd.read_csv(path)
    _W = (q[q.metric == "whole_resistance_ratio"]
          .drop_duplicates(["edta_concentration_mol_per_l", "global_time_s"])
          .sort_values(["edta_concentration_mol_per_l", "global_time_s"]))


def raw_prior(dose, t):
    g = _W[_W.edta_concentration_mol_per_l == dose]
    ts = g.global_time_s.to_numpy(float)
    vs = g.q50.to_numpy(float)
    t_s = t * 60.0
    return float(vs[-1]) if t_s > ts.max() else float(np.interp(t_s, ts, vs))


def base_attach(df):
    d = attach(df)
    d["raw"] = [raw_prior(r.dose, r.t) for r in d.itertuples()]
    d["delta"] = 1.0 - d["raw"].to_numpy(float)
    return d


def mech_delta(p4, X):
    a0, be, pe, s = p4
    t, lf, dr = _cols(X)
    return a0 * np.power(dr, be) * np.power(t / T0, pe) * np.exp(-s * lf)


def r_mech(p4, X):
    return 1.0 / (1.0 + mech_delta(p4, X))


def base_predict(name, p, X):
    p = np.asarray(p, float)
    t, lf, dr = _cols(X)
    if name == "A1.1":
        b = np.clip(p[1] + p[2] * X["dose"].to_numpy(float) + p[3] * lf, 1e-6, 5.0)
        return p[0] + (1 - p[0]) * np.exp(-np.power(b * t, p[4]))
    if name == "A1.2":
        return (p[0] + p[1] * X["raw"].to_numpy(float) + p[2] * t + p[3] * lf)
    if name == "A1.3":
        return (1.0 - p[0] * np.exp(-p[1] * lf) * X["delta"].to_numpy(float) - p[2] * t)
    if name == "A2":
        return r_mech(p[:4], X)
    if name == "A3":
        rm = r_mech(p[:4], X)
        return 1.0 - p[4] * (1.0 - rm) - p[5] * (t / T_UREA)
    raise ValueError(name)


def _fit_A2(X, y):
    c = FITCFG["A2"]
    r = least_squares(lambda p: r_mech(p, X) - y, np.asarray(c["p0"], float),
                      bounds=(np.asarray(c["lo"]), np.asarray(c["hi"])),
                      max_nfev=5000, ftol=1e-11, xtol=1e-11)
    return r.x


def base_fit(name, X, y, p_init=None):
    cfg = FITCFG[name]
    if name == "A3":
        mp = (np.asarray(p_init[:4], float) if p_init is not None and len(p_init) >= 4
              else _fit_A2(X, y))
        rm = r_mech(mp, X)
        base = 1.0 - rm
        tnorm = X["t"].to_numpy(float) / T_UREA
        r = least_squares(lambda q: (1.0 - q[0] * base - q[1] * tnorm) - y,
                          np.asarray(p_init[4:], float) if p_init is not None and len(p_init) == 6
                          else np.asarray(cfg["p0"], float),
                          bounds=(np.asarray(cfg["lo"]), np.asarray(cfg["hi"])),
                          max_nfev=4000, ftol=1e-11, xtol=1e-11)
        return np.concatenate([mp, r.x])
    p0 = np.asarray(p_init if p_init is not None else cfg["p0"], float)
    r = least_squares(lambda p: base_predict(name, p, X) - y, p0,
                      bounds=(np.asarray(cfg["lo"]), np.asarray(cfg["hi"])),
                      max_nfev=5000, ftol=1e-11, xtol=1e-11)
    return r.x


# =========================================================================
# 二、TrackA1 纯经验：每段候选公式（允许改变公式）
# =========================================================================
A1_VARIANTS = {
    "S1_acetic": {
        "s1_pow": dict(k=3, p0=[0.28, 0.10, 0.05], lo=[0.0, -3.0, -1.0], hi=[2.0, 3.0, 1.0],
                       label="经验幂律", formula="R = 1 - A*(d/D0)^b*(f/100)^(-s)"),
        "s1_lin": dict(k=4, p0=[0.72, 0.02, -0.05, 0.0], lo=[-2, -2, -2, -2], hi=[2, 2, 2, 2],
                       label="线性回归+交互", formula="R = c0 + c1*ln(f/100) + c2*(d/D0-1) + c3*交互"),
    },
    "S2_EDTA": {
        "s2_stretch": dict(k=5, p0=[0.10, 0.02, 0.01, -0.005, 0.7], lo=[0, 1e-5, -1, -1, 0.2], hi=[0.9, 1, 1, 1, 3],
                           label="拉伸指数", formula="R = C+(1-C)exp(-(b*t)^n), b=b0+b1*(d/D0)+b2*ln(f/100)"),
        "s2_hyper": dict(k=4, p0=[5.0, 0.3, 0.8, 0.2], lo=[0.01, -2, 0.05, 0], hi=[100, 3, 3, 1],
                         label="双曲/幂律导纳", formula="R = 1/(1+a0*(d/D0)^b*(t/100)^p*(f/100)^(-s))"),
        "s2_exp": dict(k=6, p0=[0.8, 0.1, -0.05, 0.02, 0.01, 0.0], lo=[0, -2, -2, 1e-5, -1, -0.5], hi=[2, 2, 2, 1, 1, 0.5],
                       label="单指数饱和+漂移", formula="R = 1 - A*(1-exp(-k*t)) - c*(t/105)"),
    },
    "S3_urea": {
        "s3_rebound": dict(k=3, p0=[0.25, 0.0, 0.0], lo=[-1, -1, -2], hi=[0.95, 1, 2],
                           label="分段衔接回升式", formula="R = R95 + lambda*(1-R95), lambda=clip(l0+l1*lnf+l2*(d/D0-1))"),
        "s3_cross": dict(k=4, p0=[0.27, 0.02, -0.05, 0.0], lo=[-2, -2, -2, -2], hi=[2, 2, 2, 2],
                         label="独立横截面回归", formula="R = c0 + c1*ln(f/100) + c2*(d/D0-1) + c3*交互"),
    },
}
A1_DEFAULT = {"S1_acetic": "s1_pow", "S2_EDTA": "s2_stretch", "S3_urea": "s3_rebound"}


def a1_predict(seg, var, p, X, ctx=None):
    """ctx: {'s2_var':..., 's2_params':...}，仅 S3/s3_rebound 需要（构造 t=95 反事实节点）。"""
    ctx = ctx or {}
    t, lf, dr = _cols(X)
    p = np.asarray(p, float)
    if seg == "S1_acetic":
        if var == "s1_pow":
            A, b, s = p
            return 1.0 - A * np.power(dr, b) * np.exp(-s * lf)
        if var == "s1_lin":
            c0, c1, c2, c3 = p
            return c0 + c1 * lf + c2 * (dr - 1) + c3 * lf * (dr - 1)
    if seg == "S2_EDTA":
        if var == "s2_stretch":
            C, b0, b1, b2, n = p
            b = np.clip(b0 + b1 * dr + b2 * lf, 1e-6, 5.0)
            return C + (1 - C) * np.exp(-np.power(b * t, n))
        if var == "s2_hyper":
            a0, be, pe, s = p
            return 1.0 / (1.0 + a0 * np.power(dr, be) * np.power(t / T0, pe) * np.exp(-s * lf))
        if var == "s2_exp":
            A0, A1, A2, k0, k1, c = p
            A = np.clip(A0 + A1 * dr + A2 * lf, 0.0, 2.0)
            k = np.clip(k0 + k1 * dr, 1e-5, 1.0)
            return 1.0 - A * (1.0 - np.exp(-k * t)) - c * (t / T_UREA)
    if seg == "S3_urea":
        if var == "s3_rebound":
            X95 = X.copy()
            X95["t"] = 95.0
            r95 = a1_predict("S2_EDTA", ctx["s2_var"], ctx["s2_params"], X95)
            l0, l1, l2 = p
            lam = np.clip(l0 + l1 * lf + l2 * (dr - 1), 0.0, 0.95)
            return r95 + lam * (1.0 - r95)
        if var == "s3_cross":
            c0, c1, c2, c3 = p
            return c0 + c1 * lf + c2 * (dr - 1) + c3 * lf * (dr - 1)
    raise ValueError((seg, var))


def a1_fit(seg, var, X, y, ctx=None):
    cfg = A1_VARIANTS[seg][var]
    y = np.asarray(y, float)
    p0 = np.asarray(cfg["p0"], float)
    r = least_squares(lambda p: a1_predict(seg, var, p, X, ctx) - y, p0,
                      bounds=(np.asarray(cfg["lo"], float), np.asarray(cfg["hi"], float)),
                      max_nfev=6000, ftol=1e-11, xtol=1e-11)
    return r.x


# =========================================================================
# 三、TrackA3 机理 + 经验：三段共用 A2 机理骨架，按段补机理项
# =========================================================================
# 机理骨架（与 TrackA2 逐字一致）：delta_eff = a0*(d/D0)^beta*(t/100)^p*(f/100)^(-s)
# S1 新增机理项：乙酸瞬时表面酸蚀 -> 与时间无关的额外有效改性量 delta_acid
# S3 新增机理项：乙醇+尿素导致蛋白/孔道再封闭 -> delta_post = delta_95*(1-rho)
#                 rho = rho0*(d/D0)^gamma*(f/100)^(-nu)（剂量/频率依赖的再封闭率）
A3_CFG = {
    "mech": dict(k=4, p0=[5.0, 0.3, 0.8, 0.2], lo=[0.01, -2, 0.05, 0.0], hi=[100, 3, 3, 1.0],
                 label="EDTA 脱钙机理骨架", formula="delta_eff = a0*(d/D0)^beta*(t/100)^p*(f/100)^(-s)"),
    "S1_acetic": dict(k=2, p0=[1.2, 1.0], lo=[0.01, 0.0], hi=[100, 5],
                      label="乙酸瞬时表面酸蚀跃变", formula="delta_acid = alpha*dr^beta*exp(-s*lnf); R = 1-eta_a*(1-R_mech_acid)"),
    "S2_EDTA": dict(k=2, p0=[1.0, 0.0], lo=[0.0, -0.5], hi=[3.0, 0.5],
                    label="EDTA 段保形校准", formula="R = 1 - eta_e*(1-R_mech) - eps_e*(t/105)"),
    "S3_urea": dict(k=5, p0=[0.25, 0.0, 0.0, 1.0, 0.0], lo=[0.0, -2, -1, 0.0, -0.5], hi=[0.95, 2, 1, 3.0, 0.5],
                    label="尿素界面重构再封闭", formula="delta_post = delta_95*(1-rho); R = 1 - eta_u*(1-R_post) - eps_u"),
}
A3_TOTAL_K = (A3_CFG["mech"]["k"] + A3_CFG["S1_acetic"]["k"]
              + A3_CFG["S2_EDTA"]["k"] + A3_CFG["S3_urea"]["k"])


def _delta_eff(p4, X, t_override=None):
    a0, be, pe, s = p4
    t, lf, dr = _cols(X)
    if t_override is not None:
        t = np.full(len(X), float(t_override))
    return a0 * np.power(dr, be) * np.power(t / T0, pe) * np.exp(-s * lf)


def a3_predict(seg, p, X, p4):
    """p4 为共用机理 4 参数；p 为该段自有参数。"""
    t, lf, dr = _cols(X)
    p = np.asarray(p, float)
    if seg == "S1_acetic":
        alpha, eta_a = p
        d_acid = alpha * np.power(dr, p4[1]) * np.exp(-p4[3] * lf)
        rm = 1.0 / (1.0 + d_acid)
        return 1.0 - eta_a * (1.0 - rm)
    if seg == "S2_EDTA":
        eta_e, eps_e = p
        rm = 1.0 / (1.0 + _delta_eff(p4, X))
        return 1.0 - eta_e * (1.0 - rm) - eps_e * (t / T_UREA)
    if seg == "S3_urea":
        rho0, ga, nu, eta_u, eps_u = p
        d95 = _delta_eff(p4, X, t_override=95.0)
        rho = np.clip(rho0 * np.power(dr, ga) * np.exp(-nu * lf), 0.0, 0.95)
        rm_post = 1.0 / (1.0 + d95 * (1.0 - rho))
        return 1.0 - eta_u * (1.0 - rm_post) - eps_u
    raise ValueError(seg)


def a3_fit_mech(X, y):
    c = A3_CFG["mech"]
    r = least_squares(lambda q: 1.0 / (1.0 + _delta_eff(q, X)) - np.asarray(y, float),
                      np.asarray(c["p0"], float),
                      bounds=(np.asarray(c["lo"], float), np.asarray(c["hi"], float)),
                      max_nfev=6000, ftol=1e-11, xtol=1e-11)
    return r.x


def a3_fit_segment(seg, X, y, p4):
    c = A3_CFG[seg]
    y = np.asarray(y, float)
    r = least_squares(lambda q: a3_predict(seg, q, X, p4) - y,
                      np.asarray(c["p0"], float),
                      bounds=(np.asarray(c["lo"], float), np.asarray(c["hi"], float)),
                      max_nfev=6000, ftol=1e-11, xtol=1e-11)
    return r.x


# =========================================================================
# 四、增量分段调整（V2）：保留全局骨架的跨段形状约束，只做最少分段修正
# =========================================================================
# 动机：V1「三段从零独立标定」虽修复尿素回升，但把 6 个时间点的联合约束拆散，
#       且参数从 6 增至 11/13，LORO 折内方差变大，整体 R² 反而下降。
# V2 改为：基底 = 冻结的全局单调模型（用全部 6 时间点拟合，形状约束不丢），
#           S1/S2 各加 1 个截距修正（消除分段系统偏差，几乎不增方差），
#           S3 加回升项（唯一被证实的定性缺陷）。
#   ADJ_A1 基底 A1.1 纯经验拉伸指数；S3 用经验线性回升系数 lambda(d,f)。
#   ADJ_A3 基底 A3 机理骨架；S3 用机理再封闭率 rho(d,f)=rho0*dr^gamma*exp(-nu*lnf)。
ADJ_BASE = {"ADJ_A1": "A1.1", "ADJ_A3": "A3"}
ADJ_K = {"ADJ_A1": BASE_K["A1.1"] + 1 + 1 + 3,
         "ADJ_A3": BASE_K["A3"] + 1 + 1 + 3}
ADJ_S3_CFG = dict(p0=[0.25, 0.0, 0.0], lo=[-1.0, -2.0, -2.0], hi=[0.95, 2.0, 2.0])


def base_at95(name, p, X):
    """基底在 t=95 的反事实预测（同一 dose/freq），用于定义可回升空间 1-R(95)。"""
    X95 = X.copy()
    X95["t"] = 95.0
    return base_predict(name, p, X95)


def adj_fit(scheme, X, y, p_base, seg_times=None):
    """在基底预测之上，按段拟合最少修正参数。返回 dict(base=..., seg=系数向量)。"""
    name = ADJ_BASE[scheme]
    y = np.asarray(y, float)
    rb = np.asarray(base_predict(name, p_base, X), float)
    t, lf, dr = _cols(X)
    tarr = X["t"].to_numpy(float)
    P = {"base": np.asarray(p_base, float)}
    for seg in SEG_ORDER:
        m = np.isin(tarr, SEGS[seg]["t"])
        res = y[m] - rb[m]
        if seg in ("S1_acetic", "S2_EDTA"):
            P[seg] = np.array([float(np.mean(res))])
        else:
            # S3 从「EDTA 末态平台」出发，而不是从基底在 t=105 的单调外推出发：
            # 尿素是独立处理阶段，其 R_Z 应由 t=95 的改性状态 + 回升项决定。
            rb95 = np.asarray(base_at95(name, p_base, X[m]), float)
            gap = np.clip(1.0 - rb95, 1e-6, None)
            if scheme == "ADJ_A1":
                # 纯经验：y105 - R95 = lambda*gap，lambda = l0+l1*lnf+l2*(dr-1)
                D = np.c_[gap, gap * lf[m], gap * (dr[m] - 1.0)]
                coef, *_ = np.linalg.lstsq(D, y[m] - rb95, rcond=None)
                P[seg] = np.asarray(coef, float)
            else:
                # 机理：界面重构/再封闭使有效改性量回落
                # delta_post = delta_95*(1-rho); R = 1/(1+delta_post)
                d95 = _delta_eff(np.asarray(p_base[:4], float), X[m], t_override=95.0)
                c = ADJ_S3_CFG

                def resid(q):
                    rho = np.clip(q[0] * np.power(dr[m], q[1]) * np.exp(-q[2] * lf[m]), 0.0, 0.95)
                    return 1.0 / (1.0 + d95 * (1.0 - rho)) - y[m]

                r = least_squares(resid, np.asarray(c["p0"], float),
                                  bounds=(np.asarray(c["lo"], float), np.asarray(c["hi"], float)),
                                  max_nfev=6000, ftol=1e-11, xtol=1e-11)
                P[seg] = r.x
    return P


def adj_predict(scheme, seg, P, X):
    name = ADJ_BASE[scheme]
    rb = np.asarray(base_predict(name, P["base"], X), float)
    t, lf, dr = _cols(X)
    if seg in ("S1_acetic", "S2_EDTA"):
        return rb + float(P[seg][0])
    rb95 = np.asarray(base_at95(name, P["base"], X), float)
    gap = np.clip(1.0 - rb95, 1e-6, None)
    if scheme == "ADJ_A1":
        l0, l1, l2 = P[seg]
        lam = np.clip(l0 + l1 * lf + l2 * (dr - 1.0), 0.0, 0.95)
        return rb95 + lam * gap
    d95 = _delta_eff(np.asarray(P["base"][:4], float), X, t_override=95.0)
    r0, ga, nu = P[seg]
    rho = np.clip(r0 * np.power(dr, ga) * np.exp(-nu * lf), 0.0, 0.95)
    return 1.0 / (1.0 + d95 * (1.0 - rho))
