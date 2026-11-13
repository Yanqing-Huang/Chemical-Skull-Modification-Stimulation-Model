#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
三条 TrackA 路线 x 四条件 拟合（本轮重构的计算核心，只算数、不出图/文档）

数据来源：上一轮已完成三层核验的冻结数据（修订版《体外EDTA数据记录汇总.xlsx》
SHA256 4c165298...，新增单位行 / 数值+1行偏移 0 差异 / 2016 行与冻结副本逐格一致）。
源文件未被修改，24 鼠全保留。

===== 本轮模型体系（用户 2026-11-13 重构）=====

TrackA1 纯经验模型（数据驱动，机理方程最多当查表特征）
  A1.1 拉伸指数   R_Z = C+(1-C)*exp(-(b*t)^n),  b=b0+b1*dose+b2*ln(f/100)   [5 参数, 完全脱离机理]
  A1.2 A+         R_Z = a + b*raw + c*t + d*ln(f/100)                        [4 参数, raw 作特征]
  A1.3 H1 混合    R_Z = 1 - kappa*(f/100)^(-s)*(1-raw) - eps*t                [3 参数, delta 作骨架]

TrackA2 纯机理模型（新：调机理、把频率等全算进去、一套物理参数覆盖所有条件）
  机理链条：Archie 电导 sigma=sigma_floor+c*phi^m ; 孔隙率脱钙演化 phi ~ d^beta*t^p ;
            并联导纳 Y(f,t,d)=G_ion(t,d)+G_pol(f), G_pol ~ f^(-s)（界面极化/CPE 旁路）;
            R_Z = Y(0)/Y(t) = 1/(1+delta_eff)
  delta_eff(t,d,f) = a0 * (d/0.10)^beta * (t/100)^p * (f/100)^(-s)
  R_Z = 1/(1+delta_eff)                                                      [4 参数, 全物理量]
  * 频率 f 内生进机理方程（原机理频率无关）；单一组参数覆盖 4 剂量 x 4 频率 x 6 时间 = 96 节点；
    t=0 天然 R_Z=1；delta 单调故无法表达尿素 t=105 回升（诚实保留）。

TrackA3 机理驱动经验模型（新：机理给骨架 + 经验做受约束保形校准）
  R_mech = A2 的机理预测（stagewise 固定物理参数）
  R_Z = 1 - eta*(1-R_mech) - eps*(t/105)                                     [2 经验系数 + 4 机理参数]
  * eta=1, eps=0 时退化为 A2（嵌套）；t=0 天然 =1；eta 缩放机理幅度、eps 补机理未覆盖的时间漂移。

四条件（评价集不同、LORO 折数不同，R2 不可横向解读为模型优劣）：
  C1 全部条件      4 剂量 x 4 频率 x 6 改性时间 = 96 条件节点 / 24 折 LORO
  C2 低频 100/120  4 剂量 x 2 频率 x 6 时间    = 48 / 24 折
  C3 高浓度 15/20  2 剂量 x 4 频率 x 6 时间    = 48 / 12 折
  C4 低频+高浓度   2 剂量 x 2 频率 x 6 时间    = 24 / 12 折

TrackB：既有 392 点 LOO 校准区间按条件切片，只报 coverage/width/Winkler，不报 R2/RMSE。
"""
from __future__ import annotations

import csv
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
FROZEN = ROOT.parent / "data" / "frozen_measured_24rat.csv"
PRIOR = ROOT.parent / "data" / "mechanism_timeseries_quantiles.csv"
TRACKB_ALL = ROOT.parent / "TrackB" / "calibrated_intervals_allfreq_392.csv"

T_MOD = [5, 20, 35, 65, 95, 105]
T_ALL = [0, 5, 20, 35, 65, 95, 105]
F0 = 100.0      # 参考频率 Hz
D0 = 0.10       # 参考剂量 mol/L
T0 = 100.0      # 参考时间 min

CONDITIONS = {
    "C1_all": dict(label="C1 全部条件（4 剂量 x 4 频率）",
                   doses=[0.05, 0.10, 0.15, 0.20], freqs=[100, 120, 1000, 10000]),
    "C2_lowfreq": dict(label="C2 低频（100 / 120 Hz）",
                       doses=[0.05, 0.10, 0.15, 0.20], freqs=[100, 120]),
    "C3_highdose": dict(label="C3 高浓度（15% / 20%）",
                        doses=[0.15, 0.20], freqs=[100, 120, 1000, 10000]),
    "C4_lf_hd": dict(label="C4 低频 + 高浓度",
                     doses=[0.15, 0.20], freqs=[100, 120]),
}

# 三条路线 / 5 个模型
ROUTES = {
    "A1": "TrackA1 纯经验模型",
    "A2": "TrackA2 纯机理模型",
    "A3": "TrackA3 机理驱动经验模型",
}
MODELS = {
    "A1.1": dict(route="A1", family="emp", label="A1.1 拉伸指数", k=5,
                 formula="R_Z = C+(1-C)exp(-(bt)^n), b=b0+b1*d+b2*ln(f/100)",
                 mechanism_input="无（完全脱离机理链路）"),
    "A1.2": dict(route="A1", family="emp", label="A1.2 A+ 协变量", k=4,
                 formula="R_Z = a + b*R_Z_raw + c*t + d*ln(f/100)",
                 mechanism_input="机理先验 R_Z_raw 作一个查表特征"),
    "A1.3": dict(route="A1", family="emp", label="A1.3 H1 混合", k=3,
                 formula="R_Z = 1 - kappa*(f/100)^(-s)*(1-R_Z_raw) - eps*t",
                 mechanism_input="机理改性量 delta=1-R_Z_raw 作形状骨架"),
    "A2": dict(route="A2", family="mech", label="A2 并联导纳纯机理（频率内生）", k=4,
               formula="R_Z = 1/(1+a0*(d/0.10)^b*(t/100)^p*(f/100)^(-s))",
               mechanism_input="全部来自机理方程（Archie + 脱钙动力学 + CPE 并联导纳）"),
    "A3": dict(route="A3", family="driven", label="A3 机理骨架 + 保形校准", k=6,
               formula="R_Z = 1 - eta*(1-R_mech) - eps*(t/105); R_mech=A2 机理预测",
               mechanism_input="机理骨架固定 + 2 个受约束经验校准系数"),
}


# ==========================================================================
# 数据
# ==========================================================================
_W = None


def load_data():
    global _W
    s = pd.read_csv(FROZEN)
    rl = (s.groupby(["rat", "edta_concentration_mol_per_l", "frequency_hz", "global_time_min"])["rz"]
          .mean().reset_index(name="y"))
    rl = rl[rl.global_time_min.isin(T_MOD)].copy()
    rl.columns = ["rat", "dose", "freq", "t", "y"]
    cond = (rl.groupby(["dose", "freq", "t"])
            .agg(y=("y", "mean"), n=("rat", "nunique")).reset_index())
    cond["sem"] = (rl.groupby(["dose", "freq", "t"])["y"].sem()
                   .reindex(pd.MultiIndex.from_frame(cond[["dose", "freq", "t"]])).to_numpy())
    q = pd.read_csv(PRIOR)
    _W = (q[q.metric == "whole_resistance_ratio"]
          .drop_duplicates(["edta_concentration_mol_per_l", "global_time_s"])
          .sort_values(["edta_concentration_mol_per_l", "global_time_s"]))
    return rl, cond


def raw_prior(dose, t):
    g = _W[_W.edta_concentration_mol_per_l == dose]
    ts = g.global_time_s.to_numpy(float)
    vs = g.q50.to_numpy(float)
    t_s = t * 60.0
    return float(vs[-1]) if t_s > ts.max() else float(np.interp(t_s, ts, vs))


def attach(df):
    d = df.copy()
    d["raw"] = [raw_prior(r.dose, r.t) for r in d.itertuples()]
    d["lf"] = np.log(d["freq"].to_numpy(float) / F0)
    d["delta"] = 1.0 - d["raw"].to_numpy(float)
    return d


# ==========================================================================
# 三类预测函数
# ==========================================================================
def mech_delta(p4, X):
    """A2 机理有效改性量 delta_eff = a0*(d/D0)^beta*(t/T0)^p*(f/F0)^(-s)"""
    a0, beta, pexp, s = p4
    d = X["dose"].to_numpy(float) / D0
    f = X["freq"].to_numpy(float) / F0
    t = X["t"].to_numpy(float) / T0
    return a0 * np.power(d, beta) * np.power(t, pexp) * np.power(f, -s)


def r_mech(p4, X):
    return 1.0 / (1.0 + mech_delta(p4, X))


def _b(p, X):
    return np.clip(p[1] + p[2] * X["dose"].to_numpy(float)
                   + p[3] * X["lf"].to_numpy(float), 1e-6, 5.0)


def predict(name, p, X):
    p = np.asarray(p, float)
    if name == "A1.1":
        return p[0] + (1 - p[0]) * np.exp(-np.power(_b(p, X) * X["t"].to_numpy(float), p[4]))
    if name == "A1.2":
        return (p[0] + p[1] * X["raw"].to_numpy(float)
                + p[2] * X["t"].to_numpy(float) + p[3] * X["lf"].to_numpy(float))
    if name == "A1.3":
        lam = np.exp(-p[1] * X["lf"].to_numpy(float))       # (f/100)^(-s)
        return (1.0 - p[0] * lam * X["delta"].to_numpy(float)
                - p[2] * X["t"].to_numpy(float))
    if name == "A2":
        return r_mech(p[:4], X)
    if name == "A3":
        rm = r_mech(p[:4], X)
        return 1.0 - p[4] * (1.0 - rm) - p[5] * (X["t"].to_numpy(float) / 105.0)
    raise ValueError(name)


FITCFG = {
    "A1.1": dict(p0=[0.10, 0.02, 0.01, -0.002, 0.8], lo=[0.0, 1e-5, -1, -1, 0.2], hi=[0.9, 1.0, 1, 1, 3.0]),
    "A1.2": dict(p0=[-4.7, 6.1, 0.0, 0.05], lo=[-50, -50, -1, -5], hi=[50, 50, 1, 5]),
    "A1.3": dict(p0=[3.3, 0.11, 0.003], lo=[0.1, -1.0, -0.05], hi=[30.0, 1.5, 0.05]),
    "A2":   dict(p0=[5.0, 0.3, 0.8, 0.2], lo=[0.01, -2, 0.05, 0.0], hi=[100, 3, 3, 1.0]),
    "A3":   dict(p0=[1.0, 0.0], lo=[0.0, -0.5], hi=[3.0, 0.5]),
}


def fit(name, X, y, p_init=None):
    """返回该模型的参数向量。A3 为 stagewise：先固定 A2 机理骨架，再拟合 2 个经验校准系数。"""
    cfg = FITCFG[name]
    if name == "A3":
        # 机理骨架：用 p_init 里已有的 A2 参数，或在训练集上重新拟合 A2
        mp = (np.asarray(p_init[:4], float) if p_init is not None and len(p_init) >= 4
              else _fit_A2(X, y))
        rm = r_mech(mp, X)
        base = 1.0 - rm
        tnorm = X["t"].to_numpy(float) / 105.0
        r = least_squares(lambda q: (1.0 - q[0] * base - q[1] * tnorm) - y,
                          np.asarray(p_init[4:], float) if p_init is not None and len(p_init) == 6
                          else np.asarray(cfg["p0"], float),
                          bounds=(np.asarray(cfg["lo"]), np.asarray(cfg["hi"])),
                          max_nfev=4000, ftol=1e-11, xtol=1e-11)
        return np.concatenate([mp, r.x])
    p0 = np.asarray(p_init if p_init is not None else cfg["p0"], float)
    r = least_squares(lambda p: predict(name, p, X) - y, p0,
                      bounds=(np.asarray(cfg["lo"]), np.asarray(cfg["hi"])),
                      max_nfev=5000, ftol=1e-11, xtol=1e-11)
    return r.x


def _fit_A2(X, y):
    c = FITCFG["A2"]
    r = least_squares(lambda p: r_mech(p, X) - y, np.asarray(c["p0"], float),
                      bounds=(np.asarray(c["lo"]), np.asarray(c["hi"])),
                      max_nfev=5000, ftol=1e-11, xtol=1e-11)
    return r.x


# ==========================================================================
# 指标
# ==========================================================================
def metrics(y, yhat):
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    sse = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    return dict(n=int(len(y)), r2=float(1 - sse / sst) if sst > 0 else float("nan"),
                rmse=float(np.sqrt(np.mean((y - yhat) ** 2))),
                mae=float(np.mean(np.abs(y - yhat))),
                bias=float(np.mean(y - yhat)), sse=sse)


def aic(n, sse, k):
    return float(n * np.log(max(sse, 1e-12) / n) + 2 * (k + 1))


def evaluate(name, cond_all, rl_all, spec):
    c = attach(cond_all[(cond_all.dose.isin(spec["doses"]))
                        & (cond_all.freq.isin(spec["freqs"]))].copy()).reset_index(drop=True)
    r = attach(rl_all[(rl_all.dose.isin(spec["doses"]))
                      & (rl_all.freq.isin(spec["freqs"]))].copy()).reset_index(drop=True)
    y_c = c.y.to_numpy(float)
    p = fit(name, c, y_c)
    m_full = metrics(y_c, predict(name, p, c))
    m_full["aic"] = aic(len(y_c), m_full["sse"], MODELS[name]["k"])

    # LOCO：留一条件节点（群体均值层）
    yh = np.empty(len(c))
    for i in range(len(c)):
        mask = np.ones(len(c), bool)
        mask[i] = False
        pp = fit(name, c[mask].reset_index(drop=True), y_c[mask], p_init=p)
        yh[i] = predict(name, pp, c.iloc[[i]].reset_index(drop=True))[0]
    m_loco = metrics(y_c, yh)

    # LORO：留一大鼠（个体级独立验证，验收口径）
    yr = r.y.to_numpy(float)
    rat = r.rat.to_numpy()
    yhat = np.empty(len(r))
    for one in np.unique(rat):
        te = rat == one
        tr = ~te
        pp = fit(name, r[tr].reset_index(drop=True), yr[tr], p_init=p)
        yhat[te] = predict(name, pp, r[te].reset_index(drop=True))
    m_loro = metrics(yr, yhat)
    per = [metrics(yr[rat == one], yhat[rat == one])["r2"] for one in np.unique(rat)]
    m_loro["per_rat_mean_r2"] = float(np.nanmean(per))
    m_loro["n_folds"] = int(len(np.unique(rat)))

    # 基线自洽性：t=0 未经人为锚定的外推值
    base = attach(pd.DataFrame([dict(dose=spec["doses"][0], freq=float(spec["freqs"][0]),
                                     t=0.0, y=1.0)]))
    baseline = float(predict(name, p, base)[0])
    return dict(name=name, label=MODELS[name]["label"], route=MODELS[name]["route"],
                family=MODELS[name]["family"], k=MODELS[name]["k"],
                formula=MODELS[name]["formula"],
                mechanism_input=MODELS[name]["mechanism_input"],
                params=[float(x) for x in p],
                baseline_pred=baseline,
                n_cond_nodes=int(len(c)), n_rat_points=int(len(r)),
                n_rats=int(r.rat.nunique()),
                full=m_full, loco=m_loco, loro=m_loro)


# ==========================================================================
# TrackB 四条件切片
# ==========================================================================
def trackb_slice(spec):
    d = pd.read_csv(TRACKB_ALL)
    s = d[(d.edta_concentration_mol_per_l.isin(spec["doses"]))
          & (d.frequency_hz.isin(spec["freqs"]))].copy()
    wcol = "winkler_alpha_mis_0.10"
    return dict(n_points=int(len(s)), n_rats=int(s.rat.nunique()),
                coverage=float(s.covered.mean()),
                mean_width=float(s.width.mean()), median_width=float(s.width.median()),
                max_width=float(s.width.max()), winkler=float(s[wcol].mean()),
                note="既有 LOO 校准区间按条件切片；未重新拟合、未重新校准")


# ==========================================================================
# 主流程
# ==========================================================================
def main():
    rl, cond = load_data()
    results = {}
    for cid, spec in CONDITIONS.items():
        results[cid] = dict(spec=spec,
                            models={n: evaluate(n, cond, rl, spec) for n in MODELS},
                            trackB=trackb_slice(spec))
        r = results[cid]["models"]
        print(f"--- {cid} ---")
        for n in MODELS:
            m = r[n]
            print(f'  {n:5s} full={m["full"]["r2"]:.3f} LOCO={m["loco"]["r2"]:.3f} '
                  f'LORO={m["loro"]["r2"]:.3f} (folds={m["loro"]["n_folds"]}) '
                  f'base={m["baseline_pred"]:.3f}')
        tb = results[cid]["trackB"]
        print(f'  TrackB n={tb["n_points"]} cov={tb["coverage"]:.3f} '
              f'width={tb["mean_width"]:.3f} winkler={tb["winkler"]:.3f}')

    # A2「所有条件都考虑进去」：C1 拟合的一套物理参数，直接迁移到各切片（零重拟合）
    c1p = np.asarray(results["C1_all"]["models"]["A2"]["params"], float)
    transfer = {}
    for cid, spec in CONDITIONS.items():
        sub = cond[(cond.dose.isin(spec["doses"])) & (cond.freq.isin(spec["freqs"]))].copy()
        sub = attach(sub)
        m = metrics(sub.y.to_numpy(float), r_mech(c1p, sub))
        transfer[cid] = dict(r2=m["r2"], rmse=m["rmse"], n_nodes=int(len(sub)))
    print("\nA2 全条件统一物理参数（C1 拟合）跨切片零重拟合迁移：")
    for cid, v in transfer.items():
        print(f"  {cid:12s} R2={v['r2']:.3f} RMSE={v['rmse']:.3f} nodes={v['n_nodes']}")

    payload = dict(meta=dict(
        data_source="修订版《体外EDTA数据记录汇总.xlsx》 SHA256 4c165298...",
        note="上一轮三层核验通过；源文件未改动、24 鼠全保留",
        T_MOD=T_MOD, T_ALL=T_ALL, F0=F0, D0=D0, T0=T0,
        routes=ROUTES, models=MODELS, conditions=CONDITIONS),
        results=results,
        A2_global_transfer=transfer,
        A2_global_params=[float(x) for x in c1p])

    (ROOT / "three_routes_results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # 主指标长表
    with open(ROOT / "three_routes_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "condition_label", "doses", "freqs_hz", "route", "model",
                    "family", "k", "n_cond_nodes", "n_rat_points", "n_rats",
                    "full_r2", "full_rmse", "full_mae", "full_bias", "full_aic",
                    "loco_r2", "loco_rmse", "loro_r2", "loro_rmse",
                    "loro_per_rat_mean_r2", "loro_folds", "baseline_pred", "params"])
        for cid, blk in results.items():
            sp = blk["spec"]
            for n, m in blk["models"].items():
                w.writerow([cid, sp["label"], "|".join(map(str, sp["doses"])),
                            "|".join(map(str, sp["freqs"])), ROUTES[m["route"]], n,
                            m["family"], m["k"], m["n_cond_nodes"], m["n_rat_points"], m["n_rats"],
                            f'{m["full"]["r2"]:.6f}', f'{m["full"]["rmse"]:.6f}',
                            f'{m["full"]["mae"]:.6f}', f'{m["full"]["bias"]:.6f}',
                            f'{m["full"]["aic"]:.2f}',
                            f'{m["loco"]["r2"]:.6f}', f'{m["loco"]["rmse"]:.6f}',
                            f'{m["loro"]["r2"]:.6f}', f'{m["loro"]["rmse"]:.6f}',
                            f'{m["loro"]["per_rat_mean_r2"]:.6f}', m["loro"]["n_folds"],
                            f'{m["baseline_pred"]:.6f}',
                            ";".join(f"{x:.6f}" for x in m["params"])])

    with open(ROOT / "trackB_four_conditions.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "condition_label", "n_points", "n_rats", "coverage",
                    "mean_width", "median_width", "max_width", "winkler_alpha_mis_0.10", "note"])
        for cid, blk in results.items():
            t = blk["trackB"]
            w.writerow([cid, blk["spec"]["label"], t["n_points"], t["n_rats"],
                        f'{t["coverage"]:.6f}', f'{t["mean_width"]:.6f}',
                        f'{t["median_width"]:.6f}', f'{t["max_width"]:.6f}',
                        f'{t["winkler"]:.6f}', t["note"]])

    print("\n计算完成：", ROOT)


if __name__ == "__main__":
    main()
