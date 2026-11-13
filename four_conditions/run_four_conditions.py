#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
四条件拟合 + 结论 + 机理原理 + 重组 GitHub 包（一次性生成）

数据权威来源：用户本轮上传的修订版《体外EDTA数据记录汇总.xlsx》
  SHA256 4c165298fbd37ab3c556413ae00dfb8e838e56b37c530e1dd1f0bfc2faa6cd71
  该 SHA 与 ../Chemical_Skull_Model_4Conditions/data_verification.json 记录一致，
  上一轮已完成三层核验（新增单位行 / 数值+1行偏移 0 差异 / 2016 行与冻结副本逐格一致），
  故本脚本直接复用已冻结的 frozen_measured_24rat.csv，不改动源文件。

四条件：
  C1 全部条件        4 剂量 x 4 频率 x 6 改性时间 = 96 条件节点
  C2 低频 100/120Hz  4 剂量 x 2 频率 x 6 时间    = 48
  C3 高浓度 15/20%   2 剂量 x 4 频率 x 6 时间    = 48
  C4 低频+高浓度     2 剂量 x 2 频率 x 6 时间    = 24

TrackA 系列（均为「矫正后」模型，本轮统一命名）：
  A1 经验线性校正        R_Z = a + b*raw
  A2 协变量扩展          R_Z = a + b*raw + c*t + d*ln(f/100)
  A3 机理融合混合式      R_Z = 1 - kappa*(f/100)^(-s)*delta - eps*t,  delta=1-raw
  REF 拉伸指数衰减（纯经验拟合上限，仅作对照，不进 TrackA 正式系列）

TrackB：既有 LOO 校准区间的四条件切片统计（未重新校准）；只报 coverage/width/Winkler，
        绝不报 R2/RMSE。
"""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
PKG = ROOT.parent
FIG = PKG / "four_conditions" / "figures"
for p in [ROOT, PKG, FIG, PKG / "TrackA", PKG / "TrackB", PKG / "data",
          PKG / "docs", PKG / "validation", PKG / "four_conditions"]:
    p.mkdir(parents=True, exist_ok=True)

FROZEN = PKG / "data" / "frozen_measured_24rat.csv"
PRIOR = PKG / "data" / "mechanism_timeseries_quantiles.csv"
TRACKB_ALL = PKG / "TrackB" / "calibrated_intervals_allfreq_392.csv"
TRACKB_LF = PKG / "TrackB" / "calibrated_intervals_lowfreq_196.csv"
VERIFY = PKG / "data" / "data_verification.json"

T_MOD = [5, 20, 35, 65, 95, 105]
T_ALL = [0, 5, 20, 35, 65, 95, 105]
F0 = 100.0

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

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False


# ==========================================================================
# 数据
# ==========================================================================
def load_data():
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
    w = (q[q.metric == "whole_resistance_ratio"]
         .drop_duplicates(["edta_concentration_mol_per_l", "global_time_s"])
         .sort_values(["edta_concentration_mol_per_l", "global_time_s"]))
    return rl, cond, w


_W = None


def raw_prior(dose, t):
    g = _W[_W.edta_concentration_mol_per_l == dose]
    ts = g.global_time_s.to_numpy(float); vs = g.q50.to_numpy(float)
    t_s = t * 60.0
    return float(vs[-1]) if t_s > ts.max() else float(np.interp(t_s, ts, vs))


def attach(df):
    d = df.copy()
    d["raw"] = [raw_prior(r.dose, r.t) for r in d.itertuples()]
    d["lf"] = np.log(d["freq"].to_numpy(float) / F0)
    d["delta"] = 1.0 - d["raw"].to_numpy(float)
    return d


# ==========================================================================
# 模型
# ==========================================================================
def _lam(s, X):
    return np.exp(-s * X["lf"].to_numpy(float))


def _b(p, X):
    return np.clip(p[1] + p[2] * X["dose"].to_numpy(float)
                   + p[3] * X["lf"].to_numpy(float), 1e-6, 5.0)


MODELS = {
    "A1": dict(label="A1 线性校正 a+b*raw", k=2, official=True,
               p0=[-4.7, 6.1], lo=[-50, -50], hi=[50, 50],
               pred=lambda p, X: p[0] + p[1] * X["raw"].to_numpy(float)),
    "A2": dict(label="A2 协变量扩展 +t+ln(f)", k=4, official=True,
               p0=[-4.7, 6.1, 0.0, 0.05], lo=[-50, -50, -1, -5], hi=[50, 50, 1, 5],
               pred=lambda p, X: (p[0] + p[1] * X["raw"].to_numpy(float)
                                  + p[2] * X["t"].to_numpy(float)
                                  + p[3] * X["lf"].to_numpy(float))),
    "A3": dict(label="A3 机理融合混合式", k=3, official=True,
               p0=[3.3, 0.11, 0.003], lo=[0.1, -1.0, -0.05], hi=[30.0, 1.5, 0.05],
               pred=lambda p, X: 1.0 - p[0] * _lam(p[1], X) * X["delta"].to_numpy(float)
               - p[2] * X["t"].to_numpy(float)),
    "REF": dict(label="对照：拉伸指数衰减", k=5, official=False,
                p0=[0.10, 0.02, 0.01, -0.002, 0.8],
                lo=[0.0, 1e-5, -1, -1, 0.2], hi=[0.9, 1.0, 1, 1, 3.0],
                pred=lambda p, X: p[0] + (1 - p[0]) * np.exp(
                    -np.power(_b(p, X) * X["t"].to_numpy(float), p[4]))),
}


def fit(name, X, y, p_init=None):
    m = MODELS[name]
    p0 = np.array(p_init if p_init is not None else m["p0"], float)
    try:
        r = least_squares(lambda p: m["pred"](p, X) - y, p0,
                          bounds=(np.array(m["lo"]), np.array(m["hi"])),
                          max_nfev=5000, ftol=1e-11, xtol=1e-11)
        return r.x
    except Exception:
        return p0


def predict(name, p, X):
    return np.asarray(MODELS[name]["pred"](p, X), float)


def metrics(y, yhat):
    y = np.asarray(y, float); yhat = np.asarray(yhat, float)
    sse = float(np.sum((y - yhat) ** 2)); sst = float(np.sum((y - y.mean()) ** 2))
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
    m_full = metrics(y_c, predict(name, p, c)); m_full["aic"] = aic(len(y_c), m_full["sse"], MODELS[name]["k"])

    yh = np.empty(len(c))
    for i in range(len(c)):
        mask = np.ones(len(c), bool); mask[i] = False
        pp = fit(name, c[mask].reset_index(drop=True), y_c[mask], p_init=p)
        yh[i] = predict(name, pp, c.iloc[[i]].reset_index(drop=True))[0]
    m_loco = metrics(y_c, yh)

    yr = r.y.to_numpy(float); rat = r.rat.to_numpy()
    yhat = np.empty(len(r))
    for one in np.unique(rat):
        te = rat == one; tr = ~te
        pp = fit(name, r[tr].reset_index(drop=True), yr[tr], p_init=p)
        yhat[te] = predict(name, pp, r[te].reset_index(drop=True))
    m_loro = metrics(yr, yhat)
    per = [metrics(yr[rat == one], yhat[rat == one])["r2"] for one in np.unique(rat)]
    m_loro["per_rat_mean_r2"] = float(np.nanmean(per))
    m_loro["n_folds"] = int(len(np.unique(rat)))

    base = attach(pd.DataFrame([dict(dose=spec["doses"][0], freq=float(spec["freqs"][0]),
                                     t=0.0, y=1.0)]))
    baseline = float(predict(name, p, base)[0])
    return dict(name=name, label=MODELS[name]["label"], k=MODELS[name]["k"],
                official=MODELS[name]["official"], params=[float(x) for x in p],
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
                max_width=float(s.width.max()),
                winkler=float(s[wcol].mean()),
                note="既有 LOO 校准区间按条件切片；未重新拟合、未重新校准")


# ==========================================================================
# 主流程
# ==========================================================================
def main():
    global _W
    rl, cond, _W = load_data()
    results = {}
    for cid, spec in CONDITIONS.items():
        results[cid] = dict(spec=spec,
                            models={n: evaluate(n, cond, rl, spec) for n in MODELS},
                            trackB=trackb_slice(spec))
        r = results[cid]["models"]
        print(f'--- {cid} ---')
        for n in MODELS:
            m = r[n]
            print(f'  {n:4s} full={m["full"]["r2"]:.3f} LOCO={m["loco"]["r2"]:.3f} '
                  f'LORO={m["loro"]["r2"]:.3f} (folds={m["loro"]["n_folds"]}) '
                  f'base={m["baseline_pred"]:.3f}')
        tb = results[cid]["trackB"]
        print(f'  TrackB n={tb["n_points"]} cov={tb["coverage"]:.3f} '
              f'width={tb["mean_width"]:.3f} winkler={tb["winkler"]:.3f}')

    (ROOT / "four_conditions_results.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")

    # CSV：TrackA 长表
    with open(ROOT / "four_conditions_metrics.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "condition_label", "doses", "freqs_hz", "model", "official",
                    "k", "n_cond_nodes", "n_rat_points", "n_rats",
                    "full_r2", "full_rmse", "full_mae", "full_bias", "full_aic",
                    "loco_r2", "loco_rmse", "loro_r2", "loro_rmse",
                    "loro_per_rat_mean_r2", "loro_folds", "baseline_pred", "params"])
        for cid, blk in results.items():
            sp = blk["spec"]
            for n, m in blk["models"].items():
                w.writerow([cid, sp["label"], "|".join(map(str, sp["doses"])),
                            "|".join(map(str, sp["freqs"])), n,
                            "yes" if m["official"] else "reference_only", m["k"],
                            m["n_cond_nodes"], m["n_rat_points"], m["n_rats"],
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

    make_figures(results, cond)
    make_documents(results)
    make_github_package(results)
    print("\n完成：", ROOT)


# ==========================================================================
# 图
# ==========================================================================
def make_figures(results, cond):
    C = {"A1": "#4C72B0", "A2": "#8172B3", "A3": "#C44E52", "REF": "#8C8C8C"}
    ids = list(CONDITIONS)

    # 图1 四条件 x 模型 full/LORO R2
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.2))
    for ax, cid in zip(axes.ravel(), ids):
        blk = results[cid]; x = np.arange(4); wd = 0.36
        fv = [blk["models"][n]["full"]["r2"] for n in C]
        lv = [blk["models"][n]["loro"]["r2"] for n in C]
        b1 = ax.bar(x - wd / 2, fv, wd, color="#A9C4DE", label="full R2（样本内）")
        b2 = ax.bar(x + wd / 2, lv, wd, color=[C[n] for n in C], label="LORO R2（留一大鼠）")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                        f"{b.get_height():.3f}", ha="center", fontsize=8.5, fontweight="bold")
        ax.set_xticks(x); ax.set_xticklabels(list(C.keys()), fontsize=10)
        ax.set_ylim(0, 1.08); ax.set_ylabel("R2")
        ax.set_title(blk["spec"]["label"] + f'  |  {blk["models"]["A1"]["n_cond_nodes"]} 节点 / LORO {blk["models"]["A1"]["loro"]["n_folds"]} 折',
                     fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle=":"); ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#A9C4DE"),
               plt.Rectangle((0, 0), 1, 1, color="#4C72B0")]
    fig.legend(handles, ["full R2（样本内，浅蓝）", "LORO R2（深色，颜色区分模型）"],
               loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, fontsize=10, frameon=False)
    fig.suptitle("图1｜四种条件下 TrackA 系列与对照模型的拟合 / 独立验证（纵轴自 0 起）",
                 fontsize=13.5, fontweight="bold", y=1.045)
    fig.text(0.5, 0.012,
             "REF 为纯经验拉伸指数（拟合上限对照，不属 TrackA 正式系列）。"
             "四个条件的评价集与 LORO 折数不同（C1/C2 为 24 折，C3/C4 为 12 折），R2 数值不可横向解读为模型优劣。",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.05, 1, 0.955])
    fig.savefig(FIG / "fig1_metrics_by_condition.png", dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 图2 C4 低频+高浓度：2 剂量 x 2 频率 = 4 个组合，每个 panel 画全 7 个时间点 + SEM
    spec = CONDITIONS["C4_lf_hd"]
    combos = [(d, f) for d in spec["doses"] for f in spec["freqs"]]
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.6), sharex=True, sharey=True)
    for ax, (dose, f) in zip(axes.ravel(), combos):
        sub = cond[(cond.dose == dose) & (cond.freq == f)].sort_values("t")
        tt = [0.0] + list(sub.t.to_numpy(float))
        yy = [1.0] + list(sub.y.to_numpy(float))
        se = [0.0] + list(sub["sem"].to_numpy(float))
        ax.errorbar(tt, yy, yerr=se, fmt="o", color="#111111", markersize=5.2,
                    capsize=3.5, elinewidth=1.2,
                    label="实测 ± SEM（N=6）", zorder=6)
        X = attach(pd.DataFrame(dict(dose=[float(dose)] * len(T_ALL),
                                    freq=[float(f)] * len(T_ALL),
                                    t=[float(t) for t in T_ALL], y=[0.0] * len(T_ALL))))
        for n, col in [("A1", "#4C72B0"), ("A2", "#8172B3"), ("A3", "#C44E52")]:
            p = np.asarray(results["C4_lf_hd"]["models"][n]["params"], float)
            v = predict(n, p, X).copy(); v[0] = 1.0  # 基线按物理定义锚定
            ax.plot(T_ALL, v, "-", color=col, lw=1.8, alpha=0.9, label=n, zorder=4)
        ax.axvspan(95, 105, color="#FFF2CC", alpha=0.7, zorder=0)
        ax.set_title(f"EDTA {dose} M / {f} Hz", fontsize=11.5, fontweight="bold")
        ax.set_ylim(0, 1.30); ax.set_xlim(-3, 108); ax.set_xticks(T_ALL)
        ax.grid(alpha=0.3, linestyle=":"); ax.set_axisbelow(True)
    for ax in axes[1]:
        ax.set_xlabel("全局时间 t (min)：0 基线 / 5 乙酸 / 20·35·65·95 EDTA / 105 尿素", fontsize=9.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("R_Z = Z / Z0", fontsize=10.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=4, fontsize=9, frameon=False)
    fig.suptitle("图2｜C4（低频 + 高浓度）完整时间序列：7 个时间点与 SEM 全画，纵坐标自 0 起",
                 fontsize=13.5, fontweight="bold", y=1.045)
    fig.text(0.5, 0.012,
             "黄色区为 95 min 后：机理先验截断为平台，实测 t=105（乙醇+尿素）回升，所有单调模型均无法表达。"
             "A1/A2 在 t=0 按物理定义锚定为 1.0；A3 因 delta(t=0)=0 天然等于 1。",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.04, 1, 0.955])
    fig.savefig(FIG / "fig2_timeseries_C4_full.png", dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 图3 TrackB 四条件
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    cov = [results[c]["trackB"]["coverage"] for c in ids]
    wid = [results[c]["trackB"]["mean_width"] for c in ids]
    win = [results[c]["trackB"]["winkler"] for c in ids]
    short = ["C1 全部", "C2 低频", "C3 高浓度", "C4 低频+高浓度"]
    for ax, vals, title, ylab, target in [
        (axes[0], cov, "(a) 逐点覆盖率", "coverage", 0.90),
        (axes[1], wid, "(b) 平均区间宽度", "mean width", None),
        (axes[2], win, "(c) Winkler (alpha_mis=0.10)", "Winkler", None)]:
        bars = ax.bar(range(4), vals, color=["#4C72B0", "#8172B3", "#C5701F", "#C44E52"])
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + max(vals) * 0.02,
                    f"{v:.3f}", ha="center", fontsize=9, fontweight="bold")
        if target:
            ax.axhline(target, color="#2F7A4F", lw=1.8, linestyle="--", label="目标 0.90")
            ax.legend(fontsize=8.5)
        ax.set_xticks(range(4)); ax.set_xticklabels(short, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.22); ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle=":"); ax.set_axisbelow(True)
    fig.suptitle("图3｜TrackB 四条件切片：覆盖率、宽度、Winkler（不报 R2 / RMSE）",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.text(0.5, 0.005,
             "仅为既有 392 点 LOO 校准区间按条件切片统计，未重新校准；"
             "区间宽度约 97% 来自 conformal padding，高频段明显更宽；逐点覆盖率不等于整条曲线联合覆盖。",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.075, 1, 0.94])
    fig.savefig(FIG / "fig3_trackB_four_conditions.png", dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # 图4 表1
    rows = []
    for cid in ids:
        blk = results[cid]
        for n in ["A1", "A2", "A3"]:
            m = blk["models"][n]
            rows.append([short[ids.index(cid)], n, str(m["k"]), str(m["n_cond_nodes"]),
                         f'{m["full"]["r2"]:.3f}', f'{m["full"]["rmse"]:.3f}',
                         f'{m["loco"]["r2"]:.3f}', f'{m["loro"]["r2"]:.3f}',
                         f'{m["loro"]["per_rat_mean_r2"]:.3f}', str(m["loro"]["n_folds"])])
    fig, ax = plt.subplots(figsize=(14.5, 6.6))
    ax.axis("off")
    cols = ["条件", "模型", "k", "节点", "full R2", "RMSE", "LOCO R2", "LORO R2", "逐鼠均值", "LORO 折数"]
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center",
                   colWidths=[0.16, 0.075, 0.05, 0.07, 0.09, 0.09, 0.09, 0.09, 0.10, 0.10])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9.2); tbl.scale(1, 1.55)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2F5D96"); tbl[0, j].set_text_props(color="white", weight="bold")
    for i in range(len(rows)):
        if i % 2 == 1:
            for j in range(len(cols)):
                tbl[i + 1, j].set_facecolor("#F2F6FB")
    ax.set_title("表1｜四条件 x TrackA 系列拟合指标（REF 对照见 CSV；口径不同不可横比）",
                 fontsize=13, fontweight="bold", pad=14)
    fig.text(0.5, 0.02,
             "LORO = 留一大鼠（个体级独立验证）；LOCO = 留一条件（群体均值层）。"
             "C1/C2 为 24 折、C3/C4 为 12 折，评价集与折数均不同。",
             ha="center", fontsize=9, color="#555555")
    fig.savefig(FIG / "fig4_table1.png", dpi=165, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("图1-4 已生成")


# ==========================================================================
# 文档
# ==========================================================================
def _tbl(results):
    ids = list(CONDITIONS)
    short = ["C1 全部", "C2 低频", "C3 高浓度", "C4 低频+高浓度"]
    out = ["| 条件 | 节点 | 模型 | k | full R2 | RMSE | LOCO R2 | **LORO R2** | 逐鼠均值 | LORO 折数 |",
           "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for i, cid in enumerate(ids):
        for n in ["A1", "A2", "A3"]:
            m = results[cid]["models"][n]
            out.append(f'| {short[i]} | {m["n_cond_nodes"]} | {n} | {m["k"]} | '
                       f'{m["full"]["r2"]:.3f} | {m["full"]["rmse"]:.3f} | '
                       f'{m["loco"]["r2"]:.3f} | **{m["loro"]["r2"]:.3f}** | '
                       f'{m["loro"]["per_rat_mean_r2"]:.3f} | {m["loro"]["n_folds"]} |')
    return "\n".join(out)


def _tb(results):
    ids = list(CONDITIONS)
    short = ["C1 全部", "C2 低频", "C3 高浓度", "C4 低频+高浓度"]
    out = ["| 条件 | 点数 | 鼠数 | 覆盖率 | 平均宽度 | 中位宽度 | 最大宽度 | Winkler |",
           "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for i, cid in enumerate(ids):
        t = results[cid]["trackB"]
        out.append(f'| {short[i]} | {t["n_points"]} | {t["n_rats"]} | {t["coverage"]:.3f} | '
                   f'{t["mean_width"]:.3f} | {t["median_width"]:.3f} | '
                   f'{t["max_width"]:.3f} | {t["winkler"]:.3f} |')
    return "\n".join(out)


def make_documents(results):
    ids = list(CONDITIONS)
    a2 = {c: results[c]["models"]["A2"]["loro"]["r2"] for c in ids}
    a3 = {c: results[c]["models"]["A3"]["loro"]["r2"] for c in ids}
    ref = {c: results[c]["models"]["REF"]["loro"]["r2"] for c in ids}
    best_loro = {c: max(["A1", "A2", "A3", "REF"], key=lambda n: results[c]["models"][n]["loro"]["r2"])
                 for c in ids}
    ref_txt = " / ".join(f"{c}={ref[c]:.3f}" for c in ids)

    conclusion = f"""# 四条件拟合结论

> 数据：本轮上传的修订版《体外EDTA数据记录汇总.xlsx》
> SHA256 `4c165298fbd37ab3c556413ae00dfb8e838e56b37c530e1dd1f0bfc2faa6cd71`，
> 与 `data/data_verification.json` 记录一致；三层核验已完成（新增单位行 /
> 数值 +1 行偏移后 0 差异 / 2016 行与冻结副本逐格一致）。源文件未被修改。

## 一句话结论

**四个条件下，「条件越窄、指标越好看」是评价范围变化，不是模型变好。**

若目标函数**只含统计拟合指标**，REF（拉伸指数衰减）在四个条件的 LORO 上**全部第一**
（相对正式系列最优者：C1 +0.005、C2 +0.056、C3 +0.018、C4 +0.047）。
但它脱离机理链路、5 个参数无物理意义、单调曲线无法表达尿素回升、不能跨浓度外推，
**不是可用模型**，故只作拟合上限对照，不纳入 TrackA 正式系列。

在正式系列里，**A2 与 A3 的 LORO 四条件均值几乎相同（约 0.596），统计上不可区分**：
A2 的样本内拟合略好且协变量可解释，A3 参数更少、基线自动 =1、频率项有物理出处。
按「拟合程度 + 模型效果 + 可用性」综合，推荐 **A2 为默认、A3 为结构优先备选**。
C4 中 A1 反而略高（0.692），但 A1/A2/A3 相差 0.012 属噪声级，且 C4 只剩 12 鼠 / 24 节点，
不能据此说「简单模型更好」。

## 1. 四条件定义（务必先看）

| 条件 | 剂量 | 频率 | 条件节点 | LORO 折数 | 鼠级点 |
|---|---|---|---:|---:|---:|
| C1 全部条件 | 0.05/0.10/0.15/0.20 M | 100/120/1000/10000 Hz | {results['C1_all']['models']['A1']['n_cond_nodes']} | 24 | {results['C1_all']['models']['A1']['n_rat_points']} |
| C2 低频 | 4 剂量 | 100/120 Hz | {results['C2_lowfreq']['models']['A1']['n_cond_nodes']} | 24 | {results['C2_lowfreq']['models']['A1']['n_rat_points']} |
| C3 高浓度 | 0.15/0.20 M | 4 频率 | {results['C3_highdose']['models']['A1']['n_cond_nodes']} | **12** | {results['C3_highdose']['models']['A1']['n_rat_points']} |
| C4 低频+高浓度 | 0.15/0.20 M | 100/120 Hz | {results['C4_lf_hd']['models']['A1']['n_cond_nodes']} | **12** | {results['C4_lf_hd']['models']['A1']['n_rat_points']} |

**四个条件的评价集、节点数、LORO 折数都不同（C3/C4 只有 12 只鼠），R² 数值不能横向解读为「模型持续改进」。**
每只大鼠只接受一种 EDTA 浓度（被试间设计），故 C3/C4 天然只有 12 鼠。

## 2. TrackA 四条件拟合（每条件独立重新拟合参数）

{_tbl(results)}

REF（拉伸指数衰减）LORO：{ref_txt}，仅作拟合上限对照，不属 TrackA 正式系列。

**读数要点**

- 每个条件下 LORO 最高者：{", ".join(f"{c}={best_loro[c]}" for c in ids)}。
- A2 的 LORO：{", ".join(f"{c}={a2[c]:.3f}" for c in ids)}；A3：{", ".join(f"{c}={a3[c]:.3f}" for c in ids)}。
  A2 与 A3 的差在多数条件下小于 0.02，属噪声级。
- 从 C1 到 C4，full R² 与 LORO 普遍升高，主要因为**剔除了高频（机理先验频率无关造成的最大误差源）和两个低剂量**，
  且 C4 的两条高浓度曲线几乎重合。**这是评价集收窄，不是模型改进**；同时 C4 只剩 12 鼠、24 节点，外推能力反而下降。
- 基线自洽性：A3 因 `delta(t=0)=0` 恒等于 1.000；A1 / A2 在 t=0 的外推值偏离 1，使用时须按物理定义锚定。

## 3. TrackB 四条件（只报区间指标，不报 R² / RMSE）

{_tb(results)}

- 四条件覆盖率都靠近 0.90 目标，但**宽度差异很大**：高频段明显更宽，这是 conformal padding 在补偿「仿真 vs 实测」的系统性偏差，
  不是 LHS 参数不确定性（B1 诊断：基础 LHS 区间仅占约 3%，padding 占约 97%）。
- 这些数字是**既有 392 点 LOO 校准区间按条件切片的统计**，本轮**没有重新拟合、没有重新校准** TrackB。
- 逐点覆盖率 ≠ 整条时间曲线同时落入区间的概率；基线(t=0) 与尿素(t=105) 无校准区间。
- 已知治理问题：验证集仅 22/24 鼠（缺 WT5 0.10 M、WT6 0.15 M）；0.05 M 只出现在欠覆盖的
  ORIGINAL_COMPATIBLE 模式，存在 role x 剂量混淆；上游机制选择非严格嵌套。

## 4. 结论

1. **若只优化统计指标（full R² / RMSE / LORO R²）：REF 最优，但不建议采用。**
   它四个条件全部第一，优势在低频子集上达 0.05 量级（不是噪声）；代价是 5 个无物理意义参数、
   丢掉机理链路、无法表达尿素回升、跨浓度外推没有物理约束、与 TrackB 口径脱节。
   **指标最优 ≠ 可用模型**，这正是本仓库反复强调「拟合变好不等于模型变好」的实例。
2. **若追求「拟合程度 + 模型效果 + 可用性」：选 A2**。`R_Z = a + b*raw + c*t + d*ln(f/100)`，
   四个条件下样本内拟合在正式系列里最好，LORO 与 A3 持平，参数可解释，保留机理先验链路，
   可跨浓度外推，能与 TrackB 口径衔接。
3. **若同时要求结构自洽：选 A3**。`R_Z = 1 - kappa*(f/100)^(-s)*delta - eps*t`，
   LORO 与 A2 基本持平（四条件均值同为约 0.596），但 t=0 天然等于 1、只需 3 个参数、
   频率项有物理出处。二者的取舍不在精度，而在「要不要把频率维度写回机理层」。
4. **不要写「C4 拟合最好所以模型最好」**：C4 的漂亮数字来自评价范围收窄 + 两高浓度曲线近似重合 + 12 鼠小样本；
   且 C4 中 A1 的 LORO 甚至略高于 A2/A3，说明自由度在极小评价集上不再带来收益。
5. **A1 是最稳的交付基线**，但只修正幅度偏差，未解决频率维度缺失；A2/A3 才补上了频率与时间维度。
6. **四个条件的 R² 不可横向比大小**：C1/C2 为 24 折 LORO、C3/C4 为 12 折；节点数 96/48/48/24 各不相同。

## 5. 遗留限制（如实保留）

- 尿素 t=105 回升：所有单调模型（含 A3、REF）都无法表达；机理先验在 95 min 后截断为平台外推。
- 机理先验 `raw` 只落在 0.794–0.883（宽 0.089），实测跨 0.12–0.92（宽 0.80）；
  同一 `raw` 值对应实测可从 0.15 到 0.95，误差来自缺失的维度，不是 `f(raw)` 的形状。
- 14 处 Excel↔measured CSV 不一致（urea_10 / Midpoint / WT21–24 / 4 频率）与 32 条重复录入属局部转录异常，
  **按要求保留全部样本，未删除 WT17 / WT9 / WT8 等不利个体**，标记未决、未闭环。
- 本地 Excel SHA 与上游 manifest 声明的 source_data SHA 不一致，上游原始文件不可得。
- 96/48/24 节点属小样本；多模型比较存在选择偏差，AIC 已惩罚参数但未做多重比较校正；
  LORO 逐鼠 R² 方差大（个别鼠为负），仅作参考。

## 附：复现与文件

`four_conditions/run_four_conditions.py`（本目录上一级）、
`four_conditions/four_conditions_metrics.csv`、`four_conditions/trackB_four_conditions.csv`、
`four_conditions/four_conditions_results.json`、`four_conditions/figures/` 四张图。
"""
    (PKG / "four_conditions" / "FOUR_CONDITIONS_CONCLUSION.md").write_text(conclusion, encoding="utf-8")
    (ROOT / "FOUR_CONDITIONS_CONCLUSION.md").write_text(conclusion, encoding="utf-8")

    principle = """# 「机理 + 经验」下的机理原理

> 这份文档回答：矫正后的 TrackA 里，哪些是机理、哪些是经验、两者怎么接、机理上还缺什么。
> 对应仓库 `TrackA/A1`、`TrackA/A2`、`TrackA/A3` 与 `TrackB/`。

## 1. 机理层（第一性原理，未改动）

原始模型是一个**准静态纯电阻网络**（绝对孔隙率 + 空间串联积分）：

```
sigma_i = sigma_floor + c * phi_i^m          (Archie 型局部电导率，phi = 孔隙率)
R       = sum_i dx_i / (sigma_i * A_cell)    (沿厚度方向串联积分)
R_Z_raw = Z(t) / Z(0) = sigma(0) / sigma(t) (阻抗比值，无量纲)
```

物理图像：EDTA 螯合颅骨羟基磷灰石中的 Ca2+，使孔隙率 phi 上升 → 离子电导 sigma 上升 →
串联电阻 R 下降 → 阻抗比值 R_Z 小于 1。基线 t=0 按定义 R_Z = 1。

**这条链路从未被改动**。A1/A2/A3 都是在它外面接校准层，或在它内部补维度，
不是"证伪机理"或"辨识出机理参数"。

## 2. 机理层的三个结构性缺口（实测核对）

| 缺口 | 证据 | 后果 |
|---|---|---|
| **频率无关** | `R_Z_raw` 与 f 完全无关（只有剂量 x 时间一条曲线族）；实测 100/120/1000/10000 Hz 明显分层，高频 R_Z 系统性偏高 | 机理先验无法表达频率维度，这是最大误差源 |
| **脱钙动力学幅度太小 / 太慢** | `raw` 只从 1.0 降到 0.794–0.883（改性量 delta = 0.12–0.21）；实测降到 0.12–0.92（delta 高达 0.8） | 需要放大因子，或补时间维度 |
| **95 min 后截断 + 尿素回升** | 先验在 95 min 取末值平台；实测 t=105（乙醇+尿素）回升 | 所有单调模型无法表达 |

补充：`raw` 全部节点只落在 0.794–0.883（宽 0.089），实测跨 0.12–0.92（宽 0.80）。
在这个窄区间上任何平滑函数一阶泰勒展开都近似线性，所以**换 `f(raw)` 的形状（反比例 / 指数 / 米氏 / 幂）改善上限很低**；
同一 `raw` 值对应的实测可以从 0.15 到 0.95，且按频率分层 —— 误差来自 raw 里没有的维度，不是函数形状。

## 3. 经验层（三种接法，对应 A1 / A2 / A3）

### A1 · 经验线性校正（幅度补丁）

```
R_Z_corr = a + b * R_Z_raw         (a ≈ -4.47, b ≈ 5.68，低频 48 节点拟合)
```

只修正**幅度偏差**，不改机理。A0 纯机理 R² 为负但 obs–pred 相关 r=0.885，
说明"趋势对、幅度错"，加一个仿射校正即可。**A1 是经验后处理，不是机理改进**；
基线 t=0 外推值偏离 1，使用时须按物理定义锚定。

### A2 · 协变量扩展（把缺失维度补进校准层）

```
R_Z = a + b * R_Z_raw + c*t + d*ln(f/100)
```

`t` 补机理脱钙动力学太慢的部分，`ln(f/100)` 补机理频率无关的部分。
只加一个频率项就能让 full R² 从 0.627 升到 0.810（+0.183），比换任何 `f(raw)` 都有效。
**但 c、d 是在校准层的回归系数，没有物理出处** —— 这是"机理 + 经验"里经验成分最重的一档。

### A3 · 机理融合混合式（把频率维度写回机理层）

机理假说：真实颅骨不是纯电阻，而是「**离子通路 ∥ 界面极化 / 电容通路**」并联。

- 低频（100/120 Hz）：电流主要走细胞外液 / 孔隙的离子通路。脱钙使 phi↑ → sigma↑ → Z 显著下降。
- 高频（1000/10000 Hz）：电容耦合与界面极化旁路占主导，对 phi 变化不敏感 → 改性后 R_Z 更接近 1。
  这正好解释实测"同一时间下高频 R_Z 系统性偏高"。

用常相位角（CPE）幂律把"改性量对测量频率的可见度"写成
`lambda(f) = (f/f0)^(-s)`（s 为频散指数），再用 kappa 放大被机理低估的脱钙幅度：

```
delta(t,d) = 1 - R_Z_raw(t,d)                     (机理预测的改性量)
R_Z = 1 - kappa * lambda(f) * delta - eps * t
```

- `delta` 与 `lambda(f)` 属于机理层；`eps*t` 是校准层唯一余项，标记机理 phi(t) 动力学待重标定之处。
- **基线自动 = 1**（t=0 时 delta=0），无需人为锚定；3 个参数，每个都有物理含义：
  kappa = 脱钙幅度放大（对应 phi(t) 增长率或 Archie 系数 c 的标定误差）；
  s = 界面极化频散指数；eps = 未覆盖的时间漂移。
- 一阶展开 `lambda ≈ 1 - s*ln(f/f0)` 给出可检验预言：频率效应应以 `ln(f)*delta` 交互项出现。
  实测交互版 LORO 0.428 vs 加性 0.436，**数据未支持该预言**，故 lambda 的幂律形式应视为最简假说下的等效参数化。

## 4. TrackB 的机理 + 经验定位

TrackB 保留机理 ensemble 框架：LHS 采样机理参数 → 逐点 `sim_q05/q50/q95` → 兼容性筛选 /
最近邻 ensemble 兜底 → **折 x 频率分层 conformal 校准**：

```
s_j = max(0, q05_j - y_j, y_j - q95_j) / tolerance_f
q_f = Quantile_0.90(内层 cross-fit 分数 s)
p_(fold,f) = tolerance_f * q_f
calibrated_lower = clip(sim_q05 - p, 0, 1)
calibrated_upper = clip(sim_q95 + p, 0, 1)
```

机理层给区间中心与形状，经验层（padding）补偿"仿真 vs 实测"的系统性 + 个体差异。
B1 诊断：**最终宽度约 97% 来自 padding，基础 LHS 参数不确定仅约 3%** —— 所以"区间过宽"的根因不是参数采样范围不足，
盲目扩大 LHS 采样范围没有用。TrackB 只报 coverage / width / Winkler，**不报 R² / RMSE**。

## 5. 必须诚实说明的边界

- **s 不是被辨识的 CPE 指数**：只有 4 个频率点，幂律 lambda(f)、加性 ln(f)、交互 ln(f)*delta 在统计上不可区分；
  要辨识需独立宽频 EIS（介电谱）实验。
- **eps 仍是经验项**：只要它还在，"结合机理"就没有彻底完成；它指向真正的 A2 工作 —— 用孔隙率 / 钙含量实测
  重标定 phi(t)，而不是继续调 sigma_floor / c / m。
- **换 `f(raw)` 不是出路**：反比例比现行线性还差（0.612 vs 0.627）；幂模型拟合出 beta=14.88
  （Archie m 物理先验 1.3–2.0），属参数互相补偿，拟合变好不等于机理更正确。
- **拟合变好 != 模型变好**：条件收窄、新增自由度、补协变量都会抬高 R²；一律以 LORO 留一大鼠为验收口径。
- 尿素回升、高频机理缺失、padding 的 role x 剂量混淆、数据转录未决项，均为 OPEN ISSUES。

## 6. 一句话总括

> **机理层**给出"为什么阻抗会降"（脱钙 → 孔隙率 → 电导率 → 串联电阻）与"频率为什么分层"（离子通路 ∥ 界面极化）；
> **经验层**只负责修正机理尚未标定的幅度、动力学与个体误差。
> A1 是纯经验补丁，A2 是把缺失维度补在校准层，A3 是把频率维度写回机理层、只留一个时间余项。
> 三者都不构成"机理被证实"，只是"有物理动机的参数化"。
"""
    (PKG / "docs" / "MECHANISM_AND_EMPIRICAL_PRINCIPLE.md").write_text(principle, encoding="utf-8")
    (ROOT / "MECHANISM_AND_EMPIRICAL_PRINCIPLE.md").write_text(principle, encoding="utf-8")
    print("结论与机理原理文档已生成")


# ==========================================================================
# GitHub 包
# ==========================================================================
MODEL_PY = {
"A1": '''#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""TrackA / A1 — 经验线性校正（矫正后的点预测模型，原 TrackA1 已统一命名为 TrackA）"""
import os, numpy as np, pandas as pd

A, B = {a}, {b}
STAGE_BY_TIME = {{0:"baseline",5:"acetic_acid",20:"edta_15",35:"edta_30",
                  65:"edta_60",95:"edta_90",105:"urea_10"}}
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
    return {{"model_id":"TrackA_A1","formula":"R_Z = a + b*R_Z_raw",
             "a":A,"b":B,"baseline_anchor":1.0,
             "label":"EMPIRICAL correction, NOT a mechanistic improvement",
             "original_mechanism_unchanged":"sigma = 1e-4 + 1.5*phi^m ; R = sum dx/(sigma*A)"}}
''',
"A2": '''#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""TrackA / A2 — 协变量扩展校正（补时间 t 与频率 ln(f/100) 两个机理先验缺失的维度）"""
import os, numpy as np, pandas as pd

P = np.array({params}, float)   # [a, b, c_t, d_lnf]
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
    return {{"model_id":"TrackA_A2","formula":"R_Z = a + b*raw + c*t + d*ln(f/100)",
             "params":P.tolist(),"f0_hz":F0,"baseline_anchor":1.0,
             "label":"empirical covariates added to the calibration layer; c,d have no physical derivation"}}
''',
"A3": '''#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""TrackA / A3 — 机理融合混合式（频率维度写回机理层，校准层只留一个时间余项）"""
import os, numpy as np, pandas as pd

KAPPA, S, EPS = {params}   # kappa, s, eps
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
    return {{"model_id":"TrackA_A3",
             "formula":"R_Z = 1 - kappa*(f/f0)^(-s)*(1-R_Z_raw) - eps*t",
             "kappa":KAPPA,"s":S,"eps_per_min":EPS,"f0_hz":F0,
             "baseline_anchor":"automatic (=1 by construction)",
             "label":"mechanism-informed parameterization; s is NOT an identified CPE exponent"}}
''',
}


def make_github_package(results):
    # 数据
    for src, dst in [(FROZEN, PKG / "data" / "frozen_measured_24rat.csv"),
                     (PRIOR, PKG / "data" / "mechanism_timeseries_quantiles.csv"),
                     (VERIFY, PKG / "data" / "data_verification.json"),
                     (TRACKB_ALL, PKG / "TrackB" / "calibrated_intervals_allfreq_392.csv"),
                     (TRACKB_LF, PKG / "TrackB" / "calibrated_intervals_lowfreq_196.csv")]:
        shutil.copy2(src, dst)

    # 四条件结果
    for src, dst in [(ROOT / "four_conditions_results.json", PKG / "four_conditions" / "four_conditions_results.json"),
                     (ROOT / "four_conditions_metrics.csv", PKG / "four_conditions" / "four_conditions_metrics.csv"),
                     (ROOT / "trackB_four_conditions.csv", PKG / "four_conditions" / "trackB_four_conditions.csv")]:
        shutil.copy2(src, dst)

    # 包内复现脚本：把沙箱绝对路径改成仓库内相对路径
    src_py = Path(__file__).read_text(encoding="utf-8")
    repl = {
        'ROOT = Path(__file__).resolve().parent':
            'ROOT = Path(__file__).resolve().parent',
        'PKG = ROOT.parent': 'PKG = ROOT.parent',
        'FROZEN = PKG / "data" / "frozen_measured_24rat.csv"':
            'FROZEN = PKG / "data" / "frozen_measured_24rat.csv"',
        'PRIOR = PKG / "data" / "mechanism_timeseries_quantiles.csv"':
            'PRIOR = PKG / "data" / "mechanism_timeseries_quantiles.csv"',
        'TRACKB_ALL = PKG / "TrackB" / "calibrated_intervals_allfreq_392.csv"':
            'TRACKB_ALL = PKG / "TrackB" / "calibrated_intervals_allfreq_392.csv"',
        'TRACKB_LF = PKG / "TrackB" / "calibrated_intervals_lowfreq_196.csv"':
            'TRACKB_LF = PKG / "TrackB" / "calibrated_intervals_lowfreq_196.csv"',
        'VERIFY = PKG / "data" / "data_verification.json"':
            'VERIFY = PKG / "data" / "data_verification.json"',
        'val_src = PKG / "validation"':
            'val_src = PKG / "validation"',
    }
    for a, b in repl.items():
        if a not in src_py:
            raise RuntimeError("包内脚本路径替换失败: " + a[:60])
        src_py = src_py.replace(a, b)
    (PKG / "four_conditions" / "run_four_conditions.py").write_text(src_py, encoding="utf-8")

    # TrackA A1/A2/A3：参数取 C1 全部条件（最通用）拟合值
    c1 = results["C1_all"]["models"]
    params = {
        "A1": dict(a=c1["A1"]["params"][0], b=c1["A1"]["params"][1],
                   plist=c1["A1"]["params"]),
        "A2": dict(plist=c1["A2"]["params"]),
        "A3": dict(plist=c1["A3"]["params"]),
    }
    for n in ["A1", "A2", "A3"]:
        d = PKG / "TrackA" / n
        d.mkdir(parents=True, exist_ok=True)
        if n == "A1":
            code = MODEL_PY["A1"].format(a=params["A1"]["a"], b=params["A1"]["b"])
        else:
            code = MODEL_PY[n].format(params=[round(x, 8) for x in params[n]["plist"]])
        (d / "model.py").write_text(code, encoding="utf-8")
        m = c1[n]
        (d / "parameters.json").write_text(json.dumps(dict(
            model_id=f"TrackA_{n}", formula=MODELS[n]["label"],
            params=m["params"], k=m["k"],
            fit_scope="C1 全部条件（4 剂量 x 4 频率 x 6 改性时间，96 条件节点）",
            baseline_pred_uncorrected=m["baseline_pred"],
            metrics=dict(full_r2=m["full"]["r2"], rmse=m["full"]["rmse"],
                         loco_r2=m["loco"]["r2"], loro_r2=m["loro"]["r2"],
                         loro_per_rat_mean_r2=m["loro"]["per_rat_mean_r2"]),
            four_conditions_loro={c: results[c]["models"][n]["loro"]["r2"] for c in CONDITIONS},
            label="corrected model; empirical calibration, NOT a mechanistic proof"
        ), ensure_ascii=False, indent=2), encoding="utf-8")

    # TrackB model.py（简版，读区间 + 统计）
    (PKG / "TrackB" / "model.py").write_text('''#!/usr/bin/env python3.11
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
    return np.clip(np.asarray(sim_q05, float) - padding, 0, 1), \\
           np.clip(np.asarray(sim_q95, float) + padding, 0, 1)

def info():
    return {"model_id":"TrackB_calibrated", "target_coverage":0.90,
            "padding_formula":"p_(fold,f)=tolerance_f*q_f ; q_f=Quantile_0.90(inner cross-fit scores)",
            "diagnosis":"~97% of final width comes from conformal padding, NOT LHS parameter uncertainty",
            "caveats":["pointwise coverage != joint curve coverage",
                       "validation set 22/24 rats (WT5 0.10M, WT6 0.15M missing)",
                       "role x dose confounding for 0.05 M",
                       "no R2/RMSE reported"]}
''', encoding="utf-8")
    (PKG / "TrackB" / "parameters.json").write_text(json.dumps(dict(
        model_id="TrackB_calibrated", target_coverage=0.90, alpha_mis=0.10,
        padding="p_(fold,f) = tolerance_f * q_f",
        intervals_allfreq_392="4 frequencies, used for four-condition slicing",
        intervals_lowfreq_196="100/120 Hz, delivery version",
        four_conditions={c: results[c]["trackB"] for c in CONDITIONS},
        label="NO R2/RMSE for TrackB"
    ), ensure_ascii=False, indent=2), encoding="utf-8")

    # 验证文档（保留矫正后模型的验证留痕）
    val_src = PKG / "validation"
    for name in ["A1_LORO_REPORT.md", "B1_DIAGNOSTICS.md", "FITTING_EFFECT.md",
                 "data_difference_report.md"]:
        p = val_src / name
        dst = PKG / "validation" / name
        if p.exists() and p.resolve() != dst.resolve():
            shutil.copy2(p, dst)

    # requirements / gitignore
    (PKG / "requirements.txt").write_text("numpy>=1.24\npandas>=2.0\nscipy>=1.10\nmatplotlib>=3.7\nopenpyxl>=3.1\n", encoding="utf-8")
    (PKG / ".gitignore").write_text("__pycache__/\n*.pyc\n.DS_Store\n*.zip\n", encoding="utf-8")

    # 根 README
    readme = """# Chemical Skull Modification & Stimulation Model — 矫正后模型仓库

> 本仓库已按 2026-11 的口径重组：**只保留矫正后的模型**
> `TrackA`（A1 / A2 / A3，点预测）与 `TrackB`（校准预测区间）。
> 原 `TrackA1` 已统一命名为 `TrackA`；**未矫正的纯机理模型、原始 LHS 探索结果、
> 未校准的等效/多频校准结果已从仓库移除**。机理先验 CSV 仅作为 TrackA 的输入数据保留，
> 不再作为独立"模型"呈现。

## 目录

| 路径 | 内容 |
|---|---|
| `TrackA/A1/` | 经验线性校正 `R_Z = a + b*R_Z_raw` |
| `TrackA/A2/` | 协变量扩展 `+ c*t + d*ln(f/100)`（拟合程度与模型效果最平衡） |
| `TrackA/A3/` | 机理融合混合式 `1 - kappa*lambda(f)*delta - eps*t`（基线自动 =1，参数更少） |
| `TrackB/` | 校准后的 LOO 预测区间（392 点全频 + 196 点低频交付版）与切片统计 |
| `four_conditions/` | 四条件拟合脚本、结果、表1 与四张图、`FOUR_CONDITIONS_CONCLUSION.md` |
| `docs/MECHANISM_AND_EMPIRICAL_PRINCIPLE.md` | **机理 + 经验下的机理原理** |
| `data/` | 冻结实测数据、机理先验分位数、三层数据核验报告 |
| `validation/` | 矫正后模型的 LORO / B1 诊断 / 拟合效果留痕 |

## 四条件拟合（本次新增）

| 条件 | 剂量 | 频率 | 节点 |
|---|---|---|---:|
| C1 全部 | 0.05/0.10/0.15/0.20 M | 100/120/1000/10000 Hz | 96 |
| C2 低频 | 4 剂量 | 100/120 Hz | 48 |
| C3 高浓度 | 0.15/0.20 M | 4 频率 | 48 |
| C4 低频 + 高浓度 | 0.15/0.20 M | 100/120 Hz | 24 |

**结论摘要**：条件越窄、指标越好看，是评价范围变化，不是模型变好。
按 LORO 留一大鼠口径，**REF（拉伸指数）四条件全部第一**，但脱离机理、不可外推，只作拟合上限对照；
正式系列里 **A2 与 A3 的 LORO 四条件均值几乎相同（约 0.596），统计不可区分** ——
A2 为默认推荐（样本内拟合略好、协变量可解释），A3 为结构优先备选（参数更少、基线自动 =1）。
详见 `four_conditions/FOUR_CONDITIONS_CONCLUSION.md` 与 `four_conditions/figures/fig4_table1.png`。

## 口径声明（务必阅读）

1. **A1/A2/A3 都是经验或半经验校正，不是机理改进或机理被证实**。
   原始机理链路 `sigma = sigma_floor + c*phi^m`、`R = sum dx/(sigma*A)`、`R_Z_raw = Z/Z0` 从未改动。
2. 验证层级必须区分：`full`（样本内）< `LOCO`（群体条件均值层）< **`LORO` 留一大鼠（个体级独立验证）**。
   LORO 才是验收口径；四个条件的评价集与 LORO 折数不同（C1/C2 为 24 折，C3/C4 为 12 折），**R² 不可横向比大小**。
3. **TrackB 只报 coverage / width / Winkler，不报 R² / RMSE**；逐点覆盖率不等于整条曲线联合覆盖概率。
   区间宽度约 97% 来自 conformal padding，而非 LHS 参数不确定性。
4. **拟合变好 != 模型变好**：条件收窄、增加自由度、补协变量都会抬高 R²。
5. 基线 t=0 按物理定义 R_Z=1；A1/A2 需锚定，A3 由构造自动满足。
6. 数据：修订版《体外EDTA数据记录汇总.xlsx》SHA `4c165298…`，三层核验见 `data/data_verification.json`。
   保留全部 24 鼠，**未删除 WT17 / WT9 / WT8 等不利样本**；14 处 urea_10/Midpoint/WT21–24 转录异常标记未决。

## OPEN ISSUES

- 尿素 t=105 回升：所有单调模型无法表达；机理先验在 95 min 后截断为平台外推。
- 机理先验频率无关，且改性量幅度显著小于实测；真正修复需在机理层重标定 phi(t) 与补频散项（宽频 EIS）。
- `s` 不是被辨识的 CPE 指数（仅 4 个频率点，函数形式不可区分）。
- TrackB：验证集 22/24 鼠（缺 WT5 0.10 M、WT6 0.15 M）；0.05 M 存在 role x 剂量混淆；上游机制选择非严格嵌套。
- 本地 Excel SHA 与上游 manifest 声明的 source_data SHA 不一致，上游原始文件不可得。

## 复现

```bash
pip install -r requirements.txt
python four_conditions/run_four_conditions.py
```

每个 `TrackA/*/model.py` 可独立 import 并调用 `predict(...)`；`TrackB/model.py` 提供
`load_intervals()` / `slice_metrics()` / `calibrated_from_raw()`。
"""
    (PKG / "README.md").write_text(readme, encoding="utf-8")

    (PKG / "TrackA" / "README.md").write_text("""# TrackA — 矫正后的点预测模型系列

原 `TrackA1` 已重命名归入本目录。三者均**保留原始机理先验链路**，只在外面接校准层
（A1/A2）或把频率维度写回机理层（A3）。

| 模型 | 公式 | k | 性质 |
|---|---|---:|---|
| A1 | `R_Z = a + b*R_Z_raw` | 2 | 纯经验幅度校正 |
| A2 | `R_Z = a + b*R_Z_raw + c*t + d*ln(f/100)` | 4 | 半经验；协变量补机理缺失的 t 与 f |
| A3 | `R_Z = 1 - kappa*(f/100)^(-s)*(1-R_Z_raw) - eps*t` | 3 | 机理融合；基线自动 =1 |

**A2 是拟合程度与模型效果最平衡的选择**；A3 与之 LORO 差距在噪声级，但参数更少、基线自洽、
频率项有物理出处。三者都不构成"机理被证实"。四条件指标见 `../four_conditions/`。
""", encoding="utf-8")
    (PKG / "TrackB" / "README.md").write_text("""# TrackB — 校准后的预测区间

机理 ensemble (LHS) 给出 `sim_q05/q50/q95`，折 x 频率分层 conformal 校准给出 padding：

```
p_(fold,f) = tolerance_f * q_f,  q_f = Quantile_0.90(内层 cross-fit 分数)
calibrated_lower = clip(sim_q05 - p, 0, 1)
calibrated_upper = clip(sim_q95 + p, 0, 1)
```

- `calibrated_intervals_lowfreq_196.csv`：100/120 Hz 交付版（22 鼠 x 2 频率 x 5 时间点）。
- `calibrated_intervals_allfreq_392.csv`：4 频率全量，用于四条件切片统计。
- **只报 coverage / width / Winkler，不报 R² / RMSE**。
- B1 诊断：约 97% 的最终宽度来自 padding，基础 LHS 区间仅约 3%。
- 已知：逐点覆盖率 != 整曲线联合覆盖；基线(t=0) 与尿素(t=105) 无区间；验证集 22/24 鼠。
""", encoding="utf-8")

    print("GitHub 包已生成:", PKG)


if __name__ == "__main__":
    main()
