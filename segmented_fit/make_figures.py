#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""分段拟合成图（8 张）

绘图铁律（沿用既有交付约定）：
  * 实测点 + SEM 全部画全（含基线 t=0，6 鼠/节点），不遗漏任何时间点；
  * R_Z 图纵坐标一律从 0 起；
  * 中文字体 NotoSansCJK，禁止缺字方框；
  * 图例 / 注释置于空白区，不遮挡数据；
  * 四条件 R² 不可横比，图上必须标注；
  * 群体均值层 full 与个体级 LORO 明确分开标注。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TAB, FIG = HERE / "tables", HERE / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FP)
plt.rcParams["font.family"] = fm.FontProperties(fname=FP).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130
plt.rcParams["savefig.dpi"] = 160

C_DOSE = {0.05: "#1F77B4", 0.10: "#2CA02C", 0.15: "#FF7F0E", 0.20: "#D62728"}
C_BASE, C_SEG, C_ADJ = "#7A8B99", "#E08A2E", "#2F7A4F"
FREQS = [100, 120, 1000, 10000]
DOSES = [0.05, 0.10, 0.15, 0.20]
TS = [0, 5, 20, 35, 65, 95, 105]
CONDS = ["C1_all", "C2_lowfreq", "C3_highdose", "C4_lf_hd"]
COND_SHORT = {"C1_all": "C1 全部", "C2_lowfreq": "C2 低频",
              "C3_highdose": "C3 高浓度", "C4_lf_hd": "C4 低频+高浓度"}
SCHEMES = ["BASE_A1.1", "BASE_A1.2", "BASE_A1.3", "BASE_A2", "BASE_A3",
           "SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]
SHORT = {"BASE_A1.1": "A1.1", "BASE_A1.2": "A1.2", "BASE_A1.3": "A1.3",
         "BASE_A2": "A2", "BASE_A3": "A3", "SEG_A1": "分段A1", "SEG_A3": "分段A3",
         "ADJ_A1": "调整A1", "ADJ_A3": "调整A3"}


def scheme_color(s):
    if s.startswith("BASE_"):
        return C_BASE
    if s.startswith("SEG_"):
        return C_SEG
    return C_ADJ


def load():
    cv = pd.read_csv(TAB / "SEGMENT_CURVES.csv")
    ov = pd.read_csv(TAB / "COMPARISON_OVERALL.csv")
    sm = pd.read_csv(TAB / "SEGMENT_METRICS.csv")
    va = pd.read_csv(TAB / "A1_VARIANT_SELECTION.csv")
    rb = pd.read_csv(TAB / "UREA_REBOUND.csv")
    fu = pd.read_csv(TAB / "FULL_GROUP_MEAN.csv")
    rs = pd.read_csv(TAB / "RESIDUALS_LONG.csv")
    return cv, ov, sm, va, rb, fu, rs


def obs_points(cv, condition):
    """实测群体均值点 + SEM，含按物理定义锚定的基线 t=0（R_Z=1）。"""
    d = cv[(cv.condition == condition) & (cv.scheme == "BASE_A3")]
    g = (d.groupby(["dose", "freq", "t"])
         .agg(y=("y", "mean"), sem=("sem", "mean")).reset_index())
    base = pd.DataFrame([dict(dose=x, freq=f, t=0, y=1.0, sem=0.0)
                         for x in DOSES if x in g.dose.unique() for f in FREQS if f in g.freq.unique()])
    return pd.concat([base, g], ignore_index=True).sort_values(["dose", "freq", "t"])


def pred_curve(cv, condition, scheme):
    d = cv[(cv.condition == condition) & (cv.scheme == scheme)]
    g = (d.groupby(["dose", "freq", "t"])["y_pred"].mean().reset_index())
    base = pd.DataFrame([dict(dose=x, freq=f, t=0, y_pred=1.0)
                         for x in DOSES if x in g.dose.unique() for f in FREQS if f in g.freq.unique()])
    return pd.concat([base, g], ignore_index=True).sort_values(["dose", "freq", "t"])


def bands(ax, ymax=1.30, annotate=False):
    ax.axvspan(0, 5, color="#FFF2CC", alpha=0.55, zorder=0)
    ax.axvspan(5, 95, color="#EAF3FB", alpha=0.55, zorder=0)
    ax.axvspan(95, 105, color="#FCE4EC", alpha=0.65, zorder=0)
    ax.axvline(5, color="#B0A070", lw=0.8, ls=":", zorder=0)
    ax.axvline(95, color="#B0A070", lw=0.8, ls=":", zorder=0)
    if annotate:
        ax.text(2.5, ymax * 0.965, "乙酸", ha="center", va="top", fontsize=7, color="#8B6508")
        ax.text(50, ymax * 0.965, "EDTA 脱钙段", ha="center", va="top", fontsize=7, color="#37618E")
        ax.text(100, ymax * 0.965, "饱和尿素", ha="center", va="top", fontsize=7, color="#A33A5C")


def finish(fig, path, title, footnote=None, rect=(0, 0, 1, 0.95)):
    fig.suptitle(title, fontsize=13, y=0.985)
    if footnote:
        fig.text(0.5, 0.012, footnote, ha="center", fontsize=8, color="#444444")
        fig.tight_layout(rect=(0, 0.035, 1, rect[3]))
    else:
        fig.tight_layout(rect=rect)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", path.name)


# ------------------------------------------------------------------ fig01
def fig01(cv):
    o = obs_points(cv, "C1_all")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    for ax, f in zip(axes.flat, FREQS):
        bands(ax, annotate=(f == 100))
        for d in DOSES:
            g = o[(o.freq == f) & (o.dose == d)].sort_values("t")
            ax.errorbar(g.t, g.y, yerr=g["sem"], fmt="o", ms=4.2, capsize=2.4,
                        color=C_DOSE[d], ecolor=C_DOSE[d], lw=1.0,
                        label=f"{d:.2f} M", zorder=3)
            ax.plot(g.t, g.y, color=C_DOSE[d], lw=1.1, alpha=0.75, zorder=2)
        ax.set_ylim(0, 1.30)
        ax.set_xlim(-3, 108)
        ax.set_xticks(TS)
        ax.set_xticklabels([str(t) for t in TS], fontsize=8)
        ax.set_xlabel("全局时间 (min)", fontsize=9)
        ax.set_ylabel("R_Z = Z/Z0", fontsize=9)
        ax.set_title(f"{f} Hz · 实测点 + SEM（每节点 6 鼠）", fontsize=10)
        ax.grid(ls=":", alpha=0.45)
        ax.set_axisbelow(True)
        if f == 100:
            ax.legend(ncol=2, fontsize=7.5, loc="lower left", framealpha=0.92)
        ax.annotate("t=105 尿素段回升", xy=(105, 0.267), xytext=(78, 1.12),
                    fontsize=7.6, color="#A33A5C", ha="center",
                    arrowprops=dict(arrowstyle="->", color="#A33A5C", lw=1.0),
                    bbox=dict(boxstyle="round,pad=0.22", fc="white", ec="none", alpha=0.9))
    finish(fig, FIG / "fig01_segments_measured_data.png",
           "三段试剂处理的实测 R_Z 全貌（50% 乙酸 / EDTA / 饱和尿素，含基线 t=0）",
           "数据源：冻结 24 鼠；每段 4 剂量 x 4 频率 x 6 鼠；黄=乙酸段 / 蓝=EDTA 段 / 粉=饱和尿素段；基线 t=0 按定义 R_Z=1。",
           rect=(0, 0, 1, 0.95))


# ------------------------------------------------------------------ fig02
def fig02(va):
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.6))
    for ax, seg in zip(axes, ["S1_acetic", "S2_EDTA", "S3_urea"]):
        d = va[va.seg == seg]
        x = np.arange(len(d))
        cols = [C_ADJ if int(s) == 1 else "#AEB9C2" for s in d.selected]
        ax.bar(x, d.loro_r2, color=cols, edgecolor="black", linewidth=0.6, width=0.62)
        for i, (r, k, sel) in enumerate(zip(d.loro_r2, d.k, d.selected)):
            ax.text(i, r + 0.006, f"{r:.3f}\nk={k}", ha="center", va="bottom",
                    fontsize=7.6, fontweight="bold" if sel else "normal")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{v}\n{'★选中' if s else ''}" for v, s in zip(d.variant, d.selected)],
                           fontsize=7.8)
        ax.set_ylim(0, max(d.loro_r2) * 1.42 + 0.02)
        ax.set_ylabel("段内 LORO R²", fontsize=9)
        ax.set_title(d.seg_label.iloc[0], fontsize=10)
        ax.grid(axis="y", ls=":", alpha=0.5)
        ax.set_axisbelow(True)
    finish(fig, FIG / "fig02_trackA1_variant_selection.png",
           "TrackA1 纯经验：三段各自改变公式，按段内 LORO R² 择优",
           "绿色=该段选中公式；择优仅在 C1 全部条件上进行，属同一数据集内的探索性选择；正式结论以拼接整体 LORO 为准。",
           rect=(0, 0, 1, 0.93))


# ------------------------------------------------------------------ fig03 / fig04
def fit_panels(cv, scheme, baseline, out, title, footnote):
    o = obs_points(cv, "C1_all")
    p = pred_curve(cv, "C1_all", scheme)
    pb = pred_curve(cv, "C1_all", baseline)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4))
    for ax, f in zip(axes.flat, FREQS):
        bands(ax, annotate=(f == 100))
        for d in DOSES:
            g = o[(o.freq == f) & (o.dose == d)].sort_values("t")
            ax.errorbar(g.t, g.y, yerr=g["sem"], fmt="o", ms=3.6, capsize=2.0,
                        color=C_DOSE[d], ecolor=C_DOSE[d], lw=0.9, alpha=0.85, zorder=3)
            q = p[(p.freq == f) & (p.dose == d)].sort_values("t")
            ax.plot(q.t, q.y_pred, color=C_DOSE[d], lw=1.7, zorder=4)
            qb = pb[(pb.freq == f) & (pb.dose == d)].sort_values("t")
            ax.plot(qb.t, qb.y_pred, color="#8A8A8A", lw=1.0, ls="--", alpha=0.85, zorder=2)
        ax.set_ylim(0, 1.30)
        ax.set_xlim(-3, 108)
        ax.set_xticks(TS)
        ax.set_xticklabels([str(t) for t in TS], fontsize=8)
        ax.set_xlabel("全局时间 (min)", fontsize=9)
        ax.set_ylabel("R_Z = Z/Z0", fontsize=9)
        ax.set_title(f"{f} Hz", fontsize=10)
        ax.grid(ls=":", alpha=0.45)
        ax.set_axisbelow(True)
        if f == 100:
            from matplotlib.lines import Line2D
            hd = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#555",
                         markeredgecolor="#555", ms=4, label="实测点 ± SEM"),
                  Line2D([0], [0], color="#333", lw=1.8, label=f"{scheme} 分段预测"),
                  Line2D([0], [0], color="#8A8A8A", lw=1.1, ls="--", label=f"{baseline} 全局单调")]
            ax.legend(handles=hd, fontsize=7.2, loc="lower left", framealpha=0.93)
    finish(fig, FIG / out, title, footnote, rect=(0, 0, 1, 0.95))


# ------------------------------------------------------------------ fig05
def fig05(ov):
    fig, axes = plt.subplots(2, 2, figsize=(13.0, 8.2))
    for ax, cid in zip(axes.flat, CONDS):
        d = ov[ov.condition == cid].set_index("scheme").loc[SCHEMES]
        x = np.arange(len(SCHEMES))
        ax.bar(x, d.r2, color=[scheme_color(s) for s in SCHEMES],
               edgecolor="black", linewidth=0.55, width=0.72)
        for i, v in enumerate(d.r2):
            ax.text(i, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=6.8, rotation=0)
        b = float(d.loc["BASE_A3", "r2"])
        ax.axhline(b, color="#B22222", ls="--", lw=1.1)
        ax.text(0.12, b + 0.030, "全局 A3 基准", color="#B22222",
                fontsize=7.0, ha="left", va="bottom")
        adj = d.loc["ADJ_A3"]
        seg = d.loc["SEG_A3"]
        txt = (f"调整A3 vs A3: Δ={adj.delta_r2_vs_best:+.4f} "
               f"(95%CI [{adj.delta_r2_boot_lo:+.3f},{adj.delta_r2_boot_hi:+.3f}])\n"
               f"独立分段A3 vs A3: Δ={seg.delta_r2_vs_best:+.4f} "
               f"(95%CI [{seg.delta_r2_boot_lo:+.3f},{seg.delta_r2_boot_hi:+.3f}])")
        ax.text(0.015, 0.965, txt, transform=ax.transAxes, ha="left", va="top", fontsize=7.0,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#CCCCCC", alpha=0.94))
        ax.set_xticks(x)
        ax.set_xticklabels([SHORT[s] for s in SCHEMES], rotation=38, ha="right", fontsize=7.6)
        ax.set_ylim(0, 1.05)
        ax.set_ylabel("拼接整体 LORO R²", fontsize=9)
        ax.set_title(COND_SHORT[cid], fontsize=10)
        ax.grid(axis="y", ls=":", alpha=0.45)
        ax.set_axisbelow(True)
    finish(fig, FIG / "fig05_loro_comparison.png",
           "四条件 x 9 方案：拼接整体 LORO R²（唯一验收口径）",
           "灰=冻结全局单调模型 / 橙=三段独立标定 / 绿=骨架保留+最少分段调整；Δ 的 95%CI 为按鼠聚类配对 bootstrap。E1：四条件评价集与折数不同，R² 不可横向解读为模型优劣。",
           rect=(0, 0, 1, 0.95))


# ------------------------------------------------------------------ fig06
def fig06(rb):
    d = rb[(rb.condition == "C1_all")].copy()
    agg = (d.groupby("scheme")
           .agg(d_pred=("d_pred", "mean"), sign=("sign_match", "mean"))
           .reindex(SCHEMES))
    obs = float(d.d_obs.mean())
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0),
                             gridspec_kw=dict(width_ratios=[1.45, 1]))
    ax = axes[0]
    nodes = d[d.scheme == "BASE_A3"].sort_values(["dose", "freq"])
    lab = [f"{a:.2f}M\n{int(b)}Hz" for a, b in zip(nodes.dose, nodes.freq)]
    x = np.arange(len(nodes))
    ax.bar(x, nodes.d_obs, color="#4C72B0", edgecolor="black", linewidth=0.5,
           width=0.62, label="实测 ΔR_Z(105-95)")
    for s, off, c in [("BASE_A3", -0.20, C_BASE), ("SEG_A3", 0.0, C_SEG), ("ADJ_A3", 0.20, C_ADJ)]:
        v = d[d.scheme == s].sort_values(["dose", "freq"]).d_pred.to_numpy()
        ax.bar(x + off, v, width=0.19, color=c, edgecolor="black", linewidth=0.4,
               label=f"{SHORT[s]} 预测 Δ")
    ax.axhline(0, color="black", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(lab, fontsize=6.6)
    ax.set_ylabel("ΔR_Z = R_Z(105) − R_Z(95)", fontsize=9)
    ax.set_title("(a) 16 个剂量×频率节点的回升量", fontsize=10)
    ax.legend(ncol=2, fontsize=7.3, loc="upper right", framealpha=0.93)
    ax.grid(axis="y", ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    ax.text(0.012, 0.02, "Δ 可为负；0 线以上=回升，以下=继续下降",
            transform=ax.transAxes, fontsize=7.2, color="#555", ha="left", va="bottom")
    ax = axes[1]
    xx = np.arange(len(SCHEMES))
    ax.bar(xx, agg.d_pred, color=[scheme_color(s) for s in SCHEMES],
           edgecolor="black", linewidth=0.55, width=0.66)
    ax.axhline(obs, color="#4C72B0", ls="--", lw=1.3)
    ax.text(len(SCHEMES) - 0.3, obs + 0.002, f"实测均值 {obs:.3f}", color="#4C72B0",
            fontsize=7.6, ha="right", va="bottom")
    ax.axhline(0, color="black", lw=0.9)
    for i, (v, sr) in enumerate(zip(agg.d_pred, agg.sign)):
        ax.text(i, v + (0.002 if v >= 0 else -0.004),
                f"{v:+.3f}\n{sr*100:.0f}%", ha="center",
                va="bottom" if v >= 0 else "top", fontsize=6.7)
    ylo = min(agg.d_pred.min() * 1.6, -0.04)
    yhi = max(obs, agg.d_pred.max()) * 1.45
    ax.set_ylim(ylo, yhi)
    ax.set_xticks(xx)
    ax.set_xticklabels([SHORT[s] for s in SCHEMES], rotation=42, ha="right", fontsize=7.4)
    ax.set_ylabel("平均 ΔR_Z", fontsize=9)
    ax.set_title("(b) 平均回升量与符号一致率（16 节点）", fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    finish(fig, FIG / "fig06_urea_rebound.png",
           "饱和尿素段回升：分段模型能否表达单调模型表达不了的定性行为",
           "实测 Δ=+0.060（R_Z 由 0.207 回升至 0.267）。5 个全局单调模型全部预测继续下降（符号一致率 0/16）；所有分段方案符号一致率 16/16。独立分段幅度最准，骨架调整偏保守。",
           rect=(0, 0, 1, 0.93))


# ------------------------------------------------------------------ fig07
def fig07(ov, fu):
    o = ov[ov.condition == "C1_all"].set_index("scheme").loc[SCHEMES]
    f = fu[fu.condition == "C1_all"].set_index("scheme").loc[SCHEMES]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2))
    ax = axes[0]
    x = np.arange(len(SCHEMES))
    ax.bar(x - 0.20, f.r2, width=0.38, color="#8172B3", edgecolor="black",
           linewidth=0.55, label="full：群体均值层 in-sample（仅参考）")
    ax.bar(x + 0.20, o.r2, width=0.38, color="#4C72B0", edgecolor="black",
           linewidth=0.55, label="LORO：个体级留出（唯一验收）")
    for i, (a, b) in enumerate(zip(f.r2, o.r2)):
        ax.text(i - 0.20, a + 0.008, f"{a:.3f}", ha="center", va="bottom", fontsize=6.6)
        ax.text(i + 0.20, b + 0.008, f"{b:.3f}", ha="center", va="bottom", fontsize=6.6)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[s] for s in SCHEMES], rotation=38, ha="right", fontsize=7.6)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("R²", fontsize=9)
    ax.set_title("(a) 两种口径的 R² 分离（C1）", fontsize=10)
    ax.legend(fontsize=7.5, loc="lower right", framealpha=0.93)
    ax.grid(axis="y", ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    ax = axes[1]
    OFF = {"BASE_A1.1": (-16, -15), "BASE_A1.2": (5, -13), "BASE_A1.3": (5, 7),
           "BASE_A2": (5, 11), "BASE_A3": (5, 7),
           "SEG_A1": (6, 12), "SEG_A3": (6, -14),
           "ADJ_A1": (6, -14), "ADJ_A3": (6, 8)}
    for s in SCHEMES:
        ax.scatter(f.loc[s, "r2"], o.loc[s, "r2"], s=70, color=scheme_color(s),
                   edgecolor="black", linewidth=0.6, zorder=3)
        dx, dy = OFF[s]
        ax.annotate(SHORT[s], (f.loc[s, "r2"], o.loc[s, "r2"]),
                    textcoords="offset points", xytext=(dx, dy), fontsize=7.2,
                    ha="right" if dx < 0 else "left")
    b = float(o.loc["BASE_A3", "r2"])
    ax.axhline(b, color="#B22222", ls="--", lw=1.1)
    ax.text(0.858, b + 0.0015, "全局 A3 LORO 基准", color="#B22222",
            fontsize=7.2, ha="left", va="bottom")
    ax.set_xlim(0.85, 0.98)
    ax.set_ylim(0.45, 0.56)
    ax.set_xlabel("full 群体均值层 R²（仅参考）", fontsize=9)
    ax.set_ylabel("LORO 个体级 R²（验收口径）", fontsize=9)
    ax.set_title("(b) full 更高 ≠ 个体级更好", fontsize=10)
    ax.grid(ls=":", alpha=0.45)
    ax.set_axisbelow(True)
    ax.text(0.02, 0.03,
            "独立分段 A1/A3：full 0.957/0.954（最高）\n但 LORO 0.496/0.494（显著低于 A3 0.520）",
            transform=ax.transAxes, fontsize=7.6, ha="left", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", fc="#FFF6E8", ec="#E08A2E", alpha=0.95))
    finish(fig, FIG / "fig07_full_vs_loro.png",
           "拟合效果是否更好，取决于用哪一层口径：群体均值层 vs 个体级留出",
           "铁律：full 为群体均值 in-sample，不能与个体验证混淆；LORO 才是唯一验收口径。分段消除了系统偏差（bias），但 LORO 误差主要由鼠间异质性主导，故 full 提升未能转化为 LORO 提升。",
           rect=(0, 0, 1, 0.93))


# ------------------------------------------------------------------ fig08
def fig08(rs, sm):
    d = rs[(rs.condition == "C1_all") & (rs.scheme.isin(["BASE_A3", "SEG_A3", "ADJ_A3"]))]
    fig, axes = plt.subplots(1, 3, figsize=(13.4, 4.5))
    ax = axes[0]
    bins = np.linspace(-0.55, 0.75, 34)
    for s, c in [("BASE_A3", C_BASE), ("SEG_A3", C_SEG), ("ADJ_A3", C_ADJ)]:
        e = d[d.scheme == s].residual.to_numpy(float)
        ax.hist(e, bins=bins, density=True, histtype="step", lw=1.5, color=c, label=SHORT[s])
    ax.axvline(0, color="black", lw=0.9)
    ax.set_xlabel("残差 e = 观测 − 预测", fontsize=9)
    ax.set_ylabel("密度", fontsize=9)
    ax.set_title("(a) 残差分布（三方案）", fontsize=10)
    ax.legend(fontsize=7.6, framealpha=0.93)
    ax.grid(ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax = axes[1]
    for s, c, mk in [("BASE_A3", C_BASE, "o"), ("SEG_A3", C_SEG, "s"), ("ADJ_A3", C_ADJ, "^")]:
        g = d[d.scheme == s].groupby("t")["residual"].agg(["mean", "std"]).reset_index()
        ax.errorbar(g.t, g["mean"], yerr=g["std"], fmt=mk + "-", ms=4.0, capsize=2.0,
                    lw=1.2, color=c, label=SHORT[s], alpha=0.9)
    ax.axhline(0, color="black", lw=0.9)
    ax.axvspan(103, 107, color="#FCE4EC", alpha=0.7)
    ax.set_xticks([5, 20, 35, 65, 95, 105])
    ax.set_xlabel("全局时间 (min)", fontsize=9)
    ax.set_ylabel("残差均值 ± SD", fontsize=9)
    ax.set_title("(b) 残差随时间的系统结构", fontsize=10)
    ax.legend(fontsize=7.6, loc="upper right", framealpha=0.93)
    ax.grid(ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    ax = axes[2]
    segs = ["S1_acetic", "S2_EDTA", "S3_urea"]
    x = np.arange(3)
    w = 0.26
    for i, (s, c) in enumerate([("BASE_A3", C_BASE), ("SEG_A3", C_SEG), ("ADJ_A3", C_ADJ)]):
        q = sm[(sm.condition == "C1_all") & (sm.scheme == s)].set_index("seg").loc[segs]
        vals = q.bias.to_numpy(float)
        ax.bar(x + (i - 1) * w, vals, width=w, color=c, edgecolor="black",
               linewidth=0.5, label=SHORT[s])
        for j, v in enumerate(vals):
            ax.text(x[j] + (i - 1) * w, v + (0.002 if v >= 0 else -0.002),
                    f"{v:+.3f}", ha="center", va="bottom" if v >= 0 else "top",
                    fontsize=6.4, rotation=0)
    ax.axhline(0, color="black", lw=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(["乙酸段", "EDTA 段", "尿素段"], fontsize=8.5)
    ax.set_ylabel("段内 bias（观测−预测）", fontsize=9)
    ax.set_title("(c) 分段后系统偏差被消除", fontsize=10)
    lo = float(sm[(sm.condition == "C1_all")].bias.min()) * 1.5
    hi = float(sm[(sm.condition == "C1_all")].bias.max()) * 1.5
    ax.set_ylim(lo, hi)
    ax.legend(fontsize=7.6, framealpha=0.93)
    ax.grid(axis="y", ls=":", alpha=0.4)
    ax.set_axisbelow(True)
    finish(fig, FIG / "fig08_residual_diagnostics.png",
           "残差诊断：分段消除的是系统偏差，不是鼠间异质性",
           "(c) 三段独立标定后 bias≈0；但 (a)(b) 显示残差离散度未同步下降，故 LORO R² 未提升。纵轴为 bias / 残差（含负值），0 线为无偏基准。",
           rect=(0, 0, 1, 0.93))


def main():
    cv, ov, sm, va, rb, fu, rs = load()
    print("生成 8 张图：")
    fig01(cv)
    fig02(va)
    fit_panels(cv, "SEG_A1", "BASE_A1.1", "fig03_fit_SEG_A1.png",
               "TrackA1 纯经验：三段独立标定（SEG_A1）对实测的拟合",
               "实线=SEG_A1 三段独立公式（乙酸经验幂律 / EDTA 双曲幂律 / 尿素独立横截面回归）；灰虚线=全局 A1.1 拉伸指数。基线 t=0 按定义锚定 R_Z=1。")
    fit_panels(cv, "ADJ_A3", "BASE_A3", "fig04_fit_ADJ_A3.png",
               "TrackA3 机理+经验：骨架保留 + 最少分段调整（ADJ_A3）",
               "实线=ADJ_A3（全局 A3 骨架 + 乙酸/EDTA 截距修正 + 尿素界面重构再封闭项 rho）；灰虚线=全局 A3 单调外推。尿素段从 t=95 平台出发，不再延续单调下降。")
    fig05(ov)
    fig06(rb)
    fig07(ov, fu)
    fig08(rs, sm)
    print("完成")


if __name__ == "__main__":
    main()
