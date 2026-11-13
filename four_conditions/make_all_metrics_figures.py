#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""全指标实算图集 fig13 / fig14 / fig15（读落盘 CSV，不重新计算、不改模型）。

硬性要求：CJK 字体无缺字、柱状/直方图纵轴自 0 起、图例与注释不遮挡数据、
5 个模型与 4 个条件全部画全。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
PKG = HERE / "github_package" if (HERE / "github_package").exists() else HERE.parent
SRC = PKG / "four_conditions"
FIG = SRC / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

M = pd.read_csv(SRC / "ALL_METRICS_FULL.csv")
R = pd.read_csv(SRC / "RESIDUALS_LONG.csv")
T = pd.read_csv(SRC / "TRACKB_FULL_METRICS.csv")
S = pd.read_csv(SRC / "STANDARDS_COMPUTED.csv")
B10 = pd.read_csv(SRC / "B10_DROP_UREA.csv")

MODELS = ["A1.1", "A1.2", "A1.3", "A2", "A3"]
CONDS = ["C1_all", "C2_lowfreq", "C3_highdose", "C4_lf_hd"]
SHORT = {"C1_all": "C1 全部", "C2_lowfreq": "C2 低频",
         "C3_highdose": "C3 高浓度", "C4_lf_hd": "C4 低频+高浓度"}
MC = {"A1.1": "#8C8C8C", "A1.2": "#4C72B0", "A1.3": "#8172B3",
      "A2": "#C44E52", "A3": "#2F7A4F"}


def _bar(ax, x, heights, colors, ylabel, title, ymax=None, errs=None, hline=None):
    bars = ax.bar(x, heights, color=colors, edgecolor="black", linewidth=0.6, width=0.78)
    if errs is not None:
        ax.errorbar(x, heights, yerr=errs, fmt="none", ecolor="black",
                    capsize=3, elinewidth=1.0)
    for b, v in zip(bars, heights):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:.3f}", ha="center",
                va="bottom", fontsize=7.2)
    ax.set_ylim(0, ymax if ymax else max(heights) * 1.28)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10, pad=8)
    ax.grid(axis="y", ls=":", alpha=0.5)
    ax.set_axisbelow(True)
    if hline is not None:
        ax.axhline(hline, ls="--", color="#B22222", lw=1.2)
        # 放在最后两根柱之间的空隙，避免压住柱顶数值标签
        ax.text(len(x) - 1.5, hline * 1.015, f"目标 {hline:.2f}", color="#B22222",
                fontsize=8, ha="center", va="bottom",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.88))


# ============================================================ fig13 全指标全景
def fig13():
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.6))
    loro = M[M.level == "LORO"]

    # (a) LORO R² 分组柱（4 条件 × 5 模型全画）
    ax = axes[0, 0]
    w, xs = 0.165, np.arange(4)
    for j, name in enumerate(MODELS):
        vals = [loro[(loro.condition == c) & (loro.model == name)].r2.iloc[0] for c in CONDS]
        ax.bar(xs + (j - 2) * w, vals, width=w, color=MC[name], label=name,
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.5)
    ax.set_ylim(0, 1.05); ax.set_ylabel("LORO R²", fontsize=9)
    ax.set_title("(a) 验收口径 LORO R²：4 条件 × 5 模型全部画全", fontsize=10)
    ax.legend(ncol=5, fontsize=8, loc="upper left", framealpha=0.9)
    ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)

    # (b) LORO RMSE
    ax = axes[0, 1]
    vals_all = []
    for j, name in enumerate(MODELS):
        vals = [loro[(loro.condition == c) & (loro.model == name)].rmse.iloc[0] for c in CONDS]
        vals_all += vals
        ax.bar(xs + (j - 2) * w, vals, width=w, color=MC[name], label=name,
               edgecolor="black", linewidth=0.5)
    ax.set_xticks(xs); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.5)
    ax.set_ylim(0, max(vals_all) * 1.30); ax.set_ylabel("LORO RMSE", fontsize=9)
    ax.set_title("(b) LORO RMSE（越小越好）", fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)

    # (c) C1 LORO 多指标归一化对比（误差类取反，越大越好）
    ax = axes[1, 0]
    c1 = loro[loro.condition == "C1_all"].set_index("model")
    keys = ["r2", "adj_r2", "ccc_lin", "willmott_d", "calib_slope", "pearson_r",
            "explained_variance", "mae", "rmse", "mape", "theil_u2"]
    labels = ["R²", "adj R²", "CCC", "Willmott d", "校准斜率", "Pearson r",
              "解释方差", "MAE", "RMSE", "MAPE", "Theil U2"]
    err_like = {"mae", "rmse", "mape", "theil_u2"}
    W = np.arange(len(keys))
    bw = 0.16
    for j, name in enumerate(MODELS):
        norm = []
        for k in keys:
            v = float(c1.loc[name, k])
            norm.append(1 - v if k in err_like else v)
        ax.bar(W + (j - 2) * bw, norm, width=bw, color=MC[name], label=name,
               edgecolor="black", linewidth=0.4)
    ax.set_xticks(W); ax.set_xticklabels(labels, rotation=38, ha="right", fontsize=7.6)
    ax.set_ylim(0, 1.30); ax.set_ylabel("归一化值（误差类取 1−x，均越大越好）", fontsize=8.5)
    ax.set_title("(c) C1 LORO 全指标画像：11 个指标 × 5 模型", fontsize=10)
    ax.legend(ncol=5, fontsize=7.6, loc="upper right", framealpha=0.92)
    ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)

    # (d) A3 LORO R² 与按鼠聚类 bootstrap 95% CI
    ax = axes[1, 1]
    a3 = loro[loro.model == "A3"].set_index("condition").loc[CONDS]
    lo = a3.r2 - a3.r2_boot_lo
    hi = a3.r2_boot_hi - a3.r2
    _bar(ax, np.arange(4), a3.r2.to_numpy(), ["#2F7A4F"] * 4,
         "A3 LORO R²", "(d) A3 LORO R² 及按鼠聚类 bootstrap 95% CI",
         ymax=1.12, errs=np.c_[lo.to_numpy(), hi.to_numpy()].T)
    ax.set_xticks(range(4)); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.5)

    fig.suptitle("全评价标准实算 · 指标全景（TrackA 五模型 × 四条件 × 三验证层级）",
                 fontsize=13, y=0.995)
    fig.text(0.5, 0.012,
             "口径：E1 四条件不可横比（评价集 96/48/48/24、折数 24/24/12/12）；"
             "E5/E6 bootstrap CI 平均半宽 0.121 → 噪声级差异不判胜负",
             ha="center", fontsize=8, color="#B22222")
    fig.tight_layout(rect=(0, 0.028, 1, 0.975))
    p = FIG / "fig13_all_metrics_overview.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(" ", p.name)


# ============================================================ fig14 残差诊断
def fig14():
    d = R[(R.condition == "C1_all") & (R.level == "LORO") & (R.model == "A3")].copy()
    e = d.residual.to_numpy(float)
    yhat = d.y_pred.to_numpy(float)

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))

    # (a) 残差 vs 拟合值
    ax = axes[0, 0]
    for name in MODELS:
        g = R[(R.condition == "C1_all") & (R.level == "LORO") & (R.model == name)]
        ax.scatter(g.y_pred, g.residual, s=9, alpha=0.42, color=MC[name],
                   edgecolors="none", label=name)
    ax.axhline(0, color="black", lw=1.1)
    ax.set_xlabel("预测值 R_Z", fontsize=9); ax.set_ylabel("残差 e = 观测 − 预测", fontsize=9)
    ax.set_title("(a) 残差 vs 拟合值：5 模型 576 鼠级点全画（C1 LORO）", fontsize=10)
    ax.legend(ncol=5, fontsize=8, loc="upper right", framealpha=0.9)
    ax.grid(ls=":", alpha=0.5); ax.set_axisbelow(True)

    # (b) 直方图 + 正态拟合（纵轴自 0 起）
    ax = axes[0, 1]
    n, bins, _ = ax.hist(e, bins=38, color="#4C72B0", edgecolor="white",
                         linewidth=0.5, density=True, alpha=0.85, label="A3 残差")
    xs = np.linspace(e.min(), e.max(), 300)
    ax.plot(xs, stats.norm.pdf(xs, e.mean(), e.std(ddof=1)), color="#B22222",
            lw=2, label="同均值/标准差正态")
    ax.set_ylim(0, ax.get_ylim()[1] * 1.18)
    ax.set_xlabel("残差 e", fontsize=9); ax.set_ylabel("密度", fontsize=9)
    m = M[(M.condition == "C1_all") & (M.level == "LORO") & (M.model == "A3")].iloc[0]
    ax.set_title(f'(b) 残差分布：Shapiro p={m.shapiro_p:.2e}，偏度={m.resid_skew:.3f}，'
                 f'峰度={m.resid_kurt_excess:.3f}', fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)

    # (c) Q-Q 图
    ax = axes[1, 0]
    (osm, osr), (slope, inter, rr) = stats.probplot(e, dist="norm")
    ax.scatter(osm, osr, s=10, alpha=0.55, color="#8172B3", edgecolors="none")
    ax.plot(osm, slope * osm + inter, color="#B22222", lw=1.8, label=f"拟合直线 R²={rr**2:.4f}")
    lim = [min(osm.min(), osr.min()), max(osm.max(), osr.max())]
    ax.plot(lim, lim, color="black", ls="--", lw=1.0, label="y=x（正态）")
    ax.set_xlabel("理论分位数", fontsize=9); ax.set_ylabel("样本分位数", fontsize=9)
    ax.set_title("(c) 正态 Q-Q：尾部偏离 → 拒绝正态性假设", fontsize=10)
    ax.legend(fontsize=8); ax.grid(ls=":", alpha=0.5); ax.set_axisbelow(True)

    # (d) 按时间分组残差（含 t=105 尿素）
    ax = axes[1, 1]
    ts = sorted(d.t_min.unique())
    for j, name in enumerate(MODELS):
        means, sds = [], []
        for t in ts:
            g = R[(R.condition == "C1_all") & (R.level == "LORO") &
                  (R.model == name) & (R.t_min == t)]
            means.append(g.residual.mean()); sds.append(g.residual.std(ddof=1))
        ax.errorbar(np.arange(len(ts)) + (j - 2) * 0.14, means, yerr=sds,
                    fmt="o-", color=MC[name], label=name, capsize=2,
                    markersize=4, lw=1.2, elinewidth=0.9, alpha=0.9)
    ax.axhline(0, color="black", lw=1.0)
    ax.axvspan(len(ts) - 1.4, len(ts) - 0.6, color="#FFD966", alpha=0.35)
    ax.set_ylim(-0.34, 0.62)
    ax.text(0.015, 0.96, "黄色区：尿素 t=105 回升，单调模型无法表达",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.0, color="#8B6508",
            bbox=dict(boxstyle="round,pad=0.25", fc="#FFF8DC", ec="none", alpha=0.9))
    ax.set_xticks(range(len(ts))); ax.set_xticklabels([f"{int(t)}" for t in ts], fontsize=8.5)
    ax.set_xlabel("全局时间 (min)", fontsize=9)
    ax.set_ylabel("残差均值 ± SD", fontsize=9)
    ax.set_title("(d) 残差随时间的系统结构（5 模型全画）", fontsize=10)
    ax.legend(ncol=5, fontsize=8, loc="lower left", framealpha=0.9)
    ax.grid(ls=":", alpha=0.5); ax.set_axisbelow(True)

    fig.suptitle("全评价标准实算 · 残差诊断（逐点残差 8640 行，验收层 C1 LORO）",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    p = FIG / "fig14_residual_diagnostics.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(" ", p.name)


# ============================================================ fig15 TrackB + 标准回填
def fig15():
    t4 = T[T.condition.isin(CONDS)].set_index("condition").loc[CONDS]
    fig, axes = plt.subplots(2, 2, figsize=(13.8, 9.4))
    xl = np.arange(4)

    # (a) coverage + Wilson CI
    ax = axes[0, 0]
    cov = t4.coverage.to_numpy()
    lo = cov - t4.coverage_wilson_lo.to_numpy()
    hi = t4.coverage_wilson_hi.to_numpy() - cov
    _bar(ax, xl, cov, ["#4C72B0"] * 4, "逐点覆盖率 PICP",
         "(a) TrackB 覆盖率 + Wilson 95% CI（验收口径）", ymax=1.12,
         errs=np.c_[lo, hi].T, hline=0.90)
    ax.set_xticks(xl); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.2)

    # (b) 平均宽度 / Winkler（两组柱全部标数值）
    ax = axes[0, 1]
    wmax = max(t4.winkler_mean.max(), t4.mean_width.max())
    b1 = ax.bar(xl - 0.19, t4.mean_width.to_numpy(), width=0.38, color="#8172B3",
                edgecolor="black", linewidth=0.6, label="平均宽度")
    b2 = ax.bar(xl + 0.19, t4.winkler_mean.to_numpy(), width=0.38, color="#C44E52",
                edgecolor="black", linewidth=0.6, label="Winkler 均值")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height(),
                    f"{b.get_height():.3f}", ha="center", va="bottom", fontsize=7.2)
    ax.set_ylim(0, wmax * 1.30)
    ax.set_ylabel("平均宽度 / Winkler", fontsize=9)
    ax.set_title("(b) 区间宽度与 Winkler 区间评分（越小越好）", fontsize=10)
    ax.set_xticks(xl); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.2)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)

    # (c) 45+1 条标准按类别实算覆盖
    ax = axes[1, 0]
    cats = ["A", "B", "C", "D", "E", "扩展"]
    quant = [int(((S.cls == c) & (S.result_type == "定量")).sum()) for c in cats]
    other = [int(((S.cls == c) & (S.result_type != "定量")).sum()) for c in cats]
    ax.bar(cats, quant, color="#2F7A4F", edgecolor="black", linewidth=0.6, label="定量实算值")
    ax.bar(cats, other, bottom=quant, color="#DD8452", edgecolor="black",
           linewidth=0.6, label="判定 / 铁律口径")
    for i, (q, o) in enumerate(zip(quant, other)):
        ax.text(i, q + o + 0.3, f"{q+o}", ha="center", fontsize=8.5)
    ax.set_ylim(0, max(np.array(quant) + np.array(other)) * 1.25)
    ax.set_ylabel("标准条数", fontsize=9)
    ax.set_title(f"(c) 评价标准逐条回填：{len(S)} 条全部有实算值/判定", fontsize=10)
    ax.legend(fontsize=8); ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)

    # (d) 宽度构成 padding vs LHS + B10 稳健性（文字放右侧空白，不压柱）
    ax = axes[1, 1]
    _bar(ax, [0, 1], [97.09, 3.00], ["#C44E52", "#4C72B0"], "%",
         "(d) 区间宽度构成：宽区间根因是 padding，不是 LHS",
         ymax=118)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["conformal padding", "基础 LHS 区间"], fontsize=8.5)
    flips = pd.read_csv(SRC / "B10_RANK_FLIPS.csv")
    ax.text(0.98, 0.93,
            f'B10 去尿素稳健性：{int(flips.rank_flipped.sum())}/{len(flips)} 条件 LORO 第一名翻转\n'
            f'（剔除 t=105 后四条件最优模型仍为 A3）',
            transform=ax.transAxes, ha="right", va="top", fontsize=8.0,
            bbox=dict(boxstyle="round,pad=0.4", fc="#F0F7F0", ec="#2F7A4F", alpha=0.95))
    ax.text(0.98, 0.70, "TrackB 铁律：R²/RMSE 仅作区间中心点诊断，\n不作验收、不与 TrackA 点预测横比",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.0, color="#B22222",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.88))
    ax.text(0.98, 0.50, "逐点覆盖 ≠ 整曲线联合覆盖（C7）",
            transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color="#B22222")

    fig.suptitle("全评价标准实算 · TrackB 区间验收 + 45 条标准回填 + 稳健性",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    p = FIG / "fig15_trackB_and_standards.png"
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(" ", p.name)


if __name__ == "__main__":
    fig13(); fig14(); fig15()
    print("图集完成:", FIG)
