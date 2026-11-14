#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""框架与全维度对比图：CJK 字体、纵轴自 0 起、点+SEM 全画、无遮挡。"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parent
TAB = ROOT / "tables"
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)
S = json.loads((ROOT / "SUMMARY.json").read_text(encoding="utf-8"))

FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if Path(FP).exists():
    fm.fontManager.addfont(FP)
    plt.rcParams["font.family"] = fm.FontProperties(fname=FP).get_name()
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

BLUE, GREEN, DGREEN, PURPLE, ORANGE, RED, GREY = (
    "#4C72B0", "#8FBC8F", "#2F7A4F", "#8172B3", "#E39B44", "#B22222", "#8C8C8C")
SHORT = {"C1_all": "C1 所有条件", "C2_lowfreq": "C2 低频",
         "C3_highdose": "C3 高浓度", "C4_lf_hd": "C4 最佳范围"}
CONDS = ["C1_all", "C2_lowfreq", "C3_highdose", "C4_lf_hd"]

MAT = pd.read_csv(TAB / "MATRIX_FITMODE_X_SCOPE.csv")
SCOPE = pd.read_csv(TAB / "ALL_VS_BEST_CONDITION.csv")
TR = pd.read_csv(TAB / "TRACKA_THREE_ROUTES.csv")
GV = pd.read_csv(TAB / "GLOBAL_VS_SEGMENTED.csv")
SEGOV = pd.read_csv(ROOT.parent / "segmented_fit" / "tables" / "COMPARISON_OVERALL.csv")
TB = pd.read_csv(TAB / "TRACKB_INTERVALS.csv")
UR = pd.read_csv(TAB / "UREA_REBOUND_SUMMARY.csv")
GAPS = pd.read_csv(TAB / "FRAMEWORK_GAPS.csv")
CUR = pd.read_csv(ROOT.parent / "segmented_fit" / "tables" / "SEGMENT_CURVES.csv")
DATA = pd.read_csv(ROOT / "data" / "frozen_measured_24rat.csv")


def finish(fig, name, title, sub=None):
    fig.suptitle(title, fontsize=13.5, y=0.985)
    if sub:
        fig.text(0.5, 0.012, sub, ha="center", fontsize=8, color="#444444")
    fig.tight_layout(rect=(0, 0.032 if sub else 0, 1, 0.965))
    fig.savefig(FIG / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  ->", name)


def bar(ax, x, vals, colors, labels=None, ymax=None, ylabel="", title="", hline=None,
        hlabel=None, width=0.72, fontsize=7.6, fmt="%.3f", rot=0):
    x = np.asarray(x, float)
    vals = np.asarray(vals, float)
    colors = list(colors)
    b = ax.bar(x, vals, width=width, color=colors, edgecolor="black", linewidth=0.6, zorder=3)
    for i, (xi, v) in enumerate(zip(x, vals)):
        if np.isnan(v):
            ax.text(xi, 0.02, "未实现", ha="center", va="bottom", fontsize=7,
                    color=RED, rotation=90)
            continue
        if rot:
            ax.text(xi, v + 0.006, fmt % v, ha="center", va="bottom",
                    rotation=rot, fontsize=fontsize, zorder=5)
        else:
            ax.text(xi, v, fmt % v, ha="center", va="bottom", fontsize=fontsize, zorder=5)
    if labels is not None:
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylim(0, ymax if ymax else max(vals[np.isfinite(vals)]) * 1.25)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.set_title(title, fontsize=10)
    ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)
    if hline is not None:
        ax.axhline(hline, ls="--", color=RED, lw=1.3, zorder=2)
        if hlabel:
            ax.text(0.985, 0.965, hlabel, color=RED, fontsize=8,
                    ha="right", va="top", transform=ax.transAxes,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    return b


# ================================================================ fig01 框架总览
fig, ax = plt.subplots(figsize=(13.5, 8.2))
ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")


def box(x, y, w, h, text, fc, ec, fontsize=10, sub=None, alpha=0.22, lw=1.6):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.012",
                       facecolor=fc, edgecolor=ec, linewidth=lw, alpha=alpha)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h * (0.66 if sub else 0.5), text, ha="center", va="center",
            fontsize=fontsize, fontweight="bold", color=ec)
    if sub:
        ax.text(x + w / 2, y + h * 0.28, sub, ha="center", va="center",
                fontsize=8.2, color="#333333")
    return (x, y, w, h)


def arrow(x1, y1, x2, y2, color=GREY):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=1.3,
                                shrinkA=2, shrinkB=2))


ax.text(0.5, 0.965, "顶层设计框架：预测对象 → 建模路线 → 两个对比维度 → 验收口径",
        ha="center", fontsize=14, fontweight="bold")

# 层1：预测对象
bA = box(0.035, 0.805, 0.575, 0.105, "TrackA · 具体阻抗降低值（点预测）", BLUE, BLUE,
         sub="输出 R_Z 的点估计（等价 Δ = 1 − R_Z）\n"
             "生物复杂 + 数据有限 → 三路线对照（灵活性 / 可解释性 / 折中）")
bB = box(0.645, 0.805, 0.320, 0.105, "TrackB · 阻抗降低区间", PURPLE, PURPLE,
         sub="输出 [R_Z^lo, R_Z^hi]，目标覆盖率 0.90")

# 层2：三路线 + TrackB 实现
b1 = box(0.035, 0.605, 0.178, 0.135, "TrackA1", BLUE, BLUE, sub="纯经验\nA1.1 / A1.2 / A1.3")
b2 = box(0.235, 0.605, 0.178, 0.135, "TrackA2", GREEN, DGREEN, sub="纯机理\n4 个物理参数")
b3 = box(0.435, 0.605, 0.175, 0.135, "TrackA3", DGREEN, DGREEN,
         sub="机理驱动经验\nη / ε 保形校准")
bB2 = box(0.645, 0.605, 0.320, 0.135, "LHS 机理 ensemble + conformal padding", PURPLE, PURPLE,
          sub="分位数 q05/q50/q95 + LOO 校准宽度")
for bx in (b1, b2, b3):
    arrow(bx[0] + bx[2] / 2, 0.805, bx[0] + bx[2] / 2, 0.740, color=BLUE)
arrow(0.805, 0.805, 0.805, 0.740, color=PURPLE)

# 层3：两个对比维度
d1 = box(0.035, 0.355, 0.435, 0.165, "对比维度 ①  拟合粒度", ORANGE, "#B87514",
         sub="全局拟合（一套参数 / 6 时间点联合约束）\nvs 分段拟合（乙酸 / EDTA / 饱和尿素三段）\n"
             "分段两代：SEG 独立标定 · ADJ 骨架保留 + 增量")
d2 = box(0.530, 0.355, 0.435, 0.165, "对比维度 ②  条件范围", ORANGE, "#B87514",
         sub="所有条件 C1（4 剂量 × 4 频率，576 点 / 24 鼠 / 24 折）\n"
             "vs 最佳条件范围 C4（0.15/0.20 M × 100/120 Hz，144 点 / 12 鼠 / 12 折）\n"
             "⚠ C4 是 C1 子集，R² 不可横比")
for bx in (b1, b2, b3):
    arrow(bx[0] + bx[2] / 2, 0.605, 0.2525, 0.520, color=GREY)
arrow(0.805, 0.605, 0.7475, 0.520, color=GREY)

# 层4：验收口径
box(0.035, 0.145, 0.930, 0.135, "唯一验收口径", RED, RED,
    sub="TrackA：LORO（Leave-One-Rat-Out，逐鼠 24/12 折）R² / RMSE / 残差诊断 / AICc / BIC　·　"
        "TrackB：coverage / width / Winkler（不报 R²、RMSE 作验收）")
arrow(0.2525, 0.355, 0.35, 0.280, color=GREY)
arrow(0.7475, 0.355, 0.65, 0.280, color=GREY)

# 层5：空格子提示
ax.text(0.5, 0.075,
        "框架空格（如实标注，未填充）：分段维度下 TrackA2 纯机理、TrackB 区间版本尚未实现 —— 共 8 格，见 FRAMEWORK_GAPS.csv",
        ha="center", fontsize=8.6, color=RED,
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF4F4", ec=RED, alpha=0.95))
ax.text(0.5, 0.025, "全部数值来源已冻结成果，跨包一致性 60 项 max|Δ| = 4.8e−07（PASS）；本轮零重拟合、零改模型",
        ha="center", fontsize=8, color="#555555")
finish(fig, "fig01_research_design_framework.png",
       "图 1 · 顶层研究设计框架总览",
       "TrackA 点值 / TrackB 区间 · 三路线 · 全局与分段 · 所有条件与最佳条件范围")

# ================================================================ fig02 3x2 矩阵
fig, axes = plt.subplots(1, 2, figsize=(13.2, 5.6))
MODES = ["全局", "分段独立", "分段增量"]
ROUTES = ["TrackA1", "TrackA2", "TrackA3"]
COLS = [BLUE, GREEN, DGREEN]
for ax, cond in zip(axes, ["C1_all", "C4_lf_hd"]):
    for j, route in enumerate(ROUTES):
        vals = []
        for mode in MODES:
            q = MAT[(MAT.condition == cond) & (MAT.fit_mode == mode) & (MAT.route == route)]
            vals.append(float(q.loro_r2.iloc[0]) if len(q) and not pd.isna(q.loro_r2.iloc[0]) else np.nan)
        xs = np.arange(3) + (j - 1) * 0.26
        b = bar(ax, xs, vals, [COLS[j]] * 3, ymax=1.02,
                ylabel="LORO R²", fmt="%.3f", width=0.25,
                title=f"({chr(97 + list(axes).index(ax))}) {SHORT[cond]} · 三路线 × 三拟合方式" if j == 0 else None)
    ax.set_xticks(np.arange(3)); ax.set_xticklabels(MODES, fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_title(f"({chr(97 + list(axes).index(ax))}) {SHORT[cond]} · 三路线 × 三拟合方式", fontsize=10)
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.6) for c in COLS],
              labels=["TrackA1 纯经验", "TrackA2 纯机理", "TrackA3 机理驱动经验"],
              fontsize=8, loc="upper left", framealpha=0.95)
    ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)
finish(fig, "fig02_matrix_fitmode_x_scope.png",
       "图 2 · 全维度矩阵：拟合方式 × 条件范围 × 三路线（唯一验收口径 LORO R²）",
       "空格：TrackA2 纯机理的分段版本尚未实现（红色「未实现」）；四条件 R² 不可横比")

# ================================================================ fig03 三路线
fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
MODELS = ["A1.1", "A1.2", "A1.3", "A2", "A3"]
MCOLS = ["#9DB7D8", "#B7C9E2", "#C9D8EC", GREEN, DGREEN]
ax = axes[0]
w = 0.17
for j, m in enumerate(MODELS):
    vals = [float(TR[(TR.condition == c) & (TR.model == m)].loro_r2.iloc[0]) for c in CONDS]
    bar(ax, np.arange(4) + (j - 2) * w, vals, [MCOLS[j]] * 4,
        ymax=1.16, ylabel="LORO R²", fmt="%.3f", width=w, fontsize=6.6, rot=90)
ax.set_xticks(np.arange(4)); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.5)
ax.set_title("(a) 三路线 × 四条件 · LORO R²（5 模型全画）", fontsize=10)
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.6) for c in MCOLS], MODELS,
          ncol=5, fontsize=7.8, loc="upper left", framealpha=0.95)
ax = axes[1]
x = np.arange(4)
full = [float(SCOPE[SCOPE.condition == c].a3_full_r2.iloc[0]) for c in CONDS]
loro = [float(SCOPE[SCOPE.condition == c].a3_loro_r2.iloc[0]) for c in CONDS]
bar(ax, x - 0.19, full, ["#B9C9E8"] * 4, ymax=1.08, ylabel="R²", fmt="%.3f",
    width=0.38, title="(b) TrackA3 · full（群体均值层）vs LORO（验收层）")
ax.bar(x + 0.19, loro, width=0.38, color=DGREEN, edgecolor="black", linewidth=0.6,
       label="LORO 验收层", zorder=3)
for i, v in enumerate(loro):
    ax.text(i + 0.19, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.6)
ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.5)
ax.legend(["full 仅参考", "LORO 验收层"], fontsize=8, loc="lower right")
ax.text(0.015, 0.94, "full 高 ≠ 个体泛化好；差值即鼠间异质性", transform=ax.transAxes,
        ha="left", va="top", fontsize=8, color=RED,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)
finish(fig, "fig03_three_routes.png",
       "图 3 · TrackA1 / TrackA2 / TrackA3 三路线对比（全局拟合，四条件）",
       "四条件排序一致 A3 > A2 ≈ A1.1 > A1.2 ≈ A1.3；C1/C4 评价集不同，不可横比")

# ================================================================ fig04 全局 vs 分段
fig, axes = plt.subplots(2, 2, figsize=(13.8, 8.4))
SCHEMES = ["BASE_A1.1", "BASE_A2", "BASE_A3", "SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]
SCOL = {"BASE_A1.1": "#9DB7D8", "BASE_A2": GREEN, "BASE_A3": DGREEN,
        "SEG_A1": "#D8B48C", "SEG_A3": "#C48C5A", "ADJ_A1": "#C9A0DC", "ADJ_A3": "#8E44AD"}
for j, cond in enumerate(["C1_all", "C4_lf_hd"]):
    ax = axes[0, j]
    q = SEGOV[SEGOV.condition == cond].set_index("scheme")
    vals = [float(q.loc[s, "r2"]) for s in SCHEMES]
    bar(ax, np.arange(len(SCHEMES)), vals, [SCOL[s] for s in SCHEMES],
        ymax=1.02, ylabel="LORO R²", fmt="%.3f", fontsize=7.2,
        title=f"({chr(97+j)}) {SHORT[cond]} · 7 个代表方案 LORO R²")
    b3 = float(q.loc["BASE_A3", "r2"])
    ax.axhline(b3, ls="--", color=RED, lw=1.3)
    ax.text(0.015, 0.955, f"全局 A3 基准 {b3:.3f}", color=RED, fontsize=8,
            transform=ax.transAxes, ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
    ax.set_xticks(np.arange(len(SCHEMES)))
    ax.set_xticklabels(SCHEMES, rotation=35, ha="right", fontsize=7.6)
    ax = axes[1, j]
    q2 = GV[GV.condition == cond].set_index("scheme")
    ss = ["SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]
    ds = [float(q2.loc[s, "delta_r2_vs_best"]) for s in ss]
    los = [float(q2.loc[s, "delta_r2_boot_lo"]) for s in ss]
    his = [float(q2.loc[s, "delta_r2_boot_hi"]) for s in ss]
    y = np.arange(len(ss))
    ax.errorbar(ds, y, xerr=[np.array(ds) - np.array(los), np.array(his) - np.array(ds)],
                fmt="o", color=DGREEN, ecolor="#555555", elinewidth=1.4, capsize=4,
                markersize=7, zorder=3)
    ax.axvline(0, ls="--", color=RED, lw=1.3)
    for i, (d, lo, hi) in enumerate(zip(ds, los, his)):
        ax.text(d, i + 0.28, f"{d:+.4f} [{lo:+.4f},{hi:+.4f}]",
                ha="center", fontsize=7.2)
    ax.set_yticks(y); ax.set_yticklabels(ss, fontsize=8.5)
    ax.set_ylim(-0.65, 4.15)
    ax.set_xlabel("ΔR² 相对全局 A3（配对 bootstrap 95% CI）", fontsize=9)
    ax.set_title(f"({chr(99+j)}) {SHORT[cond]} · 分段方案的配对 ΔR²", fontsize=10, pad=10)
    lo_lim = min(los) - 0.012; hi_lim = max(his) + 0.020
    ax.set_xlim(lo_lim, hi_lim)
    ax.grid(axis="x", ls=":", alpha=0.5); ax.set_axisbelow(True)
    ax.text(0.985, 0.05, "CI 跨 0 → 不判胜负", transform=ax.transAxes, ha="right",
            fontsize=7.6, color=RED,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
finish(fig, "fig04_global_vs_segmented.png",
       "图 4 · 全局拟合 vs 分段拟合（两代分段方案，配对 ΔR² 与 95% CI）",
       "只有 ADJ_A3 在唯一验收口径下不劣于全局 A3；SEG_A1 / SEG_A3 / ADJ_A1 的 CI 全在 0 左侧")

# ================================================================ fig05 条件收窄路径
fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.4))
x = np.arange(4)
ax = axes[0, 0]
for j, (m, c) in enumerate(zip(["A1.1", "A2", "A3"], ["#9DB7D8", GREEN, DGREEN])):
    vals = [float(TR[(TR.condition == cc) & (TR.model == m)].loro_r2.iloc[0]) for cc in CONDS]
    bar(ax, x + (j - 1) * 0.26, vals, [c] * 4, ymax=1.02, ylabel="LORO R²",
        fmt="%.3f", width=0.26, fontsize=7.2,
        title="(a) 三路线 LORO R² 随条件收窄")
ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.5)
ax.legend([plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.6)
           for c in ["#9DB7D8", GREEN, DGREEN]], ["A1.1 纯经验", "A2 纯机理", "A3 机理驱动经验"],
          fontsize=8, loc="upper left")
ax = axes[0, 1]
full = [float(SCOPE[SCOPE.condition == c].a3_full_r2.iloc[0]) for c in CONDS]
loro = [float(SCOPE[SCOPE.condition == c].a3_loro_r2.iloc[0]) for c in CONDS]
bar(ax, x - 0.19, full, ["#B9C9E8"] * 4, ymax=1.08, ylabel="R²", fmt="%.3f",
    width=0.38, title="(b) A3：full（仅参考）vs LORO（验收）")
ax.bar(x + 0.19, loro, width=0.38, color=DGREEN, edgecolor="black", linewidth=0.6, zorder=3)
for i, v in enumerate(loro):
    ax.text(i + 0.19, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.4)
ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.5)
ax.legend(["full 仅参考", "LORO 验收"], fontsize=8, loc="lower right")
ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)
ax = axes[1, 0]
cov = [float(SCOPE[SCOPE.condition == c].trackb_coverage.iloc[0]) for c in CONDS]
bar(ax, x, cov, [PURPLE] * 4, ymax=1.08, ylabel="coverage", fmt="%.3f",
    title="(c) TrackB 覆盖率随条件收窄（目标 0.90）", hline=0.90, hlabel="目标 0.90")
ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.5)
ax = axes[1, 1]
wid = [float(SCOPE[SCOPE.condition == c].trackb_mean_width.iloc[0]) for c in CONDS]
wink = [float(SCOPE[SCOPE.condition == c].trackb_winkler.iloc[0]) for c in CONDS]
bar(ax, x - 0.19, wid, [PURPLE] * 4, ymax=max(max(wid), max(wink)) * 1.30,
    ylabel="宽度 / Winkler", fmt="%.3f", width=0.38,
    title="(d) TrackB 平均宽度与 Winkler（越小越好）")
ax.bar(x + 0.19, wink, width=0.38, color="#C44E52", edgecolor="black", linewidth=0.6,
       label="Winkler", zorder=3)
for i, v in enumerate(wink):
    ax.text(i + 0.19, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.4)
ax.set_xticks(x); ax.set_xticklabels([SHORT[c] for c in CONDS], fontsize=8.5)
ax.legend(["平均宽度", "Winkler"], fontsize=8, loc="upper right")
ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)
finish(fig, "fig05_all_vs_best_condition.png",
       "图 5 · 所有条件 → 最佳条件范围的收窄路径（C1/C2/C3/C4）",
       "⚠ C4 是 C1 子集（144 点/12 鼠/12 折 vs 576/24/24），评价集不同，数值升高不可作模型改进结论")

# ================================================================ fig06 TrackB + 回升 + padding
fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.2))
ax = axes[0]
u = UR[UR.condition == "C1_all"].copy()
u["kind"] = np.where(u.scheme.str.startswith("BASE_"), "全局单调", "分段")
gl = u[u.kind == "全局单调"]; sg = u[u.kind == "分段"]
labels = list(gl.scheme) + list(sg.scheme)
vals = [float(v) * 100 for v in list(gl.sign_match_rate) + list(sg.sign_match_rate)]
cols = [GREY] * len(gl) + [DGREEN] * len(sg)
bar(ax, np.arange(len(labels)), vals, cols, ymax=118, ylabel="回升方向一致率 (%)",
    fmt="%.0f%%", fontsize=7.2, title="(a) 尿素回升 16 节点符号一致率")
ax.set_xticks(np.arange(len(labels)))
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7.2)
ax.text(0.985, 0.965, "实测回升 Δ=+0.060（0.207 → 0.267）", transform=ax.transAxes,
        ha="right", va="top", fontsize=8, color=RED,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.92))
ax = axes[1]
# 末两点 R_Z：全为正值，纵轴自 0 起；直接显示「全局继续下降 vs 实测/分段回升」
cv1 = CUR[CUR.condition == "C1_all"]
def end_points(scheme):
    q = cv1[cv1.scheme == scheme].groupby("t").y_pred.mean()
    return float(q.loc[95]), float(q.loc[105])
b95, b105 = end_points("BASE_A3")
a95, a105 = end_points("ADJ_A3")
obs95 = float(DATA[DATA.global_time_min == 95].rz.mean())
obs105 = float(DATA[DATA.global_time_min == 105].rz.mean())
series = [("实测", [obs95, obs105], "black"),
          ("全局 A3", [b95, b105], GREY),
          ("分段 ADJ_A3", [a95, a105], DGREEN)]
for j, (name, vv, c) in enumerate(series):
    ax.bar(np.arange(2) + (j - 1) * 0.26, vv, width=0.26, color=c,
           edgecolor="black", linewidth=0.6, zorder=3, label=name)
    for i, v in enumerate(vv):
        ax.text(i + (j - 1) * 0.26, v, f"{v:.3f}", ha="center", va="bottom", fontsize=7.2)
ax.set_xticks([0, 1]); ax.set_xticklabels(["t=95 EDTA 末点", "t=105 尿素段"], fontsize=8.2)
ax.set_ylim(0, 0.38); ax.set_ylabel("R_Z", fontsize=9)
ax.set_title("(b) 末两点 R_Z：实测 / 全局 / 分段", fontsize=10)
ax.legend(fontsize=7.4, loc="upper right", framealpha=0.95)
ax.grid(axis="y", ls=":", alpha=0.5); ax.set_axisbelow(True)
ax.text(0.015, 0.955, "全局 A3 在 t=105 继续下降；实测与分段回升", transform=ax.transAxes,
        ha="left", va="top", fontsize=7.6, color=RED,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.9))
ax = axes[2]
bar(ax, np.arange(2), [97.09, 2.91], [RED, BLUE], ymax=118, ylabel="占区间宽度 (%)",
    fmt="%.2f%%", fontsize=8.5,
    title="(c) 宽区间根因：padding，不是 LHS")
ax.set_xticks(np.arange(2)); ax.set_xticklabels(["conformal padding", "基础 LHS 区间"], fontsize=8.5)
ax.text(0.985, 0.965, "残差强非正态 + 厚尾 + 正自相关 + 异方差\n→ 需大幅 conformal 校准补偿",
        transform=ax.transAxes, ha="right", va="top", fontsize=7.8, color="#333333",
        bbox=dict(boxstyle="round,pad=0.3", fc="#F4F7FB", ec=BLUE, alpha=0.95))
finish(fig, "fig06_trackB_rebound_padding.png",
       "图 6 · TrackB 区间、尿素回升与 padding 构成",
       "分段路线唯一确定变好的是回升方向（16/16），不等于个体级 LORO 提升；TrackB 只报区间指标")

# ================================================================ fig07 实测点+SEM 与分段曲线
# 群体均值层：按 (dose, freq) 求 6 鼠均值 → 每个 t 得 24 个条件节点 → 均值 ± SEM
d = DATA.copy()
node = d.groupby(["edta_concentration_mol_per_l", "frequency_hz",
                  "global_time_min"], as_index=False)["rz"].mean()
g = node.groupby("global_time_min")["rz"].agg(["mean", "std", "count"])
g["sem"] = g["std"] / np.sqrt(g["count"])
ts = np.array(sorted(g.index), float)
ym = g["mean"].to_numpy(float); ys = g["sem"].to_numpy(float)

cv = CUR[CUR.condition == "C1_all"]
def curve(scheme):
    q = cv[cv.scheme == scheme].groupby("t").y_pred.mean()
    # t=0 为基线锚点，所有模型按定义 R_Z=1
    return np.array([1.0 if t == 0 else float(q.loc[t]) for t in ts], float)
p_base = curve("BASE_A3"); p_adj = curve("ADJ_A3")

fig, ax = plt.subplots(figsize=(12.5, 6.2))
ax.errorbar(ts, ym, yerr=ys, fmt="o", color="black", ecolor="black", elinewidth=1.3,
            capsize=4, markersize=6, label="实测（16 条件节点均值 ± SEM）", zorder=5)
ax.plot(ts, p_base, ls="--", lw=2, color=GREY, marker="s", markersize=4,
        label="全局 TrackA3（单调，尿素段继续下降）", zorder=3)
ax.plot(ts, p_adj, ls="-", lw=2.2, color=DGREEN, marker="^", markersize=5,
        label="分段 ADJ_A3（尿素界面重构，回升）", zorder=4)
ax.axvspan(3.5, 20, color="#F2C879", alpha=0.16)
ax.axvspan(20, 98, color="#BFD8F2", alpha=0.16)
ax.axvspan(98, 110, color="#E7B6C8", alpha=0.20)
for xx, lab, col in [(11, "50% 乙酸段", "#A67C00"), (59, "EDTA 段", "#2F5D8C"),
                     (104, "饱和尿素段", "#A33A5C")]:
    ax.text(xx, 1.12, lab, ha="center", fontsize=9, color=col, fontweight="bold")
ax.annotate("实测回升 +0.060\n全局模型预测继续下降",
            xy=(105, ym[-1]), xytext=(72, 0.40), fontsize=8.4, color="#A33A5C",
            arrowprops=dict(arrowstyle="->", color="#A33A5C", lw=1.1),
            bbox=dict(boxstyle="round,pad=0.28", fc="white", ec="none", alpha=0.92))
ax.set_xlim(-3, 112); ax.set_ylim(0, 1.32)
ax.set_xticks(ts); ax.set_xticklabels([f"{int(t)}" for t in ts], fontsize=9)
ax.set_xlabel("全局时间 (min)", fontsize=10)
ax.set_ylabel("归一化阻抗比 R_Z（纵轴自 0 起）", fontsize=10)
ax.set_title("C1 全部条件 · 7 个时间点实测点 + SEM 全画 · 全局 A3 与分段 ADJ_A3 曲线", fontsize=11)
ax.legend(loc="lower left", fontsize=8.6, framealpha=0.95)
ax.grid(ls=":", alpha=0.5); ax.set_axisbelow(True)
ax.text(0.99, 0.02, "分段修好的是尿素回升形状（16/16 节点符号一致）；LORO 个体级泛化未显著提升",
        transform=ax.transAxes, ha="right", fontsize=8, color=RED,
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.9))
finish(fig, "fig07_segmented_curves_measured.png",
       "图 7 · 全局拟合 vs 分段拟合的视觉证据（实测点 + SEM 全画）",
       "三段按 Excel 官方试剂标签划分；基线 t=0 锚点保留；纵轴自 0 起")

print("全部图生成完毕")
