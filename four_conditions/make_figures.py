#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
三条路线 x 四条件 完整拟合图集（读 three_routes_results.json，复用 run_three_routes 的预测函数）

图全部放入 four_conditions/figures/（与前几轮位置一致）。
硬性要求（用户反复强调）：7 个时间点全画（含 t=0 基线、t=105 尿素）、实测点与 SEM 画全、
5 条模型曲线全画、纵坐标自 0 起、图例/标题不遮挡、无缺字。
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
import numpy as np
import pandas as pd

import run_three_routes as R

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
fm.fontManager.addfont(FONT)
plt.rcParams["font.family"] = fm.FontProperties(fname=FONT).get_name()
plt.rcParams["axes.unicode_minus"] = False

PAY = json.loads((ROOT / "three_routes_results.json").read_text(encoding="utf-8"))
RES = PAY["results"]
IDS = list(R.CONDITIONS)
SHORT = {"C1_all": "C1 全部", "C2_lowfreq": "C2 低频",
         "C3_highdose": "C3 高浓度", "C4_lf_hd": "C4 低频+高浓度"}

MC = {
    "A1.1": dict(color="#8C8C8C", lw=1.3, ls=(0, (4, 2)), label="A1.1 拉伸指数"),
    "A1.2": dict(color="#4C72B0", lw=1.3, ls="--", label="A1.2 A+"),
    "A1.3": dict(color="#8172B3", lw=1.3, ls="-.", label="A1.3 H1 混合"),
    "A2":   dict(color="#C44E52", lw=2.4, ls="-", label="A2 纯机理"),
    "A3":   dict(color="#2F7A4F", lw=2.4, ls="-", label="A3 机理驱动经验"),
}
ORDER = ["A1.1", "A1.2", "A1.3", "A2", "A3"]

rl, cond = R.load_data()          # 同时初始化 R._W（机理先验插值）


# ==========================================================================
# 每个条件：全 panel 完整拟合效果图
# ==========================================================================
def fit_condition(cid, out):
    spec = R.CONDITIONS[cid]
    doses, freqs = spec["doses"], spec["freqs"]
    nrow, ncol = len(doses), len(freqs)
    fig, axes = plt.subplots(nrow, ncol, figsize=(3.6 * ncol + 1.4, 3.0 * nrow + 1.9),
                             sharex=True, sharey=True, squeeze=False)
    for i, dose in enumerate(doses):
        for j, f in enumerate(freqs):
            ax = axes[i, j]
            sub = cond[(cond.dose == dose) & (cond.freq == f)].sort_values("t")
            tt = [0.0] + list(sub.t.to_numpy(float))
            yy = [1.0] + list(sub.y.to_numpy(float))
            se = [0.0] + list(sub["sem"].to_numpy(float))
            ax.errorbar(tt, yy, yerr=se, fmt="o", color="#111111", markersize=4.6,
                        capsize=3, elinewidth=1.0, label="实测 ± SEM", zorder=8)

            X = R.attach(pd.DataFrame(dict(
                dose=[float(dose)] * len(R.T_ALL), freq=[float(f)] * len(R.T_ALL),
                t=[float(t) for t in R.T_ALL], y=[0.0] * len(R.T_ALL))))
            for n in ORDER:
                p = np.asarray(RES[cid]["models"][n]["params"], float)
                v = R.predict(n, p, X).copy()
                v[0] = 1.0
                ax.plot(R.T_ALL, v, color=MC[n]["color"], lw=MC[n]["lw"],
                        ls=MC[n]["ls"], alpha=0.92, zorder=5)

            ax.axvspan(95, 105, color="#FFF2CC", alpha=0.65, zorder=0)
            ax.set_title(f"{dose} M / {int(f)} Hz", fontsize=10.5, fontweight="bold")
            ax.set_ylim(0, 1.30)
            ax.set_xlim(-4, 109)
            ax.set_xticks([0, 20, 40, 60, 80, 100])
            ax.grid(alpha=0.3, linestyle=":")
            ax.set_axisbelow(True)

    for ax in axes[-1]:
        ax.set_xlabel("全局时间 t (min)", fontsize=9.5)
    for ax in axes[:, 0]:
        ax.set_ylabel("R_Z = Z / Z0", fontsize=10)

    handles = [plt.Line2D([0], [0], marker="o", color="#111111", linestyle="none",
                          markersize=5, label="实测 ± SEM（N=6/剂量）")]
    handles += [plt.Line2D([0], [0], color=MC[n]["color"], lw=MC[n]["lw"],
                           ls=MC[n]["ls"], label=MC[n]["label"]) for n in ORDER]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.995),
               ncol=6, fontsize=9.5, frameon=False)
    nn = RES[cid]["models"]["A2"]["n_cond_nodes"]
    nf = RES[cid]["models"]["A2"]["loro"]["n_folds"]
    fig.suptitle(f"图｜{spec['label']} 完整拟合效果：{nn} 条件节点 / {nrow}x{ncol} 组合 / "
                 f"每 panel 7 时间点与 SEM 全画、5 模型曲线全画、纵轴自 0 起（LORO {nf} 折）",
                 fontsize=13, fontweight="bold", y=1.055)
    fig.text(0.5, 0.008,
             "黄色区 = 95 min 后尿素段：t=105 实测回升，所有单调模型（含 A2 纯机理）均无法表达；"
             "t=0 基线按物理定义统一锚定为 1.0；实测为条件均值 ± 鼠间 SEM。",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.028, 1, 0.955])
    fig.savefig(out, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  已生成", out.name)


# ==========================================================================
# 图1 四条件 x 五模型 full/LORO R2
# ==========================================================================
def fig1_metrics():
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.4))
    for ax, cid in zip(axes.ravel(), IDS):
        x = np.arange(len(ORDER)); wd = 0.38
        fv = [RES[cid]["models"][n]["full"]["r2"] for n in ORDER]
        lv = [RES[cid]["models"][n]["loro"]["r2"] for n in ORDER]
        b1 = ax.bar(x - wd / 2, fv, wd, color="#A9C4DE", label="full R2（样本内）")
        b2 = ax.bar(x + wd / 2, lv, wd, color=[MC[n]["color"] for n in ORDER],
                    label="LORO R2（留一大鼠）")
        for bars in (b1, b2):
            for b in bars:
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.015,
                        f"{b.get_height():.3f}", ha="center", fontsize=8, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(["A1.1", "A1.2", "A1.3", "A2", "A3"], fontsize=9.5)
        ax.set_ylim(0, 1.10)
        ax.set_ylabel("R2")
        nf = RES[cid]["models"]["A2"]["loro"]["n_folds"]
        ax.set_title(SHORT[cid] + f"  |  {RES[cid]['models']['A2']['n_cond_nodes']} 节点 / LORO {nf} 折",
                     fontsize=10.5, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle=":")
        ax.set_axisbelow(True)
    handles = [plt.Rectangle((0, 0), 1, 1, color="#A9C4DE"),
               plt.Rectangle((0, 0), 1, 1, color="#C44E52")]
    fig.legend(handles, ["full R2（样本内，统一浅蓝）", "LORO R2（彩色，颜色对应各模型）"],
               loc="upper center", bbox_to_anchor=(0.5, 0.985), ncol=2, fontsize=10, frameon=False)
    fig.suptitle("图1｜四条件 x 三条路线五模型的拟合与独立验证（纵轴自 0 起）",
                 fontsize=13.5, fontweight="bold", y=1.05)
    fig.text(0.5, 0.012,
             "A1 = 纯经验 / A2 = 纯机理 / A3 = 机理驱动经验。四条件评价集与 LORO 折数不同"
             "（C1/C2 为 24 折，C3/C4 为 12 折），R2 数值不可横向解读为模型优劣。",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.05, 1, 0.955])
    fig.savefig(FIG / "fig1_metrics_by_condition.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig1")


# ==========================================================================
# 图6 三路线 LORO 跨条件汇总
# ==========================================================================
def fig6_routes():
    fig, ax = plt.subplots(figsize=(13, 6.4))
    x = np.arange(len(IDS)); wd = 0.16
    for k, n in enumerate(ORDER):
        vals = [RES[c]["models"][n]["loro"]["r2"] for c in IDS]
        bars = ax.bar(x + (k - 2) * wd, vals, wd, color=MC[n]["color"],
                      label=MC[n]["label"], alpha=0.92)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.008,
                    f"{v:.3f}", ha="center", fontsize=7.6, fontweight="bold", rotation=0)
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT[c] for c in IDS], fontsize=10.5)
    ax.set_ylim(0, 0.92)
    ax.set_ylabel("LORO R2（留一大鼠，验收口径）")
    ax.set_title("图6｜三条路线五模型在四条件下的 LORO 独立验证（纵轴自 0 起）",
                 fontsize=13.5, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9.5, framealpha=0.95, ncol=2)
    ax.grid(axis="y", alpha=0.3, linestyle=":")
    ax.set_axisbelow(True)
    fig.text(0.5, 0.012,
             "关键：A2 纯机理（4 个物理量、频率内生、零经验回归层）在 C1 全条件 LORO=0.486，"
             "反而高于三个纯经验模型（0.474-0.481）；A3 机理驱动经验四条件 LORO 全部第一。\n"
             "条件越窄指标越好看是评价集收窄，不是模型变好；四条件评价集与 LORO 折数不同，R2 不可横比。",
             ha="center", fontsize=9, color="#8A5A00",
             bbox=dict(boxstyle="round,pad=0.45", fc="#FFF8E7", ec="#D9A441", alpha=0.95))
    fig.tight_layout(rect=[0, 0.085, 1, 1])
    fig.savefig(FIG / "fig6_routes_loro.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig6")


# ==========================================================================
# 图7 A2 纯机理曲面：把频率算进去的直接证明
# ==========================================================================
def fig7_surface():
    p = np.asarray(PAY["A2_global_params"], float)   # C1 全条件一套物理参数
    ff = np.array([100, 120, 1000, 10000], float)
    tt = np.array([5, 20, 35, 65, 95], float)
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.4))
    ax = axes[0]
    cmap = plt.get_cmap("viridis")
    xpos = np.arange(len(ff))          # 4 个离散测量频率用等距分类刻度，避免 100/120 在对数轴上挤在一起
    for i, t in enumerate(tt):
        X = R.attach(pd.DataFrame(dict(dose=[0.15] * len(ff), freq=ff,
                                       t=[float(t)] * len(ff), y=[0.0] * len(ff))))
        ax.plot(xpos, R.r_mech(p, X), "o-", color=cmap(i / (len(tt) - 1)),
                lw=2, markersize=6, label=f"t={int(t)} min")
    ax.set_xticks(xpos); ax.set_xticklabels(["100", "120", "1000", "10000"])
    ax.set_xlabel("测量频率 (Hz，离散分类刻度)")
    ax.set_ylabel("机理预测 R_Z")
    ax.set_ylim(0, 1.05)
    ax.set_title("(a) 频率内生：同一时间下高频 R_Z 系统性偏高", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8.5, ncol=2); ax.grid(alpha=0.3, linestyle=":")

    ax = axes[1]
    gx = np.linspace(100, 10000, 60)
    gy = np.linspace(2, 105, 60)
    GX, GY = np.meshgrid(gx, gy)
    DD = np.zeros_like(GX)
    for a in range(GX.shape[0]):
        X = R.attach(pd.DataFrame(dict(dose=[0.15] * GX.shape[1], freq=GX[a],
                                       t=GY[a], y=[0.0] * GX.shape[1])))
        DD[a] = R.r_mech(p, X)
    cs = ax.contourf(np.log10(GX), GY, DD, levels=14, cmap="viridis")
    cb = fig.colorbar(cs, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("机理预测 R_Z", fontsize=9)
    ax.set_xticks(np.log10([100, 1000, 10000]))
    ax.set_xticklabels(["100", "1000", "10000"])
    ax.set_xlabel("测量频率 (Hz, 对数刻度；120 Hz 位于 100 与 1000 之间)")
    ax.set_ylabel("全局时间 t (min)")
    ax.set_title("(b) A2 单一组物理参数覆盖时间 x 频率全平面", fontsize=11, fontweight="bold")
    fig.suptitle("图7｜A2 纯机理模型：把频率维度写进机理方程（0.15 M 示例，C1 全条件参数）",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.text(0.5, 0.005,
             "delta_eff = a0*(d/0.10)^b*(t/100)^p*(f/100)^(-s)；R_Z = 1/(1+delta_eff)。"
             "一套 4 参数物理量覆盖 4 剂量 x 4 频率 x 6 时间 = 96 条件节点，零经验回归系数。",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(FIG / "fig7_mechanism_surface.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig7")


# ==========================================================================
# 图8 三条路线架构 / 机理含量
# ==========================================================================
def fig8_architecture():
    fig, ax = plt.subplots(figsize=(14, 7.6))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    boxes = [
        (0.03, 0.60, 0.30, 0.34, "#EAF0F8", "#2F5D96",
         "TrackA1 纯经验模型",
         "数据驱动，机理方程最多当查表特征\n"
         "A1.1 拉伸指数：R_Z=C+(1-C)exp(-(bt)^n)\n"
         "A1.2 A+：a+b*raw+c*t+d*ln(f/100)\n"
         "A1.3 H1：1-kappa*(f/100)^(-s)*delta-eps*t\n"
         "机理含量：无 / 低（raw、delta 作特征）"),
        (0.35, 0.60, 0.30, 0.34, "#FBEDEE", "#C44E52",
         "TrackA2 纯机理模型",
         "调机理：频率/剂量/时间全部内生\n"
         "Archie 电导 + 脱钙动力学 + CPE 并联导纳\n"
         "R_Z = 1/(1+delta_eff)\n"
         "delta_eff=a0*d^b*t^p*f^(-s)\n"
         "机理含量：100%，4 个全物理量参数"),
        (0.67, 0.60, 0.30, 0.34, "#EAF5EE", "#2F7A4F",
         "TrackA3 机理驱动经验模型",
         "机理给骨架（R_mech = A2 预测）\n"
         "经验做受约束保形校准\n"
         "R_Z = 1-eta*(1-R_mech)-eps*(t/105)\n"
         "eta=1,eps=0 时退化为 A2（嵌套）\n"
         "机理含量：骨架 100% + 2 个经验系数"),
    ]
    from matplotlib.patches import FancyBboxPatch
    for x, y, w, h, fc, ec, title, body in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012",
                                    fc=fc, ec=ec, lw=2, alpha=0.95))
        ax.text(x + w / 2, y + h - 0.045, title, ha="center", va="top",
                fontsize=12, fontweight="bold", color=ec)
        ax.text(x + 0.012, y + h - 0.085, body, ha="left", va="top",
                fontsize=8.9, color="#222222", linespacing=1.55)

    # 三条路线各自的机理含量（对齐各自方框正下方，不按编号暗示单调排序）
    from matplotlib.patches import FancyBboxPatch as _FBB
    caps = [
        (0.18, "#2F5D96", "#EAF0F8", "A1：机理含量低", "纯数据驱动；参数无物理量纲"),
        (0.50, "#C44E52", "#FBEDEE", "A2：机理含量 100%", "零经验回归层；4 个全物理量参数"),
        (0.82, "#2F7A4F", "#EAF5EE", "A3：机理骨架 + 经验", "机理定形状，2 个受约束校准系数"),
    ]
    for xpos, ec, fc, t1, t2 in caps:
        ax.add_patch(_FBB((xpos - 0.145, 0.405), 0.29, 0.105,
                          boxstyle="round,pad=0.008", fc=fc, ec=ec, lw=1.4, alpha=0.95))
        ax.text(xpos, 0.482, t1, ha="center", va="center", fontsize=9.8,
                fontweight="bold", color=ec)
        ax.text(xpos, 0.440, t2, ha="center", va="center", fontsize=8.2, color="#333333")
    ax.text(0.5, 0.375, "方框按路线编号 A1/A2/A3 排列；下方标签按各自机理含量标注，二者不是同一排序维度",
            ha="center", fontsize=8.4, color="#777777")

    ax.text(0.5, 0.30,
            "共同机理底座（三条路线共享的物理事实，A1 只借其输出、A2 直接改写、A3 借其骨架）",
            ha="center", fontsize=10.5, fontweight="bold", color="#333333")
    ax.text(0.5, 0.245,
            "EDTA 螯合颅骨羟基磷灰石 Ca -> 孔隙率 phi 上升 -> 离子电导 sigma = sigma_floor + c*phi^m 上升 -> 阻抗 Z 下降。\n"
            "实测「同一时间高频 R_Z 系统性偏高」-> 界面极化 / CPE 旁路导纳 G_pol(f) ~ f^(-s) 与离子通路并联 -> 频率进入机理方程。\n"
            "R_Z = Y(0)/Y(t) = 1/(1 + delta_eff)，t=0 天然等于 1，无需人为锚定。",
            ha="center", va="top", fontsize=9, color="#333333", linespacing=1.6,
            bbox=dict(boxstyle="round,pad=0.6", fc="#F7F7F2", ec="#BBBBBB", alpha=0.95))
    ax.text(0.5, 0.045,
            "诚实边界：三条路线都是「有物理动机的参数化」，不等于机理被证实；"
            "delta_eff 单调，尿素 t=105 回升所有模型均无法表达；物理量由实测标定，不是纯先验推导。",
            ha="center", fontsize=8.6, color="#8A5A00")
    fig.suptitle("图8｜三条 TrackA 路线的定义、机理含量与共享机理底座",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.savefig(FIG / "fig8_architecture.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig8")


# ==========================================================================
# 图9 TrackB 四条件
# ==========================================================================
def fig9_trackb():
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    cov = [RES[c]["trackB"]["coverage"] for c in IDS]
    wid = [RES[c]["trackB"]["mean_width"] for c in IDS]
    win = [RES[c]["trackB"]["winkler"] for c in IDS]
    sht = [SHORT[c] for c in IDS]
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
        ax.set_xticks(range(4)); ax.set_xticklabels(sht, fontsize=9)
        ax.set_ylim(0, max(vals) * 1.22); ax.set_ylabel(ylab)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.grid(axis="y", alpha=0.3, linestyle=":"); ax.set_axisbelow(True)
    fig.suptitle("图9｜TrackB 四条件切片：覆盖率、宽度、Winkler（不报 R2 / RMSE）",
                 fontsize=13.5, fontweight="bold", y=1.02)
    fig.text(0.5, 0.005,
             "仅为既有 392 点 LOO 校准区间按条件切片统计，未重新校准；"
             "区间宽度约 97% 来自 conformal padding；逐点覆盖率不等于整条曲线联合覆盖。",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=[0, 0.075, 1, 0.94])
    fig.savefig(FIG / "fig9_trackB_four_conditions.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig9")


# ==========================================================================
# 图10 表1
# ==========================================================================
def fig10_table():
    rows = []
    for cid in IDS:
        for n in ORDER:
            m = RES[cid]["models"][n]
            rows.append([SHORT[cid], m["route"], n, str(m["k"]), str(m["n_cond_nodes"]),
                         f'{m["full"]["r2"]:.3f}', f'{m["full"]["rmse"]:.3f}',
                         f'{m["loco"]["r2"]:.3f}', f'{m["loro"]["r2"]:.3f}',
                         f'{m["loro"]["per_rat_mean_r2"]:.3f}', str(m["loro"]["n_folds"]),
                         f'{m["baseline_pred"]:.3f}'])
    fig, ax = plt.subplots(figsize=(15.5, 8.2))
    ax.axis("off")
    cols = ["条件", "路线", "模型", "k", "节点", "full R2", "RMSE", "LOCO R2",
            "LORO R2", "逐鼠均值", "LORO 折数", "t=0 外推"]
    tbl = ax.table(cellText=rows, colLabels=cols, cellLoc="center", loc="center",
                   colWidths=[0.13, 0.075, 0.075, 0.045, 0.055, 0.075, 0.075,
                              0.075, 0.075, 0.085, 0.085, 0.075])
    tbl.auto_set_font_size(False); tbl.set_fontsize(8.4); tbl.scale(1, 1.42)
    for j in range(len(cols)):
        tbl[0, j].set_facecolor("#2F5D96")
        tbl[0, j].set_text_props(color="white", weight="bold")
    for i in range(len(rows)):
        r = rows[i]
        if r[1] == "A2":
            for j in range(len(cols)):
                tbl[i + 1, j].set_facecolor("#FBEDEE")
        elif r[1] == "A3":
            for j in range(len(cols)):
                tbl[i + 1, j].set_facecolor("#EAF5EE")
        elif i % 2 == 1:
            for j in range(len(cols)):
                tbl[i + 1, j].set_facecolor("#F2F6FB")
    ax.set_title("表1｜四条件 x 三条路线五模型拟合指标（红=A2 纯机理 / 绿=A3 机理驱动经验）",
                 fontsize=13, fontweight="bold", pad=14)
    fig.text(0.5, 0.02,
             "LORO = 留一大鼠（个体级独立验证，验收口径）；LOCO = 留一条件（群体均值层）。"
             "C1/C2 为 24 折、C3/C4 为 12 折；评价集与折数不同，R2 不可横向比大小。"
             "t=0 外推含未锚定的基线值，A1.2 (A+) 偏离 1 需人为锚定，其余天然等于 1。",
             ha="center", fontsize=8.8, color="#555555")
    fig.savefig(FIG / "fig10_table1.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  fig10")


def main():
    print("生成图集 ->", FIG)
    fit_condition("C1_all", FIG / "fig2_fit_C1_all.png")
    fit_condition("C2_lowfreq", FIG / "fig3_fit_C2_lowfreq.png")
    fit_condition("C3_highdose", FIG / "fig4_fit_C3_highdose.png")
    fit_condition("C4_lf_hd", FIG / "fig5_fit_C4_lf_hd.png")
    fig1_metrics()
    fig6_routes()
    fig7_surface()
    fig8_architecture()
    fig9_trackb()
    fig10_table()
    print("图集完成，共 10 张")


if __name__ == "__main__":
    main()
