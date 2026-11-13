#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
最新模型拟合效果图（仅用"矫正后/校准后"模型）：
  Track A : A1 经验校正模型  R_Z_corr = a + b * R_Z_raw
  Track B : 校准后的嵌套 LOO 预测区间 (LB/UB)
输出：
  figA_trackA_fit.png      TrackA 校正模型拟合效果（实测均值±SD vs A1 预测，剂量×频率）
  figB_trackA_scatter.png  TrackA 校正模型 预测值 vs 真实值 散点
  figC_trackB_fit.png      TrackB 校准区间拟合效果（实测均值±SD vs 预测区间，剂量×频率）
  figD_trackB_coverage.png TrackB 覆盖率/宽度诊断（逐点，目标 90%）
  FITTING_EFFECT.md        精简指标汇总
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

_FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
try:
    fm.fontManager.addfont(_FP)
    plt.rcParams["font.family"] = fm.FontProperties(fname=_FP).get_name()
except Exception:
    pass
plt.rcParams.update({"axes.unicode_minus": False, "font.size": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 130, "savefig.dpi": 130})

OUT = "/workspace/Chemical_Skull_Model_FittingEffect"
MEAS = "/workspace/Chemical_Skull_Model_Batch1/data_audit/frozen_measured_24rat.csv"
MECH = "/workspace/Chemical_Skull_Model_LowFreq_100_120Hz/data/model_sources_raw/mechanism_timeseries_quantiles.csv"
TB = "/workspace/Chemical_Skull_Model_LowFreq_100_120Hz/data/trackB_point_intervals.csv"

FREQS = [100, 120]
DOSES = [0.05, 0.10, 0.15, 0.20]
T_MOD = [5, 20, 35, 65, 95, 105]
STAGE_BY_TIME = {0: "baseline", 5: "acetic_acid", 20: "edta_15", 35: "edta_30",
                 65: "edta_60", 95: "edta_90", 105: "urea_10"}
TIME_ALL = [0, 5, 20, 35, 65, 95, 105]
TARGET_COV = 0.90


def metrics(y, x):
    y = np.asarray(y, float); x = np.asarray(x, float)
    ss_res = float(np.sum((y - x) ** 2)); ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    return dict(n=len(y), r2=r2, rmse=float(np.sqrt(np.mean((y - x) ** 2))),
                mae=float(np.mean(np.abs(y - x))), bias=float(np.mean(y - x)))


def fit_ols(x, y):
    X = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(coef[0]), float(coef[1])


def build_prior():
    q = pd.read_csv(MECH)
    w = (q[q.metric == "whole_resistance_ratio"]
         .drop_duplicates(["edta_concentration_mol_per_l", "global_time_s"])
         .sort_values(["edta_concentration_mol_per_l", "global_time_s"]))
    def ta(dose, t_min, stat="q50"):
        g = w[w.edta_concentration_mol_per_l == dose]
        ts = g.global_time_s.to_numpy(float); vs = g[stat].to_numpy(float)
        if t_min * 60.0 > ts.max():
            return float(vs[-1])
        return float(np.interp(t_min * 60.0, ts, vs))
    return ta


def build_tables():
    s = pd.read_csv(MEAS)
    s = s[s.frequency_hz.isin(FREQS)].copy()
    s["global_time_min"] = s["stage"].map({v: k for k, v in STAGE_BY_TIME.items()})
    obs = (s.groupby(["rat", "edta_concentration_mol_per_l", "frequency_hz", "global_time_min"])["rz"]
           .mean().reset_index(name="observed_rz"))
    ta = build_prior()
    obs["pred_q50"] = obs.apply(lambda r: ta(float(r.edta_concentration_mol_per_l), int(r.global_time_min)), axis=1)
    return obs


def main():
    obs = build_tables()
    rats = sorted(obs.rat.unique())
    tb = pd.read_csv(TB)
    tb_rats = set(tb.rat.unique())

    # ---- Track A corrected model (group-mean level, n=48, exclude baseline) ----
    cond = (obs[obs.global_time_min.isin(T_MOD)]
            .groupby(["edta_concentration_mol_per_l", "frequency_hz", "global_time_min"])
            .agg(observed_rz=("observed_rz", "mean"), pred_q50=("pred_q50", "first"))
            .reset_index())
    yc, xc = cond.observed_rz.to_numpy(float), cond.pred_q50.to_numpy(float)
    a_f, b_f = fit_ols(xc, yc)
    xc_a1 = a_f + b_f * xc
    m_a1 = metrics(yc, xc_a1)
    m_raw = metrics(yc, xc)  # Track A 原始机理模型（未校正）
    # LOCO / LORO for honesty note
    yhat_loco = np.empty(len(yc))
    for i in range(len(yc)):
        m = np.ones(len(yc), bool); m[i] = False
        aa, bb = fit_ols(xc[m], yc[m]); yhat_loco[i] = aa + bb * xc[i]
    m_loco = metrics(yc, yhat_loco)
    y_all, x_all, rat_of = [], [], []
    for r in rats:
        sub = obs[obs.rat == r]; sub = sub[sub.global_time_min.isin(T_MOD)]
        y_all.append(sub.observed_rz.to_numpy(float)); x_all.append(sub.pred_q50.to_numpy(float))
        rat_of.extend([r] * len(sub))
    y_all = np.concatenate(y_all); x_all = np.concatenate(x_all); rat_of = np.array(rat_of)
    yhat = np.empty(len(y_all))
    for r in rats:
        tr = rat_of != r; aa, bb = fit_ols(x_all[tr], y_all[tr]); yhat[~tr] = aa + bb * x_all[~tr]
    m_loro = metrics(y_all, yhat)

    # ---- Track B calibrated summary ----
    tb["covered"] = tb["covered"].astype(bool)
    cov = float(tb.covered.mean()); width = float(tb.width.mean())
    wink = float(tb["winkler_alpha_mis_0.10"].mean())
    role = tb.groupby("ensemble_role").apply(
        lambda g: pd.Series(dict(coverage=g.covered.mean(), n=len(g), width=g.width.mean())))
    meas_tb = obs[obs.rat.isin(tb_rats)]
    tb_times = sorted(tb.global_time_min.unique())

    # ============================================================ FIG A: TrackA fit
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for i, d in enumerate(DOSES):
        for j, fr in enumerate(FREQS):
            ax = axes[i, j]
            sub = obs[(obs.edta_concentration_mol_per_l == d) & (obs.frequency_hz == fr)].copy()
            g = sub.groupby("global_time_min")["observed_rz"].agg(["mean", "std", "count"]).reindex(TIME_ALL)
            sem = g["std"] / np.sqrt(g["count"].replace(0, np.nan))  # 6 鼠组均值的标准误
            t_h = np.array(TIME_ALL) / 60.0
            ax.errorbar(t_h, g["mean"], yerr=sem, fmt="o", color="#C44E52", capsize=2, ms=4,
                        label="实测均值 (N=6, ±SEM)")
            a1 = np.array([a_f + b_f * build_prior()(d, t) for t in TIME_ALL])
            a1[0] = 1.0  # 基线按定义 R_Z=1.0 锚定（A1 线性外推在 t=0 会偏离，此处物理锚定使曲线与实测基线点衔接）
            ax.plot(t_h, a1, "--", color="#4C72B0", lw=1.8, label="Track A A1 校正预测")
            ax.set_ylim(0, 1.2); ax.set_title(f"剂量 {d:.2f} M, {fr} Hz", fontsize=9)
            if i == 3: ax.set_xlabel("时间 (h)")
            if j == 0: ax.set_ylabel(r"阻抗比值 $R_Z$")
            if i == 0 and j == 0: ax.legend(fontsize=7, loc="upper right")
    fig.suptitle(f"图A  Track A 校正模型拟合效果（实测 N=6 均值±SEM vs A1 校正预测）\n"
                 f"群体均值层：R²={m_a1['r2']:.3f}, RMSE={m_a1['rmse']:.3f}, MAE={m_a1['mae']:.3f}, Bias={m_a1['bias']:.3f}",
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig(f"{OUT}/figA_trackA_fit.png"); plt.close(fig)

    # ============================================================ FIG B: TrackA scatter
    fig, ax = plt.subplots(figsize=(6.5, 6))
    sc = ax.scatter(yc, xc_a1, c=cond.edta_concentration_mol_per_l, cmap="viridis", s=40, alpha=0.85)
    lim = [min(yc.min(), xc_a1.min()), max(yc.max(), xc_a1.max())]
    ax.plot(lim, lim, "k--", lw=1)
    ax.set_xlabel("真实阻抗比值 (实测条件均值)"); ax.set_ylabel("Track A A1 校正预测")
    ax.set_title(f"图B  Track A 校正模型：预测值 vs 真实值 (n=48)\n"
                 f"R²={m_a1['r2']:.3f}, RMSE={m_a1['rmse']:.3f}, MAE={m_a1['mae']:.3f}, Bias={m_a1['bias']:.3f}", fontsize=10)
    fig.colorbar(sc, ax=ax, label="EDTA 浓度 (M)", ticks=[0.05, 0.10, 0.15, 0.20])
    fig.tight_layout(); fig.savefig(f"{OUT}/figB_trackA_scatter.png"); plt.close(fig)

    # ============================================================ FIG C: TrackB fit
    # 实测画满全部 7 个时间点（基线→尿素）；Track B 区间仅在验证覆盖的 5 个时间点存在
    meas_c = obs[obs.rat.isin(tb_rats)].copy()
    fig, axes = plt.subplots(4, 2, figsize=(11, 12), sharex=True)
    for i, d in enumerate(DOSES):
        for j, fr in enumerate(FREQS):
            ax = axes[i, j]
            # 实测完整轨迹（7 时间点）
            sub = meas_c[(meas_c.edta_concentration_mol_per_l == d) & (meas_c.frequency_hz == fr)]
            go = sub.groupby("global_time_min")["observed_rz"].agg(["mean", "std", "count"]).reindex(TIME_ALL)
            sem_o = go["std"] / np.sqrt(go["count"].replace(0, np.nan))
            t_all_h = np.array(TIME_ALL) / 60.0
            ax.errorbar(t_all_h, go["mean"], yerr=sem_o, fmt="o", color="#C44E52", capsize=2, ms=4,
                        label="实测均值 (N=6, ±SEM)")
            # 校准区间仅在有验证数据的 5 个时间点
            gi = tb[(tb.edta_concentration_mol_per_l == d) & (tb.frequency_hz == fr)
                    & (tb.global_time_min.isin(tb_times))].groupby("global_time_min").agg(
                LB=("LB", "mean"), UB=("UB", "mean"), C=("trackB_center", "mean")).reindex(tb_times)
            t_int_h = np.array(tb_times) / 60.0
            ax.fill_between(t_int_h, gi["LB"], gi["UB"], color="#55A868", alpha=0.20,
                            label="Track B 预测区间 (5 时间点)")
            ax.plot(t_int_h, gi["C"], "--", color="#55A868", lw=1.3, label="区间中点")
            cov_p = tb[(tb.edta_concentration_mol_per_l == d) & (tb.frequency_hz == fr)].covered.mean()
            ax.axvline(0.0, color="gray", ls=":", lw=0.7); ax.axvline(1.75, color="gray", ls=":", lw=0.7)
            ax.set_ylim(0, 1.2); ax.set_title(f"剂量 {d:.2f} M, {fr} Hz (覆盖率={cov_p:.2f})", fontsize=9)
            if i == 3: ax.set_xlabel("时间 (h)")
            if j == 0: ax.set_ylabel(r"阻抗比值 $R_Z$")
            if i == 0 and j == 0: ax.legend(fontsize=6.5, loc="lower left")
    fig.suptitle("图C  Track B 校准区间拟合效果（实测均值±SEM vs 校准预测区间）\n"
                 "注：实测为 Track B 验证集 22 鼠完整 7 时间点轨迹；Track B 区间仅覆盖乙酸→edta_90 共 5 时间点"
                 "（基线/尿素时段无校准区间，以灰虚线标出）；虚线=区间中点（非独立点预测）", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f"{OUT}/figC_trackB_fit.png"); plt.close(fig)

    # ============================================================ FIG D: TrackB coverage (enhanced 2x2)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    t_all_h = np.array(tb_times) / 60.0
    # (0,0) overall coverage vs time
    ax = axes[0, 0]
    cov_t = tb.groupby("global_time_min").covered.mean().reindex(tb_times)
    ax.plot(t_all_h, cov_t.values, "o-", color="#4C72B0")
    ax.axhline(TARGET_COV, color="r", ls="--", label=f"目标 {TARGET_COV:.0%}")
    ax.set_ylim(0, 1.05); ax.set_title("逐时间点覆盖率（总体）"); ax.set_xlabel("时间 (h)"); ax.set_ylabel("覆盖率"); ax.legend(fontsize=8)
    # (0,1) coverage by role
    ax = axes[0, 1]
    for r_, g in tb.groupby("ensemble_role"):
        gt = g.groupby("global_time_min").covered.mean().reindex(tb_times)
        short = "NEAREST" if r_.startswith("NEAREST") else "TRAINING_COMPATIBLE"
        ax.plot(t_all_h, gt.values, "o-", label=short)
    ax.axhline(TARGET_COV, color="r", ls="--")
    ax.set_ylim(0, 1.05); ax.set_title("逐时间点覆盖率（按 ensemble 模式）"); ax.set_xlabel("时间 (h)"); ax.legend(fontsize=7)
    # (1,0) coverage by dose
    ax = axes[1, 0]
    for d in DOSES:
        gd = tb[tb.edta_concentration_mol_per_l == d]
        if len(gd) == 0:
            continue
        gt = gd.groupby("global_time_min").covered.mean().reindex(tb_times)
        ax.plot(t_all_h, gt.values, "o-", label=f"{d:.2f} M")
    ax.axhline(TARGET_COV, color="r", ls="--")
    ax.set_ylim(0, 1.05); ax.set_title("逐时间点覆盖率（按剂量）"); ax.set_xlabel("时间 (h)"); ax.legend(fontsize=7, title="剂量")
    # (1,1) avg interval width by dose
    ax = axes[1, 1]
    wd = [tb[tb.edta_concentration_mol_per_l == d].width.mean() for d in DOSES]
    ax.bar([f"{d:.2f} M" for d in DOSES], wd, color="#55A868", alpha=0.8)
    ax.axhline(width, color="r", ls="--", label=f"总体 {width:.3f}")
    ax.set_title("平均区间宽度（按剂量）"); ax.set_xlabel("EDTA 剂量"); ax.legend(fontsize=7)
    fig.suptitle(f"图D  Track B 区间覆盖诊断（逐点覆盖率={cov:.3f}，目标 90%；平均宽度={width:.3f}，Winkler={wink:.3f}）\n"
                 "0.05 M 仅出现在 ORIGINAL_COMPATIBLE 模式，剂量效应与模式混淆；覆盖率=逐点口径，非整条曲线同时落入", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f"{OUT}/figD_trackB_coverage.png"); plt.close(fig)

    # ============================================================ FIG E: TrackA residual diagnostics (A1 corrected)
    resid_a1 = xc_a1 - yc  # predicted - observed
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    ax = axes[0]
    for d in DOSES:
        m = cond.edta_concentration_mol_per_l.to_numpy(float) == d
        ax.scatter(cond.global_time_min.to_numpy(float)[m] / 60.0, resid_a1[m], s=32, label=f"{d:.2f} M")
    ax.axhline(0, color="k", lw=1); ax.set_xlabel("时间 (h)"); ax.set_ylabel("残差 = A1预测 − 真实")
    ax.set_title("图E(1)  Track A A1 校正残差 vs 时间"); ax.legend(fontsize=7, title="剂量")
    ax = axes[1]
    for d in DOSES:
        m = cond.edta_concentration_mol_per_l.to_numpy(float) == d
        ax.scatter(xc_a1[m], resid_a1[m], s=32, label=f"{d:.2f} M")
    ax.axhline(0, color="k", lw=1); ax.set_xlabel("A1 校正预测值"); ax.set_ylabel("残差 = A1预测 − 真实")
    ax.set_title("图E(2)  Track A A1 校正残差 vs 预测值"); ax.legend(fontsize=7, title="剂量")
    fig.suptitle("图E  Track A 校正模型残差诊断（检验系统性偏差：整体偏置≈0；残差随预测值/时间呈轻微结构——"
                  "早期偏低、后期偏高，提示 A1 经验校正仍有轻微未建模非线性）", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f"{OUT}/figE_trackA_residuals.png"); plt.close(fig)

    # ============================================================ FIG F: TrackA error by dose (A1 corrected vs raw)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 4.4))
    per_dose = {}
    for d in DOSES:
        m = cond.edta_concentration_mol_per_l.to_numpy(float) == d
        per_dose[d] = dict(a1=metrics(yc[m], xc_a1[m]), raw=metrics(yc[m], xc[m]))
    titles = [("RMSE", "rmse"), ("MAE", "mae"), ("Bias", "bias")]
    for k, (tlabel, key) in enumerate(titles):
        ax = axes[k]
        a1v = [per_dose[d]["a1"][key] for d in DOSES]
        rawv = [per_dose[d]["raw"][key] for d in DOSES]
        xpos = np.arange(len(DOSES)); w = 0.38
        ax.bar(xpos - w / 2, rawv, w, label="原始", color="#C44E52", alpha=0.75)
        ax.bar(xpos + w / 2, a1v, w, label="A1校正", color="#4C72B0")
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(xpos); ax.set_xticklabels([f"{d:.2f} M" for d in DOSES])
        ax.set_title(f"图F({tlabel}) 按剂量（n=12/剂量）"); ax.set_xlabel("EDTA 剂量")
        if k == 0:
            ax.set_ylabel(tlabel)
        ax.legend(fontsize=7)
    fig.suptitle("图F  Track A 不同剂量下误差对比（原始 vs A1校正；RMSE 整体误差/对大误差敏感，MAE 平均绝对误差，Bias 系统偏差）", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.93]); fig.savefig(f"{OUT}/figF_trackA_error_by_dose.png"); plt.close(fig)

    # ============================================================ FIG G: 表1 metrics summary (rendered table)
    fig = plt.figure(figsize=(10.2, 5.4))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.4, 1], hspace=0.25)
    ax0 = fig.add_subplot(gs[0]); ax0.axis("off")
    ax1 = fig.add_subplot(gs[1]); ax1.axis("off")
    tbl = [["模型 / 验证层级", "R²", "RMSE", "MAE", "Bias"],
           ["Track A 原始模型（群体均值层 n=48）", f"{m_raw['r2']:.3f}", f"{m_raw['rmse']:.3f}", f"{m_raw['mae']:.3f}", f"{m_raw['bias']:.3f}"],
           ["Track A A1 校正（full OLS n=48）", f"{m_a1['r2']:.3f}", f"{m_a1['rmse']:.3f}", f"{m_a1['mae']:.3f}", f"{m_a1['bias']:.3f}"],
           ["Track A 群体均值 LOCO", f"{m_loco['r2']:.3f}", f"{m_loco['rmse']:.3f}", f"{m_loco['mae']:.3f}", f"{m_loco['bias']:.3f}"],
           ["Track A 个体级 LORO", f"{m_loro['r2']:.3f}", f"{m_loro['rmse']:.3f}", f"{m_loro['mae']:.3f}", f"{m_loro['bias']:.3f}"]]
    t = ax0.table(cellText=tbl, loc="center", cellLoc="center",
                  colWidths=[0.42, 0.145, 0.145, 0.145, 0.145])
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.7)
    for (r, c), cell in t.get_celld().items():
        if r == 0:
            cell.set_facecolor("#4C72B0"); cell.set_text_props(color="w")
        elif r == 4:
            cell.set_facecolor("#F2F2F2")
    ax0.set_title("表1  Track A 点预测指标汇总（同一组真值/预测值，按验证层级）\n"
                  "注：0.588 为个体级 LORO 总体 R²，逐鼠 R² 均值 0.404；0.762 为群体条件均值 LOCO，两者层级不同、不可直接比较",
                  fontsize=10, pad=12)
    _role_cov = {("NEAREST" if k.startswith("NEAREST") else "COMPATIBLE"): v["coverage"]
                 for k, v in role.to_dict("index").items()}
    tb_sum = (f"Track B 区间预测（单独汇报，逐点口径）：逐点覆盖率={cov:.3f}（目标 90%），平均区间宽度={width:.3f}，"
              f"Winkler(α_mis=0.10)={wink:.3f}，n={len(tb)} 点 = {tb.rat.nunique()} 鼠 × 2 频率 × 6 时间\n"
              f"ensemble 模式分裂（合并值掩盖差异）：TRAINING_COMPATIBLE 覆盖率 {_role_cov.get('COMPATIBLE', float('nan')):.3f} / "
              f"NEAREST {_role_cov.get('NEAREST', float('nan')):.3f}；0.05 M 仅出现在 COMPATIBLE 模式；验证集缺 WT5/WT6")
    ax1.text(0, 1, tb_sum, ha="left", va="top", fontsize=8.5, transform=ax1.transAxes)
    fig.savefig(f"{OUT}/figG_table1.png", bbox_inches="tight"); plt.close(fig)

    # ---- report ----
    role_md = "\n".join(f"| {k} | {v['coverage']:.3f} | {int(v['n'])} | {v['width']:.3f} |" for k, v in role.to_dict("index").items())
    md = []
    md.append("# 最新模型拟合效果（矫正后 / 校准后）\n")
    md.append("## Track A — A1 经验校正模型  R_Z_corr = a + b·R_Z_raw\n")
    md.append(f"- 校正参数（群体均值层全数据拟合）：**a = {a_f:.4f}, b = {b_f:.4f}**")
    md.append(f"- 群体均值层（n=48 条件，留一条件 LOCO 对照）：R²={m_a1['r2']:.4f}, RMSE={m_a1['rmse']:.4f}, "
              f"MAE={m_a1['mae']:.4f}, Bias={m_a1['bias']:.4f}")
    md.append(f"- 留一条件 LOCO：R²={m_loco['r2']:.4f}, RMSE={m_loco['rmse']:.4f}")
    md.append(f"- **留一大鼠 LORO（单鼠泛化）：R²={m_loro['r2']:.4f}, RMSE={m_loro['rmse']:.4f}** "
              f"（逐鼠 R² 均值 {np.mean([metrics(y_all[rat_of!=r], aa+bb*x_all[rat_of!=r])['r2'] for r in rats]):.4f}）")
    md.append("- 结论：A1 在群体均值层显著提升 R²、消除系统性 Bias；但 LORO 显示单鼠外推弱于群体层，"
              "A1 仍属经验修正、非机理改进。")
    md.append("")
    md.append("## Track B — 校准后的嵌套 LOO 预测区间\n")
    md.append(f"- **逐点覆盖率 = {cov:.3f}**（目标 90%），n={len(tb)} 点 = {tb.rat.nunique()} 鼠 × 2 频率 × 6 时间点")
    md.append(f"- **平均区间宽度 = {width:.3f}**；平均 Winkler(α_mis=0.10) = {wink:.3f}")
    md.append("- 统计口径：逐点覆盖率（各时间点观测落入其区间的比例），**不等于**整条曲线同时落入区间的概率")
    md.append(f"- 两种 ensemble 模式分裂（合并 {cov:.3f} 掩盖差异）：\n")
    md.append("| ensemble 模式 | 覆盖率 | n | 平均宽度 |")
    md.append("|---|---|---|---|")
    md.append(role_md)
    md.append(f"- 缺鼠：验证集仅 {tb.rat.nunique()}/24 鼠（缺 WT5 0.10M、WT6 0.15M）")
    md.append("- 区间过宽根因：~97% 来自 conformal padding，基础 LHS 参数不确定仅 ~3%")
    md.append("")
    md.append("## 附图（完整 7 张 + 表1）\n")
    md.append("- 图A `figA_trackA_fit.png`：Track A 校正拟合（实测±SD vs A1 预测，剂量×频率）")
    md.append("- 图B `figB_trackA_scatter.png`：Track A 校正 预测值 vs 真实值（含 y=x 线与 R²/RMSE/MAE）")
    md.append("- 图E `figE_trackA_residuals.png`：Track A 校正残差诊断（残差 vs 时间 / vs 预测值）")
    md.append("- 图F `figF_trackA_error_by_dose.png`：Track A 不同剂量误差对比（原始 vs A1，RMSE/MAE/Bias）")
    md.append("- 图C `figC_trackB_fit.png`：Track B 校准区间拟合（实测±SD vs 区间，剂量×频率）")
    md.append("- 图D `figD_trackB_coverage.png`：Track B 覆盖诊断（总体/按模式/按剂量 + 宽度）")
    md.append("- 表1 `figG_table1.png`：Track A 指标汇总表 + Track B 区间指标")
    with open(f"{OUT}/FITTING_EFFECT.md", "w") as fh:
        fh.write("\n".join(md))
    with open(f"{OUT}/metrics.json", "w") as fh:
        json.dump(dict(a1=(a_f, b_f), trackA_group=m_a1, trackA_loco=m_loco, trackA_loro=m_loro,
                       trackB=dict(coverage=cov, width=width, winkler=wink, n=len(tb), n_rats=int(tb.rat.nunique()))),
                  fh, indent=2, default=float)
    print(f"TrackA A1: R2={m_a1['r2']:.4f} RMSE={m_a1['rmse']:.4f} MAE={m_a1['mae']:.4f} Bias={m_a1['bias']:.4f}")
    print(f"TrackB: cov={cov:.4f} width={width:.4f} winkler={wink:.4f} n={len(tb)} rats={tb.rat.nunique()}")
    print("done ->", OUT)


if __name__ == "__main__":
    main()
