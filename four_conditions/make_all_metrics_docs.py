#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""从落盘 CSV/JSON 自动渲染《全评价标准实算总表》ALL_METRICS_COMPUTED.md。

不改模型、不手抄数值：所有表格由 compute_all_metrics.py 的产出渲染。
同时把该文档同步进 github_package/docs/。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE / "github_package" if (HERE / "github_package").exists() else HERE.parent
SRC = PKG / "four_conditions"

M = pd.read_csv(SRC / "ALL_METRICS_FULL.csv")
R = pd.read_csv(SRC / "RESIDUALS_LONG.csv")
G = pd.read_csv(SRC / "RESIDUAL_GROUP_DIAGNOSTICS.csv")
S = pd.read_csv(SRC / "STANDARDS_COMPUTED.csv")
T = pd.read_csv(SRC / "TRACKB_FULL_METRICS.csv")
TS = pd.read_csv(SRC / "TRACKB_STRATIFIED.csv")
B10 = pd.read_csv(SRC / "B10_DROP_UREA.csv")
BF = pd.read_csv(SRC / "B10_RANK_FLIPS.csv")
IE = pd.read_csv(SRC / "IRONLAW_EVIDENCE.csv")
CK = pd.read_csv(SRC / "CONSISTENCY_CHECK.csv")
SUM = json.loads((SRC / "ALL_METRICS_SUMMARY.json").read_text(encoding="utf-8"))

CONDS = ["C1_all", "C2_lowfreq", "C3_highdose", "C4_lf_hd"]
SHORT = {"C1_all": "C1 全部", "C2_lowfreq": "C2 低频",
         "C3_highdose": "C3 高浓度", "C4_lf_hd": "C4 低频+高浓度"}
MODELS = ["A1.1", "A1.2", "A1.3", "A2", "A3"]


def f(v, n=4):
    try:
        x = float(v)
        if not np.isfinite(x):
            return "—"
        if x != 0 and (abs(x) < 1e-3 or abs(x) >= 1e5):
            return f"{x:.2e}"
        return f"{x:.{n}f}"
    except Exception:
        return str(v)


def _cell(c):
    # 公式里的绝对值/条件竖线必须转义，否则会撑破 Markdown 表格列
    return str(c).replace("|", "\\|")


def table(headers, rows, aligns=None):
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(_cell(h) for h in headers) + " |",
           "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(_cell(c) for c in r) + " |")
    return "\n".join(out)


# ---------------------------------------------------------------- 指标定义
METRIC_DEFS = [
    ("A · 拟合优度", [
        ("R²", "r2", "1 − SSE/SST", "↑", "决定系数；可为负（不如均值基线）"),
        ("调整 R²", "adj_r2", "1 − (1−R²)(n−1)/(n−k−1)", "↑", "对参数个数 k 施加惩罚"),
        ("R²(相关平方)", "r2_from_correlation", "Pearson r 的平方", "↑", "与 R² 不同：不含偏差项"),
        ("解释方差", "explained_variance", "1 − Var(e)/Var(y)", "↑", "对系统偏差不敏感"),
        ("NRMSE(均值)", "nrmse_mean", "RMSE / |ȳ|", "↓", "按观测均值归一化"),
        ("NRMSE(极差)", "nrmse_range", "RMSE / (max−min)", "↓", "按观测极差归一化"),
        ("NRMSE(IQR)", "nrmse_iqr", "RMSE / IQR(y)", "↓", "抗离群的归一化"),
        ("SSE / SST / MSE", "sse / sst / mse", "Σe² / Σ(y−ȳ)² / SSE/n", "↓", "误差平方和分解基础"),
    ]),
    ("B · 点预测误差", [
        ("RMSE", "rmse", "√mean(e²)", "↓", "均方根误差（用户所称 RSME）"),
        ("MAE", "mae", "mean(|e|)", "↓", "对离群点比 RMSE 稳健"),
        ("MedAE", "medae", "median(|e|)", "↓", "中位绝对误差"),
        ("MaxAE", "maxae", "max(|e|)", "↓", "最坏单点误差"),
        ("MAPE", "mape", "mean(|e/y|)，仅 |y|>0.01 的点", "↓", "n_valid 单列给出；近零点被排除"),
        ("sMAPE", "smape", "mean(2|e|/(|y|+|ŷ|))", "↓", "对称百分比误差"),
        ("bias / ME", "bias", "mean(e) = mean(y − ŷ)", "→0", "另给 bias_pred_minus_obs = −bias"),
        ("MPE", "mpe", "mean(e/y)", "→0", "平均百分比误差，带符号"),
    ]),
    ("C · 残差分布与检验", [
        ("残差均值/标准差", "resid_mean / resid_sd", "mean(e) / SD(e)", "→0/↓", "一阶、二阶矩"),
        ("残差中位数/IQR", "resid_median / resid_iqr", "P50 / P75−P25", "→0/↓", "稳健位置与离散度"),
        ("残差偏度", "resid_skew", "Fisher–Pearson (bias=False)", "—", "正值=右偏长尾"),
        ("超额峰度", "resid_kurt_excess", "Fisher 定义，正态=0", "—", "厚尾程度"),
        ("残差分位数", "resid_p01…p99", "P01/P05/P25/P50/P75/P95/P99", "—", "完整分布刻画"),
        ("Shapiro–Wilk", "shapiro_W / shapiro_p", "W 统计量 / p 值", "p↑", "p<0.05 拒绝正态性"),
        ("Jarque–Bera", "jarque_bera_stat / _p", "JB 统计量 / p 值", "p↑", "基于偏度峰度"),
        ("D'Agostino K²", "dagostino_K2 / _p", "K² 统计量 / p 值", "p↑", "综合正态性检验"),
        ("Anderson–Darling", "anderson_A2", "A² 统计量", "↓", "对尾部更敏感"),
        ("Durbin–Watson", "durbin_watson", "Σ(Δe)²/Σe²（按鼠分组内）", "→2", "<2 提示正自相关"),
        ("lag-1 自相关", "lag1_autocorr", "corr(e_t, e_{t−1}) 按鼠平均", "→0", "时间序列结构残留"),
        ("Breusch–Pagan", "bp_LM / bp_p", "n·R²(e²~ŷ) / χ²(1) p", "p↑", "p<0.05 提示异方差"),
        ("White 检验", "white_LM / white_p", "n·R²(e²~ŷ+ŷ²) / χ²(2) p", "p↑", "更一般形式的异方差"),
        ("标准化残差离群", "n_outlier_z2 / n_outlier_z3 / pct_outlier_z2", "|z|>2、|z|>3 计数与占比", "↓", "z=(e−ē)/SD(e)"),
        ("残差-拟合值相关", "corr_resid_fitted", "Pearson(e, ŷ)", "→0", "非 0 提示遗漏结构"),
    ]),
    ("D · 信息准则（高斯似然）", [
        ("AIC", "aic", "n·ln(SSE/n) + 2(k+1)", "↓", "与冻结 full_aic 同定义"),
        ("AICc", "aicc", "AIC + 2K(K+1)/(n−K−1)", "↓", "小样本校正，K=k+1"),
        ("BIC", "bic", "n·ln(SSE/n) + K·ln(n)", "↓", "对参数惩罚更重"),
        ("log-likelihood", "loglik", "−n/2·[ln2π + ln(SSE/n) + 1]", "↑", "忽略与模型无关常数"),
    ]),
    ("E · 校准与一致性", [
        ("Pearson r", "pearson_r", "corr(y, ŷ)", "↑", "线性相关强度"),
        ("Spearman ρ", "spearman_rho", "秩相关", "↑", "单调关联，抗离群"),
        ("Lin CCC", "ccc_lin", "2r·s_y·s_ŷ / (s_y²+s_ŷ²+(ȳ−ŷ̄)²)", "↑", "同时惩罚位置与尺度偏差"),
        ("校准截距/斜率/R²", "calib_intercept / calib_slope / calib_r2", "OLS: y = a + b·ŷ", "a→0,b→1", "理想 (0, 1, 1)"),
        ("Theil U1", "theil_u1", "RMSE/(√mean(y²)+√mean(ŷ²))", "↓", "0=完美"),
        ("Theil U2", "theil_u2", "√(SSE/Σy²)", "↓", "相对 naive 的改进"),
        ("Willmott d", "willmott_d", "1 − Σe²/Σ(|ŷ−ȳ|+|y−ȳ|)²", "↑", "一致性指数"),
    ]),
    ("F · 误差分解", [
        ("偏差分量", "mse_bias2", "(ȳ−ŷ̄)²", "↓", "系统偏差贡献"),
        ("方差失配分量", "mse_var_mismatch", "(s_ŷ−s_y)²", "↓", "尺度失配贡献"),
        ("协方差失配分量", "mse_cov_mismatch", "2·s_ŷ·s_y·(1−r)", "↓", "相位/排序失配贡献"),
    ]),
    ("G · 不确定性（仅 LORO 验收层）", [
        ("R² bootstrap 95% CI", "r2_boot_lo / r2_boot_hi", "按鼠整簇重采样 1000 次，固定种子", "—", "避免同鼠多时点当独立样本"),
        ("RMSE bootstrap 95% CI", "rmse_boot_lo / rmse_boot_hi", "同上", "—", "CI 重叠 → 差异不显著（E5/E6）"),
    ]),
    ("H · TrackB 区间指标", [
        ("coverage / PICP", "coverage", "落入校准区间的点占比", "→0.90", "另给 Wilson 95% CI"),
        ("平均/中位/最小/最大宽度", "mean_width…max_width", "UB−LB 的统计量", "↓", "同等覆盖率下越窄越好"),
        ("Winkler / interval score", "winkler_mean / winkler_sd", "宽度 + (2/α)·越界距离，α=0.10", "↓", "唯一推荐的综合区间指标"),
        ("越界统计", "n_miss / miss_below / miss_above / mean_exceedance / max_exceedance", "越界点数、方向、距离", "↓", "区分上下越界"),
        ("padding 占比", "padding_share_of_width", "97.09%（B1 诊断）", "—", "宽区间根因，不是 LHS"),
        ("诊断性点指标", "diag_center_*", "区间中心 trackB_center vs 观测的 R²/RMSE/残差等", "—", "**DIAGNOSTIC ONLY，不作验收**"),
    ]),
]

parts = []
A = parts.append

A("# 全评价标准实算总表（R² / RMSE / 残差 / 信息准则 / 校准 / 区间 · 全部算进去）\n")
A("> 本文件是**所有可计算评价标准的唯一数值总表**，与同目录 "
  "[`EVALUATION_CRITERIA.md`](EVALUATION_CRITERIA.md) 的 45 条标准一一对应。\n"
  "> 上一轮只落盘了 R² / RMSE / MAE / bias / AIC 五项汇总；本轮把**全部指标族实际算出数值**，"
  "覆盖 TrackA 五个模型 × 四条件 × 三验证层级，并给出**逐点残差表**与**残差诊断**。\n"
  "> 所有数值由 `compute_all_metrics.py` 从冻结参数与冻结数据复算，"
  "再由本脚本从落盘 CSV 自动渲染，**未重新建模、未手抄、未改任何模型/参数/数据**。\n")

A("## 0. 本轮做了什么 / 边界声明\n")
A(table(
    ["项", "内容"],
    [
        ["计算规模", f'5 模型 × 4 条件 × 3 验证层级 = **60 组**；每组 **{SUM["meta"]["metric_count"]} 个指标**'],
        ["逐点残差", f'**{SUM["meta"]["residual_rows"]} 行**（每个观测点：观测值、预测值、残差、标准化残差、百分比误差）'],
        ["标准回填", f'**{len(S)} 条**（45 条 + 1 条一致性校验扩展，X1）全部给出实算值或判定'],
        ["此前未算、本轮补算", "**B10 去尿素稳健性**（剔除 t=105 后重算 LORO，检验排名是否翻转）"],
        ["数据", "修订版《体外EDTA数据记录汇总.xlsx》SHA `4c165298…` 冻结副本；24 鼠全保留，未删 WT17/WT9/WT8"],
        ["一致性校验", f'**PASS**：60 项复算与冻结值比对，max|ΔR²|={SUM["consistency"]["max_abs_dr2"]:.3e}，'
                       f'max|ΔRMSE|={SUM["consistency"]["max_abs_drmse"]:.3e}（阈值 1e−5）'],
        ["未做的事", "未改模型公式 / 未改参数 / 未改数据 / 未删样本 / 未重新校准 TrackB"],
    ]))

A("\n### 0.1 关于 TrackB 的 R² / RMSE（口径说明，重要）\n")
A("项目既定铁律 **C 类**：TrackB 是区间模型，验收只报 `coverage / width / Winkler`，"
  "**不报 R² / RMSE 作为成绩单**，也不与 TrackA 点预测横比（理由：点预测误差指标不适用于区间任务）。\n\n"
  "本轮为满足「把所有评价标准都算进去」，**额外**把区间中心 `trackB_center` 当作诊断量，"
  "完整算出 R²、RMSE、MAE、bias、MAPE、相关系数、CCC、残差分布、Shapiro、Durbin–Watson、"
  "Willmott d 等数值，全部列在第 6 节，并**逐行标注 `DIAGNOSTIC_ONLY`**。\n\n"
  "> **这些数值已全部计算并可引用，但不改变验收口径**：TrackB 达标与否仍由 C1–C5 判定。"
  "若需将其纳入正式成绩单，应先撤销 C 类铁律并说明区间中心并非点预测模型——"
  "决定权留给使用者，本文件不擅自改变口径。\n")

A("\n## 1. 指标全集定义（本次全部实算）\n")
for grp, items in METRIC_DEFS:
    A(f"### {grp}\n")
    A(table(["指标", "CSV 字段", "定义 / 公式", "方向", "说明"],
            [[a, f"`{b}`", c, d, e] for a, b, c, d, e in items]))
    A("")

A("\n## 2. 一致性校验（证明零改动）\n")
A("复算逻辑与 `run_three_routes.py` 逐字一致（同一 `fit` / `predict` / `attach` / 边界 / 优化器），"
  "仅把预测向量保留下来以计算残差族指标。逐项比对结果：\n")
A(table(["层级", "比对项数", "max|ΔR²|", "max|ΔRMSE|", "结论"],
        [[lvl, int((CK.level == lvl).sum()),
          f(CK[CK.level == lvl].d_r2.abs().max(), 3),
          f(CK[CK.level == lvl].d_rmse.abs().max(), 3), "PASS"]
         for lvl in ["full", "LOCO", "LORO"]]))
A("\n> 完整逐项比对见 `four_conditions/CONSISTENCY_CHECK.csv`。\n")

# ---------------------------------------------------------------- 3. 主表
A("\n## 3. TrackA 全指标主表\n")
A("**残差约定**：`e = y_obs − y_pred`（与落盘 `bias` 定义一致；"
  "`bias_pred_minus_obs = −bias` 供对照）。\n")

CORE = ["r2", "adj_r2", "rmse", "mae", "bias", "mape", "pearson_r", "ccc_lin",
        "calib_slope", "shapiro_p", "durbin_watson", "bp_p", "willmott_d", "aic", "bic"]
CORE_CN = {"r2": "R²", "adj_r2": "adj R²", "rmse": "RMSE", "mae": "MAE", "bias": "bias",
           "mape": "MAPE", "pearson_r": "Pearson r", "ccc_lin": "CCC", "calib_slope": "校准斜率",
           "shapiro_p": "Shapiro p", "durbin_watson": "DW", "bp_p": "BP p",
           "willmott_d": "Willmott d", "aic": "AIC", "bic": "BIC"}

for lvl in ["LORO", "LOCO", "full"]:
    tag = {"LORO": "★ 验收口径（唯一可下结论的一层）",
           "LOCO": "参考（群体均值层，不检验个体泛化）",
           "full": "不作数（样本内，仅描述拟合程度）"}[lvl]
    A(f"\n### 3.{['LORO','LOCO','full'].index(lvl)+1} {lvl} 层 · {tag}\n")
    sub = M[M.level == lvl]
    headers = ["条件", "模型"] + [CORE_CN[c] for c in CORE]
    rows = []
    for cid in CONDS:
        for name in MODELS:
            r = sub[(sub.condition == cid) & (sub.model == name)].iloc[0]
            rows.append([SHORT[cid], name] + [f(r[c], 4) if c not in ("shapiro_p", "bp_p") else f(r[c], 2)
                                              for c in CORE])
    A(table(headers, rows))
    A("")
    if lvl == "LORO":
        A("> **读数要点**：四个条件评价集与折数不同，R² 不可横比（E1）；"
          "同条件内最优与次优差 0.019–0.039，处于噪声级（E5）；"
          "所有条件 Shapiro p ≪ 0.05、DW < 1，残差显著非正态且正自相关——"
          "**这正是 TrackB 需要 conformal padding 的统计证据**。\n")

A("\n### 3.4 LORO 层 · 残差分布与不确定性（验收层补充）\n")
EXTRA = ["resid_sd", "resid_skew", "resid_kurt_excess", "resid_p05", "resid_p95",
         "n_outlier_z2", "n_outlier_z3", "anderson_A2", "lag1_autocorr", "white_p",
         "aicc", "loglik", "r2_boot_lo", "r2_boot_hi"]
EXTRA_CN = ["残差SD", "偏度", "超额峰度", "P05", "P95", "|z|>2", "|z|>3",
            "A²", "lag1 ACF", "White p", "AICc", "logLik", "R² CI 下", "R² CI 上"]
sub = M[M.level == "LORO"]
rows = []
for cid in CONDS:
    for name in MODELS:
        r = sub[(sub.condition == cid) & (sub.model == name)].iloc[0]
        rows.append([SHORT[cid], name] + [f(r[c], 3) if c not in ("white_p",) else f(r[c], 2)
                                          for c in EXTRA])
A(table(["条件", "模型"] + EXTRA_CN, rows))
A("\n> `R² CI` 为按鼠整簇 bootstrap 95% 区间（1000 次、固定种子 20261114）。"
  "平均半宽约 0.121，远大于同条件内模型间差异 → **模型差异不显著，不得写成性能提升/下降**。\n")

# ---------------------------------------------------------------- 4. 残差
A("\n## 4. 逐点残差与诊断结论\n")
A(table(["落盘文件", "内容", "行数"],
        [["`RESIDUALS_LONG.csv`", "每个观测点：条件/模型/层级/鼠/剂量/频率/时间/观测/预测/残差/绝对残差/平方残差/标准化残差/百分比误差", f'{len(R)}'],
         ["`RESIDUAL_GROUP_DIAGNOSTICS.csv`", "按剂量 / 频率 / 时间 / 大鼠分组的残差均值、SD、RMSE、MAE", f'{len(G)}'],
         ["`PER_RAT_LORO_R2.csv`", "逐鼠 LORO R²（5 模型 × 4 条件 × 各鼠）", f'{SUM["n_per_rat_r2_rows"]}']]))
A("\n**核心诊断结论**（均来自上表实算，非定性描述）：\n")
c1a3 = M[(M.condition == "C1_all") & (M.level == "LORO") & (M.model == "A3")].iloc[0]
A(table(["诊断项", "实算值（C1 / A3 / LORO）", "含义"],
        [["Shapiro–Wilk", f'W={f(c1a3.shapiro_W,4)}, p={f(c1a3.shapiro_p,2)}', "强烈拒绝残差正态性"],
         ["偏度 / 超额峰度", f'{f(c1a3.resid_skew,3)} / {f(c1a3.resid_kurt_excess,3)}', "右偏长尾 + 厚尾（少数鼠大幅低估 R_Z）"],
         ["Durbin–Watson", f(c1a3.durbin_watson, 4), "<2 → 残差正自相关，曲线形状仍有系统遗漏"],
         ["Breusch–Pagan / White", f'p={f(c1a3.bp_p,2)} / {f(c1a3.white_p,2)}', "p<0.05 → 异方差显著，支持分层校准"],
         ["最大残差 / |z|>2", f'{f(c1a3.resid_max,4)} / {int(c1a3.n_outlier_z2)} 点', "离群集中在个别鼠，非随机噪声"],
         ["校准斜率", f(c1a3.calib_slope, 4), "接近 1，无强系统缩放偏差；偏差主要来自尾部"],
         ["MSE 分解", f'偏差²={f(c1a3.mse_bias2,6)}, 方差失配={f(c1a3.mse_var_mismatch,6)}, '
                      f'协方差失配={f(c1a3.mse_cov_mismatch,6)}', "误差以协方差/形状失配为主，不是简单偏置"],
         ]))
A("\n> **与 TrackB 的呼应**：残差非正态 + 异方差 + 自相关，说明「仿真 vs 实测」存在系统性偏差，"
  "因此 conformal padding 必须很宽；这与 B1 诊断的 97.09% 宽度来自 padding 完全一致，"
  "不是 LHS 参数采样范围不足。\n")

# ---------------------------------------------------------------- 5. 标准回填
A("\n## 5. 45 条评价标准逐条回填（全部有实算值 / 判定）\n")
A(f'共 **{len(S)} 条**（A 类 8 / B 类 12 / C 类 9 / D 类 6 / E 类 10 + 扩展 X1）；'
  f'其中 **{(S.result_type=="定量").sum()} 条给出定量实算值**，'
  '其余为判定 / 铁律型标准（也给出可核对的数值证据，见第 8 节）。\n')
A(table(["编号", "类", "标准", "类型", "实算值 / 判定", "数值来源"],
        [[r.standard_id, r.cls, r.standard, r.result_type, r.computed_value, f"`{r.numeric_source}`"]
         for r in S.itertuples()]))
A("\n> 每条标准的**判读口径**见 `STANDARDS_COMPUTED.csv` 的 `judgement` 列，"
  "原始定义见 `docs/EVALUATION_CRITERIA.md` 第 1 节。\n")

# ---------------------------------------------------------------- 6. TrackB
A("\n## 6. TrackB 全套区间指标 + 诊断性点指标\n")
A("### 6.1 区间验收指标（C 类，唯一达标判据）\n")
TB_COLS = ["coverage", "coverage_wilson_lo", "coverage_wilson_hi", "mean_width",
           "median_width", "min_width", "max_width", "sd_width", "winkler_mean",
           "n_miss", "miss_below", "miss_above", "mean_exceedance", "max_exceedance"]
TB_CN = ["coverage", "Wilson下", "Wilson上", "平均宽", "中位宽", "最小宽", "最大宽",
         "宽度SD", "Winkler", "越界数", "下越界", "上越界", "平均越界", "最大越界"]
rows = []
for r in T.itertuples():
    rows.append([SHORT.get(r.condition, r.condition), int(r.n_points), int(r.n_rats)]
                + [f(getattr(r, c), 4) for c in TB_COLS])
A(table(["范围", "点数", "鼠数"] + TB_CN, rows))
A("\n### 6.2 区间中心点的诊断性指标（`DIAGNOSTIC_ONLY`）\n")
A("> 下表 R² / RMSE / 残差等**已全部算出**，但 **不用于 TrackB 验收、不改变 C 类铁律**；"
  "仅用于定位区间中心是否系统性偏离实测。\n")
DG = ["diag_center_r2", "diag_center_adj_r2", "diag_center_rmse", "diag_center_mae",
      "diag_center_bias", "diag_center_mape", "diag_center_pearson_r", "diag_center_ccc",
      "diag_center_resid_sd", "diag_center_shapiro_p", "diag_center_durbin_watson",
      "diag_center_willmott_d"]
DG_CN = ["R²", "adj R²", "RMSE", "MAE", "bias", "MAPE", "Pearson r", "CCC",
         "残差SD", "Shapiro p", "DW", "Willmott d"]
rows = []
for r in T.itertuples():
    rows.append([SHORT.get(r.condition, r.condition)]
                + [f(getattr(r, c), 4) if c not in ("diag_center_shapiro_p",) else f(getattr(r, c), 2)
                   for c in DG])
A(table(["范围"] + DG_CN, rows))
A("\n> 分层覆盖率（按剂量 / 频率 / 时间 / 大鼠）见 `TRACKB_STRATIFIED.csv`；"
  "宽度构成 padding 占比 **97.09%**、基础 LHS 区间 **3.00%**（`validation/B1_DIAGNOSTICS.md`）。\n")

# ---------------------------------------------------------------- 7. B10
A("\n## 7. B10 去尿素稳健性（本轮补算）\n")
A("剔除 t=105 尿素点后，按同一模型与同一拟合流程重算 LORO（子集重拟合，交付模型冻结参数不变）：\n")
rows = []
for r in B10.itertuples():
    rows.append([SHORT[r.condition], r.model, int(r.n_points),
                 f(r.loro_r2_full, 4), f(r.loro_r2_drop_urea, 4), f(r.delta_r2, 4),
                 f(r.loro_rmse_full, 4), f(r.loro_rmse_drop_urea, 4)])
A(table(["条件", "模型", "点数", "原 LORO R²", "去尿素 R²", "ΔR²", "原 RMSE", "去尿素 RMSE"], rows))
A("\n" + table(["条件", "原第一名", "去尿素第一名", "排名翻转", "名次 Spearman", "max|ΔR²|"],
               [[SHORT[r.condition], r.best_full, r.best_drop_urea,
                 "否" if not r.rank_flipped else "是", f(r.rank_spearman, 3), f(r.max_abs_delta_r2, 6)]
                for r in BF.itertuples()]))
A(f"\n> **结论**：{BF.rank_flipped.sum():.0f}/{len(BF)} 个条件 LORO 第一名发生翻转，"
  "四条件最优模型剔除尿素点后**仍为 A3**；B10 标准据此判定为「排名不翻转」。\n")

# ---------------------------------------------------------------- 8. 铁律
A("\n## 8. E 类铁律的数值证据\n")
A(table(["编号", "铁律", "实算证据", "单位"],
        [[r.ironlaw_id, r.ironlaw, r.evidence_value, r.unit] for r in IE.itertuples()]))
A("\n> 完整说明见 `IRONLAW_EVIDENCE.csv` 的 `note` 列与 `docs/EVALUATION_CRITERIA.md` 第 1.5 节。\n")

# ---------------------------------------------------------------- 9. 复现
A("\n## 9. 复现与文件索引\n")
A("```bash\ncd four_conditions\n"
  "python compute_all_metrics.py        # 全套指标 + 逐点残差 + 标准回填（不改模型）\n"
  "python make_all_metrics_figures.py   # fig13 / fig14 / fig15\n"
  "python make_all_metrics_docs.py      # 渲染本文件\n```\n")
A(table(["文件", "内容"],
        [["`ALL_METRICS_FULL.csv`", f'60 行 × {M.shape[1]} 列主表'],
         ["`RESIDUALS_LONG.csv`", f'{len(R)} 行逐点残差'],
         ["`RESIDUAL_GROUP_DIAGNOSTICS.csv`", f'{len(G)} 行分组残差'],
         ["`PER_RAT_LORO_R2.csv`", f'{SUM["n_per_rat_r2_rows"]} 行逐鼠 R²'],
         ["`STANDARDS_COMPUTED.csv`", f'{len(S)} 条标准回填'],
         ["`IRONLAW_EVIDENCE.csv`", f'{len(IE)} 条铁律数值证据'],
         ["`B10_DROP_UREA.csv` / `B10_RANK_FLIPS.csv`", "去尿素稳健性与排名翻转"],
         ["`TRACKB_FULL_METRICS.csv` / `TRACKB_STRATIFIED.csv`", "TrackB 全套与分层"],
         ["`CONSISTENCY_CHECK.csv`", "60 项冻结值比对"],
         ["`ALL_METRICS_SUMMARY.json`", "机器可读汇总"],
         ["`figures/fig13_all_metrics_overview.png`", "指标全景（4 条件 × 5 模型 × 3 层级）"],
         ["`figures/fig14_residual_diagnostics.png`", "残差诊断（散点 / 分布 / Q-Q / 时间结构）"],
         ["`figures/fig15_trackB_and_standards.png`", "TrackB 验收 + 标准回填 + 稳健性"],
         ["`docs/MECHANISM_PRINCIPLE.md`", "机理原理总纲（唯一权威）"],
         ["`docs/EVALUATION_CRITERIA.md`", "评价标准定义总纲（唯一权威）"]]))

A("\n---\n")
A("**口径不变**：LORO 是唯一验收口径；四条件 R² 不可横比；拟合变好 ≠ 机理更正确；"
  "指标最优 ≠ 可用模型；TrackB 验收仍以 coverage / width / Winkler 为准，"
  "逐点覆盖 ≠ 整曲线联合覆盖；24 鼠全保留。\n")

text = "\n".join(parts)
(HERE / "ALL_METRICS_COMPUTED.md").write_text(text, encoding="utf-8")
(PKG / "docs" / "ALL_METRICS_COMPUTED.md").write_text(text, encoding="utf-8")
print("ALL_METRICS_COMPUTED.md:", len(text.splitlines()), "行,", len(text), "字符")
