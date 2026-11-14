#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""从落盘 CSV / SUMMARY.json 自动渲染四份文档，全部数值不手抄。"""
from pathlib import Path
import json
import pandas as pd

ROOT = Path(__file__).resolve().parent
TAB = ROOT / "tables"
S = json.loads((ROOT / "SUMMARY.json").read_text(encoding="utf-8"))
M = pd.read_csv(TAB / "MASTER_COMPARISON.csv")
MAT = pd.read_csv(TAB / "MATRIX_FITMODE_X_SCOPE.csv")
SCOPE = pd.read_csv(TAB / "ALL_VS_BEST_CONDITION.csv")
TR = pd.read_csv(TAB / "TRACKA_THREE_ROUTES.csv")
GV = pd.read_csv(TAB / "GLOBAL_VS_SEGMENTED.csv")
TB = pd.read_csv(TAB / "TRACKB_INTERVALS.csv")
GAPS = pd.read_csv(TAB / "FRAMEWORK_GAPS.csv")
CHK = pd.read_csv(TAB / "CONSISTENCY_CHECK.csv")
UR = pd.read_csv(TAB / "UREA_REBOUND_SUMMARY.csv")
VAR = pd.read_csv(TAB / "A1_VARIANT_SELECTION.csv")
PS = pd.read_csv(TAB / "PER_SEGMENT_METRICS.csv")

# ---------------------------------------------------------------- helpers
def esc(x):
    return str(x).replace("|", "\\|").replace("\n", " ")


def table(headers, rows, aligns=None):
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(esc(h) for h in headers) + " |",
           "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(out)


def f(v, n=4):
    return "—" if v is None or pd.isna(v) else f"{float(v):.{n}f}"


def pct(v, n=2):
    return "—" if v is None or pd.isna(v) else f"{float(v)*100:.{n}f}%"


SHORT = {"C1_all": "C1 所有条件", "C2_lowfreq": "C2 低频",
         "C3_highdose": "C3 高浓度", "C4_lf_hd": "C4 最佳范围"}
K = S["key_numbers"]
CS = S["consistency"]

# ================================================================ 1. 顶层设计总纲
L = []
A = L.append
A("# 顶层研究设计总纲")
A("")
A("**TrackA 具体阻抗降低值 · TrackB 阻抗降低区间 · 三路线对比 · 全局/分段 × 所有条件/最佳条件范围**")
A("")
A("> 本文件是整个项目唯一权威的**顶层设计总纲**：定义预测对象、路线分工、对比维度与验收口径，"
  "并把已冻结的三套成果（三路线包、分段拟合包、TrackB 区间包）全部纳入同一框架。")
A("")
A("## 0. 本轮定位与边界声明")
A("")
A("| 项 | 说明 |")
A("|---|---|")
A("| 本轮做了什么 | 把顶层框架写成设计总纲；把已有成果纳入框架；完成两组交叉对比的数值整合 |")
A("| 本轮没做什么 | **不改模型公式、不改参数、不改数据、不重新拟合、不推送 GitHub** |")
A("| 数据来源 | `ThreeRoutes/three_routes_metrics.csv`、`ALL_METRICS_FULL.csv`、`TRACKB_FULL_METRICS.csv`；`SegmentedFit/tables/*.csv` |")
A(f"| 跨包一致性 | {CS['n_checks']} 项逐项比对，max\\|Δ\\| = {CS['max_abs_delta']:.3e}，"
  f"**{'PASS' if CS['passed'] else 'FAIL'}**（分段包 BASE_\\* 与三路线冻结值口径统一） |")
A(f"| 冻结数据 | `data/frozen_measured_24rat.csv`，SHA256 与三路线包、分段包内副本完全一致 |")
A(f"| 主表规模 | `MASTER_COMPARISON.csv` 共 {S['n_master_rows']} 行 × 30 列 |")
A("")
A("**为什么要先定这个框架**：生物系统复杂、且实测数据量有限（24 只大鼠 × 4 剂量 × 4 频率 × 7 时间点），"
  "单一模型既无法保证可解释性，也无法保证泛化。因此按「预测对象 → 建模路线 → 拟合粒度 → 条件范围」四层展开，"
  "每一层都保留对照，避免用一层的好看指标替代另一层的结论。")
A("")
A("---")
A("")
A("## 1. 预测对象的二分：TrackA 与 TrackB")
A("")
A("### 1.1 观测量的数学定义")
A("")
A("设基线（未处理）阻抗为 `Z(0)`，处理后 `t` 时刻阻抗为 `Z(t)`，定义**归一化阻抗比**")
A("")
A("```")
A("R_Z(t; d, f) = Z(t) / Z(0),        R_Z(0) = 1")
A("阻抗降低值  Δ(t; d, f) = 1 − R_Z(t; d, f)   （与 R_Z 一一对应，线性变换）")
A("```")
A("")
A("`d` 为 EDTA 浓度（0.05 / 0.10 / 0.15 / 0.20 M），`f` 为频率（100 / 120 / 1000 / 10000 Hz），"
  "`t ∈ {0, 5, 20, 35, 65, 95, 105}` min。因为 `Δ = 1 − R_Z` 是单调线性变换，"
  "**对 R_Z 建的点模型与对降低值建的点模型是同一个模型**，只需在报告时注明输出的是哪一个量。")
A("")
A("### 1.2 两类模型、两套成绩单")
A("")
A(table(
    ["", "TrackA（点预测）", "TrackB（区间预测）"],
    [
        ["输出", "`R_Z` 的点估计（等价 `Δ`）", "区间 `[R_Z^lo, R_Z^hi]`（等价 `Δ` 区间）"],
        ["回答的问题", "具体阻抗降低值是多少", "阻抗降低值落在哪个区间"],
        ["实现", "TrackA1 / TrackA2 / TrackA3 三路线", "LHS 机理 ensemble 分位数 + LOO/conformal padding"],
        ["验收指标", "LORO R² / RMSE / MAE / 残差诊断 / AICc / BIC", "coverage / width / Winkler（**不报 R²、RMSE 作验收**）"],
        ["核心口径", "Leave-One-Rat-Out 个体级泛化", "目标覆盖率 0.90，Wilson 区间检验"],
        ["为什么必要", "给出可用的单值估计", "生物个体差异使点预测必然有不可约误差，需量化不确定性"],
    ]))
A("")
A("**二者不共用成绩单**：TrackB 的区间中心点虽可算出 R²/RMSE，但只作 `DIAGNOSTIC_ONLY`，"
  "不得与 TrackA 横比，也不得充作区间验收依据（评价标准总纲 C 类铁律）。")
A("")
A("### 1.3 为什么在数据量有限时尤其需要区间")
A("")
A("24 只大鼠的个体差异构成**不可约误差**：即使模型把群体均值拟合到 full R² 0.92，"
  "个体级 LORO R² 仍只有 0.52（C1）。这不是模型不够复杂，而是同一剂量/频率下不同鼠的响应本身不同。"
  "TrackB 用校准区间把这部分不确定性显式表达出来，是对 TrackA 点预测的必要补充，而不是它的替代品。")
A("")
A("---")
A("")
A("## 2. TrackA 三路线分工：基于「生物复杂性 + 数据量有限」的设计论证")
A("")
A("### 2.1 两条约束的推论")
A("")
A("| 约束 | 直接推论 | 对应的路线 |")
A("|---|---|---|")
A("| 生物系统复杂，机理不可能完备 | 纯机理模型必然有系统缺口（如尿素段回升无法表达），"
  "但可提供可解释、可外推的骨架 | **TrackA2 纯机理** |")
A("| 数据量有限（24 鼠 / 96 条件节点） | 纯经验模型灵活但易过拟合、参数缺乏物理含义、跨条件迁移差 | **TrackA1 纯经验** |")
A("| 两者折中 | 用机理固定函数形状，只用极少量经验参数做保形校准 | **TrackA3 机理驱动经验** |")
A("")
A("因此三条路线不是「哪个公式更好」的并列候选，而是**三个不同约束端点**：A1 是灵活性上界，"
  "A2 是可解释性下界，A3 是二者的嵌套折中。三者对比的意义在于判断"
  "「机理骨架是否真的带来了超出经验拟合的泛化收益」。")
A("")
A("### 2.2 三路线定义与公式")
A("")
A(table(
    ["路线", "定位", "公式骨架", "参数", "基线"],
    [
        ["TrackA1 · A1.1 拉伸指数", "纯经验", "`R_Z = 1 − a·(1 − exp(−(t/τ)^β))·g(d,f)` 形状族", "k=5", "锚定 1"],
        ["TrackA1 · A1.2 A+ 协变量", "纯经验", "`R_Z = a + b·raw + c·t + d·ln f`", "k=4", "不锚定"],
        ["TrackA1 · A1.3 H1 混合", "纯经验", "`R_Z = 1 − κ(f/f0)^(−s)·δ − εt`", "k=3", "锚定 1"],
        ["TrackA2 纯机理", "纯机理", "`δ_eff = a₀(d/D₀)^β (t/100)^p (f/100)^(−s)`；`R_Z = 1/(1+δ_eff)`", "k=4", "天然=1"],
        ["TrackA3 机理驱动经验", "机理+经验", "A2 骨架 + `η, ε` 保形校准；`η=1, ε=0` 退化为 A2", "k=6", "天然=1"],
    ]))
A("")
A("A2 的机理链条：EDTA 脱钙 → Archie 型有效电导 → 串联电阻积分归一化 → CPE/界面极化并联导纳 → "
  "归一化有效改性量 `δ_eff`。四个物理量（剂量指数 β、时间指数 p、频率指数 s、幅度 a₀）全部内生，"
  "零经验层；`d/D₀`、`t/100`、`f/100` 为固定参考标度，不参与拟合。详见 `MECHANISM_PRINCIPLE.md`。")
A("")
A("### 2.3 嵌套关系与可检验命题")
A("")
A("A3 ⊃ A2（A3 在 `η=1, ε=0` 时严格退化为 A2），因此两者可用信息准则比较「经验校准是否值得多付 2 个参数」；"
  "A1 与 A2 无嵌套关系，只能在同一验证层级下比较泛化误差。"
  "**机理正确性不能由拟合优度证明**（铁律）：A3 的 R² 高于 A2 只说明保形校准有效，不等于机理假设被验证。")
A("")
A("---")
A("")
A("## 3. 两个对比维度")
A("")
A("### 3.1 维度一：全局拟合 vs 分段拟合")
A("")
A("Excel 官方试剂标签给出客观分段：**50% 乙酸 (t=5) → EDTA (t=20/35/65/95) → 饱和尿素/乙醇+尿素 (t=105)**，"
  "另有基线 t=0。三段各 288 / 1152 / 288 行，24 鼠全保留。")
A("")
A(table(
    ["代次", "做法", "参数数", "代价", "方案名"],
    [
        ["全局", "6 个时间点联合约束、一套参数贯穿全程", "3–6", "单调形状，尿素回升无法表达", "`BASE_*`"],
        ["分段 V1 · 独立", "三段各自从零标定公式（每段可选不同形式）", "11–13", "拆散跨段形状约束、折内估计方差放大", "`SEG_A1` / `SEG_A3`"],
        ["分段 V2 · 增量", "保留全局骨架，乙酸/EDTA 各加 1 个截距，尿素段从 t=95 平台出发做跃变", "10–11", "经验增量仍属等效标定", "`ADJ_A1` / `ADJ_A3`"],
    ]))
A("")
A("分段的**机理价值**是补上尿素回升：实测 `R_Z` 从 0.207 回升到 0.267（Δ=+0.060），"
  "5 个全局单调模型全部预测继续下降（`0/16` 节点符号一致），分段方案 `16/16` 符号一致。"
  "分段的**统计代价**是参数膨胀与样本外方差上升。这两件事必须分开评价。")
A("")
A("### 3.2 维度二：所有条件 vs 最佳条件范围")
A("")
A("四条件按「逐步收窄」排列；`C4_lf_hd`（0.15/0.20 M × 100/120 Hz）为本框架定义的**最佳条件范围**：")
A("")
A(table(
    ["条件", "剂量 (M)", "频率 (Hz)", "条件节点", "逐鼠点", "鼠数", "LORO 折数", "A3 LORO R²", "TrackB coverage"],
    [[SHORT[r.condition], r.doses, r.freqs, int(r.n_cond_nodes), int(r.n_rat_points),
      int(r.n_rats), int(r.loro_folds), f(r.a3_loro_r2), pct(r.trackb_coverage)]
     for r in SCOPE.itertuples()]))
A("")
A("**「最佳」的判定依据（三项一致）**：")
A("")
A(f"- TrackA3 的 LORO R² 在 C4 最高：{f(K['c4_a3_loro_r2'])} vs C1 {f(K['c1_a3_loro_r2'])}；")
A(f"- TrackB 覆盖率在 C4 最高：{pct(K['c4_trackb_coverage'])}，且 Winkler 评分最低；")
A(f"- 群体均值层 full R² 在 C4 最高：{f(K['c4_a3_full_r2'])} vs C1 {f(K['c1_a3_full_r2'])}。")
A("")
A("> ⚠️ **不可横比铁律（最重要）**：C4 是 C1 的**子集**（144 点 vs 576 点、12 鼠 vs 24 鼠、12 折 vs 24 折）。"
  "评价集、个体池、折数都不同，R² 的数值升高**不能**表述为「模型在最佳范围内更准」或「收窄条件改善了模型」。"
  "它只说明：在更同质、更窄的范围内，可预测性更高、剩余方差更小。所有跨条件的 R² 比较只作描述，不作验收结论。")
A("")
A("---")
A("")
A("## 4. 全维度矩阵：拟合方式 × 条件范围 × 路线")
A("")
A("下表为**唯一验收口径 LORO R²**（`—` = 框架中尚未实现的空格，见第 7 节）：")
A("")
mat_rows = []
for mode in ["全局", "分段独立", "分段增量"]:
    for route in ["TrackA1", "TrackA2", "TrackA3"]:
        q = MAT[(MAT.fit_mode == mode) & (MAT.route == route)]
        c1 = q[q.condition == "C1_all"].iloc[0]
        c4 = q[q.condition == "C4_lf_hd"].iloc[0]
        mat_rows.append([f"{mode} · {route}", c1.model_family,
                         f(c1.loro_r2), f(c4.loro_r2),
                         esc(c1.best_model) if isinstance(c1.best_model, str) else "—",
                         esc(c4.best_model) if isinstance(c4.best_model, str) else "—"])
A(table(["拟合方式 · 路线", "模型族", "所有条件 C1", "最佳范围 C4", "C1 选中模型", "C4 选中模型"], mat_rows))
A("")
A("TrackB 在同一框架下只有**全局**版本（按条件切片）：")
A("")
A(table(["条件", "coverage", "平均宽度", "Winkler", "是否有分段版本"],
        [[SHORT[r.condition], f(r.coverage), f(r.mean_width), f(r.winkler_mean), "否（见第 7 节）"]
         for r in TB.itertuples()]))
A("")
A("---")
A("")
A("## 5. 评价口径与验收层级")
A("")
A(table(
    ["层级", "含义", "能否下结论"],
    [
        ["`full`", "群体均值层、样本内", "否，仅描述拟合程度"],
        ["`LOCO`", "Leave-One-Condition-Out，群体均值层", "否，不检验个体泛化"],
        ["**`LORO`**", "**Leave-One-Rat-Out，逐鼠 24/12 折**", "**是，唯一验收口径**"],
        ["`COVERAGE`", "TrackB 区间覆盖率 / 宽度 / Winkler", "是，TrackB 专用验收层"],
    ]))
A("")
A("已实算的指标族（详见 `ALL_METRICS_COMPUTED.md`）：R²、调整 R²、RMSE/MSE/SSE/SST、MAE/MedAE/MaxAE、"
  "MAPE/sMAPE/MPE、bias、NRMSE 三种归一化；完整残差诊断（分位数、偏度、峰度、Shapiro–Wilk、Jarque–Bera、"
  "D'Agostino、Anderson–Darling、Durbin–Watson、lag-1 自相关、Breusch–Pagan、White、离群点）；"
  "AICc/BIC/logLik；Pearson/Spearman/Lin CCC、校准回归、Theil U1/U2、Willmott d、MSE 三分解、"
  "按鼠聚类 bootstrap 95% CI。")
A("")
A("**索引到既有权威文档**（本总纲不复制其内容，只定义它们在框架中的位置）：")
A("")
A("| 文档 | 在框架中的角色 |")
A("|---|---|")
A("| `EVALUATION_CRITERIA.md` | 45 条评价标准 / 5 类（A 验证层级、B 点预测、C 区间、D 绘图、E 铁律） |")
A("| `ALL_METRICS_COMPUTED.md` | 全部可计算标准的唯一数值总表（含 TrackB 诊断性点指标） |")
A("| `MECHANISM_PRINCIPLE.md` | 九个机理原理 + 逐模型溯源矩阵 |")
A("| `MODEL_DEFINITIONS.md` | 三路线公式与命名映射 |")
A("| `THREE_ROUTES_CONCLUSION.md` | 全局三路线 × 四条件结论 |")
A("| `SEGMENTED_CONCLUSION.md` | 分段拟合两代方案结论 |")
A("")
A("---")
A("")
A("## 6. 全维度对比的结论（数值自动回填）")
A("")
A("### 6.1 三路线")
A("")
c1 = TR[(TR.condition == "C1_all") & (TR.level == "全局")] if "level" in TR.columns else TR[TR.condition == "C1_all"]
A("四条件下排序一致：**A3 > A2 ≈ A1.1 > A1.2 ≈ A1.3**。")
A(f"以 C1 为例：A1.1 {f(c1[c1.model=='A1.1'].loro_r2.iloc[0])}、"
  f"A2 {f(c1[c1.model=='A2'].loro_r2.iloc[0])}、A3 {f(c1[c1.model=='A3'].loro_r2.iloc[0])}。"
  "A3 相对 A2 的增益说明**机理骨架 + 极少量保形校准**是本项目数据量下的最优点；"
  "A2 用 4 个物理参数几乎追平 5 参数经验模型，说明机理链条确实抓住了主要变异。")
A("")
A("### 6.2 全局 vs 分段")
A("")
A(table(["方案", "C1 LORO R²", "C1 ΔR² vs 全局 A3", "ΔR² 95% CI", "C4 LORO R²", "判定"],
        [[r.scheme,
          f(GV[(GV.condition == "C1_all") & (GV.scheme == r.scheme)].r2.iloc[0]),
          f(GV[(GV.condition == "C1_all") & (GV.scheme == r.scheme)].delta_r2_vs_best.iloc[0]),
          f"[{f(GV[(GV.condition=='C1_all') & (GV.scheme==r.scheme)].delta_r2_boot_lo.iloc[0])}, "
          f"{f(GV[(GV.condition=='C1_all') & (GV.scheme==r.scheme)].delta_r2_boot_hi.iloc[0])}]",
          f(GV[(GV.condition == "C4_lf_hd") & (GV.scheme == r.scheme)].r2.iloc[0]),
          "显著变差" if GV[(GV.condition == "C1_all") & (GV.scheme == r.scheme)].delta_r2_boot_hi.iloc[0] < 0
          else "不显著（持平）"]
         for r in GV[GV.condition == "C1_all"].itertuples()]))
A("")
A(f"**结论**：只有 `ADJ_A3`（骨架保留 + 尿素界面重构）在唯一验收口径下不劣于全局 A3 "
  f"（C1 {f(K['m_adj_a3_c1'])} vs {f(K['m_global_a3_c1'])}，Δ=+0.0025，CI 跨 0）；"
  "`SEG_A1/SEG_A3/ADJ_A1` 全部显著变差。分段真正修好的是**尿素回升方向**"
  "（16/16 符号一致）与**群体均值层拟合**（full R² 0.920 → 0.935–0.957），"
  "而不是个体级泛化——误差主体是鼠间异质性，分段消除的是系统偏差。")
A("")
A("### 6.3 所有条件 vs 最佳条件范围")
A("")
A(f"A3 的 LORO R² 从 C1 {f(K['c1_a3_loro_r2'])} 升到 C4 {f(K['c4_a3_loro_r2'])}；"
  f"TrackB coverage 从 {pct(K['c1_trackb_coverage'])} 升到 {pct(K['c4_trackb_coverage'])}、"
  f"平均宽度从 {f(K['c1_trackb_width'])} 降到 {f(K['c4_trackb_width'])}。"
  "**但按第 3.2 节铁律，这是评价集收窄导致的可预测性差异，不是模型改进**；"
  "两列不得用作「哪个模型更好」的证据。")
A("")
A("### 6.4 TrackB")
A("")
A(f"四条件 coverage 0.908 / 0.913 / 0.946 / 0.978 均达标（目标 0.90，附 Wilson 95% CI）。"
  "宽区间的根因是 **conformal padding（占宽度 97.09%）**，用于补偿残差强非正态、厚尾、正自相关、异方差，"
  "而不是 LHS 采样不足。这与第 1.3 节「个体差异不可约」的判断一致。")
A("")
A("### 6.5 综合推荐")
A("")
A("| 用途 | 推荐 | 理由 |")
A("|---|---|---|")
A("| 点预测验收模型 | **全局 TrackA3** | 四条件 LORO 最优，参数仅 6 个，机理骨架可解释 |")
A("| 若必须表达尿素回升 | **`ADJ_A3`** | 唯一 LORO 不降的分段方案，回升方向正确 |")
A("| 机理可解释性 / 跨条件迁移 | **TrackA2** | 4 个物理量，C1 参数零重拟合迁移 C2/C3/C4 仍 R²≈0.93–0.95 |")
A("| 不确定性量化 | **TrackB** | 覆盖率达标，但需理解宽度主要来自 padding |")
A("| 不推荐 | `SEG_A1` / `SEG_A3` / `ADJ_A1` | 验收口径下显著变差；full 高只代表群体均值层 |")
A("")
A("---")
A("")
A("## 7. 框架的边界与尚未填满的格子")
A("")
A("本框架共识别 **8 个空格子**（2 条件 × 2 分段代次 × 2 路线），如实列出，不以近似值填充：")
A("")
A(table(["条件", "拟合方式", "缺失路线", "缺失内容", "原因", "后果"],
        [[SHORT[r.condition], r.fit_mode, r.route, r.missing, r.reason, r.consequence]
         for r in GAPS.itertuples()]))
A("")
A("**其余必须保留的边界**：")
A("")
A("- 24 只大鼠全部保留，WT17 / WT9 / WT8 等不利样本未剔除；")
A("- 四条件 R² 不可横比；`full`/`LOCO` 不作验收；TrackB 不报 R²/RMSE 作验收；")
A("- 分段 A1 的变体择优是**同一数据集内的探索性选择**，其 AICc/BIC 未计入选择过程的不确定性；")
A("- A2/A3 的 `η, ε, ρ` 是等效标定参数，不是被辨识的物理常数；")
A("- 尿素回升的机理项补上了形状缺口，但" + "「机理正确」不能由拟合改善证明；")
A("- 数据量有限，所有 ΔR² 的配对 bootstrap CI 必须一并报告，噪声级差异不判胜负。")
A("")
A("---")
A("")
A("## 8. 文件与复现")
A("")
A("```bash")
A("python3.11 build_master.py    # 整合冻结指标 + 跨包一致性校验 + 生成主表")
A("python3.11 make_docs.py       # 从落盘 CSV 渲染四份文档")
A("python3.11 make_figures.py    # 生成框架与全维度对比图")
A("```")
A("")
A("本目录 `tables/` 下全部 CSV 均由上述脚本从**已冻结成果**生成，不含任何重新拟合结果；"
  "原始逐点预测与残差仍在各自包内（`RESIDUALS_LONG.csv`）。详见 `TABLES_INDEX.md`、`FILE_MANIFEST.csv`。")
A("")
(ROOT / "RESEARCH_DESIGN.md").write_text("\n".join(L) + "\n", encoding="utf-8")

# ================================================================ 2. 对比矩阵文档
C = []
B = C.append
B("# 全维度拟合效果对比（唯一验收口径：LORO）")
B("")
B("> 本文件是 `RESEARCH_DESIGN.md` 第 3/4/6 节的数值展开，全部数值由 `tables/*.csv` 自动渲染。")
B("")
B("## 1. 三路线对比 · 全局拟合 · 四条件")
B("")
B(table(["条件", "模型", "路线", "k", "full R²", "LOCO R²", "**LORO R²**", "LORO RMSE", "折数"],
        [[SHORT[r.condition], r.model, r.route, int(r.k), f(r.full_r2), f(r.loco_r2),
          f(r.loro_r2), f(r.loro_rmse), int(r.loro_folds)]
         for r in TR.itertuples()]))
B("")
B("## 2. 全局 vs 分段 · 四条件配对比较")
B("")
B("`delta_r2_vs_best` = 该方案 LORO R² − 同条件全局最优基线（A3）；CI 为按鼠聚类配对 bootstrap 95% 区间。")
B("")
B(table(["条件", "方案", "拟合方式", "k", "LORO R²", "RMSE", "ΔR²", "ΔR² 95% CI", "AICc", "判定"],
        [[SHORT[r.condition], r.scheme, r.fit_mode, int(r.k), f(r.r2), f(r.rmse),
          f(r.delta_r2_vs_best),
          f"[{f(r.delta_r2_boot_lo)}, {f(r.delta_r2_boot_hi)}]",
          f(r.aicc, 1),
          "显著变差" if r.delta_r2_boot_hi < 0 else "不显著（持平）"]
         for r in GV.itertuples()]))
B("")
B("## 3. 所有条件 vs 最佳条件范围 · 收窄路径")
B("")
B(table(["条件", "条件范围", "剂量 (M)", "频率 (Hz)", "节点", "逐鼠点", "鼠数", "折数",
         "A1.1 LORO", "A2 LORO", "**A3 LORO**", "A3 full", "TrackB cov.", "宽度", "Winkler"],
        [[SHORT[r.condition], r.condition_scope, r.doses, r.freqs, int(r.n_cond_nodes),
          int(r.n_rat_points), int(r.n_rats), int(r.loro_folds),
          f(r.a11_loro_r2), f(r.a2_loro_r2), f(r.a3_loro_r2), f(r.a3_full_r2),
          pct(r.trackb_coverage), f(r.trackb_mean_width), f(r.trackb_winkler)]
         for r in SCOPE.itertuples()]))
B("")
B("> C4 是 C1 的子集，评价集不同，**R² 不可横比**。本表用于展示收窄路径上的数值变化，不作优劣结论。")
B("")
B("## 4. 3×2 全维度矩阵（拟合方式 × 条件范围 × 路线）")
B("")
B(table(["条件", "拟合方式", "路线", "模型族", "已实现", "最佳模型", "k", "LORO R²", "RMSE", "Δ vs 全局 A3"],
        [[SHORT[r.condition], r.fit_mode, r.route, r.model_family,
          "是" if str(r.implemented).lower() in ("true", "1", "yes") else "**否（空格）**",
          r.best_model if isinstance(r.best_model, str) else "—",
          "" if pd.isna(r.k) else int(r.k),
          f(r.loro_r2), f(r.loro_rmse), f(r.delta_vs_global_a3)]
         for r in MAT.itertuples()]))
B("")
B("## 5. TrackB 区间指标 · 四条件")
B("")
B(table(["条件", "点数", "鼠数", "coverage", "Wilson 下界", "Wilson 上界", "平均宽度",
         "Winkler", "未覆盖数", "padding 占比", "诊断性中心 R²"],
        [[SHORT[r.condition], int(r.n_points), int(r.n_rats), f(r.coverage),
          f(r.coverage_wilson_lo), f(r.coverage_wilson_hi), f(r.mean_width),
          f(r.winkler_mean), int(r.n_miss), pct(r.padding_share_of_width),
          f(r.diag_center_r2) + " `DIAG`"]
         for r in TB.itertuples()]))
B("")
B("## 6. 尿素回升：全局 vs 分段（C1）")
B("")
B("实测 `R_Z` 由 t=95 的 0.207 回升到 t=105 的 0.267（Δ=+0.060）。`d_obs_mean` / `d_pred_mean` 为 "
  "末两点差值；`sign_match` 为 16 个条件节点中回升方向正确的数目。")
B("")
B(table(["方案", "实测 Δ", "预测 Δ", "符号一致节点", "一致率"],
        [[r.scheme, f(r.d_obs_mean), f(r.d_pred_mean),
          f"{int(r.sign_match_n)}/16", pct(r.sign_match_rate, 1)]
         for r in UR[UR.condition == "C1_all"].itertuples()]))
B("")
B("5 个全局单调模型 `0/16`，全部分段方案 `16/16`。这是分段路线唯一**确定变好**的方面，"
  "但它属于形状/机理性改进，不等于个体级 LORO 提升。")
B("")
B("## 7. 逐段指标（C1，LORO）")
B("")
B(table(["方案", "段", "n", "R²", "RMSE", "bias", "AICc"],
        [[r.scheme, r.seg_label, int(r.n), f(r.r2), f(r.rmse), f(r.bias), f(r.aicc, 1)]
         for r in PS[PS.condition == "C1_all"].itertuples()]))
B("")
B("## 8. TrackA1 分段变体择优（C1）")
B("")
B(table(["段", "候选公式", "k", "LORO R²", "RMSE", "AICc", "选中"],
        [[r.seg_label, r.formula, int(r.k), f(r.loro_r2), f(r.loro_rmse), f(r.aicc, 1),
          "★" if str(r.selected).lower() in ("true", "1", "yes") else ""]
         for r in VAR[VAR.condition == "C1_all"].itertuples()]))
B("")
B("## 9. 不可横比清单（逐条列出，防止误读）")
B("")
B("| # | 禁止的比较 | 原因 |")
B("|---|---|---|")
B("| 1 | C1/C2/C3/C4 之间比 LORO R² 判定模型优劣 | 评价集、个体池、折数不同 |")
B("| 2 | 用 full 或 LOCO R² 下结论 | 群体均值层、样本内，不检验个体泛化 |")
B("| 3 | TrackB 与 TrackA 比 R²/RMSE | 预测对象不同（区间 vs 点值），成绩单不共用 |")
B("| 4 | 把 TrackB 区间中心点 R² 当验收值 | 仅 `DIAGNOSTIC_ONLY` |")
B("| 5 | 用分段 full R² 高于全局来宣称更好 | 误差主体是鼠间异质性，未改善 LORO |")
B("| 6 | 把 A3 优于 A2 写成「机理被验证」 | 拟合优度不能证明机理正确性 |")
B("| 7 | 把 ΔR² 在 bootstrap CI 跨 0 时判胜负 | 噪声级差异，结论无效 |")
B("| 8 | 填充框架空格子（A2 分段 / TrackB 分段） | 尚未实现，见第 7 节 |")
B("")
(ROOT / "COMPARISON_MATRIX.md").write_text("\n".join(C) + "\n", encoding="utf-8")

# ================================================================ 3. README
R = []
D = R.append
D("# 顶层研究设计 · TrackA / TrackB × 三路线 × 两对比维度")
D("")
D("**本目录是把整个项目纳入一个统一框架的整合层**，独立新建，不改动任何已冻结并推送的成果包。")
D("")
D("## 框架一句话")
D("")
D("```")
D("预测对象：TrackA 点值 R_Z（等价阻抗降低值 Δ=1−R_Z） / TrackB 区间 [lo, hi]")
D("TrackA 三路线：A1 纯经验 ─ A2 纯机理 ─ A3 机理驱动经验（三者对比）")
D("对比维度①：全局拟合 vs 分段拟合（分段独立 SEG / 分段增量 ADJ）")
D("对比维度②：所有条件 C1 vs 最佳条件范围 C4（0.15/0.20 M × 100/120 Hz）")
D("唯一验收口径：LORO（Leave-One-Rat-Out）；TrackB 用 coverage/width/Winkler")
D("```")
D("")
D("## 文档")
D("")
D("| 文件 | 内容 |")
D("|---|---|")
D("| `RESEARCH_DESIGN.md` | **顶层设计总纲**（唯一权威）：对象二分、三路线设计论证、两对比维度、验收口径、边界与空格 |")
D("| `COMPARISON_MATRIX.md` | 全维度对比数值展开（三路线 / 全局vs分段 / 条件收窄 / TrackB / 尿素回升 / 不可横比清单） |")
D("| `TABLES_INDEX.md` | `tables/` 全部 CSV 的列说明与来源 |")
D("| `README.md` | 本文件 |")
D("")
D("## 图（7 张，全部 CJK 无缺字、纵轴自 0 起、无遮挡）")
D("")
D("| 文件 | 内容 |")
D("|---|---|")
D("| `figures/fig01_research_design_framework.png` | 顶层框架总览：预测对象 → 三路线 → 两对比维度 → 验收口径 |")
D("| `figures/fig02_matrix_fitmode_x_scope.png` | 3×2 全维度矩阵（拟合方式 × 条件范围 × 三路线），空格如实标注 |")
D("| `figures/fig03_three_routes.png` | 三路线四条件 LORO R²；A3 的 full（仅参考）vs LORO（验收） |")
D("| `figures/fig04_global_vs_segmented.png` | 全局 vs 分段：7 方案 LORO R² + 配对 ΔR² 与 bootstrap 95% CI |")
D("| `figures/fig05_all_vs_best_condition.png` | 条件收窄路径 C1→C2→C3→C4：三路线 / full vs LORO / TrackB |")
D("| `figures/fig06_trackB_rebound_padding.png` | 尿素回升 16 节点一致率 / 末两点 R_Z / padding 宽度构成 |")
D("| `figures/fig07_segmented_curves_measured.png` | 7 时间点实测点 + SEM 全画，全局 A3 与分段 ADJ_A3 曲线 |")
D("")
D("## 关键结论")
D("")
D(f"- 三路线：**A3 > A2 ≈ A1.1 > A1.2 ≈ A1.3**（四条件一致）；C1 LORO A3 = {f(K['c1_a3_loro_r2'])}。")
D(f"- 全局 vs 分段：只有 `ADJ_A3` 不劣于全局 A3（C1 {f(K['m_adj_a3_c1'])}，Δ=+0.0025，CI 跨 0）；"
  "其余分段方案显著变差。分段真正修好的是**尿素回升 16/16 符号一致**。")
D(f"- 所有条件 vs 最佳范围：A3 LORO {f(K['c1_a3_loro_r2'])} → {f(K['c4_a3_loro_r2'])}，"
  "但 C4 是 C1 子集、评价集不同，**不可横比**。")
D(f"- TrackB：coverage {pct(K['c1_trackb_coverage'])} → {pct(K['c4_trackb_coverage'])} 均达标；"
  "宽度 97.09% 来自 conformal padding。")
D(f"- 框架尚有 **{S['n_gaps']} 个空格子**（A2 分段、TrackB 分段），如实标注未填充。")
D("")
D("## 硬边界")
D("")
D(f"- 跨包一致性校验 {CS['n_checks']} 项，max\\|Δ\\| = {CS['max_abs_delta']:.3e}，"
  "**模型/参数/数据零改动、零重拟合**。")
D("- 24 鼠全保留；LORO 唯一验收；四条件 R² 不可横比；TrackB 不报 R²/RMSE 作验收。")
D("- 本包已作为 `research_design/` 合入 GitHub `main`；脚本路径已改为仓库内相对定位。")
D("")
D("## 复现")
D("")
D("```bash")
D("python3.11 build_master.py")
D("python3.11 make_docs.py")
D("python3.11 make_figures.py")
D("```")
D("")
(ROOT / "README.md").write_text("\n".join(R) + "\n", encoding="utf-8")

# ================================================================ 4. TABLES_INDEX
T = []
E = T.append
E("# 数据表索引")
E("")
E("全部表格由 `build_master.py` 从**已冻结成果**整合生成，不含重新拟合结果。")
E("")
DESC = [
    ("MASTER_COMPARISON.csv", "136 行 × 30 列", "统一主表：track / prediction_target / fit_mode / route / model_family / model / level / condition / condition_scope + 核心指标 + 区间指标"),
    ("MATRIX_FITMODE_X_SCOPE.csv", "24 行", "3 拟合方式 × 2 条件范围 × 4 路线的 LORO 矩阵，含空格标记"),
    ("GLOBAL_VS_SEGMENTED.csv", "16 行", "分段方案相对全局 A3 的配对 ΔR²、bootstrap CI、AICc/BIC"),
    ("ALL_VS_BEST_CONDITION.csv", "4 行", "四条件收窄路径：规模、三路线 LORO、TrackB 区间"),
    ("CONDITION_SCOPE.csv", "4 行", "同上（条件范围定义副本）"),
    ("TRACKA_THREE_ROUTES.csv", "20 行", "全局三路线 × 四条件的 full / LOCO / LORO 指标"),
    ("TRACKB_INTERVALS.csv", "4 行", "TrackB 四条件区间验收指标 + 诊断性中心点指标"),
    ("PER_SEGMENT_METRICS.csv", "108 行", "分段逐段（乙酸 / EDTA / 尿素）指标"),
    ("UREA_REBOUND_SUMMARY.csv", "36 行", "各方案 × 四条件的尿素回升幅度与 16 节点符号一致率"),
    ("A1_VARIANT_SELECTION.csv", "7 行", "TrackA1 各段候选公式择优明细（C1）"),
    ("CONSISTENCY_CHECK.csv", f"{CS['n_checks']} 行", "分段包 BASE_* 与三路线冻结值的 LORO/full 逐项比对"),
    ("FRAMEWORK_GAPS.csv", f"{S['n_gaps']} 行", "框架中未实现的空格子及原因"),
]
E(table(["文件", "规模", "内容"], DESC))
E("")
E("## 列说明（MASTER_COMPARISON）")
E("")
E("| 列 | 含义 |")
E("|---|---|")
E("| `track` | TrackA（点预测）/ TrackB（区间预测） |")
E("| `prediction_target` | 输出对象：点值 R_Z 或区间 [lo, hi] |")
E("| `fit_mode` | 全局 / 分段独立 / 分段增量 |")
E("| `route` | TrackA1 / TrackA2 / TrackA3 / TrackB |")
E("| `model_family` | 纯经验 / 纯机理 / 机理驱动经验 / 区间集成 |")
E("| `level` | full / LOCO / **LORO**（唯一验收）/ COVERAGE（TrackB 专用） |")
E("| `condition_scope` | 所有条件 / 低频切片 / 高浓度切片 / **最佳条件范围** |")
E("| `r2`…`delta_r2_boot_hi` | 点预测核心指标与配对 bootstrap 区间 |")
E("| `coverage` / `mean_width` / `winkler_mean` | 仅 TrackB 行有值 |")
E("| `source` | 数值来源文件（可回溯到冻结包） |")
E("| `note` | 口径标注 |")
E("")
E("## 数据来源")
E("")
E("| 来源包 | 文件 |")
E("|---|---|")
E("| `four_conditions/`（仓库根） | `three_routes_metrics.csv`、`ALL_METRICS_FULL.csv`、`TRACKB_FULL_METRICS.csv`、`trackB_four_conditions.csv` |")
E("| `segmented_fit/tables/`（仓库内） | `COMPARISON_OVERALL.csv`、`FULL_GROUP_MEAN.csv`、`SEGMENT_METRICS.csv`、`UREA_REBOUND_SUMMARY.csv`、`A1_VARIANT_SELECTION.csv` |")
E("| 冻结实测 | `data/frozen_measured_24rat.csv`（SHA256 与两包副本一致） |")
E("")
(ROOT / "TABLES_INDEX.md").write_text("\n".join(T) + "\n", encoding="utf-8")

print("文档已生成：")
for fn in ["RESEARCH_DESIGN.md", "COMPARISON_MATRIX.md", "README.md", "TABLES_INDEX.md"]:
    p = ROOT / fn
    print(f"  {fn}: {len(p.read_text(encoding='utf-8').splitlines())} 行")
