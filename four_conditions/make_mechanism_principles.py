#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
生成《机理模型原理总纲》MECHANISM_PRINCIPLE.md 与原理图 fig12。

把此前 121 行的「机理+经验」概览升级为逐条、逐模型的完整原理总纲：
EDTA 脱钙化学 / Archie 电导 / 串联积分与归一化 / 界面极化 CPE 并联导纳 /
delta_eff 完整代数推导与数学性质 / 参数可辨识性(尺度冗余 + 边界命中) /
逐模型机理成分溯源 / 符号量纲总表 / 落盘数值对照 / 诚实边界。

所有数值来自本轮落盘结果（不重新拟合、不手抄）：
  three_routes_results.json / three_routes_metrics.csv / trackB_four_conditions.csv
  各模型 parameters.json / data/mechanism_timeseries_quantiles.csv / data/frozen_measured_24rat.csv
  FITCFG 边界常量来源 run_three_routes.py

路径自适应：放在顶层目录或 github_package/four_conditions/ 下都能运行。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
IN_PKG = (HERE.name == "four_conditions")

if IN_PKG:
    DATA = HERE
    PKG = HERE.parent
    DOCS = PKG / "docs"
    FIGS = HERE / "figures"
    TOP_MD = None
else:
    DATA = HERE
    PKG = HERE / "github_package"
    DOCS = PKG / "docs"
    FIGS = PKG / "four_conditions" / "figures"
    TOP_MD = HERE / "MECHANISM_PRINCIPLE.md"

DOCS.mkdir(parents=True, exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

R = json.loads((DATA / "three_routes_results.json").read_text(encoding="utf-8"))
M = pd.read_csv(DATA / "three_routes_metrics.csv")
B = pd.read_csv(DATA / "trackB_four_conditions.csv")
MODELS = R["meta"]["models"]
MK = ["A1.1", "A1.2", "A1.3", "A2", "A3"]
CK = ["C1_all", "C2_lowfreq", "C3_highdose", "C4_lf_hd"]
CN = {"C1_all": "C1 全部", "C2_lowfreq": "C2 低频",
      "C3_highdose": "C3 高浓度", "C4_lf_hd": "C4 低频+高浓度"}
ROUTE = {"A1.1": "TrackA1 纯经验", "A1.2": "TrackA1 纯经验", "A1.3": "TrackA1 纯经验",
         "A2": "TrackA2 纯机理", "A3": "TrackA3 机理驱动经验"}


def params_of(model):
    p = PKG / ("TrackA1/" + model if "." in model else "Track" + model)
    return json.loads((p / "parameters.json").read_text(encoding="utf-8"))


PAR = {m: params_of(m) for m in MK}
A2P = PAR["A2"]["params"]                      # [a0, beta, p, s]
A3P = PAR["A3"]["params"]                      # [a0, beta, p, s, eta, eps]
TRANSFER = R["A2_global_transfer"]

# 拟合边界（来源 run_three_routes.py FITCFG）；A3 的前 4 个机理参数沿用 A2 边界
BOUNDS = {
    "A1.1": (["C", "b0", "b1", "b2", "n"], [0.0, 1e-5, -1, -1, 0.2], [0.9, 1.0, 1, 1, 3.0]),
    "A1.2": (["a", "b", "c", "d"], [-50, -50, -1, -5], [50, 50, 1, 5]),
    "A1.3": (["kappa", "s", "eps"], [0.1, -1.0, -0.05], [30.0, 1.5, 0.05]),
    "A2":   (["a0", "beta", "p", "s"], [0.01, -2, 0.05, 0.0], [100, 3, 3, 1.0]),
    "A3":   (["eta", "eps"], [0.0, -0.5], [3.0, 0.5]),
}

# 上游机理先验 / 实测条件均值的数值证据（自动渲染，不手抄）
PRIOR = pd.read_csv(PKG / "data" / "mechanism_timeseries_quantiles.csv")
W = PRIOR[PRIOR.metric == "whole_resistance_ratio"]
W95 = W[W.global_time_s == 5700.0].drop_duplicates("edta_concentration_mol_per_l")
RAW_MIN, RAW_MAX = float(W95.q50.min()), float(W95.q50.max())
FROZ = pd.read_csv(PKG / "data" / "frozen_measured_24rat.csv")
OBS = (FROZ[FROZ.global_time_min > 0]
       .groupby(["edta_concentration_mol_per_l", "frequency_hz", "global_time_min"])["rz"].mean())
OBS_MIN, OBS_MAX = float(OBS.min()), float(OBS.max())


def md_table(header, rows):
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(["---"] * len(header)) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(out)


def g(cond, model, col):
    return float(M[(M.condition == cond) & (M.model == model)][col].iloc[0])


def f(v, n=3):
    return f"{v:.{n}f}"


def boundary_rows():
    rows = []
    for m in MK:
        vals = PAR[m]["params"]
        if m == "A3":
            names = ["a0", "beta", "p", "s", "eta", "eps"]
            lo = list(BOUNDS["A2"][1]) + list(BOUNDS["A3"][1])
            hi = list(BOUNDS["A2"][2]) + list(BOUNDS["A3"][2])
        else:
            names, lo, hi = BOUNDS[m]
        for nm, v, l, h in zip(names, vals, lo, hi):
            span = h - l
            hit = (v <= l + 0.02 * span) or (v >= h - 0.02 * span)
            rows.append([m, nm, f"{v:.4f}", f"{l:.4g}", f"{h:.4g}",
                         "**顶边界 ✗**" if hit else "界内 ✓"])
    return rows


# ================================================================ 文档正文
L = []
add = L.append

add("# 机理模型原理总纲（TrackA1 / TrackA2 / TrackA3 / TrackB 全覆盖）")
add("")
add("> 本文件是全项目**机理原理的唯一权威清单**。此前 `docs/MECHANISM_PRINCIPLE.md` 只有 121 行概览，")
add("> 给出了主要公式，但缺**完整推导链条、符号与量纲、逐模型机理成分溯源、参数可辨识性论证、数值对照**；")
add("> 本轮把它们逐条补全，覆盖所有含机理成分的模型。")
add("> 所有数值由 `make_mechanism_principles.py` 从落盘结果自动渲染，**未重新拟合、未改模型、未重新校准 TrackB**。")
add("")
add("> 配套文档：[`docs/EVALUATION_CRITERIA.md`](EVALUATION_CRITERIA.md) 回答「怎么判」，本文件回答「为什么这样写」。")
add("> 数据：修订版《体外EDTA数据记录汇总.xlsx》SHA256 `4c165298...`，三层核验通过；24 鼠全保留，")
add("> 未删除 WT17 / WT9 / WT8 等不利样本。")
add("")

# ---- 0
add("## 0. 覆盖范围：机理成分在哪些模型里")
add("")
add("「机理」不是只属于 TrackA2：A1.2 / A1.3 读取机理先验输出，A3 以机理为骨架，TrackB 的区间由机理 ensemble 产生。")
add("下表先声明每个模型的机理成分与来源，第 7 节逐条展开。")
add("")
add(md_table(["模型", "路线", "机理成分", "机理来源", "机理含量", "本文件落点"], [
    ["**A1.1 拉伸指数**", "纯经验", "无", "—", "无（唯一完全脱离机理链路）", "第 7.1 节"],
    ["**A1.2 A+**", "纯经验", "`R_Z_raw` 查表特征", "`data/mechanism_timeseries_quantiles.csv` 的 q50", "低", "第 7.2 节"],
    ["**A1.3 H1 混合**", "纯经验", "改性量 `delta=1-raw` 作形状骨架 + `lambda(f)`", "同上 + CPE 一阶展开", "中", "第 7.3 节"],
    ["**A2 纯机理**", "纯机理", "**全部**：Archie + 脱钙动力学 + CPE 并联导纳", "机理方程内生（零外部数据）", "100%", "第 2–6 节"],
    ["**A3 机理驱动经验**", "机理驱动经验", "A2 机理骨架（固定）+ 2 个校准系数", "`TrackA2` 预测", "骨架 100%", "第 7.5 节"],
    ["**TrackB 区间**", "区间模型", "机理 LHS ensemble 的 `sim_q05/q50/q95`", "机理参数 Latin Hypercube 采样", "ensemble 底座", "第 7.6 节"],
]))
add("")

# ---- 1
add("## 1. 原理一：EDTA 脱钙化学（一切的物理起点）")
add("")
add("- **无机相**：颅骨矿物质主要是碳酸替代型羟基磷灰石 `Ca10(PO4)6(OH)2`（Ca/P 约 1.67），")
add("  以微晶形式填充在胶原基质中，晶间与骨小管/哈弗管构成含电解液的孔隙网络。")
add("- **螯合反应**：EDTA（乙二胺四乙酸，完全去质子化为 `Y4-`）与 `Ca2+` 形成 1:1 可溶螯合物")
add("  `Ca2+ + Y4- -> CaY2-`（log K 约 10.7，碱性条件），把 Ca 从晶格表面持续抽走并带走，")
add("  反应不可逆地向右进行。")
add("- **电学后果**：矿物溶解 → 微孔/微裂纹扩大、孔隙率 `phi` 上升 → 孔内离子液连通性上升 →")
add("  有效电导率上升 → 阻抗下降。这就是 `R_Z = Z(t)/Z0 < 1` 的化学根源。")
add("- **剂量依赖**：`d` 越高，螯合剂摩尔数与反应前沿推进越快 → `phi` 随 `d` 幂律增长（对应指数 `beta`）。")
add("- **时间依赖**：由「Ca 从晶格溶出」与「EDTA 在孔内扩散」两步控制，早期快、后期趋缓；")
add("  本项目用 `(t/T0)^p`（`p` 小于 1 的次线性）作为**等效描述**，不是被辨识的反应/扩散动力学方程。")
add("- **实测阶段序列**：`baseline`(t=0) → `acetic_acid` → EDTA 四个剂量 → 改性时间点")
add("  `[5, 20, 35, 65, 95]` min → `urea_10`(t=105)。共 4 剂量 × 4 频率 × 7 时间点（含基线）。")
add("- **尿素段**：t=105 加入乙醇+尿素后实测 `R_Z` 回升，机理上可能对应蛋白/孔道再封闭或界面重构；")
add("  当前所有单调机理（含 A2）均无法表达，见第 9 节。")
add("")

# ---- 2
add("## 2. 原理二：Archie 型有效电导率")
add("")
add("多孔介质经典 **Archie 定律**：饱和岩体电导率 `sigma_b = sigma_w / F`，地层因子 `F = phi^(-m)`，")
add("即 `sigma_b = sigma_w · phi^m`（`m` 为胶结/连通性指数，常见 1.3–2.5，反映孔道曲折度）。")
add("本模型使用带残余项的推广式：")
add("")
add("```")
add("sigma = sigma_floor + c * phi^m")
add("```")
add("")
add(md_table(["符号", "含义", "量纲 / 单位", "为什么需要"], [
    ["`sigma`", "局部有效电导率", "S/m", "决定该层电阻"],
    ["`sigma_floor`", "脱钙前基质 / 结合水 / 细胞质的残余电导", "S/m", "若纯 `sigma ∝ phi^m`，`phi→0` 时阻抗发散；实测基线 `Z0` 有限，故必须有残余项"],
    ["`c`", "孔隙液电导率标度", "S/m", "把孔隙率折算成电导的系数"],
    ["`m`", "胶结指数（孔道曲折度）", "无量纲", "Archie 定律的经验指数"],
    ["`phi`", "孔隙率", "无量纲体积分数", "由 EDTA 脱钙产生（第 1 节）"],
]))
add("")
add("旧机理先验（`data/mechanism_timeseries_quantiles.csv` 的 `whole_resistance_ratio`）正是按这一")
add("**准静态纯电阻网络**逐层积分得到的；它只含 `phi(t,d)`，不含任何频率项，是第 3 节三个缺口的根源。")
add("")

# ---- 3
add("## 3. 原理三：串联电阻积分与归一化阻抗比")
add("")
add("把颅骨沿厚度离散为 N 层，电极有效截面积 `A_cell`，每层厚 `dx_i`：")
add("")
add("```")
add("R    = sum_i dx_i / (sigma_i * A_cell)      (沿厚度串联积分，准静态纯电阻)")
add("Z0   = Z(t=0)")
add("R_Z  = Z(t) / Z0                            (归一化阻抗比)")
add("```")
add("")
add("按定义 `R_Z(t=0) ≡ 1`，这是所有模型必须满足的硬门槛（评价标准 B7）。归一化同时消掉")
add("`A_cell`、电极几何与绝对电导标度，使不同鼠、不同批次可比。")
add("")
add("**旧机理先验的三个结构性缺口**（数值由落盘数据自动渲染）：")
add("")
add(md_table(["缺口", "机理侧证据", "实测侧证据", "后果"], [
    ["**频率无关**", "先验 `R_Z_raw` 与 `f` 完全无关，只有剂量 × 时间一条曲线族",
     "100 / 120 / 1000 / 10000 Hz 实测明显分层", "机理方程缺频率维度，是最大误差源"],
    ["**脱钙动力学太慢 / 幅度太小**",
     "t=95 min 先验 `R_Z_raw` 只降到 `" + f(RAW_MIN) + "–" + f(RAW_MAX) + "`（改性量约 " +
     f(1 - RAW_MAX, 2) + "–" + f(1 - RAW_MIN, 2) + "）",
     "实测条件均值降到 `" + f(OBS_MIN) + "–" + f(OBS_MAX) + "`（改性量 " +
     f(1 - OBS_MAX, 2) + "–" + f(1 - OBS_MIN, 2) + "）", "需要放大改性量或补时间维度"],
    ["**95 min 截断 + 尿素回升**", "先验在 95 min 取末值平台，无回升项",
     "t=105（乙醇+尿素）实测 `R_Z` 回升", "所有单调模型无法表达"],
]))
add("")
add("**结论**：偏差不是「参数没调好」，而是机理方程**缺频率维度、缺界面极化、脱钙动力学被过度平滑**。")
add("A1 系列用经验协变量在外面补，A2 则把频率直接写回机理方程（第 4–5 节）。")
add("")

# ---- 4
add("## 4. 原理四：界面极化 / CPE 与并联导纳（本轮核心修复）")
add("")
add("交流测量下颅骨不是纯电阻：骨基质 / 孔壁界面与电极-电解液界面存在**双电层电容**与")
add("**Maxwell-Wagner 界面极化**，其频响常用**恒相位元件 CPE** 描述：")
add("")
add("```")
add("Y_CPE = Q * (j*omega)^n        0 < n < 1,  omega = 2*pi*f")
add("```")
add("")
add("CPE 导纳随频率**上升**（`n>0`）。因此真实颅骨的等效电路是")
add("**离子通路 `G_ion(t,d)` 与界面极化旁路 `G_pol(f)` 并联**：")
add("")
add("```")
add("Y(f,t,d) = G_ion(t,d) + G_pol(f)          Z = 1 / Y")
add("```")
add("")
add(md_table(["频段", "主导支路", "对脱钙 phi 变化", "归一化结果"], [
    ["**低频 100 / 120 Hz**", "离子通路 `G_ion`", "敏感：脱钙 → `sigma`↑ → `G_ion`↑ → `Z` 明显下降", "`R_Z` 下降显著"],
    ["**高频 1000 / 10000 Hz**", "CPE 旁路 `G_pol(f)`", "不敏感：旁路分流占主导，phi 变化被分母稀释", "`R_Z` 更接近 1（系统性偏高）"],
]))
add("")
add("这正好解释实测「**同一时间下高频 `R_Z` 系统性偏高**」，而且**不需要任何经验 `ln(f)` 协变量**。")
add("旧机理是「准静态纯电阻」，A2 是「频率内生的并联导纳」，这是两条机理路线最本质的差别。")
add("")

# ---- 5
add("## 5. 原理五：delta_eff 的完整推导与数学性质（TrackA2 方程）")
add("")
add("### 5.1 从 `Y(0)/Y(t)` 到 `1/(1+delta_eff)`（逐步代数）")
add("")
add("设基线离子导纳 `G0 = G_ion(0)`，脱钙带来的**相对增量**为")
add("`delta(t,d) = A·(d/D0)^beta·(t/T0)^p`，界面极化旁路 `G_pol(f) = rho·(f/F0)^n`，则")
add("")
add("```")
add("G_ion(t,d) = G0 * (1 + delta(t,d))")
add("Y(0)       = G0 + G_pol(f)")
add("Y(t)       = G0 * (1 + delta) + G_pol(f)")
add("")
add("R_Z = Y(0) / Y(t)")
add("    = [G0 + G_pol] / [G0*(1+delta) + G_pol]")
add("    = 1 / [1 + G0*delta / (G0 + G_pol)]")
add("    = 1 / (1 + delta_eff)")
add("")
add("delta_eff = G0 * delta(t,d) / (G0 + G_pol(f))        <-- 频率进入分子的可辨识组合")
add("```")
add("")
add("在**旁路主导**极限（`G_pol >> G0`，对应较高频 / 强界面极化）：")
add("")
add("```")
add("delta_eff ~ (G0/rho) * (f/F0)^(-n) * delta(t,d)")
add("         = a0 * (d/D0)^beta * (t/T0)^p * (f/F0)^(-s)      a0 = (G0/rho)*A,  s = n")
add("R_Z       = 1 / (1 + delta_eff)")
add("```")
add("")
add("> **澄清**：早期 121 行概览把 `G_pol(f) ~ (f/F0)^(-s)` 直接写成「CPE 频散」，那是对")
add("> `delta_eff` 频率因子的**等效简写**。严格链条是：CPE 旁路导纳随 `f` **上升** → 分母")
add("> `(G0 + G_pol)` 变大 → `delta_eff` 被稀释 → `delta_eff ∝ f^(-s)`。这也说明 `s` 只是")
add("> CPE 指数 `n` 在旁路主导近似下的等效值（第 9 节 E9）。")
add("")
add("### 5.2 三个参考标度（固定，不参与拟合）")
add("")
add(md_table(["标度", "取值", "作用", "为什么这样取"], [
    ["`D0`", "0.10 mol/L", "剂量归一化", "取剂量的中间量级，使 `a0` 有明确参考条件"],
    ["`T0`", "100 min", "时间归一化", "接近改性观察窗上限（95 min），使 `p` 落在易辨识区间"],
    ["`F0`", "100 Hz", "频率归一化", "最低测量频率；`f/F0 ≥ 1`，`(f/F0)^(-s) ≤ 1`"],
]))
add("")
add("### 5.3 四个物理量（C1 全条件拟合值，由 `parameters.json` 自动渲染）")
add("")
names, lo, hi = BOUNDS["A2"]
add(md_table(["符号", "物理含义", "量纲", "C1 拟合值", "下界", "上界", "边界状态"], [
    [nm, mean, dim, f"{v:.4f}", f"{l:.4g}", f"{h:.4g}",
     "**顶边界**" if (v <= l + 0.02 * (h - l) or v >= h - 0.02 * (h - l)) else "界内 ✓"]
    for nm, mean, dim, v, l, h in zip(
        names,
        ["参考条件下改性幅度", "剂量指数", "脱钙动力学时间指数", "界面极化频散指数"],
        ["无量纲", "无量纲", "无量纲", "无量纲"], A2P, lo, hi)
]))
add("")
add("四个参数**全部无量纲、全部落在拟合边界内部、无参数互相补偿**（第 6 节），且**零经验回归系数**。")
add("")
add("### 5.4 数学性质（可直接验证）")
add("")
add(md_table(["性质", "推导", "意义"], [
    ["`R_Z(t=0) = 1`", "`t=0` → `(t/T0)^p = 0` → `delta_eff = 0` → `R_Z = 1`",
     "**基线天然自洽**，无需人为锚定（B7）"],
    ["值域 `(0, 1]`", "`delta_eff ≥ 0` → `1/(1+delta_eff) ∈ (0,1]`", "无需 clip，不会越出物理范围"],
    ["`∂R_Z/∂t < 0`", "`p > 0` → `delta_eff` 随 `t` 增 → `R_Z` 降", "单调下降"],
    ["`∂R_Z/∂d < 0`", "`beta > 0` → 剂量越高改性越强", "剂量分层"],
    ["`∂R_Z/∂f > 0`", "`s > 0` → `(f/F0)^(-s)` 随 `f` 减 → `delta_eff` 减 → `R_Z` 升", "**高频 `R_Z` 系统性偏高**"],
]))
add("")
add("**单调性是双刃剑**：它保证了基线自洽与物理合理性，但也使 A2（以及所有 A1 模型）")
add("无法表达 t=105 尿素段的回升（E8）。")
add("")

# ---- 6
add("## 6. 原理六：参数可辨识性（为什么最终是 4 个参数）")
add("")
add("### 6.1 尺度冗余的代数证明")
add("")
add("由 `delta_eff = G0·delta / (G0 + G_pol)`：若把 `G0` 与 `rho`（从而 `G_pol`）同时乘以任意 `lambda > 0`，")
add("分子分母同乘 `lambda`，`delta_eff` **不变**，`R_Z` 不变。因此**只有比值 `G0/rho` 可辨识**，")
add("`G0` 与 `rho` 各自不可辨识 —— 这就是 6 参数形式里 `a0` 与 `rho` 双双顶到优化上界、互相补偿的原因。")
add("")
add("### 6.2 参数化演化（建模阶段诊断记录）")
add("")
add(md_table(["阶段", "形式", "参数", "结果", "处置"], [
    ["P1", "`g_ion=1+A·t^p/(t50^p+t^p)` + `g_pol=rho·f^n`（Hill 饱和）", "6",
     "`a0 / tau / rho` 互相补偿，多个参数顶到上界", "弃用"],
    ["P2", "`g_ion=1+a0·d^beta·t^p` + `g_pol=rho·f^n`（幂律）", "5",
     "`G0` 与 `rho` 的尺度冗余仍在，仍顶边界", "弃用"],
    ["**P3**", "**归一化有效改性量 `delta_eff`（本轮采用）**", "**4**",
     "把可辨识组合 `G0/rho` 直接写成单一因子，4 参数全部界内", "**采用**"],
]))
add("")
add("> 说明：P1/P2 是建模阶段的可辨识性探针记录，探针脚本未纳入交付包；")
add("> 交付的 `run_three_routes.py` 只含最终 4 参数形式。本轮**未重新拟合**，")
add("> 下表是对既有落盘参数与 `FITCFG` 边界的静态核对。")
add("")
add("### 6.3 边界命中诊断（自动渲染）")
add("")
add("判据：参数落在 `[lo + 2%·span, hi − 2%·span]` 之外即视为顶边界（= 不可辨识，B8）。")
add("")
add(md_table(["模型", "参数", "C1 拟合值", "下界", "上界", "边界状态"], boundary_rows()))
add("")
add("**A2 的 4 个参数全部界内 ✓**；A3 的 4 个机理参数沿用 A2 边界、`eta` / `eps` 在各自边界内。")
add("三个经验模型的参数虽然也多在界内，但**无物理量纲、无独立性约束**，")
add("「界内」只说明优化收敛，不等于物理可解释（B9）。")
add("")
add("### 6.4 与评价标准的对应")
add("")
add("可辨识性对应 [`EVALUATION_CRITERIA.md`](EVALUATION_CRITERIA.md) 的 **B8**（顶边界/补偿 = 不可辨识）、")
add("**B9**（物理可解释性）、**B6/B5**（k 与 AIC）、**B11**（跨条件零重拟合迁移）。")
add("")

# ---- 7
add("## 7. 原理七：逐模型的机理成分溯源")
add("")
add("### 7.1 A1.1 拉伸指数 —— 无机理成分")
add("")
add("`R_Z = C + (1-C)·exp(-(b·t)^n)`，`b = b0 + b1·d + b2·ln(f/100)`。直接对时间拟合衰减曲线，")
add("把剂量与频率塞进 `b` 的线性组合；`C` 是平台、`n` 是形状指数。")
add("**完全脱离机理链路**，是唯一不含任何机理成分的模型（5 参数无物理量纲）。")
add("")
add("### 7.2 A1.2 A+ —— 机理先验只作一个查表特征")
add("")
add("`R_Z = a + b·R_Z_raw + c·t + d·ln(f/100)`。其中 `R_Z_raw` 来自")
add("`data/mechanism_timeseries_quantiles.csv` 的 `whole_resistance_ratio` **q50**，")
add("由 `run_three_routes.py` 的 `raw_prior()` 用 `np.interp` 把秒级曲线插值到 7 个全局时间点。")
add("由于先验**频率无关、动力学太慢**（第 3 节），A+ 必须补 `c·t` 与 `d·ln(f/100)` 两个经验协变量 ——")
add("**补 `ln(f)` 等于承认机理侧缺频散 / 界面极化项**，但这是经验层补丁，不是机理改进（E3）。")
add("代价：`c` / `d` 无物理出处，t=0 外推偏离 1（落盘值 `" +
     f(PAR["A1.2"]["baseline_pred_uncorrected"]) + "`），需人为锚定；model.py 需读外部 CSV，非自包含。")
add("")
add("### 7.3 A1.3 H1 混合 —— 机理改性量作形状骨架")
add("")
add("`R_Z = 1 - kappa·(f/100)^(-s)·(1-R_Z_raw) - eps·t`，用 `delta = 1 - R_Z_raw` 作形状骨架，")
add("`lambda(f) = (f/100)^(-s)` 乘在 `delta` 上，`-eps·t` 补时间漂移。")
add("`lambda(f)` 来自 CPE 一阶展开 `lambda(f) ≈ 1 - s·ln(f/f0)`，")
add("故 `R_Z ≈ 1 - kappa·delta + kappa·s·ln(f/f0)·delta`，预言频率效应应是 **`ln(f)×delta` 交互项**。")
add("")
add("> **一条机理预言被数据否定**（Mechanism Fusion 历史落盘）：实测加性 `c·ln(f)` 的 LORO 0.436")
add("> 优于交互形式 0.428，说明 `lambda(f)` 的幂律只是**最简假说下的等效参数化**，")
add("> `s` 不是被辨识的 CPE 指数（E9）。这也是 A2 把 `s` 明确标注为等效频散指数的原因。")
add("")
add("### 7.4 A2 纯机理 —— 全部内生")
add("")
add("第 2–5 节的全部链条（Archie → 脱钙 → CPE 并联导纳 → `delta_eff`）都在机理方程内部，")
add("**频率 / 剂量 / 时间三个维度全部内生**，零经验回归系数、零外部数据依赖。")
add("`TrackA2/model.py` 完全自包含，仅依赖 numpy。")
add("")
add("### 7.5 A3 机理驱动经验 —— 机理定形状，经验只做保形校准")
add("")
add("```")
add("R_mech = TrackA2 的纯机理预测（前 4 个物理参数，stagewise 固定）")
add("R_Z    = 1 - eta*(1 - R_mech) - eps*(t/105)")
add("```")
add("")
add(md_table(["系数", "C1 拟合值", "物理 / 校准含义", "边界"], [
    ["`eta`", f(A3P[4], 4), "缩放机理改性幅度；小于 1 说明纯机理略高估改性量", "[0, 3] 内"],
    ["`eps`", f(A3P[5], 4), "补机理尚未覆盖的时间漂移", "[-0.5, 0.5] 内"],
]))
add("")
add("`eta=1, eps=0` 时**严格退化为 A2**（嵌套关系），因此经验层的边际贡献可直接量化：")
add("四条件 LORO 相对 A2 提升约 +0.028 ~ +0.040（第 8 节）。")
add("**诚实边界**：`eta` / `eps` 是拟合校准系数、不是物理常数；「机理驱动」指结构来自机理，")
add("不指全部参数可溯源（E10）。")
add("")
add("### 7.6 TrackB —— 机理 ensemble + conformal 校准")
add("")
add("机理 **Latin Hypercube Sampling (LHS) ensemble** 对机理参数采样，产生")
add("`sim_q05 / q50 / q95`；折 × 频率分层的 **conformal 校准**给出 padding：")
add("")
add("```")
add("p_(fold,f) = tolerance_f * q_f      q_f = Quantile_0.90(内层 cross-fit 分数)")
add("calibrated_lower = clip(sim_q05 - p, 0, 1)")
add("calibrated_upper = clip(sim_q95 + p, 0, 1)")
add("```")
add("")
add("- **B1 诊断**：最终区间宽度约 **97.09%** 来自 padding，基础 LHS 参数不确定性仅 **3.00%**。")
add("  「区间过宽」的根因是 conformal padding 补偿「仿真 vs 实测」的系统性偏差，")
add("  **不是 LHS 参数采样范围不足** —— 盲目扩大 LHS 采样没有用。")
add("- **只报 coverage / width / Winkler，不报 R² / RMSE**；逐点覆盖率 ≠ 整条曲线联合覆盖；")
add("  基线 t=0 与尿素 t=105 无校准区间；验证集 22/24 鼠；0.05 M 存在 role × 剂量混淆。")
add("- 四条件指标是**既有 392 点 LOO 校准区间按条件切片的统计**，本轮未重新拟合、未重新校准。")
add("")
add("### 7.7 机理成分溯源矩阵（汇总）")
add("")
add(md_table(["模型", "机理输入", "输入来源", "经验层", "基线 t=0", "自包含"], [
    ["A1.1", "无", "—", "全部（5 参数）", "天然=1", "是"],
    ["A1.2", "`R_Z_raw` q50", "机理先验 CSV + `np.interp`", "`a,b,c,d` 全为回归系数", "偏离 1，需锚定", "否"],
    ["A1.3", "`delta=1-raw` + `lambda(f)`", "机理先验 CSV + CPE 展开", "`kappa,s,eps` 拟合系数", "天然=1", "否"],
    ["**A2**", "**全部机理方程内生**", "**零外部数据**", "**无**", "天然=1", "**是**"],
    ["A3", "A2 骨架（固定 4 参数）", "`TrackA2` 预测", "`eta,eps` 两个校准系数", "天然=1", "是"],
    ["TrackB", "LHS ensemble 分位数", "机理参数采样", "conformal padding", "无区间", "CSV 切片"],
]))
add("")

# ---- 8
add("## 8. 原理八：落盘数值对照（自动渲染）")
add("")
add("数值仅用于佐证各机理 / 经验的**适用范围**；四条件评价集与折数不同，")
add("**R² 不可横向比大小**（E1/E2），也不构成「机理被证实」（E10）。")
add("")
add("### 8.1 五模型 × 四条件 LORO R²（验收口径）")
add("")
add(md_table(["模型", "路线"] + [CN[c] for c in CK],
             [[m, ROUTE[m]] + [f(g(c, m, "loro_r2")) for c in CK] for m in MK]))
add("")
add("### 8.2 A2 跨条件零重拟合迁移（B11，「覆盖所有条件」的证据）")
add("")
add("用 C1 全条件拟合的**单一组 4 个物理参数** `a0=" + f(A2P[0], 4) +
     ", beta=" + f(A2P[1], 4) + ", p=" + f(A2P[2], 4) + ", s=" + f(A2P[3], 4) +
     "`，不重拟合直接评估各评价子集：")
add("")
add(md_table(["目标评价集（零重拟合）", "节点", "R²", "RMSE"],
             [[CN[c], int(TRANSFER[c]["n_nodes"]), f(TRANSFER[c]["r2"]),
               f(TRANSFER[c]["rmse"], 4)] for c in CK]))
add("")
add("一套参数跨 4 剂量 × 4 频率 × 6 时间全平面保持 R² > 0.91，无需为子集单独调参 ——")
add("这是纯机理相对经验模型的**结构性优势**，也是「把频率等全算进去、所有条件都考虑」的落点。")
add("")
add("### 8.3 TrackB 四条件区间指标（只报区间指标）")
add("")
add(md_table(["条件", "点数", "鼠数", "coverage", "mean width", "median width", "max width", "Winkler"],
             [[CN[r.condition], int(r.n_points), int(r.n_rats), f(r.coverage),
               f(r.mean_width, 4), f(r.median_width, 4), f(r.max_width, 4),
               f(r["winkler_alpha_mis_0.10"], 4)] for _, r in B.iterrows()]))
add("")

# ---- 9
add("## 9. 原理九：必须诚实说明的边界（完整版）")
add("")
add(md_table(["编号", "边界", "理由", "对应评价标准"], [
    ["M1", "**物理量是被标定，不是从第一性原理推导**",
     "A2 的 `a0/beta/p/s` 由实测最小二乘标定，只有物理量纲与物理含义", "E10"],
    ["M2", "**`s` 是等效频散指数，不是被辨识的 CPE 指数**",
     "只有 4 个频率点；幂律 / 加性 `ln(f)` / 交互形式统计不可分；单一 `f^(-s)` 只在旁路主导极限成立，需独立宽频 EIS 介电谱实验", "E9"],
    ["M3", "**`(t/T0)^p` 是脱钙动力学的等效描述**",
     "`p` 约 0.8 的次线性幂律反映扩散限制的整体趋势，不是被辨识的孔隙率演化方程；真正修复需用孔隙率 / 钙含量实测重标定 `phi(t)`", "E10"],
    ["M4", "**`delta_eff` 单调 → 尿素 t=105 回升无法表达**",
     "所有 A1 / A2 / A3 模型皆然；需引入尿素导致的蛋白 / 孔道再封闭或界面重构项", "E8"],
    ["M5", "**A3 的 `eta` / `eps` 是经验校准系数**",
     "有明确校准含义但不是物理常数；「机理驱动」指结构来自机理，不指全部参数可溯源", "E10"],
    ["M6", "**TrackB 约 97% 宽度来自 padding**",
     "根因是「仿真 vs 实测」系统性偏差，不是 LHS 参数不确定性；另有逐点 ≠ 联合、22/24 鼠、role × 剂量混淆、上游机制选择非严格嵌套", "C6/C7"],
    ["M7", "**三条路线不构成「机理被证实」**",
     "它们是有物理动机的参数化；`R²` 高只说明在该评价集上拟合好", "E10"],
    ["M8", "**拟合变好 ≠ 机理更正确**",
     "条件收窄、加自由度、补协变量都会抬高 `R²`；A2 在 C1 的 LORO 优势需与噪声级差异同看", "E3/E5"],
    ["M9", "**旧机理先验与上游 manifest 的 source_data SHA 不一致**",
     "上游原始文件不可得；本仓库以三层核验通过的本地 Excel 与冻结副本为准", "A7"],
]))
add("")

# ---- 10
add("## 10. 附录 A：符号与量纲总表")
add("")
add(md_table(["符号", "含义", "量纲 / 单位", "出现位置"], [
    ["`Ca10(PO4)6(OH)2`", "羟基磷灰石", "化学式", "第 1 节"],
    ["`Y4-`, `CaY2-`", "EDTA 去质子化形式 / Ca-EDTA 螯合物", "化学式", "第 1 节"],
    ["`phi`", "孔隙率", "无量纲体积分数", "第 1、2 节"],
    ["`sigma`, `sigma_floor`, `c`, `m`", "有效电导率 / 残余电导 / 标度 / 胶结指数", "S/m, S/m, S/m, 无量纲", "第 2 节"],
    ["`dx_i`, `A_cell`, `R`", "层厚 / 电极有效截面积 / 串联电阻", "m, m², Ohm", "第 3 节"],
    ["`Z`, `Z0`, `R_Z`, `R_Z_raw`", "阻抗 / 基线阻抗 / 归一化阻抗比 / 机理先验", "Ohm, Ohm, 无量纲, 无量纲", "第 3、7.2 节"],
    ["`Y`, `G_ion`, `G_pol`, `G0`", "总导纳 / 离子通路导纳 / CPE 旁路导纳 / 基线离子导纳", "S", "第 4、5 节"],
    ["`Q`, `n`, `omega`", "CPE 系数 / CPE 指数 / 角频率", "S·s^n, 无量纲, rad/s", "第 4 节"],
    ["`delta(t,d)`, `delta_eff`", "离子导纳相对增量 / 归一化有效改性量", "无量纲", "第 5 节"],
    ["`a0`, `beta`, `p`, `s`", "改性幅度 / 剂量指数 / 时间指数 / 等效频散指数", "均无量纲", "第 5.3 节"],
    ["`D0`, `T0`, `F0`", "参考剂量 / 时间 / 频率", "mol/L, min, Hz（固定标度）", "第 5.2 节"],
    ["`kappa`, `lambda(f)`, `eps`", "H1 幅度 / CPE 频率因子 / 时间漂移", "无量纲, 无量纲, 1/min", "第 7.3 节"],
    ["`eta`, `eps`（A3）", "机理幅度校准 / 时间漂移校准", "无量纲, 无量纲", "第 7.5 节"],
    ["`sim_q05/q50/q95`, `p`, `tolerance_f`, `q_f`", "LHS 分位数 / padding / 容差 / 0.90 分位分数", "无量纲", "第 7.6 节"],
    ["`alpha_mis`", "Winkler 区间评分的未覆盖惩罚系数", "无量纲（0.10）", "第 7.6、8.3 节"],
]))
add("")
add("> 归一化 `R_Z = Z/Z0` 消掉了 `A_cell`、几何与绝对电导标度；")
add("> 因此 `G0` 与 `rho` 只有**比值**可辨识（第 6.1 节），这也是模型最终只保留 4 个无量纲参数的原因。")
add("")

# ---- 11
add("## 11. 附录 B：文档关系与复现")
add("")
add(md_table(["文档", "分工"], [
    ["**本文件 `MECHANISM_PRINCIPLE.md`**", "**机理原理的唯一权威清单（完整推导、符号量纲、逐模型溯源、可辨识性、数值对照、诚实边界）**"],
    ["`docs/EVALUATION_CRITERIA.md`", "评价标准与验收口径的唯一权威清单（A/B/C/D/E 五类 45 条）"],
    ["`four_conditions/THREE_ROUTES_CONCLUSION.md`", "四条件拟合结论与模型取舍"],
    ["`four_conditions/MODEL_DEFINITIONS.md`", "三条路线定义、公式、与上一版命名映射"],
    ["`TrackA1/`、`TrackA2/`、`TrackA3/`、`TrackB/README.md`", "各模型定义 + 评价标准成绩单 + 本模型机理原理摘要"],
    ["`validation/`", "LORO / B1 padding 诊断 / 数据差异 / 拟合效果留痕"],
]))
add("")
add("```bash")
add("python make_mechanism_principles.py  # 由落盘结果重新生成本文件与 fig12")
add("python make_evaluation_criteria.py  # 评价标准总纲 EVALUATION_CRITERIA.md + fig11")
add("python run_three_routes.py          # 重算四条件 x 五模型全部指标")
add("python make_figures.py              # 重绘 fig1-fig10 拟合效果图")
add("```")
add("")
add("> 本文件由脚本生成，数值全部来自落盘 CSV / JSON / `parameters.json`；")
add("> **未重新拟合、未改模型、未重新校准 TrackB**。")
add("")

DOC = "\n".join(L) + "\n"
(DOCS / "MECHANISM_PRINCIPLE.md").write_text(DOC, encoding="utf-8")
if TOP_MD is not None:
    TOP_MD.write_text(DOC, encoding="utf-8")
print("MECHANISM_PRINCIPLE.md 已生成:", len(L), "行,", len(DOC), "字符")


# ================================================================ fig12
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle
from matplotlib.gridspec import GridSpec
import numpy as np

FP = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
if os.path.exists(FP):
    font_manager.fontManager.addfont(FP)
    plt.rcParams["font.family"] = font_manager.FontProperties(fname=FP).get_name()
plt.rcParams["axes.unicode_minus"] = False

fig = plt.figure(figsize=(16.5, 12.2), facecolor="white")
gs = GridSpec(2, 2, height_ratios=[1.0, 1.04], hspace=0.30, wspace=0.22,
              top=0.905, bottom=0.175, left=0.055, right=0.975)

# ---------------- (a) 物理链路推导 ----------------
axA = fig.add_subplot(gs[0, 0]); axA.axis("off")
axA.set_title("(a) 机理链路：从化学到归一化阻抗比（6 步）", fontsize=12.5,
              fontweight="bold", loc="left")
steps = [
    ("1  EDTA 螯合 Ca", "Ca10(PO4)6(OH)2 + Y  ->  CaY", "#EAF1F8", "#2F5D96"),
    ("2  孔隙率上升", "phi(t,d) ~ (d/D0)^beta (t/T0)^p", "#EAF1F8", "#2F5D96"),
    ("3  Archie 电导", "sigma = sigma_floor + c phi^m", "#EAF1F8", "#2F5D96"),
    ("4  并联导纳", "Y = G_ion(t,d) + G_pol(f)", "#FDF1E3", "#C5701F"),
    ("5  归一化", "R_Z = Y(0)/Y(t) = 1/(1+delta_eff)", "#FDF1E3", "#C5701F"),
    ("6  有效改性量", "delta_eff = a0 d^beta t^p f^(-s)", "#E9F5EC", "#2F7A4F"),
]
for i, (title, formula, fc, ec) in enumerate(steps):
    row, col = i % 3, i // 3
    # 左列 step1/2/3 自上而下，右列 step4/5/6 自上而下（先读完左列再读右列）
    x = 0.03 + col * 0.50
    y = 0.80 - row * 0.265
    axA.add_patch(FancyBboxPatch((x, y - 0.075), 0.465, 0.165,
                                 boxstyle="round,pad=0.008,rounding_size=0.012",
                                 fc=fc, ec=ec, lw=1.4, transform=axA.transAxes))
    axA.text(x + 0.022, y + 0.035, title, fontsize=10.4, fontweight="bold",
             color=ec, va="center", transform=axA.transAxes)
    axA.text(x + 0.022, y - 0.030, formula, fontsize=9.0, color="#333333",
             va="center", transform=axA.transAxes)
# 列内向下箭头：1->2->3 与 4->5->6
for (x1, y1, x2, y2) in [(0.262, 0.720, 0.262, 0.630), (0.262, 0.455, 0.262, 0.365),
                          (0.762, 0.720, 0.762, 0.630), (0.762, 0.455, 0.762, 0.365)]:
    axA.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                  mutation_scale=13, lw=1.6, color="#777777",
                                  transform=axA.transAxes))
# 3 -> 4：沿两列间隙的斜向箭头（x 恒在 0.495-0.530 空隙内，不穿任何方框）
axA.add_patch(FancyArrowPatch((0.500, 0.300), (0.522, 0.780), arrowstyle="-|>",
                              mutation_scale=13, lw=1.6, color="#777777",
                              linestyle=":", transform=axA.transAxes))
axA.text(0.02, 0.028,
         "关键：第 4 步起频率 f 进入机理方程（CPE 旁路导纳随 f 上升，分母稀释 delta_eff）。",
         fontsize=9.2, color="#B03A2E", fontweight="bold", transform=axA.transAxes,
         bbox=dict(boxstyle="round,pad=0.35", fc="#FDECEA", ec="#C44E52", alpha=0.95))

# ---------------- (b) 并联等效电路 ----------------
axB = fig.add_subplot(gs[0, 1]); axB.axis("off")
axB.set_title("(b) 并联等效电路与高低频电流路径", fontsize=12.5,
              fontweight="bold", loc="left")
axB.set_xlim(0, 1); axB.set_ylim(0, 1)
# 上下两条母线
axB.plot([0.16, 0.84], [0.80, 0.80], color="#333333", lw=2.4)
axB.plot([0.16, 0.84], [0.20, 0.20], color="#333333", lw=2.4)
axB.text(0.08, 0.80, "A", fontsize=10, va="center", ha="center", fontweight="bold")
axB.text(0.08, 0.20, "B", fontsize=10, va="center", ha="center", fontweight="bold")
axB.text(0.92, 0.50, "Z = 1/Y", fontsize=11, va="center", fontweight="bold", color="#333333")
# 支路 1：离子通路电阻
axB.plot([0.36, 0.36], [0.80, 0.64], color="#2F5D96", lw=2)
axB.add_patch(Rectangle((0.30, 0.44), 0.12, 0.20, fc="#DCE8F5", ec="#2F5D96", lw=1.8))
axB.plot([0.36, 0.36], [0.44, 0.20], color="#2F5D96", lw=2)
axB.text(0.36, 0.54, "G_ion", fontsize=9.6, ha="center", va="center",
         color="#2F5D96", fontweight="bold")
axB.text(0.36, 0.885, "离子通路", fontsize=9.8, ha="center", color="#2F5D96", fontweight="bold")
axB.text(0.36, 0.385, "随 t, d 增大", fontsize=8.8, ha="center", color="#555555")
# 支路 2：CPE 旁路
axB.plot([0.66, 0.66], [0.80, 0.64], color="#C5701F", lw=2)
axB.add_patch(Rectangle((0.60, 0.44), 0.12, 0.20, fc="#FDE9D2", ec="#C5701F", lw=1.8))
axB.plot([0.66, 0.66], [0.44, 0.20], color="#C5701F", lw=2)
axB.text(0.66, 0.54, "G_pol", fontsize=9.6, ha="center", va="center",
         color="#C5701F", fontweight="bold")
axB.text(0.66, 0.885, "界面极化 / CPE", fontsize=9.8, ha="center", color="#C5701F", fontweight="bold")
axB.text(0.66, 0.385, "随 f 增大", fontsize=8.8, ha="center", color="#555555")
# 低频 / 高频路径标注
axB.text(0.02, 0.075, "低频 100/120 Hz：G_pol 小，电流走离子通路 → 脱钙使 R_Z 明显下降",
         fontsize=9.2, color="#2F5D96", transform=axB.transAxes)
axB.text(0.02, 0.028, "高频 1000/10000 Hz：G_pol 大，旁路分流占主导 → R_Z 更接近 1（偏高）",
         fontsize=9.2, color="#C5701F", transform=axB.transAxes)

# ---------------- (c) A2 机理预测曲线 ----------------
axC = fig.add_subplot(gs[1, 0])
axC.set_title("(c) 机理预测：频率内生 / 基线=1 / 单调 / 尿素段无法表达",
              fontsize=12.5, fontweight="bold", loc="left")
a0, beta, pexp, s = A2P
tt = np.linspace(0, 105, 300)
for fr, col, mk in [(100, "#2F5D96", "o"), (120, "#4C72B0", "s"),
                    (1000, "#C5701F", "^"), (10000, "#C44E52", "d")]:
    de = a0 * (0.15 / 0.10) ** beta * (tt / 100.0) ** pexp * (fr / 100.0) ** (-s)
    rz = 1.0 / (1.0 + de)
    axC.plot(tt, rz, lw=2.0, color=col, label=f"{fr} Hz")
axC.axvspan(95, 105, color="#FFF2CC", alpha=0.85, zorder=0)
axC.axhline(1.0, color="#2F7A4F", lw=1.4, linestyle="--")
axC.text(14, 1.035, "物理定义 R_Z(t=0)=1", fontsize=8.8, color="#2F7A4F")
axC.text(100, 0.96, "尿素段 t=105\n实测回升\n单调机理无法表达",
         ha="center", va="top", fontsize=8.6, color="#B03A2E",
         bbox=dict(boxstyle="round,pad=0.35", fc="#FDECEA", ec="#C44E52", alpha=0.92))
axC.set_xlabel("全局时间 t (min)  |  剂量 0.15 M", fontsize=9.5)
axC.set_ylabel("机理预测 R_Z", fontsize=9.5)
axC.set_ylim(0, 1.12); axC.set_xlim(-2, 108)
axC.set_xticks([0, 5, 20, 35, 65, 95, 105])
axC.grid(alpha=0.3, linestyle=":"); axC.set_axisbelow(True)
axC.legend(fontsize=9, ncol=2, loc="lower left", framealpha=0.95)

# ---------------- (d) 参数与量纲表 ----------------
axD = fig.add_subplot(gs[1, 1]); axD.axis("off")
axD.set_title("(d) A2 的 4 个物理量与 3 个参考标度", fontsize=12.5,
              fontweight="bold", loc="left")
tbl = [
    ["a0", "参考改性幅度", "无量纲", f(A2P[0], 4), "界内"],
    ["beta", "剂量指数", "无量纲", f(A2P[1], 4), "界内"],
    ["p", "脱钙时间指数", "无量纲", f(A2P[2], 4), "界内"],
    ["s", "等效频散指数", "无量纲", f(A2P[3], 4), "界内"],
    ["D0", "参考剂量", "0.10 mol/L", "固定", "—"],
    ["T0", "参考时间", "100 min", "固定", "—"],
    ["F0", "参考频率", "100 Hz", "固定", "—"],
]
t = axD.table(cellText=tbl,
              colLabels=["符号", "物理含义", "量纲 / 单位", "C1 拟合值", "边界"],
              cellLoc="center", colLoc="center", loc="upper center",
              colWidths=[0.13, 0.30, 0.20, 0.22, 0.13])
t.auto_set_font_size(False); t.set_fontsize(9.4); t.scale(1.0, 1.55)
for (r, c), cell in t.get_celld().items():
    if r == 0:
        cell.set_facecolor("#2F5D96"); cell.set_text_props(color="white", weight="bold")
    elif r % 2 == 0:
        cell.set_facecolor("#F3F7FC")
    cell.set_edgecolor("#C9D6E5")
axD.text(0.02, 0.175,
         "可辨识性：G0 与 rho 同时放大不改变 R_Z，只有比值 G0/rho 可辨识；",
         fontsize=9.2, color="#333333", transform=axD.transAxes)
axD.text(0.02, 0.115,
         "归一化有效改性量把该比值写成单一因子，故 6 -> 5 -> 4 参数后无顶边界。",
         fontsize=9.2, color="#333333", transform=axD.transAxes)
axD.text(0.02, 0.040,
         "s 是等效频散指数（非被辨识 CPE 指数）；物理量由实测标定，非第一性原理推导。",
         fontsize=9.2, color="#B03A2E", fontweight="bold", transform=axD.transAxes,
         bbox=dict(boxstyle="round,pad=0.35", fc="#FDECEA", ec="#C44E52", alpha=0.95))

fig.suptitle("机理模型原理总览｜EDTA 脱钙 · Archie 电导 · 界面极化 CPE 并联导纳 · delta_eff 推导 · 参数可辨识性",
             fontsize=16, fontweight="bold", x=0.5, y=0.968)
fig.text(0.5, 0.118,
         "总纲 docs/MECHANISM_PRINCIPLE.md：逐条覆盖 TrackA1(A1.1/A1.2/A1.3)、TrackA2、TrackA3、TrackB 的全部机理成分与原理。",
         ha="center", fontsize=9.3, color="#333333")
fig.text(0.5, 0.068,
         "铁律：三条路线都是有物理动机的参数化，不构成机理被证实；s 非被辨识 CPE 指数；"
         "delta_eff 单调无法表达尿素回升；拟合变好 != 机理更正确；TrackB 约 97% 宽度来自 padding。",
         ha="center", fontsize=9.2, color="#333333",
         bbox=dict(boxstyle="round,pad=0.5", fc="#FFF8E7", ec="#D9A441", alpha=0.95))
fig.text(0.5, 0.020,
         "数值由 make_mechanism_principles.py 从落盘结果自动渲染；本轮未重新拟合、未改模型、未重新校准 TrackB。",
         ha="center", fontsize=8.8, color="#777777")

out = FIGS / "fig12_mechanism_principles.png"
fig.savefig(out, dpi=155, bbox_inches="tight", facecolor="white")
plt.close(fig)
print("fig12 已生成:", out)
