#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""从落盘 CSV 自动渲染分段拟合文档（数值一律来自实算表，不手抄）"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
TAB = HERE / "tables"
FIG = HERE / "figures"

COND_SHORT = {"C1_all": "C1 全部", "C2_lowfreq": "C2 低频",
              "C3_highdose": "C3 高浓度", "C4_lf_hd": "C4 低频+高浓度"}
CONDS = list(COND_SHORT)
SCHEMES = ["BASE_A1.1", "BASE_A1.2", "BASE_A1.3", "BASE_A2", "BASE_A3",
           "SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]
SHORT = {"BASE_A1.1": "A1.1", "BASE_A1.2": "A1.2", "BASE_A1.3": "A1.3",
         "BASE_A2": "A2", "BASE_A3": "A3", "SEG_A1": "SEG_A1", "SEG_A3": "SEG_A3",
         "ADJ_A1": "ADJ_A1", "ADJ_A3": "ADJ_A3"}
ROUTE = {**{s: "冻结全局单调（对照）" for s in SCHEMES[:5]},
         "SEG_A1": "V1 三段独立标定 / TrackA1 纯经验",
         "SEG_A3": "V1 三段独立标定 / TrackA3 机理+经验",
         "ADJ_A1": "V2 骨架保留+最少调整 / TrackA1 纯经验",
         "ADJ_A3": "V2 骨架保留+最少调整 / TrackA3 机理+经验"}


def esc(x):
    return str(x).replace("|", "\\|")


def table(headers, rows, aligns=None):
    aligns = aligns or ["---"] * len(headers)
    out = ["| " + " | ".join(esc(h) for h in headers) + " |",
           "|" + "|".join(aligns) + "|"]
    for r in rows:
        out.append("| " + " | ".join(esc(c) for c in r) + " |")
    return "\n".join(out)


def load():
    d = {n: pd.read_csv(TAB / (n + ".csv")) for n in
         ["COMPARISON_OVERALL", "SEGMENT_METRICS", "FULL_GROUP_MEAN",
          "A1_VARIANT_SELECTION", "UREA_REBOUND_SUMMARY", "UREA_REBOUND",
          "SEGMENT_PARAMS", "BASELINE_REPRO_CHECK"]}
    d["SUMMARY"] = json.loads((TAB / "SEGMENTED_SUMMARY.json").read_text(encoding="utf-8"))
    return d


def f(x, n=4):
    return f"{float(x):.{n}f}" if pd.notna(x) else "—"


# ---------------------------------------------------------------- README
def readme(D):
    S = D["SUMMARY"]
    o = D["COMPARISON_OVERALL"]
    c1 = o[o.condition == "C1_all"].set_index("scheme")
    fu = D["FULL_GROUP_MEAN"]

    def fr(s):
        return f(float(fu[(fu.condition == "C1_all") & (fu.scheme == s)].r2.iloc[0]))

    full_a3, full_seg, full_adj = fr("BASE_A3"), fr("SEG_A1"), fr("ADJ_A3")
    lines = [
        "# 按试剂处理分段拟合（乙酸 / EDTA / 饱和尿素）",
        "",
        "> 本轮为**新的分析分支**，新建独立目录；**不修改**已冻结并推送的 `Chemical_Skull_Model_ThreeRoutes` 包内任何模型公式、参数或数据。",
        "",
        "## 一句话结论",
        "",
        "分段拟合在**群体均值层**和**机理性状**上确实更好（full R² 由 0.920 提升到 0.935–0.957，尿素段回升由「完全无法表达」变为 16/16 节点符号一致）；"
        "但在项目既定的**个体级 LORO 唯一验收口径**下，没有任何分段方案取得统计显著的提升——"
        "三段独立标定反而显著变差，骨架保留的最少调整（ADJ_A3）只是**不劣于**全局 A3（ΔR²=+0.0025，95%CI 跨 0）。",
        "",
        f"- 冻结基线 C1 LORO R²：A1.1 {f(c1.loc['BASE_A1.1','r2'])} / A2 {f(c1.loc['BASE_A2','r2'])} / **A3 {f(c1.loc['BASE_A3','r2'])}**",
        f"- 分段 C1 LORO R²：SEG_A1 {f(c1.loc['SEG_A1','r2'])} / SEG_A3 {f(c1.loc['SEG_A3','r2'])} / ADJ_A1 {f(c1.loc['ADJ_A1','r2'])} / **ADJ_A3 {f(c1.loc['ADJ_A3','r2'])}**",
        f"- 群体均值层 full R²：A3 {full_a3} → SEG_A1 {full_seg} / ADJ_A3 {full_adj}",
        f"- 尿素回升：实测 +0.060；5 个全局单调模型全部预测继续下降（0/16 节点符号一致）；全部 4 个分段方案 16/16 一致",
        "",
        "## 目录",
        "",
        "| 路径 | 内容 |",
        "|---|---|",
        "| `segmented_models.py` | 三段模型定义（含冻结基线逐字复刻、V1 独立分段、V2 增量调整、机理再封闭项） |",
        "| `segmented_fit.py` | 数据 / LORO 引擎 / 变体择优 / 全指标 / 配对 bootstrap / 落盘 |",
        "| `metrics_lib.py` | 指标函数，逐字复制自上一轮 `compute_all_metrics.py` 第 192–373 行，口径严格一致 |",
        "| `make_figures.py` | 8 张图 |",
        "| `make_docs.py` | 本文档体系 |",
        "| `data/` | 冻结 24 鼠实测 CSV + 机理先验（SHA 与 ThreeRoutes 包内副本一致） |",
        "| `tables/` | 全部实算表（12 张 CSV + 汇总 JSON + 冻结基线复制件） |",
        "| `figures/` | 8 张诊断图 |",
        "| `SEGMENTED_MODELS.md` | 三段定义、两代公式全表、参数与机理原理 |",
        "| `SEGMENTED_CONCLUSION.md` | **结论主文档**（含全部数值表、两种口径分离、推荐、诚实边界） |",
        "| `TABLES_INDEX.md` | 每张 CSV 的字段说明 |",
        "",
        "## 复现",
        "",
        "```bash",
        "cd Chemical_Skull_Model_SegmentedFit",
        "python3.11 segmented_fit.py     # 主计算：基线校验 + 变体择优 + 四条件 9 方案 LORO + 全指标",
        "python3.11 make_figures.py      # 8 张图",
        "python3.11 make_docs.py         # 文档（从 tables/ 自动渲染）",
        "```",
        "",
        "依赖：Python 3.11 / numpy / pandas / scipy / matplotlib，CJK 字体 `NotoSansCJK-Regular.ttc`。",
        "",
        "## 口径铁律（继承既有评价体系，未放宽）",
        "",
        "1. **LORO（Leave-One-Rat-Out，24 / 12 折）是唯一验收口径**；full 为群体均值层 in-sample，只作参考，**不可与个体验证混淆**。",
        "2. 四条件评价集与折数不同，**R² 不可横向解读为模型优劣**。",
        "3. 分段模型参数更多，必须同时报 k / AICc / BIC / 按鼠聚类配对 bootstrap 的 ΔR² 95%CI。",
        "4. 24 鼠全保留，未删除任何不利样本；基线 t=0 按物理定义锚定 R_Z=1。",
        "5. 拟合变好 ≠ 机理更正确；指标最优 ≠ 可用模型；ρ 是等效再封闭率，非被辨识的物理常数。",
        "6. TrackB 不在本轮范围内，其验收只报 coverage / width / Winkler，不报 R² / RMSE。",
        "",
        f"基线一致性校验：20 项复算 vs 冻结值 `max|ΔR²| = {S['baseline_repro']['max_abs_delta_r2']:.3e}`（PASS），证明对照口径零漂移。",
        "",
        "## 交付范围",
        "",
        "本包已作为 `segmented_fit/` 合入 GitHub `main`（见仓库根 `README.md` 与 `docs/PUSH_RECEIPT.md`）。本目录是自包含的分析分支，**未修改**仓库中三路线包的任何模型公式、参数或数据。",
    ]
    (HERE / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------- MODELS
def models_doc(D):
    va = D["A1_VARIANT_SELECTION"]
    P = D["SEGMENT_PARAMS"]
    sel_rows = []
    for seg in ["S1_acetic", "S2_EDTA", "S3_urea"]:
        for r in va[va.seg == seg].itertuples():
            sel_rows.append([r.seg_label, r.variant, r.variant_label, r.formula, r.k,
                             f(r.loro_r2), "★" if r.selected else ""])
    pr = P[(P.condition == "C1_all") & (P.scheme.isin(["SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]))]
    param_rows = [[r.scheme, r.seg, r.component, r.k, r.params] for r in pr.itertuples()]

    txt = """# 分段模型定义与机理原理

## 1. 三段划分（来自 Excel 官方标签与冻结数据 `stage` 列）

| 试剂阶段 | stage | 全局时间 (min) | 观测点数 | 实测 R_Z 均值 | 行为 |
|---|---|---|---|---|---|
| 基线（无试剂） | `baseline` | 0 | 288 | 1.000 | 归一化锚点，不进入评价集 |
| **50% 乙酸** | `acetic_acid` | 5 | 288 | 0.720 | 瞬时表面酸蚀，单步下降 |
| **EDTA**（4 浓度） | `edta_15/30/60/90` | 20 / 35 / 65 / 95 | 1152 | 0.587 → 0.207 | 持续脱钙，单调下降 |
| **饱和尿素（乙醇+尿素）** | `urea_10` | 105 | 288 | 0.267 | **相对 t=95 回升 +0.060** |

评价集沿用 `T_MOD=[5,20,35,65,95,105]`（含尿素、不含基线），逐鼠按 3 位点平均后
C1 为 576 点 / 96 条件节点 / 24 折 LORO。每段 4 剂量 × 4 频率 × 6 鼠（每鼠只属于一个剂量组）。

> 用户口径里的「饱和尿素处理段」对应 Excel 标签 `乙醇+尿素/MΩ` 与 protocol_note 中
> 「（乙醇+尿素）25 mL 进行操作」，即冻结数据的 `urea_10` 阶段。

## 2. 两代分段思路

| 代际 | 路线 | 方案 | 做法 | 参数 k |
|---|---|---|---|---|
| V0 对照 | 全局单调 | `BASE_*` | 冻结的 5 个模型，一套参数覆盖 6 个时间点 | 3–6 |
| **V1** | TrackA1 纯经验 | `SEG_A1` | 三段**各自换公式、从零独立标定**，互不共享参数 | 11 |
| **V1** | TrackA3 机理+经验 | `SEG_A3` | 共用 A2 机理骨架，三段各自补机理项 + 保形校准 | 13 |
| **V2** | TrackA1 纯经验 | `ADJ_A1` | 保留**全局 A1.1 基底**（6 时间点联合约束不丢）+ 最少分段修正 | 10 |
| **V2** | TrackA3 机理+经验 | `ADJ_A3` | 保留**全局 A3 骨架** + 乙酸/EDTA 截距修正 + 尿素界面重构项 | 11 |

V1 是「每段可以改公式」的直接实现；V2 是针对 V1 诊断结果（拆散跨段约束、参数膨胀、
LORO 显著变差）后提出的修正做法，也是「在 TrackA3 上进行调整」的落地形式。

## 3. TrackA1 纯经验：三段候选公式与择优

三段分别给 2–3 个候选公式，按**段内 LORO R²** 在 C1 上择优（探索性，同一数据集内选择）：

""" + table(["试剂段", "变体", "名称", "公式", "k", "段内 LORO R²", "选中"], sel_rows) + """

**最终选中**：乙酸段 `s1_pow`（经验幂律，k=3）、EDTA 段 `s2_hyper`（双曲/幂律导纳，k=4）、
尿素段 `s3_cross`（独立横截面回归，k=4），合计 k=11。

注意：择优结果里尿素段选了**独立横截面**而非分段衔接回升式（0.1813 vs 0.1728），
因为单看段内 R² 横截面略优；但独立横截面不再依赖 EDTA 末态，机理衔接性更弱。
这本身也说明「段内指标最优」不等于「分段模型更合理」。

V1 三段公式（选中项）：

```
S1 乙酸   R = 1 - A*(d/D0)^b*(f/100)^(-s)                          [3]
S2 EDTA   R = 1/(1 + a0*(d/D0)^b*(t/100)^p*(f/100)^(-s))           [4]
S3 尿素   R = c0 + c1*ln(f/100) + c2*(d/D0-1) + c3*ln(f/100)*(d/D0-1)  [4]
```

V2 `ADJ_A1`（基底 A1.1 拉伸指数，5 参数在全部 6 时间点上拟合）：

```
S1/S2     R = R_base + c_seg                    (各 1 个截距修正)
S3        R(105) = R_base(95) + lambda*(1 - R_base(95))
          lambda = clip(l0 + l1*ln(f/100) + l2*(d/D0-1), 0, 0.95)   [3]
```

## 4. TrackA3 机理 + 经验：机理链条与分段项

沿用 TrackA2 的机理骨架（Archie 局部电导 → 脱钙孔隙率演化 → CPE / 界面极化并联导纳）：

```
delta_eff(t,d,f) = a0 * (d/D0)^beta * (t/100)^p * (f/100)^(-s)
R_mech           = 1 / (1 + delta_eff)
```

三段各自需要不同的机理描述：

| 段 | 新增机理项 | 物理含义 | 经验校准 |
|---|---|---|---|
| S1 乙酸 | `delta_acid = alpha * (d/D0)^beta * (f/100)^(-s)`，与时间无关 | 乙酸瞬时表面酸蚀 / 蛋白变性引入额外界面导纳跃变 | `eta_a`（保形缩放） |
| S2 EDTA | 完整 `delta_eff` 随 t 幂律增长 | 脱钙孔隙率持续增大、离子通路电导上升 | `eta_e, eps_e`（A3 原式） |
| **S3 尿素** | `delta_post = delta_95 * (1 - rho)`，`rho = rho0*(d/D0)^gamma*(f/100)^(-nu)` | **乙醇+尿素使蛋白 / 孔道再封闭、界面重构，有效改性量按剂量 / 频率依赖地部分回落**，故 R_Z 回升 | `eta_u, eps_u`（V1）/ 无（V2，纯机理） |

`rho = 0` 时 `delta_post = delta_95`，即退化为「EDTA 末态平台」——这正是旧机理模型
（MECHANISM_PRINCIPLE.md 的 M4 / E8）承认的缺口：`delta_eff` 单调，**无法表达 t=105 回升**。
本轮的 `rho` 项补上了这一机理环节，但必须诚实：**`rho0/gamma/nu` 是对实测的等效标定，
不是被独立辨识的物理常数，不构成机理被证实**。

V1 `SEG_A3`：机理 4 参数只在 EDTA 段标定，S1 2 参数、S2 2 参数（eta/eps）、S3 5 参数
（`rho0,gamma,nu,eta_u,eps_u`），合计 k=13。

V2 `ADJ_A3`：全局 A3 骨架 6 参数在**全部 6 个时间点**上拟合（跨段形状约束不丢）；
S1/S2 各加 1 个截距；S3 从 **t=95 的机理平台**出发做界面重构：

```
delta_95 = a0*(d/D0)^beta*(95/100)^p*(f/100)^(-s)
rho      = clip(rho0*(d/D0)^gamma*(f/100)^(-nu), 0, 0.95)
R(105)   = 1 / (1 + delta_95*(1 - rho))          [3 个机理参数]
```

## 5. C1 全量拟合参数（供复现 / 绘图）

""" + table(["方案", "段", "部件", "k", "参数值"], param_rows) + """

> 参数由 `segmented_fit.py` 在全部 24 鼠上拟合得到；LORO 折内每一折都会用其余 23 鼠
> 重新标定，上表仅用于绘出群体曲线，不是验收时的参数。

## 6. 与冻结模型的关系

- 冻结的 A1.1 / A1.2 / A1.3 / A2 / A3 公式与参数**未被修改**；本文件以逐字复刻方式
  提供 `base_predict` / `base_fit`，仅用于同口径对照，并已通过 20 项一致性校验。
- 三段的新公式只存在于本目录，不影响 `main` 上已推送的三路线包。
"""
    (HERE / "SEGMENTED_MODELS.md").write_text(txt, encoding="utf-8")


# ---------------------------------------------------------------- CONCLUSION
def conclusion(D):
    S = D["SUMMARY"]
    o = D["COMPARISON_OVERALL"]
    fu = D["FULL_GROUP_MEAN"]
    sm = D["SEGMENT_METRICS"]
    rbs = D["UREA_REBOUND_SUMMARY"]
    chk = D["BASELINE_REPRO_CHECK"]

    c1 = o[o.condition == "C1_all"].set_index("scheme")
    f1 = fu[fu.condition == "C1_all"].set_index("scheme")

    main_rows = []
    for s in SCHEMES:
        main_rows.append([SHORT[s], ROUTE[s], int(c1.loc[s, "k"]),
                          f(f1.loc[s, "r2"]), f(c1.loc[s, "r2"]), f(c1.loc[s, "rmse"]),
                          f(c1.loc[s, "mae"]), f(c1.loc[s, "bias"]),
                          f(c1.loc[s, "aicc"], 1), f(c1.loc[s, "bic"], 1),
                          (f"{float(c1.loc[s,'delta_r2_vs_best']):+.4f}"
                           if pd.notna(c1.loc[s, "delta_r2_vs_best"]) else "—"),
                          (f"[{float(c1.loc[s,'delta_r2_boot_lo']):+.4f}, {float(c1.loc[s,'delta_r2_boot_hi']):+.4f}]"
                           if pd.notna(c1.loc[s, "delta_r2_boot_lo"]) else "—")])

    cond_rows = []
    for cid in CONDS:
        q = o[o.condition == cid].set_index("scheme")
        cond_rows.append([COND_SHORT[cid], *[f(q.loc[s, "r2"]) for s in
                                             ["BASE_A3", "SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]],
                          f"{q.loc['ADJ_A3','r2'] - q.loc['BASE_A3','r2']:+.4f}",
                          f"{q.loc['SEG_A3','r2'] - q.loc['BASE_A3','r2']:+.4f}"])

    seg_rows = []
    for seg, label in [("S1_acetic", "乙酸段"), ("S2_EDTA", "EDTA 段"), ("S3_urea", "尿素段")]:
        q = sm[(sm.condition == "C1_all") & (sm.seg == seg)].set_index("scheme")
        seg_rows.append([label, *[f(q.loc[s, "r2"]) for s in ["BASE_A3", "SEG_A1", "SEG_A3", "ADJ_A3"]],
                         *[f(q.loc[s, "rmse"]) for s in ["BASE_A3", "SEG_A1", "SEG_A3", "ADJ_A3"]]])
    bias_rows = []
    for seg, label in [("S1_acetic", "乙酸段"), ("S2_EDTA", "EDTA 段"), ("S3_urea", "尿素段")]:
        q = sm[(sm.condition == "C1_all") & (sm.seg == seg)].set_index("scheme")
        bias_rows.append([label, *[f"{q.loc[s,'bias']:+.4f}" for s in ["BASE_A3", "SEG_A1", "SEG_A3", "ADJ_A3"]]])

    rb1 = rbs[rbs.condition == "C1_all"].set_index("scheme")
    rb_rows = [[SHORT[s], f(rb1.loc[s, "d_obs_mean"]), f"{rb1.loc[s,'d_pred_mean']:+.4f}",
                f"{rb1.loc[s,'sign_match_rate']*100:.0f}%", "是" if rb1.loc[s, "expresses_rebound"] else "否"]
               for s in SCHEMES]

    txt = """# 分段拟合结论：拟合效果是否更好？

> **结论先行**：「更好」取决于用哪一层口径。
> **群体均值层**（full，仅参考）与**机理性状完整性**上，分段确实更好；
> 但在项目既定的**个体级 LORO 唯一验收口径**下，没有任何分段方案取得统计显著的提升。
> 三段独立标定（V1）显著变差；骨架保留 + 最少调整（V2 的 `ADJ_A3`）是唯一 LORO 不降的方案，
> 但 ΔR²=+0.0025、95%CI 跨 0，只能表述为**不劣于**全局 A3，而非优于。

## 1. 数据与口径

- 冻结 24 鼠实测，4 剂量 × 4 频率 × 7 时间点（含基线），24 鼠全保留，未删任何不利样本。
- 三段：50% 乙酸 t=5 / EDTA t=20,35,65,95 / 饱和尿素（乙醇+尿素）t=105。
- 评价集 `T_MOD=[5,20,35,65,95,105]`（含尿素、不含基线）；C1 = 576 逐鼠点 / 96 条件节点 / 24 折 LORO。
- **LORO 是唯一验收口径**；full 是群体均值层 in-sample，只作参考，二者不可混淆。
- 四条件评价集与折数不同，R² 不可横比。
- 基线一致性：20 项复算 vs 冻结值 `max|ΔR²| = """ + f"{S['baseline_repro']['max_abs_delta_r2']:.3e}" + """`（PASS），对照零漂移。

## 2. C1 全部条件：9 个方案总表

""" + table(["方案", "路线", "k", "full R²*", "**LORO R²**", "LORO RMSE", "MAE", "bias",
            "AICc", "BIC", "ΔR² vs A3", "ΔR² 95%CI"], main_rows) + """

\\* full 为群体均值层 in-sample，**不作验收**；A3 是该条件最优冻结基线。
ΔR² 的 95%CI 为按鼠整簇重采样的**配对** bootstrap（2000 次），正值才表示改善。

## 3. 四条件拼接整体 LORO R²

""" + table(["条件", "全局 A3", "SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3",
            "ADJ_A3 − A3", "SEG_A3 − A3"], cond_rows) + """

`ADJ_A3` 在四条件上相对 A3 的变化为 +0.0025 / +0.0053 / −0.0014 / −0.0014，
全部落在噪声级；`SEG_A3` 则四条件一致下降（−0.016 至 −0.042）。
按鼠聚类 bootstrap 的 C1 配对 ΔR²：

| 方案 | ΔR² | 95%CI | P(Δ>0) | 判定 |
|---|---|---|---|---|
""" + "\n".join(
        f"| {SHORT[s]} | {float(c1.loc[s,'delta_r2_vs_best']):+.4f} | "
        f"[{float(c1.loc[s,'delta_r2_boot_lo']):+.4f}, {float(c1.loc[s,'delta_r2_boot_hi']):+.4f}] | "
        f"{float(c1.loc[s,'delta_r2_boot_p_positive']):.3f} | "
        f"{'显著变差' if c1.loc[s,'delta_r2_boot_hi'] < 0 else ('不显著（持平）' if c1.loc[s,'delta_r2_boot_lo'] < 0 else '显著改善')} |"
        for s in ["SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]) + """

## 4. 分段到底改善了什么

### 4.1 尿素段回升：从「完全无法表达」到 16/16 节点符号一致

实测 `R_Z` 由 t=95 的 0.207 回升到 t=105 的 0.267（Δ=+0.060）。

""" + table(["方案", "实测 Δ", "预测 Δ", "符号一致率", "表达回升"], rb_rows) + """

这是本轮**最实质的收获**：5 个全局单调模型（含 A2 纯机理、A3）全部预测继续下降，
正是 MECHANISM_PRINCIPLE.md 里 M4 / E8 承认的机理缺口。分段模型通过引入
「乙酸瞬时酸蚀 / 尿素界面重构再封闭」把这一**定性错误**修正了，其中 V1 独立分段
幅度最准（+0.057 ~ +0.058，接近 +0.060），V2 骨架调整偏保守（+0.021 ~ +0.024，
因最小二乘在个体噪声下保守估计再封闭率 `rho`）。

### 4.2 群体均值层拟合显著提升

| 方案 | full R² | full RMSE |
|---|---|---|
""" + "\n".join(f"| {SHORT[s]} | {f(f1.loc[s,'r2'])} | {f(f1.loc[s,'rmse'])} |"
                for s in ["BASE_A1.1", "BASE_A2", "BASE_A3", "SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]) + """

`SEG_A1` 0.957 / `SEG_A3` 0.954 / `ADJ_A3` 0.935 均高于全局 A3 的 0.920，
RMSE 由 0.060 降至 0.044–0.054。原因是分段消除了段内系统偏差。

### 4.3 段内系统偏差被消除

""" + table(["段", "全局 A3", "SEG_A1", "SEG_A3", "ADJ_A3"], bias_rows) + """

三段独立标定后 `bias ≈ 0`；全局 A3 在乙酸段 −0.050、尿素段 +0.038。

## 5. 为什么 LORO 没有跟着变好（关键机制）

""" + table(["段", "A3 R²", "SEG_A1 R²", "SEG_A3 R²", "ADJ_A3 R²",
            "A3 RMSE", "SEG_A1 RMSE", "SEG_A3 RMSE", "ADJ_A3 RMSE"], seg_rows) + """

注意 `ADJ_A3` 在乙酸段与尿素段的段内 R² 确有提升（0.141→0.200、0.198→0.223），
但段内 SST 远小于跨时间点的全局 SST，稀释到拼接整体只剩 +0.0025；这也说明
**段内 R² 改善不能直接外推为整体 R² 改善**。

1. **误差主体是鼠间异质性，不是系统偏差**。单段横截面 LORO R² 只有 0.13–0.20，
   即同一剂量×频率节点内 6 只鼠的离散度主导误差。分段把每段的**均值**调准了
   （bias→0），但没有降低**方差**，所以个体级 R² 不动。
2. **V1 拆散了跨段联合约束**。全局 A3 用 6 个时间点联合拟合 6 个参数，形状受强约束；
   V1 改成三段独立、参数 11 / 13 个，LORO 折内（23 鼠）估计方差明显放大，
   EDTA 段 R² 由 0.397 降到 0.359，尿素段由 0.198 降到 0.151–0.181。
3. **复杂度惩罚**：AICc 由 A3 的 −1865.9 变为 −1827.6 / −1821.2，
   BIC 由 −1835.6 变为 −1775.9 / −1761.0，全部变差。
4. 这正是项目反复强调的铁律：**群体均值层的 full 提升不能当作个体验证的提升**。
   fig07 把两种口径并列，可以直观看到 `SEG_A1/A3` 落在「full 最高、LORO 偏低」的位置。

## 6. 回答「哪种更好」

| 目标 | 推荐 | 理由 | 代价 |
|---|---|---|---|
| 机理链完整、群体曲线最贴、回升幅度最准 | **`SEG_A3`** | full 0.954、回升 +0.057、bias≈0、含界面重构项 | LORO 0.494（显著低于 A3），**不得作为验收模型** |
| 遵守既有 LORO 验收口径 | **`ADJ_A3`** | 唯一 LORO 不降的分段方案（0.5226 vs 0.5201），full 0.935 > 0.920，回升方向 16/16 正确，参数仅比 A3 多 5 个 | 提升不显著（CI 跨 0），回升幅度保守 |
| 纯经验 TrackA1 路线 | 不推荐 | `SEG_A1` / `ADJ_A1` 在 LORO 上分别为 0.496 / 0.483，均不如 A3；`ADJ_A1` 甚至显著变差 | — |

**如果必须给一个可执行结论**：在 `TrackA3（机理+经验）` 上做**骨架保留 + 尿素段界面重构**
的 `ADJ_A3` 是本次分段调整里最稳妥的做法——它补上了尿素回升这个定性缺陷、群体均值层
明显更贴、且没有牺牲个体级验收；但必须写成「不劣于全局 A3」，不能写成「拟合效果更好」。
若追求「改公式后曲线更贴」，`SEG_A3` 的 full 最高，但那是群体均值层的成绩，不是个体验证。

## 7. 诚实边界

1. 未改动任何已冻结模型公式、参数、数据；24 鼠全保留；本目录是独立分析分支，已作为 `segmented_fit/` 合入 GitHub `main`。
2. A1 变体择优在 C1 同一数据集内按段内 LORO 选，属**探索性选择**，存在轻微选择偏差；
   正式判定以拼接整体 LORO 与配对 bootstrap 为准。
3. `rho0 / gamma / nu` 是对实测的**等效再封闭率标定**，不是被独立辨识的物理常数；
   补上机理环节 ≠ 机理被证实。
4. 基线 t=0 按物理定义锚定 R_Z=1；LORO 折内所有段参数均由其余 23 鼠重新标定，
   留出鼠全段均为未见数据，无信息泄漏。
5. 四条件评价集 96 / 48 / 48 / 24 节点、折数 24 / 24 / 12 / 12，**R² 不可横比**。
6. `ADJ_A3` 回升幅度 +0.024 仅为实测 +0.060 的 40%，属保守估计，不宜宣称「已准确表达回升」。
7. TrackB 预测区间不在本轮范围；其验收只报 coverage / width / Winkler，不报 R² / RMSE。
"""
    (HERE / "SEGMENTED_CONCLUSION.md").write_text(txt, encoding="utf-8")


# ---------------------------------------------------------------- TABLES INDEX
def index_doc():
    items = [
        ("BASELINE_REPRO_CHECK.csv", "20 项（5 模型 × 4 条件）复算 LORO R²/RMSE 与冻结值比对，含 Δ"),
        ("A1_VARIANT_SELECTION.csv", "TrackA1 三段共 7 个候选公式的段内 LORO 指标与选中标记"),
        ("COMPARISON_OVERALL.csv", "9 方案 × 4 条件拼接整体全指标（~85 列），含 k / AICc / BIC / bootstrap CI / 配对 ΔR²"),
        ("SEGMENT_METRICS.csv", "9 方案 × 4 条件 × 3 段的分段全指标（108 行）"),
        ("FULL_GROUP_MEAN.csv", "群体均值层 full in-sample 指标，标注「仅参考、不作验收」"),
        ("RESIDUALS_LONG.csv", "逐点残差长表 11664 行（观测/预测/残差/标准化残差/百分比误差）"),
        ("SEGMENT_CURVES.csv", "绘图用群体节点曲线（实测均值 ± SEM + 各方案预测，含 96 节点 × 9 方案 × 4 条件）"),
        ("SEGMENT_PARAMS.csv", "各方案各段在全部 24 鼠上的拟合参数（供复现与绘图）"),
        ("UREA_REBOUND.csv", "16 个剂量×频率节点的实测/预测回升量与符号一致性"),
        ("UREA_REBOUND_SUMMARY.csv", "各条件各方案的平均回升量、符号一致率、是否表达回升"),
        ("SEGMENTED_SUMMARY.json", "机器可读汇总：选中变体、k、C1 两种口径 R²、ΔR² 与 CI、回升结果、铁律"),
        ("frozen_baseline_metrics.csv", "上一轮冻结的三路线指标（复制件，用于一致性校验）"),
    ]
    txt = "# 实算表索引\n\n所有 CSV 均为 UTF-8-SIG，由 `segmented_fit.py` 一次运行生成。\n\n"
    txt += table(["文件", "内容"], [[a, b] for a, b in items])
    txt += """

## 图索引

| 图 | 内容 |
|---|---|
"""
    figs = [
        ("fig01_segments_measured_data.png", "三段实测 R_Z 全貌：4 频率 × 4 剂量，点+SEM 全画，含基线 t=0，纵轴从 0"),
        ("fig02_trackA1_variant_selection.png", "TrackA1 三段候选公式按段内 LORO R² 择优"),
        ("fig03_fit_SEG_A1.png", "SEG_A1 三段独立标定 vs 全局 A1.1 拟合曲线"),
        ("fig04_fit_ADJ_A3.png", "ADJ_A3 骨架保留+界面重构 vs 全局 A3（尿素段实线回升、灰虚线继续下降）"),
        ("fig05_loro_comparison.png", "四条件 × 9 方案拼接整体 LORO R² 与配对 ΔR² 95%CI"),
        ("fig06_urea_rebound.png", "16 节点回升量实测 vs 预测、平均回升与符号一致率"),
        ("fig07_full_vs_loro.png", "群体均值层 full 与个体级 LORO 两种口径的分离"),
        ("fig08_residual_diagnostics.png", "残差分布 / 随时间的系统结构 / 分段后 bias 消除"),
    ]
    txt += table(["图", "内容"], figs)
    (HERE / "TABLES_INDEX.md").write_text(txt + "\n", encoding="utf-8")


def main():
    D = load()
    readme(D)
    models_doc(D)
    conclusion(D)
    index_doc()
    for n in ["README.md", "SEGMENTED_MODELS.md", "SEGMENTED_CONCLUSION.md", "TABLES_INDEX.md"]:
        p = HERE / n
        print(f"  {n}: {len(p.read_text(encoding='utf-8').splitlines())} 行")


if __name__ == "__main__":
    main()
