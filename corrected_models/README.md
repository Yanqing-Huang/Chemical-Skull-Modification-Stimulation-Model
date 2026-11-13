# Corrected Models — Track A (A1) & Track B (calibrated intervals)

本目录包含**矫正 / 校准后的两条模型链路**，可直接调用、复现与核对。
数据口径：仅低频 **100 / 120 Hz**（高频 1000 / 10000 Hz 已剔除）；24 只大鼠、3 位点、
4 种 EDTA 剂量、7 个时间点全部保留，未删除任何不利样本。

> 这是相对上游原始模型（A0 机理先验 / B0 校准基准）的**增量交付**。
> 原始 A0/B0 代码与结果保留在本仓库原有位置，本目录不覆盖、不修改任何既有文件。

---

## 1. 目录结构

```
corrected_models/
├── README.md                         ← 本文件（总说明）
├── trackA_A1/
│   ├── model.py                      ← Track A A1 校正点预测模型（可 import）
│   ├── parameters.json               ← a, b 参数 + 全层级指标
│   └── trackA_A1_metrics.csv         ← 拟合指标明细
├── trackB_calibrated/
│   ├── model.py                      ← Track B 校准区间模型（可 import）
│   ├── parameters.json               ← 校准配置 + 覆盖率/宽度指标
│   └── calibrated_intervals.csv      ← 校准后最终区间（196 点，含 LB/UB/中点）
├── data/
│   ├── frozen_measured_24rat.csv     ← 冻结的 24 鼠实测数据（权威源）
│   ├── data_manifest.json            ← 数据 SHA-256 + 异常/未决记录
│   └── mechanism_timeseries_quantiles.csv  ← 原始机理先验（R_Z_raw 来源）
├── figures/                          ← 7 张拟合效果图（纵坐标从 0，N=6 均值±SEM，7 时间点完整）
│   ├── figA_trackA_fit.png           Track A 校正拟合时间曲线
│   ├── figB_trackA_scatter.png       预测值 vs 真实值
│   ├── figC_trackB_fit.png           Track B 校准区间时间曲线
│   ├── figD_trackB_coverage.png      逐点覆盖率/宽度诊断
│   ├── figE_trackA_residuals.png     Track A 残差诊断
│   ├── figF_trackA_error_by_dose.png 按剂量误差对比
│   └── figG_table1.png              表1 指标汇总
└── validation/                       ← 口径与验证报告
    ├── A1_LORO_REPORT.md            A1 大鼠级独立验证
    ├── B1_DIAGNOSTICS.md            Track B 区间误差来源诊断
    ├── data_difference_report.md    数据异常核查与冻结
    ├── FITTING_EFFECT.md            拟合效果精简结论
    ├── make_fitting_effect.py       全套图 + 指标复现脚本
    └── run_trackA_a1_fit.py         A1 原始拟合脚本
```

---

## 2. Track A — A1 经验校正模型（点预测）

**定义**：`R_Z_corr = a + b · R_Z_raw`，`a = −4.4715`，`b = 5.6806`。
`R_Z_raw` 为原始机理先验 `whole_resistance_ratio` 的 q50；`R_Z = Z/Z_0`。

```python
from trackA_A1.model import predict, raw_prior, info
predict(0.05, 35)          # 剂量0.05M、t=35min 的 A1 校正预测
raw_prior(0.10, 65)        # 原始机理先验（未校正）
```

| 验证层级 | R² | RMSE | MAE | Bias |
|---|---|---|---|---|
| A0 原始机理模型（群体均值层 n=48） | −7.906 | 0.533 | 0.510 | −0.510 |
| **A1 校正（full OLS, n=48）** | **0.782** | **0.083** | 0.069 | 0.000 |
| A1 群体均值 LOCO（留一条件） | 0.762 | 0.087 | — | — |
| A1 个体级 LORO（留一大鼠 n=288） | **0.588** | 0.130 | 0.105 | 0.001 |

**必须注意**：
- A1 是**经验线性校正**，不是机理改进。它只修正幅度偏差，原始公式
  `σ = 1e-4 + 1.5·φ^m`、`R = Σ Δx_i/(σ_i·A_i)` 未改动。
- **0.588 是个体级 LORO 总体 R²**，逐鼠 R² 均值仅 0.404（WT17、WT2 为负）；
  **0.762 是群体条件均值 LOCO**。两者预测层级不同（单鼠外推 vs 群体均值），
  **不能写成「性能从 0.762 下降到 0.588」**。
- 基线 t=0 按物理定义锚定 `R_Z = 1.0`；机理先验在 edta_90(95min) 后截断，
  尿素点(105min)为末值外推。

---

## 3. Track B — 校准后的嵌套 LOO 预测区间

```python
from trackB_calibrated.model import predict_interval, summary, apply_padding
predict_interval("WT1", 0.05, 100, 20)   # -> {LB, UB, center, covered, ...}
summary()                                 # -> 覆盖率/宽度/Winkler 汇总
```

**校准公式**（折×频率分层 conformal，目标覆盖 90%）：
`p_(fold,f) = tolerance_f · q_f`，`q_f = Quantile_0.90(内层 cross-fit 分数 s)`；
`lower = clip(q05 − p, 0, 1)`，`upper = clip(q95 + p, 0, 1)`。
嵌套结构：22 外层留一大鼠折 × 5 内层 grouped 折。

| 指标 | 值 |
|---|---|
| 逐点覆盖率（目标 90%） | **0.913** |
| 平均区间宽度 | 0.366 |
| 平均 Winkler (α_mis=0.10) | 0.454 |
| n | 196 点 = 22 鼠 × 2 频率 × 5 时间点 |

**必须注意**：
- Track B **不报告 R² / RMSE**；覆盖率不可与 Track A 的 R² 混比。
- 逐点覆盖率 ≠ 整条时间曲线同时落入区间的概率。
- 合并 0.913 掩盖了两种 ensemble 模式分裂：
  `TRAINING_COMPATIBLE_ENSEMBLE` 0.857（欠覆盖）/
  `NEAREST_TRAINING_ENSEMBLE_NOT_ORIGINAL_COMPATIBLE_SET` 0.969（过补偿）。
- 区间宽度约 **97% 来自 conformal padding**，基础 LHS 参数不确定仅约 3%
  —— 过宽根因是校准层补偿「仿真 vs 实测」偏差，**不是**参数采样不足。
- 0.05 M 只出现在 TRAINING_COMPATIBLE 模式，「0.05 M 欠覆盖」是**模式混淆**。
- 验证集仅 **22/24 鼠**（缺 WT5 0.10M、WT6 0.15M）。

---

## 4. 数据版本与已知治理问题（OPEN ISSUES）

- 权威数据源：`data/frozen_measured_24rat.csv`（SHA 见 `data/data_manifest.json`）。
- **未决 1**：本地 Excel 文件 SHA 与上游 manifest 声明的 `source_data` SHA 不一致，
  上游原始文件不可得 → 不能宣布数据版本完全一致。
- **未决 2**：上游 manifest 声明 `mechanism_id=157` / `two_sided_50um` 为**全数据诊断选定**，
  非每个外层训练折内重选 → 存在潜在乐观偏差，待 B3 严格折内验证。
- **未决 3**：Track B 验证集 22/24 鼠，缺鼠原因待澄清。
- 14 处 Excel↔CSV 不一致（urea_10/Midpoint/WT21–24）标记 `UNRESOLVED_TRANSCRIPTION`
  保留、未静默覆盖；32 条重复录入（16 组）已按「保留 CSV 一致值 + 冲突留痕」去重。
- 频率范围：仅 100/120 Hz；`R_Z = 0.85` 为参考阈值而非饱和值。

---

## 5. 复现

```bash
# 模型自检（无需外部依赖，仅 numpy/pandas）
python3 trackA_A1/model.py
python3 trackB_calibrated/model.py

# 全套拟合效果图 + 指标（需修改 validation/make_fitting_effect.py 顶部路径常量为本目录）
python3 validation/make_fitting_effect.py
```

所有指标均基于**同一组真值/预测值**计算并标注验证层级；N=6 的误差棒为
SEM（标准误），表示实验观测均值离散度，**不等于**模型预测区间。

---

## 6. 变更边界（相对上游）

| 项 | 上游 A0/B0 | 本交付 |
|---|---|---|
| Track A 点预测 | 原始机理先验（幅度偏差大） | + A1 经验校正（可开关，A0 保留） |
| Track B 区间 | 校准基准框架 | 保留框架，交付校准后最终区间 + 诊断 |
| 数据 | 原始 Excel | 冻结 24 鼠 CSV + SHA + 异常留痕 |
| 频率 | 含 1000/10000 Hz | 仅 100/120 Hz |
| 验证口径 | 群体/个体混用 | 明确区分群体 LOCO 与个体 LORO |

`A1` 若进入 Track B 的仿真生成链路，必须**重新生成 ensemble 预测并重新校准**，
不可直接沿用本目录的 padding（见 validation/B1_DIAGNOSTICS.md）。
