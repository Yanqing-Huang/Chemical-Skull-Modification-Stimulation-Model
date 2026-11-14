# TrackB B1 · 区间误差来源诊断 (Why is padding so wide?)

## 1. 总体 (100/120 Hz, n=196, 大鼠=22)

- 校准后覆盖率 = **0.9133**（目标 0.9）；
  按大鼠聚类的 bootstrap 95% CI = [0.864, 0.970]（避免同鼠多时间点被当独立样本）。
- 平均区间宽度 = 0.3662，中位 0.3650，最大 0.4459。
- **宽度构成**：基础 LHS 区间（sim_q95−sim_q05）占比 = **3.00%**；padding 占比 = **97.09%**（双侧均值 padding=0.3596，基础宽度=0.0110）。
- 平均 Winkler(α_mis=0.10) = 0.4100。

## 2. 结论先行：宽度从哪来？

- 区间宽度 **98%+ 来自 conformal padding**，基础 LHS 参数不确定性几乎可忽略（<2%）。即：模型本身对参数的不确定很小，
  宽区间是**校准层为补偿『仿真 vs 实测』系统性/个体差异**而加的加性余量，并非参数采样不足。
- 因此 B1 阶段**不应继续扩大参数采样范围**，而应查清 padding 是否在补偿：(i) 系统性偏差，或 (ii) 大鼠间异质性。

## 3. 分层统计

### 3.1 按 ensemble_role

| role | n | coverage | mean_width | mean_pad_share | mean_winkler | mean\|resid\| |
|---|---|---|---|---|---|---|
| NEAREST_TRAINING_ENSEMBLE_NOT_ORIGINAL_COMPATIBLE_SET | 98 | 0.969 | 0.3564 | 0.990 | 0.3630 | 0.0818 |
| TRAINING_COMPATIBLE_ENSEMBLE | 98 | 0.857 | 0.3760 | 0.951 | 0.4570 | 0.1018 |

### 3.2 按频率 / 阶段 / 剂量

| 分层 | stratum | n | coverage | mean_width | mean_pad_share | mean\|resid\| |
|---|---|---|---|---|---|---|
| frequency_hz | 100.0 | 98 | 0.918 | 0.3690 | 0.972 | 0.0926 |
| frequency_hz | 120.0 | 98 | 0.908 | 0.3634 | 0.970 | 0.0909 |
| stage | acetic_acid | 44 | 0.864 | 0.3645 | 0.987 | 0.1052 |
| stage | edta | 152 | 0.928 | 0.3667 | 0.966 | 0.0879 |
| edta_concentration_mol_per_l | 0.05 | 56 | 0.768 | 0.3709 | 0.940 | 0.1201 |
| edta_concentration_mol_per_l | 0.1 | 48 | 0.958 | 0.3414 | 0.988 | 0.0718 |
| edta_concentration_mol_per_l | 0.15 | 42 | 1.000 | 0.3665 | 0.989 | 0.0619 |
| edta_concentration_mol_per_l | 0.2 | 50 | 0.960 | 0.3845 | 0.974 | 0.1043 |

### 3.3 ensemble_role × 剂量 覆盖率（验证 0.05M 是否受 mode 混淆）

| role | dose | n | coverage | mean_width |
|---|---|---|---|---|
| NEAREST_TRAINING_ENSEMBLE_NOT_ORIGINAL_COMPATIBLE_SET | 0.1 | 38 | 0.947 | 0.3310 |
| NEAREST_TRAINING_ENSEMBLE_NOT_ORIGINAL_COMPATIBLE_SET | 0.15 | 42 | 1.000 | 0.3665 |
| NEAREST_TRAINING_ENSEMBLE_NOT_ORIGINAL_COMPATIBLE_SET | 0.2 | 18 | 0.944 | 0.3865 |
| TRAINING_COMPATIBLE_ENSEMBLE | 0.05 | 56 | 0.768 | 0.3709 |
| TRAINING_COMPATIBLE_ENSEMBLE | 0.1 | 10 | 1.000 | 0.3807 |
| TRAINING_COMPATIBLE_ENSEMBLE | 0.2 | 32 | 0.969 | 0.3834 |

## 4. 个体异质性（按大鼠）

- 覆盖率低于目标 0.9 的大鼠：['WT17', 'WT9', 'WT8', 'WT2', 'WT20', 'WT7']。
- 覆盖率最低 3 只：[{'rat': 'WT17', 'coverage': 0.4, 'mean_width': 0.35530699602005755, 'mean_abs_resid': 0.21151023269331506}, {'rat': 'WT9', 'coverage': 0.5, 'mean_width': 0.4000964295850191, 'mean_abs_resid': 0.1685408351930579}, {'rat': 'WT8', 'coverage': 0.75, 'mean_width': 0.36185963647500907, 'mean_abs_resid': 0.10693647858359678}]。
- 若各鼠 padding / 残差差异大，说明误差具异方差性，后续 B2 应优先尝试**分层校准**而非统一 padding。

## 5. OPEN ISSUE

- TrackB 外层验证集仅含 **22/24 鼠**（缺 WT5、WT6）；本报告 B1 结论仅基于这 22 鼠，进入 B2/B3 前须澄清缺失原因。
- 0.05M 剂量覆盖率偏低（前序报告 76.79%）与 ORIGINAL_COMPATIBLE 集成模式高度相关，属**模式混淆**而非单纯剂量效应，B2 比较时应控制 ensemble_role。

## 6. B1 阶段结论（不改算法）

- 区间过宽的根因是 **conformal padding 在补偿仿真-实测差异**，而非 LHS 参数不确定。
- 是否『过度补偿』不能仅看覆盖率（91%≈目标）；需看 B2 不同校准法能否在**同等覆盖率**下收窄宽度。B1 只定位来源，不调整。