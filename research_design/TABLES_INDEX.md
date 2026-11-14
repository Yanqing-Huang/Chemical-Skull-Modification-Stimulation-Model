# 数据表索引

全部表格由 `build_master.py` 从**已冻结成果**整合生成，不含重新拟合结果。

| 文件 | 规模 | 内容 |
|---|---|---|
| MASTER_COMPARISON.csv | 136 行 × 30 列 | 统一主表：track / prediction_target / fit_mode / route / model_family / model / level / condition / condition_scope + 核心指标 + 区间指标 |
| MATRIX_FITMODE_X_SCOPE.csv | 24 行 | 3 拟合方式 × 2 条件范围 × 4 路线的 LORO 矩阵，含空格标记 |
| GLOBAL_VS_SEGMENTED.csv | 16 行 | 分段方案相对全局 A3 的配对 ΔR²、bootstrap CI、AICc/BIC |
| ALL_VS_BEST_CONDITION.csv | 4 行 | 四条件收窄路径：规模、三路线 LORO、TrackB 区间 |
| CONDITION_SCOPE.csv | 4 行 | 同上（条件范围定义副本） |
| TRACKA_THREE_ROUTES.csv | 20 行 | 全局三路线 × 四条件的 full / LOCO / LORO 指标 |
| TRACKB_INTERVALS.csv | 4 行 | TrackB 四条件区间验收指标 + 诊断性中心点指标 |
| PER_SEGMENT_METRICS.csv | 108 行 | 分段逐段（乙酸 / EDTA / 尿素）指标 |
| UREA_REBOUND_SUMMARY.csv | 36 行 | 各方案 × 四条件的尿素回升幅度与 16 节点符号一致率 |
| A1_VARIANT_SELECTION.csv | 7 行 | TrackA1 各段候选公式择优明细（C1） |
| CONSISTENCY_CHECK.csv | 60 行 | 分段包 BASE_* 与三路线冻结值的 LORO/full 逐项比对 |
| FRAMEWORK_GAPS.csv | 8 行 | 框架中未实现的空格子及原因 |

## 列说明（MASTER_COMPARISON）

| 列 | 含义 |
|---|---|
| `track` | TrackA（点预测）/ TrackB（区间预测） |
| `prediction_target` | 输出对象：点值 R_Z 或区间 [lo, hi] |
| `fit_mode` | 全局 / 分段独立 / 分段增量 |
| `route` | TrackA1 / TrackA2 / TrackA3 / TrackB |
| `model_family` | 纯经验 / 纯机理 / 机理驱动经验 / 区间集成 |
| `level` | full / LOCO / **LORO**（唯一验收）/ COVERAGE（TrackB 专用） |
| `condition_scope` | 所有条件 / 低频切片 / 高浓度切片 / **最佳条件范围** |
| `r2`…`delta_r2_boot_hi` | 点预测核心指标与配对 bootstrap 区间 |
| `coverage` / `mean_width` / `winkler_mean` | 仅 TrackB 行有值 |
| `source` | 数值来源文件（可回溯到冻结包） |
| `note` | 口径标注 |

## 数据来源

| 来源包 | 文件 |
|---|---|
| `Chemical_Skull_Model_ThreeRoutes` | `three_routes_metrics.csv`、`github_package/four_conditions/ALL_METRICS_FULL.csv`、`TRACKB_FULL_METRICS.csv`、`trackB_four_conditions.csv` |
| `Chemical_Skull_Model_SegmentedFit` | `tables/COMPARISON_OVERALL.csv`、`FULL_GROUP_MEAN.csv`、`SEGMENT_METRICS.csv`、`UREA_REBOUND_SUMMARY.csv`、`A1_VARIANT_SELECTION.csv` |
| 冻结实测 | `data/frozen_measured_24rat.csv`（SHA256 与两包副本一致） |

