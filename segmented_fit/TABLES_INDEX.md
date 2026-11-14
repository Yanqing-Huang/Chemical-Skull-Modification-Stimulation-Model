# 实算表索引

所有 CSV 均为 UTF-8-SIG，由 `segmented_fit.py` 一次运行生成。

| 文件 | 内容 |
|---|---|
| BASELINE_REPRO_CHECK.csv | 20 项（5 模型 × 4 条件）复算 LORO R²/RMSE 与冻结值比对，含 Δ |
| A1_VARIANT_SELECTION.csv | TrackA1 三段共 7 个候选公式的段内 LORO 指标与选中标记 |
| COMPARISON_OVERALL.csv | 9 方案 × 4 条件拼接整体全指标（~85 列），含 k / AICc / BIC / bootstrap CI / 配对 ΔR² |
| SEGMENT_METRICS.csv | 9 方案 × 4 条件 × 3 段的分段全指标（108 行） |
| FULL_GROUP_MEAN.csv | 群体均值层 full in-sample 指标，标注「仅参考、不作验收」 |
| RESIDUALS_LONG.csv | 逐点残差长表 11664 行（观测/预测/残差/标准化残差/百分比误差） |
| SEGMENT_CURVES.csv | 绘图用群体节点曲线（实测均值 ± SEM + 各方案预测，含 96 节点 × 9 方案 × 4 条件） |
| SEGMENT_PARAMS.csv | 各方案各段在全部 24 鼠上的拟合参数（供复现与绘图） |
| UREA_REBOUND.csv | 16 个剂量×频率节点的实测/预测回升量与符号一致性 |
| UREA_REBOUND_SUMMARY.csv | 各条件各方案的平均回升量、符号一致率、是否表达回升 |
| SEGMENTED_SUMMARY.json | 机器可读汇总：选中变体、k、C1 两种口径 R²、ΔR² 与 CI、回升结果、铁律 |
| frozen_baseline_metrics.csv | 上一轮冻结的三路线指标（复制件，用于一致性校验） |

## 图索引

| 图 | 内容 |
|---|---|
| 图 | 内容 |
|---|---|
| fig01_segments_measured_data.png | 三段实测 R_Z 全貌：4 频率 × 4 剂量，点+SEM 全画，含基线 t=0，纵轴从 0 |
| fig02_trackA1_variant_selection.png | TrackA1 三段候选公式按段内 LORO R² 择优 |
| fig03_fit_SEG_A1.png | SEG_A1 三段独立标定 vs 全局 A1.1 拟合曲线 |
| fig04_fit_ADJ_A3.png | ADJ_A3 骨架保留+界面重构 vs 全局 A3（尿素段实线回升、灰虚线继续下降） |
| fig05_loro_comparison.png | 四条件 × 9 方案拼接整体 LORO R² 与配对 ΔR² 95%CI |
| fig06_urea_rebound.png | 16 节点回升量实测 vs 预测、平均回升与符号一致率 |
| fig07_full_vs_loro.png | 群体均值层 full 与个体级 LORO 两种口径的分离 |
| fig08_residual_diagnostics.png | 残差分布 / 随时间的系统结构 / 分段后 bias 消除 |
