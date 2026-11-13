# 最新模型拟合效果（矫正后 / 校准后）

## Track A — A1 经验校正模型  R_Z_corr = a + b·R_Z_raw

- 校正参数（群体均值层全数据拟合）：**a = -4.4715, b = 5.6806**
- 群体均值层（n=48 条件，留一条件 LOCO 对照）：R²=0.7824, RMSE=0.0833, MAE=0.0693, Bias=0.0000
- 留一条件 LOCO：R²=0.7624, RMSE=0.0871
- **留一大鼠 LORO（单鼠泛化）：R²=0.5883, RMSE=0.1298** （逐鼠 R² 均值 0.6092）
- 结论：A1 在群体均值层显著提升 R²、消除系统性 Bias；但 LORO 显示单鼠外推弱于群体层，A1 仍属经验修正、非机理改进。

## Track B — 校准后的嵌套 LOO 预测区间

- **逐点覆盖率 = 0.913**（目标 90%），n=196 点 = 22 鼠 × 2 频率 × 6 时间点
- **平均区间宽度 = 0.366**；平均 Winkler(α_mis=0.10) = 0.454
- 统计口径：逐点覆盖率（各时间点观测落入其区间的比例），**不等于**整条曲线同时落入区间的概率
- 两种 ensemble 模式分裂（合并 0.913 掩盖差异）：

| ensemble 模式 | 覆盖率 | n | 平均宽度 |
|---|---|---|---|
| NEAREST_TRAINING_ENSEMBLE_NOT_ORIGINAL_COMPATIBLE_SET | 0.969 | 98 | 0.356 |
| TRAINING_COMPATIBLE_ENSEMBLE | 0.857 | 98 | 0.376 |
- 缺鼠：验证集仅 22/24 鼠（缺 WT5 0.10M、WT6 0.15M）
- 区间过宽根因：~97% 来自 conformal padding，基础 LHS 参数不确定仅 ~3%

## 附图（完整 7 张 + 表1）

- 图A `figA_trackA_fit.png`：Track A 校正拟合（实测±SD vs A1 预测，剂量×频率）
- 图B `figB_trackA_scatter.png`：Track A 校正 预测值 vs 真实值（含 y=x 线与 R²/RMSE/MAE）
- 图E `figE_trackA_residuals.png`：Track A 校正残差诊断（残差 vs 时间 / vs 预测值）
- 图F `figF_trackA_error_by_dose.png`：Track A 不同剂量误差对比（原始 vs A1，RMSE/MAE/Bias）
- 图C `figC_trackB_fit.png`：Track B 校准区间拟合（实测±SD vs 区间，剂量×频率）
- 图D `figD_trackB_coverage.png`：Track B 覆盖诊断（总体/按模式/按剂量 + 宽度）
- 表1 `figG_table1.png`：Track A 指标汇总表 + Track B 区间指标