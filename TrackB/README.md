# TrackB — 校准后的预测区间

机理 ensemble (LHS) 给出 `sim_q05/q50/q95`，折 x 频率分层 conformal 校准给出 padding：

```
p_(fold,f) = tolerance_f * q_f,  q_f = Quantile_0.90(内层 cross-fit 分数)
calibrated_lower = clip(sim_q05 - p, 0, 1)
calibrated_upper = clip(sim_q95 + p, 0, 1)
```

- `calibrated_intervals_lowfreq_196.csv`：100/120 Hz 交付版（22 鼠 x 2 频率 x 5 时间点）。
- `calibrated_intervals_allfreq_392.csv`：4 频率全量，用于四条件切片统计。
- **只报 coverage / width / Winkler，不报 R² / RMSE**。
- B1 诊断：约 97% 的最终宽度来自 padding，基础 LHS 区间仅约 3%。
- 已知：逐点覆盖率 != 整曲线联合覆盖；基线(t=0) 与尿素(t=105) 无区间；验证集 22/24 鼠。

## 评价标准下的成绩单（C 类 · 区间指标）

> 完整定义与口径见 [`../docs/EVALUATION_CRITERIA.md`](../docs/EVALUATION_CRITERIA.md) 第 1.3、6 节。
> **TrackB 只报 coverage / width / Winkler，不报 R² / RMSE**（用户既定铁律），
> 也不与 TrackA 的点预测 R² 横向比较。

| 条件 | 点数 / 鼠数 | coverage | mean width | median width | max width | Winkler (α_mis=0.10) |
|---|---|---:|---:|---:|---:|---:|
| C1 全部 | 392 / 22 | 0.908 | 0.5662 | 0.4500 | 1.7663 | 1.0113 |
| C2 低频 | 196 / 22 | 0.913 | 0.3662 | 0.3650 | 0.4459 | 0.4538 |
| C3 高浓度 | 184 / 11 | 0.946 | 0.5491 | 0.4500 | 1.1414 | 0.7445 |
| C4 低频+高浓度 | 92 / 11 | 0.978 | 0.3763 | 0.3767 | 0.4459 | 0.3814 |

| 标准 | 达标情况 |
|---|---|
| C1 coverage（目标 0.90） | 四条件 0.908 / 0.913 / 0.946 / 0.978，均达标 ✓ |
| C2–C4 width | 高频段明显更宽（C1 最大 1.766）；覆盖率是靠宽区间买来的 |
| C5 Winkler | 低频 C2 / C4（0.454 / 0.381）优于含高频条件（1.011 / 0.745） |
| **C6 宽度构成** | **padding 97.09% / 基础 LHS 3.00%** —— 根因是 conformal padding 补偿「仿真 vs 实测」系统性偏差，不是 LHS 参数不确定性 |
| C7 逐点 ≠ 联合 | 逐点覆盖率不等于整条时间曲线同时落入区间的概率 |
| C8 无区间时段 | 基线 t=0 与尿素 t=105 无校准区间，绘图须标注灰虚线 |
| C9 padding 公式 | `p_(fold,f)=tolerance_f × q_f`；`lower/upper = clip(q05∓p, 0, 1)`；P0 审计复现误差 1.39e−16 |

这些是**既有 392 点 LOO 校准区间按条件切片的统计**，本轮**未重新拟合、未重新校准** TrackB。
已知限制：验证集 22/24 鼠（缺 WT5 0.10 M、WT6 0.15 M）；0.05 M 存在 role×剂量混淆；
上游机制选择非严格嵌套；覆盖率最低 3 鼠为 WT17 / WT9 / WT8（按铁律 A8 保留，未删除）。
