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
