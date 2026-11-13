# TrackA2 — 纯机理模型路线（本轮新建）

**调机理、把频率等等各种都算进去、把所有条件都考虑进去。**

```
Archie 电导     sigma = sigma_floor + c*phi^m
脱钙孔隙率演化   phi(t,d) ~ (d/D0)^beta * (t/T0)^p
界面极化 / CPE   G_pol(f) ~ (f/F0)^(-s)，与离子通路并联
总导纳           Y(f,t,d) = G_ion(t,d) + G_pol(f)
R_Z             = Y(0)/Y(t) = 1/(1+delta_eff)
delta_eff       = a0*(d/D0)^beta*(t/T0)^p*(f/F0)^(-s)
```

- 4 个参数全是物理量：`a0` 改性幅度、`beta` 剂量指数、`p` 脱钙时间指数、`s` 界面极化频散指数。
  `D0=0.10 mol/L, T0=100 min, F0=100 Hz`。
- **频率 f 内生进机理方程**，可解释实测「同一时间高频 R_Z 偏高」；t=0 天然=1；值域 (0,1]。
- **零经验回归系数**；单一组参数覆盖 4 剂量 x 4 频率 x 6 时间 = 96 条件节点，
  跨 C2/C3/C4 零重拟合迁移 R2 仍 >0.91（见 `../four_conditions/THREE_ROUTES_CONCLUSION.md` 第 3 节）。
- `model.py` **完全自包含**，不读任何外部数据文件。

诚实边界：物理量由实测标定，不是纯先验推导；`s` 是等效频散指数不是被辨识的 CPE 指数；
delta_eff 单调，无法表达尿素 t=105 回升。详见 `../docs/MECHANISM_PRINCIPLE.md`。
