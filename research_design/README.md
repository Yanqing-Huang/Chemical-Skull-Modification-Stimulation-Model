# 顶层研究设计 · TrackA / TrackB × 三路线 × 两对比维度

**本目录是把整个项目纳入一个统一框架的整合层**，独立新建，不改动任何已冻结并推送的成果包。

## 框架一句话

```
预测对象：TrackA 点值 R_Z（等价阻抗降低值 Δ=1−R_Z） / TrackB 区间 [lo, hi]
TrackA 三路线：A1 纯经验 ─ A2 纯机理 ─ A3 机理驱动经验（三者对比）
对比维度①：全局拟合 vs 分段拟合（分段独立 SEG / 分段增量 ADJ）
对比维度②：所有条件 C1 vs 最佳条件范围 C4（0.15/0.20 M × 100/120 Hz）
唯一验收口径：LORO（Leave-One-Rat-Out）；TrackB 用 coverage/width/Winkler
```

## 文档

| 文件 | 内容 |
|---|---|
| `RESEARCH_DESIGN.md` | **顶层设计总纲**（唯一权威）：对象二分、三路线设计论证、两对比维度、验收口径、边界与空格 |
| `COMPARISON_MATRIX.md` | 全维度对比数值展开（三路线 / 全局vs分段 / 条件收窄 / TrackB / 尿素回升 / 不可横比清单） |
| `TABLES_INDEX.md` | `tables/` 全部 CSV 的列说明与来源 |
| `README.md` | 本文件 |

## 图（7 张，全部 CJK 无缺字、纵轴自 0 起、无遮挡）

| 文件 | 内容 |
|---|---|
| `figures/fig01_research_design_framework.png` | 顶层框架总览：预测对象 → 三路线 → 两对比维度 → 验收口径 |
| `figures/fig02_matrix_fitmode_x_scope.png` | 3×2 全维度矩阵（拟合方式 × 条件范围 × 三路线），空格如实标注 |
| `figures/fig03_three_routes.png` | 三路线四条件 LORO R²；A3 的 full（仅参考）vs LORO（验收） |
| `figures/fig04_global_vs_segmented.png` | 全局 vs 分段：7 方案 LORO R² + 配对 ΔR² 与 bootstrap 95% CI |
| `figures/fig05_all_vs_best_condition.png` | 条件收窄路径 C1→C2→C3→C4：三路线 / full vs LORO / TrackB |
| `figures/fig06_trackB_rebound_padding.png` | 尿素回升 16 节点一致率 / 末两点 R_Z / padding 宽度构成 |
| `figures/fig07_segmented_curves_measured.png` | 7 时间点实测点 + SEM 全画，全局 A3 与分段 ADJ_A3 曲线 |

## 关键结论

- 三路线：**A3 > A2 ≈ A1.1 > A1.2 ≈ A1.3**（四条件一致）；C1 LORO A3 = 0.5201。
- 全局 vs 分段：只有 `ADJ_A3` 不劣于全局 A3（C1 0.5226，Δ=+0.0025，CI 跨 0）；其余分段方案显著变差。分段真正修好的是**尿素回升 16/16 符号一致**。
- 所有条件 vs 最佳范围：A3 LORO 0.5201 → 0.7684，但 C4 是 C1 子集、评价集不同，**不可横比**。
- TrackB：coverage 90.82% → 97.83% 均达标；宽度 97.09% 来自 conformal padding。
- 框架尚有 **8 个空格子**（A2 分段、TrackB 分段），如实标注未填充。

## 硬边界

- 跨包一致性校验 60 项，max\|Δ\| = 4.800e-07，**模型/参数/数据零改动、零重拟合**。
- 24 鼠全保留；LORO 唯一验收；四条件 R² 不可横比；TrackB 不报 R²/RMSE 作验收。
- 本包已作为 `research_design/` 合入 GitHub `main`；`build_master.py` 与 `make_figures.py`
  的路径已改为仓库内相对定位（读 `four_conditions/` 与 `segmented_fit/tables/`），clone 后可直接复现。

## 复现

```bash
# 在仓库根目录 clone 后，进入 research_design/ 执行；数据源为同仓库的
# four_conditions/ 与 segmented_fit/tables/，无需任何外部绝对路径
cd research_design
python3.11 build_master.py
python3.11 make_docs.py
python3.11 make_figures.py
```

