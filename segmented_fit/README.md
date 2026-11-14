# 按试剂处理分段拟合（乙酸 / EDTA / 饱和尿素）

> 本轮为**新的分析分支**，新建独立目录；**不修改**已冻结并推送的 `Chemical_Skull_Model_ThreeRoutes` 包内任何模型公式、参数或数据。

## 一句话结论

分段拟合在**群体均值层**和**机理性状**上确实更好（full R² 由 0.920 提升到 0.935–0.957，尿素段回升由「完全无法表达」变为 16/16 节点符号一致）；但在项目既定的**个体级 LORO 唯一验收口径**下，没有任何分段方案取得统计显著的提升——三段独立标定反而显著变差，骨架保留的最少调整（ADJ_A3）只是**不劣于**全局 A3（ΔR²=+0.0025，95%CI 跨 0）。

- 冻结基线 C1 LORO R²：A1.1 0.4806 / A2 0.4865 / **A3 0.5201**
- 分段 C1 LORO R²：SEG_A1 0.4963 / SEG_A3 0.4944 / ADJ_A1 0.4827 / **ADJ_A3 0.5226**
- 群体均值层 full R²：A3 0.9205 → SEG_A1 0.9573 / ADJ_A3 0.9353
- 尿素回升：实测 +0.060；5 个全局单调模型全部预测继续下降（0/16 节点符号一致）；全部 4 个分段方案 16/16 一致

## 目录

| 路径 | 内容 |
|---|---|
| `segmented_models.py` | 三段模型定义（含冻结基线逐字复刻、V1 独立分段、V2 增量调整、机理再封闭项） |
| `segmented_fit.py` | 数据 / LORO 引擎 / 变体择优 / 全指标 / 配对 bootstrap / 落盘 |
| `metrics_lib.py` | 指标函数，逐字复制自上一轮 `compute_all_metrics.py` 第 192–373 行，口径严格一致 |
| `make_figures.py` | 8 张图 |
| `make_docs.py` | 本文档体系 |
| `data/` | 冻结 24 鼠实测 CSV + 机理先验（SHA 与 ThreeRoutes 包内副本一致） |
| `tables/` | 全部实算表（12 张 CSV + 汇总 JSON + 冻结基线复制件） |
| `figures/` | 8 张诊断图 |
| `SEGMENTED_MODELS.md` | 三段定义、两代公式全表、参数与机理原理 |
| `SEGMENTED_CONCLUSION.md` | **结论主文档**（含全部数值表、两种口径分离、推荐、诚实边界） |
| `TABLES_INDEX.md` | 每张 CSV 的字段说明 |

## 复现

```bash
cd segmented_fit
python3.11 segmented_fit.py     # 主计算：基线校验 + 变体择优 + 四条件 9 方案 LORO + 全指标
python3.11 make_figures.py      # 8 张图
python3.11 make_docs.py         # 文档（从 tables/ 自动渲染）
```

依赖：Python 3.11 / numpy / pandas / scipy / matplotlib，CJK 字体 `NotoSansCJK-Regular.ttc`。

## 口径铁律（继承既有评价体系，未放宽）

1. **LORO（Leave-One-Rat-Out，24 / 12 折）是唯一验收口径**；full 为群体均值层 in-sample，只作参考，**不可与个体验证混淆**。
2. 四条件评价集与折数不同，**R² 不可横向解读为模型优劣**。
3. 分段模型参数更多，必须同时报 k / AICc / BIC / 按鼠聚类配对 bootstrap 的 ΔR² 95%CI。
4. 24 鼠全保留，未删除任何不利样本；基线 t=0 按物理定义锚定 R_Z=1。
5. 拟合变好 ≠ 机理更正确；指标最优 ≠ 可用模型；ρ 是等效再封闭率，非被辨识的物理常数。
6. TrackB 不在本轮范围内，其验收只报 coverage / width / Winkler，不报 R² / RMSE。

基线一致性校验：20 项复算 vs 冻结值 `max|ΔR²| = 4.758e-07`（PASS），证明对照口径零漂移。

## 交付范围

本包已作为 `segmented_fit/` 合入 GitHub `main`（见仓库根 `README.md` 与 `docs/PUSH_RECEIPT.md`）。本目录是自包含的分析分支，**未修改**仓库中三路线包的任何模型公式、参数或数据。
