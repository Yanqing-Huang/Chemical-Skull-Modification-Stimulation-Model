# P1.1 / P1.2 数据差异报告与冻结说明 (Data Difference Report)

## 1. 权威数据源与冻结口径

- **权威分析源 (authoritative)**: `measured_all24_site_level.csv` — 24 大鼠、3 位点、4 频率、7 阶段（含 edta_30/60/90），已逐鼠落盘，无缺失单元。
- **文件 SHA-256**: `d285a75365279076f87f249b3fdea1afbdb5d981bf817f79a96c3e8941a05ace`
- **冻结副本**: `frozen_measured_24rat.csv`（逐字节复制，SHA 一致：`ddc2df7b2159707191c0a1846ad1626c09d0c3dbdd97fb474547bda3a6333171`）
- **Excel 规范长表 (去重后)**: `frozen_excel_canonical.csv`，SHA-256 `bc1022ea528fec04ae1a26ff6943164fa779fc13a4e5219c0a727bcbb59020d8`
- **处理原则**: 不删除任何样本（WT17/WT9/WT8 及 WT21-24 异常均保留）；Excel 与 CSV 冲突处**以 measured CSV 为分析权威**，冲突值记录但不静默覆盖。

## 2. 32 条重复录入（16 组重复键）

- 重复键 = (rat, concentration, frequency_hz, stage, site)。Excel 在 `urea_10 / Midpoint / WT21-24` 每个 (大鼠×频率) 录入了**两次**，共 16 组 × 2 = **32 行重复**。
- 性质判定：**录入重复**（同一格点被录两遍），非原始重复（原始实验并无双份测量）。
- 去重规则：每组保留**与 measured CSV 一致**的那个值；冲突值写入下表 `duplicate_excel_values`，保留审计痕迹。

| rat | site | freq | stage | concentration | 重复Excel值 | 保留值 | measured CSV值 |
|---|---|---|---|---|---|---|---|
| WT21 | Midpoint | 100 | urea_10 | 0.2 | [11.0, 14.0] | 11.0 | 51.0 |
| WT21 | Midpoint | 120 | urea_10 | 0.2 | [10.0, 12.0] | 10.0 | 51.0 |
| WT21 | Midpoint | 1000 | urea_10 | 0.2 | [3.0, 5.0] | 3.0 | 51.0 |
| WT21 | Midpoint | 10000 | urea_10 | 0.2 | [0.003, 0.004] | 0.003 | 51.0 |
| WT22 | Midpoint | 100 | urea_10 | 0.2 | [6.0, 9.0] | 6.0 | 51.0 |
| WT22 | Midpoint | 120 | urea_10 | 0.2 | [6.0, 7.0] | 6.0 | 51.0 |
| WT22 | Midpoint | 1000 | urea_10 | 0.2 | [3.0, 3.0] | 3.0 | 51.0 |
| WT22 | Midpoint | 10000 | urea_10 | 0.2 | [0.002, 0.003] | 0.002 | 51.0 |
| WT23 | Midpoint | 100 | urea_10 | 0.2 | [6.0, 9.0] | 6.0 | 51.0 |
| WT23 | Midpoint | 120 | urea_10 | 0.2 | [6.0, 8.0] | 6.0 | 51.0 |
| WT23 | Midpoint | 1000 | urea_10 | 0.2 | [2.0, 3.0] | 2.0 | 51.0 |
| WT23 | Midpoint | 10000 | urea_10 | 0.2 | [0.002, 0.003] | 0.002 | 51.0 |
| WT24 | Midpoint | 100 | urea_10 | 0.2 | [4.0, 7.0] | 4.0 | 51.0 |
| WT24 | Midpoint | 120 | urea_10 | 0.2 | [4.0, 7.0] | 4.0 | 51.0 |
| WT24 | Midpoint | 1000 | urea_10 | 0.2 | [2.0, 3.0] | 2.0 | 51.0 |
| WT24 | Midpoint | 10000 | urea_10 | 0.2 | [0.002, 0.002] | 0.002 | 51.0 |

## 3. 14 处 Excel↔CSV 值不一致

- 在重叠阶段 (baseline/acetic_acid/edta_15/urea_10) 的 1150 个比对单元中，1136 个一致（容差 1e-06），**14 个不一致**。
- 全部 14 处集中在 **urea_10 / Midpoint / WT21-24 / 4 频率**——即第 2 节重复录入的组，且 Excel 两遍录入的值彼此也不同（如 WT21/100Hz 录 11 与 14，CSV 取 14）。
- 性质判定：**局部转录异常**（transcription anomaly），非系统性偏移。上游原始文件 `raw_file_in_repository=false` 不可得，**无法回查原始实验记录**，故标记为 `UNRESOLVED_TRANSCRIPTION` 并保留，不以任何一侧覆盖另一侧用于分析口径。

| rat | site | freq | stage | Excel值 | measured CSV值 | |Δ| |
|---|---|---|---|---|---|---|
| WT21 | Midpoint | 100 | urea_10 | 11.0 | 14.0 | 3.0 |
| WT21 | Midpoint | 120 | urea_10 | 10.0 | 12.0 | 2.0 |
| WT21 | Midpoint | 1000 | urea_10 | 3.0 | 5.0 | 2.0 |
| WT21 | Midpoint | 10000 | urea_10 | 0.003 | 0.004 | 0.001 |
| WT22 | Midpoint | 100 | urea_10 | 9.0 | 6.0 | 3.0 |
| WT22 | Midpoint | 120 | urea_10 | 7.0 | 6.0 | 1.0 |
| WT22 | Midpoint | 10000 | urea_10 | 0.003 | 0.002 | 0.001 |
| WT23 | Midpoint | 100 | urea_10 | 6.0 | 9.0 | 3.0 |
| WT23 | Midpoint | 120 | urea_10 | 6.0 | 8.0 | 2.0 |
| WT23 | Midpoint | 1000 | urea_10 | 2.0 | 3.0 | 1.0 |
| WT23 | Midpoint | 10000 | urea_10 | 0.002 | 0.003 | 0.001 |
| WT24 | Midpoint | 100 | urea_10 | 7.0 | 4.0 | 3.0 |
| WT24 | Midpoint | 120 | urea_10 | 7.0 | 4.0 | 3.0 |
| WT24 | Midpoint | 1000 | urea_10 | 3.0 | 2.0 | 1.0 |

## 4. OPEN ISSUE · 文件级 SHA 不一致（延续自前序核验）

- 本地 Excel 文件 SHA `a6add762…` 与上游 manifest 声明源数据 SHA `486ec7e9…` **不一致**；上游原始文件不可得。仅能核验「本地 Excel ↔ 本地 measured CSV」内容一致性，不声称两份原始文件相同。
- **新增验证单位问题（见 A1 / B1 任务）**：`measured_all24_condition_summary.csv`（TrackA 点指标来源）的 `n_rats=6`，即 A1 拟合的「48 节点」实为 **6 鼠条件均值**；TrackB 区间验证集仅含 **22/24 鼠**（缺 WT5、WT6）。两者均需在进入 A2/B2 前澄清。

## 5. 结论与通过条件

- 数据来源、单位、异常记录处理规则已有据可查；冻结副本可经本脚本重新生成（确定性）。
- 14 处不一致与 32 条重复均被记录并保留；分析以 24 鼠 measured CSV 为权威。
- **未了结项**：① 上游原始文件 SHA 不一致（无法闭环）；② 6 鼠子集成员待原作者提供；③ TrackB 缺失 WT5/WT6 原因待澄清。在 P1 阶段仅记录，不修改模型。