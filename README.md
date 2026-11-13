# Chemical Skull Modification & Stimulation Model — 矫正后模型仓库

> 本仓库已按 2026-11 的口径重组：**只保留矫正后的模型**
> `TrackA`（A1 / A2 / A3，点预测）与 `TrackB`（校准预测区间）。
> 原 `TrackA1` 已统一命名为 `TrackA`；**未矫正的纯机理模型、原始 LHS 探索结果、
> 未校准的等效/多频校准结果已从仓库移除**。机理先验 CSV 仅作为 TrackA 的输入数据保留，
> 不再作为独立"模型"呈现。

## 目录

| 路径 | 内容 |
|---|---|
| `TrackA/A1/` | 经验线性校正 `R_Z = a + b*R_Z_raw` |
| `TrackA/A2/` | 协变量扩展 `+ c*t + d*ln(f/100)`（拟合程度与模型效果最平衡） |
| `TrackA/A3/` | 机理融合混合式 `1 - kappa*lambda(f)*delta - eps*t`（基线自动 =1，参数更少） |
| `TrackB/` | 校准后的 LOO 预测区间（392 点全频 + 196 点低频交付版）与切片统计 |
| `four_conditions/` | 四条件拟合脚本、结果、表1 与四张图、`FOUR_CONDITIONS_CONCLUSION.md` |
| `docs/MECHANISM_AND_EMPIRICAL_PRINCIPLE.md` | **机理 + 经验下的机理原理** |
| `data/` | 冻结实测数据、机理先验分位数、三层数据核验报告 |
| `validation/` | 矫正后模型的 LORO / B1 诊断 / 拟合效果留痕 |

## 四条件拟合（本次新增）

| 条件 | 剂量 | 频率 | 节点 |
|---|---|---|---:|
| C1 全部 | 0.05/0.10/0.15/0.20 M | 100/120/1000/10000 Hz | 96 |
| C2 低频 | 4 剂量 | 100/120 Hz | 48 |
| C3 高浓度 | 0.15/0.20 M | 4 频率 | 48 |
| C4 低频 + 高浓度 | 0.15/0.20 M | 100/120 Hz | 24 |

**结论摘要**：条件越窄、指标越好看，是评价范围变化，不是模型变好。
按 LORO 留一大鼠口径，**REF（拉伸指数）四条件全部第一**，但脱离机理、不可外推，只作拟合上限对照；
正式系列里 **A2 与 A3 的 LORO 四条件均值几乎相同（约 0.596），统计不可区分** ——
A2 为默认推荐（样本内拟合略好、协变量可解释），A3 为结构优先备选（参数更少、基线自动 =1）。
详见 `four_conditions/FOUR_CONDITIONS_CONCLUSION.md` 与 `four_conditions/figures/fig4_table1.png`。

## 口径声明（务必阅读）

1. **A1/A2/A3 都是经验或半经验校正，不是机理改进或机理被证实**。
   原始机理链路 `sigma = sigma_floor + c*phi^m`、`R = sum dx/(sigma*A)`、`R_Z_raw = Z/Z0` 从未改动。
2. 验证层级必须区分：`full`（样本内）< `LOCO`（群体条件均值层）< **`LORO` 留一大鼠（个体级独立验证）**。
   LORO 才是验收口径；四个条件的评价集与 LORO 折数不同（C1/C2 为 24 折，C3/C4 为 12 折），**R² 不可横向比大小**。
3. **TrackB 只报 coverage / width / Winkler，不报 R² / RMSE**；逐点覆盖率不等于整条曲线联合覆盖概率。
   区间宽度约 97% 来自 conformal padding，而非 LHS 参数不确定性。
4. **拟合变好 != 模型变好**：条件收窄、增加自由度、补协变量都会抬高 R²。
5. 基线 t=0 按物理定义 R_Z=1；A1/A2 需锚定，A3 由构造自动满足。
6. 数据：修订版《体外EDTA数据记录汇总.xlsx》SHA `4c165298…`，三层核验见 `data/data_verification.json`。
   保留全部 24 鼠，**未删除 WT17 / WT9 / WT8 等不利样本**；14 处 urea_10/Midpoint/WT21–24 转录异常标记未决。

## OPEN ISSUES

- 尿素 t=105 回升：所有单调模型无法表达；机理先验在 95 min 后截断为平台外推。
- 机理先验频率无关，且改性量幅度显著小于实测；真正修复需在机理层重标定 phi(t) 与补频散项（宽频 EIS）。
- `s` 不是被辨识的 CPE 指数（仅 4 个频率点，函数形式不可区分）。
- TrackB：验证集 22/24 鼠（缺 WT5 0.10 M、WT6 0.15 M）；0.05 M 存在 role x 剂量混淆；上游机制选择非严格嵌套。
- 本地 Excel SHA 与上游 manifest 声明的 source_data SHA 不一致，上游原始文件不可得。

## 复现

```bash
pip install -r requirements.txt
python four_conditions/run_four_conditions.py
```

每个 `TrackA/*/model.py` 可独立 import 并调用 `predict(...)`；`TrackB/model.py` 提供
`load_intervals()` / `slice_metrics()` / `calibrated_from_raw()`。
