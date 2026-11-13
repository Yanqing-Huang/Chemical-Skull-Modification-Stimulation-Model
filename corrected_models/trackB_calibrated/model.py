#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
Track B — 校准后的嵌套 LOO 预测区间模型（矫正后的区间预测模型）

模型链路（保留原框架，未推倒重建）
----------------------------------
1. LHS 参数采样 -> 机理 ensemble，逐点得到 sim_q05 / sim_q50 / sim_q95。
2. 兼容性筛选 (compatible member > 0) 或最近邻 ensemble 兜底
   (compatible_count = 0 时固定 256 成员)。
3. 折 x 频率分层 conformal 校准：
       s_j = max(0, q05_j - y_j, y_j - q95_j) / tolerance_f
       q_f = Quantile_0.90(内层 cross-fit 分数 s)          # 目标覆盖 90%
       p_(fold,f) = tolerance_f * q_f                        # padding
       calibrated_lower = clip(sim_q05 - p, 0, 1)
       calibrated_upper = clip(sim_q95 + p, 0, 1)
   嵌套 LOO：22 个外层留一大鼠折 x 5 个内层 grouped 折。
4. 外层留出大鼠不参与 ensemble 生成以外的校准拟合（见下方「已知治理问题」）。

关键结论（B1 区间误差来源诊断）
-------------------------------
- 最终区间宽度约 **97% 来自 conformal padding**，基础 LHS 参数不确定仅约 3%。
  因此「区间过宽」根因是校准层在补偿「仿真 vs 实测」的系统性 + 个体差异，
  **不是**参数采样范围不足；不应盲目扩大 LHS 采样范围。
- 合并逐点覆盖率 0.913 掩盖了两种 ensemble 模式的分裂：
      TRAINING_COMPATIBLE_ENSEMBLE 覆盖率 0.857（欠覆盖）
      NEAREST_TRAINING_ENSEMBLE      覆盖率 0.969（过补偿）
- 0.05 M 剂量只出现在 TRAINING_COMPATIBLE 模式，故「0.05 M 欠覆盖」是
  **模式混淆**而非单纯剂量效应；B2 校准对照必须控制 ensemble_role。
- 验证集仅 22/24 鼠（缺 WT5 0.10M、WT6 0.15M）；逐点覆盖率不等于
  整条时间曲线同时落入区间的概率。

重要声明
--------
1. Track B **不使用 R2 / RMSE** 评价；只报告覆盖率、区间宽度、Winkler 分数。
   其覆盖率不可与 Track A 的 R2 放在同一张表混比。
2. 逐点覆盖率是各时间点观测落入其区间的比例，非整曲线联合覆盖概率。
3. 已知治理问题（未解决，勿隐去）：
   - 上游 manifest 声明 mechanism_id=157 / two_sided_50um 是**全数据诊断选定**，
     而非在每个外层训练折内重新选择 -> 存在潜在乐观偏差（待 B3 严格折内验证）。
   - 本地 Excel 文件 SHA 与上游 manifest 声明的 source_data SHA 不一致（原始文件不可得）。
   - 机制选择流程非严格嵌套（见上）。
4. calibrated_intervals.csv 是**已校准后的最终区间结果**（100/120 Hz，
   n=196 点 = 22 鼠 x 2 频率 x 5 时间 [5,20,35,65,95] 分钟，不含基线/尿素）。
   基线(t=0) 与尿素(t=105) 无校准区间。

本模块提供：加载已校准区间、按条件查询 [LB,UB]、以及从原始 q05/q95 + padding
重建区间的纯函数（自包含，不依赖上游代码）。
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

TARGET_COVERAGE = 0.90
ALPHA_MIS = 0.10
WINKLER_ALPHA = 0.10

_HERE = os.path.dirname(os.path.abspath(__file__))
_INTERVAL_CSV = os.path.join(_HERE, "calibrated_intervals.csv")

_DF = None


def load_intervals() -> pd.DataFrame:
    """加载已校准的最终预测区间表（100/120 Hz，逐点）。"""
    global _DF
    if _DF is None:
        _DF = pd.read_csv(_INTERVAL_CSV)
    return _DF.copy()


def predict_interval(rat: str, dose_mol_per_l: float, frequency_hz: int,
                     t_min: float) -> dict | None:
    """
    查询某（大鼠/剂量/频率/时间）点的校准后预测区间 [LB, UB] 与区间中点。
    若无该点（如基线/尿素，或缺失大鼠）返回 None。
    """
    df = load_intervals()
    sel = df[(df.rat == rat)
             & (np.isclose(df.edta_concentration_mol_per_l, dose_mol_per_l))
             & (np.isclose(df.frequency_hz, frequency_hz))
             & (np.isclose(df.global_time_min, t_min))]
    if sel.empty:
        return None
    r = sel.iloc[0]
    return {
        "rat": rat, "dose": dose_mol_per_l, "freq_hz": frequency_hz, "t_min": t_min,
        "LB": float(r.LB), "UB": float(r.UB),
        "center": float(r.trackB_center),
        "observed_rz": float(r.observed_rz),
        "covered": bool(r.covered),
        "width": float(r.width),
        "ensemble_role": str(r.ensemble_role),
        "winkler": float(r["winkler_alpha_mis_0.10"]),
    }


def apply_padding(sim_q05: float, sim_q95: float, padding: float) -> tuple[float, float]:
    """从原始 LHS 分位数 + padding 重建校准区间（双侧加 padding，clip 到 [0,1]）。"""
    return (float(np.clip(sim_q05 - padding, 0, 1)),
            float(np.clip(sim_q95 + padding, 0, 1)))


def conformal_padding(tolerance_f: float, q_f: float) -> float:
    """p_(fold,f) = tolerance_f * q_f，q_f 为内层 cross-fit 分数的 0.90 分位数。"""
    return tolerance_f * q_f


def winkler(y: float, lower: float, upper: float, alpha: float = WINKLER_ALPHA) -> float:
    """Winkler / interval score（alpha_mis=0.10 口径）。"""
    w = upper - lower
    return (w
            + (2.0 / alpha) * (lower - y) * (y < lower)
            + (2.0 / alpha) * (y - upper) * (y > upper))


def summary() -> dict:
    df = load_intervals()
    by_role = (df.groupby("ensemble_role")
               .agg(coverage=("covered", "mean"), n=("covered", "size"),
                    width=("width", "mean")).to_dict("index"))
    return {
        "target_coverage": TARGET_COVERAGE,
        "point_coverage": float(df.covered.mean()),
        "mean_width": float(df.width.mean()),
        "mean_winkler": float(df["winkler_alpha_mis_0.10"].mean()),
        "n_points": int(len(df)),
        "n_rats": int(df.rat.nunique()),
        "n_freq": int(df.frequency_hz.nunique()),
        "time_points_min": sorted(int(x) for x in df.global_time_min.unique()),
        "by_ensemble_role": by_role,
        "padding_share_of_width": "~97% (B1 diagnostics); base LHS ~3%",
        "caveats": [
            "point coverage != whole-curve joint coverage",
            "validation set 22/24 rats (missing WT5, WT6)",
            "0.05 M confounded with TRAINING_COMPATIBLE mode",
            "mechanism selection not strictly nested (B3 pending)",
            "Track B never reports R2/RMSE",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(summary(), ensure_ascii=False, indent=2, default=float))
    print("\nexample (WT1, 0.05 M, 100 Hz, t=20min):")
    print(" ", predict_interval("WT1", 0.05, 100, 20))
