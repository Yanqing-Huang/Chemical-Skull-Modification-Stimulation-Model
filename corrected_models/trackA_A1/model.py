#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
Track A — 经验校正模型 A1（矫正后的点预测模型）

模型定义
--------
    R_Z_corr = a + b * R_Z_raw

其中 R_Z_raw 是原始机理先验（绝对孔隙率 + 空间串联积分）给出的
whole_resistance_ratio 的 q50，R_Z = Z / Z_0 为阻抗比值。

    a = -4.471497120097876
    b =  5.680585691167204

拟合口径
--------
在低 100/120 Hz、48 个「剂量 x 频率 x 时间」条件均值节点上（排除基线），
对 (R_Z_raw, 实测条件均值) 做最小二乘拟合得到 a, b。

重要声明（务必阅读）
--------------------
1. A1 是 **经验线性校正（empirical post-hoc calibration）**，不是机理改进：
   它只修正幅度偏差，未改动原始机理公式
       sigma = 1e-4 + 1.5 * phi^m ,  R = sum_i dx_i / (sigma_i * A_i)
   因此不能据此声称机理参数被辨识或物理机制被证伪/证实。
2. 验证层级必须区分，不能混为一谈：
     - 群体条件均值层 (full OLS, n=48):  R2=0.782, RMSE=0.083
     - 群体条件均值 LOCO (留一条件, n=48): R2=0.762, RMSE=0.087
     - 个体级 LORO (留一大鼠, n=288):     R2=0.588, RMSE=0.130
       （逐鼠 R2 均值 0.404；WT17、WT2 逐鼠 R2 为负）
   0.762 与 0.588 预测层级不同，不能写成「性能从 0.762 下降到 0.588」。
3. 原始机理模型 A0（未校正）在相同 48 节点上 R2=-7.906、Bias=-0.510，
   说明原始模型有趋势解释力（Pearson r=0.885）但存在显著幅度偏差。
4. 基线 (t=0) 按物理定义 R_Z=1.0；A1 线性式在 t=0 外推值会偏离，
   绘图/使用时对基线做物理锚定（见 predict 的 anchor_baseline 参数）。
5. 本模块只依赖 numpy / pandas 与本包 data/ 内的机理先验 CSV，可独立运行。
"""
from __future__ import annotations
import json
import os
import numpy as np
import pandas as pd

# ---- 校正参数（全数据 OLS；同时保留 LOCO/LORO 复核值供对照） ----
A = -4.471497120097876
B = 5.680585691167204

A_LOCO, B_LOCO = -4.471875802594746, 5.680997572464842   # 留一条件均值
A_LORO, B_LORO = -4.473343733982449, 5.682741880925119   # 留一大鼠均值

_HERE = os.path.dirname(os.path.abspath(__file__))
_PRIOR_CSV = os.path.join(_HERE, "..", "data", "mechanism_timeseries_quantiles.csv")

# 时间点（分钟）-> stage 名，与冻结数据一致
STAGE_BY_TIME = {0: "baseline", 5: "acetic_acid", 20: "edta_15", 35: "edta_30",
                 65: "edta_60", 95: "edta_90", 105: "urea_10"}


def _load_prior():
    q = pd.read_csv(_PRIOR_CSV)
    w = (q[q.metric == "whole_resistance_ratio"]
         .drop_duplicates(["edta_concentration_mol_per_l", "global_time_s"])
         .sort_values(["edta_concentration_mol_per_l", "global_time_s"]))
    return w


_PRIOR = None


def raw_prior(dose_mol_per_l: float, t_min: float, stat: str = "q50") -> float:
    """原始机理先验 whole_resistance_ratio 在给定剂量/时间的插值（未校正）。"""
    global _PRIOR
    if _PRIOR is None:
        _PRIOR = _load_prior()
    g = _PRIOR[_PRIOR.edta_concentration_mol_per_l == dose_mol_per_l]
    ts = g.global_time_s.to_numpy(float)
    vs = g[stat].to_numpy(float)
    if len(ts) == 0:
        raise ValueError(f"no prior for dose {dose_mol_per_l}")
    t_s = t_min * 60.0
    if t_s > ts.max():          # 机理先验在 edta_90(95min) 后截断，尿素点取末值（外推）
        return float(vs[-1])
    return float(np.interp(t_s, ts, vs))


def predict(dose_mol_per_l: float, t_min: float,
            a: float = A, b: float = B, anchor_baseline: bool = True) -> float:
    """
    Track A A1 校正点预测 R_Z。

    Parameters
    ----------
    dose_mol_per_l : EDTA 浓度 (0.05 / 0.10 / 0.15 / 0.20)
    t_min          : 全局时间（分钟），取 {0,5,20,35,65,95,105}
    a, b           : 校正参数（默认全数据 OLS）
    anchor_baseline: t=0 时按物理定义返回 1.0（推荐 True）
    """
    if anchor_baseline and t_min == 0:
        return 1.0
    return a + b * raw_prior(dose_mol_per_l, t_min)


def predict_from_raw(rz_raw: float, a: float = A, b: float = B) -> float:
    """对已算好的原始机理先验值做 A1 校正：R_Z_corr = a + b * R_Z_raw。"""
    return a + b * rz_raw


def info() -> dict:
    return {
        "definition": "R_Z_corr = a + b * R_Z_raw (empirical linear calibration, NOT mechanistic)",
        "a": A, "b": B,
        "baseline_anchor": 1.0,
        "metrics": {
            "group_full_OLS_n48": {"r2": 0.782, "rmse": 0.083, "mae": 0.069, "bias": 0.000},
            "group_LOCO_n48": {"r2": 0.762, "rmse": 0.087},
            "individual_LORO_n288": {"r2": 0.588, "rmse": 0.130, "per_rat_mean_r2": 0.404},
            "A0_raw_n48": {"r2": -7.906, "rmse": 0.533, "bias": -0.510},
        },
        "caveats": [
            "Empirical correction only; mechanistic parameters unchanged.",
            "Low frequencies 100/120 Hz only; high-freq 1000/10000 Hz excluded.",
            "Validation levels (group LOCO vs individual LORO) are not directly comparable.",
        ],
    }


if __name__ == "__main__":
    print(json.dumps(info(), ensure_ascii=False, indent=2))
    print("\nexample predictions (dose=0.05 M):")
    for t in [0, 5, 20, 35, 65, 95, 105]:
        print(f"  t={t:3d} min  raw={raw_prior(0.05, t):.3f}  A1={predict(0.05, t):.3f}")
