#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
统一研究设计框架 · 整合脚本（零重拟合 / 零改模型 / 零改参数 / 零改数据）

把三套已冻结成果收进同一个框架（仓库内路径，clone 后可直接运行）：
  1) ThreeRoutes/three_routes_metrics.csv         全局 TrackA1/A2/A3 四条件
  2) ThreeRoutes/.../ALL_METRICS_FULL.csv         全局三验证层级全指标
  3) ThreeRoutes/.../TRACKB_FULL_METRICS.csv      TrackB 四条件区间指标
  4) SegmentedFit/tables/COMPARISON_OVERALL.csv   分段四条件 9 方案 LORO
  5) SegmentedFit/tables/FULL_GROUP_MEAN.csv      分段群体均值层
  6) SegmentedFit/tables/SEGMENT_METRICS.csv      分段逐段指标
  7) SegmentedFit/tables/UREA_REBOUND_SUMMARY.csv 尿素回升证据

输出：MASTER_COMPARISON.csv + 分组对比表 + CONDITION_SCOPE.csv
     + FRAMEWORK_GAPS.csv + CONSISTENCY_CHECK.csv + SUMMARY.json
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
TR = ROOT.parent / "four_conditions"      # 三路线冻结指标表（仓库内）
SG = ROOT.parent / "segmented_fit"        # 分段拟合包（仓库内）
TAB = ROOT / "tables"
TAB.mkdir(parents=True, exist_ok=True)
FC = TR                                   # 全指标表与三路线指标同目录

# ---------------------------------------------------------------- 定义
SCOPE = {
    "C1_all":      "所有条件",
    "C2_lowfreq":  "低频切片",
    "C3_highdose": "高浓度切片",
    "C4_lf_hd":    "最佳条件范围",
}
SCOPE_SHORT = {"C1_all": "C1 所有条件", "C2_lowfreq": "C2 低频",
               "C3_highdose": "C3 高浓度", "C4_lf_hd": "C4 最佳范围"}
COND_SPEC = {
    "C1_all":      dict(doses="0.05 / 0.10 / 0.15 / 0.20", freqs="100 / 120 / 1000 / 10000",
                        nodes=96, rat_points=576, rats=24, folds=24),
    "C2_lowfreq":  dict(doses="0.05 / 0.10 / 0.15 / 0.20", freqs="100 / 120",
                        nodes=48, rat_points=288, rats=24, folds=24),
    "C3_highdose": dict(doses="0.15 / 0.20", freqs="100 / 120 / 1000 / 10000",
                        nodes=48, rat_points=288, rats=12, folds=12),
    "C4_lf_hd":    dict(doses="0.15 / 0.20", freqs="100 / 120",
                        nodes=24, rat_points=144, rats=12, folds=12),
}

ROUTE_OF = {
    "A1.1": ("TrackA1", "纯经验"), "A1.2": ("TrackA1", "纯经验"), "A1.3": ("TrackA1", "纯经验"),
    "A2":   ("TrackA2", "纯机理"), "A3": ("TrackA3", "机理驱动经验"),
}
SCHEME_OF = {
    "BASE_A1.1": ("全局", "TrackA1", "纯经验", "A1.1 拉伸指数"),
    "BASE_A1.2": ("全局", "TrackA1", "纯经验", "A1.2 A+ 协变量"),
    "BASE_A1.3": ("全局", "TrackA1", "纯经验", "A1.3 H1 混合"),
    "BASE_A2":   ("全局", "TrackA2", "纯机理", "A2 并联导纳纯机理"),
    "BASE_A3":   ("全局", "TrackA3", "机理驱动经验", "A3 机理骨架 + 保形校准"),
    "SEG_A1":    ("分段独立", "TrackA1", "纯经验", "SEG_A1 三段独立经验择优"),
    "SEG_A3":    ("分段独立", "TrackA3", "机理驱动经验", "SEG_A3 三段独立机理+经验"),
    "ADJ_A1":    ("分段增量", "TrackA1", "纯经验", "ADJ_A1 骨架保留 + 经验增量"),
    "ADJ_A3":    ("分段增量", "TrackA3", "机理驱动经验", "ADJ_A3 骨架保留 + 机理增量"),
}
BASE_MAP = {"BASE_A1.1": "A1.1", "BASE_A1.2": "A1.2", "BASE_A1.3": "A1.3",
            "BASE_A2": "A2", "BASE_A3": "A3"}


def esc(v):
    return "" if v is None or (isinstance(v, float) and np.isnan(v)) else v


# ---------------------------------------------------------------- 读取
three = pd.read_csv(TR / "three_routes_metrics.csv")
allmet = pd.read_csv(FC / "ALL_METRICS_FULL.csv")
tbfull = pd.read_csv(FC / "TRACKB_FULL_METRICS.csv")
tb4 = pd.read_csv(TR / "trackB_four_conditions.csv").rename(
    columns={"winkler_alpha_mis_0.10": "trackb_winkler"})
segov = pd.read_csv(SG / "tables" / "COMPARISON_OVERALL.csv")
segfull = pd.read_csv(SG / "tables" / "FULL_GROUP_MEAN.csv")
segseg = pd.read_csv(SG / "tables" / "SEGMENT_METRICS.csv")
urea = pd.read_csv(SG / "tables" / "UREA_REBOUND_SUMMARY.csv")
variants = pd.read_csv(SG / "tables" / "A1_VARIANT_SELECTION.csv")

# ---------------------------------------------------------------- 跨包一致性校验
chk = []
for r in segov.itertuples():
    if r.scheme not in BASE_MAP:
        continue
    ref = three[(three.condition == r.condition) & (three.model == BASE_MAP[r.scheme])]
    if ref.empty:
        continue
    ref = ref.iloc[0]
    chk.append(dict(condition=r.condition, scheme=r.scheme, model=BASE_MAP[r.scheme],
                    metric="LORO_R2", segmented=r.r2, frozen=float(ref.loro_r2),
                    abs_delta=abs(float(r.r2) - float(ref.loro_r2))))
    chk.append(dict(condition=r.condition, scheme=r.scheme, model=BASE_MAP[r.scheme],
                    metric="LORO_RMSE", segmented=r.rmse, frozen=float(ref.loro_rmse),
                    abs_delta=abs(float(r.rmse) - float(ref.loro_rmse))))
for r in segfull.itertuples():
    if r.scheme not in BASE_MAP:
        continue
    ref = three[(three.condition == r.condition) & (three.model == BASE_MAP[r.scheme])]
    if ref.empty:
        continue
    chk.append(dict(condition=r.condition, scheme=r.scheme, model=BASE_MAP[r.scheme],
                    metric="FULL_R2", segmented=r.r2, frozen=float(ref.iloc[0].full_r2),
                    abs_delta=abs(float(r.r2) - float(ref.iloc[0].full_r2))))
chk = pd.DataFrame(chk)
chk.to_csv(TAB / "CONSISTENCY_CHECK.csv", index=False, encoding="utf-8-sig", float_format="%.10g")
max_delta = float(chk.abs_delta.max())
consistency_ok = bool(max_delta < 1e-6)

# ---------------------------------------------------------------- MASTER 主表
CORE = ["r2", "adj_r2", "rmse", "mae", "bias", "aicc", "bic",
        "r2_boot_lo", "r2_boot_hi", "delta_r2_vs_best", "delta_r2_boot_lo", "delta_r2_boot_hi"]
rows = []

# (1) 全局 TrackA：三验证层级（LORO 为验收层），来自全指标表
for r in allmet.itertuples():
    route, fam = ROUTE_OF[r.model]
    rows.append(dict(
        track="TrackA", prediction_target="点值 R_Z（等价阻抗降低量 1-R_Z）",
        fit_mode="全局", route=route, model_family=fam, model=r.model,
        level=r.level, condition=r.condition, condition_scope=SCOPE[r.condition],
        k=r.k, n=r.n, n_rats=COND_SPEC[r.condition]["rats"],
        n_folds=COND_SPEC[r.condition]["folds"],
        **{c: esc(getattr(r, c, np.nan)) for c in CORE},
        coverage=np.nan, mean_width=np.nan, winkler_mean=np.nan,
        source="ThreeRoutes/ALL_METRICS_FULL.csv",
        note="验收层=LORO；full/LOCO 仅参考" if r.level != "LORO" else "★ 唯一验收口径"))

# (2) 分段 TrackA：LORO 验收层 + 群体均值层
for r in segov.itertuples():
    mode, route, fam, label = SCHEME_OF[r.scheme]
    rows.append(dict(
        track="TrackA", prediction_target="点值 R_Z（等价阻抗降低量 1-R_Z）",
        fit_mode=mode, route=route, model_family=fam, model=r.scheme,
        level="LORO", condition=r.condition, condition_scope=SCOPE[r.condition],
        k=r.k, n=r.n, n_rats=COND_SPEC[r.condition]["rats"], n_folds=r.n_folds,
        **{c: esc(getattr(r, c, np.nan)) for c in CORE},
        coverage=np.nan, mean_width=np.nan, winkler_mean=np.nan,
        source="SegmentedFit/COMPARISON_OVERALL.csv",
        note="★ 唯一验收口径；delta 相对同条件全局最优基线"))
for r in segfull.itertuples():
    mode, route, fam, label = SCHEME_OF[r.scheme]
    rows.append(dict(
        track="TrackA", prediction_target="点值 R_Z（等价阻抗降低量 1-R_Z）",
        fit_mode=mode, route=route, model_family=fam, model=r.scheme,
        level="full", condition=r.condition, condition_scope=SCOPE[r.condition],
        k=np.nan, n=np.nan, n_rats=COND_SPEC[r.condition]["rats"], n_folds=np.nan,
        r2=r.r2, adj_r2=np.nan, rmse=r.rmse, mae=r.mae, bias=np.nan,
        aicc=np.nan, bic=np.nan, r2_boot_lo=np.nan, r2_boot_hi=np.nan,
        delta_r2_vs_best=np.nan, delta_r2_boot_lo=np.nan, delta_r2_boot_hi=np.nan,
        coverage=np.nan, mean_width=np.nan, winkler_mean=np.nan,
        source="SegmentedFit/FULL_GROUP_MEAN.csv",
        note="群体均值层样本内，仅描述拟合程度、不作验收"))

# (3) TrackB：区间指标（无分段版本）
for r in tbfull.itertuples():
    if r.condition not in SCOPE:
        continue
    rows.append(dict(
        track="TrackB", prediction_target="区间 [R_Z_lo, R_Z_hi]（等价降低量区间）",
        fit_mode="全局", route="TrackB", model_family="区间集成", model="TrackB 校准区间",
        level="COVERAGE", condition=r.condition, condition_scope=SCOPE[r.condition],
        k=np.nan, n=r.n_points, n_rats=r.n_rats, n_folds=np.nan,
        **{c: np.nan for c in CORE},
        coverage=r.coverage, mean_width=r.mean_width, winkler_mean=r.winkler_mean,
        source="ThreeRoutes/TRACKB_FULL_METRICS.csv",
        note="★ TrackB 只报 coverage/width/Winkler，不报 R²/RMSE 作验收"))

master = pd.DataFrame(rows)
front = ["track", "prediction_target", "fit_mode", "route", "model_family", "model",
         "level", "condition", "condition_scope", "k", "n", "n_rats", "n_folds"]
master = master[front + CORE + ["coverage", "mean_width", "winkler_mean", "source", "note"]]
master.to_csv(TAB / "MASTER_COMPARISON.csv", index=False, encoding="utf-8-sig", float_format="%.8g")

# ---------------------------------------------------------------- 2x2（细化为 3x2）对比矩阵
def best_loro(condition, mode=None, route=None, model=None):
    q = master[(master.track == "TrackA") & (master.level == "LORO") &
               (master.condition == condition)]
    if mode:
        q = q[q.fit_mode == mode]
    if route:
        q = q[q.route == route]
    if model:
        q = q[q.model == model]
    return None if q.empty else q.loc[q.r2.idxmax()]

matrix = []
for cond in ["C1_all", "C4_lf_hd"]:
    for mode in ["全局", "分段独立", "分段增量"]:
        for route in ["TrackA1", "TrackA2", "TrackA3"]:
            b = best_loro(cond, mode=mode, route=route)
            matrix.append(dict(condition=cond, condition_scope=SCOPE[cond],
                               fit_mode=mode, route=route,
                               model_family={"TrackA1": "纯经验", "TrackA2": "纯机理",
                                             "TrackA3": "机理驱动经验"}[route],
                               implemented=b is not None,
                               best_model=None if b is None else b.model,
                               k=None if b is None else b.k,
                               loro_r2=None if b is None else round(float(b.r2), 6),
                               loro_rmse=None if b is None else round(float(b.rmse), 6),
                               aicc=None if b is None else round(float(b.aicc), 3),
                               delta_vs_global_a3=None if b is None else
                               round(float(b.r2) - float(best_loro(cond, mode="全局", route="TrackA3").r2), 6)))
        tb = master[(master.track == "TrackB") & (master.condition == cond)]
        tbr = tb.iloc[0] if len(tb) else None
        matrix.append(dict(condition=cond, condition_scope=SCOPE[cond],
                           fit_mode=mode, route="TrackB", model_family="区间集成",
                           implemented=(mode == "全局" and tbr is not None),
                           best_model="TrackB 校准区间" if (mode == "全局" and tbr is not None) else None,
                           k=None, loro_r2=None, loro_rmse=None, aicc=None, delta_vs_global_a3=None))
matrix = pd.DataFrame(matrix)
matrix.to_csv(TAB / "MATRIX_FITMODE_X_SCOPE.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- 全局 vs 分段（配对 delta，四条件）
gv = segov[segov.scheme.str.startswith(("SEG_", "ADJ_"))][
    ["condition", "scheme", "k", "r2", "rmse", "aicc", "bic",
     "delta_r2_vs_best", "delta_r2_boot_lo", "delta_r2_boot_hi", "delta_r2_boot_p_positive"]].copy()
gv["condition_scope"] = gv.condition.map(SCOPE)
gv["fit_mode"] = gv.scheme.map(lambda s: SCHEME_OF[s][0])
gv["route"] = gv.scheme.map(lambda s: SCHEME_OF[s][1])
gv["model_family"] = gv.scheme.map(lambda s: SCHEME_OF[s][2])
gv["baseline_global_a3_r2"] = gv.condition.map(
    lambda c: float(three[(three.condition == c) & (three.model == "A3")].loro_r2.iloc[0]))
gv.to_csv(TAB / "GLOBAL_VS_SEGMENTED.csv", index=False, encoding="utf-8-sig", float_format="%.8g")

# ---------------------------------------------------------------- 所有条件 vs 最佳条件范围
scope_rows = []
for cond, spec in COND_SPEC.items():
    a3 = three[(three.condition == cond) & (three.model == "A3")].iloc[0]
    tb = tb4[tb4.condition == cond].iloc[0]
    scope_rows.append(dict(condition=cond, condition_scope=SCOPE[cond],
                           doses=spec["doses"], freqs=spec["freqs"],
                           n_cond_nodes=spec["nodes"], n_rat_points=spec["rat_points"],
                           n_rats=spec["rats"], loro_folds=spec["folds"],
                           a3_full_r2=float(a3.full_r2), a3_loro_r2=float(a3.loro_r2),
                           a3_loro_rmse=float(a3.loro_rmse),
                           a11_loro_r2=float(three[(three.condition == cond) &
                                                   (three.model == "A1.1")].loro_r2.iloc[0]),
                           a2_loro_r2=float(three[(three.condition == cond) &
                                                  (three.model == "A2")].loro_r2.iloc[0]),
                           trackb_coverage=float(tb.coverage),
                           trackb_mean_width=float(tb.mean_width),
                           trackb_winkler=float(tb.trackb_winkler)))
scope = pd.DataFrame(scope_rows)
scope.to_csv(TAB / "ALL_VS_BEST_CONDITION.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
scope.to_csv(TAB / "CONDITION_SCOPE.csv", index=False, encoding="utf-8-sig", float_format="%.8g")

# ---------------------------------------------------------------- 三路线对比（LORO，四条件）
tr = three[["condition", "model", "route", "family", "k", "n_cond_nodes", "n_rat_points",
            "n_rats", "loro_folds", "full_r2", "full_rmse", "loco_r2", "loro_r2",
            "loro_rmse", "loro_per_rat_mean_r2"]].copy()
tr["condition_scope"] = tr.condition.map(SCOPE)
tr["fit_mode"] = "全局"
tr.to_csv(TAB / "TRACKA_THREE_ROUTES.csv", index=False, encoding="utf-8-sig", float_format="%.8g")

# ---------------------------------------------------------------- TrackB 四条件
tbo = tbfull[tbfull.condition.isin(SCOPE)][
    ["condition", "n_points", "n_rats", "coverage", "coverage_wilson_lo", "coverage_wilson_hi",
     "target_coverage", "mean_width", "median_width", "min_width", "max_width",
     "winkler_mean", "n_miss", "miss_below", "miss_above", "padding_share_of_width",
     "diag_center_r2", "diag_center_rmse", "diagnostic_notice"]].copy()
tbo["condition_scope"] = tbo.condition.map(SCOPE)
tbo.to_csv(TAB / "TRACKB_INTERVALS.csv", index=False, encoding="utf-8-sig", float_format="%.8g")

# ---------------------------------------------------------------- 尿素回升证据与逐段指标
urea.to_csv(TAB / "UREA_REBOUND_SUMMARY.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
segseg[["condition", "scheme", "seg", "seg_label", "n", "k", "r2", "rmse", "mae", "bias",
        "aicc", "bic"]].to_csv(TAB / "PER_SEGMENT_METRICS.csv", index=False,
                               encoding="utf-8-sig", float_format="%.8g")
variants.to_csv(TAB / "A1_VARIANT_SELECTION.csv", index=False, encoding="utf-8-sig", float_format="%.8g")

# ---------------------------------------------------------------- 框架空格子（诚实标注）
gaps = []
for cond in ["C1_all", "C4_lf_hd"]:
    for mode in ["分段独立", "分段增量"]:
        gaps.append(dict(condition=cond, condition_scope=SCOPE[cond], fit_mode=mode,
                         route="TrackA2", missing="纯机理 A2 的分段版本",
                         reason="分段任务只在 TrackA1 与 TrackA3 上实现（用户上一轮明确指定）；A2 未做分段改造",
                         consequence="2x2 矩阵中该格为空，三条路线在分段维度下不构成完整对比"))
        gaps.append(dict(condition=cond, condition_scope=SCOPE[cond], fit_mode=mode,
                         route="TrackB", missing="TrackB 的分段区间版本",
                         reason="TrackB 校准区间按条件切片得到，未做分段重建/重新校准",
                         consequence="区间维度只有全局版本，不能与分段 TrackA 做同维度横比"))
pd.DataFrame(gaps).to_csv(TAB / "FRAMEWORK_GAPS.csv", index=False, encoding="utf-8-sig")

# ---------------------------------------------------------------- SUMMARY.json
c1 = scope[scope.condition == "C1_all"].iloc[0]
c4 = scope[scope.condition == "C4_lf_hd"].iloc[0]
def mat(cond, mode, route):
    q = matrix[(matrix.condition == cond) & (matrix.fit_mode == mode) & (matrix.route == route)]
    return None if q.empty or not bool(q.implemented.iloc[0]) else float(q.loro_r2.iloc[0])

summary = dict(
    framework=dict(
        trackA="点预测：具体阻抗降低值 R_Z（等价 Δ=1−R_Z）",
        trackB="区间预测：阻抗降低区间 [lo, hi]",
        routes=["TrackA1 纯经验", "TrackA2 纯机理", "TrackA3 机理驱动经验"],
        dimensions=["全局拟合 vs 分段拟合", "所有条件 vs 最佳条件范围"],
        best_scope="C4_lf_hd（0.15/0.20 M × 100/120 Hz）"),
    consistency=dict(n_checks=len(chk), max_abs_delta=max_delta, passed=consistency_ok,
                     note="分段包 BASE_* 与三路线冻结值的 LORO/full 逐项比对"),
    condition_scope=scope.to_dict("records"),
    matrix=matrix.to_dict("records"),
    key_numbers=dict(
        c1_a3_loro_r2=float(c1.a3_loro_r2), c4_a3_loro_r2=float(c4.a3_loro_r2),
        c1_a3_full_r2=float(c1.a3_full_r2), c4_a3_full_r2=float(c4.a3_full_r2),
        c1_trackb_coverage=float(c1.trackb_coverage), c4_trackb_coverage=float(c4.trackb_coverage),
        c1_trackb_width=float(c1.trackb_mean_width), c4_trackb_width=float(c4.trackb_mean_width),
        m_global_a3_c1=mat("C1_all", "全局", "TrackA3"),
        m_global_a3_c4=mat("C4_lf_hd", "全局", "TrackA3"),
        m_seg_a3_c1=mat("C1_all", "分段独立", "TrackA3"),
        m_seg_a3_c4=mat("C4_lf_hd", "分段独立", "TrackA3"),
        m_adj_a3_c1=mat("C1_all", "分段增量", "TrackA3"),
        m_adj_a3_c4=mat("C4_lf_hd", "分段增量", "TrackA3"),
    ),
    urea_rebound=urea[urea.condition == "C1_all"][
        ["scheme", "d_obs_mean", "d_pred_mean", "sign_match_n", "sign_match_rate"]].to_dict("records"),
    n_master_rows=len(master), n_gaps=len(gaps),
)
(ROOT / "SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

print("=== 跨包一致性校验 ===")
print(f"  比对项 {len(chk)} | max|Δ| = {max_delta:.3e} | {'PASS' if consistency_ok else 'FAIL'}")
print(f"\n=== MASTER_COMPARISON: {master.shape} ===")
print("  track:", master.track.value_counts().to_dict())
print("  fit_mode:", master.fit_mode.value_counts().to_dict())
print("  level:", master.level.value_counts().to_dict())
print("\n=== 3x2 对比矩阵（LORO R²）===")
print(matrix.pivot_table(index=["fit_mode", "route"], columns="condition",
                         values="loro_r2", aggfunc="first").round(4).to_string())
print("\n=== 空缺格子 ===", len(gaps))
print("SUMMARY.json 已写出")
