#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""按试剂处理分段拟合 —— 主计算脚本

三段：50% 乙酸(t=5) / EDTA(t=20,35,65,95) / 饱和尿素(t=105)
两条路线：TrackA1 纯经验（每段可改公式、按 LORO 择优）、TrackA3 机理+经验（按段补机理项）
对照：已冻结的 5 个全局单调模型，在同一 LORO 框架内逐字复刻重算，并校验与冻结值一致。

评价口径（继承既有铁律）：
  * LORO（Leave-One-Rat-Out，24/12 折）是唯一验收口径；full 仅作参考，不混淆群体均值与个体验证；
  * 四条件评价集不同、折数不同，R² 不可横向解读为模型优劣；
  * 拼接整体 R² 用同一 576/288/288/144 个逐鼠点，与冻结基线严格可比；
  * 分段模型参数更多，必须同时报 k / AICc / BIC / 按鼠聚类 bootstrap 的配对 ΔR² CI。

不改动任何已冻结模型公式、参数、数据；24 鼠全保留。
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import segmented_models as M
from metrics_lib import compute_metrics, cluster_bootstrap

warnings.filterwarnings("ignore")

HERE = Path(__file__).resolve().parent
DATA, TAB, FIG = HERE / "data", HERE / "tables", HERE / "figures"
for d in (TAB, FIG):
    d.mkdir(parents=True, exist_ok=True)

FROZEN = DATA / "frozen_measured_24rat.csv"
PRIOR = DATA / "mechanism_timeseries_quantiles.csv"
FROZEN_BASE = TAB / "frozen_baseline_metrics.csv"

T_MOD = [5, 20, 35, 65, 95, 105]
CONDITIONS = {
    "C1_all": dict(label="C1 全部条件（4 剂量 x 4 频率）",
                   doses=[0.05, 0.10, 0.15, 0.20], freqs=[100, 120, 1000, 10000]),
    "C2_lowfreq": dict(label="C2 低频（100 / 120 Hz）",
                       doses=[0.05, 0.10, 0.15, 0.20], freqs=[100, 120]),
    "C3_highdose": dict(label="C3 高浓度（15% / 20%）",
                        doses=[0.15, 0.20], freqs=[100, 120, 1000, 10000]),
    "C4_lf_hd": dict(label="C4 低频 + 高浓度", doses=[0.15, 0.20], freqs=[100, 120]),
}
SCHEMES = ["BASE_A1.1", "BASE_A1.2", "BASE_A1.3", "BASE_A2", "BASE_A3",
           "SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]
# V1 = 三段从零独立标定（SEG_*）；V2 = 保留全局骨架 + 最少分段增量修正（ADJ_*）


# ---------------------------------------------------------------- 数据
def load_data():
    s = pd.read_csv(FROZEN)
    rl = (s.groupby(["rat", "edta_concentration_mol_per_l", "frequency_hz", "global_time_min"])["rz"]
          .mean().reset_index(name="y"))
    rl = rl[rl.global_time_min.isin(T_MOD)].copy()
    rl.columns = ["rat", "dose", "freq", "t", "y"]
    rl = M.base_attach(M.attach(rl)).reset_index(drop=True)
    cond = (rl.groupby(["dose", "freq", "t"])
            .agg(y=("y", "mean"), sem=("y", "sem"), n=("rat", "nunique")).reset_index())
    cond = M.base_attach(M.attach(cond))
    return rl, cond


def filt(df, spec):
    return (df[df.dose.isin(spec["doses"]) & df.freq.isin(spec["freqs"])]
            .copy().reset_index(drop=True))


def seg_mask(df, seg):
    return df.t.isin(M.SEGS[seg]["t"]).to_numpy()


# ---------------------------------------------------------------- 分段拟合（折内）
def fit_a1_all(train, selection):
    s2v = selection["S2_EDTA"]
    X2 = train[seg_mask(train, "S2_EDTA")]
    p2 = M.a1_fit("S2_EDTA", s2v, X2, X2.y.to_numpy(float))
    ctx = dict(s2_var=s2v, s2_params=p2)
    X1 = train[seg_mask(train, "S1_acetic")]
    p1 = M.a1_fit("S1_acetic", selection["S1_acetic"], X1, X1.y.to_numpy(float))
    X3 = train[seg_mask(train, "S3_urea")]
    p3 = M.a1_fit("S3_urea", selection["S3_urea"], X3, X3.y.to_numpy(float), ctx=ctx)
    return {"S1_acetic": p1, "S2_EDTA": p2, "S3_urea": p3, "ctx": ctx}


def fit_a3_all(train):
    Xm = train[seg_mask(train, "S2_EDTA")]
    p4 = M.a3_fit_mech(Xm, Xm.y.to_numpy(float))
    P = {"mech": p4}
    for seg in M.SEG_ORDER:
        Xs = train[seg_mask(train, seg)]
        P[seg] = M.a3_fit_segment(seg, Xs, Xs.y.to_numpy(float), p4)
    return P


def predict_scheme(scheme, seg, P, X):
    if scheme == "SEG_A1":
        return M.a1_predict(seg, P["selection"][seg], P[seg], X, ctx=P["ctx"])
    if scheme == "SEG_A3":
        return M.a3_predict(seg, P[seg], X, P["mech"])
    if scheme.startswith("ADJ_"):
        return M.adj_predict(scheme, seg, P, X)
    raise ValueError(scheme)


def k_of(scheme, selection=None, base_name=None):
    if scheme == "SEG_A1":
        return sum(M.A1_VARIANTS[s][selection[s]]["k"] for s in M.SEG_ORDER)
    if scheme == "SEG_A3":
        return M.A3_TOTAL_K
    if scheme.startswith("ADJ_"):
        return M.ADJ_K[scheme]
    return M.BASE_K[base_name]


# ---------------------------------------------------------------- LORO 主引擎
def loro(scheme, r, cond, spec, selection=None, base_name=None):
    """返回 (yhat, P_full, k)。LORO 折内三段分别用其余鼠重新标定，留出鼠全段皆为未见数据。"""
    y = r.y.to_numpy(float)
    rats = r.rat.to_numpy()
    yhat = np.full(len(r), np.nan)
    uniq = np.unique(rats)

    if scheme.startswith("BASE_"):
        p_full = M.base_fit(base_name, cond, cond.y.to_numpy(float))
        for one in uniq:
            te, tr = rats == one, rats != one
            pp = M.base_fit(base_name, r[tr], y[tr], p_init=p_full)
            yhat[te] = M.base_predict(base_name, pp, r[te])
        return yhat, p_full, k_of(scheme, base_name=base_name)

    if scheme.startswith("ADJ_"):
        name = M.ADJ_BASE[scheme]
        p_full = M.base_fit(name, cond, cond.y.to_numpy(float))
        for one in uniq:
            te, tr = rats == one, rats != one
            pp = M.base_fit(name, r[tr], y[tr], p_init=p_full)
            P = M.adj_fit(scheme, r[tr], y[tr], pp)
            test = r[te]
            for seg in M.SEG_ORDER:
                m = seg_mask(test, seg)
                if m.any():
                    yhat[np.flatnonzero(te)[m]] = M.adj_predict(scheme, seg, P, test[m])
        return yhat, M.adj_fit(scheme, r, y, p_full), k_of(scheme)

    if scheme == "SEG_A1":
        for one in uniq:
            te, tr = rats == one, rats != one
            P = fit_a1_all(r[tr], selection)
            P["selection"] = selection
            test = r[te]
            for seg in M.SEG_ORDER:
                m = seg_mask(test, seg)
                if m.any():
                    yhat[np.flatnonzero(te)[m]] = predict_scheme(scheme, seg, P, test[m])
        P_full = fit_a1_all(r, selection)
        P_full["selection"] = selection
        return yhat, P_full, k_of(scheme, selection=selection)

    if scheme == "SEG_A3":
        for one in uniq:
            te, tr = rats == one, rats != one
            P = fit_a3_all(r[tr])
            test = r[te]
            for seg in M.SEG_ORDER:
                m = seg_mask(test, seg)
                if m.any():
                    yhat[np.flatnonzero(te)[m]] = predict_scheme(scheme, seg, P, test[m])
        return yhat, fit_a3_all(r), k_of(scheme)

    raise ValueError(scheme)


# ---------------------------------------------------------------- 配对 bootstrap
def paired_delta_boot(y, ya, yb, rats, B=2000, seed=20261115):
    """按鼠整簇重采样的 ΔR² = R²(b) - R²(a) 均值与 95% CI。"""
    rng = np.random.default_rng(seed)
    uniq = np.unique(rats)
    idx_by = {u: np.flatnonzero(rats == u) for u in uniq}
    ds = []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        yy, aa, bb = y[idx], ya[idx], yb[idx]
        sst = float(np.sum((yy - yy.mean()) ** 2))
        if sst <= 1e-15:
            continue
        ra = 1 - np.sum((yy - aa) ** 2) / sst
        rb = 1 - np.sum((yy - bb) ** 2) / sst
        ds.append(rb - ra)
    if not ds:
        return (np.nan,) * 4
    return (float(np.mean(ds)), float(np.percentile(ds, 2.5)),
            float(np.percentile(ds, 97.5)), float(np.mean(np.asarray(ds) > 0)))


# ---------------------------------------------------------------- 变体择优（仅 C1）
def select_variants(rl, cond, spec):
    r = filt(rl, spec)
    c = filt(cond, spec)
    y = r.y.to_numpy(float)
    rats = r.rat.to_numpy()
    selection = dict(M.A1_DEFAULT)
    rows = []
    for seg in M.SEG_ORDER:
        best_var, best_r2 = None, -np.inf
        for var, cfg in M.A1_VARIANTS[seg].items():
            trial = dict(selection)
            trial[seg] = var
            yhat, _, _ = loro("SEG_A1", r, c, spec, selection=trial)
            m = seg_mask(r, seg)
            mm = compute_metrics(y[m], yhat[m], cfg["k"], rats=rats[m])
            rows.append(dict(seg=seg, seg_label=M.SEGS[seg]["label"], variant=var,
                             variant_label=cfg["label"], formula=cfg["formula"], k=cfg["k"],
                             n=int(m.sum()), loro_r2=mm["r2"], loro_rmse=mm["rmse"],
                             loro_mae=mm["mae"], bias=mm["bias"], aicc=mm["aicc"], bic=mm["bic"]))
            if mm["r2"] > best_r2:
                best_r2, best_var = mm["r2"], var
        selection[seg] = best_var
        print(f"  择优 {seg:11s} -> {best_var:12s} (段内 LORO R2={best_r2:.4f})")
    for row in rows:
        row["selected"] = int(M.A1_VARIANTS[row["seg"]][row["variant"]] is not None
                              and selection[row["seg"]] == row["variant"])
    return selection, pd.DataFrame(rows)


# ---------------------------------------------------------------- 曲线 / 参数 / 回升
def curves_for(scheme, r, cond, P, selection=None, base_name=None):
    out = []
    for seg in M.SEG_ORDER:
        c = cond[seg_mask(cond, seg)].copy()
        if scheme.startswith("BASE_"):
            pred = M.base_predict(base_name, P, c)
        else:
            pred = predict_scheme(scheme, seg, P, c)
        c = c.copy()
        c["y_pred"] = np.asarray(pred, float)
        c["scheme"] = scheme
        c["seg"] = seg
        out.append(c)
    return pd.concat(out, ignore_index=True)


def params_rows(scheme, condition, P, selection=None, base_name=None):
    rows = []
    if scheme.startswith("BASE_"):
        rows.append(dict(scheme=scheme, condition=condition, seg="ALL", component="base",
                         k=M.BASE_K[base_name], params=";".join(f"{v:.8g}" for v in np.asarray(P, float))))
        return rows
    if scheme.startswith("ADJ_"):
        name = M.ADJ_BASE[scheme]
        rows.append(dict(scheme=scheme, condition=condition, seg="ALL", component="global_base",
                         k=M.BASE_K[name], params=";".join(f"{v:.8g}" for v in np.asarray(P["base"], float))))
        for seg in M.SEG_ORDER:
            rows.append(dict(scheme=scheme, condition=condition, seg=seg,
                             component="segment_correction",
                             k=(1 if seg != "S3_urea" else 3),
                             params=";".join(f"{v:.8g}" for v in np.asarray(P[seg], float))))
        return rows
    if scheme == "SEG_A1":
        for seg in M.SEG_ORDER:
            rows.append(dict(scheme=scheme, condition=condition, seg=seg,
                             component=selection[seg], k=M.A1_VARIANTS[seg][selection[seg]]["k"],
                             params=";".join(f"{v:.8g}" for v in np.asarray(P[seg], float))))
    else:
        rows.append(dict(scheme=scheme, condition=condition, seg="ALL", component="mech_shared",
                         k=M.A3_CFG["mech"]["k"],
                         params=";".join(f"{v:.8g}" for v in np.asarray(P["mech"], float))))
        for seg in M.SEG_ORDER:
            rows.append(dict(scheme=scheme, condition=condition, seg=seg, component="segment",
                             k=M.A3_CFG[seg]["k"],
                             params=";".join(f"{v:.8g}" for v in np.asarray(P[seg], float))))
    return rows


def rebound_rows(scheme, condition, r, yhat):
    d = r[["dose", "freq", "t", "y"]].copy()
    d["pred"] = yhat
    g = d.groupby(["dose", "freq", "t"]).agg(y=("y", "mean"), pred=("pred", "mean")).reset_index()
    rows = []
    for (dose, freq), gg in g.groupby(["dose", "freq"]):
        gg = gg.set_index("t")
        if 95 not in gg.index or 105 not in gg.index:
            continue
        rows.append(dict(scheme=scheme, condition=condition, dose=dose, freq=freq,
                         y95=float(gg.loc[95, "y"]), y105=float(gg.loc[105, "y"]),
                         d_obs=float(gg.loc[105, "y"] - gg.loc[95, "y"]),
                         p95=float(gg.loc[95, "pred"]), p105=float(gg.loc[105, "pred"]),
                         d_pred=float(gg.loc[105, "pred"] - gg.loc[95, "pred"]),
                         sign_match=int((gg.loc[105, "y"] - gg.loc[95, "y"])
                                        * (gg.loc[105, "pred"] - gg.loc[95, "pred"]) > 0)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- 主流程
def main():
    M.init_prior(PRIOR)
    rl, cond = load_data()
    frozen = pd.read_csv(FROZEN_BASE)
    print(f"数据：逐鼠 {len(rl)} 行 / 条件节点 {len(cond)} / 鼠 {rl.rat.nunique()} / 时间点 {sorted(rl.t.unique())}")

    # 1) 冻结基线同框架复算 + 一致性校验
    print("\n[1/5] 冻结基线 LORO 同框架复算与一致性校验")
    chk = []
    for cid, spec in CONDITIONS.items():
        r, c = filt(rl, spec), filt(cond, spec)
        for name in M.BASE_K:
            yhat, _, _ = loro("BASE_" + name, r, c, spec, base_name=name)
            m = compute_metrics(r.y.to_numpy(float), yhat, M.BASE_K[name])
            f = frozen[(frozen.condition == cid) & (frozen.model == name)].iloc[0]
            chk.append(dict(condition=cid, model=name, rec_r2=m["r2"], frozen_r2=float(f.loro_r2),
                            delta_r2=m["r2"] - float(f.loro_r2),
                            rec_rmse=m["rmse"], frozen_rmse=float(f.loro_rmse),
                            delta_rmse=m["rmse"] - float(f.loro_rmse)))
    chk = pd.DataFrame(chk)
    maxdr = float(chk.delta_r2.abs().max())
    print(f"  20 项复算 vs 冻结：max|ΔR²| = {maxdr:.3e}，max|ΔRMSE| = {chk.delta_rmse.abs().max():.3e}")
    if maxdr > 1e-5:
        raise RuntimeError("基线复算与冻结值不一致，口径漂移，终止")
    chk.to_csv(TAB / "BASELINE_REPRO_CHECK.csv", index=False, encoding="utf-8-sig", float_format="%.8g")

    # 2) TrackA1 每段候选公式择优（仅 C1）
    print("\n[2/5] TrackA1 三段候选公式按 LORO 择优（C1）")
    selection, var_df = select_variants(rl, cond, CONDITIONS["C1_all"])
    var_df.insert(0, "condition", "C1_all")
    var_df.to_csv(TAB / "A1_VARIANT_SELECTION.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    print("  最终选中：", selection)

    # 3) 四条件全量评估
    print("\n[3/5] 四条件 x 7 方案 LORO 全量评估")
    overall, segmetrics, residuals, curves, params, rebound = [], [], [], [], [], []
    results = {}
    for cid, spec in CONDITIONS.items():
        r, c = filt(rl, spec), filt(cond, spec)
        y = r.y.to_numpy(float)
        rats = r.rat.to_numpy()
        yhats = {}
        for scheme in SCHEMES:
            bn = scheme.replace("BASE_", "") if scheme.startswith("BASE_") else None
            yhat, P, k = loro(scheme, r, c, spec, selection=selection, base_name=bn)
            yhats[scheme] = yhat
            if not np.isfinite(yhat).all():
                raise RuntimeError(f"非有限预测：{cid}/{scheme}")
            mm = compute_metrics(y, yhat, k, rats=rats, do_boot=True)
            mm.update(condition=cid, condition_label=spec["label"], scheme=scheme, k=k,
                      n_folds=int(len(np.unique(rats))))
            overall.append(mm)
            for seg in M.SEG_ORDER:
                m = seg_mask(r, seg)
                if scheme.startswith("BASE_"):
                    kseg = k
                elif scheme == "SEG_A1":
                    kseg = M.A1_VARIANTS[seg][selection[seg]]["k"]
                elif scheme == "SEG_A3":
                    kseg = M.A3_CFG[seg]["k"]
                else:
                    kseg = 1 if seg != "S3_urea" else 3
                sm = compute_metrics(y[m], yhat[m], kseg, rats=rats[m])
                sm.update(condition=cid, scheme=scheme, seg=seg,
                          seg_label=M.SEGS[seg]["label"], k=kseg, n=int(m.sum()))
                segmetrics.append(sm)
            e = y - yhat
            esd = e.std(ddof=1)
            rr = r[["rat", "dose", "freq", "t"]].copy()
            rr["condition"], rr["scheme"] = cid, scheme
            rr["seg"] = [next(s for s in M.SEG_ORDER if int(t) in M.SEGS[s]["t"]) for t in rr.t]
            rr["y_obs"], rr["y_pred"] = y, yhat
            rr["residual"] = e
            rr["std_residual"] = (e - e.mean()) / esd if esd > 0 else 0.0
            rr["abs_pct_error"] = np.abs(e) / np.where(np.abs(y) > 1e-6, np.abs(y), np.nan)
            residuals.append(rr)
            params.extend(params_rows(scheme, cid, P, selection=selection, base_name=bn))
            cc = curves_for(scheme, r, c, P, selection=selection, base_name=bn)
            cc["condition"] = cid
            curves.append(cc)
            rebound.append(rebound_rows(scheme, cid, r, yhat))
        # 配对 ΔR²：分段 vs 冻结最优基线（按 C1 的 A3 为锚，各条件取该条件最优冻结模型）
        best_base = max([s for s in SCHEMES if s.startswith("BASE_")],
                        key=lambda s: next(o["r2"] for o in overall
                                          if o["condition"] == cid and o["scheme"] == s))
        ya = yhats[best_base]
        for seg_scheme in ["SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]:
            dm, dlo, dhi, pfrac = paired_delta_boot(y, ya, yhats[seg_scheme], rats)
            for o in overall:
                if o["condition"] == cid and o["scheme"] == seg_scheme:
                    o.update(best_baseline=best_base, delta_r2_vs_best=o["r2"] - next(
                        x["r2"] for x in overall if x["condition"] == cid and x["scheme"] == best_base),
                        delta_r2_boot_mean=dm, delta_r2_boot_lo=dlo, delta_r2_boot_hi=dhi,
                        delta_r2_boot_p_positive=pfrac)
        results[cid] = dict(y=y, rats=rats, yhats=yhats, r=r)
        print(f"  {cid}: " + " | ".join(
            f"{s}={next(o['r2'] for o in overall if o['condition']==cid and o['scheme']==s):.4f}"
            for s in ["BASE_A3", "SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]))

    ov = pd.DataFrame(overall)
    sm = pd.DataFrame(segmetrics)
    rs = pd.concat(residuals, ignore_index=True)
    cv = pd.concat(curves, ignore_index=True)
    pm = pd.DataFrame(params)
    rb = pd.concat(rebound, ignore_index=True)
    ov.to_csv(TAB / "COMPARISON_OVERALL.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    sm.to_csv(TAB / "SEGMENT_METRICS.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    rs.to_csv(TAB / "RESIDUALS_LONG.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    cv.to_csv(TAB / "SEGMENT_CURVES.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    pm.to_csv(TAB / "SEGMENT_PARAMS.csv", index=False, encoding="utf-8-sig")
    rb.to_csv(TAB / "UREA_REBOUND.csv", index=False, encoding="utf-8-sig", float_format="%.8g")

    # 3b) 群体均值层 full（in-sample）指标：仅作参考，明确不用于验收
    kmap = {(o["condition"], o["scheme"]): o["k"] for o in overall}
    frows = []
    for (cid, scheme), g in cv.groupby(["condition", "scheme"]):
        mm = compute_metrics(g.y.to_numpy(float), g.y_pred.to_numpy(float), kmap[(cid, scheme)])
        mm.update(condition=cid, scheme=scheme, level="FULL_GROUP_MEAN_IN_SAMPLE",
                  note="群体均值层 in-sample，仅参考；LORO 才是唯一验收口径，二者不可混淆")
        frows.append(mm)
    fdf = pd.DataFrame(frows)
    fdf.to_csv(TAB / "FULL_GROUP_MEAN.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    c1full = fdf[fdf.condition == "C1_all"].set_index("scheme")

    # 4) 尿素回升汇总
    print("\n[4/5] 尿素回升表达核验")
    rbs = (rb.groupby(["condition", "scheme"])
           .agg(n_nodes=("dose", "size"), d_obs_mean=("d_obs", "mean"),
                d_pred_mean=("d_pred", "mean"), sign_match_n=("sign_match", "sum"))
           .reset_index())
    rbs["sign_match_rate"] = rbs.sign_match_n / rbs.n_nodes
    rbs["expresses_rebound"] = (rbs.d_pred_mean > 0).astype(int)
    rbs.to_csv(TAB / "UREA_REBOUND_SUMMARY.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    c1 = rbs[rbs.condition == "C1_all"]
    print(c1[["scheme", "d_obs_mean", "d_pred_mean", "sign_match_rate", "expresses_rebound"]]
          .to_string(index=False))

    # 5) 汇总 JSON
    c1ov = ov[ov.condition == "C1_all"].set_index("scheme")
    summary = dict(
        task="按试剂处理分段拟合（乙酸 / EDTA / 饱和尿素）",
        segments={s: M.SEGS[s]["label"] for s in M.SEG_ORDER},
        a1_selected=selection,
        a3_total_k=M.A3_TOTAL_K,
        a1_total_k=k_of("SEG_A1", selection=selection),
        adj_total_k=dict(ADJ_A1=M.ADJ_K["ADJ_A1"], ADJ_A3=M.ADJ_K["ADJ_A3"]),
        baseline_repro=dict(max_abs_delta_r2=maxdr, passed=True),
        c1_loro_r2={s: float(c1ov.loc[s, "r2"]) for s in SCHEMES},
        c1_loro_rmse={s: float(c1ov.loc[s, "rmse"]) for s in SCHEMES},
        c1_full_group_mean_r2={s: float(c1full.loc[s, "r2"]) for s in SCHEMES},
        c1_delta_vs_best={s: float(c1ov.loc[s, "delta_r2_vs_best"])
                          for s in ["SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]},
        c1_delta_boot_ci={s: [float(c1ov.loc[s, "delta_r2_boot_lo"]),
                              float(c1ov.loc[s, "delta_r2_boot_hi"])]
                          for s in ["SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]},
        c1_delta_boot_p_positive={s: float(c1ov.loc[s, "delta_r2_boot_p_positive"])
                                  for s in ["SEG_A1", "SEG_A3", "ADJ_A1", "ADJ_A3"]},
        urea_rebound_c1={s: dict(d_obs=float(c1.loc[c1.scheme == s, "d_obs_mean"].iloc[0]),
                                 d_pred=float(c1.loc[c1.scheme == s, "d_pred_mean"].iloc[0]),
                                 sign_rate=float(c1.loc[c1.scheme == s, "sign_match_rate"].iloc[0]))
                         for s in SCHEMES},
        n_tables=dict(overall=len(ov), segment=len(sm), residuals=len(rs), curves=len(cv)),
        ironlaw=dict(loro_only=True, conditions_not_comparable=True,
                     full_fit_is_group_mean=True, more_params_penalized_by_aicc_bic=True),
    )
    (TAB / "SEGMENTED_SUMMARY.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2),
                                                 encoding="utf-8")
    print("\n[5/5] 落盘完成 ->", TAB)
    print(json.dumps(summary["c1_loro_r2"], ensure_ascii=False, indent=2))
    return summary


if __name__ == "__main__":
    main()
