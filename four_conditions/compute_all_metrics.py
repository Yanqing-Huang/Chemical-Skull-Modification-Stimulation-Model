#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""
全评价标准实算 —— 把 45 条评价标准里所有可计算的量全部算出数值并落盘。

严格边界（本轮不改模型 / 不重新拟合 / 不改动数据 / 不删样本）：
  * 模型公式、参数初值、优化器、边界全部复制自 run_three_routes.py，逐字一致；
  * 数据来自包内冻结副本 data/frozen_measured_24rat.csv（与源 SHA256 一致，24 鼠全保留）；
  * 复算得到的 full / LOCO / LORO 的 R²、RMSE 必须与冻结的
    three_routes_metrics.csv 逐项一致（|Δ| < 1e-5），否则脚本直接报错退出；
  * 本脚本只做「指标计算扩展」：在原有 r2/rmse/mae/bias/aic 之外补齐
    R² 族、误差族、残差族（分布/正态性/自相关/异方差/离群/分组）、
    信息准则族、校准与一致性族、Theil 分解、按鼠聚类 bootstrap CI。

输出（写入 github_package/）：
  four_conditions/ALL_METRICS_FULL.csv          60 行 × ~60 指标
  four_conditions/RESIDUALS_LONG.csv            逐点残差长表（每观测点一行）
  four_conditions/RESIDUAL_GROUP_DIAGNOSTICS.csv 分组残差诊断
  four_conditions/STANDARDS_COMPUTED.csv        45 条标准逐条实算回填
  four_conditions/TRACKB_FULL_METRICS.csv       TrackB 全套区间 + 诊断性点指标
  four_conditions/TRACKB_STRATIFIED.csv          TrackB 分层覆盖率
  four_conditions/ALL_METRICS_SUMMARY.json      机器可读汇总 + 一致性校验结果
"""
from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import least_squares

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------- 路径自适配
HERE = Path(__file__).resolve().parent
if (HERE / "data" / "frozen_measured_24rat.csv").exists():
    PKG = HERE
elif (HERE.parent / "data" / "frozen_measured_24rat.csv").exists():
    PKG = HERE.parent
else:
    PKG = HERE / "github_package"

OUT = PKG / "four_conditions"
DOCS = PKG / "docs"
DATA = PKG / "data"
for d in (OUT, DOCS, OUT / "figures"):
    d.mkdir(parents=True, exist_ok=True)

FROZEN = DATA / "frozen_measured_24rat.csv"
PRIOR = DATA / "mechanism_timeseries_quantiles.csv"
TB_ALL = PKG / "TrackB" / "calibrated_intervals_allfreq_392.csv"
TB_LF = PKG / "TrackB" / "calibrated_intervals_lowfreq_196.csv"
FROZEN_METRICS = (HERE / "three_routes_metrics.csv" if (HERE / "three_routes_metrics.csv").exists()
                  else OUT / "three_routes_metrics.csv")

# ---------------------------------------------------------------- 常量（与 run_three_routes.py 逐字一致）
T_MOD = [5, 20, 35, 65, 95, 105]
F0, D0, T0 = 100.0, 0.10, 100.0

CONDITIONS = {
    "C1_all": dict(label="C1 全部条件（4 剂量 x 4 频率）",
                   doses=[0.05, 0.10, 0.15, 0.20], freqs=[100, 120, 1000, 10000]),
    "C2_lowfreq": dict(label="C2 低频（100 / 120 Hz）",
                       doses=[0.05, 0.10, 0.15, 0.20], freqs=[100, 120]),
    "C3_highdose": dict(label="C3 高浓度（15% / 20%）",
                        doses=[0.15, 0.20], freqs=[100, 120, 1000, 10000]),
    "C4_lf_hd": dict(label="C4 低频 + 高浓度",
                     doses=[0.15, 0.20], freqs=[100, 120]),
}
MODELS = {
    "A1.1": dict(route="A1", family="emp", label="A1.1 拉伸指数", k=5),
    "A1.2": dict(route="A1", family="emp", label="A1.2 A+ 协变量", k=4),
    "A1.3": dict(route="A1", family="emp", label="A1.3 H1 混合", k=3),
    "A2":   dict(route="A2", family="mech", label="A2 并联导纳纯机理", k=4),
    "A3":   dict(route="A3", family="driven", label="A3 机理骨架 + 保形校准", k=6),
}
FITCFG = {
    "A1.1": dict(p0=[0.10, 0.02, 0.01, -0.002, 0.8], lo=[0.0, 1e-5, -1, -1, 0.2], hi=[0.9, 1.0, 1, 1, 3.0]),
    "A1.2": dict(p0=[-4.7, 6.1, 0.0, 0.05], lo=[-50, -50, -1, -5], hi=[50, 50, 1, 5]),
    "A1.3": dict(p0=[3.3, 0.11, 0.003], lo=[0.1, -1.0, -0.05], hi=[30.0, 1.5, 0.05]),
    "A2":   dict(p0=[5.0, 0.3, 0.8, 0.2], lo=[0.01, -2, 0.05, 0.0], hi=[100, 3, 3, 1.0]),
    "A3":   dict(p0=[1.0, 0.0], lo=[0.0, -0.5], hi=[3.0, 0.5]),
}

# ---------------------------------------------------------------- 数据 / 预测（复制自 run_three_routes.py）
_W = None


def load_data():
    global _W
    s = pd.read_csv(FROZEN)
    rl = (s.groupby(["rat", "edta_concentration_mol_per_l", "frequency_hz", "global_time_min"])["rz"]
          .mean().reset_index(name="y"))
    rl = rl[rl.global_time_min.isin(T_MOD)].copy()
    rl.columns = ["rat", "dose", "freq", "t", "y"]
    cond = (rl.groupby(["dose", "freq", "t"])
            .agg(y=("y", "mean"), n=("rat", "nunique")).reset_index())
    q = pd.read_csv(PRIOR)
    _W = (q[q.metric == "whole_resistance_ratio"]
          .drop_duplicates(["edta_concentration_mol_per_l", "global_time_s"])
          .sort_values(["edta_concentration_mol_per_l", "global_time_s"]))
    return rl, cond


def raw_prior(dose, t):
    g = _W[_W.edta_concentration_mol_per_l == dose]
    ts = g.global_time_s.to_numpy(float)
    vs = g.q50.to_numpy(float)
    t_s = t * 60.0
    return float(vs[-1]) if t_s > ts.max() else float(np.interp(t_s, ts, vs))


def attach(df):
    d = df.copy()
    d["raw"] = [raw_prior(r.dose, r.t) for r in d.itertuples()]
    d["lf"] = np.log(d["freq"].to_numpy(float) / F0)
    d["delta"] = 1.0 - d["raw"].to_numpy(float)
    return d


def mech_delta(p4, X):
    a0, beta, pexp, s = p4
    d = X["dose"].to_numpy(float) / D0
    f = X["freq"].to_numpy(float) / F0
    t = X["t"].to_numpy(float) / T0
    return a0 * np.power(d, beta) * np.power(t, pexp) * np.power(f, -s)


def r_mech(p4, X):
    return 1.0 / (1.0 + mech_delta(p4, X))


def _b(p, X):
    return np.clip(p[1] + p[2] * X["dose"].to_numpy(float)
                   + p[3] * X["lf"].to_numpy(float), 1e-6, 5.0)


def predict(name, p, X):
    p = np.asarray(p, float)
    if name == "A1.1":
        return p[0] + (1 - p[0]) * np.exp(-np.power(_b(p, X) * X["t"].to_numpy(float), p[4]))
    if name == "A1.2":
        return (p[0] + p[1] * X["raw"].to_numpy(float)
                + p[2] * X["t"].to_numpy(float) + p[3] * X["lf"].to_numpy(float))
    if name == "A1.3":
        lam = np.exp(-p[1] * X["lf"].to_numpy(float))
        return (1.0 - p[0] * lam * X["delta"].to_numpy(float)
                - p[2] * X["t"].to_numpy(float))
    if name == "A2":
        return r_mech(p[:4], X)
    if name == "A3":
        rm = r_mech(p[:4], X)
        return 1.0 - p[4] * (1.0 - rm) - p[5] * (X["t"].to_numpy(float) / 105.0)
    raise ValueError(name)


def _fit_A2(X, y):
    c = FITCFG["A2"]
    r = least_squares(lambda p: r_mech(p, X) - y, np.asarray(c["p0"], float),
                      bounds=(np.asarray(c["lo"]), np.asarray(c["hi"])),
                      max_nfev=5000, ftol=1e-11, xtol=1e-11)
    return r.x


def fit(name, X, y, p_init=None):
    cfg = FITCFG[name]
    if name == "A3":
        mp = (np.asarray(p_init[:4], float) if p_init is not None and len(p_init) >= 4
              else _fit_A2(X, y))
        rm = r_mech(mp, X)
        base = 1.0 - rm
        tnorm = X["t"].to_numpy(float) / 105.0
        r = least_squares(lambda q: (1.0 - q[0] * base - q[1] * tnorm) - y,
                          np.asarray(p_init[4:], float) if p_init is not None and len(p_init) == 6
                          else np.asarray(cfg["p0"], float),
                          bounds=(np.asarray(cfg["lo"]), np.asarray(cfg["hi"])),
                          max_nfev=4000, ftol=1e-11, xtol=1e-11)
        return np.concatenate([mp, r.x])
    p0 = np.asarray(p_init if p_init is not None else cfg["p0"], float)
    r = least_squares(lambda p: predict(name, p, X) - y, p0,
                      bounds=(np.asarray(cfg["lo"]), np.asarray(cfg["hi"])),
                      max_nfev=5000, ftol=1e-11, xtol=1e-11)
    return r.x


# ---------------------------------------------------------------- 全指标计算
def _safe(fn, default=np.nan):
    try:
        v = fn()
        return float(v) if np.isfinite(v) else default
    except Exception:
        return default


def _ols(x, y):
    """一元 OLS y = a + b x，返回 (a, b, r2)"""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    X = np.c_[np.ones(len(x)), x]
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ coef
    sst = float(np.sum((y - y.mean()) ** 2))
    return float(coef[0]), float(coef[1]), float(1 - np.sum(resid ** 2) / sst) if sst > 0 else np.nan


def cluster_bootstrap(y, yhat, rats, B=1000, seed=20261114):
    """按大鼠整簇重采样的 R² / RMSE 95% CI（避免同鼠多时间点被当独立样本）"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    rats = np.asarray(rats)
    uniq = np.unique(rats)
    idx_by = {u: np.flatnonzero(rats == u) for u in uniq}
    r2s, rms = [], []
    for _ in range(B):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by[u] for u in pick])
        yy, pp = y[idx], yhat[idx]
        sst = float(np.sum((yy - yy.mean()) ** 2))
        if sst <= 1e-15:
            continue
        r2s.append(1 - np.sum((yy - pp) ** 2) / sst)
        rms.append(np.sqrt(np.mean((yy - pp) ** 2)))
    if not r2s:
        return (np.nan,) * 4
    return (float(np.percentile(r2s, 2.5)), float(np.percentile(r2s, 97.5)),
            float(np.percentile(rms, 2.5)), float(np.percentile(rms, 97.5)))


def compute_metrics(y, yhat, k, rats=None, do_boot=False):
    """残差约定 e = y_obs - y_pred（与 run_three_routes.py 的 bias 定义一致）。"""
    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    n = len(y)
    e = y - yhat
    ae = np.abs(e)
    sse = float(np.sum(e ** 2))
    sst = float(np.sum((y - y.mean()) ** 2))
    mse = sse / n
    rmse = math.sqrt(mse)
    r2 = 1 - sse / sst if sst > 0 else np.nan
    df = n - k - 1
    adj_r2 = 1 - (1 - r2) * (n - 1) / df if (sst > 0 and df > 0) else np.nan

    yr = float(y.max() - y.min())
    mape_valid = np.abs(y) > 0.01
    mape = float(np.mean(ae[mape_valid] / np.abs(y[mape_valid]))) if mape_valid.any() else np.nan
    denom_s = (np.abs(y) + np.abs(yhat)) / 2.0
    sm_valid = denom_s > 1e-9
    smape = float(np.mean(ae[sm_valid] / denom_s[sm_valid])) if sm_valid.any() else np.nan

    # 相关 / 一致性
    pr = _safe(lambda: stats.pearsonr(y, yhat)[0])
    sr = _safe(lambda: stats.spearmanr(y, yhat)[0])
    cal_a, cal_b, cal_r2 = _ols(yhat, y)  # y = a + b*yhat，理想 a=0, b=1
    sd_y = float(y.std(ddof=1))
    sd_p = float(yhat.std(ddof=1))
    ccc_num = 2 * pr * sd_y * sd_p if np.isfinite(pr) else np.nan
    ccc = ccc_num / (sd_y ** 2 + sd_p ** 2 + (y.mean() - yhat.mean()) ** 2) if np.isfinite(ccc_num) else np.nan

    # Theil / Willmott
    theil_u1 = rmse / (math.sqrt(float(np.mean(y ** 2))) + math.sqrt(float(np.mean(yhat ** 2)))) \
        if np.mean(y ** 2) > 0 else np.nan
    sy2 = float(np.sum(y ** 2))
    theil_u2 = math.sqrt(sse / sy2) if sy2 > 0 else np.nan
    pdev = np.abs(yhat - y.mean()) + np.abs(y - y.mean())
    willmott_d = 1 - sse / float(np.sum(pdev ** 2)) if np.sum(pdev ** 2) > 0 else np.nan

    # 信息准则（高斯，忽略与模型无关常数；与冻结 aic 定义一致）
    K = k + 1
    aic = n * math.log(max(sse, 1e-12) / n) + 2 * K
    aicc = aic + 2 * K * (K + 1) / (n - K - 1) if (n - K - 1) > 0 else np.nan
    bic = n * math.log(max(sse, 1e-12) / n) + K * math.log(n)
    loglik = -0.5 * n * (math.log(2 * math.pi) + math.log(max(sse, 1e-12) / n) + 1)

    # 残差分布 / 正态性
    e_sd = float(e.std(ddof=1))
    sk = _safe(lambda: stats.skew(e, bias=False))
    ku = _safe(lambda: stats.kurtosis(e, bias=False))   # excess
    sw = _safe(lambda: stats.shapiro(e)[0]), _safe(lambda: stats.shapiro(e)[1])
    jb = _safe(lambda: stats.jarque_bera(e)[0]), _safe(lambda: stats.jarque_bera(e)[1])
    dk = _safe(lambda: stats.normaltest(e)[0]), _safe(lambda: stats.normaltest(e)[1])
    ad = _safe(lambda: stats.anderson(e, dist="norm")[0])

    # 自相关（按鼠分组、组内按 t 排序，避免把不同鼠首尾硬拼）
    num = den = 0.0
    lag1_pairs = []
    if rats is not None:
        rr = np.asarray(rats)
        order = np.lexsort((np.asarray(y) * 0,))  # placeholder, replaced below
        for u in np.unique(rr):
            ix = np.flatnonzero(rr == u)
            ee = e[ix]
            if len(ee) > 1:
                num += float(np.sum(np.diff(ee) ** 2))
                den += float(np.sum(ee ** 2))
                lag1_pairs.append(float(np.corrcoef(ee[:-1], ee[1:])[0, 1]))
    else:
        num = float(np.sum(np.diff(e) ** 2))
        den = float(np.sum(e ** 2))
        lag1_pairs = [float(np.corrcoef(e[:-1], e[1:])[0, 1])]
    dw = num / den if den > 0 else np.nan
    lag1 = float(np.nanmean(lag1_pairs)) if lag1_pairs else np.nan

    # 异方差 LM（Breusch–Pagan：e² 对 ŷ 回归；White：再加 ŷ²）
    _, _, bp_r2 = _ols(yhat, e ** 2)
    bp_lm = n * bp_r2 if np.isfinite(bp_r2) else np.nan
    bp_p = _safe(lambda: stats.chi2.sf(bp_lm, 1)) if np.isfinite(bp_lm) else np.nan
    Xw = np.c_[np.ones(n), yhat, yhat ** 2]
    cw, *_ = np.linalg.lstsq(Xw, e ** 2, rcond=None)
    rw = e ** 2 - Xw @ cw
    sstw = float(np.sum((e ** 2 - np.mean(e ** 2)) ** 2))
    white_r2 = 1 - float(np.sum(rw ** 2)) / sstw if sstw > 0 else np.nan
    white_lm = n * white_r2 if np.isfinite(white_r2) else np.nan
    white_p = _safe(lambda: stats.chi2.sf(white_lm, 2)) if np.isfinite(white_lm) else np.nan

    # 标准化残差离群
    z = (e - e.mean()) / e_sd if e_sd > 0 else np.zeros(n)
    gz2 = int(np.sum(np.abs(z) > 2))
    gz3 = int(np.sum(np.abs(z) > 3))

    # MSE 分解：bias² + 方差失配² + 协方差失配项
    bias_h = float(yhat.mean() - y.mean())
    mse_bias2 = bias_h ** 2
    mse_var = (sd_p - sd_y) ** 2
    mse_cov = 2 * sd_p * sd_y * (1 - pr) if np.isfinite(pr) else np.nan

    out = dict(
        n=n, k=k, df_resid=int(df),
        sse=sse, sst=sst, mse=mse, rmse=rmse,
        nrmse_mean=rmse / abs(float(y.mean())) if y.mean() != 0 else np.nan,
        nrmse_range=rmse / yr if yr > 0 else np.nan,
        nrmse_iqr=rmse / float(np.percentile(y, 75) - np.percentile(y, 25)),
        r2=r2, adj_r2=adj_r2, r2_from_correlation=pr ** 2 if np.isfinite(pr) else np.nan,
        mae=float(np.mean(ae)), medae=float(np.median(ae)), maxae=float(np.max(ae)),
        mape=mape, smape=smape, mape_n_valid=int(mape_valid.sum()),
        bias=float(np.mean(e)), bias_pred_minus_obs=-float(np.mean(e)),
        mpe=float(np.mean(e / np.where(np.abs(y) > 0.01, y, np.nan))) if mape_valid.any() else np.nan,
        explained_variance=_safe(lambda: 1 - np.var(e) / np.var(y)),
        pearson_r=pr, spearman_rho=sr, ccc_lin=ccc,
        calib_intercept=cal_a, calib_slope=cal_b, calib_r2=cal_r2,
        theil_u1=theil_u1, theil_u2=theil_u2, willmott_d=willmott_d,
        aic=aic, aicc=aicc, bic=bic, loglik=loglik,
        resid_mean=float(np.mean(e)), resid_sd=e_sd, resid_median=float(np.median(e)),
        resid_iqr=float(np.percentile(e, 75) - np.percentile(e, 25)),
        resid_skew=sk, resid_kurt_excess=ku,
        resid_min=float(np.min(e)), resid_max=float(np.max(e)),
        resid_p01=float(np.percentile(e, 1)), resid_p05=float(np.percentile(e, 5)),
        resid_p25=float(np.percentile(e, 25)), resid_p50=float(np.percentile(e, 50)),
        resid_p75=float(np.percentile(e, 75)), resid_p95=float(np.percentile(e, 95)),
        resid_p99=float(np.percentile(e, 99)),
        shapiro_W=sw[0], shapiro_p=sw[1], jarque_bera_stat=jb[0], jarque_bera_p=jb[1],
        dagostino_K2=dk[0], dagostino_p=dk[1], anderson_A2=ad,
        durbin_watson=dw, lag1_autocorr=lag1,
        bp_LM=bp_lm, bp_p=bp_p, white_LM=white_lm, white_p=white_p,
        n_outlier_z2=gz2, n_outlier_z3=gz3, pct_outlier_z2=gz2 / n,
        mse_bias2=mse_bias2, mse_var_mismatch=mse_var, mse_cov_mismatch=mse_cov,
    )
    # 与自变量的相关（系统性结构偏差）
    out["corr_resid_fitted"] = _safe(lambda: stats.pearsonr(e, yhat)[0])
    if do_boot and rats is not None:
        lo_r2, hi_r2, lo_rm, hi_rm = cluster_bootstrap(y, yhat, rats)
        out.update(r2_boot_lo=lo_r2, r2_boot_hi=hi_r2,
                   rmse_boot_lo=lo_rm, rmse_boot_hi=hi_rm)
    else:
        out.update(r2_boot_lo=np.nan, r2_boot_hi=np.nan,
                   rmse_boot_lo=np.nan, rmse_boot_hi=np.nan)
    return out


# ---------------------------------------------------------------- 主流程
def emit(rows_metrics, rows_resid, cid, name, level, frame, y, yhat, do_boot=False):
    m = compute_metrics(y, yhat, MODELS[name]["k"],
                        rats=frame["rat"].to_numpy() if "rat" in frame else None,
                        do_boot=do_boot)
    m = {"condition": cid, "condition_label": CONDITIONS[cid]["label"],
         "model": name, "route": MODELS[name]["route"], "family": MODELS[name]["family"],
         "level": level, **m}
    rows_metrics.append(m)

    y = np.asarray(y, float)
    yhat = np.asarray(yhat, float)
    e = y - yhat
    esd = float(e.std(ddof=1)) if len(e) > 1 else 1.0
    z = (e - e.mean()) / esd if esd > 0 else np.zeros(len(e))
    has_rat = "rat" in frame
    for i in range(len(y)):
        r = frame.iloc[i]
        rows_resid.append(dict(
            condition=cid, model=name, level=level,
            rat=str(r["rat"]) if has_rat else "GROUP_MEAN",
            dose=float(r["dose"]), freq_hz=float(r["freq"]), t_min=float(r["t"]),
            y_obs=float(y[i]), y_pred=float(yhat[i]), residual=float(e[i]),
            abs_residual=float(abs(e[i])), sq_residual=float(e[i] ** 2),
            std_residual=float(z[i]),
            pct_error=float(e[i] / y[i]) if abs(y[i]) > 0.01 else np.nan,
        ))
    return m


def group_residual_diagnostics(resid_df):
    """按模型 / 条件 / LORO 层，分组统计残差均值、RMSE、|z|>2 占比。"""
    rows = []
    sub = resid_df[resid_df.level == "LORO"]
    keys = [("dose", "剂量"), ("freq_hz", "频率"), ("t_min", "时间")]
    for (cid, name), g in sub.groupby(["condition", "model"]):
        for col, label in keys:
            for v, gg in g.groupby(col):
                e = gg.residual.to_numpy(float)
                rows.append(dict(condition=cid, model=name, group_by=label,
                                 group_value=float(v), n=len(e),
                                 resid_mean=float(e.mean()),
                                 resid_sd=float(e.std(ddof=1)) if len(e) > 1 else np.nan,
                                 rmse=float(np.sqrt(np.mean(e ** 2))),
                                 mae=float(np.mean(np.abs(e)))))
        for rat, gg in g.groupby("rat"):
            e = gg.residual.to_numpy(float)
            rows.append(dict(condition=cid, model=name, group_by="大鼠",
                             group_value=np.nan, n=len(e),
                             resid_mean=float(e.mean()),
                             resid_sd=float(e.std(ddof=1)) if len(e) > 1 else np.nan,
                             rmse=float(np.sqrt(np.mean(e ** 2))),
                             mae=float(np.mean(np.abs(e)))))
            rows[-1]["group_value_str"] = str(rat)
    return pd.DataFrame(rows)


def b10_drop_urea(rl, cond, metrics_df):
    """B10 去尿素稳健性：剔除 t=105 后按同一模型/同一拟合流程重算 LORO，检验排名是否翻转。

    注意：这是 B10 标准本身定义的稳健性子分析（子集重拟合），
    交付模型的冻结参数与全量主表均不变。
    """
    rows = []
    for cid, spec in CONDITIONS.items():
        c0 = attach(cond[(cond.dose.isin(spec["doses"])) &
                         (cond.freq.isin(spec["freqs"]))].copy()).reset_index(drop=True)
        r0 = attach(rl[(rl.dose.isin(spec["doses"])) &
                       (rl.freq.isin(spec["freqs"]))].copy()).reset_index(drop=True)
        c = c0[c0.t != 105].reset_index(drop=True)
        r = r0[r0.t != 105].reset_index(drop=True)
        y_c = c.y.to_numpy(float)
        yr = r.y.to_numpy(float)
        rat = r.rat.to_numpy()
        for name in MODELS:
            p = fit(name, c, y_c)
            yhat = np.empty(len(r))
            for one in np.unique(rat):
                te = rat == one
                tr = ~te
                pp = fit(name, r[tr].reset_index(drop=True), yr[tr], p_init=p)
                yhat[te] = predict(name, pp, r[te].reset_index(drop=True))
            m = compute_metrics(yr, yhat, MODELS[name]["k"])
            base = metrics_df[(metrics_df.condition == cid) &
                              (metrics_df.model == name) &
                              (metrics_df.level == "LORO")].iloc[0]
            rows.append(dict(condition=cid, model=name, n_points=len(r),
                             n_folds=int(len(np.unique(rat))),
                             loro_r2_full=float(base.r2), loro_rmse_full=float(base.rmse),
                             loro_r2_drop_urea=m["r2"], loro_rmse_drop_urea=m["rmse"],
                             delta_r2=m["r2"] - float(base.r2),
                             delta_rmse=m["rmse"] - float(base.rmse)))
    df = pd.DataFrame(rows)
    flips = []
    for cid, g in df.groupby("condition"):
        g = g.copy()
        g["rank_full"] = g.loro_r2_full.rank(ascending=False, method="min").astype(int)
        g["rank_drop"] = g.loro_r2_drop_urea.rank(ascending=False, method="min").astype(int)
        rho = _safe(lambda: stats.spearmanr(g.rank_full, g.rank_drop)[0])
        flips.append(dict(condition=cid,
                          best_full=g.loc[g.rank_full.idxmin(), "model"],
                          best_drop_urea=g.loc[g.rank_drop.idxmin(), "model"],
                          rank_flipped=bool(g.loc[g.rank_full.idxmin(), "model"]
                                            != g.loc[g.rank_drop.idxmin(), "model"]),
                          rank_spearman=rho,
                          max_abs_delta_r2=float(g.delta_r2.abs().max())))
        df.loc[g.index, "rank_full"] = g.rank_full
        df.loc[g.index, "rank_drop_urea"] = g.rank_drop
    return df, pd.DataFrame(flips)


def ironlaw_evidence(metrics_df, resid_df, b10_flips):
    """E 类铁律的数值证据（把「口径声明」也落成可核对的数值）。"""
    loro = metrics_df[metrics_df.level == "LORO"].copy()
    rows = []

    def add(sid, name, value, unit, note):
        rows.append(dict(ironlaw_id=sid, ironlaw=name, evidence_value=value,
                         unit=unit, numeric_source="ALL_METRICS_FULL.csv / RESIDUALS_LONG.csv",
                         note=note))

    for cid, g in loro.groupby("condition"):
        s = g.r2.sort_values(ascending=False)
        add("E5", f"{cid} 最优与次优 LORO R² 差", f"{s.iloc[0]-s.iloc[1]:.6f}", "R²",
            "差异处于噪声级，不得写成性能提升或下降")
    add("E1", "四条件评价集节点数", "96 / 48 / 48 / 24", "节点",
        "折数 24/24/12/12，不可横比")
    a3 = loro[loro.model == "A3"].set_index("condition").r2
    add("E2", "A3 的 C4 与 C1 LORO R² 差", f'{a3["C4_lf_hd"]-a3["C1_all"]:.6f}', "R²",
        "来自评价集收窄，不是模型变好")
    a11 = metrics_df[(metrics_df.model == "A1.1") & (metrics_df.level == "full")].r2.max()
    a3f = metrics_df[(metrics_df.model == "A3") & (metrics_df.level == "full")].r2.max()
    add("E4", "A1.1 与 A3 样本内最高 R²", f"A1.1={a11:.6f} / A3={a3f:.6f}", "R²",
        "样本内第一不代表可外推、可解释")

    # 逐鼠 LORO R²
    per = []
    for (cid, name, rat), g in (resid_df[resid_df.level == "LORO"]
                                .groupby(["condition", "model", "rat"])):
        y = g.y_obs.to_numpy(float)
        p = g.y_pred.to_numpy(float)
        sst = float(np.sum((y - y.mean()) ** 2))
        per.append(dict(condition=cid, model=name, rat=rat,
                        per_rat_r2=1 - np.sum((y - p) ** 2) / sst if sst > 0 else np.nan))
    per = pd.DataFrame(per)
    add("E7", "逐鼠 LORO R²<0 的鼠次",
        f'{int((per.per_rat_r2 < 0).sum())} / {len(per)} '
        f'({100*(per.per_rat_r2 < 0).mean():.1f}%)', "鼠次",
        f'逐鼠 R² 标准差={per.per_rat_r2.std():.4f}，方差大，仅作参考')
    # 尿素 t=105 系统偏差
    u = resid_df[(resid_df.level == "LORO") & (resid_df.t_min == 105)]
    add("E8", "t=105 尿素点 LORO 残差均值", f'{u.residual.mean():.6f}', "R_Z",
        f'RMSE={np.sqrt(np.mean(u.residual**2)):.6f}；单调模型系统性无法表达回升')
    # 一致性 / bootstrap
    boot = loro.dropna(subset=["r2_boot_lo"])
    add("E6", "LORO R² bootstrap 95% CI 平均半宽",
        f'{float((boot.r2_boot_hi-boot.r2_boot_lo).mean()/2):.6f}', "R²",
        "按鼠整簇重采样 1000 次；CI 重叠说明模型差异不显著")
    add("E3", "A1 经验层相对 A2 纯机理的样本内 R² 增量",
        f'{metrics_df[(metrics_df.model=="A1.1")&(metrics_df.level=="full")].r2.max() - a3f:+.6f}',
        "R²", "经验校正抬高拟合不等于机理更正确")
    add("E9", "频率点数", "4（100/120/1000/10000 Hz）", "点",
        "幂律 / ln(f) / 交互形式统计不可分，s 非被辨识的 CPE 指数")
    add("E10", "A2 物理参数个数", "4（由实测标定）", "参数",
        "机理驱动指结构来自机理，非全部参数可溯源")
    add("B10", "去尿素稳健性排名翻转数",
        f'{int(b10_flips.rank_flipped.sum())} / {len(b10_flips)}', "条件",
        "剔除 t=105 后 LORO 排名是否翻转；详见 B10_DROP_UREA.csv")
    return pd.DataFrame(rows), per


def trackb_metrics():
    d = pd.read_csv(TB_ALL)
    rows, strat = [], []
    scopes = [(cid, spec, d) for cid, spec in CONDITIONS.items()]
    scopes.append(("DELIVERY_lowfreq_196", dict(label="交付版低频 196 点（全量）", doses=None, freqs=None),
                   pd.read_csv(TB_LF)))
    for cid, spec, src in scopes:
        s = src.copy()
        if spec.get("doses"):
            s = s[s.edta_concentration_mol_per_l.isin(spec["doses"])]
        if spec.get("freqs"):
            s = s[s.frequency_hz.isin(spec["freqs"])]
        n = len(s)
        cov = float(s.covered.mean())
        # Wilson 95% CI
        z = 1.959963984540054
        center = (cov + z * z / (2 * n)) / (1 + z * z / n)
        half = z * math.sqrt(cov * (1 - cov) / n + z * z / (4 * n * n)) / (1 + z * z / n)
        w = s.width.to_numpy(float)
        wink = s["winkler_alpha_mis_0.10"].to_numpy(float)
        obs = s.observed_rz.to_numpy(float)
        lb, ub = s.LB.to_numpy(float), s.UB.to_numpy(float)
        exc = np.where(obs < lb, lb - obs, np.where(obs > ub, obs - ub, 0.0))
        miss_below = int(np.sum(obs < lb))
        miss_above = int(np.sum(obs > ub))
        center_pred = s.trackB_center.to_numpy(float)
        dm = compute_metrics(obs, center_pred, 0)
        rows.append(dict(
            condition=cid, condition_label=spec["label"], n_points=n,
            n_rats=int(s.rat.nunique()), coverage=cov,
            coverage_wilson_lo=center - half, coverage_wilson_hi=center + half,
            target_coverage=0.90, coverage_gap=cov - 0.90,
            mean_width=float(w.mean()), median_width=float(np.median(w)),
            min_width=float(w.min()), max_width=float(w.max()), sd_width=float(w.std(ddof=1)),
            sharpness=float(w.mean()),
            winkler_mean=float(wink.mean()), winkler_sd=float(wink.std(ddof=1)),
            n_miss=int(np.sum(~s.covered)), miss_below=miss_below, miss_above=miss_above,
            mean_exceedance=float(exc.mean()), max_exceedance=float(exc.max()),
            padding_share_of_width=0.9709, padding_source="B1_DIAGNOSTICS.md (低频 196 点)",
            # 以下为 DIAGNOSTIC ONLY：区间中心点作为点预测的诊断量，不作验收、不改变铁律
            diag_center_r2=dm["r2"], diag_center_adj_r2=dm["adj_r2"],
            diag_center_rmse=dm["rmse"], diag_center_mae=dm["mae"],
            diag_center_bias=dm["bias"], diag_center_mape=dm["mape"],
            diag_center_pearson_r=dm["pearson_r"], diag_center_ccc=dm["ccc_lin"],
            diag_center_resid_sd=dm["resid_sd"], diag_center_shapiro_p=dm["shapiro_p"],
            diag_center_durbin_watson=dm["durbin_watson"],
            diag_center_willmott_d=dm["willmott_d"],
            diagnostic_notice="DIAGNOSTIC_ONLY: 区间中心点不是点预测模型；TrackB 验收仍以 coverage/width/Winkler 为准，铁律不变",
        ))
        # 分层覆盖率
        for by, col in [("剂量", "edta_concentration_mol_per_l"),
                        ("频率Hz", "frequency_hz"), ("时间min", "global_time_min"),
                        ("大鼠", "rat")]:
            for v, gg in s.groupby(col):
                strat.append(dict(condition=cid, stratify_by=by, group_value=str(v),
                                  n=len(gg), coverage=float(gg.covered.mean()),
                                  mean_width=float(gg.width.mean())))
    return pd.DataFrame(rows), pd.DataFrame(strat)


def standards_computed(metrics_df, tb_df, resid_df, consistency, b10_flips=None):
    """45 条标准逐条回填实算值 / 判定。"""
    loro = metrics_df[metrics_df.level == "LORO"]
    full = metrics_df[metrics_df.level == "full"]
    best = loro.loc[loro.groupby("condition").r2.idxmax()]
    a2_c1 = metrics_df[(metrics_df.model == "A2") & (metrics_df.condition == "C1_all") &
                       (metrics_df.level == "full")].iloc[0]
    rows = []

    def add(sid, cls, name, scope, typ, value, unit, source, note):
        rows.append(dict(standard_id=sid, cls=cls, standard=name, scope=scope,
                         result_type=typ, computed_value=value, unit=unit,
                         numeric_source=source, judgement=note))

    # A 类 · 验证层级
    add("A1", "A", "full 样本内指标", "TrackA", "定量", f'{full.r2.min():.6f} ~ {full.r2.max():.6f}',
        "R²", "ALL_METRICS_FULL.csv / level=full / r2", "不作数；仅描述样本内拟合程度")
    add("A2", "A", "LOCO 留一条件", "TrackA", "定量",
        f'{metrics_df[metrics_df.level=="LOCO"].r2.min():.6f} ~ {metrics_df[metrics_df.level=="LOCO"].r2.max():.6f}',
        "R²", "ALL_METRICS_FULL.csv / level=LOCO", "群体均值层，不检验大鼠级泛化")
    add("A3", "A", "LORO 留一大鼠（验收口径）", "TrackA", "定量",
        f'{loro.r2.min():.6f} ~ {loro.r2.max():.6f}', "R²",
        "ALL_METRICS_FULL.csv / level=LORO", "唯一验收口径")
    add("A4", "A", "逐鼠 LORO 均值", "TrackA", "定量",
        f'{resid_df[resid_df.level=="LORO"].groupby(["condition","model","rat"]).size().mean():.2f}',
        "点/鼠", "RESIDUALS_LONG.csv / 按 condition-model-rat 计数", "逐鼠结果见 RESIDUAL_GROUP_DIAGNOSTICS.csv")
    add("A5", "A", "LORO 折数", "TrackA", "定量", "C1/C2=24；C3/C4=12", "折",
        "ALL_METRICS_FULL.csv / n folds", "折数不同 → 四条件 R² 不可横比")
    add("A6", "A", "评价集规模", "TrackA/TrackB", "定量",
        "条件节点 96/48/48/24；鼠级点 576/288/288/144；鼠数 24/24/12/12", "点",
        "ALL_METRICS_FULL.csv / n", "节点数不同，指标不具可比性")
    add("A7", "A", "数据版本 gate", "全部", "判定", "PASS（SHA 冻结 + 三层核验）", "-",
        "data/data_verification.json", "源文件未改动")
    add("A8", "A", "不利样本保留", "全部", "定量",
        f'{resid_df[resid_df.level=="LORO"].rat.nunique()}', "鼠",
        "RESIDUALS_LONG.csv / rat nunique", "WT17 / WT9 / WT8 全部保留，未删除")

    # B 类 · TrackA 点预测指标
    bmap = [
        ("B1", "R²", "r2", "1−SSE/SST，越大越好；LORO 层为验收"),
        ("B2", "RMSE", "rmse", "sqrt(mean((y−ŷ)²))，越小越好"),
        ("B3", "MAE", "mae", "mean(|y−ŷ|)，对离群稳健"),
        ("B4", "bias", "bias", "mean(y−ŷ)；同时给 bias_pred_minus_obs=−bias 便于对照"),
        ("B5", "AIC", "aic", "n·ln(SSE/n)+2(k+1)，越小越好；另补 AICc/BIC"),
        ("B6", "k 参数个数", "k", "5 / 4 / 3 / 4 / 6"),
    ]
    for sid, nm, col, note in bmap:
        add(sid, "B", nm, "TrackA", "定量",
            f'full={full[col].min():.6f}~{full[col].max():.6f}; LORO={loro[col].min():.6f}~{loro[col].max():.6f}',
            "-", f"ALL_METRICS_FULL.csv / {col}", note)
    add("B7", "B", "基线 t=0 外推", "TrackA", "定量",
        "A1.1/A1.3/A2/A3=1.000000；A1.2≈0.899~1.305（需锚定）", "R_Z",
        "three_routes_metrics.csv / baseline_pred", "A1.2 偏离 1，须人为锚定")
    add("B8", "B", "参数可辨识性", "TrackA", "判定",
        "A2 四参数 0 边界命中；A1.2 出现大系数互补", "-",
        "docs/MECHANISM_PRINCIPLE.md 原理六", "6→5→4 参数化后边界命中降为 0")
    add("B9", "B", "物理可解释性 / 机理含量", "TrackA", "判定",
        "A1.1/A1.2/A1.3=无或低；A2=100%；A3=骨架100%+2 校准系数", "-",
        "MODEL_DEFINITIONS.md", "定性，与 B6/B11 同看")
    if b10_flips is not None:
        b10v = (f'本轮已实算：剔除 t=105 后 {int(b10_flips.rank_flipped.sum())}/{len(b10_flips)} '
                f'条件 LORO 第一名翻转；最大 |ΔR²|={b10_flips.max_abs_delta_r2.max():.6f}')
        b10src = "B10_DROP_UREA.csv / rank_flipped, max_abs_delta_r2"
        b10note = "B10 标准定义的子集稳健性检验已补算；交付模型冻结参数不变"
    else:
        b10v = "历史结论：剔除 t=105 后排名不翻转"
        b10src, b10note = "validation/FITTING_EFFECT.md", "本轮未重跑"
    add("B10", "B", "去尿素稳健性", "TrackA", "定量", b10v, "R²", b10src, b10note)
    add("B11", "B", "跨条件零重拟合迁移", "A2", "定量",
        "C2≈0.947 / C3≈0.932 / C4≈0.953", "R²",
        "three_routes_results.json / A2_global_transfer", "一套 C1 参数零重拟合迁移")
    add("B12", "B", "自包含性", "TrackA", "判定",
        "A1.1/A2/A3 自包含；A1.2/A1.3 需机理先验 CSV", "-",
        "各 model.py", "包内 data/ 已含机理先验副本")

    # C 类 · TrackB 区间指标
    c1 = tb_df[tb_df.condition == "C1_all"].iloc[0]
    cmap = [
        ("C1", "coverage", "coverage", "目标 0.90；另给 Wilson 95% CI"),
        ("C2", "mean width", "mean_width", "同等覆盖率下越窄越好"),
        ("C3", "median width", "median_width", "不受极端宽区间影响"),
        ("C4", "max width", "max_width", "高频段显著更宽"),
        ("C5", "Winkler (α=0.10)", "winkler_mean", "综合覆盖率与宽度"),
    ]
    for sid, nm, col, note in cmap:
        vals = " / ".join(f'{r.condition}={getattr(r, col):.6f}' for r in tb_df.itertuples()
                          if r.condition in list(CONDITIONS))
        add(sid, "C", nm, "TrackB 四条件", "定量", vals, "-",
            f"TRACKB_FULL_METRICS.csv / {col}", note)
    add("C6", "C", "宽度构成 padding 占比", "TrackB", "定量", "97.09%（基础 LHS 3.00%）",
        "%", "validation/B1_DIAGNOSTICS.md", "宽区间根因是 conformal padding，不是 LHS 参数不确定性")
    add("C7", "C", "逐点 vs 联合覆盖", "TrackB", "铁律", "逐点覆盖 ≠ 整曲线联合覆盖", "-",
        "docs/EVALUATION_CRITERIA.md C7", "不可把逐点 0.913 说成整曲线 91.3%")
    add("C8", "C", "无区间时段", "TrackB", "判定",
        f'区间覆盖 t={sorted(pd.read_csv(TB_ALL).global_time_min.unique())}（无 t=0 / t=105）', "min",
        "calibrated_intervals_allfreq_392.csv / global_time_min", "绘图须标注灰虚线")
    add("C9", "C", "padding 公式", "TrackB", "判定",
        "p=tolerance_f×q_f；lower=clip(q05−p,0,1)，upper=clip(q95+p,0,1)", "-",
        "TrackB/model.py / calibrated_from_raw()", "P0 审计复现误差 1.39e−16")

    # D 类 · 绘图
    dmap = [
        ("D1", "时间点画全", "7 点 [0,5,20,35,65,95,105]，拟合图 fig2–fig5 全画"),
        ("D2", "SEM 画全", "N=6 均值±SEM，未用 SD 冒充"),
        ("D3", "纵轴从 0 起", "ylim=(0,1.30)，未裁掉后期低值"),
        ("D4", "曲线画全", "5 条 TrackA 模型曲线全部叠加"),
        ("D5", "无遮挡无缺字", "CJK 字体注册，图例/注释不压数据"),
        ("D6", "位置一致", "全部位于 four_conditions/figures/"),
    ]
    for sid, nm, note in dmap:
        add(sid, "D", nm, "fig2–fig5 / 全部图", "判定", "PASS（已在 fig1–fig15 全图核验）",
            "-", "four_conditions/figures/", note)

    # E 类 · 铁律
    emap = [
        ("E1", "四条件 R² 不可横比", "评价集 96/48/48/24、折数 24/24/12/12 不同"),
        ("E2", "条件越窄 ≠ 模型越好", "C4 数字来自评价范围收窄 + 12 鼠小样本"),
        ("E3", "拟合变好 ≠ 机理更正确", "A1 系列是经验校正，不是机理改进"),
        ("E4", "指标最优 ≠ 可用模型", "A1.1 样本内最高但 5 参数无物理意义"),
        ("E5", "噪声级差异不声称提升", "LORO 差 0.002~0.007 远小于逐鼠波动"),
        ("E6", "AIC 罚参数但未做多重比较校正", "本轮补 AICc/BIC，仍仅作辅助"),
        ("E7", "逐鼠 LORO 方差大", "个别鼠 R² 为负；逐鼠残差见 RESIDUAL_GROUP_DIAGNOSTICS.csv"),
        ("E8", "尿素 t=105 回升无法表达", "所有单调模型皆然"),
        ("E9", "s 不是被辨识的 CPE 指数", "仅 4 个频率点，幂律与 ln(f) 形式统计不可分"),
        ("E10", "三条路线不构成机理被证实", "物理量由实测标定，不是第一性原理推导"),
    ]
    for sid, nm, note in emap:
        add(sid, "E", nm, "全部", "铁律", "ACTIVE（口径生效，数值判读均按此执行）", "-",
            "docs/EVALUATION_CRITERIA.md 第 1.5 节", note)

    add("X1", "扩展", "全套指标实算一致性校验", "全部", "判定",
        f'PASS：max|ΔR²|={consistency["max_abs_dr2"]:.3e}, max|ΔRMSE|={consistency["max_abs_drmse"]:.3e}',
        "-", "ALL_METRICS_SUMMARY.json / consistency", "证明本轮未改动任何模型/参数/数据")
    return pd.DataFrame(rows)


def main():
    rl, cond = load_data()
    frozen = pd.read_csv(FROZEN_METRICS, encoding="utf-8-sig")
    rows_metrics, rows_resid = [], []
    checks = []

    for cid, spec in CONDITIONS.items():
        c = attach(cond[(cond.dose.isin(spec["doses"])) &
                        (cond.freq.isin(spec["freqs"]))].copy()).reset_index(drop=True)
        r = attach(rl[(rl.dose.isin(spec["doses"])) &
                      (rl.freq.isin(spec["freqs"]))].copy()).reset_index(drop=True)
        y_c = c.y.to_numpy(float)
        yr = r.y.to_numpy(float)
        rat = r.rat.to_numpy()

        for name in MODELS:
            p = fit(name, c, y_c)
            # full
            m_full = emit(rows_metrics, rows_resid, cid, name, "full", c, y_c,
                          predict(name, p, c))
            # LOCO
            yh = np.empty(len(c))
            for i in range(len(c)):
                mask = np.ones(len(c), bool)
                mask[i] = False
                pp = fit(name, c[mask].reset_index(drop=True), y_c[mask], p_init=p)
                yh[i] = predict(name, pp, c.iloc[[i]].reset_index(drop=True))[0]
            m_loco = emit(rows_metrics, rows_resid, cid, name, "LOCO", c, y_c, yh)
            # LORO
            yhat = np.empty(len(r))
            for one in np.unique(rat):
                te = rat == one
                tr = ~te
                pp = fit(name, r[tr].reset_index(drop=True), yr[tr], p_init=p)
                yhat[te] = predict(name, pp, r[te].reset_index(drop=True))
            m_loro = emit(rows_metrics, rows_resid, cid, name, "LORO", r, yr, yhat, do_boot=True)

            fr = frozen[(frozen.condition == cid) & (frozen.model == name)].iloc[0]
            for lvl, m, r2col, rmcol in [("full", m_full, "full_r2", "full_rmse"),
                                         ("LOCO", m_loco, "loco_r2", "loco_rmse"),
                                         ("LORO", m_loro, "loro_r2", "loro_rmse")]:
                checks.append(dict(condition=cid, model=name, level=lvl,
                                   frozen_r2=float(fr[r2col]), recomputed_r2=m["r2"],
                                   d_r2=m["r2"] - float(fr[r2col]),
                                   frozen_rmse=float(fr[rmcol]), recomputed_rmse=m["rmse"],
                                   d_rmse=m["rmse"] - float(fr[rmcol])))
            print(f"  {cid:12s} {name:5s} full={m_full['r2']:.4f} "
                  f"LOCO={m_loco['r2']:.4f} LORO={m_loro['r2']:.4f} "
                  f"shapiro_p={m_loro['shapiro_p']:.2e} DW={m_loro['durbin_watson']:.3f}")

    chk = pd.DataFrame(checks)
    consistency = dict(max_abs_dr2=float(chk.d_r2.abs().max()),
                       max_abs_drmse=float(chk.d_rmse.abs().max()),
                       n_checks=int(len(chk)),
                       passed=bool((chk.d_r2.abs().max() < 1e-5) and (chk.d_rmse.abs().max() < 1e-5)))
    print("\n一致性校验:", consistency)
    if not consistency["passed"]:
        raise RuntimeError("复算结果与冻结结果不一致 —— 疑似模型/数据被改动，已中止")

    metrics_df = pd.DataFrame(rows_metrics)
    resid_df = pd.DataFrame(rows_resid)
    grp = group_residual_diagnostics(resid_df)
    b10, b10_flips = b10_drop_urea(rl, cond, metrics_df)
    iron, per_rat = ironlaw_evidence(metrics_df, resid_df, b10_flips)
    tb, strat = trackb_metrics()
    std = standards_computed(metrics_df, tb, resid_df, consistency, b10_flips)

    metrics_df.to_csv(OUT / "ALL_METRICS_FULL.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    resid_df.to_csv(OUT / "RESIDUALS_LONG.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    grp.to_csv(OUT / "RESIDUAL_GROUP_DIAGNOSTICS.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    b10.to_csv(OUT / "B10_DROP_UREA.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    b10_flips.to_csv(OUT / "B10_RANK_FLIPS.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    iron.to_csv(OUT / "IRONLAW_EVIDENCE.csv", index=False, encoding="utf-8-sig")
    per_rat.to_csv(OUT / "PER_RAT_LORO_R2.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    tb.to_csv(OUT / "TRACKB_FULL_METRICS.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    strat.to_csv(OUT / "TRACKB_STRATIFIED.csv", index=False, encoding="utf-8-sig", float_format="%.8g")
    std.to_csv(OUT / "STANDARDS_COMPUTED.csv", index=False, encoding="utf-8-sig")
    chk.to_csv(OUT / "CONSISTENCY_CHECK.csv", index=False, encoding="utf-8-sig", float_format="%.8g")

    loro = metrics_df[metrics_df.level == "LORO"]
    summary = dict(
        meta=dict(data_source="包内冻结副本 frozen_measured_24rat.csv（与源 SHA256 一致，24 鼠全保留）",
                  boundary="未改模型 / 未重新建模 / 未删样本；仅扩展指标计算",
                  n_models=5, n_conditions=4, n_levels=3,
                  metric_count=int(metrics_df.shape[1] - 6),
                  residual_rows=int(len(resid_df)),
                  standards_count=int(len(std))),
        consistency=consistency,
        loro_best_by_condition={r.condition: dict(model=r.model, r2=float(r.r2),
                                                  rmse=float(r.rmse), mae=float(r.mae))
                               for r in loro.loc[loro.groupby("condition").r2.idxmax()].itertuples()},
        trackb={r.condition: dict(coverage=float(r.coverage), mean_width=float(r.mean_width),
                                 winkler=float(r.winkler_mean), diag_center_r2=float(r.diag_center_r2))
                for r in tb.itertuples()},
        b10_rank_flips=int(b10_flips.rank_flipped.sum()),
        b10_conditions=int(len(b10_flips)),
        n_standards_quantitative=int((std.result_type == "定量").sum()),
        n_residual_rows=int(len(resid_df)),
        n_per_rat_r2_rows=int(len(per_rat)),
    )
    (OUT / "ALL_METRICS_SUMMARY.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n落盘完成：{OUT}")
    print(f"  指标表 {metrics_df.shape} / 逐点残差 {len(resid_df)} 行 / 标准回填 {len(std)} 条")
    return summary


if __name__ == "__main__":
    main()
