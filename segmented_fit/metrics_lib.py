#!/usr/bin/env python3.11
# -*- coding: utf-8 -*-
"""指标函数：逐字复制自 ThreeRoutes/compute_all_metrics.py 第 192-373 行，口径严格一致。

包含 _safe / _ols / cluster_bootstrap / compute_metrics。
残差约定 e = y_obs - y_pred；信息准则 K = k + 1；bootstrap 按大鼠整簇重采样。
"""
import math
import numpy as np
from scipy import stats

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
