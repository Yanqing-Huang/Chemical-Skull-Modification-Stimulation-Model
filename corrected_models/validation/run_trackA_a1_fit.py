#!/usr/bin/env python3.11
"""
TrackA A1 empirical linear correction fitting effect.

Primary deliverable for TrackA A1. Reads the TrackA point-metrics file
(low-frequency 100/120 Hz nodes), fits R_Z_corr = a + b * R_Z_raw, and
reports an HONEST (leave-one-out) and a leaky (full-OLS) fitting effect,
plus a constant-offset diagnostic, with a per-frequency breakdown.

Outputs:
  - trackA_A1_metrics.csv  (one row per scope x method)
  - trackA_A1_summary.json (machine-readable numbers for the report)
"""
import json
import hashlib
import numpy as np
import pandas as pd
from scipy import stats

OUTDIR = "/workspace/Chemical_Skull_Model_TrackA_A1_Fit"
SRC = "/workspace/Chemical_Skull_Model_LowFreq_100_120Hz/data/trackA_point_metrics.csv"
LOW_FREQS = [100, 120]


def metrics(y, x):
    y = np.asarray(y, float)
    x = np.asarray(x, float)
    n = len(y)
    ss_res = float(np.sum((y - x) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = (1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    rmse = float(np.sqrt(np.mean((y - x) ** 2)))
    mae = float(np.mean(np.abs(y - x)))
    bias = float(np.mean(y - x))
    pearson = float(np.corrcoef(y, x)[0, 1]) if np.std(y) > 0 and np.std(x) > 0 else float("nan")
    spearman = float(stats.spearmanr(y, x).correlation)
    return dict(n=n, r2=r2, rmse=rmse, mae=mae, bias=bias,
                pearson_r=pearson, spearman_rho=spearman)


def fit_ols(x, y):
    """Return (a, b) for y = a + b*x via least squares."""
    X = np.column_stack([np.ones_like(x), x])
    coef, _res, _rank, _sv = np.linalg.lstsq(X, y, rcond=None)
    a, b = coef
    return float(a), float(b)


def a1_loo(x, y):
    """Leave-one-out linear fit. Returns (yhat_loo array, fold a[], fold b[])."""
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    n = len(x)
    yhat = np.empty(n, dtype=float)
    a_folds, b_folds = [], []
    for i in range(n):
        mask = np.ones(n, dtype=bool)
        mask[i] = False
        a, b = fit_ols(x[mask], y[mask])
        a_folds.append(a)
        b_folds.append(b)
        yhat[i] = a + b * x[i]
    return yhat, a_folds, b_folds


def main():
    df = pd.read_csv(SRC)
    lf = df[df["frequency_hz"].isin(LOW_FREQS)].copy()
    assert len(lf) == 48, f"expected 48 low-freq nodes, got {len(lf)}"

    y = lf["observed_rz_N6_mean"].to_numpy(float)
    x = lf["trackA_q50"].to_numpy(float)

    rows = []
    summary = {}

    # ---- A0 raw (no correction) ----
    m_a0 = metrics(y, x)
    rows.append(dict(scope="all_48", method="A0_raw", **m_a0))

    # ---- A1 full OLS (FLAGGED LEAKY: fit+eval on same 48) ----
    a_full, b_full = fit_ols(x, y)
    yhat_full = a_full + b_full * x
    m_a1full = metrics(y, yhat_full)
    rows.append(dict(scope="all_48", method="A1_full_OLS_LEAKY", **m_a1full))

    # ---- A1 LOO (HONEST) ----
    yhat_loo, a_folds, b_folds = a1_loo(x, y)
    m_a1loo = metrics(y, yhat_loo)
    rows.append(dict(scope="all_48", method="A1_LOO", **m_a1loo))

    # ---- constant offset diagnostic ----
    c = float(np.mean(y - x))  # c = mean(y - x)
    m_const = metrics(y, x + c)
    rows.append(dict(scope="all_48", method="A0_const_offset", **m_const))

    summary["all_48"] = dict(
        A0_raw=m_a0, A1_full_OLS=dict(a=a_full, b=b_full, **m_a1full),
        A1_LOO=dict(a_mean=float(np.mean(a_folds)), b_mean=float(np.mean(b_folds)),
                    a_std=float(np.std(a_folds)), b_std=float(np.std(b_folds)), **m_a1loo),
        A0_const_offset=dict(c=c, **m_const),
    )

    # ---- per-frequency breakdown (A0 and A1-LOO only) ----
    for f in LOW_FREQS:
        sub = lf[lf["frequency_hz"] == f]
        yf = sub["observed_rz_N6_mean"].to_numpy(float)
        xf = sub["trackA_q50"].to_numpy(float)
        mf_a0 = metrics(yf, xf)
        rows.append(dict(scope=f"freq_{f}", method="A0_raw", **mf_a0))
        yhat_f, af, bf = a1_loo(xf, yf)
        mf_loo = metrics(yf, yhat_f)
        rows.append(dict(scope=f"freq_{f}", method="A1_LOO", **mf_loo))
        summary[f"freq_{f}"] = dict(
            A0_raw=mf_a0,
            A1_LOO=dict(a_mean=float(np.mean(af)), b_mean=float(np.mean(bf)), **mf_loo),
        )

    out = pd.DataFrame(rows, columns=["scope", "method", "n", "r2", "rmse", "mae",
                                      "bias", "pearson_r", "spearman_rho"])
    out_path = f"{OUTDIR}/trackA_A1_metrics.csv"
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}")
    print(out.to_string(index=False))

    with open(f"{OUTDIR}/trackA_A1_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"\nwrote {OUTDIR}/trackA_A1_summary.json")

    # sanity echo for report
    print("\n=== KEY NUMBERS (all_48) ===")
    print(f"A0_raw      R2={m_a0['r2']:.4f} RMSE={m_a0['rmse']:.4f} Bias={m_a0['bias']:.4f}")
    print(f"A1_full_OLS R2={m_a1full['r2']:.4f} RMSE={m_a1full['rmse']:.4f} Bias={m_a1full['bias']:.4f}  (a={a_full:.4f}, b={b_full:.4f})")
    print(f"A1_LOO      R2={m_a1loo['r2']:.4f} RMSE={m_a1loo['rmse']:.4f} Bias={m_a1loo['bias']:.4f}")
    print(f"A0_const    R2={m_const['r2']:.4f} RMSE={m_const['rmse']:.4f} Bias={m_const['bias']:.4f}  (c={c:.4f})")


if __name__ == "__main__":
    main()
