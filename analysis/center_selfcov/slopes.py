"""傾き・到達時刻の推定ユーティリティ (spec_center_selfcov_0814 §2.3)。

  from analysis.center_selfcov.slopes import slope_ols, t50_reach, boot_ci, paired_boot_ci

- slope_ols: 前半区間 [0, t_half] の OLS 傾き
- t50_reach: y が (初期値 + 最終値)/2 を最初に超える step の線形内挿
- boot_ci / paired_boot_ci: seed 間 bootstrap (既定 n=2000) の 95%CI
"""
import numpy as np

N_BOOT = 2000


def slope_ols(x, y, t_half):
    """[0, t_half] の OLS 傾き (単位: 指標/step)。有効点が2未満なら NaN。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y) & (x <= t_half)
    if m.sum() < 2:
        return np.nan
    xs, ys = x[m], y[m]
    xc = xs - xs.mean()
    den = (xc ** 2).sum()
    return float((xc * (ys - ys.mean())).sum() / den) if den > 0 else np.nan


def t50_reach(x, y):
    """y が y0 + 0.5*(y_end − y0) を最初に超える step (線形内挿)。未到達は NaN。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(y) < 3:
        return np.nan
    y0, y1 = y[0], y[-1]
    if not np.isfinite(y0) or not np.isfinite(y1) or abs(y1 - y0) < 1e-12:
        return np.nan
    thr = y0 + 0.5 * (y1 - y0)
    cross = (y >= thr) if y1 > y0 else (y <= thr)
    if not cross.any():
        return np.nan
    i = int(np.argmax(cross))
    if i == 0:
        return float(x[0])
    x0, x1_, ya, yb = x[i - 1], x[i], y[i - 1], y[i]
    if abs(yb - ya) < 1e-12:
        return float(x1_)
    return float(x0 + (thr - ya) * (x1_ - x0) / (yb - ya))


def boot_ci(vals, rng, n_boot=N_BOOT):
    """1 群の平均と 95%CI。"""
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if len(v) < 2:
        return dict(mean=float(v.mean()) if len(v) else np.nan,
                    lo=np.nan, hi=np.nan, n=len(v))
    bs = rng.choice(len(v), (n_boot, len(v)), replace=True)
    bm = v[bs].mean(axis=1)
    return dict(mean=float(v.mean()), lo=float(np.quantile(bm, 0.025)),
                hi=float(np.quantile(bm, 0.975)), n=len(v))


def paired_boot_ci(a, b, rng, n_boot=N_BOOT):
    """対応のある差 a−b の平均と 95%CI (seed が揃っている前提)。"""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    d = a[m] - b[m]
    if len(d) < 2:
        return dict(mean=float(d.mean()) if len(d) else np.nan,
                    lo=np.nan, hi=np.nan, n=len(d), excl_zero=False)
    bs = rng.choice(len(d), (n_boot, len(d)), replace=True)
    bm = d[bs].mean(axis=1)
    lo, hi = float(np.quantile(bm, 0.025)), float(np.quantile(bm, 0.975))
    return dict(mean=float(d.mean()), lo=lo, hi=hi, n=len(d),
                excl_zero=bool(lo > 0 or hi < 0))
