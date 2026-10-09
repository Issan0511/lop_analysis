"""Small stand-ins for the few scipy.stats functions the aggregation needs (the shared venv has
no scipy; stdlib statistics.NormalDist gives the normal quantile)."""
from __future__ import annotations

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

# two-sided 95% t quantiles t_{0.975, df}
_T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262,
         10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131, 16: 2.120, 17: 2.110,
         18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042, 40: 2.021, 60: 2.000, 120: 1.980}


def t975(df: int) -> float:
    if df in _T975:
        return _T975[df]
    keys = sorted(_T975)
    for k in keys:
        if k > df:
            return _T975[k]
    return 1.96


def norm_ppf(p: float) -> float:
    return NormalDist().inv_cdf(p)


def spearman(a, b) -> float:
    ra = pd.Series(np.asarray(a, float)).rank().to_numpy()
    rb = pd.Series(np.asarray(b, float)).rank().to_numpy()
    if ra.std() == 0 or rb.std() == 0:
        return float("nan")
    return float(np.corrcoef(ra, rb)[0, 1])


def auc(score, label) -> float:
    """P(score_pos > score_neg) + 0.5 P(tie)  (Mann-Whitney U / (n_pos n_neg))."""
    score = np.asarray(score, float); label = np.asarray(label, bool)
    n1, n0 = label.sum(), (~label).sum()
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = pd.Series(score).rank().to_numpy()
    u = r[label].sum() - n1 * (n1 + 1) / 2
    return float(u / (n1 * n0))
