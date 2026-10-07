"""Registered seed-level decisions (A2's, imported unchanged) and the calibration-only segmentation.

window() is A2's rule (analysis/drive_cifar_c_0920/stats.py) with the number of epoch points as a
parameter: 80 here (A2: 400).  BIC0 = n log(RSS0/n) + log n, BIC1 = n log(RSS1/n) + 3 log n; ties go to
the simpler model and to the smallest break; a constant series is the simple model; RSS1 = 0 alone is
the two-level model.  Identified only if BIC1 < BIC0, early < 0 and early < late.
"""
import math

import numpy as np

from analysis.drive_cifar_c_0920.stats import directional, certificate_status  # noqa: F401

EPOCH_POINTS = 80
UPDATES_PER_EPOCH = 75


def window(v, n=EPOCH_POINTS):
    v = np.asarray(v, float)
    assert v.shape == (n,) and np.isfinite(v).all()
    rss0 = float(((v - v.mean()) ** 2).sum())
    if (v == v[0]).all():
        return dict(label='WINDOW_NOT_IDENTIFIED', reason='constant', break_epoch=None, boundary_updates=None)
    candidates = []
    for b in range(1, n):
        early, late = float(v[:b].mean()), float(v[b:].mean())
        rss = float(((v[:b] - early) ** 2).sum() + ((v[b:] - late) ** 2).sum())
        candidates.append((rss, b, early, late))
    rss1, b, early, late = min(candidates)
    bic0 = n * math.log(rss0 / n) + math.log(n)
    bic1 = n * math.log(rss1 / n) + 3 * math.log(n) if rss1 else -math.inf
    identified = bic1 < bic0 and early < 0 and early < late
    return dict(label='WINDOW_IDENTIFIED' if identified else 'WINDOW_NOT_IDENTIFIED',
                break_epoch=b if identified else None,
                boundary_updates=UPDATES_PER_EPOCH * b if identified else None,
                rss0=rss0, rss1=rss1, bic0=bic0, bic1=bic1, early=early, late=late, n=n,
                reason='operational segmentation; not a significance test')
