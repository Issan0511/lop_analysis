# -*- coding: utf-8 -*-
"""drift_law_0907: 前活性 1 次元の drift 則（読み規則 `specs/spec_drift_law_0907.md`）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.drift_law_0907

新しい走は無い。既存ログから、タスク境界のストロボ写像の期待変位

    E[N | ζ]  （N = z̄_{k+1} − z̄_k、ζ = **k を除く** 8 タスクの移動平均）

を、p̂_up（ζ と w_free から作る支持 32 点のうち折れ目より上の割合）の関数として測る。
**current-task の z̄ で条件付けない**——ζ を leave-one-out で作るのがこの解析の要
（そうしないと平均回帰が「力」に化ける）。
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from .common import ROOT, load_config

T = 10_000
WINDOW = (10, 199)                  # タスク番号（spec §3）
HALF = 4                            # 移動平均の片側タスク数
# ζ は N = z̄_{k+1} − z̄_k の **両端（k と k+1）を除く**（spec 追補 1）。k だけ除くと
# ζ に z̄_{k+1} が 1/8 残り、Cov(N, ζ) > 0 の**正の**汚染が入る（対称窓で k を含める版は
# k と k+1 の寄与が打ち消して偶然 0 になる——検査 `test_loo_kills_mean_reversion` が実測）。
EXCLUDE = (0, 1)
NBOOT = 2000
RNG_SEED = 20260907
FLOOR_TOP_BIN = 0.005               # 最上位ビンの E[−N] の床（spec §4）
MIN_BINS, MIN_SPAN = 5, 3.0
SIGNS = np.array(list(itertools.product([-0.5, 0.5], repeat=5)))   # 32 点（半幅 = 0.5Σ|w|）

DIRS = (Path(ROOT) / "results/offset_grid_0906/logs_tail",
        Path(ROOT) / "results/act_offset_0906/logs_tail",
        Path(ROOT) / "results/boundary_dense_0907/logs")
JUDGED = (("LRoff0_1216", 0.1), ("LRa0p2_off0_1216", 0.2), ("LRa0p3_off0_1216", 0.3),
          ("LRa0p5_off0_1216", 0.5), ("LRa0p7_off0_1216", 0.7), ("BDref_1216", 0.1))
AUX = (("LRoffm0p5_1216", 0.1), ("FBLRoff0_1216", 0.1), ("LRvf1_1216", 0.1))


def _find(arm: str, seed: int) -> Path | None:
    for d in DIRS:
        p = d / f"{arm}_seed{seed}.npz"
        if p.exists():
            return p
    return None


def seed_table(arm: str, seed: int) -> dict | None:
    """1 seed 分の (タスク, ユニット) 表: N・ζ・p̂_up・W。"""
    p = _find(arm, seed)
    if p is None:
        return None
    with np.load(p, allow_pickle=True) as z:
        step = z["step"].astype(np.int64)
        zb = z["layer1_zbar"].astype(np.float64)
        wf = z["layer1_w_free"].astype(np.float64)
        ws = z["layer1_w_free_step"].astype(np.int64)
    ends = {int(v): i for i, v in enumerate(step) if int(v) % T == 0}
    wend = {int(v): i for i, v in enumerate(ws)}
    lo, hi = WINDOW
    ks = [k for k in range(lo, hi + 1)
          if all((k + d) * T in ends for d in range(-HALF, HALF + 2)) and k * T in wend]
    # 近傍は k−HALF..k+HALF+1 の 2·HALF+2 点から k, k+1 を除いた 2·HALF 点
    if not ks:
        return None
    N, Z, P, W = [], [], [], []
    for k in ks:
        zk = zb[ends[k * T]]
        N.append(zb[ends[(k + 1) * T]] - zk)
        nb = [zb[ends[(k + d) * T]] for d in range(-HALF, HALF + 2)
              if d not in EXCLUDE]
        zeta = np.mean(nb, axis=0)
        Z.append(zeta)
        w = wf[wend[k * T]]                                   # (unit, 5)
        pts = zeta[:, None] + np.einsum("kj,hj->hk", SIGNS, w)  # (unit, 32)
        P.append((pts > 0).mean(axis=1))
        W.append(0.5 * np.abs(w).sum(axis=1))
    f = lambda x: np.asarray(x, dtype=np.float64)
    return dict(N=f(N), zeta=f(Z), p_up=f(P), W=f(W), n_task=len(ks))


def arm_table(arm: str) -> list[dict]:
    out = []
    for s in range(10):
        t = seed_table(arm, s)
        if t is not None:
            out.append(t)
    return out


def deciles(p: np.ndarray, n: int = 10) -> np.ndarray:
    """量子化された p̂_up を跨ぐビン境界（spec §6-4）。"""
    q = np.unique(np.quantile(p, np.linspace(0, 1, n + 1)))
    return q


def bin_stats(tabs: list[dict], a: float) -> list[dict]:
    """腕の十分位ビンごとの (p̂_up 中央値, E[−N])。seed をプール。"""
    p = np.concatenate([t["p_up"].ravel() for t in tabs])
    n = np.concatenate([t["N"].ravel() for t in tabs])
    m = np.isfinite(p) & np.isfinite(n)
    p, n = p[m], n[m]
    edges = deciles(p)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (p >= lo) & (p < hi) if hi < edges[-1] else (p >= lo) & (p <= hi)
        if sel.sum() < 200:
            continue
        rows.append(dict(lo=float(lo), hi=float(hi), n=int(sel.sum()),
                         p_med=float(np.median(p[sel])), mN=float(-np.mean(n[sel])),
                         a=a))
    return rows


def slope_from(rows: list[dict]) -> tuple[float, int, float]:
    """log E[−N/(1−a)] 対 log p̂_up の傾き（腕をプールした点群）。"""
    x = np.array([r["p_med"] for r in rows])
    y = np.array([r["mN"] / (1.0 - r["a"]) for r in rows])
    m = (x > 0) & (y > 0)
    x, y = x[m], y[m]
    if x.size < MIN_BINS:
        return float("nan"), int(x.size), 0.0
    span = float(x.max() / x.min())
    return float(np.polyfit(np.log(x), np.log(y), 1)[0]), int(x.size), span


def analyze(out: Path) -> dict:
    rng = np.random.default_rng(RNG_SEED)
    arms, tabs = {}, {}
    for arm, a in JUDGED + AUX:
        t = arm_table(arm)
        if t:
            tabs[arm] = (t, a)
            arms[arm] = dict(arm=arm, a=a, n_seed=len(t), n_task=t[0]["n_task"])

    # ビン統計（判定腕をプール）
    rows_by_arm = {arm: bin_stats(t, a) for arm, (t, a) in tabs.items()}
    judged_rows = [r for arm, _ in JUDGED if arm in rows_by_arm for r in rows_by_arm[arm]]
    s_pt, n_bin, span = slope_from(judged_rows)

    def boot_slope(idx_by_arm):
        rr = []
        for arm, a in JUDGED:
            if arm not in tabs:
                continue
            t, _ = tabs[arm]
            pick = [t[i] for i in idx_by_arm[arm]]
            rr += bin_stats(pick, a)
        return slope_from(rr)[0]

    draws = []
    for _ in range(NBOOT // 4):
        idx = {arm: rng.integers(0, len(tabs[arm][0]), len(tabs[arm][0]))
               for arm, _ in JUDGED if arm in tabs}
        v = boot_slope(idx)
        if np.isfinite(v):
            draws.append(v)
    s_ci = ([float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]
            if len(draws) >= 100 else [float("nan"), float("nan")])
    top = max((r["mN"] for r in judged_rows), default=0.0)
    nd = (n_bin < MIN_BINS) or (span < MIN_SPAN) or (top < FLOOR_TOP_BIN)
    shape = ("NOT_DETERMINED" if nd or not np.isfinite(s_ci[0]) else
             "PROPORTIONAL" if 0.8 <= s_ci[0] and s_ci[1] <= 1.2 else
             "SUPERLINEAR" if s_ci[0] > 1.2 else
             "NO_DEPENDENCE" if s_ci[0] <= 0 <= s_ci[1] else
             "SUBLINEAR" if s_ci[1] < 0.8 else "OTHER")

    # 第 2 判定: κ = E[−N] / [(1−a)·p̂_up]
    kappa = {}
    for arm, a in JUDGED:
        if arm not in tabs:
            continue
        t, _ = tabs[arm]
        p = np.concatenate([x["p_up"].ravel() for x in t])
        n = np.concatenate([x["N"].ravel() for x in t])
        m = np.isfinite(p) & np.isfinite(n) & (p > 0)
        kappa[arm] = dict(a=a, kappa=float(-np.mean(n[m]) / ((1 - a) * np.mean(p[m]))),
                          mean_p=float(np.mean(p[m])), mean_mN=float(-np.mean(n[m])),
                          n=int(m.sum()))
    ks = [v["kappa"] for v in kappa.values() if np.isfinite(v["kappa"]) and v["kappa"] > 0]
    ratio = float(max(ks) / min(ks)) if len(ks) >= 3 else float("nan")
    collapse = ("NOT_DETERMINED" if len(ks) < 3 else
                "COLLAPSE" if ratio <= 2.0 else "RESIDUAL")

    # 補助 1: p̂_up = 0 群は動くか
    zero = {}
    for arm, a in JUDGED + AUX:
        if arm not in tabs:
            continue
        t, _ = tabs[arm]
        p = np.concatenate([x["p_up"].ravel() for x in t])
        n = np.concatenate([x["N"].ravel() for x in t])
        m = np.isfinite(p) & np.isfinite(n)
        z0 = m & (p == 0.0)
        per_seed = []
        for x in t:
            pp, nn = x["p_up"].ravel(), x["N"].ravel()
            sel = np.isfinite(pp) & np.isfinite(nn) & (pp == 0.0)
            if sel.sum() >= 50:
                per_seed.append(float(np.mean(nn[sel])))
        ci = ([float(np.percentile(np.array([np.mean(np.array(per_seed)[rng.integers(0, len(per_seed), len(per_seed))])
                                             for _ in range(NBOOT)]), q)) for q in (2.5, 97.5)]
              if len(per_seed) >= 3 else [float("nan"), float("nan")])
        zero[arm] = dict(frac=float(z0.mean()), E_N=float(np.mean(n[z0])) if z0.any() else float("nan"),
                         ci=ci, n=int(z0.sum()))

    # 補助 2: W で層別（1 次元に畳めるか）
    strat = {}
    arm0 = "LRoff0_1216"
    if arm0 in tabs:
        t, a = tabs[arm0]
        W = np.concatenate([x["W"].ravel() for x in t])
        p = np.concatenate([x["p_up"].ravel() for x in t])
        n = np.concatenate([x["N"].ravel() for x in t])
        qs = np.quantile(W, [0, 1 / 3, 2 / 3, 1])
        for i, (lo, hi) in enumerate(zip(qs[:-1], qs[1:])):
            sel = (W >= lo) & (W <= hi) if i == 2 else (W >= lo) & (W < hi)
            good = sel & np.isfinite(p) & np.isfinite(n) & (p > 0)
            strat[f"W_{i}"] = dict(W_med=float(np.median(W[sel])), n=int(good.sum()),
                                   kappa=float(-np.mean(n[good]) / ((1 - a) * np.mean(p[good]))))

    res = dict(experiment="drift_law_0907", read_rule="specs/spec_drift_law_0907.md",
               window=list(WINDOW), n_boot=NBOOT, boot_seed=RNG_SEED,
               DRIFT_SHAPE=dict(label=shape, slope=s_pt, slope_ci=s_ci, n_bin=n_bin,
                                span=span, top_bin_mN=top, floor=FLOOR_TOP_BIN),
               A_COLLAPSE=dict(label=collapse, max_over_min=ratio, kappa=kappa),
               zero_straddle=zero, W_stratified=strat,
               bins={arm: rows_by_arm[arm] for arm in rows_by_arm},
               arms=list(arms.values()))
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    L = [f"# drift_law_0907 判定", "",
         f"- **DRIFT_SHAPE = `{shape}`**  傾き s = {s_pt:.3f} [{s_ci[0]:.3f}, {s_ci[1]:.3f}]"
         f"（ビン {n_bin}・p̂_up の幅 {span:.1f} 倍・最上位ビン E[−N] {top:.5f}）",
         f"- **A_COLLAPSE = `{collapse}`**  κ の max/min = {ratio:.2f}", "",
         "## κ = E[−N] / [(1−a)·p̂_up]", "",
         "| 腕 | a | κ | 平均 p̂_up | 平均 E[−N] | n |", "|---|---|---|---|---|---|"]
    for arm, v in kappa.items():
        L.append(f"| `{arm}` | {v['a']} | {v['kappa']:.4f} | {v['mean_p']:.4f} | "
                 f"{v['mean_mN']:+.5f} | {v['n']:,} |")
    L += ["", "## 跨ぎゼロ（p̂_up = 0）の群は動くか", "",
          "| 腕 | 割合 | E[N] | [CI] | n |", "|---|---|---|---|---|"]
    for arm, v in zero.items():
        L.append(f"| `{arm}` | {v['frac']:.3f} | {v['E_N']:+.5f} | "
                 f"[{v['ci'][0]:+.5f}, {v['ci'][1]:+.5f}] | {v['n']:,} |")
    if strat:
        L += ["", "## 幅 W で層別（`LRoff0_1216`・1 次元に畳めるか）", "",
              "| 層 | W 中央値 | κ | n |", "|---|---|---|---|"]
        for k, v in strat.items():
            L.append(f"| {k} | {v['W_med']:.2f} | {v['kappa']:.4f} | {v['n']:,} |")
    L += ["", "## 腕ごとのビン（p̂_up 中央値 → E[−N]）", ""]
    for arm in rows_by_arm:
        rr = rows_by_arm[arm]
        L.append(f"- `{arm}` (a={rr[0]['a'] if rr else '—'}): "
                 + " / ".join(f"{r['p_med']:.3f}→{r['mN']:+.5f}" for r in rr))
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(Path(ROOT) / "results/drift_law_0907"))
    a = ap.parse_args()
    analyze(Path(a.out))


if __name__ == "__main__":
    main()
