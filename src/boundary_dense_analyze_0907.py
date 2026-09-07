# -*- coding: utf-8 -*-
"""boundary_dense_0907 の判定（spec `specs/spec_boundary_dense_0907.md` §4）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.boundary_dense_analyze_0907

窓の各タスク k（開始行 t0 = 10,000k・flip 前の旧支持）について
    J = [z̄−b](t0+1) − [z̄−b](t0)     反転の幾何の跳び
    B = b(t0+10000) − b(t0)          バイアスの学習
    A = [z̄−b](t0+10000) − [z̄−b](t0+1)  タスク内の w_flip の整列
    N = z̄(t0+10000) − z̄(t0) = J+B+A
主判定 SINK_CHANNEL = E[B]/E[N]、第 2 判定 SINK_TIMING = b の累積が半分に達する τ。
**`layer1_dzbar` は境界密格子では定義されないので読まない**（spec §8-2）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import boundary_dense_0907 as B
from .common import ROOT, load_config

PERIOD = B.PERIOD


def load_seed(logdir: Path, arm: str, seed: int) -> dict | None:
    p = Path(logdir) / f"{arm}_seed{seed}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        return dict(step=z["step"].astype(np.int64),
                    zb=z["layer1_zbar"].astype(np.float64),
                    b=z["layer1_b"].astype(np.float64),
                    zx=z["layer1_zmax"].astype(np.float64),
                    zn=z["layer1_zmin"].astype(np.float64))


def per_task(d: dict, window) -> dict:
    """(タスク, ユニット) の J/B/A/N と、b の累積プロファイル。"""
    idx = {int(v): i for i, v in enumerate(d["step"])}
    lo, hi = int(window[0]), int(window[1])
    taus = sorted({int(s) - t0 for t0 in (lo,) for s in d["step"]
                   if t0 < int(s) <= t0 + PERIOD} | {PERIOD})
    J, Bv, A, N, prof, aprof, zx0, straddle = [], [], [], [], [], [], [], []
    for t0 in range(lo, hi + 1, PERIOD):
        i0, i1, i2 = idx.get(t0), idx.get(t0 + 1), idx.get(t0 + PERIOD)
        if None in (i0, i1, i2):
            continue
        g0, g1, g2 = d["zb"][i0] - d["b"][i0], d["zb"][i1] - d["b"][i1], d["zb"][i2] - d["b"][i2]
        J.append(g1 - g0)
        Bv.append(d["b"][i2] - d["b"][i0])
        A.append(g2 - g1)
        N.append(d["zb"][i2] - d["zb"][i0])
        zx0.append(d["zx"][i0])
        straddle.append((d["zx"][i0] > 0) & (d["zn"][i0] <= 0))
        prof.append([d["b"][idx[t0 + t]] - d["b"][i0] if (t0 + t) in idx else np.nan
                     for t in taus])
        aprof.append([(d["zb"][idx[t0 + t]] - d["b"][idx[t0 + t]]) - g1
                      if (t0 + t) in idx else np.nan for t in taus])
    f = lambda x: np.asarray(x, dtype=np.float64)
    return dict(J=f(J), B=f(Bv), A=f(A), N=f(N), taus=taus,
                prof=np.asarray(prof, dtype=np.float64),
                aprof=np.asarray(aprof, dtype=np.float64),
                straddle=np.asarray(straddle), zmax0=f(zx0))


def boot(per_seed: list[np.ndarray], rng, nboot: int, stat=np.mean):
    """seed 単位の復元抽出（各 seed の中はプール）。"""
    vals = [np.asarray(v, dtype=np.float64).ravel() for v in per_seed]
    vals = [v[np.isfinite(v)] for v in vals]
    keep = [v for v in vals if v.size]
    if not keep:
        return float("nan"), [float("nan"), float("nan")]
    point = float(stat(np.concatenate(keep)))
    S = len(keep)
    draws = np.array([float(stat(np.concatenate([keep[i] for i in rng.integers(0, S, S)])))
                      for _ in range(nboot)])
    return point, [float(np.nanpercentile(draws, 2.5)), float(np.nanpercentile(draws, 97.5))]


def analyze(logdir: Path, out: Path) -> dict:
    cfg = load_config(str(B.CONFIG))
    an = cfg["analysis"]
    nboot, seed = int(an["boot"]["n"]), int(an["boot"]["seed"])
    lab, floor = an["labels"], float(an["floor"]["net_per_task"])
    window = [int(v) for v in an["window"]]
    arms = [str(r["name"]) for r in cfg["arms"]]
    rng = np.random.default_rng(seed)

    rows, per_arm = [], {}
    for arm in arms:
        seeds = [load_seed(logdir, arm, s) for s in range(10)]
        pt = [per_task(d, window) for d in seeds if d is not None]
        if not pt:
            continue
        per_arm[arm] = pt
        r = dict(arm=arm, n_seed=len(pt), n_task=int(pt[0]["J"].shape[0]))
        for k in ("J", "B", "A", "N"):
            p, ci = boot([x[k] for x in pt], np.random.default_rng(seed), nboot)
            r[k], r[f"{k}_ci"] = p, ci
        r["identity_max_abs"] = float(max(
            np.abs(x["J"] + x["B"] + x["A"] - x["N"]).max() for x in pt))
        r["straddle_frac"] = float(np.mean([x["straddle"].mean() for x in pt]))
        r["zmax0"] = float(np.median(np.concatenate([x["zmax0"].ravel() for x in pt])))
        rows.append(r)

    judged = str(an["judged_arm"])
    jr = [r for r in rows if r["arm"] == judged][0]
    pt = per_arm[judged]
    # 主判定
    fb_num = np.random.default_rng(seed + 1)
    Bs = [x["B"] for x in pt]
    Ns = [x["N"] for x in pt]
    def _ratio(idx):
        b = np.concatenate([Bs[i].ravel() for i in idx])
        n = np.concatenate([Ns[i].ravel() for i in idx])
        return float(np.mean(b) / np.mean(n)) if np.mean(n) != 0 else np.nan
    S = len(pt)
    f_b = _ratio(range(S))
    fb_draws = np.array([_ratio(fb_num.integers(0, S, S)) for _ in range(nboot)])
    f_b_ci = [float(np.nanpercentile(fb_draws, 2.5)), float(np.nanpercentile(fb_draws, 97.5))]
    n_ok = abs(jr["N"]) >= floor and not (jr["N_ci"][0] <= 0 <= jr["N_ci"][1])
    channel = ("NOT_DETERMINED" if not n_ok else
               "BIAS_CARRIED" if f_b >= float(lab["bias_carried_min"]) else
               "GEOMETRY_CARRIED" if f_b <= float(lab["geometry_carried_max"]) else "MIXED")
    # 第 2 判定
    taus = pt[0]["taus"]
    itot = taus.index(PERIOD)
    # prof は (タスク, τ, ユニット)。**reshape(-1, n_tau) は τ 軸とユニット軸を混ぜる**ので
    # 使わない（2026-09-07 の実装バグ・第 2 判定だけが影響した）。τ 以外を潰す。
    curve = np.nanmean(np.concatenate([x["prof"] for x in pt], axis=0), axis=(0, 2))
    total = float(curve[itot])
    tot_p, tot_ci = boot([x["prof"][:, itot, :] for x in pt],
                         np.random.default_rng(seed + 2), nboot)
    tau_half = None
    if np.isfinite(total) and total != 0:
        for t, v in zip(taus, curve):
            if (v / total) >= 0.5:
                tau_half = int(t)
                break
    timing = ("NOT_DETERMINED" if (tot_ci[0] <= 0 <= tot_ci[1]) or tau_half is None else
              "TRANSIENT" if tau_half <= int(lab["transient_max_tau"]) else
              "STEADY" if tau_half >= int(lab["steady_min_tau"]) else "INTERMEDIATE")

    acurve = np.nanmean(np.concatenate([x["aprof"] for x in pt], axis=0), axis=(0, 2))
    res = dict(experiment="boundary_dense_0907", judged_arm=judged, window=window,
               n_boot=nboot, boot_seed=seed, floor=floor,
               SINK_CHANNEL=dict(label=channel, f_b=f_b, f_b_ci=f_b_ci,
                                 E_N=jr["N"], E_N_ci=jr["N_ci"], floor_ok=bool(n_ok)),
               SINK_TIMING=dict(label=timing, tau_half=tau_half,
                                E_B_total=tot_p, E_B_total_ci=tot_ci,
                                taus=taus, curve=[float(v) for v in curve],
                                a_curve=[float(v) for v in acurve]),
               arms=rows)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=float),
                                      encoding="utf-8")
    L = [f"# boundary_dense_0907 判定",
         f"", f"- **SINK_CHANNEL = `{channel}`**  f_b = E[B]/E[N] = {f_b:.3f} "
         f"[{f_b_ci[0]:.3f}, {f_b_ci[1]:.3f}]・E[N] = {jr['N']:+.5f} "
         f"[{jr['N_ci'][0]:+.5f}, {jr['N_ci'][1]:+.5f}]（床 {floor}）",
         f"- **SINK_TIMING = `{timing}`**  τ½ = {tau_half}・E[B(10000)] = {tot_p:+.5f} "
         f"[{tot_ci[0]:+.5f}, {tot_ci[1]:+.5f}]", "",
         "## 1 タスクあたりの収支（seed ブートストラップ 2000）", "",
         "| 腕 | n タスク | J（幾何） | B（バイアス） | A（整列） | N（正味） | 跨ぎ率 | zmax(t0) | 恒等式残差 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        g = lambda k: f"{r[k]:+.5f} [{r[k+'_ci'][0]:+.5f}, {r[k+'_ci'][1]:+.5f}]"
        L.append(f"| `{r['arm']}` | {r['n_task']} | {g('J')} | {g('B')} | {g('A')} | {g('N')} | "
                 f"{r['straddle_frac']:.2f} | {r['zmax0']:+.3f} | {r['identity_max_abs']:.2e} |")
    L += ["", "## b の累積プロファイル（判定腕・単位 = z̄ の 1 タスクあたり）", "",
          "| τ | " + " | ".join(str(t) for t in taus) + " |",
          "|---|" + "---|" * len(taus),
          "| E[b(t0+τ)−b(t0)] | " + " | ".join(f"{v:+.5f}" for v in curve) + " |",
          "| 総量に対する割合 | " + " | ".join(f"{v/total:.2f}" if total else "—" for v in curve) + " |",
          "| E[A(τ)] = [z̄−b](t0+τ)−[z̄−b](t0+1) | " + " | ".join(f"{v:+.5f}" for v in acurve) + " |",
          "| A の総量に対する割合 | " + " | ".join(f"{v/acurve[itot]:.2f}" if acurve[itot] else "—"
                                                 for v in acurve) + " |"]
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default=str(Path(ROOT) / "results/boundary_dense_0907/logs"))
    ap.add_argument("--out", default=str(Path(ROOT) / "results/boundary_dense_0907"))
    a = ap.parse_args()
    analyze(Path(a.logs), Path(a.out))


if __name__ == "__main__":
    main()
