# -*- coding: utf-8 -*-
"""zeta_field_0908: 位置そのものの drift 場 E[Δζ | ζ]（読み規則 `specs/spec_zeta_field_0908.md`）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.zeta_field_0908

新しい走は無い。`drift_law_0907` と同じ既存ログ・同じ ζ（N の**両端**を除く 8 タスク平均）
を使い、条件付ける変数を p̂_up から **ζ そのもの**に変える。

問い: drift 場の零点（安定不動点）は固定した ζ にあるのか、それとも W に比例して動くのか。
    A = max|ζ*|/min|ζ*|（W 三分位横断）  対  B = max|ζ*/W̃|/min|ζ*/W̃|
どちらが 1 に近いかで `CLOSES_IN_ZETA` / `CLOSES_IN_RATIO` を分ける（排他）。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import drift_law_0907 as D
from .common import ROOT

OUT = Path(ROOT) / "results/zeta_field_0908"
JUDGED_ARM = "LRoff0_1216"
NBIN = 10
MIN_PER_BIN = 20
NBOOT = 2000
RNG_SEED = 20260908
RATIO_MAX = 1.25            # spec §4 のラベル境界
FLOOR_W_SPREAD = 1.5        # 床 1: W̃ の max/min
FLOOR_DRIFT = 0.005         # 床 2: 両端ビンの |E[N]|（/タスク）
ARMS = (("LRoff0_1216", 0.1), ("LRa0p2_off0_1216", 0.2), ("LRa0p3_off0_1216", 0.3),
        ("LRa0p5_off0_1216", 0.5), ("LRa0p7_off0_1216", 0.7))
AUX = (("LRoffm0p5_1216", 0.1), ("FBLRoff0_1216", 0.1), ("LRvf1_1216", 0.1))


# ---------------------------------------------------------------------------
def gather(arm: str) -> list[dict]:
    """seed ごとに (タスク×ユニット) を平らにした N / ζ / p̂_up / W。"""
    rows = []
    for t in D.arm_table(arm):
        rows.append({k: t[k].ravel() for k in ("N", "zeta", "p_up", "W")})
    return rows


def profile(x: np.ndarray, n: np.ndarray, nbin: int = NBIN) -> dict:
    """E[n | x] の十分位ビン（ビン中央値 x と平均 n）。"""
    m = np.isfinite(x) & np.isfinite(n)
    x, n = x[m], n[m]
    if x.size < nbin * MIN_PER_BIN:
        return dict(x=np.array([]), y=np.array([]), n=x.size)
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, nbin + 1)))
    if edges.size < 4:
        return dict(x=np.array([]), y=np.array([]), n=x.size)
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, edges.size - 2)
    bx, by = [], []
    for b in range(edges.size - 1):
        sel = idx == b
        if int(sel.sum()) < MIN_PER_BIN:
            continue
        bx.append(float(np.median(x[sel])))
        by.append(float(np.mean(n[sel])))
    return dict(x=np.asarray(bx), y=np.asarray(by), n=x.size)


def zero_crossing(prof: dict) -> float:
    """E[n|x] が **正 → 負** に変わる点を線形内挿。交差が 1 本でなければ nan。

    低い x で上がり、高い x で沈むなら + → − が安定不動点（spec §3-2）。
    複数あるときは「最も低いほうを採る」ではなく **判定しない**。
    """
    x, y = prof["x"], prof["y"]
    if x.size < 3:
        return float("nan")
    cross = [i for i in range(y.size - 1) if y[i] > 0.0 and y[i + 1] < 0.0]
    if len(cross) != 1:
        return float("nan")
    i = cross[0]
    t = y[i] / (y[i] - y[i + 1])
    return float(x[i] + t * (x[i + 1] - x[i]))


def drift_floor_ok(prof: dict) -> bool:
    """床 2: 両端ビンの E[N] が逆符号で、どちらも |E[N]| > FLOOR_DRIFT。"""
    y = prof["y"]
    if y.size < 3:
        return False
    return bool(y[0] * y[-1] < 0 and abs(y[0]) > FLOOR_DRIFT and abs(y[-1]) > FLOOR_DRIFT)


def tercile_edges(rows: list[dict]) -> np.ndarray:
    W = np.concatenate([r["W"] for r in rows])
    W = W[np.isfinite(W)]
    return np.quantile(W, [1.0 / 3.0, 2.0 / 3.0])


def terciles(rows: list[dict], edges: np.ndarray) -> list[dict]:
    """W の三分位ごとに ζ*・p*・W̃・床の可否を返す（seed はプール）。"""
    W = np.concatenate([r["W"] for r in rows])
    Z = np.concatenate([r["zeta"] for r in rows])
    P = np.concatenate([r["p_up"] for r in rows])
    N = np.concatenate([r["N"] for r in rows])
    g = np.digitize(W, edges)
    out = []
    for b in range(3):
        s = g == b
        pz, pp = profile(Z[s], N[s]), profile(P[s], N[s])
        out.append(dict(
            n=int(s.sum()), W_med=float(np.median(W[s])) if s.any() else float("nan"),
            zeta_star=zero_crossing(pz), p_star=zero_crossing(pp),
            drift_ok=bool(drift_floor_ok(pz)),
            prof_zeta=dict(x=pz["x"].tolist(), y=pz["y"].tolist()),
            prof_p=dict(x=pp["x"].tolist(), y=pp["y"].tolist())))
    return out


def ratios(ter: list[dict]) -> dict:
    """A（ζ* の広がり）・B（ζ*/W̃ の広がり）・C（p* の広がり）と床の可否。"""
    zs = np.array([t["zeta_star"] for t in ter], dtype=np.float64)
    ws = np.array([t["W_med"] for t in ter], dtype=np.float64)
    ps = np.array([t["p_star"] for t in ter], dtype=np.float64)
    ok_z = np.isfinite(zs) & np.isfinite(ws) & (zs != 0)
    ok_p = np.isfinite(ps) & (ps > 0)
    w_ok = np.isfinite(ws) & (ws > 0)
    spread = (float(np.max(ws[w_ok]) / np.min(ws[w_ok]))
              if int(w_ok.sum()) == 3 else float("nan"))
    def mm(v):
        v = np.abs(v)
        return float(np.max(v) / np.min(v)) if v.size == 3 and np.min(v) > 0 else float("nan")
    return dict(
        A=mm(zs[ok_z]) if int(ok_z.sum()) == 3 else float("nan"),
        B=mm((zs / ws)[ok_z]) if int(ok_z.sum()) == 3 else float("nan"),
        C=mm(ps[ok_p]) if int(ok_p.sum()) == 3 else float("nan"),
        W_spread=spread, n_valid_zeta=int(ok_z.sum()), n_valid_p=int(ok_p.sum()),
        drift_ok=bool(all(t["drift_ok"] for t in ter)))


def boot_ratios(rows: list[dict], edges: np.ndarray, nboot: int = NBOOT,
                seed: int = RNG_SEED) -> dict:
    """seed 単位の復元抽出。三分位の境界は全データで一度だけ切る（spec §6-7）。"""
    rng = np.random.default_rng(seed)
    S = len(rows)
    keys = ("A", "B", "C")
    draws = {k: [] for k in keys}
    for _ in range(nboot):
        pick = [rows[i] for i in rng.integers(0, S, S)]
        r = ratios(terciles(pick, edges))
        for k in keys:
            draws[k].append(r[k])
    out = {}
    for k in keys:
        v = np.asarray(draws[k], dtype=np.float64)
        v = v[np.isfinite(v)]
        out[k] = ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                  if v.size >= nboot // 2 else [float("nan"), float("nan")])
        out[k + "_n_finite"] = int(v.size)
    return out


# ---------------------------------------------------------------------------
def label_closure(r: dict, ci: dict) -> str:
    if not (np.isfinite(r["A"]) and np.isfinite(r["B"])):
        return "NOT_DETERMINED"
    if not r["drift_ok"] or not (np.isfinite(r["W_spread"])
                                 and r["W_spread"] >= FLOOR_W_SPREAD):
        return "NOT_DETERMINED"
    a_hi, a_lo = ci["A"][1], ci["A"][0]
    b_hi, b_lo = ci["B"][1], ci["B"][0]
    if not all(np.isfinite(v) for v in (a_lo, a_hi, b_lo, b_hi)):
        return "NOT_DETERMINED"
    if a_hi <= RATIO_MAX and b_lo > RATIO_MAX:
        return "CLOSES_IN_ZETA"
    if b_hi <= RATIO_MAX and a_lo > RATIO_MAX:
        return "CLOSES_IN_RATIO"
    if a_lo > RATIO_MAX and b_lo > RATIO_MAX:
        return "NEITHER"
    return "NOT_DETERMINED"


def label_pstar(r: dict, ci: dict) -> str:
    if not np.isfinite(r["C"]) or not r["drift_ok"]:
        return "NOT_DETERMINED"
    if not (np.isfinite(r["W_spread"]) and r["W_spread"] >= FLOOR_W_SPREAD):
        return "NOT_DETERMINED"
    lo, hi = ci["C"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NOT_DETERMINED"
    if hi <= RATIO_MAX:
        return "INVARIANT"
    if lo > RATIO_MAX:
        return "VARIES"
    return "NOT_DETERMINED"


def analyze(out: Path = OUT) -> dict:
    res = {"arms": {}, "config": dict(window=list(D.WINDOW), nbin=NBIN,
                                      ratio_max=RATIO_MAX, boot=NBOOT,
                                      rng=RNG_SEED, judged_arm=JUDGED_ARM,
                                      floor_W_spread=FLOOR_W_SPREAD,
                                      floor_drift=FLOOR_DRIFT)}
    for arm, a in list(ARMS) + list(AUX):
        rows = gather(arm)
        if not rows:
            continue
        edges = tercile_edges(rows)
        ter = terciles(rows, edges)
        r = ratios(ter)
        entry = dict(a=a, n_seeds=len(rows), tercile_edges=edges.tolist(),
                     terciles=ter, **r)
        if arm == JUDGED_ARM:
            entry["ci"] = boot_ratios(rows, edges)
        res["arms"][arm] = entry
    j = res["arms"].get(JUDGED_ARM)
    if j is None:
        res["ZETA_CLOSURE"] = res["PSTAR_INVARIANT"] = "NOT_DETERMINED"
        res["reason"] = "judged arm missing"
    else:
        res["ZETA_CLOSURE"] = label_closure(j, j["ci"])
        res["PSTAR_INVARIANT"] = label_pstar(j, j["ci"])
    return _write(res, out)


def _write(res: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    j = res["arms"].get(JUDGED_ARM, {})
    lines = ["# zeta_field_0908 の判定", "",
             f"- 主 `ZETA_CLOSURE` = **{res.get('ZETA_CLOSURE')}**",
             f"- 第 2 `PSTAR_INVARIANT` = **{res.get('PSTAR_INVARIANT')}**", ""]
    if j:
        ci = j.get("ci", {})
        lines += [f"判定腕 `{JUDGED_ARM}`: A = {j['A']:.3f} {ci.get('A')}  /  "
                  f"B = {j['B']:.3f} {ci.get('B')}  /  C = {j['C']:.3f} {ci.get('C')}",
                  f"床: W̃ の広がり {j['W_spread']:.2f}（要 {FLOOR_W_SPREAD}）・"
                  f"drift {j['drift_ok']}", "",
                  "| 三分位 | n | W̃ | ζ\\* | ζ\\*/W̃ | p\\* |", "|---|---|---|---|---|---|"]
        for i, t in enumerate(j["terciles"]):
            rr = t["zeta_star"] / t["W_med"] if t["W_med"] else float("nan")
            lines.append(f"| {i+1} | {t['n']} | {t['W_med']:.3f} | {t['zeta_star']:.4f} | "
                         f"{rr:.4f} | {t['p_star']:.4f} |")
    lines += ["", "| 腕 | a | A | B | C | W̃ の広がり | drift 床 |", "|---|---|---|---|---|---|---|"]
    for arm, e in res["arms"].items():
        lines.append(f"| {arm} | {e['a']} | {e['A']:.3f} | {e['B']:.3f} | {e['C']:.3f} | "
                     f"{e['W_spread']:.2f} | {e['drift_ok']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    res = analyze(Path(a.out) if a.out else OUT)
    print(json.dumps({k: res[k] for k in ("ZETA_CLOSURE", "PSTAR_INVARIANT")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
