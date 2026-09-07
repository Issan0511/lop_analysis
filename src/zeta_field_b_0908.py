# -*- coding: utf-8 -*-
"""zeta_field_b_0908: 位置の drift 場の**持ち出し検証**（読み規則 `specs/spec_zeta_field_b_0908.md`）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.zeta_field_b_0908

`zeta_field_0908` は床の設計ミスで落ちた。直した推定量を、**まだ 1 度も見ていない
11 腕**に当てる（焼けた 8 腕は判定に使わない・参考のみ）。

訂正 3 点（spec §4）:
  A 単調（非増加）回帰 PAVA を掛けてから零点を取る
  B 床は「両端のビン」でなく「プロファイルの振れ幅」に掛ける
  C 床を全部の AND にしない（六分位は 6 本中 4 本・腕は 11 本中 5 本）
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import drift_law_0907 as D
from .common import ROOT

OUT = Path(ROOT) / "results/zeta_field_b_0908"
NSEX = 6                     # W の六分位
NBIN = 10                    # ζ の十分位
MIN_PER_BIN = 20
NBOOT = 2000
RNG_SEED = 20260908
FLOOR_RANGE = 0.03           # プロファイルの振れ幅（/タスク）・追補 1 で 0.02 → 0.03
FLOOR_RHO = -0.6             # Spearman ρ(ζ, E[N]) の上限（追補 1）
FLOOR_W_SPREAD = 1.5         # 有効な六分位の W̃ の max/min
MIN_SEX = 4                  # 腕の資格: 有効な六分位が 6 本中 4 本以上
MIN_ARMS = 5                 # 主判定の床: 資格を満たす腕が 5 本以上
MIN_ARMS_P = 3               # 第 2 判定
PSTAR_RESOLVE = 2.0 / 32.0   # p* がこれ未満の腕は量子化で分解できないので除く

HELD_OUT = ("LRa0p2_offm0p5_1216", "LRa0p2_offp0p5_1216",
            "LRa0p3_offm0p5_1216", "LRa0p3_offp0p5_1216",
            "LRa0p5_offm0p5_1216", "LRa0p5_offp0p5_1216",
            "LRa0p7_offm0p5_1216", "LRa0p7_offp0p5_1216",
            "LRoffm0p25_1216", "LRoffp0p25_1216", "LRoffp0p5_1216")
PSTAR_ARMS = ("LRa0p3_offm0p5_1216", "LRa0p3_offp0p5_1216",
              "LRa0p5_offm0p5_1216", "LRa0p5_offp0p5_1216",
              "LRa0p7_offm0p5_1216", "LRa0p7_offp0p5_1216")
# 焼けた腕（参考のみ・判定に入れない・spec §0）
BURNED = ("LRoff0_1216", "LRa0p2_off0_1216", "LRa0p3_off0_1216",
          "LRa0p5_off0_1216", "LRa0p7_off0_1216", "LRoffm0p5_1216")


# ------------------------------------------- 採用しなかった推定量（追補 1）
def pava_nonincreasing(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """重み付き最小二乗の**非増加**当てはめ（pool-adjacent-violators）。

    **判定には使わない（追補 1 で棄却）。** 折れ目の上下に線形域が 2 つあり、
    どちらも釘付けなので E[N|ζ] は「上がる → 沈む → 0 へ戻る」で**単調でない**。
    単調制約を掛けると沈下の谷が上側の平坦域に吸収されて零点が消える
    （実プロファイルで実測・`test_pava_would_have_destroyed_the_crossing`）。
    診断として出力にだけ残す。
    """
    blocks: list[list[float]] = []          # [値, 重み, 長さ]
    for yi, wi in zip(np.asarray(y, float), np.asarray(w, float)):
        blocks.append([float(yi), float(wi), 1.0])
        while len(blocks) > 1 and blocks[-2][0] < blocks[-1][0]:
            v2, w2, c2 = blocks.pop()
            v1, w1, c1 = blocks.pop()
            wn = w1 + w2
            blocks.append([(v1 * w1 + v2 * w2) / wn, wn, c1 + c2])
    out = []
    for v, _, c in blocks:
        out.extend([v] * int(c))
    return np.asarray(out, dtype=np.float64)


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """順位相関（scipy を使わない）。ビンが 3 本未満なら 0。"""
    if len(x) < 3:
        return 0.0
    rx = np.argsort(np.argsort(np.asarray(x, float))).astype(float)
    ry = np.argsort(np.argsort(np.asarray(y, float))).astype(float)
    if rx.std() == 0 or ry.std() == 0:
        return 0.0
    return float(np.corrcoef(rx, ry)[0, 1])


def stable_zero(x: np.ndarray, y: np.ndarray) -> float:
    """**+ → −** の交差がちょうど 1 本のときだけ、その内挿点を返す。

    − → + の交差（不安定な不動点・上側の線形域へ戻る側）は数に入れない。
    """
    cross = [i for i in range(len(y) - 1) if y[i] > 0.0 and y[i + 1] < 0.0]
    if len(cross) != 1:
        return float("nan")
    i = cross[0]
    return float(x[i] + (x[i + 1] - x[i]) * y[i] / (y[i] - y[i + 1]))


def profile(x: np.ndarray, n: np.ndarray) -> dict:
    """ζ（または p̂_up）の十分位ビンごとの (中央値, E[N], 点数)。"""
    m = np.isfinite(x) & np.isfinite(n)
    x, n = x[m], n[m]
    if x.size < NBIN * MIN_PER_BIN:
        return dict(x=np.array([]), y=np.array([]), c=np.array([]))
    edges = np.unique(np.quantile(x, np.linspace(0.0, 1.0, NBIN + 1)))
    if edges.size < 4:
        return dict(x=np.array([]), y=np.array([]), c=np.array([]))
    idx = np.clip(np.digitize(x, edges[1:-1]), 0, edges.size - 2)
    bx, by, bc = [], [], []
    for b in range(edges.size - 1):
        sel = idx == b
        k = int(sel.sum())
        if k < MIN_PER_BIN:
            continue
        bx.append(float(np.median(x[sel]))); by.append(float(np.mean(n[sel]))); bc.append(k)
    return dict(x=np.asarray(bx), y=np.asarray(by), c=np.asarray(bc, dtype=np.float64))


def star_of(prof: dict) -> tuple[float, float, bool]:
    """零点・振れ幅・床の可否（追補 1 で改めた訂正 A / B）。

    六分位が有効なのは次の 3 つを**すべて**満たすとき:
      1. `+ → −` の交差がちょうど 1 本（＝安定不動点が定義できる）
      2. 振れ幅 max − min > `FLOOR_RANGE`（両端ビンでなく振れ幅に掛ける）
      3. Spearman ρ(ζ, E[N]) ≤ `FLOOR_RHO`（＝そもそも右下がりである）
    3 つの AND は**六分位の中**の話で、腕の判定は 6 本中 4 本・11 本中 5 本（訂正 C）。
    """
    x, y = prof["x"], prof["y"]
    if x.size < 3:
        return float("nan"), float("nan"), False
    rng = float(y.max() - y.min())
    star = stable_zero(x, y)
    ok = bool(np.isfinite(star) and rng > FLOOR_RANGE
              and spearman(x, y) <= FLOOR_RHO)
    return (star if ok else float("nan")), rng, ok


# ------------------------------------------------------------------ 腕
def gather(arm: str) -> list[dict]:
    return [{k: t[k].ravel() for k in ("N", "zeta", "p_up", "W")}
            for t in D.arm_table(arm)]


def sextile_edges(rows: list[dict]) -> np.ndarray:
    W = np.concatenate([r["W"] for r in rows])
    W = W[np.isfinite(W)]
    return np.quantile(W, np.linspace(0.0, 1.0, NSEX + 1)[1:-1])


def arm_fit(rows: list[dict], edges: np.ndarray, want_p: bool = False) -> dict:
    """六分位ごとに ζ*（と p*）を出し、ζ*(W) を重み付き最小二乗で当てる。"""
    W = np.concatenate([r["W"] for r in rows])
    Z = np.concatenate([r["zeta"] for r in rows])
    N = np.concatenate([r["N"] for r in rows])
    P = np.concatenate([r["p_up"] for r in rows]) if want_p else None
    g = np.digitize(W, edges)
    sex = []
    for b in range(NSEX):
        s = g == b
        if not s.any():
            sex.append(dict(n=0, W_med=float("nan"), zeta_star=float("nan"),
                            rng=float("nan"), ok=False, p_star=float("nan")))
            continue
        zs, rng, ok = star_of(profile(Z[s], N[s]))
        ps = star_of(profile(P[s], N[s]))[0] if want_p else float("nan")
        sex.append(dict(n=int(s.sum()), W_med=float(np.median(W[s])),
                        zeta_star=zs, rng=rng, ok=ok, p_star=ps))
    good = [t for t in sex if t["ok"] and np.isfinite(t["zeta_star"])]
    w_med_all = float(np.median(W[np.isfinite(W)]))
    res = dict(sextiles=sex, n_good=len(good), W_med_all=w_med_all,
               beta=float("nan"), alpha=float("nan"), alpha_hat=float("nan"),
               W_spread=float("nan"), qualified=False)
    if len(good) >= MIN_SEX:
        ws = np.array([t["W_med"] for t in good])
        zs = np.array([t["zeta_star"] for t in good])
        cn = np.array([t["n"] for t in good], dtype=np.float64)
        spread = float(ws.max() / ws.min()) if ws.min() > 0 else float("nan")
        res["W_spread"] = spread
        if np.isfinite(spread) and spread >= FLOOR_W_SPREAD:
            b, a0 = np.polyfit(ws, zs, 1, w=np.sqrt(cn))
            res.update(beta=float(b), alpha=float(a0),
                       alpha_hat=float(a0 / w_med_all) if w_med_all else float("nan"),
                       qualified=True)
    if want_p:
        ps = np.array([t["p_star"] for t in sex if t["ok"] and np.isfinite(t["p_star"])])
        res["p_stars"] = ps.tolist()
        res["p_median"] = float(np.median(ps)) if ps.size else float("nan")
        res["C"] = (float(np.max(ps) / np.min(ps))
                    if ps.size >= MIN_SEX and np.min(ps) > 0 else float("nan"))
    return res


def medians(fits: dict, arms: tuple[str, ...]) -> dict:
    b = [f["beta"] for a, f in fits.items() if a in arms and f["qualified"]]
    ah = [abs(f["alpha_hat"]) for a, f in fits.items() if a in arms and f["qualified"]]
    return dict(beta=float(np.median(b)) if b else float("nan"),
                abs_alpha_hat=float(np.median(ah)) if ah else float("nan"),
                n_arms=len(b))


def median_C(fits: dict, arms: tuple[str, ...]) -> dict:
    v = [f["C"] for a, f in fits.items()
         if a in arms and np.isfinite(f.get("C", np.nan))
         and np.isfinite(f.get("p_median", np.nan)) and f["p_median"] >= PSTAR_RESOLVE]
    return dict(C=float(np.median(v)) if v else float("nan"), n_arms=len(v))


# ------------------------------------------------------------ 判定
def label_form(m: dict, ci: dict) -> str:
    if m["n_arms"] < MIN_ARMS:
        return "NOT_DETERMINED"
    bl, bh = ci["beta"]
    al, ah = ci["abs_alpha_hat"]
    if not all(np.isfinite(v) for v in (bl, bh, al, ah)):
        return "NOT_DETERMINED"
    if -0.25 <= bl and bh <= 0.25:
        return "CLOSES_IN_ZETA"
    if -1.25 <= bl and bh <= -0.75 and ah <= 0.25:
        return "EDGE_LAW"
    if bh < -0.25:
        return "AFFINE"
    return "NOT_DETERMINED"


def label_pstar(m: dict, ci: dict) -> str:
    if m["n_arms"] < MIN_ARMS_P:
        return "NOT_DETERMINED"
    lo, hi = ci["C"]
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "NOT_DETERMINED"
    if hi <= 1.25:
        return "INVARIANT"
    if lo > 1.25:
        return "VARIES"
    return "NOT_DETERMINED"


def analyze(out: Path = OUT, nboot: int = NBOOT) -> dict:
    rows_of, edges_of = {}, {}
    for arm in HELD_OUT:
        r = gather(arm)
        if r:
            rows_of[arm] = r
            edges_of[arm] = sextile_edges(r)
    fits = {a: arm_fit(rows_of[a], edges_of[a], want_p=(a in PSTAR_ARMS))
            for a in rows_of}
    m_form = medians(fits, HELD_OUT)
    m_p = median_C(fits, PSTAR_ARMS)

    # ---- ブートストラップ（seed 添字は 1 抽出を全腕に共通に当てる）--------
    rng = np.random.default_rng(RNG_SEED)
    S = min(len(v) for v in rows_of.values())
    draws = {"beta": [], "abs_alpha_hat": [], "C": []}
    for _ in range(nboot):
        pick = rng.integers(0, S, S)
        f2 = {}
        for a, r in rows_of.items():
            f2[a] = arm_fit([r[i] for i in pick], edges_of[a],
                            want_p=(a in PSTAR_ARMS))
        mm = medians(f2, HELD_OUT)
        draws["beta"].append(mm["beta"])
        draws["abs_alpha_hat"].append(mm["abs_alpha_hat"])
        draws["C"].append(median_C(f2, PSTAR_ARMS)["C"])
    ci = {}
    for k, v in draws.items():
        v = np.asarray(v, dtype=np.float64)
        v = v[np.isfinite(v)]
        ci[k] = ([float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                 if v.size >= nboot // 2 else [float("nan"), float("nan")])
        ci[k + "_n"] = int(v.size)

    res = dict(config=dict(held_out=list(HELD_OUT), pstar_arms=list(PSTAR_ARMS),
                           nsex=NSEX, nbin=NBIN, boot=nboot, rng=RNG_SEED,
                           floor_range=FLOOR_RANGE, floor_W_spread=FLOOR_W_SPREAD,
                           min_sex=MIN_SEX, min_arms=MIN_ARMS,
                           pstar_resolve=PSTAR_RESOLVE),
               arms={a: {k: v for k, v in f.items() if k != "sextiles"} | {
                   "sextiles": f["sextiles"]} for a, f in fits.items()},
               point=dict(form=m_form, pstar=m_p), ci=ci)
    res["FIELD_FORM"] = label_form(m_form, ci)
    res["PSTAR_INVARIANT_B"] = label_pstar(m_p, ci)
    return _write(res, out)


def _write(res: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    p, ci = res["point"], res["ci"]
    lines = ["# zeta_field_b_0908 の判定（持ち出し 11 腕）", "",
             f"- 主 `FIELD_FORM` = **{res['FIELD_FORM']}**",
             f"- 第 2 `PSTAR_INVARIANT_B` = **{res['PSTAR_INVARIANT_B']}**", "",
             f"β の中央値 = {p['form']['beta']:.3f}  CI {ci['beta']}  "
             f"（資格 {p['form']['n_arms']}/{len(res['config']['held_out'])} 腕・要 {MIN_ARMS}）",
             f"|α/W̃| の中央値 = {p['form']['abs_alpha_hat']:.3f}  CI {ci['abs_alpha_hat']}",
             f"C（p\\* の max/min）の中央値 = {p['pstar']['C']:.3f}  CI {ci['C']}  "
             f"（資格 {p['pstar']['n_arms']} 腕・要 {MIN_ARMS_P}）", "",
             "| 腕 | 有効六分位 | W̃ の広がり | β | α | α/W̃ | 資格 |",
             "|---|---|---|---|---|---|---|"]
    for a, f in res["arms"].items():
        lines.append(f"| {a} | {f['n_good']}/{NSEX} | {f['W_spread']:.2f} | "
                     f"{f['beta']:.3f} | {f['alpha']:.3f} | {f['alpha_hat']:.3f} | "
                     f"{f['qualified']} |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--boot", type=int, default=NBOOT)
    a = ap.parse_args()
    res = analyze(Path(a.out) if a.out else OUT, nboot=a.boot)
    print(json.dumps({k: res[k] for k in ("FIELD_FORM", "PSTAR_INVARIANT_B")},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
