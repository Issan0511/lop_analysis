# -*- coding: utf-8 -*-
"""wflip_split_0907 の判定（spec `specs/spec_wflip_split_0907.md` §4）。

    OMP_NUM_THREADS=1 PYTHONPATH=. python3 -m src.wflip_split_analyze_0907

主判定 `WFLIP_ALIGN` = 蓄積した Δw_flip(199) の全 1 方向への分散寄与率 s。
第 2 判定 `J_SOURCE`  = 境界の戻り J を [ビット項 / γ-flip 項 / γ-free 項] に分けたシェア。
S 検査 4 本（S-null / S-rank1 / S-zbar / S-J）はすべて変異対照つき。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from . import wflip_split_0907 as W
from .common import ROOT, load_config

PERIOD = W.PERIOD
N_FLIP = W.N_FLIP
ONES = np.ones(N_FLIP, dtype=np.float64)
ONES_HAT = ONES / np.sqrt(N_FLIP)


# ---------------------------------------------------------------------------
# 幾何（spec §2・記録せず式から出す）
# ---------------------------------------------------------------------------
def gamma_of(k, target: float) -> np.ndarray:
    """`dose_const_5m.gamma_for_k` と同じ登録済みの小さい方の根（float64）。"""
    kd = np.asarray(k, dtype=np.float64)
    disc = np.square(kd + 2.5) - 20.0 * (kd + 1.25 - target ** 2)
    if np.any(disc < 0):
        raise ValueError("negative gamma discriminant")
    return ((kd + 2.5) - np.sqrt(disc)) / 10.0


def u_of(flip: np.ndarray, target: float):
    """u = x̃ − 0.5γ·1（flip 15 成分）と m = 0.5(1−γ) を返す。flip は (..., 15)。"""
    g = gamma_of(flip.sum(axis=-1), target)
    return flip - 0.5 * g[..., None], 0.5 * (1.0 - g)


# ---------------------------------------------------------------------------
def load_seed(logdir: Path, arm: str, seed: int) -> dict | None:
    p = Path(logdir) / f"{arm}_seed{seed}.npz"
    if not p.exists():
        return None
    with np.load(p, allow_pickle=True) as z:
        keys = set(z.files)
        if "layer1_w_flip" not in keys:
            raise KeyError(f"{p.name} has no layer1_w_flip")
        return dict(
            step=z["step"].astype(np.int64),
            zb=z["layer1_zbar"].astype(np.float64),
            b=z["layer1_b"].astype(np.float64),
            flip=z["flip_state"].astype(np.float64),
            wfl=z["layer1_w_flip"].astype(np.float64),
            wfl_step=z["layer1_w_flip_step"].astype(np.int64),
            wfr=z["layer1_w_free"].astype(np.float64),
            wfr_step=z["layer1_w_free_step"].astype(np.int64),
            state_hash_final=str(z["state_hash_final"]))


# S-null で比べない列（記録格子に依存する・`boundary_dense_0907` §8-2 と同じ理由）
GRID_DEPENDENT_COLUMNS = ("layer1_dzbar",)


def s_null_pair(logdir: Path, mine: str, ref_logdir: Path, theirs: str,
                seeds) -> dict:
    """学習が参照とビット一致か（追補 3）。

    **`state_hash_final` は使えない**——本走は 2M step、参照 `LRoff0_1216` は 5M step
    なので終端が違う。共通なのは (1) 1M step のチェックポイント `state_hash_1m` と
    (2) **共有する記録 step における共通列**。両方を要求する（`final` 1 つより強い）。
    `layer1_dzbar` は記録格子に依存するので除く。
    """
    rows = []
    for s in seeds:
        pm = Path(logdir) / f"{mine}_seed{s}.npz"
        pr = Path(ref_logdir) / f"{theirs}_seed{s}.npz"
        if not (pm.exists() and pr.exists()):
            rows.append(dict(seed=int(s), pass_=False, reason="missing"))
            continue
        with np.load(pm, allow_pickle=True) as a, np.load(pr, allow_pickle=True) as b:
            h = str(a["state_hash_1m"]) == str(b["state_hash_1m"])
            sa = a["step"].astype(np.int64)
            sb = b["step"].astype(np.int64)
            shared = np.intersect1d(sa, sb)
            ia = {int(v): i for i, v in enumerate(sa)}
            ib = {int(v): i for i, v in enumerate(sb)}
            ra = np.array([ia[int(v)] for v in shared], dtype=np.int64)
            rb = np.array([ib[int(v)] for v in shared], dtype=np.int64)
            mism, n_cmp = [], 0
            for k in sorted(set(a.files) & set(b.files)):
                if k in GRID_DEPENDENT_COLUMNS:
                    continue
                x, y = a[k], b[k]
                if x.ndim == 0 or y.ndim == 0 or x.dtype != y.dtype:
                    continue
                if x.shape[0] != sa.size or y.shape[0] != sb.size:
                    continue                      # step 軸を持たない列（w_free 等）
                n_cmp += 1
                xa = np.ascontiguousarray(x[ra])
                yb = np.ascontiguousarray(y[rb])
                if xa.shape != yb.shape or not np.array_equal(
                        xa.view(np.uint8), yb.view(np.uint8)):
                    mism.append(k)
            rows.append(dict(seed=int(s), pass_=bool(h and not mism), hash_1m=bool(h),
                             n_shared=int(shared.size), n_compared=int(n_cmp),
                             mismatched=mism))
    return dict(rows=rows, n=len(rows), match=sum(1 for r in rows if r["pass_"]),
                pass_=bool(rows and all(r["pass_"] for r in rows)),
                excluded=list(GRID_DEPENDENT_COLUMNS))


def _aligned(d: dict) -> dict:
    """w_flip / w_free の記録行に、同じ step の本体行（zb, b, flip）を並べる。"""
    idx = {int(v): i for i, v in enumerate(d["step"])}
    fri = {int(v): i for i, v in enumerate(d["wfr_step"])}
    rows, main, free = [], [], []
    for j, s in enumerate(d["wfl_step"]):
        i, k = idx.get(int(s)), fri.get(int(s))
        if i is None or k is None:
            continue
        rows.append(j); main.append(i); free.append(k)
    return dict(j=np.array(rows), i=np.array(main), k=np.array(free),
                step=d["wfl_step"][np.array(rows)])


# ---------------------------------------------------------------------------
# S 検査
# ---------------------------------------------------------------------------
EPS32 = float(np.finfo(np.float32).eps)


def s_rank1(d: dict, target: float, *, mutant: bool = False) -> dict:
    """タスク内 Δw_flip の u 直交成分。(タスク, ユニット) の 2 通りで返す。

    - `rho` = ‖Δw_⊥‖ / (eps₃₂·‖w_flip‖) …… **判定に使う**（追補 2）
    - `rel` = ‖Δw_⊥‖ / ‖Δw‖ ………………… 参考（登録時の形）

    `rel` は「ユニットがどれだけ動いたか」を測ってしまう: 直交残差の絶対値は
    float32 の記録雑音で決まる定数なので、`rel` は 1/‖Δw‖ に比例する
    （preflight 実測: log-log の傾き −0.98 / −1.02・‖Δw‖ が 69 倍動く間
    ‖Δw_⊥‖ は 2.3e−6 で一定）。平行性そのものを測るのは `rho` の側。

    `mutant=True` では u の 1 ビット目を反転させた u′ を使う（変異対照）。
    タスク k の実効入力は「そのタスクの終端行に載っている flip_state」で決まる。
    """
    a = _aligned(d)
    out, out_rho = [], []
    for n in range(len(a["j"]) - 1):
        j0, j1, i1 = a["j"][n], a["j"][n + 1], a["i"][n + 1]
        if int(a["step"][n + 1]) - int(a["step"][n]) != PERIOD:
            continue
        flip = d["flip"][i1].copy()
        if mutant:
            flip[0] = 1.0 - flip[0]
        u, _ = u_of(flip, target)
        uh = u / np.linalg.norm(u)
        dw = d["wfl"][j1] - d["wfl"][j0]                     # (h, 15)
        nrm = np.linalg.norm(dw, axis=1)
        # 直交成分は「差の平方根」ではなく**直接**引く（追補 1）。
        # sqrt(‖dw‖² − (dw·û)²) は平行な入力で桁落ちし、相対誤差が √eps まで
        # 悪化する（float64 で 2e-8・float32 ログでは 3e-4）。
        perp = np.linalg.norm(dw - (dw @ uh)[:, None] * uh[None, :], axis=1)
        wn = np.linalg.norm(d["wfl"][j1], axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            out.append(np.where(nrm > 0, perp / nrm, np.nan))
            out_rho.append(np.where(wn > 0, perp / (EPS32 * wn), np.nan))
    return dict(rho=np.asarray(out_rho, dtype=np.float64),
                rel=np.asarray(out, dtype=np.float64))


def s_zbar(d: dict, target: float, *, drop_m: bool = False) -> np.ndarray:
    """3 分解 z̄ = b + w_flip·u + m·Σw_free の残差（(記録, ユニット)）。

    分母は `max(|z̄|, 1)`（追補 1）。素の |z̄| で割ると z̄ ≈ 0 のユニットで
    残差が発散し、検査が中央値でしか意味を持たなくなる。
    `drop_m=True` では m 項を落とす（変異対照）。
    """
    a = _aligned(d)
    out = []
    for n in range(len(a["j"])):
        j, i, k = a["j"][n], a["i"][n], a["k"][n]
        u, m = u_of(d["flip"][i], target)
        pred = d["b"][i] + d["wfl"][j] @ u
        if not drop_m:
            pred = pred + m * d["wfr"][k].sum(axis=1)
        z = d["zb"][i]
        out.append(np.abs(pred - z) / np.maximum(np.abs(z), 1.0))
    return np.asarray(out, dtype=np.float64)


# ---------------------------------------------------------------------------
# 判定量
# ---------------------------------------------------------------------------
def align_share(d: dict, task: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """s = (Δw·1̂)²/‖Δw‖²、‖Δw‖、‖w_flip(0)‖（ユニットごと）。"""
    step = int(task) * PERIOD
    fj = {int(v): i for i, v in enumerate(d["wfl_step"])}
    j0, j1 = fj.get(0), fj.get(step)
    if j0 is None or j1 is None:
        return (np.array([]),) * 3
    w0 = d["wfl"][j0]
    dw = d["wfl"][j1] - w0
    nrm = np.linalg.norm(dw, axis=1)
    par = dw @ ONES_HAT
    with np.errstate(invalid="ignore", divide="ignore"):
        s = np.where(nrm > 0, (par ** 2) / (nrm ** 2), np.nan)
    return s, nrm, np.linalg.norm(w0, axis=1)


def j_terms(d: dict, target: float, window) -> dict:
    """境界 t0 の J を 3 項に厳密分解し、実測 J（g1−g0）と並べる。

    J_geo = w_flip·(u_new − u_old)                        … 全体
          = [ビット項] w_flip·(x̃_new − x̃_old)
          + [γ-flip 項] −0.5Δγ·Σw_flip
          + [γ-free 項] −0.5Δγ·Σw_free
    """
    idx = {int(v): i for i, v in enumerate(d["step"])}
    fj = {int(v): i for i, v in enumerate(d["wfl_step"])}
    fri = {int(v): i for i, v in enumerate(d["wfr_step"])}
    lo, hi = int(window[0]), int(window[1])
    bit, gfl, gfr, geo, meas = [], [], [], [], []
    for t0 in range(lo, hi + 1, PERIOD):
        i0, i1, j0, k0 = idx.get(t0), idx.get(t0 + 1), fj.get(t0), fri.get(t0)
        if None in (i0, i1, j0, k0):
            continue
        x_old, x_new = d["flip"][i0], d["flip"][i1]
        g_old = gamma_of(x_old.sum(), target)
        g_new = gamma_of(x_new.sum(), target)
        w, sfr = d["wfl"][j0], d["wfr"][k0].sum(axis=1)      # (h,15), (h,)
        dg = float(g_new - g_old)
        bit.append(w @ (x_new - x_old))
        gfl.append(-0.5 * dg * w.sum(axis=1))
        gfr.append(-0.5 * dg * sfr)
        u_old, m_old = u_of(x_old, target)
        u_new, m_new = u_of(x_new, target)
        geo.append(w @ (u_new - u_old) + (m_new - m_old) * sfr)
        meas.append((d["zb"][i1] - d["b"][i1]) - (d["zb"][i0] - d["b"][i0]))
    f = lambda x: np.asarray(x, dtype=np.float64)
    return dict(bit=f(bit), gfl=f(gfl), gfr=f(gfr), geo=f(geo), meas=f(meas))


def boot(per_seed, rng, nboot: int, stat=np.median):
    vals = [np.asarray(v, dtype=np.float64).ravel() for v in per_seed]
    vals = [v[np.isfinite(v)] for v in vals]
    keep = [v for v in vals if v.size]
    if not keep:
        return float("nan"), [float("nan"), float("nan")]
    point = float(stat(np.concatenate(keep)))
    S = len(keep)
    draws = np.array([float(stat(np.concatenate([keep[i] for i in rng.integers(0, S, S)])))
                      for _ in range(nboot)])
    return point, [float(np.nanpercentile(draws, 2.5)),
                   float(np.nanpercentile(draws, 97.5))]


# ---------------------------------------------------------------------------
def analyze(logdir: Path, out: Path, ref_logdir: Path | None = None) -> dict:
    cfg = load_config(str(W.CONFIG))
    an = cfg["analysis"]
    target = float(an["target_mu_norm"])
    nboot, bseed = int(an["boot"]["n"]), int(an["boot"]["seed"])
    lab, floor, tol = an["labels"], an["floor"], an["tol"]
    window = [int(v) for v in an["window"]]
    task = int(an["align_task"])
    judged = str(an["judged_arm"])
    seeds = [int(v) for v in cfg["common_overrides"]["seeds"]]
    arms = [str(r["name"]) for r in cfg["arms"]]
    rng = np.random.default_rng(bseed)
    res = {"arms": {}, "config": dict(window=window, align_task=task,
                                      target_mu_norm=target, judged_arm=judged)}

    for arm in arms:
        per = [load_seed(logdir, arm, s) for s in seeds]
        per = [p for p in per if p is not None]
        if not per:
            continue
        S, NRM, W0 = [], [], []
        R1, R1M, ZB, ZBM = [], [], [], []
        JT = {k: [] for k in ("bit", "gfl", "gfr", "geo", "meas")}
        for d in per:
            s, nrm, w0 = align_share(d, task)
            S.append(s); NRM.append(nrm); W0.append(w0)
            R1.append(s_rank1(d, target))
            R1M.append(s_rank1(d, target, mutant=True))
            ZB.append(s_zbar(d, target))
            ZBM.append(s_zbar(d, target, drop_m=True))
            t = j_terms(d, target, window)
            for k in JT:
                JT[k].append(t[k])
        s_pt, s_ci = boot(S, np.random.default_rng(bseed), nboot, np.median)
        dw_pt = float(np.median(np.concatenate([v for v in NRM if v.size])))
        w0_pt = float(np.median(np.concatenate([v for v in W0 if v.size])))
        jm_pt, jm_ci = boot(JT["meas"], np.random.default_rng(bseed + 1), nboot, np.mean)
        shares = {}
        for k in ("bit", "gfl", "gfr"):
            v = float(np.mean(np.concatenate([x.ravel() for x in JT[k]])))
            shares[k] = dict(mean=v, share=v / jm_pt if jm_pt else float("nan"))
        geo_pt = float(np.mean(np.concatenate([x.ravel() for x in JT["geo"]])))
        res["arms"][arm] = dict(
            n_seeds=len(per),
            s=dict(point=s_pt, ci=s_ci),
            dw_norm=dw_pt, w0_norm=w0_pt, dw_over_w0=dw_pt / w0_pt if w0_pt else np.nan,
            J=dict(measured=jm_pt, ci=jm_ci, geometric=geo_pt,
                   rel_gap=abs(geo_pt - jm_pt) / abs(jm_pt) if jm_pt else np.nan,
                   terms=shares,
                   bit_plus_gamma=shares["bit"]["share"] + shares["gfl"]["share"]
                   + shares["gfr"]["share"]),
            s_rank1=dict(
                rho=float(np.nanmedian(np.concatenate([x["rho"].ravel() for x in R1]))),
                rho_max=float(np.nanmax(np.concatenate([x["rho"].ravel() for x in R1]))),
                rho_mutant=float(np.nanmedian(np.concatenate([x["rho"].ravel() for x in R1M]))),
                rel=float(np.nanmedian(np.concatenate([x["rel"].ravel() for x in R1]))),
                rel_mutant=float(np.nanmedian(np.concatenate([x["rel"].ravel() for x in R1M])))),
            s_zbar=dict(value=float(np.nanmedian(np.concatenate([x.ravel() for x in ZB]))),
                        mutant=float(np.nanmedian(np.concatenate([x.ravel() for x in ZBM])))))

    # ---- S-null（state_hash が参照腕と一致するか）--------------------------
    pair = an.get("s_null_pair") or {}
    null = {}
    if ref_logdir is not None:
        for mine, theirs in pair.items():
            null[f"{mine}<->{theirs}"] = s_null_pair(
                logdir, mine, Path(ref_logdir), theirs, seeds)
    res["s_null"] = null

    # ---- 判定 ---------------------------------------------------------------
    j = res["arms"].get(judged, {})
    checks = dict(
        s_rank1=dict(rho=j.get("s_rank1", {}).get("rho"),
                     rho_mutant=j.get("s_rank1", {}).get("rho_mutant"),
                     pass_=bool(j and j["s_rank1"]["rho"] < float(tol["rank1_rho_max"])
                                and j["s_rank1"]["rho_mutant"] > float(tol["rank1_rho_mutant_min"]))),
        s_zbar=dict(value=j.get("s_zbar", {}).get("value"),
                    mutant=j.get("s_zbar", {}).get("mutant"),
                    pass_=bool(j and j["s_zbar"]["value"] < float(tol["zbar_rel"])
                               and j["s_zbar"]["mutant"] > float(tol["zbar_mutant_min"]))),
        s_null=dict(pass_=bool(null and all(v["pass_"] for v in null.values()))
                    if ref_logdir is not None else None))
    res["checks"] = checks

    if not j:
        res["WFLIP_ALIGN"] = "NOT_DETERMINED"
        res["J_SOURCE"] = "NOT_DETERMINED"
        res["reason"] = "judged arm missing"
        return _write(res, out)
    if not checks["s_rank1"]["pass_"]:                     # 追補 2: ρ で見る
        res["WFLIP_ALIGN"] = "NOT_DETERMINED"
        res["J_SOURCE"] = "NOT_DETERMINED"
        res["reason"] = "S-rank1 failed: (G2) is broken, judgments withheld (spec §4)"
        return _write(res, out)

    lo, hi = j["s"]["ci"]
    if j["dw_over_w0"] < float(floor["dw_over_w0"]):
        res["WFLIP_ALIGN"] = "NOT_DETERMINED"
    elif lo > float(lab["bias_like_min"]):
        res["WFLIP_ALIGN"] = "BIAS_LIKE"
    elif hi < float(lab["pattern_like_max"]):
        res["WFLIP_ALIGN"] = "PATTERN_LIKE"
    elif lo >= float(lab["pattern_like_max"]) and hi <= float(lab["bias_like_min"]):
        res["WFLIP_ALIGN"] = "MIXED"
    else:
        res["WFLIP_ALIGN"] = "NOT_DETERMINED"

    jl, jh = j["J"]["ci"]
    fl = float(floor["j_per_task"])
    bit = j["J"]["terms"]["bit"]["share"]
    gam = j["J"]["terms"]["gfl"]["share"] + j["J"]["terms"]["gfr"]["share"]
    if (jl <= fl <= jh) or (jl <= -fl <= jh) or abs(j["J"]["measured"]) < fl:
        res["J_SOURCE"] = "NOT_DETERMINED"
    elif j["J"]["rel_gap"] > float(tol["j_rel"]):
        res["J_SOURCE"] = "NOT_DETERMINED"
        res["j_reason"] = f"geometric J differs from measured by {j['J']['rel_gap']:.2e}"
    elif bit >= float(lab["j_carried_min"]):
        res["J_SOURCE"] = "BIT_CARRIED"
    elif gam >= float(lab["j_carried_min"]):
        res["J_SOURCE"] = "GAMMA_CARRIED"
    else:
        res["J_SOURCE"] = "MIXED"
    return _write(res, out)


def _write(res: dict, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(
        json.dumps(res, indent=2, ensure_ascii=False, default=float), encoding="utf-8")
    lines = ["# wflip_split_0907 の判定", "",
             f"- 主 `WFLIP_ALIGN` = **{res.get('WFLIP_ALIGN')}**",
             f"- 第 2 `J_SOURCE` = **{res.get('J_SOURCE')}**", ""]
    if res.get("reason"):
        lines += [f"> {res['reason']}", ""]
    lines += ["| 腕 | s (全 1 方向の寄与率) | CI | ‖Δw‖/‖w0‖ | E[J] | ビット項 | γ-flip | γ-free | S-rank1 ρ | S-zbar |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for arm, a in res.get("arms", {}).items():
        t = a["J"]["terms"]
        lines.append(
            f"| {arm} | {a['s']['point']:.4f} | [{a['s']['ci'][0]:.4f}, {a['s']['ci'][1]:.4f}] | "
            f"{a['dw_over_w0']:.3f} | {a['J']['measured']:+.5f} | {t['bit']['share']:+.2f} | "
            f"{t['gfl']['share']:+.2f} | {t['gfr']['share']:+.2f} | "
            f"{a['s_rank1']['rho']:.1f} (変異 {a['s_rank1']['rho_mutant']:.3g}) | "
            f"{a['s_zbar']['value']:.2e} (変異 {a['s_zbar']['mutant']:.2e}) |")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logdir", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--ref-logdir", default=str(Path(ROOT) / "results/offset_grid_0906/logs"))
    a = ap.parse_args()
    reg = load_config(str(W.CONFIG))
    logdir = Path(a.logdir) if a.logdir else Path(ROOT) / reg["output"]["dir"] / "logs"
    out = Path(a.out) if a.out else Path(ROOT) / reg["output"]["dir"]
    ref = Path(a.ref_logdir) if a.ref_logdir else None
    res = analyze(logdir, out, ref)
    print(json.dumps({k: res[k] for k in ("WFLIP_ALIGN", "J_SOURCE") if k in res},
                     indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
