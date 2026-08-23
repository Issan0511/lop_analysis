"""function_blind_0823: 同時刻ハザード＋オラクル bias 修復。

実装と判定の唯一の仕様は ``specs/spec_function_blind_0823.md``。新規学習走は行わず、
ratchet_log_0819 の保存ログと posreset_0819 の step=500k snapshot だけを読む。

実行:
  OMP_NUM_THREADS=1 .venv/bin/python -m analysis.function_blind.function_blind \
    --logs results/ratchet_log_0819/logs \
    --snapshot results/posreset_0819/snapshots/A_w100_cont_step500000.pt \
    --outdir results/function_blind_0823
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[2]
T0S = (200_000, 300_000, 400_000, 500_000, 600_000)
ENDPOINT_T0S = tuple(range(200_000, 600_001, 1_000))
HORIZON = 300_000
TAU = 0.05
BOOT_N = 10_000
H_BOOT_SEED = 20260823
O_BOOT_SEED = 20260824
ENDPOINT_BOOT_SEED = 20260825
LEGACY_BOOT_SEED = 20260827
EQUIV_MARGIN = 0.05
KICKS = (0.1, 0.25, 0.5, 1.0)
OPT_LR = 0.03
OPT_STEPS = 20_000
TRACE_STEPS = (0, 5_000, 10_000, 20_000)
GROUPS = ("low", "mid", "high")

plt.rcParams["font.family"] = ["Noto Sans CJK JP", "Noto Sans CJK TC",
                               "Noto Sans CJK KR", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def git_hash() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tensor_hash(x: torch.Tensor) -> str:
    a = x.detach().cpu().contiguous().numpy()
    return hashlib.sha256(a.tobytes()).hexdigest()


def relpath(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_logs(logdir: Path) -> list[dict]:
    out = []
    for path in sorted(logdir.glob("seed*.npz")):
        with np.load(path, allow_pickle=False) as z:
            need = ("step", "seed", "width", "period", "cos_u_mu", "w_norm",
                    "p_hat", "flip_state")
            miss = [k for k in need if k not in z.files]
            if miss:
                raise SystemExit(f"[H] {path}: missing keys {miss}")
            d = {k: np.array(z[k]) for k in need}
            d["path"] = path
            d["run_id"] = str(z["run_id"]) if "run_id" in z.files else path.stem
        out.append(d)
    out.sort(key=lambda d: int(d["seed"]))
    if [int(d["seed"]) for d in out] != list(range(10)):
        raise SystemExit("[H] seed0..9 の10本が揃っていない")
    return out


def qlabels(values: pd.Series) -> pd.Series:
    """spec §3.2 の quantile cut と同値規則を実装する。"""
    a = values.to_numpy(dtype=np.float64)
    q1, q2 = np.quantile(a, (1 / 3, 2 / 3))
    lab = np.where(a <= q1, "low", np.where(a <= q2, "mid", "high"))
    return pd.Series(lab, index=values.index, dtype="object")


def hazard_sanity(logs: list[dict]) -> dict:
    rows = []
    all_ok = True
    max_quant_err = 0.0
    max_geom_relerr = 0.0
    for d in logs:
        step = d["step"]
        p = d["p_hat"].astype(np.float64)
        cos = d["cos_u_mu"].astype(np.float64)
        wn = d["w_norm"].astype(np.float64)
        r = wn * np.sqrt(np.maximum(0.0, 1.0 - cos * cos))
        x = wn * cos
        denom = np.maximum(wn * wn, 1e-30)
        geom = float(np.max(np.abs(x * x + r * r - wn * wn) / denom))
        quant = float(np.max(np.abs(p * 32.0 - np.rint(p * 32.0))))
        present = all((t in step) and ((t + HORIZON) in step) for t in T0S)
        finite = bool(np.isfinite(r).all() and (r >= 0).all())
        good = (int(d["width"]) == 100 and int(step[0]) == 0
                and int(step[-1]) == 1_000_000 and present and finite
                and quant < 1e-7 and geom < 1e-6)
        all_ok &= good
        max_quant_err = max(max_quant_err, quant)
        max_geom_relerr = max(max_geom_relerr, geom)
        rows.append(dict(seed=int(d["seed"]), n_step=int(step.size),
                         width=int(d["width"]), period=int(d["period"]),
                         t0_present=present, finite_r=finite,
                         phat_quant_maxerr=quant, geom_max_relerr=geom,
                         ok=good))
    return dict(pass_all=bool(all_ok), rows=rows,
                phat_quant_maxerr=max_quant_err,
                geom_max_relerr=max_geom_relerr)


def build_exposures(logs: list[dict]) -> pd.DataFrame:
    rows = []
    for t0 in T0S:
        for d in logs:
            step = d["step"]
            i0s = np.flatnonzero(step == t0)
            i1s = np.flatnonzero(step == t0 + HORIZON)
            if i0s.size != 1 or i1s.size != 1:
                raise SystemExit(f"[H] seed={int(d['seed'])} t0={t0}: endpoint不在")
            i0, i1 = int(i0s[0]), int(i1s[0])
            p0 = d["p_hat"][i0].astype(np.float64)
            at_risk = p0 >= TAU
            future = d["p_hat"][i0 + 1:i1 + 1].astype(np.float64)
            ev05 = (future < TAU).any(axis=0)
            ev0 = (future == 0.0).any(axis=0)
            cos = d["cos_u_mu"][i0].astype(np.float64)
            wn = d["w_norm"][i0].astype(np.float64)
            x = wn * cos
            r = wn * np.sqrt(np.maximum(0.0, 1.0 - cos * cos))
            for unit in np.flatnonzero(at_risk):
                rows.append(dict(seed=int(d["seed"]), unit=int(unit), t0=t0,
                                 p_hat=float(p0[unit]), x=float(x[unit]),
                                 r=float(r[unit]), w_norm=float(wn[unit]),
                                 cos_u_mu=float(cos[unit]),
                                 event_dead_0_05=int(ev05[unit]),
                                 event_strict_dead=int(ev0[unit])))
    df = pd.DataFrame(rows)
    for _, idx in df.groupby("t0", sort=True).groups.items():
        df.loc[idx, "r_group"] = qlabels(df.loc[idx, "r"])
        df.loc[idx, "p_bin"] = qlabels(df.loc[idx, "p_hat"])
        df.loc[idx, "x_bin"] = qlabels(df.loc[idx, "x"])
    for c in ("r_group", "p_bin", "x_bin"):
        df[c] = pd.Categorical(df[c], categories=GROUPS, ordered=True)
    return df.sort_values(["seed", "unit", "t0"]).reset_index(drop=True)


def build_endpoint_exposures(logs: list[dict]) -> pd.DataFrame:
    """追補 §2: bulk 401起点から300k後の状態占有率を作る。"""
    rows = []
    for t0 in ENDPOINT_T0S:
        for d in logs:
            step = d["step"]
            i0s = np.flatnonzero(step == t0)
            i1s = np.flatnonzero(step == t0 + HORIZON)
            if i0s.size != 1 or i1s.size != 1:
                raise SystemExit(f"[H-end] seed={int(d['seed'])} t0={t0}: endpoint不在")
            i0, i1 = int(i0s[0]), int(i1s[0])
            p0 = d["p_hat"][i0].astype(np.float64)
            p1 = d["p_hat"][i1].astype(np.float64)
            at_risk = p0 >= TAU
            cos = d["cos_u_mu"][i0].astype(np.float64)
            wn = d["w_norm"][i0].astype(np.float64)
            x = wn * cos
            r = wn * np.sqrt(np.maximum(0.0, 1.0 - cos * cos))
            for unit in np.flatnonzero(at_risk):
                rows.append(dict(seed=int(d["seed"]), unit=int(unit), t0=t0,
                                 p_hat=float(p0[unit]), x=float(x[unit]),
                                 r=float(r[unit]), w_norm=float(wn[unit]),
                                 cos_u_mu=float(cos[unit]),
                                 end_dead_0_05=int(p1[unit] < TAU),
                                 end_strict_dead=int(p1[unit] == 0.0)))
    df = pd.DataFrame(rows)
    for _, idx in df.groupby("t0", sort=True).groups.items():
        df.loc[idx, "r_group"] = qlabels(df.loc[idx, "r"])
        df.loc[idx, "p_bin"] = qlabels(df.loc[idx, "p_hat"])
        df.loc[idx, "x_bin"] = qlabels(df.loc[idx, "x"])
    for c in ("r_group", "p_bin", "x_bin"):
        df[c] = pd.Categorical(df[c], categories=GROUPS, ordered=True)
    return df.sort_values(["seed", "unit", "t0"]).reset_index(drop=True)


def seed_group_rates(df: pd.DataFrame, outcome: str) -> pd.DataFrame:
    return (df.groupby(["seed", "r_group"], observed=False)[outcome]
            .agg(["sum", "count", "mean"]).reset_index())


def boot_hazard(seed_rates: pd.DataFrame, B: int, rng: np.random.Generator):
    seeds = np.arange(10)
    mat = np.full((10, 3), np.nan)
    for _, row in seed_rates.iterrows():
        s = int(row.seed)
        g = GROUPS.index(str(row.r_group))
        mat[s, g] = float(row["mean"])
    if not np.isfinite(mat).all():
        raise SystemExit("[H] seed×r_group に欠損がある")
    point = mat.mean(axis=0)
    draws = rng.integers(0, 10, size=(B, 10))
    bs = mat[draws].mean(axis=1)
    lo, hi = np.quantile(bs, (0.025, 0.975), axis=0)
    rd = point[2] - point[0]
    rd_bs = bs[:, 2] - bs[:, 0]
    rd_lo, rd_hi = np.quantile(rd_bs, (0.025, 0.975))
    return point, lo, hi, float(rd), float(rd_lo), float(rd_hi), draws


def classify_rd(lo: float, hi: float) -> str:
    if lo >= -EQUIV_MARGIN and hi <= EQUIV_MARGIN:
        return "EQUIV"
    if hi < 0:
        return "PROTECTIVE"
    if lo > 0:
        return "HIGHER_R_HIGHER_HAZARD"
    return "INCONCLUSIVE"


def adjusted_rd(df: pd.DataFrame, outcome: str, draws: np.ndarray):
    cells = [(p, x) for p in GROUPS for x in GROUPS]
    # [seed, cell, r(low/high), (events,n)]
    a = np.zeros((10, 9, 2, 2), dtype=np.float64)
    for si in range(10):
        ds = df[df.seed == si]
        for ci, (pb, xb) in enumerate(cells):
            dc = ds[(ds.p_bin == pb) & (ds.x_bin == xb)]
            for gi, rg in enumerate(("low", "high")):
                v = dc.loc[dc.r_group == rg, outcome].to_numpy(dtype=float)
                a[si, ci, gi, 0] = v.sum()
                a[si, ci, gi, 1] = v.size

    def one(sum_a):
        ev = sum_a[:, :, 0]
        nn = sum_a[:, :, 1]
        valid = (nn[:, 0] > 0) & (nn[:, 1] > 0)
        risks = np.divide(ev, nn, out=np.zeros_like(ev), where=nn > 0)
        w = np.minimum(nn[:, 0], nn[:, 1])
        if not valid.any() or w[valid].sum() == 0:
            return np.nan
        return float(np.sum(w[valid] * (risks[valid, 1] - risks[valid, 0]))
                     / np.sum(w[valid]))

    point = one(a.sum(axis=0))
    vals = np.empty(draws.shape[0], dtype=float)
    for j, idx in enumerate(draws):
        vals[j] = one(a[idx].sum(axis=0))
    vals = vals[np.isfinite(vals)]
    lo, hi = np.quantile(vals, (0.025, 0.975)) if vals.size else (np.nan, np.nan)
    return point, float(lo), float(hi)


def run_hazard(logs: list[dict], outdir: Path):
    sanity = hazard_sanity(logs)
    if not sanity["pass_all"]:
        raise SystemExit("[H] sanity failed; output前に中止")
    df = build_exposures(logs)
    df.to_csv(outdir / "hazard_exposures.csv", index=False)

    # H-S4: 反復曝露とリスク集合
    repeat = (df.groupby(["seed", "unit"]).size().value_counts().sort_index()
              .rename_axis("n_exposure").reset_index(name="n_unit"))
    risk_counts = (df.groupby(["seed", "t0"]).size().reset_index(name="n_at_risk"))
    sanity["repeat_exposure_distribution"] = repeat.to_dict(orient="records")
    sanity["risk_counts"] = risk_counts.to_dict(orient="records")

    rate_rows, verdict_rows, cell_rows = [], [], []
    outcome_names = (("event_dead_0_05", "dead_0.05", True),
                     ("event_strict_dead", "strict_dead", False))
    for oi, (outcome, label, primary) in enumerate(outcome_names):
        sr = seed_group_rates(df, outcome)
        rng = np.random.default_rng(H_BOOT_SEED + oi)
        point, lo, hi, rd, rd_lo, rd_hi, draws = boot_hazard(sr, BOOT_N, rng)
        for gi, rg in enumerate(GROUPS):
            d = df[df.r_group == rg]
            rate_rows.append(dict(outcome=label, r_group=rg,
                                  n_exposure=int(d.shape[0]),
                                  n_event=int(d[outcome].sum()),
                                  pooled_risk=float(d[outcome].mean()),
                                  seed_equal_risk=float(point[gi]),
                                  ci_lo=float(lo[gi]), ci_hi=float(hi[gi])))

        adj, adj_lo, adj_hi = adjusted_rd(df, outcome, draws)
        verdict_rows.append(dict(outcome=label, primary=primary,
                                 rd_high_low=rd, ci_lo=rd_lo, ci_hi=rd_hi,
                                 equiv_margin=EQUIV_MARGIN,
                                 verdict=classify_rd(rd_lo, rd_hi),
                                 rd_adj_3x3=adj, rd_adj_ci_lo=adj_lo,
                                 rd_adj_ci_hi=adj_hi))

        ctab = (df.groupby(["p_bin", "x_bin", "r_group"], observed=False)[outcome]
                .agg(n_event="sum", n_exposure="count", risk="mean").reset_index())
        ctab.insert(0, "outcome", label)
        for (pb, xb), dc in ctab.groupby(["p_bin", "x_bin"], observed=False):
            rr = {str(r.r_group): r for _, r in dc.iterrows()}
            rd_cell = (float(rr["high"].risk - rr["low"].risk)
                       if rr["high"].n_exposure > 0 and rr["low"].n_exposure > 0
                       else np.nan)
            ctab.loc[dc.index, "rd_high_low_cell"] = rd_cell
        cell_rows.append(ctab)

    rates = pd.DataFrame(rate_rows)
    verdict = pd.DataFrame(verdict_rows)
    cells = pd.concat(cell_rows, ignore_index=True)
    rates.to_csv(outdir / "hazard_rates.csv", index=False)
    verdict.to_csv(outdir / "hazard_verdict.csv", index=False)
    cells.to_csv(outdir / "hazard_cells_3x3.csv", index=False)
    risk_counts.to_csv(outdir / "hazard_risk_counts.csv", index=False)
    repeat.to_csv(outdir / "hazard_repeat_exposure.csv", index=False)
    return df, rates, verdict, cells, sanity


def run_endpoint(logs: list[dict], outdir: Path):
    """追補 §2 の300k後状態占有率。H-any の主判定は置換しない。"""
    df = build_endpoint_exposures(logs)
    df.to_csv(outdir / "endpoint_exposures.csv", index=False)
    repeat = (df.groupby(["seed", "unit"]).size().value_counts().sort_index()
              .rename_axis("n_exposure").reset_index(name="n_unit"))
    full_grid = pd.MultiIndex.from_product(
        [range(10), ENDPOINT_T0S], names=["seed", "t0"])
    risk_counts = (df.groupby(["seed", "t0"]).size().reindex(full_grid, fill_value=0)
                   .rename("n_at_risk").reset_index())
    sanity = dict(
        pass_all=bool(df.t0.nunique() == len(ENDPOINT_T0S)
                      and int(df.t0.min()) == ENDPOINT_T0S[0]
                      and int(df.t0.max()) == ENDPOINT_T0S[-1]
                      and risk_counts.shape[0] == 10 * len(ENDPOINT_T0S)),
        n_t0=int(df.t0.nunique()), n_exposure=int(df.shape[0]),
        repeat_exposure_distribution=repeat.to_dict(orient="records"),
        risk_counts=risk_counts.to_dict(orient="records"))
    if not sanity["pass_all"]:
        raise SystemExit("[H-end] sanity failed")

    rate_rows, verdict_rows, cell_rows = [], [], []
    outcomes = (("end_strict_dead", "end_strict_dead", True),
                ("end_dead_0_05", "end_dead_0.05", False))
    for oi, (outcome, label, primary) in enumerate(outcomes):
        sr = seed_group_rates(df, outcome)
        rng = np.random.default_rng(ENDPOINT_BOOT_SEED + oi)
        point, lo, hi, rd, rd_lo, rd_hi, draws = boot_hazard(sr, BOOT_N, rng)
        for gi, rg in enumerate(GROUPS):
            d = df[df.r_group == rg]
            rate_rows.append(dict(outcome=label, r_group=rg,
                                  n_exposure=int(d.shape[0]), n_event=int(d[outcome].sum()),
                                  pooled_risk=float(d[outcome].mean()),
                                  seed_equal_risk=float(point[gi]),
                                  ci_lo=float(lo[gi]), ci_hi=float(hi[gi])))
        adj, adj_lo, adj_hi = adjusted_rd(df, outcome, draws)
        verdict_rows.append(dict(outcome=label, primary=primary,
                                 rd_high_low=rd, ci_lo=rd_lo, ci_hi=rd_hi,
                                 equiv_margin=EQUIV_MARGIN,
                                 verdict=classify_rd(rd_lo, rd_hi),
                                 rd_adj_3x3=adj, rd_adj_ci_lo=adj_lo,
                                 rd_adj_ci_hi=adj_hi))
        ctab = (df.groupby(["p_bin", "x_bin", "r_group"], observed=False)[outcome]
                .agg(n_event="sum", n_exposure="count", risk="mean").reset_index())
        ctab.insert(0, "outcome", label)
        for (_, _), dc in ctab.groupby(["p_bin", "x_bin"], observed=False):
            rr = {str(r.r_group): r for _, r in dc.iterrows()}
            rd_cell = (float(rr["high"].risk - rr["low"].risk)
                       if rr["high"].n_exposure > 0 and rr["low"].n_exposure > 0
                       else np.nan)
            ctab.loc[dc.index, "rd_high_low_cell"] = rd_cell
        cell_rows.append(ctab)
    rates = pd.DataFrame(rate_rows)
    verdict = pd.DataFrame(verdict_rows)
    cells = pd.concat(cell_rows, ignore_index=True)
    rates.to_csv(outdir / "endpoint_rates.csv", index=False)
    verdict.to_csv(outdir / "endpoint_verdict.csv", index=False)
    cells.to_csv(outdir / "endpoint_cells_3x3.csv", index=False)
    risk_counts.to_csv(outdir / "endpoint_risk_counts.csv", index=False)
    repeat.to_csv(outdir / "endpoint_repeat_exposure.csv", index=False)
    return df, rates, verdict, cells, sanity


def exact_snapshot(snapshot: Path):
    snap = torch.load(snapshot, map_location="cpu", weights_only=False)
    if int(snap.get("step", -1)) != 500_000:
        raise SystemExit("[O] snapshot step != 500000")
    net = {k: snap["net"][k].detach().cpu().double().clone()
           for k in ("W", "b", "v", "c")}
    teach = {k: snap["teacher"][k].detach().cpu().double().clone()
             for k in ("W", "b", "v", "cout", "tau")}
    flip = snap["env"]["flip_state"].detach().cpu().double().clone()
    R, h, d = net["W"].shape
    f = flip.shape[1]
    bits = d - f
    pat = ((torch.arange(2 ** bits, dtype=torch.int64)[:, None]
            >> torch.arange(bits, dtype=torch.int64)) & 1).double()
    X = torch.cat([flip.unsqueeze(0).expand(pat.shape[0], -1, -1),
                   pat[:, None, :].expand(-1, R, -1)], dim=2)
    pre_t = torch.einsum("rhd,nrd->nrh", teach["W"], X) + teach["b"]
    ht = (pre_t >= teach["tau"]).double()
    y = (ht * teach["v"]).sum(dim=-1) + teach["cout"]
    return snap, net, X, y, dict(R=R, h=h, d=d, f=f, patterns=int(pat.shape[0]))


def predict(W, b, v, c, X):
    pre = torch.einsum("rhd,nrd->nrh", W, X) + b
    return (torch.relu(pre) * v).sum(dim=-1) + c, pre


def metrics(yhat: torch.Tensor, y: torch.Tensor):
    resid = yhat - y
    var_y = y.var(dim=0, unbiased=False)
    if bool((var_y <= 0).any()):
        raise SystemExit("[O] target variance <= 0")
    unfit = resid.var(dim=0, unbiased=False) / var_y
    nmse = resid.square().mean(dim=0) / var_y
    return unfit, nmse


def prepare_oracle_arms(net, X, y, rng_base=20260823):
    W0, b0, v0, c0 = (net[k] for k in ("W", "b", "v", "c"))
    current, pre0 = predict(W0, b0, v0, c0, X)
    p_hat = (pre0 > 0).double().mean(dim=0)
    masks = {"dead_0.05": p_hat < TAU, "strict_dead": p_hat == 0}
    labels, Ws, bs, cs = [], [], [], []
    kick_err = {}

    for mask_name in ("dead_0.05", "strict_dead"):
        mask = masks[mask_name]
        for k in KICKS:
            b = b0.clone()
            zmax = pre0.max(dim=0).values
            b[mask] += float(k) - zmax[mask]
            _, pre_k = predict(W0, b, v0, c0, X)
            err = (pre_k.max(dim=0).values[mask] - float(k)).abs()
            kick_err[f"repair_{mask_name}_k{k:g}"] = float(err.max()) if err.numel() else 0.0
            labels.append(f"repair_{mask_name}_k{k:g}")
            Ws.append(W0.clone()); bs.append(b); cs.append(c0.clone())

    # 容量対照。乱数消費順は spec §4.3 に固定。
    control_W = {"control_learned": W0.clone()}
    bw = math.sqrt(6.0 / W0.shape[-1])
    fresh = torch.empty_like(W0)
    shuffled = torch.empty_like(W0)
    rndrand = W0.clone()
    for s in range(W0.shape[0]):
        rng = np.random.default_rng(rng_base + s)
        fresh[s] = torch.from_numpy(rng.uniform(-bw, bw, size=W0[s].shape))
        perm = rng.permutation(W0.shape[1])
        shuffled[s] = W0[s, torch.from_numpy(perm)]
        rndrand[s, :, 15:] = torch.from_numpy(
            rng.uniform(-bw, bw, size=(W0.shape[1], W0.shape[2] - 15)))
    control_W["control_fresh_he"] = fresh
    control_W["control_row_shuffle"] = shuffled
    control_W["control_rnd_randomized"] = rndrand
    cmean = y.mean(dim=0)
    for label in ("control_learned", "control_fresh_he", "control_row_shuffle",
                  "control_rnd_randomized"):
        labels.append(label); Ws.append(control_W[label]); bs.append(torch.zeros_like(b0)); cs.append(cmean.clone())

    return (labels, torch.stack(Ws), torch.stack(bs), torch.stack(cs),
            p_hat, masks, current, kick_err)


def optimize_oracle(labels, W, b_init, c_init, v, X, y):
    """全 arm を一つの独立なバッチとして Adam で最適化する。"""
    A, R, h, d = W.shape
    base = torch.einsum("arhd,nrd->anrh", W, X)
    b = torch.nn.Parameter(b_init.clone())
    c = torch.nn.Parameter(c_init.clone())
    opt = torch.optim.Adam([b, c], lr=OPT_LR)
    var_y = y.var(dim=0, unbiased=False)

    best_nmse = torch.full((A, R), float("inf"), dtype=torch.float64)
    best_b = b.detach().clone()
    best_c = c.detach().clone()
    trace = []

    def eval_state(step):
        with torch.no_grad():
            yhat = (torch.relu(base + b[:, None]) * v[None, None]).sum(dim=-1) + c[:, None]
            resid = yhat - y[None]
            nmse = resid.square().mean(dim=1) / var_y[None]
            unfit = resid.var(dim=1, unbiased=False) / var_y[None]
            if not bool(torch.isfinite(nmse).all() and torch.isfinite(unfit).all()):
                raise SystemExit(f"[O] non-finite metric at step {step}")
            improve = nmse < best_nmse
            best_nmse[improve] = nmse[improve]
            best_b[improve] = b.detach()[improve]
            best_c[improve] = c.detach()[improve]
            for ai, label in enumerate(labels):
                trace.append(dict(step=step, arm=label,
                                  current_nmse_median=float(nmse[ai].median()),
                                  current_unfit_median=float(unfit[ai].median()),
                                  best_nmse_median=float(best_nmse[ai].median())))

    eval_state(0)
    checkpoints = set(TRACE_STEPS[1:])
    for step in range(1, OPT_STEPS + 1):
        opt.zero_grad(set_to_none=True)
        yhat = (torch.relu(base + b[:, None]) * v[None, None]).sum(dim=-1) + c[:, None]
        resid = yhat - y[None]
        per = resid.square().mean(dim=1) / var_y[None]
        loss = per.mean()
        if not bool(torch.isfinite(loss)):
            raise SystemExit(f"[O] non-finite loss at step {step}")
        with torch.no_grad():
            improve = per < best_nmse
            best_nmse[improve] = per[improve]
            best_b[improve] = b.detach()[improve]
            best_c[improve] = c.detach()[improve]
        loss.backward()
        opt.step()
        if step in checkpoints:
            eval_state(step)

    with torch.no_grad():
        ybest = (torch.relu(base + best_b[:, None]) * v[None, None]).sum(dim=-1) + best_c[:, None]
        resid = ybest - y[None]
        unfit = resid.var(dim=1, unbiased=False) / var_y[None]
        nmse = resid.square().mean(dim=1) / var_y[None]
        yinit = (torch.relu(base + b_init[:, None]) * v[None, None]).sum(dim=-1) + c_init[:, None]
        init_nmse = (yinit - y[None]).square().mean(dim=1) / var_y[None]
    if bool((nmse > init_nmse + 1e-12).any()):
        raise SystemExit("[O] best objective is worse than initialization")
    return unfit, nmse, init_nmse, pd.DataFrame(trace), best_b, best_c


def boot_median_diff(a: np.ndarray, b: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(BOOT_N, len(a)))
    d = a - b
    bs = np.median(d[idx], axis=1)
    point = float(np.median(d))
    lo, hi = np.quantile(bs, (0.025, 0.975))
    return point, float(lo), float(hi)


def boot_mean_recovery(current: np.ndarray, repair: np.ndarray, seed: int):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(current), size=(BOOT_N, len(current)))
    den = current[idx].mean(axis=1)
    num = repair[idx].mean(axis=1)
    valid = den > 0
    bs = 1.0 - num[valid] / den[valid]
    point = 1.0 - float(repair.mean()) / float(current.mean())
    lo, hi = np.quantile(bs, (0.025, 0.975))
    return point, float(lo), float(hi), int((~valid).sum())


def run_oracle(snapshot: Path, outdir: Path):
    snap, net, X, y, shape = exact_snapshot(snapshot)
    W_hash, v_hash = tensor_hash(net["W"]), tensor_hash(net["v"])
    labels, Ws, bs, cs, p_hat, masks, current, kick_err = prepare_oracle_arms(net, X, y)
    current_unfit, current_nmse = metrics(current, y)

    unfit, nmse, init_nmse, trace, _, _ = optimize_oracle(
        labels, Ws, bs, cs, net["v"], X, y)
    trace.to_csv(outdir / "oracle_trace.csv", index=False)

    rows = []
    for s in range(shape["R"]):
        rows.append(dict(seed=s, arm="current", unfit_var=float(current_unfit[s]),
                         nmse=float(current_nmse[s]), init_nmse=float(current_nmse[s])))
        for ai, label in enumerate(labels):
            rows.append(dict(seed=s, arm=label, unfit_var=float(unfit[ai, s]),
                             nmse=float(nmse[ai, s]), init_nmse=float(init_nmse[ai, s])))
    per = pd.DataFrame(rows)
    per.to_csv(outdir / "oracle_per_seed.csv", index=False)

    def vals(arm):
        return (per[per.arm == arm].sort_values("seed").unfit_var
                .to_numpy(dtype=float))

    cur = vals("current")
    repair = vals("repair_dead_0.05_k0.5")
    recovery = 1.0 - float(np.median(repair)) / float(np.median(cur))
    verdict_rows = [dict(test="O1_RECOVER", comparison="repair_dead_0.05_k0.5/current",
                         estimate=recovery, ci_lo=np.nan, ci_hi=np.nan,
                         threshold=0.90, verdict="PASS" if recovery >= 0.90 else "FAIL")]

    cur_nmse = (per[per.arm == "current"].sort_values("seed").nmse
                .to_numpy(dtype=float))
    repair_nmse = (per[per.arm == "repair_dead_0.05_k0.5"].sort_values("seed").nmse
                   .to_numpy(dtype=float))
    legacy_rec, legacy_lo, legacy_hi, n_zero = boot_mean_recovery(
        cur_nmse, repair_nmse, LEGACY_BOOT_SEED)
    verdict_rows.append(dict(test="O1_LEGACY_MEAN_NMSE",
                             comparison="repair_dead_0.05_k0.5/current",
                             estimate=legacy_rec, ci_lo=legacy_lo, ci_hi=legacy_hi,
                             threshold=0.90,
                             verdict="PASS" if legacy_rec >= 0.90 else "FAIL"))
    learned = vals("control_learned")
    all_better = True
    for j, control in enumerate(("control_fresh_he", "control_row_shuffle",
                                 "control_rnd_randomized")):
        point, lo, hi = boot_median_diff(learned, vals(control), O_BOOT_SEED + j)
        passed = hi < 0
        all_better &= passed
        verdict_rows.append(dict(test="O2_PAIR", comparison=f"learned-{control}",
                                 estimate=point, ci_lo=lo, ci_hi=hi,
                                 threshold=0.0, verdict="PASS" if passed else "FAIL"))
    verdict_rows.append(dict(test="O2_INFORMATIVE_W", comparison="all_3_controls",
                             estimate=np.nan, ci_lo=np.nan, ci_hi=np.nan,
                             threshold=np.nan, verdict="PASS" if all_better else "FAIL"))
    verdict = pd.DataFrame(verdict_rows)
    verdict.to_csv(outdir / "oracle_verdict.csv", index=False)

    # O-S2: predict() を成分式で独立に再計算（einsumを展開したbroadcast sum）。
    pre_ref = (X[:, :, None, :] * net["W"][None]).sum(dim=-1) + net["b"]
    y_ref = (torch.relu(pre_ref) * net["v"]).sum(dim=-1) + net["c"]
    pred_err = float((current - y_ref).abs().max())
    quant_err = float((p_hat * 32 - torch.round(p_hat * 32)).abs().max())
    per_check = pd.read_csv(outdir / "oracle_per_seed.csv")
    check_current = float(per_check[per_check.arm == "current"].nmse.mean())
    check_repair = float(per_check[per_check.arm == "repair_dead_0.05_k0.5"].nmse.mean())
    legacy_recalc_err = max(abs(check_current - float(cur_nmse.mean())),
                            abs(check_repair - float(repair_nmse.mean())))
    sanity = dict(
        pass_all=bool(shape == dict(R=10, h=100, d=20, f=15, patterns=32)
                      and pred_err < 1e-10 and quant_err < 1e-7
                      and W_hash == tensor_hash(net["W"])
                      and v_hash == tensor_hash(net["v"])
                      and max(kick_err.values(), default=0.0) < 1e-10
                      and legacy_recalc_err < 1e-12),
        shape=shape, prediction_max_abs_error=pred_err,
        phat_quant_maxerr=quant_err, W_hash_unchanged=W_hash == tensor_hash(net["W"]),
        v_hash_unchanged=v_hash == tensor_hash(net["v"]),
        kick_max_errors=kick_err,
        legacy_nmse_current_mean=float(cur_nmse.mean()),
        legacy_nmse_repair_mean=float(repair_nmse.mean()),
        legacy_recovery=float(legacy_rec),
        legacy_csv_recalc_max_abs_error=legacy_recalc_err,
        legacy_bootstrap_zero_denominator=n_zero,
        n_dead_0_05=[int(x) for x in masks["dead_0.05"].sum(dim=1)],
        n_strict_dead=[int(x) for x in masks["strict_dead"].sum(dim=1)])
    if not sanity["pass_all"]:
        raise SystemExit("[O] sanity failed")
    return per, verdict, trace, sanity


def make_figures(outdir, rates, cells, endpoint_rates, oracle):
    fdir = outdir / "figures"
    fdir.mkdir(exist_ok=True)

    main = rates[rates.outcome == "dead_0.05"]
    x = np.arange(3)
    y = main.seed_equal_risk.to_numpy()
    lo, hi = main.ci_lo.to_numpy(), main.ci_hi.to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.errorbar(x, y, yerr=[y - lo, hi - y], fmt="o-", capsize=4)
    ax.set_xticks(x, GROUPS); ax.set_ylabel("300k内 dead_0.05 リスク")
    ax.set_title("r 三分位別の同時刻ハザード"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(fdir / "fig_hazard_r_tertile.png", dpi=150); plt.close(fig)

    end = endpoint_rates[endpoint_rates.outcome == "end_strict_dead"]
    ey = end.seed_equal_risk.to_numpy()
    elo, ehi = end.ci_lo.to_numpy(), end.ci_hi.to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 4.0))
    ax.errorbar(x, ey, yerr=[ey - elo, ehi - ey], fmt="o-", capsize=4,
                color="tab:purple")
    ax.set_xticks(x, GROUPS); ax.set_ylabel("300k後 strict_dead 占有率")
    ax.set_title("追補: r 三分位別の300k後凍結")
    ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(fdir / "fig_endpoint_strict_r_tertile.png", dpi=150); plt.close(fig)

    cm = cells[cells.outcome == "dead_0.05"].drop_duplicates(["p_bin", "x_bin"])
    mat = np.full((3, 3), np.nan)
    for _, row in cm.iterrows():
        mat[GROUPS.index(str(row.p_bin)), GROUPS.index(str(row.x_bin))] = row.rd_high_low_cell
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    im = ax.imshow(mat, cmap="coolwarm", vmin=-.15, vmax=.15)
    ax.set_xticks(range(3), GROUPS); ax.set_yticks(range(3), GROUPS)
    ax.set_xlabel("x 三分位"); ax.set_ylabel("p_hat 三分位")
    ax.set_title("3×3セル内 RD (r high-low)")
    fig.colorbar(im, ax=ax, label="risk difference")
    fig.tight_layout(); fig.savefig(fdir / "fig_hazard_cells_3x3.png", dpi=150); plt.close(fig)

    controls = ["control_learned", "control_row_shuffle", "control_rnd_randomized",
                "control_fresh_he"]
    med = [float(oracle[oracle.arm == a].unfit_var.median()) for a in controls]
    fig, ax = plt.subplots(figsize=(7.0, 4.0))
    ax.bar(range(4), med, color=["tab:blue", "tab:orange", "tab:green", "tab:red"])
    ax.set_xticks(range(4), [a.replace("control_", "") for a in controls], rotation=15)
    ax.set_ylabel("unfit_var (10 seed median)"); ax.set_title("固定Wの bias-only 容量対照")
    fig.tight_layout(); fig.savefig(fdir / "fig_oracle_controls.png", dpi=150); plt.close(fig)

    sens = []
    for mask in ("dead_0.05", "strict_dead"):
        for k in KICKS:
            arm = f"repair_{mask}_k{k:g}"
            sens.append(dict(mask=mask, kick=k,
                             unfit=float(oracle[oracle.arm == arm].unfit_var.median())))
    sd = pd.DataFrame(sens)
    fig, ax = plt.subplots(figsize=(5.8, 4.0))
    for mask, d in sd.groupby("mask"):
        ax.plot(d.kick, d.unfit, "o-", label=mask)
    ax.set_xlabel("kick k"); ax.set_ylabel("unfit_var (10 seed median)")
    ax.set_title("オラクル修復の kick 感度"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(fdir / "fig_oracle_kick_sensitivity.png", dpi=150); plt.close(fig)


def make_summary(outdir, h_rates, h_verdict, h_cells, h_sanity,
                 endpoint_rates, endpoint_verdict, endpoint_sanity,
                 oracle, o_verdict, o_sanity, elapsed):
    lines = ["# function_blind_0823 結果", "",
             "`specs/spec_function_blind_0823.md` の再現仕様どおりに、既存ログと既存スナップショットだけを解析した。**既知の事後値を見た後の再現解析であり、盲検事前登録・独立確認ではない。**", "",
             "## 1. サニティ", "",
             f"- H: PASS={h_sanity['pass_all']}、p_hat量子化最大誤差={h_sanity['phat_quant_maxerr']:.3g}、x/r幾何最大相対誤差={h_sanity['geom_max_relerr']:.3g}",
             f"- H-end追補: PASS={endpoint_sanity['pass_all']}、起点={endpoint_sanity['n_t0']}、曝露={endpoint_sanity['n_exposure']:,}",
             f"- O: PASS={o_sanity['pass_all']}、予測式最大誤差={o_sanity['prediction_max_abs_error']:.3g}、kick最大誤差={max(o_sanity['kick_max_errors'].values()):.3g}",
             f"- 経過時間: {elapsed:.1f} sec", "", "## 2. H: 同時刻ハザード", ""]
    for outcome in ("dead_0.05", "strict_dead"):
        d = h_rates[h_rates.outcome == outcome]
        vals = " / ".join(f"{r.seed_equal_risk:.3f}" for _, r in d.iterrows())
        v = h_verdict[h_verdict.outcome == outcome].iloc[0]
        lines += [f"### {outcome}", "",
                  f"- seed等重みリスク（r low / mid / high）: **{vals}**",
                  f"- RD(high-low) = **{v.rd_high_low:+.4f}** [{v.ci_lo:+.4f}, {v.ci_hi:+.4f}] → **{v.verdict}**",
                  f"- p_hat×x 3×3調整 RD = {v.rd_adj_3x3:+.4f} [{v.rd_adj_ci_lo:+.4f}, {v.rd_adj_ci_hi:+.4f}]", ""]
    lines += ["主判定の EQUIV は95% CI全体が ±0.05 に入った場合だけ。0を含むだけなら無相関とは呼ばない。", "",
              "## 3. H-end追補: 300k後の状態占有率", "",
              "初回H-anyの天井効果を見た後に追加した事後追補で、主判定を置換しない。", ""]
    for outcome in ("end_strict_dead", "end_dead_0.05"):
        d = endpoint_rates[endpoint_rates.outcome == outcome]
        vals = " / ".join(f"{r.seed_equal_risk:.3f}" for _, r in d.iterrows())
        v = endpoint_verdict[endpoint_verdict.outcome == outcome].iloc[0]
        lines += [f"### {outcome}", "",
                  f"- seed等重み占有率（r low / mid / high）: **{vals}**",
                  f"- RD(high-low) = **{v.rd_high_low:+.4f}** [{v.ci_lo:+.4f}, {v.ci_hi:+.4f}] → **{v.verdict}**",
                  f"- p_hat×x 3×3調整 RD = {v.rd_adj_3x3:+.4f} [{v.rd_adj_ci_lo:+.4f}, {v.rd_adj_ci_hi:+.4f}]", ""]
    lines += ["## 4. O: オラクル bias 修復", "",
              f"- 旧見出しの集約（seed平均 NMSE）: **{o_sanity['legacy_nmse_current_mean']:.6f} → {o_sanity['legacy_nmse_repair_mean']:.6f}**（回復率 {o_sanity['legacy_recovery']:.3%}）", ""]
    arms = ["current", "repair_dead_0.05_k0.5", "control_learned",
            "control_row_shuffle", "control_rnd_randomized", "control_fresh_he"]
    lines += ["| arm | nmse mean | unfit_var median | nmse median |", "|---|---:|---:|---:|"]
    for arm in arms:
        d = oracle[oracle.arm == arm]
        lines.append(f"| {arm} | {d.nmse.mean():.6f} | {d.unfit_var.median():.6f} | {d.nmse.median():.6f} |")
    lines += [""]
    for _, row in o_verdict.iterrows():
        est = "NA" if pd.isna(row.estimate) else f"{row.estimate:+.6f}"
        ci = "" if pd.isna(row.ci_lo) else f" [{row.ci_lo:+.6f}, {row.ci_hi:+.6f}]"
        lines.append(f"- **{row.test} {row.verdict}** ({row.comparison}): {est}{ci}")
    lines += ["", "対照3種は本specで新しく固定した操作の結果であり、元の使い捨て解析の対照値の再現ではない。", "",
              "## 5. 解釈と限界", "",
              "- H は保存記録点上の300k累積転帰で、一時消灯と反復曝露を含む。連続時間ハザードでも恒久死でもない。",
              "- r は入力応答重みの大きさの代理で、教師への因果的寄与そのものではない。",
              "- O は W と v を固定したオラクル容量診断。動的再開を含まず、学習手法ではない。",
              "- 本結果から dead unit 単体の有用性、修復効果の持続、condB・他幅への一般化を主張しない。",
              "- 既知の事後概数との一致・不一致にかかわらず、この出力をそのまま正本とする。", ""]
    (outdir / "summary.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="results/ratchet_log_0819/logs")
    ap.add_argument("--snapshot", default="results/posreset_0819/snapshots/A_w100_cont_step500000.pt")
    ap.add_argument("--outdir", default="results/function_blind_0823")
    args = ap.parse_args(argv)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("OMP_NUM_THREADS=1 が必要")
    logdir = (ROOT / args.logs).resolve() if not Path(args.logs).is_absolute() else Path(args.logs)
    snapshot = (ROOT / args.snapshot).resolve() if not Path(args.snapshot).is_absolute() else Path(args.snapshot)
    outdir = (ROOT / args.outdir).resolve() if not Path(args.outdir).is_absolute() else Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    t_start = time.time()

    logs = load_logs(logdir)
    _, h_rates, h_verdict, h_cells, h_sanity = run_hazard(logs, outdir)
    _, endpoint_rates, endpoint_verdict, endpoint_cells, endpoint_sanity = run_endpoint(
        logs, outdir)
    oracle, o_verdict, _, o_sanity = run_oracle(snapshot, outdir)
    make_figures(outdir, h_rates, h_cells, endpoint_rates, oracle)
    elapsed = time.time() - t_start
    make_summary(outdir, h_rates, h_verdict, h_cells, h_sanity,
                 endpoint_rates, endpoint_verdict, endpoint_sanity,
                 oracle, o_verdict, o_sanity, elapsed)

    spec = ROOT / "specs/spec_function_blind_0823.md"
    addendum = ROOT / "specs/spec_function_blind_0823_addendum.md"
    inputs = {relpath(d["path"]): sha256(d["path"]) for d in logs}
    inputs[relpath(snapshot)] = sha256(snapshot)
    meta = dict(git_hash=git_hash(), spec=relpath(spec), spec_sha256=sha256(spec),
                addendum=relpath(addendum), addendum_sha256=sha256(addendum),
                inputs_sha256=inputs, elapsed_sec=elapsed,
                omp_threads=os.environ.get("OMP_NUM_THREADS"),
                python=sys.version, platform=platform.platform(),
                numpy=np.__version__, pandas=pd.__version__, torch=torch.__version__,
                rng=dict(hazard=H_BOOT_SEED, oracle=O_BOOT_SEED,
                         endpoint=ENDPOINT_BOOT_SEED, legacy=LEGACY_BOOT_SEED,
                         controls="20260823 + seed"),
                constants=dict(t0=list(T0S), horizon=HORIZON, tau=TAU,
                               endpoint_t0_start=ENDPOINT_T0S[0],
                               endpoint_t0_stop=ENDPOINT_T0S[-1],
                               endpoint_t0_step=1_000,
                               bootstrap_n=BOOT_N, equiv_margin=EQUIV_MARGIN,
                               kicks=list(KICKS), opt_lr=OPT_LR, opt_steps=OPT_STEPS),
                sanity=dict(hazard=h_sanity, endpoint=endpoint_sanity, oracle=o_sanity))
    (outdir / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
    print(f"[function_blind] wrote {outdir} ({elapsed:.1f}s)")


if __name__ == "__main__":
    main()
