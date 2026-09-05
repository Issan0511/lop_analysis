# -*- coding: utf-8 -*-
"""snake_flip_0906 の登録判定（spec `specs/spec_snake_flip_0906.md` §4）。

    OMP_NUM_THREADS=1 PYTHONPATH=. .venv/bin/python -m src.snake_flip_analyze_0906
    ... --w100 results/snake_flip_0906/logs --w5 results/_diag_w5_snake_0905 --out results/snake_flip_0906

H1（`SN1` 対 `SN`）: (a) 幅 5 の対応 seed Δlog10U、(b) 幅 100 の罠占有率差、(c) mob 中央値差。
H2（`SNA05`/`SNA025` 対 `SN`）: (i) ゲート、(ii) 沈まない、(iii) 罠の位置、(iv) 幅 5 の適合。
REPORT: 罠まわりのドリフト場の符号反転、W と 1−sinc(αW)、‖w‖・|v|。
CI は seed bootstrap 2000 回・percentile 95%・`default_rng(20260906)`。numpy のみ。
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .common import ROOT, load_config
from .mlp2_phase0b import _window_indices

CFG = Path(ROOT) / "configs" / "snake_flip_0906.yaml"
T = 10_000
ALPHA = 1.0
Z1 = -math.pi / (4.0 * ALPHA)                 # 第 1 罠（φ'=0・φ''=0）
HALF = math.pi / (4.0 * ALPHA)                # 罠占有の帯 |z̄−z1| < π/4α（spec §3）
RNG_SEED = 20260906
NBOOT = 2000


# ---------------------------------------------------------------------------
# 読み込み
# ---------------------------------------------------------------------------
def _load_arm(logdir: Path, arm: str) -> list[dict]:
    files = sorted(logdir.glob(f"{arm}_seed*.npz"))
    out = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        out.append({k: z[k] for k in z.files})
    return out


def _tail(seeds: list[dict], key: str, tasks: tuple[int, int]) -> np.ndarray:
    """(seed, unit) の窓平均。"""
    rows = []
    for z in seeds:
        idx = _window_indices(z["step"], T, list(tasks))
        rows.append(z[key][idx].astype(np.float64).mean(axis=0))
    return np.stack(rows)                     # (S, h)


def _scalar_tail(seeds: list[dict], key: str, tasks: tuple[int, int]) -> np.ndarray:
    rows = []
    for z in seeds:
        idx = _window_indices(z["step"], T, list(tasks))
        rows.append(float(np.nanmean(z[key][idx].astype(np.float64))))
    return np.asarray(rows)                   # (S,)


def _alive(seeds: list[dict], tasks) -> np.ndarray:
    return _tail(seeds, "layer1_denom", tasks) > 0.25


# ---------------------------------------------------------------------------
# 統計
# ---------------------------------------------------------------------------
def boot_median(per_seed_values: np.ndarray, rng: np.random.Generator,
                stat=np.median) -> tuple[float, tuple[float, float]]:
    """seed 単位の復元抽出。`per_seed_values` は (S,) か (S, n) の配列（unit をプール）。"""
    S = per_seed_values.shape[0]
    def pooled(idx):
        v = per_seed_values[idx]
        v = v.reshape(-1) if v.ndim > 1 else v
        v = v[np.isfinite(v)]
        return float(stat(v)) if v.size else np.nan
    point = pooled(np.arange(S))
    bs = np.array([pooled(rng.integers(0, S, S)) for _ in range(NBOOT)])
    return point, (float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5)))


def paired_diff(a: np.ndarray, b: np.ndarray, rng) -> tuple[float, tuple[float, float]]:
    """seed 対応の差（a−b）。a, b は (S,) または (S, h)（unit 対応・両腕で ALIVE の unit だけ渡す）。"""
    d = a - b
    return boot_median(d, rng)


def ci_within(ci, lo, hi) -> bool:
    return np.isfinite(ci[0]) and np.isfinite(ci[1]) and ci[0] >= lo and ci[1] <= hi


# ---------------------------------------------------------------------------
# 幅 100
# ---------------------------------------------------------------------------
def w100_summary(seeds: list[dict], tasks=(451, 500)) -> dict:
    zb = _tail(seeds, "layer1_zbar", tasks); zmax = _tail(seeds, "layer1_zmax", tasks)
    zmin = _tail(seeds, "layer1_zmin", tasks); mob = _tail(seeds, "layer1_mob", tasks)
    v = _tail(seeds, "layer1_v_unit", tasks); w = _tail(seeds, "layer1_w_norm", tasks)
    alive = _alive(seeds, tasks)
    occ = np.abs(zb - Z1) < HALF
    occ_tight = np.abs(zb - Z1) < 0.3
    W = zmax - zmin
    return dict(zbar=zb, zmax=zmax, zmin=zmin, mob=mob, v=v, w=w, alive=alive,
                occ=occ, occ_tight=occ_tight, W=W,
                nan_seeds=int(sum(not np.all(np.isfinite(z["layer1_zbar"])) for z in seeds)))


def settle_drift(seeds: list[dict], key: str, fn) -> list[float]:
    vals = []
    for tasks in ((301, 350), (376, 425), (451, 500)):
        x = _tail(seeds, key, tasks); a = _alive(seeds, tasks)
        vals.append(float(fn(x[a])))
    return vals


def drift_field(seeds: list[dict], band=0.3, t_from=100) -> dict:
    """罠まわりのドリフト場: タスク終端記録の Δz̄ を phase = z̄−z1 の帯で条件づける（REPORT）。"""
    below, above = [], []
    for z in seeds:
        step = z["step"]; zb = z["layer1_zbar"].astype(np.float64)
        idx = np.array([int(np.argmin(np.abs(step - t * T))) for t in range(t_from, 500)])
        cur, nxt = zb[idx[:-1]], zb[idx[1:]]
        ph = cur - Z1; d = nxt - cur
        below.append(d[(ph > -band) & (ph < 0)]); above.append(d[(ph > 0) & (ph < band)])
    rng = np.random.default_rng(RNG_SEED)
    def summ(lst):
        # seed 単位 bootstrap（各 seed の帯内 Δ をプール）
        S = len(lst)
        def pooled(ix):
            v = np.concatenate([lst[i] for i in ix]); return float(np.median(v)) if v.size else np.nan
        pt = pooled(np.arange(S)); bs = [pooled(rng.integers(0, S, S)) for _ in range(NBOOT)]
        return pt, (float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))), int(sum(len(x) for x in lst))
    b, a = summ(below), summ(above)
    return dict(below=b, above=a,
                sign_reversal=bool(b[1][0] > 0 and a[1][1] < 0))


# ---------------------------------------------------------------------------
# 幅 5
# ---------------------------------------------------------------------------
def w5_logU(logdir: Path, tasks=(491, 500)) -> np.ndarray:
    seeds = _load_arm(logdir, "*") if False else None
    files = sorted(logdir.glob("*_seed*.npz"))
    vals = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        idx = _window_indices(z["step"], T, list(tasks))
        u = z["unfit"][idx].astype(np.float64)
        vals.append(np.log10(np.nanmean(u)) if np.all(np.isfinite(z["layer1_zbar"][idx])) else np.nan)
    return np.asarray(vals)


def w5_mob(logdir: Path, tasks=(491, 500)) -> np.ndarray:
    files = sorted(logdir.glob("*_seed*.npz")); rows = []
    for f in files:
        z = np.load(f, allow_pickle=True); idx = _window_indices(z["step"], T, list(tasks))
        rows.append(z["layer1_mobility"][idx].astype(np.float64).mean(0))
    return np.stack(rows)


# ---------------------------------------------------------------------------
# 本体
# ---------------------------------------------------------------------------
def analyze(w100: Path, w5: Path, out: Path) -> dict:
    cfg = load_config(str(CFG))
    refs = cfg["width5_arms"]["references"]
    rng = np.random.default_rng(RNG_SEED)
    rows: list[dict] = []
    def row(**kw):
        rows.append(kw); return kw

    # ---- 幅 100 ----
    arms = ["SN_a1_1216", "SN1_a1_1216", "SNA05_a1_1216", "SNA025_a1_1216"]
    S = {a: _load_arm(w100, a) for a in arms}
    missing = [a for a in arms if not S[a]]
    Q = {a: w100_summary(S[a]) for a in arms if S[a]}
    for a in Q:
        q = Q[a]; al = q["alive"]
        occ_pt, occ_ci = boot_median(q["occ"][..., None].astype(float).mean(axis=1) if False else q["occ"].astype(float), rng, stat=np.mean)
        zb_pt, zb_ci = boot_median(np.where(al, q["zbar"], np.nan), rng)
        mob_pt, mob_ci = boot_median(np.where(al, q["mob"], np.nan), rng)
        W_pt, _ = boot_median(np.where(al, q["W"], np.nan), rng)
        x = ALPHA * W_pt; trap_gate = 1.0 - math.sin(x) / x if x > 0 else np.nan
        row(section="w100", arm=a, judgment="summary", role="report", label="—",
            n_alive=int(al.sum()), death_rate=float(1 - al.mean()), nan_seeds=q["nan_seeds"],
            zbar_med=zb_pt, zbar_ci=zb_ci, mob_med=mob_pt, mob_ci=mob_ci,
            occ=occ_pt, occ_ci=occ_ci, occ_tight=float(q["occ_tight"].mean()),
            W_med=W_pt, trap_gate_pred=trap_gate,
            w_med=float(np.median(q["w"][al])), v_abs_med=float(np.median(np.abs(q["v"][al]))),
            iqr_zbar=float(np.subtract(*np.percentile(q["zbar"][al], [75, 25]))),
            settle_zbar=settle_drift(S[a], "layer1_zbar", np.median),
            drift=drift_field(S[a]))

    # ---- H1 ----
    h1 = {}
    if all(a in Q for a in ("SN_a1_1216", "SN1_a1_1216")):
        A, B = Q["SN1_a1_1216"], Q["SN_a1_1216"]
        both = A["alive"] & B["alive"]
        occ_d = paired_diff(np.where(both, A["occ"], np.nan).astype(float), np.where(both, B["occ"], np.nan).astype(float), rng)
        # 占有率は seed ごとの平均の差で（unit 中央値は 0/1 で退化する）
        occ_seed = A["occ"].mean(1) - B["occ"].mean(1)
        occ_pt, occ_ci = boot_median(occ_seed, rng, stat=np.mean)
        mob_d = paired_diff(np.where(both, A["mob"], np.nan), np.where(both, B["mob"], np.nan), rng)
        h1.update(occ=(occ_pt, occ_ci), mob=mob_d)
    # 幅 5 (a)
    ref_sn = Path(refs["SN5_a1_lr00037"]) / "logs"
    u_sn = w5_logU(ref_sn) if ref_sn.exists() else None
    for tag, role in (("SN1_a1", "confirmatory"), ("SN1_a1_lr00037", "report")):
        d = w5 / tag / "logs"
        if d.exists() and u_sn is not None:
            u = w5_logU(d)
            pt, ci = paired_diff(u, u_sn, rng)
            row(section="w5", arm=tag, judgment="H1-a dlog10U vs SN5_a1", role=role, point=pt, ci=ci,
                n_seeds=int(np.isfinite(u).sum()), label=("EQUIV" if ci_within(ci, -0.1, 0.1) else
                                                          "DIFFERS" if (ci[0] > 0 or ci[1] < 0) and abs(pt) > 0.1 else "INCONCLUSIVE"))
            if role == "confirmatory": h1["U"] = (pt, ci)
    if len(h1) == 3:
        ok = ci_within(h1["U"][1], -0.1, 0.1) and ci_within(h1["occ"][1], -0.05, 0.05) and ci_within(h1["mob"][1], -0.05, 0.05)
        bad = any((c[0] > 0 or c[1] < 0) and abs(p) > b for (p, c), b in
                  ((h1["U"], 0.1), (h1["occ"], 0.05), (h1["mob"], 0.05)))
        lab = "ONE_FLIP_EQUIV" if ok else ("LOBE_MATTERS" if bad else "INCONCLUSIVE")
    else:
        lab = "NOT_DETERMINED"
    row(section="H1", arm="SN1 vs SN", judgment="H1 overall", role="confirmatory", label=lab,
        U=h1.get("U"), occ=h1.get("occ"), mob=h1.get("mob"))

    # ---- H2 ----
    u_lr = None
    ref_lr = Path(refs["LR5x_lr00037"]) / "logs"
    if ref_lr.exists():
        u_lr = w5_logU(ref_lr)
    h2 = {}
    for a, tag in (("SNA05_a1_1216", "SNA05_a1"), ("SNA025_a1_1216", "SNA025_a1")):
        if a not in Q or "SN_a1_1216" not in Q:
            row(section="H2", arm=a, judgment="H2 overall", role="confirmatory" if "05" in a and "025" not in a else "report", label="NOT_DETERMINED"); continue
        A, B = Q[a], Q["SN_a1_1216"]; al = A["alive"]
        mobA, _ = boot_median(np.where(al, A["mob"], np.nan), rng); mobB, _ = boot_median(np.where(B["alive"], B["mob"], np.nan), rng)
        gate_ok = mobA >= mobB - 0.05
        zb_pt, zb_ci = boot_median(np.where(al, A["zbar"], np.nan), rng)
        settle = settle_drift(S[a], "layer1_zbar", np.median)
        drift = max(settle) - min(settle); monotone = (settle[0] > settle[1] > settle[2])
        no_sink = zb_pt > -1.5 and not (monotone and drift > (zb_ci[1] - zb_ci[0]))
        trap_at = abs(zb_pt - Z1) <= 0.3
        lab = "FLIP_SUFFICES" if (gate_ok and no_sink and trap_at) else (
              "ZERO_NEEDED" if (not no_sink or not trap_at) else "GATE_FAIL")
        if A["nan_seeds"] > 3: lab = "NOT_DETERMINED"
        fit = {}
        d = w5 / tag / "logs"
        if d.exists() and u_sn is not None:
            u = w5_logU(d); p1, c1 = paired_diff(u, u_sn, rng)
            fit["vs_SN"] = (p1, c1, "FIT_EQUAL" if ci_within(c1, -0.1, 0.1) else "FIT_WORSE" if c1[0] > 0 else "FIT_BETTER" if c1[1] < 0 else "FIT_INCONCLUSIVE")
            if u_lr is not None:
                p2, c2 = paired_diff(u, u_lr, rng)
                fit["vs_LR5x"] = (p2, c2, "BEATS_LEAKY" if c2[1] < 0 else "NOT")
            fit["U_med"] = float(10 ** np.nanmedian(u))
        row(section="H2", arm=a, judgment="H2 overall", role="confirmatory" if a == "SNA05_a1_1216" else "report",
            label=lab, gate_ok=gate_ok, mob=(mobA, mobB), no_sink=no_sink, zbar=(zb_pt, zb_ci),
            settle=settle, trap_at_inflection=trap_at, z1=Z1, fit=fit)
        h2[a] = lab

    result = dict(rows=rows, missing=missing, H1=lab if False else rows[-3 if h2 else -1]["label"], H2=h2)
    out.mkdir(parents=True, exist_ok=True)
    (out / "verdict.json").write_text(json.dumps(result, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    with (out / "verdict.csv").open("w", newline="", encoding="utf-8") as fh:
        keys = sorted({k for r in rows for k in r})
        wr = csv.DictWriter(fh, fieldnames=keys); wr.writeheader()
        for r in rows: wr.writerow({k: json.dumps(r.get(k), default=str, ensure_ascii=False) if not isinstance(r.get(k), (str, int, float, type(None))) else r.get(k) for k in keys})
    (out / "summary.md").write_text(_markdown(rows, missing), encoding="utf-8")
    return result


def _fmt(x, nd=3):
    if x is None: return "—"
    if isinstance(x, tuple) and len(x) == 2 and isinstance(x[1], tuple):
        return f"{x[0]:.{nd}f} [{x[1][0]:.{nd}f}, {x[1][1]:.{nd}f}]"
    if isinstance(x, float): return f"{x:.{nd}f}"
    return str(x)


def _markdown(rows, missing) -> str:
    L = ["# snake_flip_0906 判定まとめ（spec §4）", "",
         f"z₁ = −π/4α = {Z1:.3f}（α=1）。窓: 幅 100 = タスク 451–500、幅 5 = 491–500。ALIVE = denom>0.25。CI = seed bootstrap 2000 回。", ""]
    if missing: L += [f"**欠落腕**: {missing}", ""]
    L += ["## 幅 100 の腕", "", "| 腕 | 生存 | z̄ 中央値 | mob | 罠占有 (±π/4α) | 罠占有 (±0.3) | W | 1−sinc(αW) | ‖w‖ | \\|v\\| | z̄ IQR | settle z̄ (3 窓) | 場: 下帯 / 上帯 | 符号反転 |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        if r.get("section") != "w100": continue
        d = r["drift"]
        L.append(f"| `{r['arm']}` | {r['n_alive']} ({1-r['death_rate']:.2f}) | {_fmt((r['zbar_med'], r['zbar_ci']))} | {_fmt((r['mob_med'], r['mob_ci']))} | {_fmt((r['occ'], r['occ_ci']))} | {r['occ_tight']:.2f} | {r['W_med']:.2f} | {r['trap_gate_pred']:.2f} | {r['w_med']:.2f} | {r['v_abs_med']:.3f} | {r['iqr_zbar']:.2f} | {['%.2f' % v for v in r['settle_zbar']]} | {_fmt(d['below'][:2])} / {_fmt(d['above'][:2])} | **{d['sign_reversal']}** |")
    L += ["", "## 幅 5 の適合（Δlog₁₀U・対応 seed）", "", "| 腕 | 判定 | 役割 | Δlog₁₀U | ラベル |", "|---|---|---|---|---|"]
    for r in rows:
        if r.get("section") != "w5": continue
        L.append(f"| `{r['arm']}` | {r['judgment']} | {r['role']} | {_fmt((r['point'], r['ci']))} | **{r['label']}** |")
    L += ["", "## H1 / H2", ""]
    for r in rows:
        if r.get("section") == "H1":
            L.append(f"- **H1 `{r['label']}`** — Δlog₁₀U {_fmt(r.get('U'))}・罠占有差 {_fmt(r.get('occ'))}・mob 差 {_fmt(r.get('mob'))}")
        if r.get("section") == "H2":
            f = r.get("fit", {}) or {}
            L.append(f"- **H2 `{r['arm']}` → `{r['label']}`**（{r['role']}）: gate_ok={r.get('gate_ok')} (mob {r.get('mob')})・no_sink={r.get('no_sink')} (z̄ {_fmt(r.get('zbar'))}・settle {r.get('settle')})・trap_at_inflection={r.get('trap_at_inflection')}"
                     + (f"・適合 vs SN {_fmt(f['vs_SN'][:2])} **{f['vs_SN'][2]}**" if 'vs_SN' in f else "")
                     + (f"・vs LR5x {_fmt(f['vs_LR5x'][:2])} **{f['vs_LR5x'][2]}**" if 'vs_LR5x' in f else "")
                     + (f"・U 中央値 {f['U_med']:.4f}" if 'U_med' in f else ""))
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w100", default=str(Path(ROOT) / "results/snake_flip_0906/logs"))
    ap.add_argument("--w5", default=str(Path(ROOT) / "results/_diag_w5_snake_0905"))
    ap.add_argument("--out", default=str(Path(ROOT) / "results/snake_flip_0906"))
    a = ap.parse_args()
    r = analyze(Path(a.w100), Path(a.w5), Path(a.out))
    print(json.dumps({k: r[k] for k in ("H1", "H2", "missing")}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
