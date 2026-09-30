#!/usr/bin/env python3
"""Round 2 valley (spec_sink_roots_0930_round2.md §7 R2_valley_long, §8 R1_valley_N_ladder, §10 S_small_checks).
Open = the unit's top (max_n z) at the task end > 0; k = open inputs at the task end.  mu+ = P(closed at t+1 | open at t),
mu- = P(open at t+1 | closed at t), pooled over the transitions of a window."""
import glob, json, re, sys
from pathlib import Path
import numpy as np

R2 = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/round2")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"
ZC = {"GELU": -0.7517915246935645, "SILU": -1.278464542761074}
med = lambda x: float(np.median(x)) if len(x) else float("nan")


def rates(top, lo, hi):
    oc = oo = co = cc = 0
    for t in range(lo - 1, min(hi, top.shape[0] - 1)):
        a, b = top[t] > 0, top[t + 1] > 0
        oc += (a & ~b).sum(); oo += a.sum(); co += (~a & b).sum(); cc += (~a).sum()
    return oc / max(oo, 1), co / max(cc, 1)


def valley_long():
    L = ["## §7 R2_valley_long (v4 box, 1,000 tasks)"]
    J = {}
    for f in sorted(glob.glob(str(R2 / "valley_long" / "*_v4.npz"))):
        act, seed = re.match(r"(\w+?)_s(\d)_v4", Path(f).stem).groups()
        d = np.load(f); top = d["task_zmax"].astype(float); k = np.rint(d["task_pplus"] * 1200); T = top.shape[0]
        zc = ZC[act]; deep = -6.75 if act == "GELU" else -22.0
        L.append(f"-- {act} seed {seed} ({T} tasks)")
        for lo in range(1, T + 1, 200):
            hi = min(lo + 199, T); ti = slice(lo - 1, hi)
            mp, mm = rates(top, lo, hi)
            cl = top[ti] <= 0
            bands = dict(A=((top[ti] > zc) & cl).mean(), B=((top[ti] <= zc) & (top[ti] > 2 * zc)).mean(),
                         C=((top[ti] <= 2 * zc) & (top[ti] > 3 * zc)).mean(), D=(top[ti] <= 3 * zc).mean())
            al = top[ti] > 0
            L.append(f"   t{lo}-{hi}: all-closed {cl.mean():.3f} | mu+ {mp:.4f} mu- {mm:.4f} | bands A {bands['A']:.3f} B {bands['B']:.3f} C {bands['C']:.3f} D {bands['D']:.3f}"
                     f" | top<={deep} {(top[ti] <= deep).mean():.3f} | min closed top {top[ti][cl].min() if cl.any() else float('nan'):.2f}"
                     f" | alive top med {med(top[ti][al]):.2f} k med {med(k[ti][al]):.1f}")
        # entries into the deep zone and later reopenings
        thr = -6.9 if act == "GELU" else -22.0
        ent = reo = 0
        for u in range(top.shape[1]):
            hit = np.where(top[:, u] <= thr)[0]
            if len(hit):
                ent += 1; reo += bool((top[hit[0] + 1:, u] > 0).any())
        occ65 = float((top[-1] <= (-6.5 if act == "GELU" else -20.0)).mean())
        w = slice(800, T)
        mp, mm = rates(top, 801, T)
        al = top[w] > 0
        J[f"{act}_s{seed}"] = dict(allclosed_801_1000=float((top[w] <= 0).mean()), mu_minus_801_1000=float(mm), mu_plus_801_1000=float(mp),
                                   occ_last=occ65, alive_top_med=med(top[w][al]), alive_k_med=med(k[w][al]),
                                   deep_entries=ent, deep_reopen=reo, min_closed_top=float(top[top <= 0].min()) if (top <= 0).any() else float("nan"))
        L.append(f"   unit entries to top<={thr}: {ent}, later reopened {reo} | occupancy at the last task of top<={-6.5 if act == 'GELU' else -20.0}: {occ65:.3f}"
                 f" | min closed top over the run {J[f'{act}_s{seed}']['min_closed_top']:.2f}")
    # predictions
    g = [v for k, v in J.items() if k.startswith("GELU")]; s = [v for k, v in J.items() if k.startswith("SILU")]
    if g:
        L.append(f"(a) GELU all-closed 801-1000 {[round(x['allclosed_801_1000'], 3) for x in g]} in [0.80,0.95] and mu- {[round(x['mu_minus_801_1000'], 4) for x in g]} <= 0.07")
        L.append(f"(b) GELU top<=-6.5 at the last task {[round(x['occ_last'], 3) for x in g]} >= 0.25")
        L.append(f"(e) GELU entries to <=-6.9 later reopened {[(x['deep_reopen'], x['deep_entries']) for x in g]} (< 5%); min closed top {[round(x['min_closed_top'], 2) for x in g]} > -7.3")
    if s:
        L.append(f"(c) SiLU all-closed 801-1000 {[round(x['allclosed_801_1000'], 3) for x in s]} in [0.40,0.60]; top<=-20 at the last task {[round(x['occ_last'], 3) for x in s]} < 0.10")
    if g or s:
        L.append(f"(d) alive top median {[round(x['alive_top_med'], 2) for x in g + s]} in [2,6]; alive k median {[x['alive_k_med'] for x in g + s]} in [1,2]")
    return L, J


def nladder():
    L = ["\n## §8 R1_valley_N_ladder (nk_sweep box, 200 tasks; alive = end top > 0)"]
    D = {}
    for f in sorted(glob.glob(str(R2 / "nladder" / "*_N*_s*.npz"))):
        act, N, seed = re.match(r"(\w+?)_N(\d+)_s(\d)", Path(f).stem).groups()
        D[(act, int(N), int(seed))] = np.load(f)
    J = {}
    meank_all = {}
    for act in ("gelu", "silu"):
        k0 = 2 if act == "gelu" else 3
        Ns = sorted({k[1] for k in D if k[0] == act})
        medk = {}
        for N in Ns:
            ks, kss, dr, mins, clo = [], [], [], [], []
            for seed in (0, 1):
                d = D.get((act, N, seed))
                if d is None:
                    continue
                top, k, sk = d["end_top"].astype(float), d["end_k"].astype(float), d["sw_k"].astype(float)
                al = top[150:200] > 0; ks += list(k[150:200][al]); kss += list(sk[150:200][sk[150:200] > 0])
                for t in range(50, min(199, top.shape[0] - 1)):
                    m = k[t] >= k0; dr += list(k[t + 1][m] - k[t][m])
                cl = top[50:200] <= 0; mins.append(float(top[50:200][cl].min()) if cl.any() else float("nan"))
                clo.append(float((top[150:200] <= 0).mean()))
            medk[N] = med(ks); meank = float(np.mean(ks)) if ks else float("nan")
            meank_all.setdefault(act, {})[N] = meank
            J[f"{act}_N{N}"] = dict(k_alive_med=med(ks), k_alive_mean=meank, k_sw_alive_med=med(kss), drift=float(np.mean(dr)) if dr else float("nan"),
                                    n_drift=len(dr), min_closed_top=mins, allclosed_151_200=clo)
            L.append(f"   {act} N{N:5d}: alive k median (t151-200, end) {med(ks):.1f} (switch {med(kss):.1f}) | drift E[k'-k | k>={k0}] {J[f'{act}_N{N}']['drift']:+.3f} (n {len(dr)})"
                     f" | min closed top per seed {np.round(mins, 2).tolist()} | all-closed t151-200 {np.round(clo, 3).tolist()}")
        if len(medk) >= 2 and all(v > 0 for v in medk.values()):
            sl = np.polyfit(np.log10(list(medk)), np.log10(list(medk.values())), 1)[0]
            L.append(f"   {act}: (a) medians {medk} in [1,2]; log-log slope {sl:+.3f} (|.| < 0.2)")
            mk = meank_all[act]
            sl2 = np.polyfit(np.log10(list(mk)), np.log10(list(mk.values())), 1)[0]
            L.append(f"   {act}: [descriptive] alive k mean {{{', '.join(f'{n}: {v:.2f}' for n, v in mk.items())}}}; log-log slope of the mean {sl2:+.3f}")
            J[f"{act}_slope"] = float(sl)
        if act == "gelu" and (act, 300, 0) in D and (act, 2400, 0) in D:
            d300 = np.nanmean(J["gelu_N300"]["min_closed_top"]); d2400 = np.nanmean(J["gelu_N2400"]["min_closed_top"])
            L.append(f"   gelu: (c) wall N2400 - N300 = {d2400 - d300:+.2f} (predicted +0.3 +- 0.2)")
    # determinism against the existing N 1,200 runs
    ref = {"gelu": "/home/issan/Projects/obsidian-research-data/push_lift_ladder_1layer_0928/chimera/runs_chim/GG_s0.npz",
           "silu": "/home/issan/Projects/obsidian-research-data/general_force_0929/runs/SS_s0.npz"}
    for act, p in ref.items():
        d = D.get((act, 1200, 0))
        if d is not None and Path(p).exists():
            e = np.load(p); n = min(d["end_top"].shape[0], e["end_top"].shape[0])
            same = [bool(np.array_equal(d["end_top"][t], e["end_top"][t])) for t in range(n)]
            first = same.index(False) + 1 if False in same else None
            L.append(f"   {act} N1200 s0 against {Path(p).name}: identical end tops for {sum(same)} of {n} tasks (first differing task {first})")
    return L, J


def lr_ladder():
    L = ["\n## §10 S_small_checks (2): count map (tasks 51-100, units alive at the switch; push loss k_sw - k_mid = a_p k_sw + c_p; lift k_end - k_mid = a_g k_mid + c_g)"]
    J = {}
    runs = {}
    for f in glob.glob(str(R2 / "nk_lr" / "*_lr*_s*.npz")):
        act, lr, seed = re.match(r"(\w+?)_lr([\d.e-]+)_s(\d)", Path(f).stem).groups()
        runs.setdefault((act, lr), []).append(np.load(f))
    for f in glob.glob(str(R2 / "nladder" / "*_N1200_s*.npz")):
        act, seed = re.match(r"(\w+?)_N1200_s(\d)", Path(f).stem).groups()
        runs.setdefault((act, "1e-3"), []).append(np.load(f))
    for (act, lr) in sorted(runs, key=lambda x: (x[0], float(x[1]))):
        ks, km, ke = [], [], []
        for d in runs[(act, lr)]:
            a, b, c = d["sw_k"][50:100].astype(float), d["mid_k"][50:100].astype(float), d["end_k"][50:100].astype(float)
            al = a > 0; ks += list(a[al]); km += list(b[al]); ke += list(c[al])
        ks, km, ke = map(np.array, (ks, km, ke))
        for d in runs[(act, lr)]:                      # survival: the fraction of units open at the task end, and accuracy
            al = (d["end_top"] > 0).mean(1); acc = d["acc_end"]
            first = next((t + 1 for t in range(len(al)) if al[t] == 0), None)
            L.append(f"      {act} lr {lr} run: open at the task end t1 {al[0]:.2f} t5 {al[4]:.2f} t10 {al[9]:.2f} t100 {al[99]:.2f} | "
                     f"first task with every unit closed {first} | accuracy t1 {acc[0]:.2f} t10 {acc[9]:.2f} t100 {acc[99]:.2f}")
        if len(ks) < 10:
            L.append(f"   {act} lr {lr}: too few alive units ({len(ks)})"); continue
        ap, cp = np.polyfit(ks, ks - km, 1); ag, cg = np.polyfit(km, ke - km, 1)
        J[f"{act}_lr{lr}"] = dict(a_p=float(ap), c_p=float(cp), a_g=float(ag), c_g=float(cg), k_alive_med=med(ks), n=len(ks), runs=len(runs[(act, lr)]))
        L.append(f"   {act} lr {lr:5s} ({len(runs[(act, lr)])} runs, n {len(ks)}): a_p {ap:+.2f} c_p {cp:+.2f} | a_g {ag:+.2f} c_g {cg:+.2f} | alive k median (switch) {med(ks):.1f}")
    return L, J


def codex():
    L = ["\n## §10 S_small_checks (1): codex_rerun (full-batch GD 0.01 rows: crosses z_c but max top < 0?)"]
    p = R2 / "logs" / "S1_codex_rerun.log"
    if not p.exists():
        return L + ["   (not run yet)"], {}
    rows = []
    for line in p.read_text().splitlines():
        parts = line.split()
        if len(parts) == 10 and parts[0] in ("GELU", "SiLU"):
            rows.append(parts)
    for r in rows:
        if r[3] == "gd":
            L.append(f"   {r[0]} start {r[1]} seed {r[2]}: first step top>z_c {r[5]} | first top>0 {r[6]} | max top {r[7]} | final {r[8]}")
    L.append(f"   ({len(rows)} rows; all rows in the log)")
    return L, {"rows": rows}


if __name__ == "__main__":
    out = {}
    lines = []
    for fn in (valley_long, nladder, lr_ladder, codex):
        L, J = fn(); lines += L; out[fn.__name__] = J
    txt = "\n".join(lines)
    print(txt)
    (RES / "round2_valley.txt").write_text(txt + "\n")
    (RES / "round2_valley.json").write_text(json.dumps(out, indent=1, default=str))
