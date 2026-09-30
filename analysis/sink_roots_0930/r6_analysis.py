#!/usr/bin/env python3
"""Round 6 (spec_sink_roots_0930_round6.md): RR-1..RR-4 (two-layer replays), G6 (cap shrink ledger), K_sweep, RR4 two-layer caps.
Readings are those of the spec §2-§3; 'live' = layer-2 units with an open input at the switch."""
import glob, json, re, sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930")
R6 = RAW / "round6"
RES = ROOT / "results" / "sink_roots_0930"
sys.path.insert(0, str(ROOT / "src"))


def rr1():
    L = ["## RR-1 replay_factors (actual labels, full arm, 200 updates)"]
    p13 = {}
    for f in sorted(glob.glob(str(R6 / "RR1" / "rr1_*.npz"))):
        d = np.load(f); name = Path(f).stem[4:]; live = d["live"]
        taus = list(d["led_taus"]); own = d["led_own"][:, live].sum(1); up = d["led_up"][:, live].sum(1)
        a6b = {t: d["a6b"][list(d["taus"]).index(t)] for t in (1, 2, 3)}
        r = []
        for t in (1, 2, 3):
            i = taus.index(t); sh = up[i] / (own[i] + up[i])
            r.append(f"t{t}: share real {sh:.3f} A6B {a6b[t][0]:.3f} (d {sh - a6b[t][0]:+.3f}) | own real/A6B {own[i] / a6b[t][1]:.2f} up {up[i] / a6b[t][2]:.2f}")
        ok11 = all(abs(own[taus.index(t)] / a6b[t][1] - 1) <= 0.15 and abs(up[taus.index(t)] / a6b[t][2] - 1) <= 0.15
                   and abs(up[taus.index(t)] / (own[taus.index(t)] + up[taus.index(t)]) - a6b[t][0]) <= 0.05 for t in (1, 2, 3))
        L.append(f"   {name}: replay fidelity {float(d['fid']):.1e} | " + " | ".join(r) + f" | P1-1 {'HIT' if ok11 else 'MISS'}")
        cors = []
        for tau, key in ((50, "pred50"), (200, "pred200")):
            c = np.corrcoef(d["up_path"][tau - 1][live], d[key][live])[0, 1]; cors.append(c)
        C = d["C"]; drop = 1 - C[25] / C[1]; p13[name] = drop
        own_s = d["own"][:, live].sum(1); up_s = d["up"][:, live].sum(1); rr = up_s / own_s
        G = d["G"]; pred = G * C
        rn = rr / rr[1:6].mean(); pn = pred / pred[1:6].mean()
        dev = float(np.mean(np.abs(rn[11:51] - pn[11:51]) / np.abs(rn[11:51])))
        L.append(f"      P1-2 corr(path upstream, uniform-shift predictor) tau50 {cors[0]:+.3f} tau200 {cors[1]:+.3f} | "
                 f"P1-3 C(1) {C[1]:.3f} C(25) {C[25]:.3f} drop {drop:+.2f} | P1-4 mean |ratio - G C| / ratio over 11-50 updates {dev:.2f} "
                 f"| G(1)->G(50) {G[1]:.3f}->{G[50]:.3f} E(1)->E(50) {d['E'][1]:.3f}->{d['E'][50]:.3f}")
    return L


def rr2():
    L = ["\n## RR-2 gate_frozen_backward (16 label draws, 200 updates; F1 ledger, label means)"]
    for f in sorted(glob.glob(str(R6 / "RR2" / "rr2_*.npz"))):
        d = np.load(f); name = Path(f).stem[4:]; lv = d["live"]
        sh = {a: d[f"{a}_up200"][lv].sum() / (d[f"{a}_own200"][lv].sum() + d[f"{a}_up200"][lv].sum()) for a in ("nat", "frozen")}
        sh50 = {a: d[f"{a}_up50"][lv].sum() / (d[f"{a}_own50"][lv].sum() + d[f"{a}_up50"][lv].sum()) for a in ("nat", "frozen")}
        dm1 = {a: float(np.median(d[f"{a}_dm1"])) for a in ("nat", "frozen")}
        L.append(f"   {name}: up share tau50 nat {sh50['nat']:.3f} frozen {sh50['frozen']:.3f} | tau200 nat {sh['nat']:.3f} frozen {sh['frozen']:.3f} "
                 f"| up amount frozen/nat {d['frozen_up200'][lv].sum() / d['nat_up200'][lv].sum():.2f} | dm1 median nat {dm1['nat']:+.3f} frozen {dm1['frozen']:+.3f} "
                 f"(frozen - nat {dm1['frozen'] - dm1['nat']:+.3f}) | dm2 down nat {np.mean(d['nat_dm2'] < 0):.3f} frozen {np.mean(d['frozen_dm2'] < 0):.3f}")
    return L


def rr3():
    L = ["\n## RR-3 center_input_full (actual labels, 200 updates; the original arms from round 1's replay)"]
    for f in sorted(glob.glob(str(R6 / "RR3" / "rr3_*.npz"))):
        d = np.load(f); name = Path(f).stem[4:]
        o = np.load(RAW / "round1" / "replay" / f"replay_{name}.npy", allow_pickle=True)[0]
        of, ou = o["full"][200][1], o["up"][200][1]
        cf, cu = d["full_200"], d["up_200"]
        L.append(f"   {name}: push median orig full {np.median(of):+.3f} centred full {np.median(cf):+.3f} (ratio {np.median(cf) / np.median(of):.2f}) | "
                 f"down orig full {np.mean(of < 0):.2f} orig up {np.mean(ou < 0):.2f} centred full {np.mean(cf < 0):.2f} centred up {np.mean(cu < 0):.2f}")
    return L


def rr4():
    L = ["\n## RR-4 refit_2layer (12 label draws, 6,000 updates; means over draws)"]
    for f in sorted(glob.glob(str(R6 / "RR4" / "rr4_*.npz"))):
        d = np.load(f); name = Path(f).stem[4:]
        end = {a: d[a][:, -1, :].mean(0) for a in ("vc", "l2", "full")}
        line = f"   {name}: end S2>0 / S2oh>0 / acc " + " | ".join(f"{a} {v[0]:.3f}/{v[1]:.3f}/{v[2]:.2f}" for a, v in end.items())
        # the full arm's S2oh>0 at the vc arm's final accuracy (mean curves over draws)
        cf = d["full"].mean(0); accv = end["vc"][2]
        i = int(np.argmin(np.abs(cf[:, 2] - accv)))
        line += f" | full at acc {cf[i, 2]:.2f} (update {(i + 1) * 25}): S2oh>0 {cf[i, 1]:.3f}"
        L.append(line)
    return L


def g6():
    L = ["\n## G6 cap_shrink_ledger (replay_geom --arm wcap + v_top ledger, ELU, tasks 5-30; medians over units, means over tasks)"]
    for d_ in sorted(glob.glob(str(R6 / "G6" / "ELU_s*"))):
        f = Path(d_) / "replay.npz"
        if not f.exists():
            continue
        d = np.load(f); cap2 = (d["W"][0, 0].astype(np.float64) ** 2).sum(1)          # |w|^2 at task 5's head = the cap^2
        sh_sq = d["L_P_sh_sq"] + d["L_R_sh_sq"]; sh_dm = d["L_P_sh_dm"] + d["L_R_sh_dm"]
        adam_dm = d["L_P_dm"] + d["L_R_dm"] + d["L_P_db"] + d["L_R_db"]
        vt_a = d["L_P_vt_adam"] + d["L_R_vt_adam"]; vt_s = d["L_P_vt_sh"] + d["L_R_vt_sh"]
        mlnf = -0.5 * sh_sq / cap2[None, :]                                       # -sum ln f ~ -1/2 sum dnorm2 / cap^2 (per task, per unit)
        L.append(f"   {Path(d_).name}: -sum ln f per task {np.mean(np.median(mlnf, 1)):.3f} | dm shrink {np.mean(np.median(sh_dm, 1)):+.3f} Adam {np.mean(np.median(adam_dm, 1)):+.3f} "
                 f"sum {np.mean(np.median(sh_dm + adam_dm, 1)):+.3f} | dv_top Adam {np.mean(np.median(vt_a, 1)):+.3f} shrink {np.mean(np.median(vt_s, 1)):+.3f} "
                 f"sum {np.mean(np.median(vt_a + vt_s, 1)):+.3f} | shrink events per unit-task {np.mean(np.median(d['L_P_sh_n'] + d['L_R_sh_n'], 1)):.0f}")
    return L


def ksweep():
    L = ["\n## K_sweep_label_luck (v4 GELU, tasks 51-100 switches; favourable = v_i.(e_y - p) > 0)"]
    mu = {}
    for K in (2, 5, 10, 20):
        fs = sorted(glob.glob(str(R6 / "K_sweep" / f"GELU_K{K}_s*_v4.npz")))
        if not fs:
            continue
        ko, kf, top_fav, mps = [], [], [], []
        for f in fs:
            d = np.load(f)
            ko.append(d["sw_kopen"][49:99]); kf.append(d["sw_kfav"][49:99])
            al = d["sw_kopen"][49:99] > 0; top_fav.append((d["sw_demand_top"][49:99] < 0)[al])
            top = d["task_zmax"]; oc = oo = 0
            for t in range(50, 99):
                x, y = top[t] > 0, top[t + 1] > 0; oc += (x & ~y).sum(); oo += x.sum()
            mps.append(oc / max(oo, 1))
        ko, kf = np.concatenate(ko).ravel(), np.concatenate(kf).ravel()
        fbar = kf[ko > 0].sum() / ko[ko > 0].sum()
        qs = []
        for k in (1, 2, 3, 4):
            m = ko == k
            if m.sum() >= 20:
                qs.append((k, float(np.mean(kf[m] == 0)), float((1 - fbar) ** k), int(m.sum())))
        mu[K] = float(np.mean(mps))
        L.append(f"   K{K:2d}: f-bar {fbar:.3f} | " + " ".join(f"q{k} {q:.3f} vs (1-f)^k {p:.3f} (n {n}, d {q - p:+.3f})" for k, q, p, n in qs)
                 + f" | top input favourable {np.mean(np.concatenate(top_fav)):.3f} | mu+ (51-100) {mu[K]:.3f}")
    if all(K in mu for K in (2, 10, 20)):
        L.append(f"   (c) mu+ K20 > K10 > K2: {mu[20]:.3f} > {mu[10]:.3f} > {mu[2]:.3f} -> {mu[20] > mu[10] > mu[2]}")
    return L


def rr4tl():
    import sink_roots_mnist_0930 as E
    L = ["\n## RR4 two-layer caps (2-layer ELU from task 5; |mu2| = norm of the mean layer-1 output, |w2| = median row norm, a2 = w2 . mu2_hat)"]
    cache = {}
    for name in ["R9_V3rl2_ELU"] + [f"RR4tl_{a}_ELU" for a in ("L1cap", "L2cap", "L12cap")]:
        rows = []
        for s in (0, 1, 2):
            d = RAW / "mnist" / f"{name}_s{s}"
            if not (d / "provenance.json").exists():
                continue
            if s not in cache:
                X, _, _ = E.load_mnist(s); cache[s] = X.double().numpy()
            X = cache[s]
            vals = {}
            for t in (5, 10, 20):
                f = d / f"state_before_t{t:03d}.npz"
                if not f.exists():
                    continue
                st = np.load(f); W1, b1, W2 = st["P0"].astype(np.float64), st["P1"].astype(np.float64), st["P2"].astype(np.float64)
                z1 = X @ W1.T + b1; a1 = np.where(z1 > 0, z1, np.expm1(np.minimum(z1, 0)))
                mu2 = a1.mean(0); mh = mu2 / np.linalg.norm(mu2)
                vals[t] = (np.linalg.norm(mu2), float(np.median(np.linalg.norm(W2, axis=1))), float(np.median((W2 @ mh) / np.linalg.norm(W2, axis=1))))
            a = np.load(d / "arrays.npz"); m2 = a["u_m2"]
            dm2 = float(np.mean([np.median(m2[t, -1] - m2[t, 0]) for t in range(5, min(20, m2.shape[0]))]))
            dm2_510 = float(np.mean([np.median(m2[t, -1] - m2[t, 0]) for t in range(5, min(10, m2.shape[0]))]))
            rows.append((s, vals, dm2, dm2_510))
        for s, vals, dm2, dm2_510 in rows:
            v5, v10 = vals.get(5), vals.get(10)
            txt = f"   {name} s{s}: "
            if v5 and v10:
                txt += f"|mu2| x{v10[0] / v5[0]:.2f} (5->10), |w2| x{v10[1] / v5[1]:.2f}, a2/|w2| {v5[2]:+.3f} -> {v10[2]:+.3f}"
            txt += f" | dm2 per task: tasks 6-10 {dm2_510:+.3f}, tasks 6-20 {dm2:+.3f}"
            L.append(txt)
    return L


if __name__ == "__main__":
    out = []
    for fn in (rr1, rr2, rr3, rr4, g6, ksweep, rr4tl):
        try:
            out += fn()
        except Exception as e:                       # a part whose runs are not all there yet
            out.append(f"\n## {fn.__name__}: not available ({type(e).__name__}: {e})")
    txt = "\n".join(out); print(txt)
    (RES / "round6_analysis.txt").write_text(txt + "\n")
