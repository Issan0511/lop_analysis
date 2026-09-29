#!/usr/bin/env python3
"""R5 (spec §3), R9 (§5), R11a (§6.1): registered labels and descriptive tables."""
import csv, json
from pathlib import Path
import numpy as np

RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/mnist")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"
SEEDS = [0, 1, 2]


def load(name):
    d = RAW / name
    if not (d / "provenance.json").exists():
        return None
    return np.load(d / "arrays.npz"), json.loads((d / "rows.json").read_text())


# ------------------------------------------------------------------ R5
def r5():
    out, labels = [], {}
    modes = ["nat", "long", "adam0", "ro_reinit", "l1_reinit", "recenter", "fresh"]
    for act in ["ELU", "LR"]:
        # main runs: fit over tasks, coverage
        for s in SEEDS:
            L = load(f"R5_main_{act}_s{s}")
            if L is None:
                continue
            a, rows = L
            cov = a["cover"]                    # (tasks, 2, N): open units per input at switch / end
            corr = a["correct_end"]             # (tasks, N)
            for t in (10, 30, 50, 100, 150, 200):
                c_end, ok = cov[t - 1, 1], corr[t - 1]
                out.append({"study": "R5_main", "act": act, "seed": s, "task": t,
                            "acc_end": rows[t - 1]["acc_end"], "online": rows[t - 1]["online_acc"],
                            "cov0_frac": float((c_end == 0).mean()),
                            "acc_cov0": float(ok[c_end == 0].mean()) if (c_end == 0).any() else np.nan,
                            "acc_cov1": float(ok[c_end == 1].mean()) if (c_end == 1).any() else np.nan,
                            "acc_cov2p": float(ok[c_end >= 2].mean()) if (c_end >= 2).any() else np.nan,
                            "cov_mean": float(c_end.mean())})
        for t in (10, 50, 200):
            F = {}
            for mode in modes:
                for s in SEEDS:
                    L = load(f"R5_fork_{act}_s{s}_t{t}_{mode}")
                    if L is None:
                        continue
                    a, rows = L
                    F[(mode, s)] = rows[0]["acc_end"]
                    out.append({"study": "R5_fork", "act": act, "seed": s, "task": t + 1, "mode": mode,
                                "acc_end": rows[0]["acc_end"], "online": rows[0]["online_acc"],
                                "cov_mean_end": float(a["cover"][0, 1].mean()),
                                "cov0_end": float((a["cover"][0, 1] == 0).mean())})
            for mode in modes:
                if mode in ("fresh", "nat"):
                    continue
                if not all((m, s) in F for m in (mode, "fresh", "nat") for s in SEEDS):
                    labels[f"{act}|t{t}|{mode}"] = "INCOMPLETE"
                    continue
                f = [F[(mode, s)] for s in SEEDS]
                fr = [F[("fresh", s)] for s in SEEDS]
                na = [F[("nat", s)] for s in SEEDS]
                if all(x >= y - 0.05 for x, y in zip(f, fr)):
                    lab = "RECOVERS"
                elif all(x <= y + 0.05 for x, y in zip(f, na)):
                    lab = "NONE"
                elif all(x > y + 0.05 for x, y in zip(f, na)):
                    lab = "PARTIAL"
                else:
                    lab = "MIXED"
                labels[f"{act}|t{t}|{mode}"] = lab
    return out, labels


# ------------------------------------------------------------------ R9
def windows(grid, T):
    g = np.array(grid)
    w1 = g < 200
    if T > 1000:
        w2, w3 = (g >= 200) & (g < 1000), g >= 1000
    else:
        w2, w3 = g >= 200, None
    return w1, w2, w3


def r9():
    out, labels = [], {}
    boxes = {"V1rl1": ("R3_base_{act}_s{s}", [1], 5), "V2sgd": ("R9_V2sgd_{act}_s{s}", [1], 5),
             "V3rl2": ("R9_V3rl2_{act}_s{s}", [1, 2], 5), "V4pm6000": ("R9_V4pm6000_{act}_s{s}", [1, 2], 5),
             "V5pm625": ("R9_V5pm625_{act}_s{s}", [1, 2], 5)}
    for box, (pat, layers, t0) in boxes.items():
        for act in ["GELU", "SILU"]:
            for l in layers:
                E = {}
                for s in SEEDS:
                    L = load(pat.format(act=act, s=s))
                    if L is None:
                        continue
                    a, rows = L
                    grid = list(a["grid"])
                    T = grid[-1]
                    w1, w2, w3 = windows(grid, T)
                    key = f"s_v{l}_below_u_neg"
                    if key not in a.files:
                        continue
                    X = a[key][t0 - 1:]                          # (tasks, probes)
                    Up = a[f"s_v{l}_moved_up"][t0 - 1:]
                    Cn = a[f"s_v{l}_below_canc_neg"][t0 - 1:]
                    Ln = a[f"s_v{l}_below_lab_neg"][t0 - 1:]
                    NB = a[f"s_v{l}_n_below"][t0 - 1:]
                    row = {"box": box, "act": act, "layer": l, "seed": s}
                    for wn, w in (("w1", w1), ("w2", w2), ("w3", w3)):
                        if w is None:
                            continue
                        # the realized move is recorded at the LATER probe; attribute it to the window of the earlier one
                        wl = np.zeros_like(w); wl[1:] = w[:-1]
                        row[f"E_{wn}"] = float(np.nanmean(X[:, w]))
                        row[f"U_{wn}"] = float(np.nanmean(Up[:, wl]))
                        row[f"canc_neg_{wn}"] = float(np.nanmean(Cn[:, w]))
                        row[f"lab_neg_{wn}"] = float(np.nanmean(Ln[:, w]))
                        row[f"n_below_{wn}"] = float(np.nanmean(NB[:, w]))
                    E[s] = row
                    out.append(row)
                if len(E) < 3:
                    labels[f"{box}|{act}|L{l}"] = "INCOMPLETE"
                    continue
                last = "w3" if "E_w3" in E[0] else "w2"
                e1 = [E[s]["E_w1"] for s in SEEDS]
                eL = [E[s][f"E_{last}"] for s in SEEDS]
                if all(x > 0.5 for x in e1) and all(y < 0.5 for y in eL):
                    lab = "FLIP_ESCAPE_THEN_RESTORE"
                elif all(x < 0.5 for x in e1) and all(y > 0.5 for y in eL):
                    lab = "FLIP_RESTORE_THEN_ESCAPE"
                elif all(x > 0.5 for x in e1 + eL):
                    lab = "ESCAPE_BOTH"
                elif all(x < 0.5 for x in e1 + eL):
                    lab = "RESTORE_BOTH"
                else:
                    lab = "MIXED"
                agree = []
                for s in SEEDS:
                    for wn in ("w1", "w2", "w3"):
                        if f"E_{wn}" in E[s] and np.isfinite(E[s][f"U_{wn}"]):
                            agree.append((E[s][f"E_{wn}"] > 0.5) == (E[s][f"U_{wn}"] < 0.5))
                labels[f"{box}|{act}|L{l}"] = lab
                labels[f"{box}|{act}|L{l}|Q9b_agree"] = float(np.mean(agree)) if agree else None
    return out, labels


# ------------------------------------------------------------------ R11a
def r11a():
    out, labels = [], {}
    for act in ["ELU", "GELU", "SILU", "LR"]:
        frac_by_seed = {}
        for s in SEEDS:
            pairs = []
            for tau in [0.0, 0.25, 0.5, 0.75, 0.9]:
                name = f"R3_base_{act}_s{s}" if tau == 0 else f"R11a_tau{tau}_{act}_s{s}"
                L = load(name)
                if L is None:
                    continue
                a, rows = L
                g = list(a["grid"]); i0, i200 = g.index(0), g.index(200)
                m, k = a["u_m1"], a["u_k1"]
                for t in range(5, 31):
                    ti = t - 1
                    sel = k[ti, i0] > 0
                    if not sel.any():
                        continue
                    P = float(np.median(m[ti, i200][sel] - m[ti, i0][sel]))
                    gam = (float(a["s_p_old"][ti, i0]) - 0.1) / 0.9
                    Aold = float(np.median(a["u_A_old"][ti, i0][sel])) if "u_A_old" in a.files else np.nan
                    pairs.append((tau, t, P, gam, Aold))
                    out.append({"act": act, "seed": s, "tau": tau, "task": t, "push_med": P, "gamma_hat": gam,
                                "A_old_med": Aold})
            use = [(tau, P, gam) for tau, t, P, gam, A in pairs if abs(gam - tau) > 0.1]
            if use:
                frac_by_seed[s] = float(np.mean([np.sign(P) == np.sign(tau - gam) for tau, P, gam in use]))
        if len(frac_by_seed) < 3:
            labels[act] = "INCOMPLETE"
        elif all(v >= 0.9 for v in frac_by_seed.values()):
            labels[act] = "SIGN_FOLLOWS_GAMMA_MINUS_TAU"
        elif all(v <= 0.6 for v in frac_by_seed.values()):
            labels[act] = "SIGN_NOT_EXPLAINED"
        else:
            labels[act] = "PARTIAL"
        labels[act + "|frac"] = frac_by_seed
    return out, labels


def write(rows, path):
    if not rows:
        return
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys); w.writeheader(); w.writerows(rows)


def main():
    RES.mkdir(parents=True, exist_ok=True)
    allab = {}
    for name, fn in (("R5", r5), ("R9", r9), ("R11a", r11a)):
        rows, labels = fn()
        write(rows, RES / f"{name}_table.csv")
        allab[name] = labels
        print(name, json.dumps(labels, indent=0)[:3000])
    (RES / "R5_R9_R11a_labels.json").write_text(json.dumps(allab, indent=1))


if __name__ == "__main__":
    main()
