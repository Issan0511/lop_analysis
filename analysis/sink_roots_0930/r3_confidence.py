#!/usr/bin/env python3
"""Round 3 R3p_confidence (spec_sink_roots_0930_round3.md): what sets the push depth -- p_old, the logit spread, or the
task length?  Each task ends when its own fit reaches a target (engine --stop pold / lsd, 500 <= updates <= 16,000).
Per task t = 5..30 (the task after a stopped fit): depth = median over units open at the switch of
(nanmin over the task's probes of m - m(switch)); p_old and logit_sd at the switch (probe values); the previous task's
length (steps_used).  Registered readings:
  (1) ELU, lsd arms: -depth / lsd per run in [1.6, 2.3]; the depth near -1.6..-2.3 / -3.8..-5.5 / -11..-16 at lsd 1 / 2.4 / 7
  (2) per activation, task level over all 6 arms: R^2 of log(-depth) on log(lsd) against R^2 on log(p_old) (lsd wins -> as predicted)
  (3) within each p_old arm (p_old held fixed), the task-level correlation of log(-depth) with log(lsd) (> 0 -> as predicted)."""
import glob, json, re
from pathlib import Path
import numpy as np

RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/mnist")
RES = Path(__file__).resolve().parents[2] / "results" / "sink_roots_0930"


def per_task(name):
    a = np.load(RAW / name / "arrays.npz"); rows = json.load(open(RAW / name / "rows.json"))
    m, k = a["u_m1"], a["u_k1"]
    steps = {r["task"]: r["steps_used"] for r in rows}
    out = []
    for t in range(5, min(31, m.shape[0] + 1)):
        ti = t - 1; live = k[ti, 0] > 0
        if live.sum() < 5:
            continue
        tr = m[ti][:, live] - m[ti][0, live]
        dep = float(np.median(np.nanmin(tr, axis=0)))
        out.append(dict(task=t, depth=dep, p_old=float(a["s_p_old"][ti, 0]), lsd=float(a["s_logit_sd"][ti, 0]),
                        prev_steps=int(steps.get(t - 1, -1)), n_live=int(live.sum())))
    return out


def r2(x, y):
    if len(x) < 3:
        return float("nan")
    c = np.corrcoef(x, y)[0, 1]
    return float(c * c)


def main():
    L = ["arm act seed | tasks | depth median | p_old median | lsd median | prev task length median | -depth/lsd median"]
    D = {}
    for d in sorted(glob.glob(str(RAW / "R3p_*_s[0-9]"))):
        name = Path(d).name
        m = re.match(r"R3p_(pold|lsd)([\d.]+)_(\w+)_s(\d)", name)
        if not m or not (Path(d) / "provenance.json").exists():
            continue
        kind, tgt, act, seed = m.group(1), float(m.group(2)), m.group(3), int(m.group(4))
        rows = per_task(name)
        D[(kind, tgt, act, seed)] = rows
        dep = np.array([r["depth"] for r in rows]); lsd = np.array([r["lsd"] for r in rows])
        L.append(f"{kind}{tgt:<5g} {act:4s} s{seed} | {len(rows)} | {np.median(dep):+.2f} | {np.median([r['p_old'] for r in rows]):.3f} | "
                 f"{np.median(lsd):.2f} | {np.median([r['prev_steps'] for r in rows]):.0f} | {np.median(-dep / lsd):.2f}")
    L.append("")
    for act in ("ELU", "SILU", "LR"):
        rows = [r for k, v in D.items() if k[2] == act for r in v]
        if not rows:
            continue
        dep = np.array([r["depth"] for r in rows]); ok = dep < 0
        ld = np.log(-dep[ok]); lp = np.log(np.array([r["p_old"] for r in rows])[ok]); ll = np.log(np.array([r["lsd"] for r in rows])[ok])
        L.append(f"{act}: (2) task level over all arms, n {ok.sum()}: R^2 log(-depth)~log(lsd) {r2(ll, ld):.3f} vs ~log(p_old) {r2(lp, ld):.3f}")
        if act == "ELU":
            for tgt in (1.0, 2.4, 7.0):
                q = [np.median([-r["depth"] / r["lsd"] for r in D[k]]) for k in D if k[0] == "lsd" and k[1] == tgt and k[2] == act]
                dd = [np.median([r["depth"] for r in D[k]]) for k in D if k[0] == "lsd" and k[1] == tgt and k[2] == act]
                L.append(f"   (1) lsd {tgt:g}: -depth/lsd per seed {np.round(q, 2).tolist()} | depth per seed {np.round(dd, 2).tolist()}")
        for tgt in (0.26, 0.43, 0.57):
            rr = [r for k, v in D.items() if k[0] == "pold" and k[1] == tgt and k[2] == act for r in v]
            dep = np.array([r["depth"] for r in rr]); ok = dep < 0
            if ok.sum() >= 3:
                c = np.corrcoef(np.log(-dep[ok]), np.log(np.array([r["lsd"] for r in rr])[ok]))[0, 1]
                L.append(f"   (3) p_old arm {tgt}: n {ok.sum()} p_old median {np.median([r['p_old'] for r in rr]):.3f} lsd range {min(r['lsd'] for r in rr):.2f}..{max(r['lsd'] for r in rr):.2f}; corr(log -depth, log lsd) {c:+.2f}")
    txt = "\n".join(L); print(txt)
    (RES / "round3_R3p_confidence.txt").write_text(txt + "\n")
    (RES / "round3_R3p_confidence.json").write_text(json.dumps({"|".join(map(str, k)): v for k, v in D.items()}))


if __name__ == "__main__":
    main()
