#!/usr/bin/env python3
"""c1_direction_1010 Q2 (spec sec. 2.2, P9): tables of the LR replays t10 -> t11 and t20 -> t21.

    python analysis/c1_direction_1010/q2_tables.py

Input:  results/c1_direction_1010/replay/replay_LR_t{10,20}.npz (+ _R1.json)
        results/c1_direction_1010/cnn_task/LR_s{seed}_t{10,20}_step{75,750,30000}_{task,Lu}.npz
Output: results/c1_direction_1010/cnn_tables/q2_replay.md and q2_*.csv

SIGN CONVENTION: G > 0 = sinking side (the SGD expected step of that loss lowers the channel mean).
"task" = the CE of the task's actual labels (mean over the 1200 images) at that in-task state;
"L_u" = the expected CE under fresh iid uniform labels (the registered switch push) at the same state.
Step 0 = task start (after the new labels); 30000 = task end.  zbar1_j = <W1[j], M> + b1_j exactly
(M = the run's raw mean patch, fixed), so its motion splits into the mu-path a_j |M| and the bias b1_j.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "analysis" / "cnn_drive_verify_1009"))
from statlite import t975                                  # noqa: E402

REP = ROOT / "results" / "c1_direction_1010" / "replay"
SNAP = ROOT / "results" / "c1_direction_1010" / "cnn_task"
OUT = ROOT / "results" / "c1_direction_1010" / "cnn_tables"
TASKS = (10, 20)
CH = 16


def ci(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), 0
    m = float(x.mean())
    h = t975(n - 1) * float(x.std(ddof=1)) / math.sqrt(n) if n > 1 else float("nan")
    return m, m - h, m + h, n


def fmt(x, nd=2):
    m, lo, hi, n = ci(x)
    if not np.isfinite(m):
        return "—"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}]" if np.isfinite(lo) else f"{m:.{nd}f}"


def series(d):
    """Per-step scalars per run (n_steps, R)."""
    Mn = d["M_norm"][None, :, None]
    z1 = d["zbar"][:, :, :CH]
    return {"sum_k <W2[k],mu2^>": d["w2mu"].sum(2),
            "sum_kj s_kj": d["s_kj"].sum((2, 3)),
            "median_j zbar1": np.median(z1, 2),
            "mean_j zbar1": z1.mean(2),
            "mean_j a_j|M| (mu-path of zbar1)": (d["a"] * Mn).mean(2),
            "mean_j b1_j (bias of zbar1)": d["b1"].mean(2),
            "median_k zbar2": np.median(d["zbar"][:, :, CH:], 2),
            "mean_j a_j": d["a"].mean(2),
            "sum_k <W2[k],mu2^> / 16": d["w2mu"].sum(2) / CH,
            "mean_kj s_kj": d["s_kj"].mean((2, 3)),
            "mean_k b2_k": d["b2"].mean(2),
            "mean_k |W2^AC[k]|": d["w2ac_norm"].mean(2)}


def table_a(d, task) -> tuple[pd.DataFrame, list[str]]:
    steps = list(d["step"])
    i0, i75, i750, iE = steps.index(0), steps.index(75), steps.index(750), steps.index(30000)
    ser = series(d)
    keys = ["sum_k <W2[k],mu2^>", "sum_kj s_kj", "median_j zbar1", "mean_j zbar1",
            "mean_j a_j|M| (mu-path of zbar1)", "mean_j b1_j (bias of zbar1)"]
    rows = []
    for r, s in enumerate(d["seeds"]):
        row = {"task": f"t{task}->t{task + 1}", "seed": int(s)}
        for k in keys:
            x = ser[k][:, r]
            tot = x[iE] - x[i0]
            row[f"{k}: total change"] = tot
            row[f"{k}: frac by 75"] = (x[i75] - x[i0]) / tot
            row[f"{k}: frac by 750"] = (x[i750] - x[i0]) / tot
        # mu-path share of c1's in-task motion (mean over channels; zbar1 = a|M| + b1 exactly)
        a = ser["mean_j a_j|M| (mu-path of zbar1)"][:, r]
        z = ser["mean_j zbar1"][:, r]
        for lab, (i, j) in (("0->75", (i0, i75)), ("0->750", (i0, i750)), ("0->30000", (i0, iE)),
                            ("75->30000", (i75, iE))):
            row[f"mu-path share of d(mean zbar1) {lab}"] = (a[j] - a[i]) / (z[j] - z[i])
        rows.append(row)
    return pd.DataFrame(rows), keys


def snap_rates(task, step, loss) -> pd.DataFrame:
    rows = []
    for fn in sorted(SNAP.glob(f"LR_s*_t{task:02d}_step{step}_{loss}.npz")):
        d = np.load(fn, allow_pickle=True)
        cD = float(d["cD2_sum"])
        rows.append({"seed": int(d["seed"]),
                     "G_c1>0": float(np.mean(d["G_c1_full"] > 0)),
                     "G_c1_DC>0": float(np.mean(d["G_c1_DC"] > 0)),
                     "G_c1_AC>0": float(np.mean(d["G_c1_AC"] > 0)),
                     "G_c1_drift>0": float(np.mean(d["G_c1_drift"] > 0)),
                     "G_c1_init>0": float(np.mean(d["G_c1_init"] > 0)),
                     "G_c1_rest>0": float(np.mean(d["G_c1_rest"] > 0)),
                     "Dbar_k>0 [c2 ch]": float(np.mean(d["D2bar"] > 0)),
                     "c_k>0 [c2 ch]": float(np.mean(d["c_k"] > 0)),
                     "sum_k c_k Dbar_k>0": float(cD > 0),
                     "margin sum c*D / sum|c*D|": cD / float(d["cD2_abs"]),
                     "Pbar1>0": float(np.mean(d["Pbar1"] > 0)),
                     "G_c2>0": float(np.mean(d["G_c2_full"] > 0)),
                     "G_c2_DC>0": float(np.mean(d["G_c2_DC"] > 0)),
                     "G_c2_drift>0": float(np.mean(d["G_c2_drift"] > 0))})
    return pd.DataFrame(rows)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L = ["# c1_direction_1010 — Q2 replay (LR, t10→t11 and t20→t21)\n",
         "Numbers as computed, no judgement. **G > 0 = sinking side.** \"task\" = CE of the task's actual "
         "labels at the in-task state; \"L_u\" = the registered uniform-label switch push at the same state. "
         "Step 0 = task start (new labels drawn), 30000 = task end. Cells: seed mean over 10 seeds "
         "[t-based 95% CI]. zbar1_j = ⟨W1[j], M⟩ + b1_j exactly (M = the raw mean patch), so "
         "the c1 mean splits into the µ-path a_j‖M‖ and the bias b1_j. µ̂₂ in ⟨W2[k], µ̂₂⟩ is fixed at the "
         "task-start state.\n"]
    # ---- R1
    L.append("## R1 (fidelity of the replay)\n")
    for task in TASKS:
        fj = REP / f"replay_LR_t{task:02d}_R1.json"
        if not fj.exists():
            L.append(f"- t{task}: missing\n")
            continue
        f = json.loads(fj.read_text())
        L.append(f"- t{task}→t{task + 1}: R1 pass = **{f['R1_pass']}**; end-of-task evaluate() and online acc vs "
                 f"per_task.csv max rel {f['R1_evaluate_and_online_max_rel']:.1e}; switch CE/acc max rel "
                 f"{f['R1_switch_max_rel']:.1e}; per-epoch online accuracy bit-equal to curves.npy for all runs: "
                 f"{f['R1_curves_bit_equal_all_runs']}; zbar records vs the earlier verify replay (max abs): "
                 f"{ {k: float(v) for k, v in f['R1_zbar_records_vs_verify_replay_max_abs'].items()} } "
                 f"(wall {f['wall_s']:.0f}s)")
    L.append("\n" + json.loads((REP / f"replay_LR_t{TASKS[0]:02d}_R1.json").read_text())["note"]
             if (REP / f"replay_LR_t{TASKS[0]:02d}_R1.json").exists() else "")
    # ---- (a)
    L.append("\n## (a) Share of the whole-task change (step 0 → 30000) that happens by step 75 / 750\n")
    all_a = []
    for task in TASKS:
        fn = REP / f"replay_LR_t{task:02d}.npz"
        if not fn.exists():
            continue
        d = np.load(fn)
        da, keys = table_a(d, task)
        all_a.append(da)
        L.append(f"### t{task} → t{task + 1}, per seed\n")
        cols = ["seed"] + [f"{k}: {w}" for k in keys for w in ("total change", "frac by 75", "frac by 750")]
        head = ["seed"] + [f"{k} [{w}]" for k in keys for w in ("Δ0→30000", "by 75", "by 750")]
        L.append("| " + " | ".join(head) + " |")
        L.append("|" + "---|" * len(head))
        for _, r in da.iterrows():
            L.append("| " + " | ".join([str(int(r.seed))] + [f"{r[c]:.3g}" for c in cols[1:]]) + " |")
        L.append("| mean [CI] | " + " | ".join(fmt(da[c]) for c in cols[1:]) + " |")
        L.append("| median | " + " | ".join(f"{np.median(da[c]):.3g}" for c in cols[1:]) + " |")
        # absolute changes (the fractions above divide by the net task change, which can be ~0)
        ser = series(d)
        steps = list(d["step"])
        i0, i75, i750, iE = steps.index(0), steps.index(75), steps.index(750), steps.index(30000)
        L.append(f"\nAbsolute changes from step 0 (seed mean [CI]; `max|Δ|` = the largest |X(s) − X(0)| over the "
                 f"recorded steps, and the step where it occurs most often):\n")
        L.append("| quantity | Δ by 75 | Δ by 750 | Δ by 30000 | max abs Δ | modal step of max |")
        L.append("|---|---|---|---|---|---|")
        for k in keys:
            x = ser[k]
            dx = x - x[i0][None]
            imax = np.abs(dx).argmax(0)
            vals, cnts = np.unique(np.asarray(steps)[imax], return_counts=True)
            L.append(f"| {k} | {fmt(dx[i75])} | {fmt(dx[i750])} | {fmt(dx[iE])} | "
                     f"{fmt(np.abs(dx).max(0))} | {int(vals[cnts.argmax()])} ({cnts.max()}/{len(imax)}) |")
        n50 = int((da["sum_k <W2[k],mu2^>: frac by 75"] >= 0.5).sum())
        L.append(f"\nSeeds with ≥ 50% of the in-task change of Σ_k⟨ΔW2[k], µ̂₂⟩ within the first 75 steps: "
                 f"**{n50}/{len(da)}** (by 750: {int((da['sum_k <W2[k],mu2^>: frac by 750'] >= 0.5).sum())}/{len(da)}).\n")
        mu_cols = [c for c in da.columns if c.startswith("mu-path share")]
        L.append("µ-path share of the change of mean_j zbar1 (Δ(mean a‖M‖)/Δ(mean zbar1)):\n")
        L.append("| seed | " + " | ".join(c.replace("mu-path share of d(mean zbar1) ", "") for c in mu_cols) + " |")
        L.append("|---|" + "---|" * len(mu_cols))
        for _, r in da.iterrows():
            L.append(f"| {int(r.seed)} | " + " | ".join(f"{r[c]:.3f}" for c in mu_cols) + " |")
        L.append("| mean [CI] | " + " | ".join(fmt(da[c]) for c in mu_cols) + " |\n")
    if all_a:
        pd.concat(all_a).to_csv(OUT / "q2_a_fractions_by_seed.csv", index=False)
    # ---- (b)
    L.append("## (b) In-task push at steps 75 / 750 / 30000: task loss vs L_u (rates over channels, seed mean [CI])\n")
    L.append("c1 rates are over the 16 c1 channels, `[c2 ch]` over the 16 c2 channels; `sum_k c_k Dbar_k>0` is "
             "the fraction of seeds; the margin is Σ c·D̄ / Σ|c·D̄| (seed mean). c_k = −⟨W2[k] − W2⁰[k], µ̂₂⟩ "
             "with µ̂₂ of that in-task state.\n")
    rows_b = []
    cols_b = None
    for task in TASKS:
        for step in (75, 750, 30000):
            for loss in ("task", "Lu"):
                df = snap_rates(task, step, loss)
                if df.empty:
                    continue
                cols_b = [c for c in df.columns if c != "seed"]
                rows_b.append((task, step, loss, df))
    if rows_b:
        L.append("| task | step | loss | n | " + " | ".join(cols_b) + " |")
        L.append("|---|---|---|---|" + "---|" * len(cols_b))
        out_rows = []
        for task, step, loss, df in rows_b:
            L.append(f"| t{task}→t{task + 1} | {step} | {'task CE' if loss == 'task' else 'L_u'} | {len(df)} | "
                     + " | ".join(fmt(df[c]) for c in cols_b) + " |")
            for _, r in df.iterrows():
                out_rows.append({"task": task, "step": step, "loss": loss, **r.to_dict()})
        pd.DataFrame(out_rows).to_csv(OUT / "q2_b_rates_by_seed.csv", index=False)
    # ---- (c)
    L.append("\n## (c) Time course (seed mean over 10 runs)\n")
    tc_rows = []
    for task in TASKS:
        fn = REP / f"replay_LR_t{task:02d}.npz"
        if not fn.exists():
            continue
        d = np.load(fn)
        ser = series(d)
        keys = ["median_j zbar1", "median_k zbar2", "mean_j a_j", "mean_j a_j|M| (mu-path of zbar1)",
                "mean_j b1_j (bias of zbar1)", "sum_k <W2[k],mu2^> / 16", "mean_kj s_kj", "mean_k b2_k",
                "mean_k |W2^AC[k]|"]
        L.append(f"### t{task} → t{task + 1}\n")
        L.append("| step | " + " | ".join(keys) + " |")
        L.append("|---|" + "---|" * len(keys))
        for i, st in enumerate(d["step"]):
            L.append(f"| {int(st)} | " + " | ".join(f"{ser[k][i].mean():.4g}" for k in keys) + " |")
            tc_rows.append({"task": task, "step": int(st), **{k: float(ser[k][i].mean()) for k in keys}})
        L.append("")
    if tc_rows:
        pd.DataFrame(tc_rows).to_csv(OUT / "q2_c_timecourse.csv", index=False)
    (OUT / "q2_replay.md").write_text("\n".join(L) + "\n")
    print("wrote", OUT / "q2_replay.md")


if __name__ == "__main__":
    main()
