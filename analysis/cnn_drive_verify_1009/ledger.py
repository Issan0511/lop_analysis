#!/usr/bin/env python3
"""(c) long-term sinking and the push / return ledger.

    python analysis/cnn_drive_verify_1009/ledger.py --root results/cnn_drive_verify_1009

Inputs: zbar/zbar_checkpoints.csv (channel means at init and at each checkpoint),
replay/replay_tXX.npy (engine re-runs of task t+1 from checkpoint t, channel means at steps
0, 1, 10, 75, 750, 7500, 30000), measure/ (G_full and the first-step predictions),
and the run's per_task.csv (channel medians, every task).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
import statlite as SL  # noqa: E402

CH = 16
PER_TASK = Path("/home/issan/Projects/claude/wt/sna_cnn_cause_1009/results/sna_cnn_cause_1009/A/per_task.csv")


def seed_rate(d, col, by):
    per = d.groupby(by + ["seed"])[col].mean().reset_index()
    out = []
    for key, g in per.groupby(by):
        x = g[col].to_numpy(float)
        n = len(x)
        m = x.mean(); sd = x.std(ddof=1) if n > 1 else float("nan")
        hw = SL.t975(n - 1) * sd / math.sqrt(n) if n > 1 else float("nan")
        key = key if isinstance(key, tuple) else (key,)
        out.append(dict(zip(by, key)) | {"rate": m, "sd_seed": sd, "lo": m - hw, "hi": m + hw, "n_seed": n})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--per-task", default=str(PER_TASK), help="the run's per_task.csv")
    args = ap.parse_args()
    root = Path(args.root)
    out = root / "tables_c"; out.mkdir(parents=True, exist_ok=True)
    tabs, rep = {}, {}
    # ---- net change from init and per-checkpoint changes
    zb = pd.read_csv(root / "zbar" / "zbar_checkpoints.csv")
    piv = zb.pivot_table(index=["arm", "seed", "layer", "ch"], columns="task", values="zbar")
    tasks = sorted(c for c in piv.columns if c > 0)
    rows = []
    for t in tasks:
        dd = (piv[t] - piv[0]).rename("d").reset_index()
        dd["task"] = t
        rows.append(dd)
    net = pd.concat(rows)
    net["down"] = net.d < 0
    tabs["net_from_init_down"] = seed_rate(net, "down", ["layer", "arm", "task"])
    tabs["net_from_init_down_pooled"] = seed_rate(net, "down", ["layer", "task"])
    # consecutive checkpoints (t -> t+1 only where both exist)
    rows = []
    for t in tasks:
        if t + 1 in piv.columns:
            dd = (piv[t + 1] - piv[t]).rename("d").reset_index(); dd["task"] = t; rows.append(dd)
    if rows:
        one = pd.concat(rows); one["down"] = one.d < 0
        tabs["one_task_down_ckpt"] = seed_rate(one, "down", ["layer", "task"])
        tabs["one_task_down_ckpt_arm"] = seed_rate(one, "down", ["layer", "arm", "task"])
    # ---- per_task.csv medians, every task
    pt = pd.read_csv(args.per_task)
    pr = []
    for (arm, seed), g in pt.groupby(["arm", "seed"]):
        g = g.sort_values("task")
        for tag in ("c1", "c2"):
            dz = np.diff(g[f"zbar_{tag}"].to_numpy())
            for k, v in enumerate(dz):
                pr.append({"arm": arm, "seed": seed, "layer": tag, "task": int(g.task.iloc[k]), "d": v})
    pr = pd.DataFrame(pr); pr["down"] = pr.d < 0
    pr["phase"] = np.where(pr.task < 10, "t1-9", "t10+")
    tabs["per_task_median_down"] = seed_rate(pr, "down", ["layer", "arm", "phase"])
    rep["per_task_last_task"] = int(pt.task.max())
    # ---- replays: ledger
    ms = {}
    for fn in sorted((root / "measure").glob("measure_*.npy")):
        r = np.load(fn, allow_pickle=True).item()
        ms[(r["arm"], r["seed"], r["task"])] = r
    led = []
    fid = []
    for fn in sorted((root / "replay").glob("replay_t*.npy")):
        r = np.load(fn, allow_pickle=True).item()
        t = r["task"]
        rec = {int(k): v for k, v in r["rec"].items()}
        for i, (arm, seed) in enumerate(r["slots"]):
            m = ms.get((arm, seed, t))
            for k in range(2 * CH):
                layer, j = divmod(k, CH)
                row = {"arm": arm, "seed": seed, "task": t, "layer": f"c{layer + 1}", "ch": j}
                z0 = rec[0][i, k]
                for s in sorted(rec):
                    if s > 0:
                        row[f"d{s}"] = rec[s][i, k] - z0
                if m is not None:
                    row["G_full"] = float(m["G_full"][k])
                    row["fs_adam"] = float(m["fs_adam"][k])
                    row["fs_adam_se"] = float(m["fs_adam_se"][k])
                    row["real_adam"] = float(m["real_adam"][k])
                    if "G_self" in m:
                        row["G_self"] = float(m["G_self"][k])
                led.append(row)
        for key, v in r["param_rel_diff_vs_saved"].items():
            fid.append({"task": t, "run": key, "param_rel_diff": v})
    if led:
        L = pd.DataFrame(led)
        L["push"] = L["d75"]
        L["ret"] = L["d30000"] - L["d75"]
        L["total"] = L["d30000"]
        L["total_down"] = L.total < 0
        L["push_down"] = L.push < 0
        L["ret_opposite"] = np.sign(L.ret) == -np.sign(L.push)
        L["ret_smaller"] = L.ret.abs() < L.push.abs()
        if "G_full" in L:
            L["push_vs_sgd"] = np.sign(L.push) == -np.sign(L.G_full)
            L["step1_vs_sgd"] = np.sign(L.d1) == -np.sign(L.G_full)
            L["push_vs_adam_exp"] = np.sign(L.push) == np.sign(L.fs_adam)
            L["step1_vs_adam_exp"] = np.sign(L.d1) == np.sign(L.fs_adam)
            L["step1_vs_real_lin"] = np.sign(L.d1) == np.sign(L.real_adam)
            L["total_vs_sgd"] = np.sign(L.total) == -np.sign(L.G_full)
        L.to_csv(out / "ledger_channels.csv", index=False)
        for col in ("total_down", "push_down", "ret_opposite", "ret_smaller", "push_vs_sgd", "step1_vs_sgd",
                    "push_vs_adam_exp", "step1_vs_adam_exp", "step1_vs_real_lin", "total_vs_sgd"):
            if col in L:
                tabs[f"ledger_{col}"] = seed_rate(L, col, ["layer", "task"])
        sp = []
        for (layer, t), g in L.groupby(["layer", "task"]):
            if "G_full" in g:
                sp.append({"layer": layer, "task": t,
                           "spearman_negG_total": SL.spearman(-g.G_full, g.total),
                           "spearman_negG_push": SL.spearman(-g.G_full, g.push),
                           "spearman_adam_push": SL.spearman(g.fs_adam, g.push),
                           "spearman_push_total": SL.spearman(g.push, g.total),
                           "median_push": g.push.median(), "median_ret": g.ret.median(),
                           "median_total": g.total.median()})
        tabs["ledger_spearman"] = pd.DataFrame(sp)
        # fraction of the 30000-step change that the first k steps carry (medians)
        fr = []
        for (layer, t), g in L.groupby(["layer", "task"]):
            row = {"layer": layer, "task": t}
            for s in (1, 10, 75, 750, 7500):
                if f"d{s}" in g:
                    row[f"med_d{s}"] = g[f"d{s}"].median()
            row["med_d30000"] = g.d30000.median()
            fr.append(row)
        tabs["ledger_medians"] = pd.DataFrame(fr)
    if fid:
        F = pd.DataFrame(fid)
        tabs["replay_fidelity_params"] = F.groupby("task").param_rel_diff.describe().reset_index()
        # channel-mean fidelity: replay end vs saved t+1
        fz = []
        for fn in sorted((root / "replay").glob("replay_t*.npy")):
            r = np.load(fn, allow_pickle=True).item()
            t = r["task"]
            if t + 1 not in piv.columns:
                continue
            for i, (arm, seed) in enumerate(r["slots"]):
                for k in range(2 * CH):
                    layer, j = divmod(k, CH)
                    saved = piv.loc[(arm, seed, f"c{layer + 1}", j), t + 1]
                    base = piv.loc[(arm, seed, f"c{layer + 1}", j), t]
                    fz.append({"task": t, "layer": f"c{layer + 1}", "err": r["rec"]["30000"][i, k] - saved,
                               "d_saved": saved - base})
        if fz:
            FZ = pd.DataFrame(fz)
            FZ["abs_err"] = FZ.err.abs()
            tabs["replay_fidelity_zbar"] = FZ.groupby(["task", "layer"]).agg(
                max_abs_err=("abs_err", "max"), med_abs_err=("abs_err", "median"),
                med_abs_d=("d_saved", lambda x: np.median(np.abs(x)))).reset_index()
    # ---- frozen reference (real labels and batch order, parameters frozen) vs the moving replay
    if led and (root / "frozen").exists():
        fr = []
        for fn in sorted((root / "frozen").glob("frozen_*.npy")):
            r = np.load(fn, allow_pickle=True).item()
            for s, v in r["rec"].items():
                for k in range(2 * CH):
                    layer, j = divmod(k, CH)
                    fr.append({"arm": r["arm"], "seed": r["seed"], "task": r["task"], "layer": f"c{layer + 1}",
                               "ch": j, "S": int(s), "frozen_real": float(v["adam"][0, k])})
        FR = pd.DataFrame(fr)
        if len(FR):
            LL = L.melt(id_vars=["arm", "seed", "task", "layer", "ch"],
                        value_vars=[c for c in L.columns if c.startswith("d") and c[1:].isdigit()],
                        var_name="S", value_name="moving")
            LL["S"] = LL.S.str[1:].astype(int)
            M = FR.merge(LL, on=["arm", "seed", "task", "layer", "ch", "S"])
            M["same_sign"] = np.sign(M.frozen_real) == np.sign(M.moving)
            M["ratio"] = M.moving / M.frozen_real
            tabs["frozen_vs_moving"] = M.groupby(["layer", "task", "S"]).agg(
                same_sign=("same_sign", "mean"), median_ratio=("ratio", "median"),
                spearman=("moving", lambda x: SL.spearman(x, M.loc[x.index, "frozen_real"])),
                n=("same_sign", "size")).reset_index()
    for k, t in tabs.items():
        t.to_csv(out / f"{k}.csv", index=False)
    (out / "report.json").write_text(json.dumps(rep, indent=1, default=float))
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
    for k, t in tabs.items():
        print(f"\n## {k}\n{t.round(4).to_string(index=False)}")


if __name__ == "__main__":
    main()
