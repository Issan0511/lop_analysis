#!/usr/bin/env python3
"""eps_rejudge_0917 sections A, B, C1/C3, D and checks S3, S4: read the saved records, no training.

    python3 analysis/eps_rejudge_0917/tables.py

A   lineage.csv        which derivative the optimizer received in each cited run (spec 1.1)
B   l3_readout.csv     act_chimera_0913 rlmnist (VM), ELU1 and SMINH, seeds 0-2, per task
    l3_seeds.csv       T_freeze, the death bounds, the per-seed classes (spec 2.2)
C1  l1_timecourse.csv  mucap_ee_0917 (4 arms x 10 seeds x 100 task ends), per task
    l1_seeds.csv       T_dead, T_freeze, the C1 and C3 classes per seed (spec 2.3)
D1  l2_variant.csv     l2_wall_0916 variant, RL x (ELU->ELU, leaky->ELU) x eps {1e-8, 1e-6, 1e-30} x seeds
D2  depth_report.csv   unit-mean zbar2 over time, L1 ref and L2 none-EE (report only)
S3, S4 -> tables_checks.json
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
RAWROOT = Path.home() / "Projects" / "obsidian-research-data"
OUT = ROOT / "results" / "eps_rejudge_0917"
LN_R = math.log(2.0 ** -25)                  # any float32 expm1 that rounds to nearest or truncates
Z_T = -16.635534286499023                    # K1: white-san's truncating kernel (largest zero z)
EXP0 = -103.97208404541016                   # K3: largest z whose float32 exp is 0 (CPU no flush, CUDA)
LB_LIVE = -70.0                              # spec 2.2: a pair above this has a nonzero SMINH gradient
TINY = 2.0 ** -149
EPS = 1e-8
ACH_STEPS = 400 * 75


def write_csv(path: Path, rows: list[dict]) -> None:
    keys: list[str] = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})


def t_freeze(frozen: dict[int, bool], last: int):
    """First task t such that frozen holds for every task t..last (None if frozen[last] is False)."""
    if not frozen.get(last, False):
        return None
    t = last
    while t - 1 in frozen and frozen[t - 1]:
        t -= 1
    return t


def t_all(flag: dict[int, bool], last: int):
    return t_freeze(flag, last)


# ----------------------------------------------------------------------------------------------
# A
# ----------------------------------------------------------------------------------------------

LINEAGE = [
    dict(lineage="L1", run="mucap_el_0916 (ELU in layer 1 only)", machine="white-san CPU, 1 thread",
         derivative="autograd expm1(z)+1", zero_below=Z_T, flush="off",
         evidence="src/mucap_el_run_0916.py forward2 + torch.autograd.grad (run_one)"),
    dict(lineage="L1", run="mucap_ee_0917 / resp_ee_0917 / l2cap_ee_0917", machine="white-san CPU, 1 thread",
         derivative="autograd expm1(z)+1", zero_below=Z_T, flush="on",
         evidence="src/mucap_el_run_0916.py run_one(act2_name='ELU1'); src/resp_ee_0917.py train_task, main set_flush_denormal(True)"),
    dict(lineage="L2", run="elu_environment_0913 / layer_chimera_rl_0914 / relu_gelu_silu_rl_0914 (+ext150)",
         machine="white-san RTX 5060 Ti, CUDA graph", derivative="hand-written gate exp(min(z,0))",
         zero_below=EXP0, flush="n/a (GPU)",
         evidence="src/layer_chimera_rl_0914.py:44-57 gate/gradients; src/elu_environment_0913.py:39-40; src/relu_gelu_silu_rl_0914.py:66-67"),
    dict(lineage="L2", run="l2_wall_0916 (chimera, ext150, variant incl. Part B eps arms)",
         machine="white-san RTX 5060 Ti", derivative="hand-written gate exp(min(z,0))", zero_below=EXP0,
         flush="n/a (GPU)", evidence="src/l2_wall_0916.py vgradients uses C.gate (layer_chimera_rl_0914)"),
    dict(lineage="L3", run="act_chimera_0913 rlmnist ELU1 (400 epochs; 0914 section 5 table)",
         machine="GCP c2d-standard-32 AMD EPYC 7B13 CPU", derivative="autograd expm1(z)+1",
         zero_below=f">= {LN_R} (kernel not measurable)", flush="off",
         evidence="src/act_chimera_rlmnist_0913.py run (torch.autograd.grad); provenance machine.cpu"),
    dict(lineage="L3", run="act_chimera_0913 rlmnist SMINH (same box)", machine="GCP AMD EPYC 7B13 CPU",
         derivative="autograd of exp(min(z,ln .1)) - C: exp(z) in the deep branch", zero_below=EXP0,
         flush="off", evidence="src/act_chimera_0913.py:92-94 _phi_SMINH"),
]


# ----------------------------------------------------------------------------------------------
# B
# ----------------------------------------------------------------------------------------------

def section_b():
    base = RAWROOT / "act_chimera_0913" / "results" / "act_chimera_0913" / "rlmnist"
    rows, seeds = [], []
    for arm in ("ELU1", "SMINH"):
        for s in (0, 1, 2):
            d = dict(np.load(base / arm / f"s{s}" / f"readout_s{s}.npz"))
            with (ROOT / "results" / "act_chimera_0913" / "rlmnist" / arm / f"s{s}" / "per_task.csv").open() as fh:
                online = {int(r["task"]): float(r["online_acc"]) for r in csv.DictReader(fh)}
            g = d["gbar_l2"].astype(np.float64)                       # (50, 100)
            with np.errstate(divide="ignore"):
                ub = np.log(1200.0 * g)
                lb = np.log(g)
            dv, sn = d["depth_vel"], d["step_num"]
            ar, ef = d["adam_ratio"].astype(np.float64), d["adam_epsfrac"].astype(np.float64)
            frozen, liveR, liveT = {}, {}, {}
            T = g.shape[0]
            for i in range(T):
                t = i + 1
                mv = float(max(np.abs(dv[i]).max(), sn[i].max()))
                frozen[t] = mv == 0.0
                r = {"arm": arm, "seed": s, "task": t, "online_acc": online[t], "move_max": mv,
                     "frozen": frozen[t], "n_ub_ge_lnR": int((ub[i] >= LN_R).sum()),
                     "n_ub_gt_ZT": int((ub[i] > Z_T).sum()), "ub_max": float(ub[i].max()),
                     "lb_max": float(lb[i].max()), "n_lb_gt_m70": int((lb[i] > LB_LIVE).sum()),
                     "n_ub_ge_exp0": int((ub[i] >= EXP0).sum()), "gbar_median": float(np.median(g[i])),
                     "adam_ratio_l1_med": float(np.median(ar[i, 0])), "adam_ratio_l2_med": float(np.median(ar[i, 1])),
                     "adam_ratio_l2_max": float(ar[i, 1].max()),
                     "epsfrac10_l1_med": float(np.median(ef[i, 0])), "epsfrac10_l2_med": float(np.median(ef[i, 1]))}
                # the stuck first-moment residue, in quanta of 2**-149 (K4): ratio * eps (sqrt(v^) << eps)
                r["l2_ratio_in_quanta"] = r["adam_ratio_l2_med"] * EPS * (1 - 0.9 ** (t * ACH_STEPS)) / TINY
                liveR[t] = r["n_ub_ge_lnR"] == 0
                liveT[t] = r["n_ub_gt_ZT"] == 0
                rows.append(r)
            Tf = t_freeze(frozen, T)
            first_eps = next((t for t in range(1, T + 1) if np.median(g[t - 1]) < EPS), None)
            srow = {"arm": arm, "seed": s, "T_freeze": Tf, "T_gate_median_below_eps": first_eps}
            if arm == "ELU1":
                srow["T_dead_R"] = t_all(liveR, T)
                srow["T_dead_T"] = t_all(liveT, T)
                if Tf is None:
                    cls = "OPEN"
                elif liveR[Tf - 1]:
                    cls = "ZERO_R"
                elif liveT[Tf - 1]:
                    cls = "ZERO_T"
                else:
                    cls = "OPEN"
                # was the hidden stack ever still while a pair could be live? (report)
                srow["frozen_while_ub_ge_lnR"] = [t for t in range(2, T + 1)
                                                  if frozen[t] and not liveR[t - 1]]
            else:
                if Tf is None:
                    cls = "TRAVEL"
                else:
                    live_all = all(rows_by(rows, arm, s, t)["n_lb_gt_m70"] >= 1 for t in range(Tf - 1, T + 1))
                    dead_x = rows_by(rows, arm, s, Tf - 1)["n_ub_ge_exp0"] == 0
                    cls = "EPS_REAL" if live_all else ("ZERO_X" if dead_x else "OPEN")
                srow["lb_max_at_freeze_minus1"] = (rows_by(rows, arm, s, Tf - 1)["lb_max"] if Tf else None)
                srow["n_lb_gt_m70_min_frozen"] = (min(rows_by(rows, arm, s, t)["n_lb_gt_m70"]
                                                      for t in range(Tf - 1, T + 1)) if Tf else None)
            srow["class"] = cls
            if Tf is not None:
                srow["l2_ratio_quanta_frozen_median"] = float(np.median(
                    [rows_by(rows, arm, s, t)["l2_ratio_in_quanta"] for t in range(Tf, T + 1)]))
            seeds.append(srow)
    return rows, seeds


def rows_by(rows, arm, s, t):
    for r in rows:
        if r["arm"] == arm and r["seed"] == s and r["task"] == t:
            return r
    raise KeyError((arm, s, t))


# ----------------------------------------------------------------------------------------------
# C1 / C3 and S3, S4
# ----------------------------------------------------------------------------------------------

MU_ARMS = ("ref", "cap_perp", "cap_par", "cap_both")


def taskend(d: dict, s: int) -> dict[int, int]:
    task, step = d[f"s{s}_task"], d[f"s{s}_step"]
    ends = {}
    for i, (t, st) in enumerate(zip(task, step)):
        if st == 6000:
            ends[int(t)] = i
    return ends


def section_c():
    base = RAWROOT / "mucap_ee_0917" / "results" / "mucap_ee_0917" / "runs"
    rows, seeds = [], []
    s3 = {"n": 0, "upper_fail": 0, "lower_fail": 0, "mutant_fail": 0, "skipped_U_pos": 0, "skipped_g0": 0}
    for arm in MU_ARMS:
        for s in range(10):
            d = dict(np.load(base / f"{arm}_s{s}" / "units.npz"))
            ends = taskend(d, s)
            T = max(ends)
            k = lambda name: d[f"s{s}_{name}"]
            U2, b1, b2 = k("U_l2"), k("b1"), k("b2")
            rn1, rn2, zb1, zb2 = k("row_norm_l1"), k("row_norm_l2"), k("zbar_l1"), k("zbar_l2")
            mh, ef, gm = k("mhat_rms_l1"), k("eps_frac_l1"), k("gate_mean_l2")
            dead, frozen, decay = {}, {}, {}
            for t in range(1, T + 1):
                i = ends[t]
                nd = int((U2[i] <= Z_T).sum())
                dead[t] = nd == 100
                if t > 1:
                    j = ends[t - 1]
                    frozen[t] = all(np.array_equal(a[i], a[j]) for a in (b1, b2, rn1, rn2, zb1, zb2))
                decay[t] = bool((mh[i] == 0).all())
                rows.append({"arm": arm, "seed": s, "task": t, "n_dead2": nd,
                             "n_margin2": int((np.abs(U2[i] - Z_T) < 1e-4).sum()),
                             "U2_max": float(U2[i].max()), "zbar2_mean": float(zb2[i].mean()),
                             "frozen": frozen.get(t), "mhat_l1_max": float(mh[i].max()),
                             "mhat_l1_zero_units": int((mh[i] == 0).sum()),
                             "epsfrac_l1_mean": float(ef[i].mean())})
                # S3 on this row
                for u in range(100):
                    if U2[i, u] > 0:
                        s3["skipped_U_pos"] += 1
                        continue
                    if gm[i, u] <= 0:
                        s3["skipped_g0"] += 1
                        continue
                    s3["n"] += 1
                    lg = math.log(gm[i, u])
                    s3["upper_fail"] += int(math.log(1200.0) + lg < U2[i, u] - 1e-5)
                    s3["lower_fail"] += int(lg > U2[i, u] + 1e-5)
                    s3["mutant_fail"] += int(lg < U2[i, u] - 1e-5)
            frozen[1] = False
            Td = t_all(dead, T)
            Tf = t_freeze(frozen, T)
            cls = []
            if Td is not None and Tf is not None and Tf - Td in (1, 2):
                cls.append("FREEZE_AT_DEATH")
            if Td is not None and any(not frozen[t] for t in range(Td + 3, T + 1)):
                cls.append("MOVES_WHILE_DEAD")
            fwl = [t for t in range(2, T + 1) if frozen[t] and not dead[t - 1]]
            if fwl:
                cls.append("FROZEN_WHILE_LIVE")
            if Td is None and Tf is None:
                cls.append("NO_FULL_DEATH")
            if not cls:
                cls.append("OTHER")
            if Td is None or Td >= 99:
                c3 = "NOT_REACHED"
            elif all(decay[t] for t in range(Td + 2, T + 1)):
                c3 = "HISTORY_DECAY"
            else:
                c3 = "NOT_DECAY"
            last = rows[-1]
            seeds.append({"arm": arm, "seed": s, "T": T, "T_dead": Td, "T_freeze": Tf, "C1": "+".join(cls),
                          "frozen_while_live_tasks": fwl[:10], "C3": c3, "n_dead2_end": last["n_dead2"],
                          "U2_max_end": last["U2_max"], "zbar2_mean_end": last["zbar2_mean"],
                          "epsfrac_l1_end": last["epsfrac_l1_mean"], "mhat_l1_zero_units_end": last["mhat_l1_zero_units"],
                          "first_task_all_dead": next((t for t in range(1, T + 1) if dead[t]), None),
                          "n_tasks_frozen": sum(1 for t in range(2, T + 1) if frozen[t])})
    s3["pass"] = s3["n"] > 0 and s3["upper_fail"] == 0 and s3["lower_fail"] == 0 and s3["mutant_fail"] > 0
    return rows, seeds, s3


def check_s4():
    """dead (U2 <= Z_T) <=> dtrain_zero_l2 == 1 on l2cap_ee_0917's ref (task ends), and that ref's U2
    equals mucap_ee_0917's ref bit for bit."""
    base = RAWROOT / "l2cap_ee_0917" / "results" / "l2cap_ee_0917" / "runs"
    mu = RAWROOT / "mucap_ee_0917" / "results" / "mucap_ee_0917" / "runs"
    res = {"n": 0, "mismatch": 0, "mismatch_margin": 0, "U2_equal_mucap_ref": True}
    for s in range(10):
        d = dict(np.load(base / f"ref_s{s}" / "units.npz"))
        m = dict(np.load(mu / f"ref_s{s}" / "units.npz"))
        ends, mends = taskend(d, s), taskend(m, s)
        U2, dz = d[f"s{s}_U_l2"], d[f"s{s}_dtrain_zero_l2"]
        for t, i in ends.items():
            a = U2[i] <= Z_T
            b = dz[i] == 1.0
            res["n"] += a.size
            mis = a != b
            res["mismatch"] += int(mis.sum())
            res["mismatch_margin"] += int((mis & (np.abs(U2[i] - Z_T) < 1e-4)).sum())
            if not np.array_equal(U2[i], m[f"s{s}_U_l2"][mends[t]]):
                res["U2_equal_mucap_ref"] = False
    res["pass"] = res["mismatch"] == res["mismatch_margin"] and res["U2_equal_mucap_ref"]
    return res


# ----------------------------------------------------------------------------------------------
# D
# ----------------------------------------------------------------------------------------------

def section_d(c_rows):
    from src import l2_wall_0916 as LW          # imports torch and the layer_chimera engine (no GPU use)
    models = list(LW.VARIANT_MODELS)
    L = np.load(RAWROOT / "l2_wall_0916" / "results" / "l2_wall_0916" / "logs" / "diag_variant.npz")
    task, step = L["task"], L["step"]
    endstep = int(step.max())
    ends = {int(t): i for i, (t, st) in enumerate(zip(task, step)) if st == endstep}
    U2 = L["u_U_l2"]
    tiny = L["u_tiny_l2"]
    epsf = L["u_adam_epsfrac_l2"]
    zb2 = L["u_zbar_l2"]
    rows, dep = [], []
    pick = [(j, m) for j, m in enumerate(models) if m["env"] == "RL" and m["act2"] == "ELU1"
            and m["arm"] in ("none", "eps1e-6", "eps1e-30")]
    for j, m in pick:
        for t in (10, 20, 30, 40, 50):
            i = ends[t]
            rows.append({"pair": f'{m["act1"]}->{m["act2"]}', "arm": m["arm"], "eps": m["eps"], "seed": m["seed"],
                         "task": t, "exp_zero_units": int((U2[i, j] < EXP0).sum()),
                         "U2_max": float(U2[i, j].max()), "U2_median": float(np.median(U2[i, j])),
                         "tiny_mean": float(tiny[i, j].mean()), "epsfrac_l2_mean": float(epsf[i, j].mean()),
                         "zbar2_mean": float(zb2[i, j].mean()),
                         "frozen_vs_prev_end": (bool(np.array_equal(zb2[i, j], zb2[ends[t - 1], j]))
                                                if t - 1 in ends else None)})
        if m["arm"] == "none" and m["act1"] == "ELU1":
            for t in (10, 20, 30, 40, 50):
                dep.append({"lineage": "L2", "run": "l2_wall_0916 none EE", "seed": m["seed"], "task": t,
                            "zbar2_mean": float(zb2[ends[t], j].mean())})
            dep.append({"lineage": "L2", "run": "l2_wall_0916 none EE", "seed": m["seed"], "task": "slope41_50",
                        "zbar2_mean": float((zb2[ends[50], j].mean() - zb2[ends[40], j].mean()) / 10)})
    for r in c_rows:
        if r["arm"] == "ref" and r["task"] in (10, 20, 30, 40, 50, 100):
            dep.append({"lineage": "L1", "run": "mucap_ee_0917 ref", "seed": r["seed"], "task": r["task"],
                        "zbar2_mean": r["zbar2_mean"]})
    for s in range(10):
        z = {r["task"]: r["zbar2_mean"] for r in c_rows if r["arm"] == "ref" and r["seed"] == s}
        dep.append({"lineage": "L1", "run": "mucap_ee_0917 ref", "seed": s, "task": "slope41_50",
                    "zbar2_mean": (z[50] - z[40]) / 10})
    at50 = [r for r in rows if r["task"] == 50]
    d1 = "L2_NO_FLOOR" if len(at50) == 18 and all(r["exp_zero_units"] / 100 < 0.1 for r in at50) else "L2_FLOOR_REACHED"
    return rows, dep, d1


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUT / "lineage.csv", LINEAGE)
    b_rows, b_seeds = section_b()
    write_csv(OUT / "l3_readout.csv", b_rows)
    write_csv(OUT / "l3_seeds.csv", b_seeds)
    c_rows, c_seeds, s3 = section_c()
    write_csv(OUT / "l1_timecourse.csv", c_rows)
    write_csv(OUT / "l1_seeds.csv", c_seeds)
    s4 = check_s4()
    d_rows, dep, d1 = section_d(c_rows)
    write_csv(OUT / "l2_variant.csv", d_rows)
    write_csv(OUT / "depth_report.csv", dep)
    chk = {"S3": s3, "S4": s4, "D1": d1}
    (OUT / "tables_checks.json").write_text(json.dumps(chk, indent=2))
    print(json.dumps(chk, indent=2))
    for r in b_seeds:
        print(r)


if __name__ == "__main__":
    main()
