#!/usr/bin/env python3
"""(B) 折れ目の喪失（部品 5）: 本走 10 seed の層ごとの応答の状態と、学ぶ力の 3 相（spec §3）。

  PYTHONDONTWRITEBYTECODE=1 /home/issan/Projects/claude/proj_004_drift/.venv/bin/python \
      analysis/chargpt_load_fold_1021/fold/fold.py

読むだけ: /home/issan/Projects/claude/wt/chargpt_cl_1019/results/chargpt_cl_1019（HEAD f71be184）。
書く: results/chargpt_load_fold_1021/fold/。新しい学習はしない（numpy だけ）。

unit_stats.npz の q は z < 0 の token の割合（閉じた割合、src/chargpt_cl_1019.py 427 行）。
ReLU では φ′ = 1[z > 0] なので、入力間の傾きの分散 Var_x(φ′) = q(1 − q) が厳密。
"""
import csv
import json
import subprocess
from pathlib import Path

import numpy as np

SRC = Path("/home/issan/Projects/claude/wt/chargpt_cl_1019/results/chargpt_cl_1019")
OUT = Path(__file__).resolve().parents[3] / "results/chargpt_load_fold_1021/fold"
ARMS = ("C3", "B3", "C4", "N4")
SEEDS = range(10)
PROBE = "uniform"           # spec §3: 一様 probe（seed だけから作る、全課題で同じ）
DEAD_Q, POS_Q = 0.995, 0.01  # 死 = p < 0.005（ただし判定は記録の dead を使う）、正側片側化 = p >= 0.99
RESP_JUMP = 0.10            # judge_m2 と同じ
T_NEVER = 10_000


def run_mean_first(e, thr, w=50):
    c = np.convolve(e, np.ones(w) / w, mode="valid")
    hit = np.nonzero(c <= thr)[0]
    return float(hit[0]) if hit.size else float(T_NEVER)


def phases(loss, H):
    e = loss - H
    return dict(spike=float(e[:5].mean()), T015=run_mean_first(e, 0.15), T03=run_mean_first(e, 0.3),
                e1k=float(e[950:1050].mean()), eend=float(e[-1000:].mean()), all=float(e.mean()))


def load_run(run_dir):
    loss = np.load(run_dir / "online.npz")["loss"]
    H = {int(r["task"]): float(r["H"]) for r in csv.DictReader(open(run_dir / "per_task.csv"))}
    U = np.load(run_dir / "unit_stats.npz")
    return loss, H, U


def layer_states(U, probe=PROBE):
    q = U[f"{probe}_q"].astype(np.float64)            # (tasks, layers, units)
    dead = U[f"{probe}_dead"].astype(bool)
    pos1 = (q <= POS_Q) & ~dead
    nonresp = dead | pos1
    var = q * (1.0 - q)
    return dict(dead=dead.mean(-1), pos1=pos1.mean(-1), nonresp=nonresp.mean(-1),
                var_med=np.median(var, -1), q=q, deadu=dead)


def checks():
    """q の意味と 3 相の再現（spec §4 (B)）。"""
    out = {}
    U = np.load(SRC / "quicklook/M/C3/seed1000/unit_stats.npz")
    q, dead = U["uniform_q"], U["uniform_dead"]
    out["dead_implies_q1"] = bool(np.all(q[dead] == 1.0))
    out["q0_never_dead"] = bool(not np.any(dead[q == 0.0]))
    out["n_dead_units_checked"] = int(dead.sum())
    loss, H, _ = load_run(SRC / "quicklook/M/C3/seed1000")
    t_early = np.median([phases(loss[t - 1], H[t])["T015"] for t in range(2, 11)])
    t_late = np.median([phases(loss[t - 1], H[t])["T015"] for t in range(41, 51)])
    out["T015_seed1000_C3_early_late"] = [float(t_early), float(t_late)]
    out["T015_reproduces_515_1449"] = bool(abs(t_early - 515) <= 1 and abs(t_late - 1449) <= 1)
    out["all_pass"] = out["dead_implies_q1"] and out["q0_never_dead"] and out["T015_reproduces_515_1449"]
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    ck = checks()
    json.dump(ck, open(OUT / "checks.json", "w"), indent=1)
    if not ck["all_pass"]:
        raise SystemExit(f"checks failed: {ck}")

    rows_phase, rows_layer, rows_onset = [], [], []
    for arm in ARMS:
        per_seed = {}
        for s in SEEDS:
            d = SRC / f"M/{arm}/seed{s}"
            loss, H, U = load_run(d)
            n = loss.shape[0]
            ph = [phases(loss[t - 1], H[t]) for t in range(1, n + 1)]
            st = layer_states(U)
            fr = {}
            for tf in (10, 25, 50):
                f = d / f"fresh_online_t{tf}.npz"
                if f.exists():
                    fr[tf] = phases(np.load(f)["loss"], H[tf])
            per_seed[s] = (ph, st, fr)
            for t in range(1, n + 1):
                rows_phase.append(dict(arm=arm, seed=s, task=t, **ph[t - 1]))
                for l in range(st["dead"].shape[1]):
                    rows_layer.append(dict(arm=arm, seed=s, task=t, layer=l, dead=st["dead"][t - 1, l],
                                           pos1=st["pos1"][t - 1, l], nonresp=st["nonresp"][t - 1, l],
                                           var_med=st["var_med"][t - 1, l]))
        # δ_T = 3 × seed 間 SE（課題 2〜10 の T015 の seed ごとの平均）。算術で決める（spec §3）
        early_T = np.array([np.mean([per_seed[s][0][t - 1]["T015"] for t in range(2, 11)]) for s in SEEDS])
        dT = 3.0 * early_T.std(ddof=1) / np.sqrt(len(early_T))
        L3 = 3
        for s in SEEDS:
            ph, st, fr = per_seed[s]
            n = len(ph)
            T = np.array([p["T015"] for p in ph])
            ref = T[1:10].mean()
            onset_T = next((t for t in range(11, n + 1) if T[t - 5:t].mean() - ref >= dT), None)
            nr = st["nonresp"][:, L3]
            onset_r = next((t for t in range(2, n + 1) if nr[t - 1] - nr[0] >= RESP_JUMP), None)
            nr_all = st["nonresp"][9]  # 課題 10、全層
            rows_onset.append(dict(arm=arm, seed=s, dT=dT, ref_T015=ref, onset_T=onset_T, onset_resp_L3=onset_r,
                                   resp_leads=(onset_r is not None and (onset_T is None or onset_r <= onset_T + 5)),
                                   var_med_L3_t1=st["var_med"][0, L3], var_med_L3_t50=st["var_med"][-1, L3],
                                   dead_L3_t50=st["dead"][-1, L3], pos1_L3_t50=st["pos1"][-1, L3],
                                   nonresp_t10_min_over_layers=float(nr_all.min()),
                                   **{f"fresh{k}_{m}": v for k, f in fr.items() for m, v in f.items() if m in ("T015", "e1k", "eend")}))

    for name, rows in (("per_task_phase.csv", rows_phase), ("per_task_layer.csv", rows_layer), ("onsets.csv", rows_onset)):
        with open(OUT / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    # 事後（spec に無い、計算後に足した）: 課題を通した L3 の死と学ぶ速さの共変、L3 の生き返り、onset_T の打ち切り
    rows_post = []
    for arm in ARMS[:3]:
        for s in SEEDS:
            d = SRC / f"M/{arm}/seed{s}"
            U = np.load(d / "unit_stats.npz")
            du = U["uniform_dead"][:, 3].mean(-1)
            dc = U["chain_dead"][:, 3].mean(-1)
            ph = [r for r in rows_phase if r["arm"] == arm and r["seed"] == s]
            T = np.array([r["T015"] for r in ph])
            e1 = np.array([r["e1k"] for r in ph])
            on = next(r for r in rows_onset if r["arm"] == arm and r["seed"] == s)
            rows_post.append(dict(arm=arm, seed=s,
                                  corr_deadU_T015=float(np.corrcoef(du[1:], T[1:])[0, 1]),
                                  corr_deadC_T015=float(np.corrcoef(dc[1:], T[1:])[0, 1]),
                                  corr_deadU_e1k=float(np.corrcoef(du[1:], e1[1:])[0, 1]),
                                  peak_dead_L3=float(du.max()), task_of_peak=int(du.argmax()) + 1,
                                  dead_L3_t50=float(du[-1]), revived=bool(du.max() >= 0.3 and du[-1] <= 0.05),
                                  onset_T_censored=(on["onset_T"] == 11)))
    with open(OUT / "posthoc_L3.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows_post[0].keys()))
        w.writeheader()
        w.writerows(rows_post)

    prov = dict(src=str(SRC), src_head=subprocess.run(["git", "-C", str(SRC.parents[1]), "rev-parse", "HEAD"],
                                                      capture_output=True, text=True).stdout.strip(),
                probe=PROBE, pos_q=POS_Q, resp_jump=RESP_JUMP, t_never=T_NEVER)
    json.dump(prov, open(OUT / "provenance.json", "w"), indent=1)
    print("ok", len(rows_phase), len(rows_layer), len(rows_onset))


if __name__ == "__main__":
    main()
