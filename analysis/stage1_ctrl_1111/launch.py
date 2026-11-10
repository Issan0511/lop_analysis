#!/usr/bin/env python3
"""stage1_ctrl_1111 の起動（spec §7）。同時 MAXJ 本まで・空きメモリが MINFREE_GB を切ったら新しい本を起動しない。
順: C-SGD → B1 → D → B2（C-SGD の選びの後）→ T → B3。出力の json が在る走は飛ばす（再開できる）。
使い方: python3 analysis/stage1_ctrl_1111/launch.py
"""
import json, os, subprocess, sys, time
from pathlib import Path
import numpy as np

W = Path(__file__).resolve().parents[2]
PY = "/home/issan/Projects/claude/proj_004_drift/.venv/bin/python"
SRC = str(W / "src/stage1_ctrl_1111.py")
OUT = W / "results/stage1_ctrl_1111"
RAW, CAL, LOGS = OUT / "raw", OUT / "raw_calib", OUT / "logs"
MAXJ, MINFREE_GB = int(os.environ.get("MAXJ", 12)), 4.0
ACTS = ["LR", "R", "ELU", "SILU", "GELU"]
SEEDS = list(range(10))
SGD_GRID = [0.01, 0.03, 0.1, 0.3, 1.0]
LOGS.mkdir(parents=True, exist_ok=True)
LOG = open(OUT / "launch_log.jsonl", "a")


def avail_gb():
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) / 2**20
    return 0.0


def job(name, args, out_json):
    return dict(name=name, cmd=[PY, SRC] + args, out_json=out_json)


def jobs_calib():
    J = []
    for act in ACTS:
        for lr in SGD_GRID:
            for s in (100, 101, 102):
                J.append(job(f"cal_sgd_{act}_lr{lr:g}_s{s}",
                             ["--mode", "branch", "--opt", "sgd", "--lr", str(lr), "--act", act, "--seed", str(s),
                              "--arms", "none", "--fresh", "0", "--out", str(CAL)],
                             CAL / f"branch_sgd_lr{lr:g}_{act}_s{s}.json"))
    return J


def jobs_b1():
    return [job(f"B1_{act}_s{s}", ["--mode", "branch", "--opt", "adam", "--lr", "0.001", "--act", act, "--seed", str(s),
                                   "--save-states", "1", "--out", str(RAW)], RAW / f"branch_adam_lr0.001_{act}_s{s}.json")
            for s in SEEDS for act in ACTS]


def jobs_d():
    return [job(f"D_{act}_s{s}", ["--mode", "dial", "--act", act, "--seed", str(s), "--draws", "5", "--out", str(RAW)],
                RAW / f"dial_{act}_s{s}.json") for s in (0, 1) for act in ACTS]


def select_sgd():
    sel = {}
    for act in ACTS:
        score = {}
        for lr in SGD_GRID:
            vals, ok = [], True
            for s in (100, 101, 102):
                f = CAL / f"branch_sgd_lr{lr:g}_{act}_s{s}.npz"
                d = np.load(f)
                accs = [float(d[f"S|t{t:02d}|eval"][-1, 0]) for t in range(1, 31)]
                ces = np.concatenate([d[f"S|t{t:02d}|eval"][:, 1] for t in range(1, 31)])
                if not np.all(np.isfinite(ces)): ok = False
                vals.append(np.mean(accs))
            score[lr] = float(np.mean(vals)) if ok else None
        cand = [(v, lr) for lr, v in score.items() if v is not None]
        best = max(v for v, _ in cand)
        chosen = min(lr for v, lr in cand if v >= best - 0.005)
        sel[act] = dict(lr=chosen, score=score)
    (OUT / "sgd_lr.json").write_text(json.dumps(sel, indent=1))
    return sel


def jobs_b2(sel):
    arms = "P0,P0m,P,Q,R,H0,F200"
    return [job(f"B2_{act}_s{s}", ["--mode", "branch", "--opt", "sgd", "--lr", str(sel[act]["lr"]), "--act", act, "--seed", str(s),
                                   "--arms", arms, "--save-states", "1", "--out", str(RAW)],
                RAW / f"branch_sgd_lr{sel[act]['lr']:g}_{act}_s{s}.json") for s in SEEDS for act in ACTS]


def jobs_t():
    return [job(f"T_{arm}_{act}_s{s}", ["--mode", "persist", "--arm", arm, "--opt", "adam", "--lr", "0.001", "--act", act,
                                        "--seed", str(s), "--out", str(RAW)], RAW / f"persist_{arm}_adam_lr0.001_{act}_s{s}.json")
            for arm in ("H0", "P", "A0", "R", "Q", "F200") for s in SEEDS for act in ACTS]


def jobs_b3():
    return [job(f"B3_{act}_s{s}", ["--mode", "branch", "--opt", "adaml", "--lr", "0.001", "--act", act, "--seed", str(s),
                                   "--arms", "P,H0,A0", "--out", str(RAW)], RAW / f"branch_adaml_lr0.001_{act}_s{s}.json")
            for s in SEEDS for act in ACTS]


running = {}


def poll():
    for name in list(running):
        p, j, t0, fh = running[name]
        rc = p.poll()
        if rc is not None:
            fh.close()
            LOG.write(json.dumps(dict(name=name, rc=rc, start=t0, end=time.time(), cmd=j["cmd"])) + "\n"); LOG.flush()
            print(f"[{time.strftime('%H:%M:%S')}] done {name} rc={rc} ({time.time() - t0:.0f}s) running={len(running) - 1}", flush=True)
            del running[name]


def run_queue(J):
    q = [j for j in J if not Path(j["out_json"]).exists()]
    print(f"[{time.strftime('%H:%M:%S')}] queue {len(q)} / {len(J)}", flush=True)
    while q or running:
        poll()
        while q and len(running) < MAXJ and avail_gb() >= MINFREE_GB:
            j = q.pop(0)
            fh = open(LOGS / f"{j['name']}.log", "w")
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            p = subprocess.Popen(j["cmd"], stdout=fh, stderr=subprocess.STDOUT, cwd=str(W), env=env)
            running[j["name"]] = (p, j, time.time(), fh)
            print(f"[{time.strftime('%H:%M:%S')}] start {j['name']} running={len(running)} avail={avail_gb():.1f}GB", flush=True)
        time.sleep(2)


def main():
    """一つの列で回す。C-SGD が全部終わった時点で、選んだ学習率の B2 を、まだ起動していない B1・D の後ろに差し込む。"""
    calib = jobs_calib()
    q = [j for j in calib + jobs_b1() + jobs_d() if not Path(j["out_json"]).exists()]
    tail = [j for j in jobs_t() + jobs_b3() if not Path(j["out_json"]).exists()]
    b2_added = False
    print(f"[{time.strftime('%H:%M:%S')}] queue {len(q)} + later {len(tail)}", flush=True)
    while q or tail or running or not b2_added:
        poll()
        if not b2_added and all(Path(j["out_json"]).exists() for j in calib):
            sel = select_sgd(); print("SGD lr:", {a: v["lr"] for a, v in sel.items()}, flush=True)
            b2 = [j for j in jobs_b2(sel) if not Path(j["out_json"]).exists()]
            q = q + b2; b2_added = True
            print(f"[{time.strftime('%H:%M:%S')}] B2 added: {len(b2)}", flush=True)
        if not q and b2_added and tail:
            q, tail = tail, []
        while q and len(running) < MAXJ and avail_gb() >= MINFREE_GB:
            j = q.pop(0)
            fh = open(LOGS / f"{j['name']}.log", "w")
            env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
            p = subprocess.Popen(j["cmd"], stdout=fh, stderr=subprocess.STDOUT, cwd=str(W), env=env)
            running[j["name"]] = (p, j, time.time(), fh)
            print(f"[{time.strftime('%H:%M:%S')}] start {j['name']} running={len(running)} avail={avail_gb():.1f}GB", flush=True)
        if not running and not q and not tail and b2_added:
            break
        if not running and not q and not b2_added and not all(Path(j["out_json"]).exists() for j in calib):
            raise SystemExit("C-SGD の走が出力を残さずに終わった（logs を見る）")
        time.sleep(2)
    failed = [json.loads(l) for l in open(OUT / "launch_log.jsonl") if json.loads(l)["rc"] != 0]
    print(f"[{time.strftime('%H:%M:%S')}] ALL DONE. failed={len(failed)}", flush=True)


if __name__ == "__main__":
    main()
