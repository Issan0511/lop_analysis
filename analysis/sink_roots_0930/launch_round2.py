#!/usr/bin/env python3
"""sink_roots_0930 round 2 (spec_sink_roots_0930_round2.md): one queue for the 10 requests.
A job starts when its dependencies' `.done` markers exist and the number of my experiment processes on the machine
(python processes running a script under this worktree's src/, i.e. R7, round 1b and round 2; paused ones not counted) is below the budget
`total` of results/sink_roots_0930/caps_round2.json (read every loop).  `.done` is written on rc 0.
results/sink_roots_0930/STOP_round2 stops new starts."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930")
R2R = RAW / "round2"
RES = ROOT / "results" / "sink_roots_0930"
LOG = RES / "launch_round2_log.jsonl"
STOP = RES / "STOP_round2"
CAPS = RES / "caps_round2.json"
PY = sys.executable
SRC = ROOT / "src"
S2 = SRC / "sink_roots_round2"
ENG = [PY, str(SRC / "sink_roots_mnist_0930.py")]
MARK = str(SRC) + "/"


def budget():
    try:
        return int(json.loads(CAPS.read_text())["total"])
    except Exception:
        return 12


def my_procs():
    n = 0
    for p in Path("/proc").iterdir():
        if not p.name.isdigit():
            continue
        try:
            args = (p / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if args and b"python" in args[0] and any(MARK.encode() in x for x in args[1:3]):
            try:
                state = (p / "stat").read_text().rsplit(")", 1)[1].split()[0]
            except OSError:
                continue
            if state not in ("T", "t"):              # a job paused with SIGSTOP does not use a core
                n += 1
    return n


def job(name, cmd, deps=(), cwd=None, env=None, out=None):
    out = Path(out) if out else R2R / "jobs" / name
    return {"name": name, "cmd": [str(c) for c in cmd], "deps": [Path(d) for d in deps], "cwd": cwd or ROOT,
            "env": env or {}, "out": out}


def jobs():
    J = []
    # 1 R3prime_bwrelu (engine; next to the other R3 arms)
    for act in ("GELU", "SILU"):
        for s in (0, 1, 2):
            n = f"R3_bwrelu_{act}_s{s}"
            J.append(job(n, ENG + ["--name", n, "--out", RAW / "mnist" / n, "--act", act, "--seed", s, "--tasks", 30,
                                   "--bwmode", "relu", "--bw-from", 200], out=RAW / "mnist" / n))
    # 5 F2_branch_scale, 6 R5_rf_probe, 10(1) codex_rerun
    for act in ("ELU", "GELU"):
        for s in (0, 1, 2):
            J.append(job(f"F2_{act}_s{s}", [PY, S2 / "adam" / "branch_scale_probe.py", "--act", act, "--seed", s,
                                            "--switches", "10,20", "--out", R2R / "F2" / "branches_scale"], cwd=S2 / "adam"))
    for act in ("ELU", "GELU"):
        J.append(job(f"R5rf_{act}_s1", [PY, S2 / "adam" / "rf_probe.py", "--act", act, "--seed", 1,
                                        "--out", R2R / "R5rf" / "rf_runs"], cwd=S2 / "adam"))
    J.append(job("S1_codex_rerun", [PY, S2 / "valley" / "codex_rerun.py"], cwd=S2 / "valley"))
    # 2 tau_eff_vs_fit_speed: T 4000 first (states for 3 and 4), then the other lengths
    def tau(act, K, s, T):
        return job(f"tau_{act}_K{K}_s{s}_T{T}", [PY, S2 / "sign" / "dump_states.py", "--act", act, "--K", K, "--seed", s,
                                                  "--tasks", 12, "--steps", T, "--dump", "2,3,5,8,12",
                                                  "--out", R2R / "tau" / f"T{T}"], cwd=S2 / "sign")
    ACTS, KS, SEEDS, TS = ("ELU", "LR", "GELU"), (2, 10), (0, 1, 2), (500, 1000, 4000, 16000)
    for act in ACTS:
        for K in KS:
            for s in SEEDS:
                J.append(tau(act, K, s, 4000))
    J.append(tau("SILU", 10, 0, 4000))
    done = lambda name: R2R / "jobs" / name / ".done"
    st4 = R2R / "tau" / "T4000"
    # 3 leaky_C2_intervention
    J.append(job("leakyC2", [PY, S2 / "sign" / "leaky_c2_intervention.py"]
                 + [st4 / f"L1_LR_K10_s{s}_t{t:03d}.npz" for s in (0, 1) for t in (8, 12)] + ["--out", R2R / "leakyC2"],
                 deps=[done("tau_LR_K10_s0_T4000"), done("tau_LR_K10_s1_T4000")], cwd=S2 / "sign"))
    # 4 df_test
    (R2R / "df").mkdir(parents=True, exist_ok=True)
    for act, t in (("ELU", 5), ("ELU", 12), ("LR", 8), ("GELU", 8), ("SILU", 8)):
        for arm in ("full", "vc", "wb"):
            J.append(job(f"df_{act}_t{t:03d}_{arm}", [PY, S2 / "sign" / "df_test.py", st4 / f"L1_{act}_K10_s0_t{t:03d}.npz", arm, 12],
                         deps=[done(f"tau_{act}_K10_s0_T4000")], cwd=S2 / "sign", env={"DF_OUT": str(R2R / "df")}))
    for T in (500, 1000, 16000):
        for act in ACTS:
            for K in KS:
                for s in SEEDS:
                    J.append(tau(act, K, s, T))
    for T in TS:
        for act in ACTS:
            for K in KS:
                J.append(job(f"ftau_{act}_K{K}_T{T}", [PY, S2 / "sign" / "filter_tau.py", "--dir", R2R / "tau" / f"T{T}",
                                                        "--pattern", f"L1_{act}_K{K}_s*_t*.npz",
                                                        "--out", R2R / "tau" / "filter" / f"ftau_{act}_K{K}_T{T}.npy"],
                             deps=[done(f"tau_{act}_K{K}_s{s}_T{T}") for s in SEEDS], cwd=S2 / "sign"))
    # 7 R2_valley_long
    for act in ("GELU", "SILU"):
        for s in (0, 1):
            J.append(job(f"vlong_{act}_s{s}", [PY, S2 / "valley" / "alpha_ladder_1layer.py", "--act", act, "--seeds", s,
                                               "--tasks", 1000, "--snap_every", 100, "--out", R2R / "valley_long", "--tag", act],
                         cwd=S2 / "valley"))
    # 9 R3_readout_scale_clamp (+ the unclamped reference)
    for r in ("1.8", "14", "none"):
        for s in (0, 1):
            n = f"R3ro_{'r' + r if r != 'none' else 'none'}_GELU_s{s}"
            extra = [] if r == "none" else ["--ro-clamp", r, "--ro-clamp-from", 51]
            J.append(job(n, ENG + ["--name", n, "--out", RAW / "mnist" / n, "--act", "GELU", "--seed", s, "--tasks", 150] + extra,
                         out=RAW / "mnist" / n))
    # 8 R1_valley_N_ladder, 10(2) lr ladder
    for N in (2400, 1200, 600, 300):
        for act in ("gelu", "silu"):
            for s in (0, 1):
                J.append(job(f"nk_{act}_N{N}_s{s}", [PY, S2 / "valley" / "nk_sweep.py", "--act", act, "--N", N, "--seed", s,
                                                     "--tasks", 200, "--tag", f"{act}_N{N}", "--out", R2R / "nladder"],
                             cwd=S2 / "valley"))
    for lr in ("1e-2", "3e-2"):
        for act in ("gelu", "silu"):
            for s in (0, 1):
                J.append(job(f"nklr_{act}_lr{lr}_s{s}", [PY, S2 / "valley" / "nk_sweep.py", "--act", act, "--lr", lr, "--seed", s,
                                                          "--tasks", 100, "--tag", f"{act}_lr{lr}", "--out", R2R / "nk_lr"],
                             cwd=S2 / "valley"))
    return J


def main():
    J = [j for j in jobs() if not (j["out"] / ".done").exists()]
    print(f"{len(J)} round-2 jobs", flush=True)
    (R2R / "logs").mkdir(parents=True, exist_ok=True)
    running = {}
    while J or running:
        for name, (p, j, t0, fh) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                fh.close()
                if rc == 0:
                    (j["out"] / ".done").write_text(f"{time.time()}\n")
                with LOG.open("a") as f:
                    f.write(json.dumps({"name": name, "rc": rc, "start": t0, "end": time.time(), "cmd": j["cmd"]}) + "\n")
                del running[name]
        if not STOP.exists():
            for j in list(J):
                if my_procs() >= budget():
                    break
                if not all(d.exists() for d in j["deps"]):
                    continue
                j["out"].mkdir(parents=True, exist_ok=True)
                fh = open(R2R / "logs" / f"{j['name']}.log", "w")
                env = dict(os.environ, **j["env"], OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                p = subprocess.Popen(j["cmd"], stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     cwd=j["cwd"], env=env)
                running[j["name"]] = (p, j, time.time(), fh)
                J.remove(j)
                time.sleep(0.5)                       # let the new process appear in /proc before counting again
        elif not running:
            break
        time.sleep(5)
    print("round-2 queue finished" if not J else f"stopped with {len(J)} left", flush=True)


if __name__ == "__main__":
    main()
