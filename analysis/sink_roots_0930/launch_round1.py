#!/usr/bin/env python3
"""sink_roots_0930 round 1 (spec_sink_roots_0930_round1.md): the parent's requests 1-4, at most MAX jobs at once,
in the parent's priority order.  A job is done when its `.done` marker exists (written on rc 0).
`results/sink_roots_0930/STOP_round1` stops new starts."""
import json, os, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930")
R1 = RAW / "round1"
RES = ROOT / "results" / "sink_roots_0930"
LOG = RES / "launch_round1_log.jsonl"
STOP = RES / "STOP_round1"
MAX = 2
PY = sys.executable
SR = ROOT / "src" / "sink_roots_round1"
ENG = [PY, str(ROOT / "src" / "sink_roots_mnist_0930.py")]


def job(name, cmd, out, env=None):
    return {"name": name, "cmd": cmd, "out": Path(out), "env": env or {}}


def jobs():
    J = []
    for act in ("GELU", "SILU"):                                         # 1 valley R_eps_wall
        for eps in ("1e-6", "1e-12"):
            for s in (0, 1):
                J.append(job(f"Reps_{act}_eps{eps}_s{s}",
                             [PY, str(SR / "valley" / "alpha_ladder_1layer.py"), "--act", act, "--seeds", str(s),
                              "--eps1", eps, "--tasks", "200", "--out", str(R1 / "R_eps_wall"), "--tag", f"{act}_eps{eps}"],
                             R1 / "R_eps_wall" / f"{act}_eps{eps}_s{s}"))
    J.append(job("Q1_ELU_s0", [PY, str(SR / "return" / "rho_test.py"), "--act", "ELU", "--seed", "0"], R1 / "Q1" / "ELU_s0"))
    J.append(job("Q1_LR_s0", [PY, str(SR / "return" / "rho_test.py"), "--act", "LR", "--seed", "0", "--w0", "500"],
                 R1 / "Q1" / "LR_s0"))
    for act in ("ELU", "LR", "SILU"):                                    # 3 R6
        for s in (0, 1, 2):
            n = f"R6ink_{act}_s{s}"
            J.append(job(n, ENG + ["--name", n, "--out", str(RAW / "mnist" / n), "--act", act, "--seed", str(s),
                                   "--tasks", "30", "--ink-normalize", "--snap-before", "5", "10", "20", "30"],
                         RAW / "mnist" / n))
    for act in ("ELU", "SILU"):                                          # 3 R1
        for s in (0, 1, 2):
            n = f"R1wcap_{act}_s{s}"
            J.append(job(n, ENG + ["--name", n, "--out", str(RAW / "mnist" / n), "--act", act, "--seed", str(s),
                                   "--tasks", "30", "--wcap-from", "5"], RAW / "mnist" / n))
    S = R1 / "sign_states"                                               # 4 replay
    groups = [[f"L2_ELU_K10_s0_t00{t}.npz" for t in (2, 3, 4, 6)], [f"L2_ELU_K10_s1_t00{t}.npz" for t in (2, 3, 4, 6)],
              [f"L2_LR_K10_s0_t00{t}.npz" for t in (2, 3, 4, 6)], [f"L2_LR_K10_s1_t00{t}.npz" for t in (2, 3, 4, 6)],
              [f"L2_SILU_K10_s0_t00{t}.npz" for t in (2, 3, 4, 6)], [f"L2_SILU_K10_s1_t00{t}.npz" for t in (2, 3, 4, 6)],
              [f"L1_ELU_K10_s0_t00{t}.npz" for t in (2, 3, 5)] + [f"L1_LR_K10_s0_t00{t}.npz" for t in (2, 3, 5)]]
    for i, g in enumerate(groups):
        J.append(job(f"replay_{i}", [PY, str(SR / "sign" / "replay.py")] + [str(S / f) for f in g],
                     R1 / "replay" / f"g{i}", env={"REPLAY_OUT": str(R1 / "replay")}))
    A6 = [PY, str(SR / "adam" / "twolayer_a6_probe.py")]                 # 4 F1
    J.append(job("F1_dry", A6 + ["--steps", "50", "--probe", "2", "--R", "2", "--out", str(R1 / "F1_dry")], R1 / "F1_dry" / "job"))
    J.append(job("F1_s0_p23", A6 + ["--seed", "0", "--probe", "2,3", "--R", "16", "--out", str(R1 / "F1")], R1 / "F1" / "s0_p23"))
    J.append(job("F1_s1_p23", A6 + ["--seed", "1", "--probe", "2,3", "--R", "16", "--out", str(R1 / "F1")], R1 / "F1" / "s1_p23"))
    J.append(job("F1_s0_p10", A6 + ["--seed", "0", "--probe", "10", "--R", "12", "--out", str(R1 / "F1")], R1 / "F1" / "s0_p10"))
    return J


def main():
    J = [j for j in jobs() if not (j["out"] / ".done").exists()]
    print(f"{len(J)} round-1 jobs", flush=True)
    RES.mkdir(parents=True, exist_ok=True)
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
            while J and len(running) < MAX:
                j = J.pop(0)
                if j["name"] == "F1_s0_p23" and not (R1 / "F1_dry" / "job" / ".done").exists():
                    J.insert(0, j)                                        # the dry run gates the F1 mains
                    break
                j["out"].mkdir(parents=True, exist_ok=True)
                fh = open(j["out"] / "stdout.log", "w")
                env = dict(os.environ, **j["env"], OMP_NUM_THREADS="1", MKL_NUM_THREADS="1")
                p = subprocess.Popen(j["cmd"], stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     cwd=ROOT, env=env)
                running[j["name"]] = (p, j, time.time(), fh)
        elif not running:
            break
        time.sleep(5)
    print("round-1 queue finished" if not J else f"stopped with {len(J)} left", flush=True)


if __name__ == "__main__":
    main()
