#!/usr/bin/env python3
"""sink_roots_0930 launcher (spec §10): a job queue with at most MAX_ALL processes, R7 at most MAX_R7.

    python3 analysis/sink_roots_0930/launch.py            # stage 1: everything but R7's S mains
    python3 analysis/sink_roots_0930/launch.py --s-eta X  # stage 2: R7's S mains with eta_S = X

A job starts only when its dependency file exists; `results/sink_roots_0930/STOP` stops new starts.
Finished jobs (their `done` file exists) are skipped, so a relaunch resumes the queue.
"""
import argparse, json, subprocess, sys, time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930")
RES = ROOT / "results" / "sink_roots_0930"
STOP = RES / "STOP"
LOG = RES / "launch_log.jsonl"
MAX_ALL, MAX_R7 = 10, 7
ENG = [sys.executable, str(ROOT / "src" / "sink_roots_mnist_0930.py")]
R7 = [sys.executable, str(ROOT / "src" / "postfit_elu_cifar_0924.py")]
ACTS = ["ELU", "GELU", "SILU", "LR"]
SEEDS = [0, 1, 2]
ETA_GRID = [0.001, 0.003, 0.01, 0.03, 0.1]


def mn(name, args, dep=None, group="mnist"):
    out = RAW / "mnist" / name
    return {"name": name, "group": group, "cmd": ENG + ["--name", name, "--out", str(out)] + args,
            "done": out / "provenance.json", "dep": dep, "out": out}


def jobs(s_eta=None):
    J = []
    if s_eta is not None:
        for s in range(10):
            out = RAW / "r7" / "S" / f"s{s}"
            J.append({"name": f"R7_S_s{s}", "group": "r7", "done": out / "provenance.json", "dep": None, "out": out,
                      "cmd": R7 + ["--arm", "S", "--eta", str(s_eta), "--out", str(out), "--device", "cpu",
                                   "--threads", "1", "--seeds", str(s), "--no-keep-ckpts"]})
        return J
    # R7 pilot first (it gates the S mains), then the long mains
    for eta in ETA_GRID:
        for s in range(10):
            out = RAW / "r7" / "_pilot" / f"eta{eta}" / f"s{s}"
            J.append({"name": f"R7_pilot_eta{eta}_s{s}", "group": "r7", "done": out / "provenance.json",
                      "dep": None, "out": out,
                      "cmd": R7 + ["--arm", "S", "--eta", str(eta), "--tasks", "2", "--pilot", "--out", str(out),
                                   "--device", "cpu", "--threads", "1", "--seeds", str(s), "--no-keep-ckpts"]})
    for arm in ["A", "F", "F99"]:
        for s in range(10):
            out = RAW / "r7" / arm / f"s{s}"
            J.append({"name": f"R7_{arm}_s{s}", "group": "r7", "done": out / "provenance.json", "dep": None,
                      "out": out, "cmd": R7 + ["--arm", arm, "--out", str(out), "--device", "cpu", "--threads", "1",
                                               "--seeds", str(s), "--no-keep-ckpts"]})
    # R3 (the parent waits for it)
    arms = {"base": ["--snap-before", "2", "5", "10", "20", "30"], "b2_099": ["--b2", "0.99"],
            "b2_09": ["--b2", "0.9"], "T1k": ["--T", "1000"], "T16k": ["--T", "16000"],
            "adamreset": ["--adamreset"], "vrestore": ["--vrestore", "200"], "ls01": ["--ls", "0.1"],
            "sq003": ["--sq", "0.03"]}
    for arm, extra in arms.items():
        for act in ACTS:
            for s in SEEDS:
                J.append(mn(f"R3_{arm}_{act}_s{s}", ["--act", act, "--seed", str(s), "--tasks", "30"] + extra))
    for act in ["ELU", "GELU", "SILU"]:
        for s in SEEDS:
            J.append(mn(f"R3_bwfloor_{act}_s{s}", ["--act", act, "--seed", str(s), "--tasks", "30",
                                                   "--bwmode", "floor", "--bw-from", "200"]))
    for act in ["GELU", "SILU"]:
        for s in SEEDS:
            J.append(mn(f"R3_bwabs_{act}_s{s}", ["--act", act, "--seed", str(s), "--tasks", "30",
                                                 "--bwmode", "abs", "--bw-from", "200"]))
    # R5 mains, then forks
    for act in ["ELU", "LR"]:
        for s in SEEDS:
            J.append(mn(f"R5_main_{act}_s{s}", ["--act", act, "--seed", str(s), "--tasks", "200", "--probes", "ends",
                                                "--ckpt-after", "10", "50", "200"]))
    # R11a
    for tau in [0.25, 0.5, 0.75, 0.9]:
        for act in ACTS:
            for s in SEEDS:
                J.append(mn(f"R11a_tau{tau}_{act}_s{s}", ["--act", act, "--seed", str(s), "--tasks", "30",
                                                          "--tau", str(tau)]))
    # R9
    for act in ["GELU", "SILU"]:
        for s in SEEDS:
            J.append(mn(f"R9_V2sgd_{act}_s{s}", ["--act", act, "--seed", str(s), "--tasks", "30", "--opt", "sgd",
                                                 "--lr", "0.1"]))
    for act in ["GELU", "SILU", "ELU", "LR"]:
        for s in SEEDS:
            J.append(mn(f"R9_V3rl2_{act}_s{s}", ["--act", act, "--seed", str(s), "--layers", "2", "--T", "6000",
                                                 "--tasks", "20", "--snap-before", "2", "3", "4", "5", "10"]))
    for act in ["GELU", "SILU"]:
        for s in SEEDS:
            J.append(mn(f"R9_V4pm6000_{act}_s{s}", ["--act", act, "--seed", str(s), "--layers", "2", "--env", "pm",
                                                    "--T", "6000", "--tasks", "20"]))
            J.append(mn(f"R9_V5pm625_{act}_s{s}", ["--act", act, "--seed", str(s), "--layers", "2", "--env", "pm",
                                                   "--T", "625", "--tasks", "100"]))
    # R5 forks (depend on the mains' checkpoints)
    for act in ["ELU", "LR"]:
        for s in SEEDS:
            main = RAW / "mnist" / f"R5_main_{act}_s{s}"
            for t in [10, 50, 200]:
                ck = main / f"ckpt_after_t{t:03d}.pt"
                for mode in ["nat", "long", "adam0", "ro_reinit", "l1_reinit", "recenter", "fresh"]:
                    extra = ["--T", "16000"] if mode == "long" else []
                    J.append(mn(f"R5_fork_{act}_s{s}_t{t}_{mode}",
                                ["--act", act, "--seed", str(s), "--tasks", "1", "--probes", "ends",
                                 "--fork-ckpt", str(ck), "--fork-mode", mode] + extra, dep=ck))
    return J


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--s-eta", type=float, default=None)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    J = [j for j in jobs(a.s_eta) if not Path(j["done"]).exists()]
    print(f"{len(J)} jobs to run", flush=True)
    if a.dry:
        for j in J[:5] + J[-5:]:
            print(j["name"], " ".join(j["cmd"][1:]))
        return
    RES.mkdir(parents=True, exist_ok=True)
    (RAW / "logs").mkdir(parents=True, exist_ok=True)
    running = {}
    while J or running:
        for name, (p, j, t0, fh) in list(running.items()):
            rc = p.poll()
            if rc is not None:
                fh.close()
                with LOG.open("a") as f:
                    f.write(json.dumps({"name": name, "rc": rc, "start": t0, "end": time.time(),
                                        "cmd": j["cmd"]}) + "\n")
                del running[name]
        if not STOP.exists():
            for j in list(J):
                n_all = len(running)
                n_r7 = sum(1 for v in running.values() if v[1]["group"] == "r7")
                mn_left = any(x["group"] == "mnist" for x in J)
                cap_r7 = MAX_R7 if mn_left else MAX_ALL
                if n_all >= MAX_ALL:
                    break
                if j["group"] == "r7" and n_r7 >= cap_r7:
                    continue
                if j["dep"] is not None and not Path(j["dep"]).exists():
                    continue
                Path(j["out"]).mkdir(parents=True, exist_ok=True)
                fh = open(RAW / "logs" / f"{j['name']}.log", "w")
                p = subprocess.Popen(j["cmd"], stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL,
                                     cwd=ROOT)
                running[j["name"]] = (p, j, time.time(), fh)
                J.remove(j)
        elif not running:
            break
        time.sleep(5)
    print("queue finished" if not J else f"stopped with {len(J)} jobs left", flush=True)


if __name__ == "__main__":
    main()
