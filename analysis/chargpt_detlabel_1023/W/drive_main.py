#!/usr/bin/env python3
"""chargpt_detlabel_1023 の本走の駆動（spec §10.2 の 4・§10.3 (ii)・§12.1）。担当 compute_W が書いた。主の承認の後に起こす。

腕 × seed の単位を、1 枚の GPU に最大 --slots 本（既定 4、chargpt_seed10_1022 §8 の上限）並べて回す。
完走した単位（out/provenance.json に detlabel_complete = true）は飛ばす（再開できる）。半端な単位は最初からやり直す。
失敗した単位はやり直さずに記録して、ほかの単位を続ける（S-chain・S-entropy で止まった単位もここに入る）。
腕: A4 = α 1e−4・サンプルの目標、A3 = α 1e−3・サンプルの目標、S2 = α 0.01・真の分布の目標（§4.5）。
π は --pi_dir（pi_precompute.py の出力）から読み、無い課題は走の中で計算する（provenance の detlabel.pi_source_counts に残る）。
使い方:
  python analysis/chargpt_detlabel_1023/W/drive_main.py --arms A4 S2 A3 --seeds 0-9 --slots 4 \
      --pi_dir results/chargpt_detlabel_1023/pi_cache [--dry]
"""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = "/home/kubo/project/Nakatsuka/lop_analysis_chargpt_cl_1019/.venv/bin/python"
ARMS = {"A4": ("1e-4", "sample"), "A3": ("1e-3", "sample"), "S2": ("0.01", "soft")}
RES = REPO / "results/chargpt_detlabel_1023/M2"


def done(out):
    pv = out / "provenance.json"
    try:
        return json.loads(pv.read_text()).get("detlabel_complete") is True
    except Exception:
        return False


def seeds_arg(xs):
    out = []
    for x in xs:
        if "-" in x:
            a, b = x.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(x))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", nargs="+", default=["A4", "S2", "A3"], choices=list(ARMS))
    ap.add_argument("--seeds", nargs="+", default=["0-9"])
    ap.add_argument("--slots", type=int, default=4)
    ap.add_argument("--pi_dir", default=str(REPO / "results/chargpt_detlabel_1023/pi_cache"))
    ap.add_argument("--order", choices=["seed", "arm"], default="seed", help="seed: seed ごとに腕を並べる（A4 の seed 0、S2 の seed 0、…）")
    ap.add_argument("--entropy_check", choices=["stop", "record"], default="stop", help="S-entropy で止めるか（主の決定が要る）")
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args()
    seeds = seeds_arg(a.seeds)
    units = [(arm, s) for s in seeds for arm in a.arms] if a.order == "seed" else [(arm, s) for arm in a.arms for s in seeds]
    todo = [(arm, s) for arm, s in units if not done(RES / arm / f"seed{s}")]
    logd = REPO / "results/chargpt_detlabel_1023/logs"
    logd.mkdir(parents=True, exist_ok=True)
    print(f"units {len(units)}  done {len(units) - len(todo)}  todo {len(todo)}", flush=True)
    env = dict(os.environ, CHARGPT_NLAYER="2", CHARGPT_DECOMP_CKPT="1", DETLABEL_GPU_CONCURRENCY=str(a.slots))
    running, results = {}, []
    t0 = time.time()
    while todo or running:
        while todo and len(running) < a.slots:
            arm, s = todo.pop(0)
            alpha, target = ARMS[arm]
            out = RES / arm / f"seed{s}"
            cmd = [PY, "src/chargpt_detlabel_1023.py", "run", "--box", "M", "--arm", "C3", "--seed", str(s), "--tasks", "100",
                   "--n_per_task", "10000", "--alpha", alpha, "--target", target, "--fresh_tasks", "10", "25", "50", "100",
                   "--pi_dir", a.pi_dir, "--entropy_check", a.entropy_check, "--out", str(out)]
            print(f"[{time.time() - t0:7.0f}s] start {arm} seed{s}: {' '.join(cmd)}", flush=True)
            if a.dry:
                continue
            log = open(logd / f"{arm}_seed{s}.log", "a")
            running[(arm, s)] = (subprocess.Popen(cmd, cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT), time.time())
        if a.dry:
            break
        time.sleep(10)
        for key, (p, ts) in list(running.items()):
            rc = p.poll()
            if rc is None:
                continue
            ok = rc == 0 and done(RES / key[0] / f"seed{key[1]}")
            results.append(dict(arm=key[0], seed=key[1], rc=rc, complete=ok, sec=time.time() - ts))
            print(f"[{time.time() - t0:7.0f}s] end {key[0]} seed{key[1]} rc {rc} complete {ok} {time.time() - ts:.0f}s", flush=True)
            del running[key]
    (logd / f"drive_{int(t0)}.json").write_text(json.dumps(dict(argv=sys.argv, results=results, wall=time.time() - t0), indent=1))


if __name__ == "__main__":
    main()
