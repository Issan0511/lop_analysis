#!/usr/bin/env python3
"""chargpt_cl_1019 のデータ列の機種間照合（S-portable、追補 1・2）。

  make  : 基準機で期待値を作る  python src/chargpt_cl_1019_streams.py make --out results/chargpt_cl_1019/expected_streams.json
  verify: 別マシンで照合する    python src/chargpt_cl_1019_streams.py verify --expected results/chargpt_cl_1019/expected_streams.json --out <dir>/streams_verify.json

照合するもの（すべて CPU で作るので GPU の機種によらないはず）:
  箱 M の課題 1〜3 の遷移表・batch 列・probe・評価用系列の sha256 と H_t、一様乱数の probe、4 腕の初期値の sha256。
"""
import argparse
import json
import socket
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chargpt_cl_1019 as C  # noqa: E402

ALPHA, N_M, TASKS = 0.01, 10000, (1, 2, 3)


def compute(seeds):
    out = {}
    for s in seeds:
        rec = dict(tasks={}, init={})
        for t in TASKS:
            c = C.task_cpu_M(s, t, N_M, ALPHA)
            rec["tasks"][str(t)] = dict(rule_sha=c["rule_sha"], data_sha=c["data_sha"], probe_sha=c["probe_sha"],
                                        eval_sha=C.tensor_sha(c["eval_seq"]), H=round(c["chain"].H, 12))
        rec["uniform_probe_sha"] = C.tensor_sha(C.uniform_probe_M(s, "cpu"))
        for arm in C.ARMS:
            rec["init"][arm] = C.state_sha(C.build_model(arm, "M", s, "cpu"))
        out[str(s)] = rec
        print(f"seed {s} done", flush=True)
    return out


def env():
    return dict(host=socket.gethostname(), torch=torch.__version__, numpy=np.__version__,
                python=sys.version.split()[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["make", "verify"])
    ap.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    ap.add_argument("--out", required=True)
    ap.add_argument("--expected", default=None)
    a = ap.parse_args()
    C.setup_torch()
    got = compute(a.seeds)
    if a.mode == "make":
        Path(a.out).write_text(json.dumps(dict(env=env(), alpha=ALPHA, n_per_task=N_M, streams=got), indent=1))
        print("written", a.out)
        return
    exp = json.loads(Path(a.expected).read_text())
    diffs = []
    for s, rec in got.items():
        e = exp["streams"][s]
        for t, r in rec["tasks"].items():
            for k, v in r.items():
                if e["tasks"][t][k] != v:
                    diffs.append(f"seed{s}/task{t}/{k}")
        if e["uniform_probe_sha"] != rec["uniform_probe_sha"]:
            diffs.append(f"seed{s}/uniform_probe")
        for arm, v in rec["init"].items():
            if e["init"][arm] != v:
                diffs.append(f"seed{s}/init/{arm}")
    res = dict(env=env(), expected_env=exp["env"], seeds=a.seeds, ok=not diffs, differing=diffs)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1))
    print("S-portable", "PASS" if not diffs else f"FAIL {diffs[:10]}")
    sys.exit(0 if not diffs else 1)


if __name__ == "__main__":
    main()
