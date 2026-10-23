#!/usr/bin/env python3
"""chargpt_detlabel_1023 §10.3 (i): 箱 M の連鎖の定常分布 π を先に計算して置く（学習なし、CPU）。

学習コード src/chargpt_cl_1019.py（blob cc524f85、書き換えない）の ChainM をそのまま呼び、その π を置く。
走（src/chargpt_detlabel_1023.py）は rule_sha で π を引くので、同じ関数の値がそのまま使われる（S-pi で照合）。
1 連鎖 1 file: <out>/a<α>/seed<k>/t<t>.npz
  pi (65,65 float64)・pi_iters・rule_sha・pi_sha・H・sec と、§8 の S-chain・S-entropy の下見
  （行の和の誤差・NaN・最後の 1 歩の L1 の変化 d・B = 64·d·log 65・eval 系列での真の CE と標準誤差）。
  S-chain の「P = 0 の遷移が引かれた数」は課題の全データが要るので、走の中（裏のスレッド）でだけ数える。
既にある file は飛ばす（再開できる）。書き込みは .part に書いてから名前を替える。

使い方:
  python -I pi_precompute.py --out results/chargpt_detlabel_1023/pi_cache --alphas 1e-4 1e-3 0.01 \
      --seeds 0 1 2 3 4 5 6 7 8 9 --tasks 1-100 --procs 8 [--order task]
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path

for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[k] = "1"

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(1)
import chargpt_cl_1019 as C  # noqa: E402

V = C.V


def tag(alpha):
    return f"a{alpha:g}"


def one(job):
    out, alpha, seed, task = job
    f = Path(out) / tag(alpha) / f"seed{seed}" / f"t{task}.npz"
    if f.exists():
        return dict(alpha=alpha, seed=seed, task=task, skipped=True)
    f.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ch = C.ChainM(seed, task, alpha)
    sec = time.time() - t0
    P, pi = ch.P, ch.pi
    new = 0.5 * pi + 0.5 * np.einsum("ab,abc->bc", pi, P)
    d = float(np.abs(new - pi).sum())
    _, ev = C.probes_M(ch, seed, task)
    x = ev.numpy().astype(np.int64)
    cx, tg = x[:, :-2] * V + x[:, 1:-1], x[:, 2:]
    pe = ch.P.reshape(V * V, V)[cx, tg]
    lp = -np.log(np.maximum(pe, 1e-300))
    ce, se = float(lp.mean()), float(lp.std(ddof=1) / math.sqrt(lp.size))
    rec = dict(alpha=alpha, seed=seed, task=task, pi_iters=int(ch.pi_iters), H=ch.H, sec=sec, d=d, B=64 * d * math.log(65),
               rowsum_err=ch.rowsum_err, nan=bool(np.isnan(P).any() or np.isnan(pi).any()), zero_drawn_eval=int((pe == 0).sum()),
               true_ce_eval=ce, true_ce_eval_se=se, ok_entropy=bool(abs(ce - ch.H) <= 4 * se),
               rule_sha=hashlib.sha256(P.tobytes()).hexdigest(), pi_sha=hashlib.sha256(pi.tobytes()).hexdigest())
    rec["ok_chain_part"] = bool(rec["rowsum_err"] <= 1e-12 and not rec["nan"] and rec["B"] <= 2.5e-4 and rec["zero_drawn_eval"] == 0)
    part = f.with_name(f".{f.stem}.{os.getpid()}.part")
    with open(part, "wb") as fh:
        np.savez(fh, pi=pi, **{k: np.array(v) for k, v in rec.items()})
    os.replace(part, f)
    rec["pid"] = os.getpid()
    rec["host_load"] = os.getloadavg()[0]
    return rec


def parse_tasks(s):
    out = []
    for p in s:
        if "-" in p:
            a, b = p.split("-")
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(p))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--tasks", nargs="+", required=True)
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--order", choices=["task", "seed"], default="task",
                    help="task: 課題の番号の順（走が課題 1 から読むので、どの seed も先頭から揃う）")
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    tasks = parse_tasks(a.tasks)
    if a.order == "task":
        jobs = [(a.out, al, s, t) for t in tasks for al in a.alphas for s in a.seeds]
    else:
        jobs = [(a.out, al, s, t) for al in a.alphas for s in a.seeds for t in tasks]
    log = open(a.log or Path(a.out) / f"precompute_{os.getpid()}.jsonl", "a")
    Path(a.out).mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    import multiprocessing as mp
    with mp.get_context("fork").Pool(a.procs, maxtasksperchild=20) as pool:
        for i, r in enumerate(pool.imap_unordered(one, jobs, chunksize=1)):
            r["wall"] = time.time() - t0
            log.write(json.dumps(r) + "\n")
            log.flush()
            if not r.get("skipped"):
                print(f"[{i + 1}/{len(jobs)}] a{r['alpha']:g} s{r['seed']} t{r['task']} iters {r['pi_iters']} {r['sec']:.1f}s "
                      f"B {r['B']:.2e} ok_entropy {r['ok_entropy']}", flush=True)
    print(f"done {len(jobs)} jobs {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
