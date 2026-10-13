#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""v1_redraw.py — 保存状態 θ(T)（課題 T 末 = end0[T-1]、M1 の約束）と replay の Adam 状態から、
新ラベル（とバッチ順）を引き直して課題 T+1 を R 回走らせ、課題末の持ち上げ役の数 N_i(T+1) の
「ラベル引き直しの測度」での分布を直接測る。D1 の縛り (4.5)(4.6) はこの測度の文なので、その測度で検算する。

- 訓練ループは cumulative_push_1013/replay_adam.py（= run_slope.py）を字句どおりに写した（Act・loss・Adam・バッチ順の作り方）。
- 自己検査: 本来のラベル（ref['y'][T]）と本来のバッチ順（gbatch を T 課題分進める）で走らせ、ref end[T] と bit 一致するか。
- 出力（obsidian-research-data/cobirth_1013/v1/redraw_<arm>_T<T>.npz）: isb0（出発状態の橋）、isb[R,H,H]、N[R,H]、B[R,H]（出発状態に無かった対の数）、
  k[R,H]、alive[R,H]、Z_end[R,N,H] float32（課題末の前活性。累積 Δz の unit 間相関の検算用）、self_test の結果。
使い方: v1_redraw.py --arm a0.3 --T 17 --R 240 --seed 7
"""
import os
for _k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_k] = "1"
import sys, json, math, time, argparse, hashlib, gzip
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import v1_lib as L  # noqa: E402

REF_ROOT = Path("/home/issan/Projects/obsidian-research-data/push_direction_proof_0930")
REPLAY = Path("/home/issan/Projects/obsidian-research-data/cumulative_push_1013/replay")
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
OUT = Path("/home/issan/Projects/obsidian-research-data/cobirth_1013/v1")
N, BATCH, H, K, STEPS, LR = 1200, 16, 100, 10, 4000, 1e-3


def stream(role, seed):
    h = hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))


def read_idx(p):
    b = gzip.open(p, "rb").read()
    dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)


class Act(torch.autograd.Function):
    @staticmethod
    def forward(ctx, z, a):
        out = torch.where(z > 0, z, z * a)
        d = torch.where(z > 0, 1.0, a)
        ctx.save_for_backward(d)
        return out

    @staticmethod
    def backward(ctx, gout):
        d, = ctx.saved_tensors
        return gout * d, None


def bits_equal(x, r):
    x, r = np.ascontiguousarray(x), np.ascontiguousarray(r)
    return x.shape == r.shape and x.dtype == r.dtype == np.float32 and np.array_equal(x.view(np.uint32), r.view(np.uint32))


def make_optim(params, rep):
    optim = torch.optim.Adam(params, lr=LR, betas=(0.9, 0.999), eps=1e-8)
    sd = optim.state_dict()
    step = float(rep["adam_step"])
    sd["state"] = {j: {"step": torch.tensor(step), "exp_avg": torch.tensor(rep[f"adam_m_{j}"]).clone(),
                       "exp_avg_sq": torch.tensor(rep[f"adam_v_{j}"]).clone()} for j in range(4)}
    optim.load_state_dict(sd)
    return optim


def run_task(theta0, rep, X, Y, order, slope):
    P = [torch.tensor(theta0[j]).clone().requires_grad_(True) for j in range(4)]
    optim = make_optim(P, rep)
    for it in range(STEPS):
        idx = order[it]; xb, yb = X[idx], Y[idx]
        z = xb @ P[0].T + P[1]; h = Act.apply(z, slope); logits = h @ P[2].T + P[3]
        loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
        optim.zero_grad(set_to_none=True); loss.backward()
        optim.step()
    return [q.detach().numpy().copy() for q in P]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="a0.3"); ap.add_argument("--T", type=int, default=17)
    ap.add_argument("--R", type=int, default=240); ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-selftest", action="store_true")
    args = ap.parse_args()
    arm, T, R = args.arm, args.T, args.R
    d, slope = L.load_ref(arm)
    rep = np.load(REPLAY / arm / f"task_{T:03d}.npz", allow_pickle=True)
    assert int(rep["task"]) == T and float(rep["a"]) == slope and bool(rep["bit_ok"])
    OUT.mkdir(parents=True, exist_ok=True)
    logf = open(OUT / f"redraw_{arm}_T{T}.log", "a")

    def log(s):
        print(s, flush=True); logf.write(s + "\n"); logf.flush()

    ax = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = d["subset"]
    X = torch.tensor(ax[subset], dtype=torch.float32)
    X64 = ax[subset].astype(np.float64)
    theta0 = [d[f"end{j}"][T - 1].copy() for j in range(4)]       # float32、課題 T 末
    log(f"arm {arm} a {slope} start state T={T} (end0[{T-1}]), adam_step {int(rep['adam_step'])}, R {R}, seed {args.seed}")

    # ---- self-test: 本来のラベルとバッチ順で課題 T+1 を走らせ ref end[T] と bit 比較 ----
    st_res = {}
    if not args.no_selftest:
        glabel, gbatch = stream("env_labels_0913", 0), stream("env_batch_0913", 0)
        epochs = -(-(STEPS * BATCH) // N)
        for _ in range(T):
            torch.randint(K, (N,), generator=glabel)
            for _ in range(epochs):
                torch.randperm(N, generator=gbatch)
        y_new = torch.randint(K, (N,), generator=glabel)
        y_ok = bool(np.array_equal(y_new.numpy(), d["y"][T]))
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:STEPS * BATCH].reshape(STEPS, BATCH)
        Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32)
        t0 = time.monotonic()
        e = run_task(theta0, rep, X, Y, order, slope)
        per = [bits_equal(e[j], d[f"end{j}"][T]) for j in range(4)]
        nd = [int((e[j].view(np.uint32) != d[f"end{j}"][T].view(np.uint32)).sum()) for j in range(4)]
        maxrel = [float(np.abs(e[j].astype(np.float64) - d[f"end{j}"][T]).max() / (np.abs(d[f"end{j}"][T]).max() + 1e-30)) for j in range(4)]
        st_res = dict(labels_match=y_ok, bits_equal=per, n_diff_elems=nd, max_rel_diff=maxrel, task_sec=time.monotonic() - t0)
        log(f"[self-test] labels match {y_ok}; end{T} bit-equal per param {per}; differing elements {nd}; max rel diff {['%.1e' % m for m in maxrel]}; {st_res['task_sec']:.1f}s")

    # ---- 出発状態の橋 ----
    st0 = L.state_from_params(arm, slope, X64, theta0[0], theta0[1], theta0[2], theta0[3])
    isb0, _, _, _ = L.bridges(st0)
    N0 = isb0.sum(1)
    log(f"start state: alive {int(st0.alive.sum())}, pairs {int(isb0.sum())}, max N_i {int(N0.max())}")

    # ---- 引き直し ----
    isb = np.zeros((R, H, H), bool); Nn = np.zeros((R, H), np.int16); Bn = np.zeros((R, H), np.int16)
    kk = np.zeros((R, H), np.int16); al = np.zeros((R, H), bool); Zend = np.zeros((R, N, H), np.float32)
    acc_end = np.zeros(R); seeds = np.zeros(R, np.int64)
    t_all = time.monotonic()
    for r in range(R):
        s = int.from_bytes(hashlib.sha256(f"cobirth_v1_redraw|{arm}|{T}|{args.seed}|{r}".encode()).digest()[:8], "little") & ((1 << 63) - 1)
        seeds[r] = s
        g = torch.Generator().manual_seed(s)
        y_new = torch.randint(K, (N,), generator=g)
        epochs = -(-(STEPS * BATCH) // N)
        order = torch.stack([torch.randperm(N, generator=g) for _ in range(epochs)]).reshape(-1)[:STEPS * BATCH].reshape(STEPS, BATCH)
        Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32)
        e = run_task(theta0, rep, X, Y, order, slope)
        st = L.state_from_params(arm, slope, X64, e[0], e[1], e[2], e[3])
        b, _, _, _ = L.bridges(st)
        isb[r] = b; Nn[r] = b.sum(1); Bn[r] = (b & ~isb0).sum(1); kk[r] = st.k; al[r] = st.alive
        Zend[r] = st.Z.astype(np.float32)
        logits = np.where(st.Z > 0, st.Z, slope * st.Z) @ st.W2.T + st.b3
        acc_end[r] = float((logits.argmax(1) == y_new.numpy()).mean())
        if r % 10 == 0 or r == R - 1:
            log(f"redraw {r+1}/{R}: max N {int(Nn[r].max())}, pairs {int(b.sum())}, births {int(Bn[r].sum())}, acc {acc_end[r]:.3f}, elapsed {time.monotonic()-t_all:.0f}s")
    out = OUT / f"redraw_{arm}_T{T}.npz"
    np.savez_compressed(out, isb0=isb0, isb=isb, N=Nn, B=Bn, k=kk, alive=al, Z_end=Zend, Z0=st0.Z.astype(np.float32),
                        acc_end=acc_end, seeds=seeds, T=np.int64(T), arm=arm, a=np.float64(slope), R=np.int64(R),
                        selftest=json.dumps(st_res), note="M1 convention: start = end0[T-1] (task T end), redraw task T+1 with fresh labels AND fresh batch order; Adam state from cumulative_push replay task_T")
    log(f"wrote {out} ({time.monotonic()-t_all:.0f}s)")


if __name__ == "__main__":
    main()
