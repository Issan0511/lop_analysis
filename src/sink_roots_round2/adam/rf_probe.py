"""反証役の小さい走（1 層 RL-MNIST・導出役と同じ箱・seed 1）。導出役の adam_window_probe.py のコピー（awp_copy.py）の関数を使う。
切替 3・10 で、同じ状態からラベルだけ 32 回引き直して 200 更新（バッチ順は本走と同じ）。32 回を前半 16・後半 16 に分け、
  (a) 共分散の割合 (E[k^T D M] − k^T E[D] E[M]) / E[k^T D M] の割れ半分の一致（A4 の大きさが引き直しの雑音でないか）
  (b) A6 の予測（切替時の状態だけ）と、ラベル平均の実変位 E[累積] の比
  (c) 1 歩目: ラベル平均が下向きの割合と、旧モーメント（履歴）を除いた 1 歩目のラベル平均が下向きの割合（A4-1 と M1）
"""
import argparse, math, time, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).parent))
import awp_copy as W
torch.set_num_threads(1)
N, BATCH, H, K = W.N, W.BATCH, W.H, W.K

def redraw_halves(P, opt, X, order, act, kvec, R, steps, seed, task):
    saveP = [q.detach().clone() for q in P]; saveM = [m.clone() for m in opt.m]; saveV = [v.clone() for v in opt.v]; savet = opt.t
    b1, lr = opt.b1, opt.lr
    M0, _, _, _ = opt.layer1(); M0 = M0.clone()
    half = R // 2
    sumD = torch.zeros(2, steps, H, 785, dtype=torch.float32); sumM = torch.zeros(2, steps, H, 785, dtype=torch.float32)
    dm = np.zeros((R, steps, H)); dm_nohist1 = np.zeros((R, H)); bc1s = np.zeros(steps)
    gl = torch.Generator().manual_seed(int(__import__("hashlib").sha256(f"refute|{seed}|{task}".encode()).hexdigest()[:12], 16))
    for r in range(R):
        with torch.no_grad():
            for q, s0 in zip(P, saveP): q.copy_(s0)
            for m, s0 in zip(opt.m, saveM): m.copy_(s0)
            for v, s0 in zip(opt.v, saveV): v.copy_(s0)
        opt.t = savet
        y = torch.randint(K, (N,), generator=gl); Y = torch.nn.functional.one_hot(y, K).to(torch.float32)
        th = W.theta1(P); h = 0 if r < half else 1
        for s in range(1, steps + 1):
            W.train_step(P, opt, X[order[s - 1]], Y[order[s - 1]], act); opt.step()
            M, V, bc1, bc2 = opt.layer1(); D = 1.0 / ((V.sqrt() / math.sqrt(bc2)) + opt.eps)
            th_new = W.theta1(P); dm[r, s - 1] = ((th_new - th) @ kvec).numpy(); th = th_new
            if s == 1: dm_nohist1[r] = (-(lr / bc1) * (((M - b1 * M0) * D) @ kvec)).numpy()
            sumD[h, s - 1] += D.float(); sumM[h, s - 1] += M.float(); bc1s[s - 1] = bc1
    with torch.no_grad():
        for q, s0 in zip(P, saveP): q.copy_(s0)
        for m, s0 in zip(opt.m, saveM): m.copy_(s0)
        for v, s0 in zip(opt.v, saveV): v.copy_(s0)
    opt.t = savet
    out = dict(dm=dm, dm_nohist1=dm_nohist1)
    for h in (0, 1):
        ED = sumD[h].double() / half; EM = sumM[h].double() / half
        out[f"kEDEM_h{h}"] = np.stack([(-(lr / bc1s[s]) * ((ED[s] * EM[s]) @ kvec)).numpy() for s in range(steps)])
    return out

def run(a):
    torch.manual_seed(a.seed)
    ax = W.read_idx(W.DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=W.stream("rl_subset", a.seed))[:N].numpy()
    X = torch.tensor(ax[subset], dtype=torch.float32)
    Xt = torch.cat([X.double(), torch.ones(N, 1, dtype=torch.float64)], 1)
    kvec = torch.cat([X.double().mean(0), torch.ones(1, dtype=torch.float64)])
    glabel, gbatch = W.stream("env_labels_0913", a.seed), W.stream("env_batch_0913", a.seed)
    P = [q.to(dtype=torch.float32).requires_grad_(True) for q in W.initial(a.seed, (784, H, K))]
    opt = W.Adam(P, 1e-3, 0.9, 0.999, 1e-8)
    probes = set(int(x) for x in a.probe.split(",")); out = {}; t0 = time.monotonic()
    for task in range(1, max(probes) + 1):
        y_new = torch.randint(K, (N,), generator=glabel); Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32)
        epochs = -(-(a.steps * BATCH) // N)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:a.steps * BATCH].reshape(a.steps, BATCH)
        if task in probes:
            st = W.switch_state(P, opt, X, Xt, kvec, a.act)
            for kk in ("S", "SD", "gbar", "sig2", "k"): out[f"t{task}_{kk}"] = st[kk].float().numpy()
            _, V, _, bc2 = opt.layer1(); out[f"t{task}_V0"] = (V / bc2).float().numpy()
            M0, _, _, _ = opt.layer1(); D0 = opt.D1()
            out[f"t{task}_hist"] = np.stack([(-(opt.lr / (1 - opt.b1 ** (opt.t + s))) * (((opt.b1 ** s) * M0 * D0) @ kvec)).numpy() for s in range(1, 201)])  # 履歴の近似（分母を D0 に固定）
            rd = redraw_halves(P, opt, X, order, a.act, kvec, a.R, 200, a.seed, task)
            for kk, vv in rd.items(): out[f"t{task}_{kk}"] = vv
            print(f"probe {task} done ({time.monotonic() - t0:.0f}s)", flush=True)
        for it in range(a.steps):
            idx = order[it]; W.train_step(P, opt, X[idx], Y[idx], a.act); opt.step()
        print(f"{a.act} s{a.seed} task {task} ({time.monotonic() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / f"rf_{a.act}_s{a.seed}.npz", **out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="ELU"); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--probe", default="3,10"); ap.add_argument("--R", type=int, default=32)
    ap.add_argument("--out", default="rf_runs")
    run(ap.parse_args())
