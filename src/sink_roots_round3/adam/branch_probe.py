"""同じ切替状態から新課題 1 つを腕ごとに回す（1 層・箱は adam_window_probe.py と同じ・本走の状態とビット一致）。
腕: base（b2 0.999）/ b2_0.9 / b2_0.99 / b2_0.9999 / v10 / v01（切替時に v を ×10・×0.1）/ sgd_0.01 / sgd_0.1 / long（b2 0.999・16,000 更新）
記録: 10 更新ごとの m_i = mean_n z_in の変位（切替時からの差）、unit ごとの切替時の k。
"""
import argparse, math, time
from pathlib import Path
import numpy as np, torch
from adam_window_probe import (read_idx, stream, initial, activ, Adam, train_step, DATA, N, BATCH, H, K)
torch.set_num_threads(1)

ARMS = ("base", "b2_0.9", "b2_0.99", "b2_0.9999", "v10", "v01", "sgd_0.01", "sgd_0.1", "long")

def m_of(P, X):
    with torch.no_grad():
        return P[0].double() @ X.double().mean(0) + P[1].double()   # m_i = w_i.mu + b_i（線形なので全入力の平均と同じ）

def run_arm(arm, P0, opt0, X, Y, order_long, act, steps):
    P = [q.detach().clone().requires_grad_(True) for q in P0]
    T = 16000 if arm == "long" else steps
    if arm.startswith("sgd"):
        lr = float(arm.split("_")[1]); opt = None
    else:
        b2 = float(arm.split("_")[1]) if arm.startswith("b2_") else 0.999
        opt = Adam(P, opt0.lr, 0.9, b2, 1e-8)
        opt.m = [m.clone() for m in opt0.m]; opt.v = [v.clone() for v in opt0.v]; opt.t = opt0.t
        if arm == "v10": opt.v = [v * 10 for v in opt.v]
        if arm == "v01": opt.v = [v * 0.1 for v in opt.v]
    m0 = m_of(P, X); traj = []
    for s in range(1, T + 1):
        idx = order_long[s - 1]
        train_step(P, opt if opt else None, X[idx], Y[idx], act)
        if opt: opt.step()
        else:
            with torch.no_grad():
                for q in P: q.add_(q.grad, alpha=-lr)
        if s % 10 == 0: traj.append((m_of(P, X) - m0).numpy())
    return np.array(traj)

def main(a):
    torch.manual_seed(a.seed)
    ax = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=stream("rl_subset", a.seed))[:N].numpy()
    X = torch.tensor(ax[subset], dtype=torch.float32)
    glabel, gbatch = stream("env_labels_0913", a.seed), stream("env_batch_0913", a.seed)
    P = [q.to(dtype=torch.float32).requires_grad_(True) for q in initial(a.seed, (784, H, K))]
    opt = Adam(P, 1e-3, 0.9, 0.999, 1e-8)
    switches = set(int(x) for x in a.switches.split(",")); arms = a.arms.split(",")
    out = {}; t0 = time.monotonic()
    for task in range(1, max(switches) + 1):
        y_new = torch.randint(K, (N,), generator=glabel)
        Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32)
        epochs = -(-(a.steps * BATCH) // N)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:a.steps * BATCH].reshape(a.steps, BATCH)
        if task in switches:
            # 長い腕のバッチ順: 本走の 4,000 更新の後ろに、別の生成器の順を足す
            ge = torch.Generator().manual_seed(1000 + task)
            extra = torch.stack([torch.randperm(N, generator=ge) for _ in range(-(-(12000 * BATCH) // N))]).reshape(-1)[:12000 * BATCH].reshape(12000, BATCH)
            order_long = torch.cat([order, extra])
            with torch.no_grad():
                z = X @ P[0].T + P[1]
            out[f"t{task}_k"] = (z > 0).sum(0).numpy()
            for arm in arms:
                out[f"t{task}_{arm}"] = run_arm(arm, P, opt, X, Y, order_long, a.act, a.steps).astype(np.float32)
                tr = out[f"t{task}_{arm}"]; live = out[f"t{task}_k"] > 0
                push = np.median(tr[19, live]); end = np.median(tr[399, live])
                print(f"{a.act} 切替 {task} {arm:10s}: 押し {push:+.3f}  課題末(4000) {end:+.3f}  比 {-(end - push) / push:.3f}" + (f"  16000 {np.median(tr[-1, live]):+.3f}" if arm == "long" else "") + f" ({time.monotonic() - t0:.0f}s)", flush=True)
        if task == max(switches): break
        for it in range(a.steps):
            idx = order[it]; train_step(P, opt, X[idx], Y[idx], a.act); opt.step()
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / f"branch_{a.act}_s{a.seed}_sw{a.switches.replace(',', '-')}.npz", **out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="ELU"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--switches", default="3,10")
    ap.add_argument("--arms", default=",".join(ARMS)); ap.add_argument("--out", default="branches")
    main(ap.parse_args())
