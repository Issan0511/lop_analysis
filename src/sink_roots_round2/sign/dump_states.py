"""切替時点の状態を丸ごと保存する短い走（検算用）。
箱は drive_recon_0930/align_probe_ref.py（1 層）・align_probe3.py（2 層）と同じ:
生 MNIST/255・subset 1,200（pmnist_0905 の hash）・init U(±1/sqrt(din))・Adam 1e-3（状態は課題をまたぐ）・batch 16・
1 層 4,000 更新/課題、2 層 6,000 更新/課題・float32・CPU 1 スレッド・ELU は expm1 の自動微分。
保存: 切替 t（課題 t が始まる直前 = 課題 t-1 に当てはめた状態）ごとに
  params（float32）、Adam の exp_avg / exp_avg_sq / step、y_old、y_new、
  課題 t の 50/200/1000/末 更新後の m（各隠れ層の入力平均の前活性）の変位（実変位）。
活性化: ELU・LR（leaky 0.1）・R（ReLU）・GELU・SILU・SNK（Snake α=1 固定: z + sin^2(z)）。
"""
import argparse, gzip, hashlib, math, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
N, BATCH, H = 1200, 16, 100
KS = (50, 200, 1000)
SQRT2 = math.sqrt(2.0)

def stream(role, seed):
    h = hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))
def read_idx(p):
    b = gzip.open(p, "rb").read()
    dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)
def activ(z, act):
    if act == "LR":   return torch.where(z > 0, z, z * 0.1)
    if act == "ELU":  return torch.where(z > 0, z, torch.expm1(z.clamp_max(0)))
    if act == "R":    return torch.relu(z)
    if act == "GELU": return z * 0.5 * (1.0 + torch.erf(z / SQRT2))
    if act == "SILU": return z * torch.sigmoid(z)
    if act == "SNK":  return z + torch.sin(z) ** 2
    raise ValueError(act)
def initial(seed, dims):
    g = stream("init", seed); p = []
    for din, dout in zip(dims[:-1], dims[1:]):
        p += [(torch.rand(dout, din, generator=g) * 2 - 1) * (1.0 / math.sqrt(din)),
              (torch.rand(dout, generator=g) * 2 - 1) * (1.0 / math.sqrt(din))]
    return p
@torch.no_grad()
def means(P, X, act, L):
    Xd = X.double(); out = []; a = Xd
    for l in range(L):
        z = a @ P[2 * l].double().T + P[2 * l + 1].double(); out.append(z.mean(0)); a = activ(z, act)
    return out

def run(act, K, seed, tasks, steps, L, dump_at, out):
    torch.manual_seed(seed)
    ax = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=stream("rl_subset", seed))[:N].numpy()
    X = torch.tensor(ax[subset], dtype=torch.float32)
    glabel, gbatch = stream("env_labels_0913", seed), stream("env_batch_0913", seed)
    dims = (784,) + (H,) * L + (K,)
    P = [q.to(dtype=torch.float32).requires_grad_(True) for q in initial(seed, dims)]
    optim = torch.optim.Adam(P, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
    y_old = None; t0 = time.monotonic(); tfit = []; curves = []   # sink_roots_0930 round 2: fitting speed per task
    out.mkdir(parents=True, exist_ok=True)
    for task in range(1, tasks + 1):
        y_new = torch.randint(K, (N,), generator=glabel)
        Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32)
        epochs = -(-(steps * BATCH) // N)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:steps * BATCH].reshape(steps, BATCH)
        dump = (task in dump_at) and (y_old is not None)
        if dump:
            snap = dict(params=[q.detach().clone().numpy() for q in P],
                        exp_avg=[optim.state[q]["exp_avg"].clone().numpy() for q in P],
                        exp_avg_sq=[optim.state[q]["exp_avg_sq"].clone().numpy() for q in P],
                        step=float(optim.state[P[0]]["step"]))
            m0 = means(P, X, act, L)
        dm = []; curve = []
        for it in range(steps):
            idx = order[it]; a = X[idx]
            for l in range(L):
                a = activ(a @ P[2 * l].T + P[2 * l + 1], act)
            logits = a @ P[2 * L].T + P[2 * L + 1]
            loss = (logits.logsumexp(-1) - (logits * Y[idx]).sum(-1)).mean()
            optim.zero_grad(set_to_none=True); loss.backward(); optim.step()
            if (it + 1) % 25 == 0:                         # full-data accuracy on the task's labels (touches no RNG / params)
                with torch.no_grad():
                    a2 = X
                    for l in range(L):
                        a2 = activ(a2 @ P[2 * l].T + P[2 * l + 1], act)
                    curve.append(float(((a2 @ P[2 * L].T + P[2 * L + 1]).argmax(-1) == y_new).float().mean()))
                if len(tfit) < task and curve[-1] >= 0.9:
                    tfit.append(it + 1)
            if dump and ((it + 1) in KS or it + 1 == steps):
                m1 = means(P, X, act, L); dm.append(np.stack([(b - a0).numpy() for a0, b in zip(m0, m1)]))
        if dump:
            np.savez_compressed(out / f"L{L}_{act}_K{K}_s{seed}_t{task:03d}.npz",
                                **{f"p{i}": v for i, v in enumerate(snap["params"])},
                                **{f"m{i}": v for i, v in enumerate(snap["exp_avg"])},
                                **{f"v{i}": v for i, v in enumerate(snap["exp_avg_sq"])},
                                step=snap["step"], subset=subset, y_old=y_old.numpy(), y_new=y_new.numpy(),
                                dm=np.array(dm), ks=np.array(sorted({k for k in KS if k <= steps} | {steps})), L=L, K=K, act=act)
        with torch.no_grad():
            a = X.double()
            for l in range(L):
                z = a @ P[2 * l].double().T + P[2 * l + 1].double(); a = activ(z, act)
            lg = a @ P[2 * L].double().T + P[2 * L + 1].double()
            acc = float((lg.argmax(-1) == y_new).double().mean())
        print(f"L{L} {act} K{K} s{seed} task {task} acc {acc:.3f} pplus_last {float((z > 0).double().mean()):.4f} ({time.monotonic()-t0:.0f}s)", flush=True)
        if len(tfit) < task:
            tfit.append(-1)
        curves.append(curve)
        y_old = y_new
    np.savez_compressed(out / f"tfit_L{L}_{act}_K{K}_s{seed}.npz", tfit=np.array(tfit), steps=steps, acc_curve=np.array(curves), curve_every=25)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", required=True); ap.add_argument("--K", type=int, default=10); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=12); ap.add_argument("--L", type=int, default=1)
    ap.add_argument("--dump", default="2,3,5,8,12"); ap.add_argument("--out", default="states")
    ap.add_argument("--steps", type=int, default=None)   # sink_roots_0930 round 2
    a = ap.parse_args()
    steps = a.steps if a.steps is not None else (4000 if a.L == 1 else 6000)
    run(a.act, a.K, a.seed, a.tasks, steps, a.L, set(int(s) for s in a.dump.split(",")), Path(a.out))
