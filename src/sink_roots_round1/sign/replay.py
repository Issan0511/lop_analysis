"""保存状態から、切替後の実際の 200 更新（同じバッチ順・同じ新ラベル・Adam 状態を継続・float32）を再生し、
更新するパラメタを限った腕で第 2 層（最終隠れ層）の m の変位を分ける。訓練の走なので、門（CPU を使う python ≤ 8）を通してから起動する。
腕: full（全部・実変位 dm との一致を確認）、own（最終隠れ層の W,b だけ）、up（それより上流の層だけ）、
    head（読み出しだけ固定: 全部から出力層を除く）、sgd（全部・SGD・lr は Adam の 1 歩目の平均の大きさに合わせる）
使い方: python3 replay.py <state.npz> ...
"""
import sys, math, hashlib, gzip, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
sys.path.insert(0, str(Path(__file__).parent))
from common_f import activ

DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
N, BATCH = 1200, 16
def stream(role, seed):
    h = hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))
def read_idx(p):
    b = gzip.open(p, "rb").read()
    dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)
AX = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255

@torch.no_grad()
def means(P, X, act, L):
    Xd = X.double(); out = []; a = Xd
    for l in range(L):
        z = a @ P[2 * l].double().T + P[2 * l + 1].double(); out.append(z.mean(0)); a = activ(z, act)
    return out

def replay(path, arm, steps=200):
    d = np.load(path, allow_pickle=True)
    L, K, act = int(d["L"]), int(d["K"]), str(d["act"])
    m = __import__("re").match(r".*_s(\d)_t(\d+)", Path(path).stem); seed, task = int(m.group(1)), int(m.group(2))
    X = torch.tensor(AX[d["subset"]])
    gbatch = stream("env_batch_0913", seed); glabel = stream("env_labels_0913", seed)
    tsteps = 4000 if L == 1 else 6000
    epochs = -(-(tsteps * BATCH) // N)
    for t in range(1, task + 1):
        y = torch.randint(K, (N,), generator=glabel)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:tsteps * BATCH].reshape(tsteps, BATCH)
    assert (y.numpy() == d["y_new"]).all(), "新ラベルが再現できない"
    Y = torch.nn.functional.one_hot(y, K).float()
    P = [torch.tensor(d[f"p{i}"]).float().requires_grad_(True) for i in range(2 * L + 2)]
    li = 2 * (L - 1)
    train_idx = {"full": list(range(2 * L + 2)), "own": [li, li + 1], "up": list(range(li)),
                 "head": list(range(2 * L)), "sgd": list(range(2 * L + 2))}[arm]
    params = [P[i] for i in train_idx]
    if arm == "sgd" or not params:
        opt = None
    else:
        opt = torch.optim.Adam(params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8)
        for i in train_idx:
            opt.state[P[i]] = {"step": torch.tensor(float(d["step"])), "exp_avg": torch.tensor(d[f"m{i}"]).float().clone(),
                               "exp_avg_sq": torch.tensor(d[f"v{i}"]).float().clone()}
    m0 = means(P, X, act, L); rec = {}
    if not params:                                  # sink_roots_0930: 'up' has no layer to train in a 1-layer net
        z0 = [np.zeros(x.shape) for x in m0]
        return {50: z0, 200: z0}, d["dm"], L
    lr_sgd = None
    for it in range(steps):
        idx = order[it]; a = X[idx]
        for l in range(L):
            a = activ(a @ P[2 * l].T + P[2 * l + 1], act)
        logits = a @ P[2 * L].T + P[2 * L + 1]
        loss = (logits.logsumexp(-1) - (logits * Y[idx]).sum(-1)).mean()
        gs = torch.autograd.grad(loss, params)
        with torch.no_grad():
            if opt is None:
                if lr_sgd is None:                      # 1 歩目の Adam の歩幅（|Δθ| の平均）に SGD の歩幅の平均を合わせる
                    lr_sgd = [1e-3 / (g.abs().mean() + 1e-30) for g in gs]
                for q, g, lr in zip(params, gs, lr_sgd): q -= lr * g
            else:
                for q, g in zip(params, gs): q.grad = g
                opt.step()
        if it + 1 in (50, 200):
            m1 = means(P, X, act, L); rec[it + 1] = [(b - a0).numpy() for a0, b in zip(m0, m1)]
    return rec, d["dm"], L

import os
OUTDIR = Path(os.environ.get("REPLAY_OUT", str(Path(__file__).parent / "out")))  # sink_roots_0930: outputs to the raw-data dir

if __name__ == "__main__":
    OUTDIR.mkdir(parents=True, exist_ok=True)
    t0 = time.monotonic()
    for f in sys.argv[1:]:
        res = {}
        for arm in ("full", "own", "up", "head", "sgd"):
            rec, dm, L = replay(f, arm); res[arm] = rec
        real = {50: dm[0, L - 1], 200: dm[1, L - 1]}
        z0 = None
        line = f"{Path(f).stem:22s}"
        for T in (50, 200):
            fu = res["full"][T][L - 1]
            line += f" | T{T} real dn {np.mean(real[T] < 0):.2f} fid {np.max(np.abs(fu - real[T])) / (np.max(np.abs(real[T])) + 1e-30):.1e}"
            for arm in ("own", "up", "head", "sgd"):
                x = res[arm][T][L - 1]
                line += f" {arm} dn {np.mean(x < 0):.2f} ag {np.mean(np.sign(x) == np.sign(real[T])):.2f}"
            if L == 2:
                inter = fu - res["own"][T][L - 1] - res["up"][T][L - 1]
                line += f" inter<0 {np.mean(inter < 0):.2f} med own/up/inter {np.median(res['own'][T][L-1]):+.2f}/{np.median(res['up'][T][L-1]):+.2f}/{np.median(inter):+.2f}"
        print(line, flush=True)
        np.save(OUTDIR / f"replay_{Path(f).stem}.npy", np.array([res, real], dtype=object), allow_pickle=True)
    print(f"({time.monotonic() - t0:.0f}s)")
