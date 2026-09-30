"""Re-run of the Codex configuration (derive/lemma_checks.py section 3; its .txt shows only 6 of the 8 claimed settings, one crossing at step 280).
One GELU/SiLU unit, v = (1,-1), output bias (6,-6) (old labels all class 1), new labels uniform in {1,2}, N 1,200 real MNIST/255, float64.
All z start below z_c (top = 2 z_c or z_c - 3).  Optimizers: SGD 0.1 (the derivation's), SGD 0.01, full-batch GD 0.01 (near gradient flow),
Adam 1e-3 (box), Adam 1e-4.  Batch 16 except GD.  2 seeds (init of w, labels, batch order).  top checked every 5 steps."""
import math, gzip, sys, numpy as np, torch
torch.set_num_threads(1); torch.set_default_dtype(torch.float64)
SQ2 = math.sqrt(2)
gelu = lambda z: z * 0.5 * (1 + torch.erf(z / SQ2)); silu = lambda z: z * torch.sigmoid(z)
ACT = {"GELU": (gelu, -0.7517915246935645), "SiLU": (silu, -1.278464542761074)}
def read_idx(p):
    b = gzip.open(p, "rb").read(); dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)
ax = read_idx("/home/issan/Projects/claude/proj_004_drift/data/mnist/train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float64) / 255
X = torch.tensor(ax[np.random.default_rng(0).choice(len(ax), 1200, replace=False)]); del ax
N = 1200; STEPS = 4000
out = []
for name, (f, zc) in ACT.items():
    for top0 in (2 * zc, zc - 3.0):
        for seed in (0, 1):
            for opt, lr in (("sgd", 0.1), ("sgd", 0.01), ("gd", 0.01), ("adam", 1e-3), ("adam", 1e-4)):
                rng = np.random.default_rng(100 + seed); g0 = torch.Generator().manual_seed(seed)
                w = (torch.rand(784, generator=g0) * 2 - 1) / 28.0
                b = top0 - (X @ w).max()
                w = w.clone().requires_grad_(True); b = b.clone().detach().requires_grad_(True)
                v = torch.tensor([1.0, -1.0]); c = torch.tensor([6.0, -6.0]); y = torch.tensor(rng.integers(0, 2, N))
                o = torch.optim.Adam([w, b], lr=lr) if opt == "adam" else torch.optim.SGD([w, b], lr=lr)
                first_zc = first_0 = None; mx = -1e9
                for it in range(1, STEPS + 1):
                    idx = torch.arange(N) if opt == "gd" else torch.tensor(rng.integers(0, N, 16))
                    z = X[idx] @ w + b; logits = f(z)[:, None] * v + c
                    loss = torch.nn.functional.cross_entropy(logits, y[idx]); o.zero_grad(); loss.backward(); o.step()
                    if it % 5 == 0 or it <= 40:
                        with torch.no_grad():
                            t_ = float((X @ w + b).max()); mx = max(mx, t_)
                            if first_zc is None and t_ > zc: first_zc = it
                            if first_0 is None and t_ > 0: first_0 = it
                with torch.no_grad():
                    zz = X @ w + b
                r = (name, round(top0, 2), seed, opt, lr, first_zc, first_0, round(mx, 3), round(float(zz.max()), 3), round(float(((zz - zc).abs() < 0.1).double().mean()), 3))
                out.append(r); print(*r, flush=True)
print("\ncolumns: act, start top, seed, opt, lr, first step top>z_c, first step top>0, max top over run, final top, share of inputs within 0.1 of z_c at end")
