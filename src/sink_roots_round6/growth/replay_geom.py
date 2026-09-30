"""replay_geom: 保存した課題 5 の頭の状態から、1 層の箱を課題 5〜t_end まで再生し、重みの空間の押し P と戻り R を取る。
エンジンは fork2 の src/sink_roots_mnist_0930.py の関数をそのまま import（読み取りのみ）。乱数列は課題 1〜4 の
ラベルとバッチの引きを空で消費して合わせる（tau=0・env rl なので glabel と gbatch だけ）。
arm: base（R3_base の状態から）/ wcap（同じ状態から、課題 5 の頭の行ノルムで毎更新の後に縮める。追補 1 の R1 と同じ計算）/
ink（R6ink の状態から、入力の墨をそろえる）。
検算: 課題の頭・200・課題末の m を fork2 の arrays.npz の u_m1 と照合する（ビット一致を期待）。
記録（unit ごと）: 課題ごとに W,b を 0・200・T で保存。毎更新の Adam の一歩 dW（縮める前）について窓ごとに
  sum|dW|^2、sum 2 w.dW、sum dW.mu（m への寄与）、sum db、縮める分の sum dW_sh.mu・sum (|w|^2 の減り)。
"""
import argparse, sys, time, json
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "/home/issan/Projects/claude/wt/sink_roots_0930/src")
import sink_roots_mnist_0930 as E
torch.set_num_threads(1)
RAW = Path("/home/issan/Projects/obsidian-research-data/sink_roots_0930/mnist")
ap = argparse.ArgumentParser()
ap.add_argument("--act", default="ELU"); ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arm", default="base", choices=["base", "wcap", "ink"])
ap.add_argument("--t0", type=int, default=5); ap.add_argument("--t1", type=int, default=30)
ap.add_argument("--out", required=True)
a = ap.parse_args()
K, T, N = 10, 4000, E.N
src = {"base": "R3_base", "wcap": "R3_base", "ink": "R6ink"}[a.arm]
ref = {"base": "R3_base", "wcap": "R1wcap", "ink": "R6ink"}[a.arm]
st = np.load(RAW / f"{src}_{a.act}_s{a.seed}" / f"state_before_t{a.t0:03d}.npz")
X, _, _ = E.load_mnist(a.seed)
if a.arm == "ink":
    ink = X.sum(1, keepdim=True); X = X * (ink.mean() / ink)
X64 = X.double(); mu64 = X64.mean(0); mu32 = mu64.float()
muh = mu64 / mu64.norm()                           # round 6: x_top per unit is re-found every 25 updates
glabel, gbatch = E.stream("env_labels_0913", a.seed), E.stream("env_batch_0913", a.seed)
epochs = -(-(T * E.BATCH) // N)
for task in range(1, a.t0):                        # consume tasks 1..t0-1
    torch.randint(K, (N,), generator=glabel)
    for _ in range(epochs): torch.randperm(N, generator=gbatch)
P = [torch.tensor(st[f"P{i}"]).requires_grad_(True) for i in range(4)]
opt = E.Adam(P, 1e-3, 0.9, 0.999, 1e-8)
opt.load({"m": [torch.tensor(st[f"m{i}"]) for i in range(4)], "v": [torch.tensor(st[f"v{i}"]) for i in range(4)],
          "tm": [int(x) for x in st["tm"]], "tv": [int(x) for x in st["tv"]]})
refarr = np.load(RAW / f"{ref}_{a.act}_s{a.seed}" / "arrays.npz"); rgrid = list(refarr["grid"])
SNAP = [0, 25, 50, 100, 200, 300, 500, 1000, 2000, 3000, 4000]
out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
wcap = None
rec = {k: [] for k in ("W", "b", "acc")}
acc_keys = ("sq", "rad", "dm", "db", "sh_dm", "sh_sq", "sh_n", "vt_adam", "vt_sh")   # sink_roots_0930 round 6: the v_top ledger (G6)
led = []                     # per task: dict window -> arrays [H]
check = []
snapm = []
t_start = time.time()
for task in range(a.t0, a.t1 + 1):
    y_new = torch.randint(K, (N,), generator=glabel)
    order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:T * E.BATCH].reshape(T, E.BATCH)
    if task == a.t0:
        assert np.array_equal(y_new.numpy(), st["y_new"]), "label stream mismatch"
    Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32)
    if a.arm == "wcap" and task == 5:
        wcap = P[0].detach().norm(dim=1).clone()
    W_snap, b_snap = [], []
    L = {w: {k: torch.zeros(100, dtype=torch.float64) for k in acc_keys} for w in ("P", "R")}
    mrow = []
    for s in range(T + 1):
        if s in SNAP:
            W = P[0].detach().double(); b = P[1].detach().double()
            if s in (0, 200, T):
                W_snap.append(W.float().numpy().copy()); b_snap.append(b.float().numpy().copy())
            z = X64 @ W.T + b
            mrow.append(np.stack([z.mean(0).numpy(), z.std(0).numpy(), z.max(0).values.numpy(), (z > 0).sum(0).double().numpy(),
                                  (W @ mu64 / mu64.norm()).numpy(), W.norm(dim=1).numpy(), b.numpy()]))
            if s in rgrid:
                gi = rgrid.index(s); check.append(float(np.abs(z.mean(0).numpy() - refarr["u_m1"][task - 1, gi]).max()))
        if s == T: break
        if s % 25 == 0:
            with torch.no_grad():
                itop = (X64 @ P[0].detach().double().T + P[1].detach().double()).argmax(0)
                Xtop = X64[itop]; ptop = Xtop @ muh             # (H, 784), (H,)
        win = "P" if s < 200 else "R"
        idx = order[s]; xb, yb = X[idx], Y[idx]
        _, _, logits = E.forward(P, xb, a.act, 1)
        loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
        grads = torch.autograd.grad(loss, P)
        W0 = P[0].detach().clone(); b0 = P[1].detach().clone()
        opt.step(grads)
        with torch.no_grad():
            dW = (P[0] - W0).double(); db = (P[1] - b0).double(); W0d = W0.double()
            Lw = L[win]
            Lw["sq"] += (dW * dW).sum(1); Lw["rad"] += 2 * (W0d * dW).sum(1); Lw["dm"] += dW @ mu64; Lw["db"] += db
            Lw["vt_adam"] += (dW * Xtop).sum(1) - (dW @ muh) * ptop            # d(w_perp . x_top) from Adam's step
            if wcap is not None:
                Wm = P[0].detach().clone()
                nr = P[0].norm(dim=1)
                P[0].mul_(torch.clamp(wcap / nr, max=1.0)[:, None])
                dsh = (P[0] - Wm).double(); Wmd = Wm.double()
                Lw["sh_dm"] += dsh @ mu64; Lw["sh_sq"] += ((P[0].double() ** 2).sum(1) - (Wmd ** 2).sum(1)); Lw["sh_n"] += (nr > wcap).double()
                Lw["vt_sh"] += (dsh * Xtop).sum(1) - (dsh @ muh) * ptop            # ... and from the shrink
    rec["W"].append(np.stack(W_snap)); rec["b"].append(np.stack(b_snap))
    led.append({w: {k: v.numpy() for k, v in L[w].items()} for w in L})
    snapm.append(np.stack(mrow))
    print(f"task {task} maxdiff_m {max(check[-3:]):.2e} ({time.time()-t_start:.0f}s)", flush=True)
np.savez_compressed(out / "replay.npz", W=np.stack(rec["W"]), b=np.stack(rec["b"]), snap=np.stack(snapm), snapsteps=np.array(SNAP),
                    **{f"L_{w}_{k}": np.stack([l[w][k] for l in led]) for w in ("P", "R") for k in acc_keys},
                    tasks=np.arange(a.t0, a.t1 + 1), mu=mu64.numpy())
json.dump({"args": vars(a), "maxdiff_m_vs_fork2": max(check), "wall_s": time.time() - t_start}, open(out / "meta.json", "w"))
print("done", max(check))
