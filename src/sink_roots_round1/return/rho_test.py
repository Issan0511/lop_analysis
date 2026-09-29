"""反証 R7（導出役の R5 の縮小版・1 seed）: 戻りの窓で Adam の実変位が 全データの勾配 / sqrt(vhat) の積分の約半分（ρ≈0.5）になる理由を切り分ける。
箱は retprobe.py と同じ（生 MNIST/255・subset 1,200・init/ラベル/バッチの乱数列の hash も同じ・Adam 1e-3・batch 16・4,000 更新/課題・float32・1 スレッド・ELU は expm1）。
課題 PT（既定 6,8,10,12,14）の切替から W0 更新（谷の近く）で状態を複製し、4 つの変種で W1 更新まで進める:
  rr_b9 : 基準（同じ順列のバッチ・β1 0.9）= 基準の軌道と一致するはず
  wr_b9 : 復元抽出（毎歩 16 個を一様に）・β1 0.9
  rr_b0 : 同じ順列・β1 0
  wr_b0 : 復元抽出・β1 0
各変種で REC 更新ごとに、全データの勾配（float32 autograd）ḡ と自分の vhat で R_mem = −lr Σ_j μ̃_j ḡ_ij/(sqrt(vhat_ij)+eps) を記録し、
ρ = (実際の m の変位) / (R_mem の台形積分) を、切替時に開いた unit で平均の比として出す。"""
import argparse, copy, gzip, hashlib, math, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
N, BATCH, H, K = 1200, 16, 100, 10
def stream(role, seed):
    h = hashlib.sha256(f"pmnist_0905|{role}|{seed}".encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))
def read_idx(p):
    b = gzip.open(p, "rb").read(); dims = int(b[3]); shape = [int.from_bytes(b[4 + 4 * i:8 + 4 * i], "big") for i in range(dims)]
    return np.frombuffer(b, dtype=np.uint8, offset=4 + 4 * dims).reshape(shape)
SQ2 = math.sqrt(2.0)
def activ(z, act):
    if act == "LR": return torch.where(z > 0, z, z * 0.1)
    if act == "ELU": return torch.where(z > 0, z, torch.expm1(z.clamp_max(0)))
    if act == "GELU": return z * 0.5 * (1.0 + torch.erf(z / SQ2))
    if act == "SILU": return z * torch.sigmoid(z)
def initial(seed, dims):
    g = stream("init", seed); p = []
    for din, dout in zip(dims[:-1], dims[1:]):
        p += [(torch.rand(dout, din, generator=g) * 2 - 1) * (1.0 / math.sqrt(din)), (torch.rand(dout, generator=g) * 2 - 1) * (1.0 / math.sqrt(din))]
    return p
def lossf(P, xb, yb, act):
    z = xb @ P[0].T + P[1]; lg = activ(z, act) @ P[2].T + P[3]
    return (lg.logsumexp(-1) - (lg * yb).sum(-1)).mean()
def main(a):
    torch.manual_seed(a.seed)
    ax = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=stream("rl_subset", a.seed))[:N].numpy()
    X = torch.tensor(ax[subset]); X64 = X.double(); mu = X64.mean(0)
    mut = torch.cat([mu, torch.ones(1, dtype=torch.float64)])
    glabel, gbatch = stream("env_labels_0913", a.seed), stream("env_batch_0913", a.seed)
    gwr = torch.Generator().manual_seed(12345 + a.seed)
    P = [q.float().requires_grad_(True) for q in initial(a.seed, (784, H, K))]
    opt = torch.optim.Adam(P, lr=a.lr, betas=(0.9, 0.999), eps=1e-8)
    PT = set(int(x) for x in a.ptasks.split(",")); maxT = max(PT)
    res = {v: {"act": [], "pred": [], "pred_coarse": []} for v in ("rr_b9", "wr_b9", "rr_b0", "wr_b0")}
    chk = []
    t0 = time.monotonic()
    for task in range(1, maxT + 1):
        y = torch.randint(K, (N,), generator=glabel); Y = torch.nn.functional.one_hot(y, K).float()
        epochs = -(-(a.steps * BATCH) // N)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:a.steps * BATCH].reshape(a.steps, BATCH)
        with torch.no_grad():
            k0 = ((X64 @ P[0].double().T + P[1].double()) > 0).sum(0)
        openu = (k0 > 0).numpy()
        for it in range(a.steps):
            if task in PT and it == a.w0:
                base_state = copy.deepcopy(opt.state_dict()); baseP = [p.detach().clone() for p in P]
                for v in res:
                    Q = [p.clone().requires_grad_(True) for p in baseP]
                    o2 = torch.optim.Adam(Q, lr=a.lr, betas=(0.9, 0.999), eps=1e-8); o2.load_state_dict(copy.deepcopy(base_state))
                    if v.endswith("b0"):
                        for gr in o2.param_groups: gr["betas"] = (0.0, 0.999)
                    ms, Rs, ts = [], [], []
                    def rec(step):
                        Qd = [q.detach().double() for q in Q]
                        ms.append((Qd[0] @ mu + Qd[1]).numpy())
                        Qg = [q.detach().clone().requires_grad_(True) for q in Q]
                        L = lossf(Qg, X, Y, a.act); gW, gb = torch.autograd.grad(L, [Qg[0], Qg[1]])
                        st = o2.state[Q[0]]; stb = o2.state[Q[1]]
                        b2c = 1 - 0.999 ** float(st["step"])
                        vh = torch.cat([st["exp_avg_sq"].double(), stb["exp_avg_sq"].double()[:, None]], 1) / b2c
                        gg = torch.cat([gW.double(), gb.double()[:, None]], 1)
                        Rs.append((-a.lr * (gg / (vh.sqrt() + 1e-8)) * mut).sum(1).numpy()); ts.append(step)
                    rec(0)
                    for s2 in range(a.w1 - a.w0):
                        if v.startswith("rr"): idx = order[a.w0 + s2]
                        else: idx = torch.randint(N, (BATCH,), generator=gwr)
                        L = lossf(Q, X[idx], Y[idx], a.act); o2.zero_grad(set_to_none=True); L.backward(); o2.step()
                        if (s2 + 1) % a.rec == 0: rec(s2 + 1)
                    ms, Rs, ts = np.array(ms), np.array(Rs), np.array(ts, float)
                    actd = ms[-1] - ms[0]
                    pred = (0.5 * (Rs[1:] + Rs[:-1]) * np.diff(ts)[:, None]).sum(0)
                    # 粗い grid（導出役の probe の刻み 700,1000,1400,2000 に相当する点だけ）での台形
                    cg = [c for c in (0, 300, 600, 1000, 1600) if c <= ts[-1]]
                    ci = [int(np.where(ts == c)[0][0]) for c in cg]
                    predc = (0.5 * (Rs[ci][1:] + Rs[ci][:-1]) * np.diff(np.array(cg, float))[:, None]).sum(0)
                    res[v]["act"].append(actd[openu]); res[v]["pred"].append(pred[openu]); res[v]["pred_coarse"].append(predc[openu])
                    if v == "rr_b9": chk.append(ms[-1])
            idx = order[it]
            L = lossf(P, X[idx], Y[idx], a.act); opt.zero_grad(set_to_none=True); L.backward(); opt.step()
            if task in PT and it + 1 == a.w1:
                with torch.no_grad(): mb = (P[0].double() @ mu + P[1].double()).numpy()
                print(f"  task {task}: rr_b9 の複製と基準の m の差 max {np.abs(chk[-1] - mb).max():.2e}", flush=True)
        print(f"task {task} done ({time.monotonic() - t0:.0f}s)", flush=True)
    print(f"act {a.act} seed {a.seed} tasks {sorted(PT)} window {a.w0}->{a.w1} rec {a.rec}")
    print("変種 | 実変位（平均） | ∫R_mem 細かい刻み | ρ = 実/∫ | ∫R_mem 粗い刻み | ρ（粗）")
    for v, r in res.items():
        A = np.concatenate(r["act"]); Pp = np.concatenate(r["pred"]); Pc = np.concatenate(r["pred_coarse"])
        print(f"{v} | {A.mean():+.4f} | {Pp.mean():+.4f} | {A.mean()/Pp.mean():.3f} | {Pc.mean():+.4f} | {A.mean()/Pc.mean():.3f}  (n={len(A)})")
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="ELU"); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--ptasks", default="6,8,10,12,14")
    ap.add_argument("--w0", type=int, default=400); ap.add_argument("--w1", type=int, default=2000); ap.add_argument("--rec", type=int, default=20)
    main(ap.parse_args())
