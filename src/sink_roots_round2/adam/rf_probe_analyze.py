"""rf_probe.py（seed 1）の集計: 共分散の割合の割れ半分・A6 予測 / ラベル平均・1 歩目の符号"""
import glob, gzip, hashlib, numpy as np, torch
b1, eps, lr, beta2 = 0.9, 1e-8, 1e-3, 0.999
def kvec_for(seed):
    h = hashlib.sha256(f"pmnist_0905|rl_subset|{seed}".encode()).digest()
    g = torch.Generator().manual_seed(int.from_bytes(h[:8], "little") & ((1 << 63) - 1))
    b = gzip.open("/home/issan/Projects/claude/proj_004_drift/data/mnist/train-images-idx3-ubyte.gz", "rb").read()
    axx = np.frombuffer(b, dtype=np.uint8, offset=16).reshape(-1, 784).astype(np.float32) / 255
    sub = torch.randperm(len(axx), generator=g)[:1200].numpy()
    return np.concatenate([axx[sub].astype(np.float64).mean(0), [1.0]])
for fn in sorted(glob.glob("rf_runs/rf_*_s*.npz")):
    d = np.load(fn); act = fn.split("rf_")[-1].split("_s")[0]; seed = int(fn.split("_s")[-1].split(".")[0]); kv = kvec_for(seed)
    print(f"==== {act} seed {seed}")
    for t in (3, 10):
        p = f"t{t}_"
        if p + "dm" not in d: continue
        live = d[p + "k"] > 0; k = d[p + "k"][live]
        dm = d[p + "dm"][:, :, live]; R = dm.shape[0]; h = R // 2
        E = [np.cumsum(dm[:h].mean(0), 0), np.cumsum(dm[h:].mean(0), 0)]
        kE = [np.cumsum(d[p + f"kEDEM_h{i}"][:, live], 0) for i in (0, 1)]
        cov = [(E[i][-1] - kE[i][-1]) / E[i][-1] for i in (0, 1)]
        Eall = np.cumsum(dm.mean(0), 0)
        print(f" 切替 {t}（生きた unit {live.sum()}、k 中央値 {np.median(k):.0f}）")
        print(f"   共分散の割合（200 更新）: 前半 16 回 {np.median(cov[0]):+.3f}  後半 16 回 {np.median(cov[1]):+.3f}  unit ごとの割れ半分の相関 {np.corrcoef(cov[0], cov[1])[0,1]:+.2f}  両半分とも負の unit {np.mean((cov[0]<0)&(cov[1]<0)):.2f}")
        for lo, hi in ((1, 3), (4, 10), (11, 30), (31, 100), (101, 10000)):
            m = (k >= lo) & (k <= hi)
            if m.sum() >= 3: print(f"     k {lo}-{hi}（{m.sum()}）: 前半 {np.median(cov[0][m]):+.3f}  後半 {np.median(cov[1][m]):+.3f}")
        # A6
        V0 = d[p + "V0"][live].astype(np.float64); gbar = d[p + "gbar"][live].astype(np.float64); G2 = gbar ** 2 + d[p + "sig2"][live].astype(np.float64)
        s = np.arange(1, 201)
        fac = np.cumsum((1 - b1 ** s)[:, None, None] / (np.sqrt(beta2 ** s[:, None, None] * V0[None] + (1 - beta2 ** s)[:, None, None] * G2[None]) + eps), 0)
        row = []
        for tau in (10, 30, 50, 100, 200):
            pc = -lr * (gbar * fac[tau - 1]) @ kv
            row.append(f"τ{tau} {np.median(pc)/np.median(Eall[tau-1]):.2f}")
        print("   A6 予測（履歴なし）/ ラベル平均の実変位（中央値どうし）: " + "  ".join(row))
        print(f"   ラベル平均の 200 更新が下向き {np.mean(Eall[-1]<0):.2f}  S>0 {np.mean(d[p+'S'][live]>0):.2f}  S と符号一致 {np.mean((d[p+'S'][live]>0)==(Eall[-1]<0)):.2f}")
        # 1 歩目
        e1 = dm[:, 0].mean(0); e1n = d[p + "dm_nohist1"][:, live].mean(0)
        S = d[p + "S"][live]; SD = d[p + "SD"][live]
        print(f"   1 歩目のラベル平均が下向き: 実 {np.mean(e1<0):.2f}  履歴を除く {np.mean(e1n<0):.2f}   S（U2）との符号一致: 実 {np.mean((S>0)==(e1<0)):.2f}  履歴を除く {np.mean((S>0)==(e1n<0)):.2f}  S^D と 履歴を除く {np.mean((SD>0)==(e1n<0)):.2f}")
