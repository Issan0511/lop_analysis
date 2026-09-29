"""改訂役（0930）: B5 の検査 — 2 層 RL-MNIST の第 2 層で、切替時の状態だけから出す「A6 の 2 層版」（期待勾配 × 分母の記憶 × 信号対雑音）の
自分・上流の割合が、ラベルを引き直した平均の実変位（200 更新）の割合に合うか。

箱: 784-100-100-10・両層 ELU（--act）・Adam 1e-3（β 0.9/0.999・ε 1e-8）・batch 16・6,000 更新/課題・N 1,200・float32・CPU 1 スレッド。
乱数列は導出役 twolayer_kernel_probe.py（= drive_recon_0930/align_probe2.py）と同じ。課題ごとの正解率などを印字して導出役のログと照合する。

切替 t（--probe）で、新ラベルを引く前の状態から:
 (1) 座標ごと（第 2 層の行 i: 101 座標、第 1 層の行 j: 785 座標）に
       ḡ   = 一様ラベルの期待の全データ勾配（U2 の式）
       σ²  = ラベルの揺らぎ（1 枚ずつ独立・一様）＋ 非復元のバッチ抽選の揺らぎ（batch 16）
       V0  = 切替時の v̂（補正済み）、M0 = 切替時の m
     A6: Δθ_c(τ) = −η [ ḡ_c Σ_{s≤τ} (1−β1^s)/(√(β2^s V0_c + (1−β2^s) G_c²) + ε) + M0_c Σ_{s≤τ} β1^s/(同じ分母) ]、G² = ḡ² + σ²
     1 次の帳簿: 自分 = μ̃1(切替時)·Δθ2_i、上流 = Σ_j W2_ij ν_j·Δθ1_j（ν_j = mean_n φ′(z1_j(n)) x̃_n、切替時）
     τ→1 は導出役の Adam 核（S2D・U2D）、分母を V0 に固定し σ を捨てると「凍結の Adam 核 × τ」、τ→∞ は Σ k ḡ/G（信号対雑音）の核。
 (2) 同じ状態からラベルだけ R 回引き直して 200 更新（バッチ順は本走と同じ）。毎更新の
       1 次の帳簿の増分: 自分 δw·μ1(0) + δb、上流 w(0)·δμ1、端点の交差は残り
       経路の帳簿: 自分 δw·μ1(s−1) + δb、上流 w(s−1)·δμ1、経路の交差 δw·δμ1
     をラベル平均と、τ = 10, 30, 50, 100, 200 の引きごとの累積で保存する。
 (3) 状態を戻して本走の課題を普通に回す。
"""
import argparse, hashlib, math, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from awp import read_idx, stream, initial, activ, gate, Adam, DATA, N, BATCH
torch.set_num_threads(1)
H, K = 100, 10
TAUS = (1, 3, 10, 30, 50, 100, 200)
B1, B2, EPS, LR = 0.9, 0.999, 1e-8, 1e-3

def rows(opt, li):
    W, b = 2 * li, 2 * li + 1
    bc1 = 1 - opt.b1 ** opt.t; bc2 = 1 - opt.b2 ** opt.t
    M = torch.cat([opt.m[W], opt.m[b][:, None]], 1).double()
    V = torch.cat([opt.v[W], opt.v[b][:, None]], 1).double()
    return M, V / bc2, bc1

@torch.no_grad()
def a6_predict(P, opt, X, Xt, act, steps=200):
    W1, b1, W2, b2, W3, b3 = [q.detach().double() for q in P]
    z1 = X @ W1.T + b1; a1 = activ(z1, act); z2 = a1 @ W2.T + b2; lg = activ(z2, act) @ W3.T + b3
    g1, g2 = gate(z1, act), gate(z2, act)
    p = lg.softmax(-1); Vr = W3.T; Vc = Vr - Vr.mean(1, keepdim=True)            # [H2,K]
    a1t = torch.cat([a1, torch.ones(N, 1, dtype=torch.float64)], 1)               # [N,H1+1]
    mu1t = a1t.mean(0)
    # 第 2 層
    Ed2 = g2 * (p @ Vc.T)                                                          # [N,H2]
    gb2 = Ed2.T @ a1t / N                                                          # [H2,H1+1]
    vc2 = (Vc ** 2).sum(1)                                                         # [H2]
    var_lab2 = ((g2 ** 2 * vc2[None, :] / K).T @ (a1t ** 2)) / N / BATCH
    var_b2 = (((Ed2 ** 2).T @ (a1t ** 2)) / N - gb2 ** 2) * (N - BATCH) / (N - 1) / BATCH
    # 第 1 層: δ1_j(n) = φ′1_j(n) u_j(n)·e_n、u_j(n) = Σ_i W2_ij φ′2_i(n) v_i
    U = torch.einsum('ij,ni,ik->njk', W2, g2, Vr)                                  # [N,H1,K]
    Uc = U - U.mean(-1, keepdim=True)
    Ed1 = g1 * torch.einsum('njk,nk->nj', Uc, p)                                   # [N,H1]
    chk = (Ed1 - g1 * (Ed2 @ W2)).abs().max().item()
    gb1 = Ed1.T @ Xt / N                                                           # [H1,785]
    uc2 = (Uc ** 2).sum(-1)                                                        # [N,H1]
    var_lab1 = ((g1 ** 2 * uc2 / K).T @ (Xt ** 2)) / N / BATCH
    var_b1 = (((Ed1 ** 2).T @ (Xt ** 2)) / N - gb1 ** 2) * (N - BATCH) / (N - 1) / BATCH
    nu = (g1.T @ Xt) / N                                                           # [H1,785]
    M1, V1, bc1 = rows(opt, 0); M2, V2, _ = rows(opt, 1)
    out = {"chk_Ed1": chk}
    s = torch.arange(1, steps + 1, dtype=torch.float64)
    for li, gb, var, V0, M0 in ((2, gb2, var_lab2 + var_b2, V2, M2), (1, gb1, var_lab1 + var_b1, V1, M1)):
        G2 = gb ** 2 + var.clamp_min(0)
        # 累積の係数 [steps, rows, cols] は大きいので τ ごとに逐次和
        cf = torch.zeros_like(gb); ch = torch.zeros_like(gb); cf0 = torch.zeros_like(gb)
        pred = {}; hist = {}; frz = {}
        for si in range(1, steps + 1):
            den = torch.sqrt(B2 ** si * V0 + (1 - B2 ** si) * G2) + EPS
            cf += (1 - B1 ** si) / den; ch += B1 ** si / den
            cf0 += (1 - B1 ** si) / (torch.sqrt(V0) + EPS)
            if si in TAUS:
                pred[si] = -LR * gb * cf; hist[si] = -LR * M0 * ch; frz[si] = -LR * gb * cf0
        snr = gb / torch.sqrt(G2)                                                  # τ→∞ の一歩 ÷ η
        rho = V0 / G2
        if li == 2:
            proj = lambda d: d @ mu1t                                               # [H2]
        else:
            proj = lambda d: W2 @ (nu * d).sum(1)                                   # [H2]  Σ_j W2_ij ν_j·Δθ1_j
        for tau in TAUS:
            out[f"L{li}_pred_{tau}"] = proj(pred[tau]); out[f"L{li}_hist_{tau}"] = proj(hist[tau]); out[f"L{li}_frz_{tau}"] = proj(frz[tau])
        out[f"L{li}_snr"] = proj(-LR * snr)                                        # 1 更新あたりの漸近の変位
        out[f"L{li}_sgd"] = proj(-LR * gb)                                         # SGD の 1 歩（η 1e-3）
        out[f"L{li}_rho_med"] = rho.median()
        out[f"L{li}_snr_absmed"] = snr.abs().median()
    out["k2"] = (z2 > 0).double().sum(0); out["m2"] = z2.mean(0); out["p_old"] = p.max(1).values.mean()
    return out

def run(a):
    torch.manual_seed(a.seed)
    ax = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=stream("rl_subset", a.seed))[:N].numpy()
    X = torch.tensor(ax[subset], dtype=torch.float32); Xd = X.double()
    Xt = torch.cat([Xd, torch.ones(N, 1, dtype=torch.float64)], 1)
    glabel, gbatch = stream("env_labels_0913", a.seed), stream("env_batch_0913", a.seed)
    P = [q.to(dtype=torch.float32).requires_grad_(True) for q in initial(a.seed, (784, H, H, K))]
    opt = Adam(P, LR, B1, B2, EPS)
    probes = sorted(int(x) for x in a.probe.split(","))
    out = {}; t0 = time.monotonic()
    def mu1_of():
        with torch.no_grad():
            return activ((X @ P[0].detach().T + P[1].detach()), a.act).double().mean(0)
    def fwd_loss(idx, Y):
        z = X[idx] @ P[0].T + P[1]; h1 = activ(z, a.act); z = h1 @ P[2].T + P[3]; h2 = activ(z, a.act); lg = h2 @ P[4].T + P[5]
        loss = (lg.logsumexp(-1) - (lg * Y[idx]).sum(-1)).mean()
        for q in P: q.grad = None
        loss.backward(); opt.step()
    for task in range(1, max(probes) + 1):
        y = torch.randint(K, (N,), generator=glabel); Y = torch.nn.functional.one_hot(y, K).to(torch.float32)
        epochs = -(-(a.steps * BATCH) // N)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:a.steps * BATCH].reshape(a.steps, BATCH)
        if task in probes:
            pr = a6_predict(P, opt, Xd, Xt, a.act, 200)
            for k_, v_ in pr.items(): out[f"t{task}_{k_}"] = v_.numpy() if torch.is_tensor(v_) else v_
            saveP = [q.detach().clone() for q in P]; saveM = [m.clone() for m in opt.m]; saveV = [v.clone() for v in opt.v]; savet = opt.t
            W2_0 = P[2].detach().double().clone(); b2_0 = P[3].detach().double().clone(); mu0 = mu1_of()
            mu1t0 = torch.cat([mu0, torch.ones(1, dtype=torch.float64)])
            with torch.no_grad():
                z1s = Xd @ P[0].detach().double().T + P[1].detach().double()
                nu0 = (gate(z1s, a.act).T @ Xt) / N                                           # [H1,785] 切替時の ν_j
            M1_0 = rows(opt, 0)[0].clone(); M2_0 = rows(opt, 1)[0].clone()
            W1t_0 = torch.cat([P[0].detach().double(), P[1].detach().double()[:, None]], 1)
            keys = ("s1", "u1", "ps", "pu", "pc", "dm", "hs", "hu", "u1lin")
            acc = {k_: np.zeros((200, H)) for k_ in keys}
            cum = {k_: np.zeros((a.R, len(TAUS), H)) for k_ in keys}
            gl = torch.Generator().manual_seed(int(hashlib.sha256(f"a6redraw|{a.seed}|{task}".encode()).hexdigest()[:12], 16))
            for r in range(a.R):
                with torch.no_grad():
                    for q, s0 in zip(P, saveP): q.copy_(s0)
                    for m, s0 in zip(opt.m, saveM): m.copy_(s0)
                    for v, s0 in zip(opt.v, saveV): v.copy_(s0)
                opt.t = savet
                yr = torch.randint(K, (N,), generator=gl); Yr = torch.nn.functional.one_hot(yr, K).to(torch.float32)
                W2p, b2p, mup = W2_0.clone(), b2_0.clone(), mu0.clone(); W1tp = W1t_0.clone()
                run_c = {k_: np.zeros(H) for k_ in keys}
                for s in range(1, 201):
                    fwd_loss(order[s - 1], Yr)
                    W2n = P[2].detach().double(); b2n = P[3].detach().double(); mun = mu1_of()
                    W1tn = torch.cat([P[0].detach().double(), P[1].detach().double()[:, None]], 1)
                    dW = W2n - W2p; db = b2n - b2p; dmu = mun - mup
                    _, V1h, bc1 = rows(opt, 0); _, V2h, _ = rows(opt, 1)
                    D1s = 1.0 / (V1h.sqrt() + EPS); D2s = 1.0 / (V2h.sqrt() + EPS)
                    hs = -(LR / bc1) * ((D2s * M2_0 * B1 ** s) @ mu1t0)                           # 旧モーメントの分（自分・1 次）
                    hu = W2_0 @ (nu0 * (-(LR / bc1) * D1s * M1_0 * B1 ** s)).sum(1)                # 旧モーメントの分（上流・線形化）
                    u1lin = W2_0 @ (nu0 * (W1tn - W1tp)).sum(1)                                  # 上流の線形化（切替時の ν）
                    inc = {"s1": (dW @ mu0 + db).numpy(), "u1": (W2_0 @ dmu).numpy(),
                           "ps": (dW @ mup + db).numpy(), "pu": (W2p @ dmu).numpy(), "pc": (dW @ dmu).numpy(),
                           "dm": (W2n @ mun + b2n - W2p @ mup - b2p).numpy(),
                           "hs": hs.numpy(), "hu": hu.numpy(), "u1lin": u1lin.numpy()}
                    for k_ in keys:
                        acc[k_][s - 1] += inc[k_] / a.R; run_c[k_] += inc[k_]
                    if s in TAUS:
                        ti = TAUS.index(s)
                        for k_ in keys: cum[k_][r, ti] = run_c[k_]
                    W2p, b2p, mup = W2n.clone(), b2n.clone(), mun; W1tp = W1tn
            for k_ in keys:
                out[f"t{task}_rd_{k_}"] = acc[k_].astype(np.float32); out[f"t{task}_cum_{k_}"] = cum[k_].astype(np.float32)
            with torch.no_grad():
                for q, s0 in zip(P, saveP): q.copy_(s0)
                for m, s0 in zip(opt.m, saveM): m.copy_(s0)
                for v, s0 in zip(opt.v, saveV): v.copy_(s0)
            opt.t = savet
        for s in range(a.steps):
            fwd_loss(order[s], Y)
        with torch.no_grad():
            z1 = Xd @ P[0].detach().double().T + P[1].detach().double(); a1 = activ(z1, a.act)
            z2 = a1 @ P[2].detach().double().T + P[3].detach().double(); lg = activ(z2, a.act) @ P[4].detach().double().T + P[5].detach().double()
        print(f"L2 {a.act} s{a.seed} task {task} acc {float((lg.argmax(-1) == y).double().mean()):.3f} p1 {float((z1 > 0).double().mean()):.3f} p2 {float((z2 > 0).double().mean()):.3f} m2 {float(z2.mean()):.2f} ({time.monotonic() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / f"A6L2_{a.act}_s{a.seed}_p{'-'.join(map(str, probes))}.npz", taus=np.array(TAUS), **out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="ELU"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=6000); ap.add_argument("--probe", default="2,3")
    ap.add_argument("--R", type=int, default=16); ap.add_argument("--out", default="runs_a6")
    run(ap.parse_args())
