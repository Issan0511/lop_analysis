# 導出役 ../derive/adam_window_probe.py の複製（import 用・無改変）
"""P2/P6 の検算用の小さい走: 1 層 RL-MNIST（784-100-10・生 MNIST/255・N 1,200・batch 16・Adam 1e-3・float32・CPU 1 スレッド）。
箱と乱数列の hash は drive_recon_0930/align_probe_ref.py（= pplus_sign_1layer_0925）と同じ。Adam は torch と同じ式を手で書く（状態を覗くため）。

記録（第 1 層の unit i の行 theta_i = (w_i, b_i)、方向 k = (mu, 1)、m_i = k.theta_i）:
  各 probe 切替の課題全体（T 更新）で毎更新:
    dm_act   実変位（float32 の param 差を float64 で射影）
    dm_hist  -(eta/bc1) k^T D_s (b1^s M_0)          旧課題のモーメントの残り
    dm_new   -(eta/bc1) k^T D_s (M_s - b1^s M_0)     新しい勾配のモーメント
    dm_inst  -eta k^T D_s g_s                        運動量なしの Adam 核
    dm_raw   -eta k.g_s                              生の勾配（SGD の核、同じ軌道の上）
    dm_frz   -(eta/bc1) k^T D_0 M_s                  分母を切替時に凍結
  座標ごと: M_0、v-hat のスナップショット（s = 0,10,50,200,1000,2000,T）、窓ごとの g の和・g^2 の和（押し 1..200 / 戻り 201..T）
  切替時の状態量: S（ユークリッド核）、S^D（Adam 核 = 切替時の D）、自己項（床基準・0 基準）、
    座標ごとの期待勾配 gbar（U2）、揺らぎ sig2、G2 = gbar^2 + sig2（ラベルとバッチの抽選についての式）
  --redraw_task: その切替で、同じ状態からラベルだけ R 回引き直して 200 更新（バッチ順は本走と同じ）。
    毎更新の dm_act と、E[D_s]・E[M_s]（座標ごと）を貯め、E[k^T D M] と k^T E[D] E[M] を比べる。
"""
import argparse, gzip, hashlib, math, time
from pathlib import Path
import numpy as np, torch
torch.set_num_threads(1)
DATA = Path("/home/issan/Projects/claude/proj_004_drift/data/mnist")
N, BATCH, H, K = 1200, 16, 100, 10
SQRT2 = math.sqrt(2.0); INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)
SNAP = (0, 10, 50, 200, 1000, 2000)
FLOOR = {"ELU": -1.0, "GELU": -0.16997, "SILU": -0.27846, "R": 0.0, "LR": 0.0}

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
def gate(z, act):
    if act == "LR":   return torch.where(z > 0, torch.ones_like(z), torch.full_like(z, 0.1))
    if act == "ELU":  return torch.where(z > 0, torch.ones_like(z), z.clamp_max(0).exp())
    if act == "R":    return (z > 0).to(z.dtype)
    if act == "GELU": return 0.5 * (1.0 + torch.erf(z / SQRT2)) + z * torch.exp(-0.5 * z * z) * INV_SQRT_2PI
    if act == "SILU":
        s = torch.sigmoid(z); return s * (1.0 + z * (1.0 - s))
def initial(seed, dims):
    g = stream("init", seed); p = []
    for din, dout in zip(dims[:-1], dims[1:]):
        p += [(torch.rand(dout, din, generator=g) * 2 - 1) * (1.0 / math.sqrt(din)),
              (torch.rand(dout, generator=g) * 2 - 1) * (1.0 / math.sqrt(din))]
    return p

class Adam:
    """torch.optim.Adam（amsgrad なし・減衰なし）と同じ式"""
    def __init__(self, P, lr, b1, b2, eps):
        self.P, self.lr, self.b1, self.b2, self.eps = P, lr, b1, b2, eps
        self.m = [torch.zeros_like(p) for p in P]; self.v = [torch.zeros_like(p) for p in P]; self.t = 0
    @torch.no_grad()
    def step(self):
        self.t += 1; t = self.t
        bc1 = 1 - self.b1 ** t; bc2s = math.sqrt(1 - self.b2 ** t)
        for p, m, v in zip(self.P, self.m, self.v):
            g = p.grad
            m.lerp_(g, 1 - self.b1); v.mul_(self.b2).addcmul_(g, g, value=1 - self.b2)
            denom = (v.sqrt() / bc2s).add_(self.eps)
            p.addcdiv_(m, denom, value=-self.lr / bc1)
    def layer1(self):
        """第 1 層の (M, v, bc1, bc2) を [H, 785] の float64 で返す"""
        M = torch.cat([self.m[0], self.m[1][:, None]], 1).double()
        V = torch.cat([self.v[0], self.v[1][:, None]], 1).double()
        return M, V, 1 - self.b1 ** self.t, 1 - self.b2 ** self.t
    def D1(self):
        M, V, bc1, bc2 = self.layer1()
        return 1.0 / ((V.sqrt() / math.sqrt(bc2)) + self.eps)

def theta1(P):
    return torch.cat([P[0].detach(), P[1].detach()[:, None]], 1).double()

@torch.no_grad()
def switch_state(P, opt, X, Xt, kvec, act):
    """切替時（新ラベルを引く前）の状態量。ラベル期待値は E[y] = 一様"""
    W1, b1, W2, b2 = [q.detach().double() for q in P]
    z = X.double() @ W1.T + b1; g = gate(z, act); a = activ(z, act)
    logits = a @ W2.T + b2; p = logits.softmax(-1)
    V = W2.T; vbar = V.mean(1); Vc = V - vbar[:, None]
    vp = p @ Vc.T                                   # [N,H]  v_i^c . p_n
    delta_bar = g * vp                              # [N,H]  E[delta_in]（U2）
    gbar = (delta_bar.T @ Xt) / N                   # [H,785] 座標ごとの期待勾配（全データ・1 歩）
    vc2 = (Vc ** 2).sum(1)                          # |v_i^c|^2
    # 揺らぎ: ラベル（一様）と非復元のバッチ抽選
    var_lab = ((g ** 2).T @ (Xt ** 2)) * (vc2[:, None] / K) / N / BATCH
    u = delta_bar[:, :, None] * Xt[:, None, :] if False else None
    m1 = gbar; m2 = ((delta_bar ** 2).T @ (Xt ** 2)) / N
    var_batch = (m2 - m1 ** 2) * (N - BATCH) / (N - 1) / BATCH
    sig2 = var_lab + var_batch
    D0 = opt.D1()
    Kn = Xt @ kvec                                  # [N]  x_n.mu + 1
    KD = Xt @ (D0 * kvec[None, :]).T                # [N,H] Adam 核 k^T D_i x~_n
    S = (Kn[:, None] * delta_bar).sum(0)
    SD = (KD * delta_bar).sum(0)
    out = dict(S=S, SD=SD, gbar=gbar, sig2=sig2, D0=D0)
    for name, ref in (("fl", FLOOR[act]), ("0", 0.0)):
        selfK = torch.zeros(H, dtype=torch.float64); selfKD = torch.zeros(H, dtype=torch.float64)
        for i in range(H):
            lm = logits - (a[:, i:i + 1] - ref) * V[i][None, :]
            st = (p - lm.softmax(-1)) @ Vc[i]
            selfK[i] = (Kn * g[:, i] * st).sum(); selfKD[i] = (KD[:, i] * g[:, i] * st).sum()
        out[f"selfS_{name}"] = selfK; out[f"selfSD_{name}"] = selfKD
    out["k"] = (z > 0).double().sum(0); out["top"] = z.max(0).values; out["m"] = z.mean(0)
    out["acc_old_pmax"] = p.max(1).values.mean()
    return out

def train_step(P, opt, xb, yb, act):
    z = xb @ P[0].T + P[1]; a = activ(z, act); logits = a @ P[2].T + P[3]
    loss = (logits.logsumexp(-1) - (logits * yb).sum(-1)).mean()
    for q in P: q.grad = None
    loss.backward()
    return loss

def window(P, opt, X, Y, order, act, kvec, T, rec_every=True):
    """1 課題分（T 更新）を回し、第 1 層の毎更新の分解を記録"""
    b1, lr = opt.b1, opt.lr
    M0, _, _, _ = opt.layer1(); M0 = M0.clone(); D0 = opt.D1()
    Vsnap = {}; Mv, Vv, _, bc2 = opt.layer1(); Vsnap[0] = (Vv / bc2).float().numpy()
    keys = ("dm_act", "dm_hist", "dm_new", "dm_inst", "dm_raw", "dm_frz")
    rec = {k: np.zeros((T, H), np.float64) for k in keys}
    gs = {w: torch.zeros(H, 785, dtype=torch.float64) for w in ("p", "r")}
    g2 = {w: torch.zeros(H, 785, dtype=torch.float64) for w in ("p", "r")}
    th = theta1(P)
    for s in range(1, T + 1):
        idx = order[s - 1]
        train_step(P, opt, X[idx], Y[idx], act)
        g = torch.cat([P[0].grad, P[1].grad[:, None]], 1).double()
        opt.step()
        M, V, bc1, bc2 = opt.layer1()
        D = 1.0 / ((V.sqrt() / math.sqrt(bc2)) + opt.eps)
        th_new = theta1(P)
        hist = (b1 ** s) * M0
        rec["dm_act"][s - 1] = ((th_new - th) @ kvec).numpy()
        rec["dm_hist"][s - 1] = (-(lr / bc1) * ((hist * D) @ kvec)).numpy()
        rec["dm_new"][s - 1] = (-(lr / bc1) * (((M - hist) * D) @ kvec)).numpy()
        rec["dm_inst"][s - 1] = (-lr * ((g * D) @ kvec)).numpy()
        rec["dm_raw"][s - 1] = (-lr * (g @ kvec)).numpy()
        rec["dm_frz"][s - 1] = (-(lr / bc1) * ((M * D0) @ kvec)).numpy()
        w = "p" if s <= 200 else "r"
        gs[w] += g; g2[w] += g * g
        if s in SNAP: Vsnap[s] = (V / bc2).float().numpy()
        th = th_new
    Vsnap[T] = (V / bc2).float().numpy()
    return rec, dict(M0=M0.float().numpy(), gsum_p=gs["p"].numpy(), gsum_r=gs["r"].numpy(),
                     g2sum_p=g2["p"].numpy(), g2sum_r=g2["r"].numpy(),
                     **{f"V_{s}": v for s, v in Vsnap.items()})

def redraw(P, opt, X, order, act, kvec, R, steps, seed, task):
    """同じ状態からラベルだけ R 回引き直して steps 更新。状態は最後に元へ戻す"""
    saveP = [q.detach().clone() for q in P]; saveM = [m.clone() for m in opt.m]; saveV = [v.clone() for v in opt.v]; savet = opt.t
    b1, lr = opt.b1, opt.lr
    D0 = opt.D1()
    sumD = torch.zeros(steps, H, 785, dtype=torch.float32); sumM = torch.zeros(steps, H, 785, dtype=torch.float32)
    sumG = torch.zeros(steps, H, 785, dtype=torch.float32)
    dm = np.zeros((R, steps, H)); dmf = np.zeros((R, steps, H)); dmg0 = np.zeros((R, steps, H))
    bc1s = np.zeros(steps)
    g1 = torch.zeros(H, 785, dtype=torch.float64)
    gl = torch.Generator().manual_seed(int(hashlib.sha256(f"redraw|{seed}|{task}".encode()).hexdigest()[:12], 16))
    for r in range(R):
        with torch.no_grad():
            for q, s0 in zip(P, saveP): q.copy_(s0)
            for m, s0 in zip(opt.m, saveM): m.copy_(s0)
            for v, s0 in zip(opt.v, saveV): v.copy_(s0)
        opt.t = savet
        y = torch.randint(K, (N,), generator=gl)
        Y = torch.nn.functional.one_hot(y, K).to(torch.float32)
        th = theta1(P)
        for s in range(1, steps + 1):
            idx = order[s - 1]
            train_step(P, opt, X[idx], Y[idx], act)
            g = torch.cat([P[0].grad, P[1].grad[:, None]], 1).double()
            if s == 1: g1 += g
            opt.step()
            M, V, bc1, bc2 = opt.layer1()
            D = 1.0 / ((V.sqrt() / math.sqrt(bc2)) + opt.eps)
            th_new = theta1(P)
            dm[r, s - 1] = ((th_new - th) @ kvec).numpy()
            dmf[r, s - 1] = (-(lr / bc1) * ((M * D0) @ kvec)).numpy()
            dmg0[r, s - 1] = (-lr * ((g * D0) @ kvec)).numpy()
            sumD[s - 1] += D.float(); sumM[s - 1] += M.float(); sumG[s - 1] += g.float()
            bc1s[s - 1] = bc1
            th = th_new
    ED = sumD.double() / R; EM = sumM.double() / R; EG = sumG.double() / R
    kEDEM = np.stack([(-(lr / bc1s[s]) * ((ED[s] * EM[s]) @ kvec)).numpy() for s in range(steps)])
    kD0EM = np.stack([(-(lr / bc1s[s]) * ((D0 * EM[s]) @ kvec)).numpy() for s in range(steps)])
    kD0EG = np.stack([(-lr * ((D0 * EG[s]) @ kvec)).numpy() for s in range(steps)])
    # D のラベルによる揺らぎ（座標ごとの変動係数の k 重み中央値）: 2 回目の和が要るので近似として最後の draw と平均の差
    with torch.no_grad():
        for q, s0 in zip(P, saveP): q.copy_(s0)
        for m, s0 in zip(opt.m, saveM): m.copy_(s0)
        for v, s0 in zip(opt.v, saveV): v.copy_(s0)
    opt.t = savet
    return dict(rd_dm=dm, rd_dmf=dmf, rd_dmg0=dmg0, rd_kEDEM=kEDEM, rd_kD0EM=kD0EM, rd_kD0EG=kD0EG, rd_g1mean=(g1 / R).float().numpy(),
                rd_ED200=ED[-1].float().numpy(), rd_ED50=ED[49].float().numpy(), rd_ED10=ED[9].float().numpy())

def run(a):
    torch.manual_seed(a.seed)
    ax = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=stream("rl_subset", a.seed))[:N].numpy()
    X = torch.tensor(ax[subset], dtype=torch.float32)
    Xt = torch.cat([X.double(), torch.ones(N, 1, dtype=torch.float64)], 1)
    kvec = torch.cat([X.double().mean(0), torch.ones(1, dtype=torch.float64)])
    glabel, gbatch = stream("env_labels_0913", a.seed), stream("env_batch_0913", a.seed)
    P = [q.to(dtype=torch.float32).requires_grad_(True) for q in initial(a.seed, (784, H, K))]
    opt = Adam(P, a.lr, 0.9, a.beta2, 1e-8)
    probes = set(int(x) for x in a.probe.split(","))
    out = {}; t0 = time.monotonic()
    for task in range(1, a.tasks + 1):
        y_new = torch.randint(K, (N,), generator=glabel)
        Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32)
        epochs = -(-(a.steps * BATCH) // N)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:a.steps * BATCH].reshape(a.steps, BATCH)
        if task in probes:
            st = switch_state(P, opt, X, Xt, kvec, a.act)
            for kk, vv in st.items():
                out[f"t{task}_{kk}"] = vv.float().numpy() if torch.is_tensor(vv) else vv
            if task in set(int(x) for x in a.redraw_tasks.split(",")):
                rd = redraw(P, opt, X, order, a.act, kvec, a.R, 200, a.seed, task)
                for kk, vv in rd.items(): out[f"t{task}_{kk}"] = vv
            rec, co = window(P, opt, X, Y, order, a.act, kvec, a.steps)
            for kk, vv in rec.items(): out[f"t{task}_{kk}"] = vv.astype(np.float32)
            for kk, vv in co.items(): out[f"t{task}_{kk}"] = vv
        else:
            for it in range(a.steps):
                idx = order[it]; train_step(P, opt, X[idx], Y[idx], a.act); opt.step()
        with torch.no_grad():
            zz = X @ P[0].T + P[1]
            acc = float(((activ(zz, a.act) @ P[2].T + P[3]).argmax(-1) == y_new).float().mean())
        print(f"{a.act} s{a.seed} b2 {a.beta2} T {a.steps} task {task} acc {acc:.3f} pplus {float((zz > 0).float().mean()):.4f} ({time.monotonic() - t0:.0f}s)", flush=True)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / f"{a.act}_s{a.seed}_b2{a.beta2}_T{a.steps}.npz", probes=np.array(sorted(probes)), **out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="ELU"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", type=int, default=12); ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--beta2", type=float, default=0.999)
    ap.add_argument("--probe", default="2,3,6,10"); ap.add_argument("--redraw_tasks", default="3,10"); ap.add_argument("--R", type=int, default=24)
    ap.add_argument("--out", default="runs")
    run(ap.parse_args())
