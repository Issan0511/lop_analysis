"""改訂役: 押しの長さ（転回点）と深さは、旧課題の予測の大きさ（取り消す logit の量）で決まるか — 同じ切替状態からの枝（1 層・ELU）。
箱は導出役の adam_window_probe.py と同じ（本走の状態とビット一致）。切替 t の状態から新課題 1 つを腕ごとに回す。
腕:
  base      そのまま
  ro{c}     切替時に読み出し W2・b2 を c 倍（logit が c 倍）。第 1 層の Adam の状態も m×c・v×c² にして、
            第 1 層の最初の一歩を基準と同じにする（勾配は v に比例するので、Adam の尺度不変で打ち消す）。W2 の状態はそのまま。
  ro{c}f    同じく c 倍にして、読み出しをその課題の間は凍結（取り消しは第 1 層だけ）
  basef     読み出しを凍結（c = 1）
記録（10 更新ごと）: m_i = mean_n z_in の変位、旧ラベルの余裕 mean_n (f_{n,y_old} − mean_k f_{n,k})、logit の SD、新ラベルの正解率。
"""
import argparse, sys, time
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, str(Path(__file__).resolve().parent))
from awp import read_idx, stream, initial, activ, Adam, train_step, DATA, N, BATCH, H, K
torch.set_num_threads(1)

def run_arm(arm, P0, opt0, X, Y, y_new, y_old, order, act, steps):
    P = [q.detach().clone().requires_grad_(True) for q in P0]
    c = 1.0; freeze = arm.endswith("f")
    if arm.startswith("ro"):
        c = float(arm[2:].rstrip("f").replace("p", "."))
    opt = Adam(P, opt0.lr, 0.9, 0.999, 1e-8)
    opt.m = [m.clone() for m in opt0.m]; opt.v = [v.clone() for v in opt0.v]; opt.t = opt0.t
    with torch.no_grad():
        P[2].mul_(c); P[3].mul_(c)
        opt.m[0].mul_(c); opt.m[1].mul_(c); opt.v[0].mul_(c * c); opt.v[1].mul_(c * c)
    mu = X.double().mean(0)
    def probe():
        with torch.no_grad():
            z = X @ P[0].T + P[1]; f = (activ(z, act) @ P[2].T + P[3]).double()
            fc = f - f.mean(1, keepdim=True)
            return ((P[0].double() @ mu + P[1].double()).numpy(), float(fc.gather(1, y_old[:, None]).mean()),
                    float(fc.std(1).mean()), float((f.argmax(1) == y_new).double().mean()))
    W2f, b2f = P[2].detach().clone(), P[3].detach().clone()
    m0, marg0, sd0, _ = probe()
    tr = []; mg = [marg0]; sd = [sd0]; acc = [np.nan]
    for s in range(1, steps + 1):
        idx = order[s - 1]
        train_step(P, opt, X[idx], Y[idx], act)
        opt.step()
        if freeze:
            with torch.no_grad():
                P[2].copy_(W2f); P[3].copy_(b2f)          # 読み出しを厳密に凍結（旧モーメントでも動かさない）
        if s % 10 == 0:
            m, a_, b_, c_ = probe(); tr.append(m - m0); mg.append(a_); sd.append(b_); acc.append(c_)
    return np.array(tr, np.float32), np.array(mg), np.array(sd), np.array(acc)

def main(a):
    torch.manual_seed(a.seed)
    ax = read_idx(DATA / "train-images-idx3-ubyte.gz").reshape(-1, 784).astype(np.float32) / 255
    subset = torch.randperm(len(ax), generator=stream("rl_subset", a.seed))[:N].numpy()
    X = torch.tensor(ax[subset], dtype=torch.float32)
    glabel, gbatch = stream("env_labels_0913", a.seed), stream("env_batch_0913", a.seed)
    P = [q.to(dtype=torch.float32).requires_grad_(True) for q in initial(a.seed, (784, H, K))]
    opt = Adam(P, 1e-3, 0.9, 0.999, 1e-8)
    switches = set(int(x) for x in a.switches.split(",")); arms = a.arms.split(",")
    out = {}; t0 = time.monotonic(); y_old = None
    for task in range(1, max(switches) + 1):
        y_new = torch.randint(K, (N,), generator=glabel)
        Y = torch.nn.functional.one_hot(y_new, K).to(torch.float32)
        epochs = -(-(a.steps * BATCH) // N)
        order = torch.stack([torch.randperm(N, generator=gbatch) for _ in range(epochs)]).reshape(-1)[:a.steps * BATCH].reshape(a.steps, BATCH)
        if task in switches:
            with torch.no_grad():
                z = X @ P[0].T + P[1]
            out[f"t{task}_k"] = (z > 0).sum(0).numpy()
            for arm in arms:
                tr, mg, sd, acc = run_arm(arm, P, opt, X, Y, y_new, y_old, order, a.act, a.steps)
                for nm, v in (("tr", tr), ("mg", mg), ("sd", sd), ("acc", acc)): out[f"t{task}_{arm}_{nm}"] = v
                live = out[f"t{task}_k"] > 0
                mn = tr[:, live].min(0); tmin = (tr[:, live].argmin(0) + 1) * 10
                print(f"{a.act} s{a.seed} 切替 {task} {arm:8s}: 深さ {np.median(mn):+.3f}  転回 {np.median(tmin):.0f}  課題末 {np.median(tr[-1, live]):+.3f}  "
                      f"余裕 {mg[0]:+.2f}→{mg[20]:+.2f}(200)→{mg[-1]:+.2f}  logitSD {sd[0]:.2f}  acc末 {acc[-1]:.3f} ({time.monotonic() - t0:.0f}s)", flush=True)
        if task == max(switches): break
        for it in range(a.steps):
            idx = order[it]; train_step(P, opt, X[idx], Y[idx], a.act); opt.step()
        y_old = y_new
    Path(a.out).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(a.out) / f"bscale_{a.act}_s{a.seed}_sw{a.switches.replace(',', '-')}.npz", **out)

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--act", default="ELU"); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=4000); ap.add_argument("--switches", default="10,20")
    ap.add_argument("--arms", default="base,ro0p5,ro2,basef,ro0p5f,ro2f"); ap.add_argument("--out", default="branches_scale")
    main(ap.parse_args())
