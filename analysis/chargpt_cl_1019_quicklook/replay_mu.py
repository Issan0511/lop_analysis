#!/usr/bin/env python3
"""課題 T の末（重み＋Adam）から次の課題を N 更新だけ再生し、層ごとの μ/σ の動きを 自分の重み（Δw·ȳ）と 上流（w·Δȳ＋交差）に割る（下見・未登録）。
  python analysis/chargpt_cl_1019_quicklook/replay_mu.py --arm SiLU --ckpt <dir> --task 50 --updates 1000
probe は次の課題（T+1）の chain probe で固定。100 更新ごとに、unit 平均の Δμ/σ（窓の始まりの σ）と、入力の担い手の割合 ‖ȳ‖/√trΣ、L の中央値を出す。"""
import argparse, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C


def ffn_io(m, probe):
    cap = []
    with torch.no_grad(): m.body(probe, cap)
    out = []
    for l, blk in enumerate(m.h):
        y = blk.n2(cap[2*l]).reshape(-1, C.D).double(); z = cap[2*l+1].reshape(-1, C.FF).double()
        out.append(dict(W=blk.mlp.up.weight.double().clone(), ybar=y.mean(0), tr=float(torch.cov(y.T).trace()), mu=z.mean(0), sd=z.std(0)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--arm", default="SiLU"); ap.add_argument("--ckpt", required=True); ap.add_argument("--task", type=int, required=True)
    ap.add_argument("--updates", type=int, default=1000); ap.add_argument("--seed", type=int, default=1000); ap.add_argument("--threads", type=int, default=4)
    a = ap.parse_args(); torch.set_num_threads(a.threads); d = Path(a.ckpt)
    m = C.build_model(a.arm, "M", a.seed, "cpu"); m.load_state_dict(torch.load(d / f"task{a.task}.pt", map_location="cpu")); m.train()
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0, foreach=True)
    opt.load_state_dict(torch.load(d / f"task{a.task}_adam.pt", map_location="cpu"))
    for g in opt.param_groups: g["lr"] = 1e-3; g["capturable"] = False
    nxt = a.task + 1; data = C.task_cpu_M(a.seed, nxt, a.updates, 0.01)["data"]; probe = C.task_cpu_M(a.seed, nxt, 10, 0.01)["probe"]
    m.eval(); A = ffn_io(m, probe); m.train()
    print(f"== {a.arm} 課題 {a.task} の末 → 課題 {nxt} を {a.updates} 更新再生（probe は課題 {nxt}）")
    print("  更新 | 層: Δμ/σ 累積（自分, 上流） | μ/σ 中央 | 担い手 ‖ȳ‖/√trΣ")
    losses = []
    for u in range(a.updates):
        x = data[u][:, :C.T].long(); y = data[u][:, 1:C.T+1].long()
        out = m(x); loss = F.cross_entropy(out[:, 1:].reshape(-1, C.V), y[:, 1:].reshape(-1)); opt.zero_grad(set_to_none=True); loss.backward(); opt.step(); losses.append(float(loss))
        if (u + 1) in (100, 300, 600, 1000, 1500, 2000) or u + 1 == a.updates:
            m.eval(); B = ffn_io(m, probe); m.train(); s = f"  u{u+1:5d} 損失 {np.mean(losses[-100:]):.2f} |"
            for l in range(4):
                x0, x1 = A[l], B[l]; dW = x1["W"] - x0["W"]; dy = x1["ybar"] - x0["ybar"]
                own = float(((dW @ x0["ybar"]) / x0["sd"]).mean()); up = float((((x0["W"] @ dy) + (dW @ dy)) / x0["sd"]).mean())
                s += f" L{l}: {own+up:+.2f} ({own:+.2f}, {up:+.2f}) r {float((x1['mu']/x1['sd']).median()):+.2f} 担 {float(x1['ybar'].norm()/np.sqrt(x1['tr'])):.2f} |"
            print(s, flush=True)


if __name__ == "__main__":
    main()
