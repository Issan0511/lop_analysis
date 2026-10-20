#!/usr/bin/env python3
"""ρ の分割の追加 2 項目（事後・判定外、Issa の依頼を Neokunekune の fork 経由、1021 未明）。

1. unit ごとの κ_R,j（kappa_reload_cnn_1020 の定義をそのまま、感度は網の全経路で取る）
   R_{n·j} = K_nj ∂l_n/∂z_nj（K_nj = y_n·ȳ、l = ロジット、z = FFN の前活性。位置をまたぐ影響は逆伝播がまとめて取る）
   κ_R,j = Σ_n ⟨R_nj, y_n − 1/K⟩ / (N(1 − 1/K))、ε_j = Σ_n ⟨R_nj, π_n − y_n⟩ / N、G_own,j = Σ_n ⟨R_nj, π_n − 1/K⟩ / N
   恒等式 G_own = (1 − 1/K) κ_R + ε（Σ_c (y − 1/K) = 0 なのでクラスの中心化は不要）。G_own > 0 が沈める側。
   y_n（古いラベル）は前の課題 k の連鎖の遷移確率 P_k(· | x_{n−1}, x_n)（入力は新しい課題 k+1 の probe）。新しい課題の期待は一様ラベル。
   位置 n = 1..T−2 を使う（学習の損失と同じ位置）。
2. L2 の有効な読み出しを「直通」（FFN3 の出力を定数とみなす。注意 3 を通る分は直通に入れる）と「L3 経由」（全体 − 直通）に割る。
   ‖s‖ は ∂l/∂a（a = 活性化の出力）の Frobenius ノルムを Hutchinson（ガウスの 16 本）で推定。κ_R も直通・L3 経由で別に出す。
3. L3 の非線形の残り: FFN3 の出力を入力 y の線形写像（切片つき）で最小二乗に当てた残差の分散の割合（固定 probe = 一様乱数の probe）。
  python analysis/chargpt_cl_1019_quicklook/kappa_split.py --arm SiLU --state <ckpt> --task <k+1> [--state_b <u100 ckpt>] --out <json>
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT / "src"))
import chargpt_cl_1019_kune as K  # noqa: E402
C = K.C
SEED, ALPHA = 1000, 0.01


def build(arm, sd, dev):
    m = C.build_model(arm, "M", SEED, "cpu"); m.load_state_dict(sd); m = m.double().to(dev); m.eval()
    for p in m.parameters(): p.requires_grad_(False)
    return m


def forward(m, X, detach_ffn3=False):
    """ロジット・各層の z（前活性）・a（活性化の出力）・y（LN2 の出力）を返す。detach_ffn3 で最後の FFN の出力を定数に。"""
    zs, as_, ys = [], [], []
    h = m.wte(X)
    nl = len(m.h)
    for l, b in enumerate(m.h):
        h = h + b.sc * b.attn(b.n1(h))
        y = b.n2(h); z = b.mlp.up(y); z.retain_grad() if z.requires_grad else None
        a = b.mlp.phi(z); a.retain_grad() if a.requires_grad else None
        f = b.mlp.down(a)
        if detach_ffn3 and l == nl - 1: f = f.detach()
        h = h + f; zs.append(z); as_.append(a); ys.append(y)
    return m.head(m.nf(h)), zs, as_, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True); ap.add_argument("--state", required=True); ap.add_argument("--state_b", default=None)
    ap.add_argument("--task", type=int, required=True); ap.add_argument("--nseq", type=int, default=32); ap.add_argument("--device", default="cuda")
    ap.add_argument("--hutch", type=int, default=16); ap.add_argument("--out", required=True)
    a = ap.parse_args(); dev = torch.device(a.device); t = a.task
    m = build(a.arm, torch.load(a.state, map_location="cpu"), dev)
    for p in m.parameters(): p.requires_grad_(True)
    X = C.task_cpu_M(SEED, t, 10, ALPHA)["probe"][:a.nseq].to(dev)
    Pold = torch.from_numpy(C.ChainM(SEED, t - 1, ALPHA).P).to(dev)                    # 前の課題の遷移表 [V, V, V]
    nseq, T = X.shape; KC = C.V
    pos = torch.zeros(nseq, T, dtype=torch.bool, device=dev); pos[:, 1:T - 1] = True     # 学習の損失と同じ位置
    Yold = torch.zeros(nseq, T, KC, dtype=torch.float64, device=dev); Yold[:, 1:] = Pold[X[:, :-1], X[:, 1:]]
    N = int(pos.sum())
    out = dict(arm=a.arm, task=t, nseq=a.nseq, layers={})
    logits, zs, as_, ys = forward(m, X)
    pi = logits.softmax(-1).detach()
    mask = pos[..., None].double()
    targets = {"kR": (Yold - 1.0 / KC) * mask, "eps": (pi - Yold) * mask, "G": (pi - 1.0 / KC) * mask}
    grads = {}
    for k, w in targets.items():
        grads[k] = torch.autograd.grad((logits * w).sum(), zs, retain_graph=True)
    D = N * (1 - 1.0 / KC)
    for l in range(len(m.h)):
        y = ys[l].detach().reshape(-1, C.D); Kn = y @ y.mean(0)
        g = {k: grads[k][l].detach().reshape(-1, C.FF) for k in targets}
        kR = (Kn[:, None] * g["kR"]).sum(0) / D; eps = (Kn[:, None] * g["eps"]).sum(0) / N; G = (Kn[:, None] * g["G"]).sum(0) / N
        W = m.h[l].mlp.up.weight.detach(); zd = (y @ W.T); alive = (zd > 0).any(0)
        out["layers"][str(l)] = dict(kappa_R=kR.tolist(), eps=eps.tolist(), G_own=G.tolist(), alive=alive.tolist(), sigma=zd.std(0).tolist(),
                                     check_identity=float(((1 - 1.0 / KC) * kR + eps - G).abs().max() / (G.abs().max() + 1e-300)))
    # 2. L2（読み出しの 1 つ手前の FFN）の読み出しを直通と L3 経由に割る
    nl = len(m.h)
    if nl >= 2:
        l2 = nl - 2
        logits_d, zs_d, as_d, ys_d = forward(m, X, detach_ffn3=True)
        gd = torch.autograd.grad((logits_d * targets["kR"]).sum(), zs_d[l2], retain_graph=True)[0].reshape(-1, C.FF)
        y2 = ys[l2].detach().reshape(-1, C.D); Kn2 = y2 @ y2.mean(0)
        kR_dir = (Kn2[:, None] * gd).sum(0) / D; kR_all = torch.tensor(out["layers"][str(l2)]["kappa_R"], device=dev, dtype=torch.float64)
        gen = torch.Generator(device="cpu").manual_seed(0)
        sq_all = torch.zeros(nseq * T, C.FF, dtype=torch.float64, device=dev); sq_dir = torch.zeros_like(sq_all); sq_via = torch.zeros_like(sq_all)
        for _ in range(a.hutch):
            r = (torch.randn(nseq, T, KC, generator=gen, dtype=torch.float64).to(dev)) * mask
            ga = torch.autograd.grad((logits * r).sum(), as_[l2], retain_graph=True)[0].reshape(-1, C.FF)
            gda = torch.autograd.grad((logits_d * r).sum(), as_d[l2], retain_graph=True)[0].reshape(-1, C.FF)
            sq_all += ga ** 2; sq_dir += gda ** 2; sq_via += (ga - gda) ** 2
        n_all = (sq_all.sum(0) / a.hutch).sqrt(); n_dir = (sq_dir.sum(0) / a.hutch).sqrt(); n_via = (sq_via.sum(0) / a.hutch).sqrt()
        out["L2_split"] = dict(layer=l2, kappa_R_direct=kR_dir.tolist(), kappa_R_viaL3=(kR_all - kR_dir).tolist(),
                               s_norm_all=n_all.tolist(), s_norm_direct=n_dir.tolist(), s_norm_viaL3=n_via.tolist())
    # 3. L3 の非線形の残り（固定 probe = 一様乱数の probe、最後の FFN）
    with torch.no_grad():
        U = C.uniform_probe_M(SEED, "cpu")[:a.nseq].to(dev)
        h = m.wte(U)
        for l, b in enumerate(m.h):
            h = h + b.sc * b.attn(b.n1(h)); y = b.n2(h); f = b.mlp(y)
            if l == nl - 1:
                Yy = torch.cat([y.reshape(-1, C.D), torch.ones(y.numel() // C.D, 1, dtype=torch.float64, device=dev)], 1); Ff = f.reshape(-1, C.D)
                A = torch.linalg.lstsq(Yy, Ff).solution; res = Ff - Yy @ A
                out["L3_nonlin_resid"] = float((res ** 2).sum() / ((Ff - Ff.mean(0)) ** 2).sum())
            h = h + f
    # 4. 窓の実際の Δm_j（課題 k の末 → 課題 k+1 の更新 100、同じ probe）
    if a.state_b:
        mb = build(a.arm, torch.load(a.state_b, map_location="cpu"), dev)
        with torch.no_grad():
            _, za, _, ya_ = forward(build(a.arm, torch.load(a.state, map_location="cpu"), dev), X)
            _, zb, _, yb_ = forward(mb, X)
        for l in range(len(m.h)):
            dmu = (zb[l].reshape(-1, C.FF).mean(0) - za[l].reshape(-1, C.FF).mean(0)) / (za[l].reshape(-1, C.FF).std(0) + 1e-30)
            out["layers"][str(l)]["dmu_window"] = dmu.tolist()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); Path(a.out).write_text(json.dumps(out))
    print("written", a.out, {l: f"{v['check_identity']:.1e}" for l, v in out["layers"].items()}, out.get("L3_nonlin_resid"))


if __name__ == "__main__":
    main()
