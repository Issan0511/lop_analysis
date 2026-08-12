"""重み凍結法による E[g] 推定 (仕様書 §2, §3)。

チェックポイントで theta を凍結し、非定常性 (A: flipping bits / B: 教師再サンプル) だけを
動かしながらサンプル勾配 g^(1)..g^(M) を採取する。batch=1 なのでサンプル勾配 = ミニバッチ勾配。

実装メモ:
- 周期境界 (全 T, K は 100 の倍数) でセグメントを切り、セグメント内 (環境定数) は
  時間方向に完全ベクトル化して閉形式勾配を積算する。
- 中心化系列の走行平均 EMA はセグメント内でも正確に逐次値を再現する
  (下三角 Toeplitz 行列による厳密展開)。
- 乱数 generator は測定専用に分離 (学習系列とは独立)。
- cos の基準 µ̂ は生入力 x_raw の平均 (潮流)。µ̂_intra は A: 測定終了時の flip_state + 0.5
  (周期内平均の解析値), B: 真の µ。µ̂_inter は測定窓全体の経験平均。
- Ĉov(e·1_i, x) は学習器が実際に見る入力 x_in で計算 (標準系列では x_raw と同一)。
"""
import os
import csv
import math
import torch

from .common import group_name
from .envs import SCREnv, LTUTarget, GaussEnv, MLPTeacher
from .nets import VecMLP
from .train import make_gens

SEG = 100  # セグメント長 = 全周期値の最大公約数


def _restore(gkey, ckpt, cfg, device):
    exp, width = gkey[0], gkey[1]
    runs = ckpt["runs"]
    R = len(runs)
    A, B = cfg["condA"], cfg["condB"]
    gens = make_gens(exp, width, device, offset=999_000 + ckpt["step"])  # 測定専用乱数
    period = torch.tensor([r["period"] for r in runs], dtype=torch.long)

    if exp == "A":
        d = A["m"]
        env = SCREnv(R, A["m"], A["f"], period, gens["input"], device)
        teacher = LTUTarget(R, A["m"], A["target_hidden"], A["beta"], gens["teacher"], device)
    else:
        d = B["d"]
        cvals = [r["c"] for r in runs]
        kvals = [r.get("kappa", 1) for r in runs]   # 旧 ckpt (kappa なし) は等方で復元
        env = GaussEnv(R, d, cvals, gens["input"], device, kappa=kvals,
                       spike_dir=B.get("spike_dir", "ones"))
        teacher = MLPTeacher(R, width, d, period, gens["teacher"], device)

    # 状態復元 (教師 A は ckpt に完全保存されている; B は再サンプルで進むので状態から再開)
    env.load_state({k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ckpt["env"].items()})
    tstate = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in ckpt["teacher"].items()}
    teacher.load_state(tstate)

    net = VecMLP(R, width, d, gens["init"], device)
    net.load_state({k: v.to(device) for k, v in ckpt["net"].items()})
    rm = ckpt["running_mean"].to(device)
    centered = torch.tensor([r["enc"] == "centered" for r in runs], device=device)
    return runs, env, teacher, net, rm, centered, period, d


def _ema_toeplitz(alpha, C, device):
    """rm_prev[c] = decay[c]*rm0 + sum_{j<c} L[c,j] x[j] の係数。"""
    j = torch.arange(C, device=device)
    Cm = j[:, None] - 1 - j[None, :]
    L = alpha * (1 - alpha) ** Cm.clamp_min(0).float()
    L = L * (j[None, :] < j[:, None]).float()
    decay = (1 - alpha) ** j.float()
    return decay, L


def freeze_measure(gkey, ckpt_path, cfg, device):
    """1 チェックポイントの凍結測定。global 行と neuron 行のリストを返す。"""
    C = cfg["common"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt["step"]
    runs, env, teacher, net, rm0, centered, period, d = _restore(gkey, ckpt, cfg, device)
    exp, width = gkey[0], gkey[1]
    R, h = len(runs), width
    alpha = cfg["condA"]["center_alpha"]
    cmask = centered[:, None].float()

    M_r = torch.minimum(C["freeze_min_periods"] * period,
                        torch.tensor(C["freeze_M_cap"])).to(device)      # [R]
    M_max = int(M_r.max().item())
    decay_full, L_full = _ema_toeplitz(alpha, SEG, device)

    z = lambda *s: torch.zeros(*s, device=device)
    sW, sW2 = z(R, h, d), z(R, h, d)
    sb, sb2 = z(R, h), z(R, h)
    sv, sv2 = z(R, h), z(R, h)
    sc, sc2 = z(R), z(R)
    s_adelta, s_e1, s_e1x = z(R, h), z(R, h), z(R, h, d)
    s_x, s_xin = z(R, d), z(R, d)

    rm = rm0.clone()
    s = 0
    while s < M_max:
        Cs = min(SEG, M_max - s)
        if exp == "B":
            teacher.t = env.t
            teacher.maybe_resample()
        xseg = env.segment(Cs)                                   # [Cs,R,d]
        y = teacher(xseg)                                        # [Cs,R]

        decay, L = decay_full[:Cs], L_full[:Cs, :Cs]
        rm_prev = decay[:, None, None] * rm[None] + torch.einsum("cj,jrd->crd", L, xseg)
        x_in = xseg - cmask[None] * rm_prev
        Lend = alpha * (1 - alpha) ** torch.arange(Cs - 1, -1, -1, device=device).float()
        rm = (1 - alpha) ** Cs * rm + torch.einsum("j,jrd->rd", Lend, xseg)

        pre, a, yhat = net.forward_batch(x_in)
        delta = yhat - y
        gW, gb, gv, gc = net.grads_batch(x_in, pre, a, delta)

        w = ((s + torch.arange(Cs, device=device))[:, None] < M_r[None, :]).float()  # [Cs,R]
        w1, w2, w3 = w[..., None], w[..., None, None], w
        sW += (gW * w2).sum(0);   sW2 += (gW ** 2 * w2).sum(0)
        sb += (gb * w1).sum(0);   sb2 += (gb ** 2 * w1).sum(0)
        sv += (gv * w1).sum(0);   sv2 += (gv ** 2 * w1).sum(0)
        sc += (gc * w3).sum(0);   sc2 += (gc ** 2 * w3).sum(0)
        s_adelta += (a * delta[..., None] * w1).sum(0)
        gate = (pre > 0).float()
        ew = delta * w
        s_e1 += (ew[..., None] * gate).sum(0)
        s_e1x += torch.einsum("crh,crd->rhd", ew[..., None] * gate, x_in)
        s_x += (xseg * w1).sum(0)
        s_xin += (x_in * w1).sum(0)
        s += Cs

    M = M_r.float()
    mW, mb, mv, mc = sW / M[:, None, None], sb / M[:, None], sv / M[:, None], sc / M
    vW = (sW2 / M[:, None, None] - mW ** 2).clamp_min(0)
    vb = (sb2 / M[:, None] - mb ** 2).clamp_min(0)
    vv = (sv2 / M[:, None] - mv ** 2).clamp_min(0)
    vc = (sc2 / M - mc ** 2).clamp_min(0)

    def snr(m_norm2, v_sum):
        return (m_norm2.sqrt() / v_sum.clamp_min(1e-30).sqrt())

    nW, nb, nv, nc = (mW ** 2).sum((1, 2)), (mb ** 2).sum(1), (mv ** 2).sum(1), mc ** 2
    tW, tb, tv, tc = vW.sum((1, 2)), vb.sum(1), vv.sum(1), vc
    snr_all = snr(nW + nb + nv + nc, tW + tb + tv + tc)

    mu_inter = s_x / M[:, None]                                  # 生入力の窓平均
    if exp == "A":
        f = env.f
        mu_intra = torch.cat([env.flip_state, 0.5 * torch.ones(R, d - f, device=device)], 1)
    else:
        mu_intra = env.mu
    mu_in_hat = torch.nn.functional.normalize(mu_inter, dim=1, eps=1e-12)

    def cosine(vec, ref):                                        # vec [R,h,d], ref [R,d]
        rn = torch.nn.functional.normalize(ref, dim=1, eps=1e-12)
        vn = torch.nn.functional.normalize(vec, dim=2, eps=1e-12)
        return torch.einsum("rhd,rd->rh", vn, rn)

    cos_intra = cosine(mW, mu_intra)
    cos_inter = cosine(mW, mu_inter)
    if exp == "B":                                               # スパイク方向 u との整列
        cos_spike = cosine(mW, env.u[None].expand(R, d))
    else:
        cos_spike = torch.full((R, h), float("nan"), device=device)
    snr_i = (mW ** 2).sum(2).sqrt() / vW.sum(2).clamp_min(1e-30).sqrt()
    E_adelta = s_adelta / M[:, None]
    cov = s_e1x / M[:, None, None] - (s_e1 / M[:, None])[..., None] * (s_xin / M[:, None])[:, None, :]
    cov_norm = cov.norm(dim=2)
    cov_proj = torch.einsum("rhd,rd->rh", cov, mu_in_hat)
    w_norm = net.W.norm(dim=2)

    n_periods = (M_r.float() / period.to(device).float())
    floor = 1.0 / M.sqrt()

    g_rows, n_rows = [], []
    for i, r in enumerate(runs):
        g_rows.append(dict(run_id=r["run_id"], ckpt=step, M=int(M_r[i]),
                           n_periods=float(n_periods[i]), noise_floor=float(floor[i]),
                           snr_all=float(snr_all[i]),
                           snr_W=float(snr(nW, tW)[i]), snr_b=float(snr(nb, tb)[i]),
                           snr_v=float(snr(nv, tv)[i]), snr_c=float(snr(nc, tc)[i])))
        for j in range(h):
            n_rows.append(dict(run_id=r["run_id"], ckpt=step, neuron=j,
                               w_norm=float(w_norm[i, j]), snr_i=float(snr_i[i, j]),
                               cos_intra=float(cos_intra[i, j]), cos_inter=float(cos_inter[i, j]),
                               cos_spike=float(cos_spike[i, j]),
                               E_adelta=float(E_adelta[i, j]), sign_v=float(torch.sign(net.v[i, j])),
                               cov_norm=float(cov_norm[i, j]), cov_proj=float(cov_proj[i, j])))
    return g_rows, n_rows


def run_freeze_all(gkey, cfg, device, outdir, ckpt_steps=None):
    gname = group_name(gkey)
    steps = ckpt_steps if ckpt_steps is not None else cfg["common"]["checkpoints"]
    g_all, n_all = [], []
    for step in steps:
        p = os.path.join(outdir, "ckpts", f"{gname}_step{step}.pt")
        if not os.path.exists(p):
            continue
        g, n = freeze_measure(gkey, p, cfg, device)
        g_all += g
        n_all += n
    for name, rows in [("freeze_global", g_all), ("freeze_neurons", n_all)]:
        if not rows:
            continue
        path = os.path.join(outdir, f"{name}_{gname}.csv")
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
    return len(g_all), len(n_all)
