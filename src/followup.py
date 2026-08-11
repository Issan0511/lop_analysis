"""Part A 追加解析用の拡張凍結測定 (task_074)。

  python -m src.followup results/drift_0809

既存チェックポイント (`<results>/ckpts/*.pt`) から、初回スイープと**同一の測定乱数系列**で
凍結測定を再走し、初回は保存していなかった生の推定量を取り出す:

  - Ê[g_W] (ニューロン別入力重み勾配の期待値) 本体       -> A1 (ニューロン間 pairwise cos)
                                                          -> A4 (ckpt 間の方向相関)
  - 奇数番/偶数番サンプル別の Ê[g_W] と µ̂                -> A6 (split-half 検定)
  - 非定常周期 τ ごとの周期内平均勾配 ḡ_τ                 -> A5 (方向分散)
  - 各ニューロンの dead 判定 (eval バッチ上, [J] App.B)   -> A2 (cos の符号と生死の対応)

再学習は不要。測定用 generator は freeze.py と同一 (offset=999000+step) なので、
サンプル列は初回 `freeze_*.csv` を生成したときと完全に一致する。

config は対象ディレクトリの `config_used.yaml` を読む (その実行を再現するのが目的なので)。

出力: <results>/followup_Eg_{exp}_w{width}_step{step}.npz (解析は followup_analysis.py)
"""
import argparse
import json
import os
import time

import numpy as np
import torch

from .common import load_config, pick_device, build_runs, group_runs, group_name
from .freeze import SEG, _restore, _ema_toeplitz
from .train import make_gens

P_MAX = 64          # 周期ビンの上限 (freeze_min_periods=50 + 端数)


def per_neuron_dead(exp, width, cfg, env, teacher, net, rm, cmask, device):
    """train.py の eval バッチ再現による per-neuron dead 判定 [R,h] (bool)。

    eval_fixed は学習時と同じ generator (offset=0 の 'eval') から同順で引くため、
    lop_metrics.csv の dead_frac と整合する。
    """
    C, A = cfg["common"], cfg["condA"]
    N = C["eval_batch"]
    gens = make_gens(exp, width, device)
    with torch.no_grad():
        if exp == "A":
            ef = torch.randint(0, 2, (N, A["m"] - A["f"]), generator=gens["eval"],
                               device=device).float()
            flip = env.flip_state[None].expand(N, -1, -1)
            x = torch.cat([flip, ef[:, None, :].expand(-1, env.R, -1)], dim=2)
        else:
            ef = torch.randn(N, env.d, generator=gens["eval"], device=device)
            x = env.mu[None] + ef[:, None, :]
        x_in = x - cmask[None] * rm[None]
        _, a, _ = net.forward_batch(x_in)
        finite = a.isfinite().all(dim=2).all(dim=0)                      # [R]
        a = torch.where(finite[None, :, None], a, torch.zeros_like(a))
        dead = (a.abs() < C["dead_tol"]).float().mean(dim=0) > C["dead_tau"]   # [R,h]
    return dead, finite


def followup_measure(gkey, ckpt_path, cfg, device):
    """1 チェックポイントの拡張凍結測定。dict of numpy arrays を返す。"""
    C = cfg["common"]
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    step = ckpt["step"]
    runs, env, teacher, net, rm0, centered, period, d = _restore(gkey, ckpt, cfg, device)
    exp, width = gkey[0], gkey[1]
    R, h = len(runs), width
    alpha = cfg["condA"]["center_alpha"]
    cmask = centered[:, None].float()

    # --- A2: 測定でenvを進める前に dead を判定する
    dead_i, finite = per_neuron_dead(exp, width, cfg, env, teacher, net, rm0, cmask, device)

    M_r = torch.minimum(C["freeze_min_periods"] * period,
                        torch.tensor(C["freeze_M_cap"])).to(device)      # [R]
    M_max = int(M_r.max().item())
    decay_full, L_full = _ema_toeplitz(alpha, SEG, device)
    per_dev = period.to(device)
    t0 = env.t                                    # 測定開始時刻 (絶対 step)
    bin0 = t0 // per_dev                          # [R]

    z = lambda *s: torch.zeros(*s, device=device)
    sW = z(R, h, d)                                # 全サンプル
    sW_o, sW_e = z(R, h, d), z(R, h, d)            # 奇数番 / 偶数番 (A6)
    s_x, s_x_o, s_x_e = z(R, d), z(R, d), z(R, d)
    sb, sv = z(R, h), z(R, h)
    sW_per = z(R, P_MAX, h, d)                     # 周期別 (A5)
    cnt_per = z(R, P_MAX)
    ridx = torch.arange(R, device=device)

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
        gW, gb, gv, _ = net.grads_batch(x_in, pre, a, delta)

        sidx = s + torch.arange(Cs, device=device)                     # [Cs]
        w = (sidx[:, None] < M_r[None, :]).float()                     # [Cs,R]
        par = (sidx % 2).float()[:, None]                              # [Cs,1] 1=奇数番
        w_o, w_e = w * par, w * (1 - par)

        sW += (gW * w[..., None, None]).sum(0)
        sW_o += (gW * w_o[..., None, None]).sum(0)
        sW_e += (gW * w_e[..., None, None]).sum(0)
        sb += (gb * w[..., None]).sum(0)
        sv += (gv * w[..., None]).sum(0)
        s_x += (xseg * w[..., None]).sum(0)
        s_x_o += (xseg * w_o[..., None]).sum(0)
        s_x_e += (xseg * w_e[..., None]).sum(0)

        # 周期ビン: 全周期は SEG=100 の倍数かつ t0 は 100 の倍数なので
        # 1 セグメントは必ず単一の周期内に収まる
        b = ((t0 + s) // per_dev - bin0).clamp(0, P_MAX - 1)            # [R]
        sW_per[ridx, b] += (gW * w[..., None, None]).sum(0)
        cnt_per[ridx, b] += w.sum(0)
        s += Cs

    M = M_r.float()
    Eg_W = sW / M[:, None, None]
    n_o = torch.div(M_r, 2, rounding_mode="floor").float()   # 0..M-1 の奇数番個数
    n_e = M - n_o
    Eg_W_o = sW_o / n_o.clamp_min(1)[:, None, None]
    Eg_W_e = sW_e / n_e.clamp_min(1)[:, None, None]
    mu = s_x / M[:, None]
    mu_o = s_x_o / n_o.clamp_min(1)[:, None]
    mu_e = s_x_e / n_e.clamp_min(1)[:, None]

    gbar = sW_per / cnt_per.clamp_min(1)[:, :, None, None]             # [R,P,h,d]
    gbar_norm = gbar.flatten(2).norm(dim=2)                            # [R,P] 全ニューロン結合
    gbar_norm_i = gbar.norm(dim=3)                                     # [R,P,h] ニューロン別

    mu_true = env.mu if exp == "B" else torch.cat(
        [env.flip_state, 0.5 * torch.ones(R, d - env.f, device=device)], 1)

    np_ = lambda t: t.detach().cpu().numpy().astype(np.float32)
    return dict(
        step=np.int64(step),
        run_ids=np.array([r["run_id"] for r in runs]),
        Eg_W=np_(Eg_W), Eg_b=np_(sb / M[:, None]), Eg_v=np_(sv / M[:, None]),
        Eg_W_odd=np_(Eg_W_o), Eg_W_even=np_(Eg_W_e),
        mu_inter=np_(mu), mu_odd=np_(mu_o), mu_even=np_(mu_e), mu_true=np_(mu_true),
        gbar_norm=np_(gbar_norm), gbar_norm_i=np_(gbar_norm_i), gbar_count=np_(cnt_per),
        dead=dead_i.detach().cpu().numpy(), finite=finite.detach().cpu().numpy(),
        W=np_(net.W), v=np_(net.v), M=np_(M), period=np_(per_dev.float()),
    )


def run_group(gkey, cfg, device, outdir, steps=None):
    gname = group_name(gkey)
    steps = steps if steps is not None else cfg["common"]["checkpoints"]
    done = []
    for step in steps:
        p = os.path.join(outdir, "ckpts", f"{gname}_step{step}.pt")
        if not os.path.exists(p):
            continue
        t0 = time.time()
        out = followup_measure(gkey, p, cfg, device)
        dst = os.path.join(outdir, f"followup_Eg_{gname}_step{step}.npz")
        np.savez_compressed(dst, **out)
        print(f"    {gname} step={step} -> {os.path.basename(dst)} "
              f"({time.time()-t0:.1f}s)", flush=True)
        done.append(step)
    return done


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", help="実験ディレクトリ (例: results/drift_0809)")
    ap.add_argument("--groups", nargs="*", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", nargs="*", type=int, default=None)
    args = ap.parse_args()

    cfg = load_config(os.path.join(args.results, "config_used.yaml"))
    cfg["common"]["device"] = args.device
    device = pick_device(cfg)
    groups = group_runs(build_runs(cfg))

    meta = {}
    for gkey in groups:
        gname = group_name(gkey)
        if args.groups and gname not in args.groups:
            continue
        print(f"=== followup measure {gname}", flush=True)
        meta[gname] = run_group(gkey, cfg, device, args.results, steps=args.steps)
    mpath = os.path.join(args.results, "followup_meta.json")
    old = json.load(open(mpath)) if os.path.exists(mpath) else {}
    old.update(meta)
    json.dump(old, open(mpath, "w"), indent=1)
    print("FOLLOWUP MEASURE DONE", flush=True)


if __name__ == "__main__":
    main()
