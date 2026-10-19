#!/usr/bin/env python3
"""chargpt_cl_1019 の事後解析 1（未登録）: Transformer の FFN で沈降の担い手を測る。

  python src/chargpt_cl_1019_posthoc_carrier.py --root results/chargpt_cl_1019/crosscheck_white-san/M \
      --arms C3 C4 B3 --out results/chargpt_cl_1019/posthoc_carrier_white-san

MLP の箱の記号に合わせる（κ の再装填 1020 のノート）。層 l の FFN の入力を y_n（LN の affine の後）とする。
  位置 m_i = w_i·ȳ + b_i（ȳ = probe のトークン平均）、担い手 K_n = ȳ·y_n（bias がある腕は +1）。
  期待の押し S^E_i = Σ_n K_n δ^E_in。δ^E は新ラベルを一様と置いた CE の勾配（∂L/∂logit = π − 1/V）を z まで逆伝播したもの。
  実ラベルの押し S^GD_i = Σ_n K_n δ_in。δ は新課題の実際の目標での CE の勾配。
  どちらも S > 0 が沈める側（全バッチ GD の自分の重みの一歩で Δm_i = −η S_i / N）。
  κ（最終層だけ）: 実効読み出し v[c,i] = W_head diag(γ_f) J̄_LN W_down[:,i]（クラスで中心化）を v⁰ + κF に最小二乗で当てる。
  F[c,i] = Σ_m (y_mc − 1/V) φ_im は課題 t のラベルと課題 t の末の特徴で作る。v⁰ は初期値の網の同じ量。
  自己形 S_self_i = κ(1 − 1/V) Σ_n K_n φ′_in φ_in（ReLU では φ′φ = φ）。
実測は、課題 t+1 の更新 0 と 100 の μ_i の差（Adam・全パラメータ・100 更新、unit_stats.npz）。
状態は課題 t の末の checkpoint（= 課題 t+1 への切替の時点）。Adam の状態は無いので、押しは全バッチ GD の一歩で置く。
"""
import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chargpt_cl_1019 as C  # noqa: E402

ALPHA, N_M, SEED = 0.01, 10000, 0
SWITCH_FROM = (1, 10, 25)
GEOM_AT = ("init", 1, 10, 25, 50)
N_BATCH = 20                      # 新課題の最初の 20 バッチ（640 系列）で押しを平均する


def load(arm, d, tag):
    m = C.build_model(arm, "M", SEED, "cpu")
    name = "init.pt" if tag == "init" else f"task{tag}.pt"
    m.load_state_dict(torch.load(d / "ckpts" / name, map_location="cpu"))
    return m.eval()


def ffn_inputs(model, x):
    """層ごとの FFN の入力 y（LN の affine の後）・z・LN の正規化後 x̂ を返す（勾配なし）。"""
    cap = []
    with torch.no_grad():
        model.body(x, cap)
    out = []
    for li, blk in enumerate(model.h):
        xin, z = cap[2 * li], cap[2 * li + 1]
        y = blk.n2(xin)
        xhat = F.layer_norm(xin, (C.D,), eps=blk.n2.eps) if isinstance(blk.n2, nn.LayerNorm) else xin
        out.append((y.reshape(-1, C.D).double(), z.reshape(-1, C.FF).double(), xhat.reshape(-1, C.D).double()))
    return out


def has_bias(arm):
    return C.ARMS[arm]["lin_bias"]


def geometry(model, probe, arm):
    rows = []
    for li, (y, z, xhat) in enumerate(ffn_inputs(model, probe)):
        ybar = y.mean(0)
        spread = (y - ybar).pow(2).sum(1).mean().sqrt()
        K = y @ ybar + (1.0 if has_bias(arm) else 0.0)
        mhat = xhat.mean(0)
        shared = float(mhat.pow(2).sum() / xhat.pow(2).sum(1).mean())
        rows.append(dict(layer=li, ybar_norm=float(ybar.norm()), spread=float(spread),
                         ratio=float(ybar.norm() / spread), shared_energy=shared,
                         K_pos=float((K > 0).double().mean()), K_med_over_ybar2=float(K.median() / ybar.pow(2).sum())))
    return rows


def pushes(model, arm, probe_next, data_next):
    """切替の時点で、層ごとに S^E・S^GD（unit ごと）と、検算用の μ・K の正の割合を返す。"""
    ins = ffn_inputs(model, probe_next)
    ybar = [y.mean(0) for (y, _, _) in ins]
    mu_probe = [z.mean(0) for (_, z, _) in ins]
    bias1 = 1.0 if has_bias(arm) else 0.0
    SE = [torch.zeros(C.FF, dtype=torch.float64) for _ in range(C.L)]
    SG = [torch.zeros(C.FF, dtype=torch.float64) for _ in range(C.L)]
    Kpos = [[] for _ in range(C.L)]
    phiK = [torch.zeros(C.FF, dtype=torch.float64) for _ in range(C.L)]     # Σ_n K_n φ_in（自己形用）
    alive = [torch.zeros(C.FF, dtype=torch.bool) for _ in range(C.L)]
    for b in range(N_BATCH):
        seq = data_next[b].long()
        x, tgt = seq[:, :C.T], seq[:, 1:]
        for mode in ("E", "GD"):
            cap = []
            model.zero_grad(set_to_none=True)
            logits = model.head(model.body(x, cap))
            zs = [cap[2 * li + 1] for li in range(C.L)]
            for z in zs:
                z.retain_grad()
            lg = logits[:, 1:].reshape(-1, C.V)
            if mode == "GD":
                loss = F.cross_entropy(lg, tgt[:, 1:].reshape(-1))
            else:
                pi = lg.detach().softmax(-1)
                loss = ((pi - 1.0 / C.V) * lg).sum() / lg.shape[0]
            loss.backward()
            for li in range(C.L):
                xin = cap[2 * li].detach()
                y = model.h[li].n2(xin).detach().reshape(-1, C.D).double()
                K = y @ ybar[li] + bias1
                d = zs[li].grad.reshape(-1, C.FF).double()
                s = (K[:, None] * d).sum(0)
                (SE if mode == "E" else SG)[li] += s
                if mode == "E":
                    zz = zs[li].detach().reshape(-1, C.FF).double()
                    phiK[li] += (K[:, None] * zz.clamp(min=0)).sum(0)
                    alive[li] |= (zz > 0).any(0)
                    Kpos[li].append(float((K > 0).double().mean()))
    return dict(SE=SE, SG=SG, phiK=phiK, alive=alive, mu_probe=mu_probe,
                Kpos=[float(np.mean(k)) for k in Kpos])


def eff_readout(model, probe):
    """最終層の unit i の実効読み出し v[c,i]（クラスで中心化）。LN のヤコビアンはトークンで平均する。"""
    with torch.no_grad():
        # 最終 LN の入力（最終ブロックの出力）
        hh = model.wte(probe)
        for blk in model.h:
            hh = blk(hh)
    hh = hh.reshape(-1, C.D).double()
    nf = model.nf
    Wd = model.h[-1].mlp.down.weight.detach().double()          # (D, FF)
    Wh = model.head.weight.detach().double()                     # (V, D)
    if isinstance(nf, nn.LayerNorm):
        mu = hh.mean(1, keepdim=True)
        var = hh.var(1, unbiased=False, keepdim=True)
        sig = (var + nf.eps).sqrt()
        hhat = (hh - mu) / sig
        d = C.D
        I = torch.eye(d, dtype=torch.float64)
        # J̄ = mean_m (1/σ_m)(I − 11ᵀ/d − ĥ_m ĥ_mᵀ/d)
        inv = (1.0 / sig).squeeze(1)
        Jbar = inv.mean() * (I - torch.ones(d, d, dtype=torch.float64) / d) - (hhat * inv[:, None]).T @ hhat / (d * len(hh))
        gam = nf.weight.detach().double()
        v = Wh @ (gam[:, None] * (Jbar @ Wd))
    else:
        v = Wh @ Wd
    return v - v.mean(0, keepdim=True)                          # (V, FF)


def label_feature_F(model, probe):
    """F[c,i] = Σ_m (y_mc − 1/V) φ_im（課題 t の probe の目標と、最終層の特徴）。"""
    cap = []
    with torch.no_grad():
        model.body(probe[:, :C.T - 1], cap)
    phi = cap[2 * (C.L - 1) + 1].clamp(min=0).reshape(-1, C.FF).double()
    tgt = probe[:, 1:C.T].reshape(-1)
    Y = F.one_hot(tgt, C.V).double() - 1.0 / C.V
    return Y.T @ phi                                             # (V, FF)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--arms", nargs="+", default=["C3", "C4", "B3"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    torch.manual_seed(0)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    geo_rows, push_rows, kappa_rows, q_rows, checks = [], [], [], [], []
    probes = {}
    for t in sorted(set([1, 10, 25, 50] + [s + 1 for s in SWITCH_FROM])):
        c = C.task_cpu_M(SEED, t, N_M, ALPHA)
        probes[t] = dict(probe=c["probe"], data=c["data"][:N_BATCH].clone(), probe_sha=c["probe_sha"])
        print("task", t, "ready", flush=True)
    for arm in a.arms:
        d = Path(a.root) / arm / f"seed{SEED}"
        U = np.load(d / "unit_stats.npz")
        prov = json.loads((d / "provenance.json").read_text())
        exp_probe = {ti["task"]: ti["probe_sha"] for ti in prov["tasks_info"]}
        m_init = load(arm, d, "init")
        # 1. 担い手の幾何
        for tag in GEOM_AT:
            t = 1 if tag == "init" else tag
            m = m_init if tag == "init" else load(arm, d, tag)
            for r in geometry(m, probes[t]["probe"], arm):
                geo_rows.append(dict(arm=arm, state=str(tag), **r))
        # 2. 切替の押し
        for t in SWITCH_FROM:
            assert probes[t + 1]["probe_sha"] == exp_probe[t + 1], "probe が走のものと違う"
            m = load(arm, d, t)
            P = pushes(m, arm, probes[t + 1]["probe"], probes[t + 1]["data"])
            dmu = U["chain_mu_u100"][t] - U["chain_mu_u0"][t]
            for li in range(C.L):
                mu_saved = U["chain_mu_u0"][t, li].astype(np.float64)
                err = float(np.abs(P["mu_probe"][li].numpy() - mu_saved).max())
                checks.append(dict(arm=arm, switch_from=t, layer=li, mu_recon_maxerr=err,
                                   mu_scale=float(np.abs(mu_saved).mean())))
                al = P["alive"][li].numpy()
                SE, SG = P["SE"][li].numpy(), P["SG"][li].numpy()
                meas = dmu[li].astype(np.float64)
                row = dict(arm=arm, switch_from=t, layer=li, n_alive=int(al.sum()), K_pos=P["Kpos"][li],
                           SE_sink=float((SE[al] > 0).mean()), SG_sink=float((SG[al] > 0).mean()),
                           meas_sink=float((meas[al] < 0).mean()),
                           agree_SE_meas=float((np.sign(-SE[al]) == np.sign(meas[al])).mean()),
                           agree_SG_meas=float((np.sign(-SG[al]) == np.sign(meas[al])).mean()),
                           agree_SE_SG=float((np.sign(SE[al]) == np.sign(SG[al])).mean()))
                if li == C.L - 1:
                    v = eff_readout(m, probes[t]["probe"])
                    v0 = eff_readout(m_init, probes[t]["probe"])
                    Fm = label_feature_F(m, probes[t]["probe"])
                    kappa = float(((v - v0) * Fm).sum() / (Fm * Fm).sum())
                    resid = (v - v0 - kappa * Fm)
                    r2 = float(1 - resid.pow(2).sum() / (v - v0).pow(2).sum())
                    Sself = kappa * (1 - 1.0 / C.V) * P["phiK"][li].numpy()
                    row.update(kappa=kappa, kappa_r2=r2,
                               agree_self_SE=float((np.sign(Sself[al]) == np.sign(SE[al])).mean()),
                               agree_self_meas=float((np.sign(-Sself[al]) == np.sign(meas[al])).mean()))
                    kappa_rows.append(dict(arm=arm, switch_from=t, kappa=kappa, r2=r2,
                                           v_norm=float(v.norm()), dv_norm=float((v - v0).norm()), F_norm=float(Fm.norm())))
                push_rows.append(row)
            print(arm, "switch", t, "done", flush=True)
        # 3. 開閉の二極（課題末の q の分布）
        qa = U["chain_q"]
        for t in (1, 10, 25, 50):
            for li in range(C.L):
                q = qa[t - 1, li]
                q_rows.append(dict(arm=arm, task=t, layer=li, always_on=float((q <= 0.05).mean()),
                                   never_on=float((q >= 0.95).mean()), middle=float(((q > 0.05) & (q < 0.95)).mean()),
                                   dead=float(U["chain_dead"][t - 1, li].mean())))

    def wcsv(name, rows):
        keys = list(dict.fromkeys(k for r in rows for k in r))
        with open(out / name, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow({k: (f"{v:.6g}" if isinstance(v, float) else v) for k, v in r.items()})
    wcsv("geometry.csv", geo_rows)
    wcsv("switch_push.csv", push_rows)
    wcsv("kappa.csv", kappa_rows)
    wcsv("q_modes.csv", q_rows)
    wcsv("checks.csv", checks)
    gh, dirty = C.git_info()
    (out / "provenance.json").write_text(json.dumps(dict(script="chargpt_cl_1019_posthoc_carrier.py", git_hash=gh,
                                                         dirty_src=dirty, root=a.root, arms=a.arms, n_batch=N_BATCH,
                                                         torch=torch.__version__, grade="事後・未登録"), indent=1))
    print("written", out)


if __name__ == "__main__":
    main()
