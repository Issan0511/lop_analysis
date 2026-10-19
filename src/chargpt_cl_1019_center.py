#!/usr/bin/env python3
"""chargpt_cl_1019 の追加腕（spec 追補 3 の案・未登録）: FFN の入力の EMA 中心化（MLP の扉 H の相当）と up の bias の除去（扉 B）。

本走のモジュール chargpt_cl_1019.py は書き換えず、読み込んだうえで腕と部品を足す。
  C3H : C3 の各 FFN の入力 y から特徴ごとの EMA 平均 m を引く（β = 0.1、勾配なし、訓練と評価で同じ値、偏りの補正つき）
        β は MLP の扉 H の 0.01 から 0.1 に上げた。箱 M は課題ごとに入力の分布が変わり、0.01（時定数 100 更新）では
        押しが集まる切替直後の約 200 更新で m が遅れるため（Neokunekune のセッションの反論 1）。遅れは measure で記録する。
  B3H : B3 に同じ中心化。up の bias は学習する
  B3HB: B3H から up の bias を外す（恒等的に 0）
EMA はその forward で中心化に使ったあとで更新する（同じ系列の未来のトークンが混ざらない）。更新は勾配が有効な forward（学習の 1 更新）だけで行い、測定（no_grad）では動かさない。

使い方は本走と同じ:
  python src/chargpt_cl_1019_center.py run --box M --arm C3H --seed 0 --tasks 50 --n_per_task 10000 --alpha 0.01 --out ...
"""
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chargpt_cl_1019 as C  # noqa: E402

EMA_BETA = 0.1

C.ARMS.update({
    "C3H": dict(norm="ln", ln_bias=False, lin_bias=False, lr=1e-3, center=True),
    "B3H": dict(norm="ln", ln_bias=True, lin_bias=True, lr=1e-3, center=True),
    "B3HB": dict(norm="ln", ln_bias=True, lin_bias=True, lr=1e-3, center=True, up_bias=False),
})


class MLPCenter(nn.Module):
    def __init__(s, arm):
        super().__init__()
        a = C.ARMS[arm]
        b = a["lin_bias"]
        s.up = nn.Linear(C.D, C.FF, bias=a.get("up_bias", b))
        s.down = nn.Linear(C.FF, C.D, bias=b)
        s.center = a.get("center", False)
        if s.center:
            s.register_buffer("ema_m", torch.zeros(C.D))
            s.register_buffer("ema_c", torch.zeros(()))

    def mean_vec(s):
        return s.ema_m / s.ema_c.clamp(min=1e-12)

    def forward(s, y, cap=None):
        if s.center:
            yc = y - s.mean_vec()
            if torch.is_grad_enabled():
                with torch.no_grad():
                    s.ema_m.mul_(1 - EMA_BETA).add_(y.detach().mean((0, 1)), alpha=EMA_BETA)
                    s.ema_c.mul_(1 - EMA_BETA).add_(EMA_BETA)
            y = yc
        z = s.up(y)
        if cap is not None:
            cap.append(z)
        return s.down(F.relu(z))


C.MLP = MLPCenter

_orig_capture = C.Engine._capture


def _capture_keep_buffers(s):
    """グラフの暖機で動いた EMA の buffer を初期の値に戻す（記憶はそのまま）。"""
    bufs = [b for n, b in s.m.named_buffers() if n.endswith("ema_m") or n.endswith("ema_c")]
    snap = [b.detach().clone() for b in bufs]
    _orig_capture(s)
    with torch.no_grad():
        for b, q in zip(bufs, snap):
            b.copy_(q)
    torch.cuda.synchronize()


C.Engine._capture = _capture_keep_buffers

@torch.no_grad()
def measure_center(model, x):
    """本走の measure と同じ量。分解に中心化の項 −wᵢ·m を足す（float64）。unit ごとの −wᵢ·m は μ − 三項で出せるので保存しない。"""
    cap = []
    model.body(x, cap)
    out = []
    for li, blk in enumerate(model.h):
        xin, z = cap[2 * li], cap[2 * li + 1]
        z2 = z.reshape(-1, C.FF)
        z64 = z2.double()
        mu = z64.mean(0)
        sd = z64.std(0)
        q = (z2 < 0).double().mean(0)
        dead = (z2 > 0).sum(0) == 0
        n2 = blk.n2
        xhat = F.layer_norm(xin, (C.D,), eps=n2.eps).reshape(-1, C.D).double()
        gam = n2.weight.double()
        beta = n2.bias.double() if n2.bias is not None else None
        mhat = xhat.mean(0)
        mlp = blk.mlp
        W = mlp.up.weight.double()
        gm = gam * mhat
        t1 = W @ gm
        t2 = W @ beta if beta is not None else torch.zeros_like(t1)
        t3 = mlp.up.bias.double() if mlp.up.bias is not None else torch.zeros_like(t1)
        t4 = -(W @ mlp.mean_vec().double()) if getattr(mlp, "center", False) else torch.zeros_like(t1)
        cos = t1 / (W.norm(dim=1) * gm.norm() + 1e-30)
        if getattr(mlp, "center", False):
            ybar = blk.n2(xin).reshape(-1, C.D).double().mean(0)
            lag = (ybar - mlp.mean_vec().double()).norm()
            lag_abs, lag_rel = float(lag), float(lag / (ybar.norm() + 1e-30))
        else:
            lag_abs = lag_rel = 0.0
        r = mu / (sd + 1e-12)
        rq = torch.quantile(r, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64, device=r.device))
        agg = dict(neg_frac=float((z2 < 0).double().mean()), units_mean_neg=float((mu < 0).double().mean()),
                   r_q25=float(rq[0]), r_med=float(rq[1]), r_q75=float(rq[2]),
                   sunk=float((q >= 0.95).double().mean()), dead=float(dead.double().mean()),
                   decomp_err=float((mu - (t1 + t2 + t3 + t4)).abs().max()),
                   mu_absmean=float(mu.abs().mean()),
                   center_norm=float(mlp.mean_vec().norm()) if getattr(mlp, "center", False) else 0.0,
                   t_wcenter_absmean=float(t4.abs().mean()),
                   center_lag=lag_abs, center_lag_rel=lag_rel)
        units = dict(mu=mu, sd=sd, q=q, t_wgm=t1, t_wbeta=t2, t_b=t3, cos=cos)
        out.append(dict(agg=agg, units={k: v.float().cpu().numpy() for k, v in units.items()},
                        dead=dead.cpu().numpy()))
    return out


C.measure = measure_center

# 課題末の checkpoint に Adam の状態も残す（切替直後の更新を後から再生できるように。Neokunekune のセッションの (c)(1)）
C._ENGINES = []
_orig_engine_init = C.Engine.__init__


def _engine_init(s, *a, **k):
    _orig_engine_init(s, *a, **k)
    C._ENGINES.append(s)


C.Engine.__init__ = _engine_init
_orig_save = torch.save


def _save_with_adam(obj, f, *a, **k):
    _orig_save(obj, f, *a, **k)
    p = str(f)
    if "/ckpts/task" in p and p.endswith(".pt") and not p.endswith("_adam.pt") and C._ENGINES:
        _orig_save(C._ENGINES[0].opt.state_dict(), p[:-3] + "_adam.pt")


torch.save = _save_with_adam

if __name__ == "__main__":
    C.main()
