#!/usr/bin/env python3
"""chargpt_cl_1019 の追加腕（spec 追補 5 の案）: FFN の活性化を kunekune 族と GELU に替える。

本走のモジュール chargpt_cl_1019.py は書き換えず、読み込んだうえで腕と部品を足す。土台は C3（LN γ のみ・bias なし・lr 1e−3）。
  SNA    : Snake φ(z) = z + sin²(αz)/α。α_i = clip(c/W_i, 0.005, 3)、W_i = √V_i、V_i は前活性のバッチ分散の EMA（β 0.01）。c = 0.6
  KKT1   : θ = 2αz が [−2π, π] の内は Snake、外は直線（左 z、右 z + 1/α）。α は SNA と同じ規則
  SNAfrz1: SNA。課題 1 の末で α と V を固定する
  GELU   : F.gelu（erf の正確な形）
α と V の扱いは MLP（layer_chimera_cifar_0921.SnakeFamily）・CNN（neokunekune_cifar5p1_cnn_1018）の実装と同じ。
  1 更新の順: 更新前の V から α → forward → backward・Adam → この forward の前活性の分散で V を更新。
違いは V の初期値だけで、MLP 版の V = 1 ではなく、初期値の網に課題 1 の probe を流した前活性の unit ごとの分散で較正する
（依頼「α の初期値は FFN 前活性の初期の幅に合わせる」）。
V の更新は勾配が有効な forward（学習の 1 更新）だけで行い、測定（no_grad）では動かさない。
"""
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chargpt_cl_1019 as C  # noqa: E402

PI = math.pi
SN_C, SN_BETA, SN_LO, SN_HI = 0.6, 0.01, 0.005, 3.0

C.ARMS.update({
    "KKT1": dict(norm="ln", ln_bias=False, lin_bias=False, lr=1e-3, act="kkt1"),
    "SNA": dict(norm="ln", ln_bias=False, lin_bias=False, lr=1e-3, act="snake"),
    "SNAfrz1": dict(norm="ln", ln_bias=False, lin_bias=False, lr=1e-3, act="snake", freeze_after_task1=True),
    "GELU": dict(norm="ln", ln_bias=False, lin_bias=False, lr=1e-3, act="gelu"),
    # 下見（未登録、Issa 1020「下限に張り付いたからじゃないの？撤廃したら？」）: α の下限 0.005 だけ外す。上限 3 は残す
    "KKT1nolo": dict(norm="ln", ln_bias=False, lin_bias=False, lr=1e-3, act="kkt1", alpha_lo=0.0),
    # 下見（未登録、Issa 1020 午後）: 共通の書き込みを止める 2 つの形
    "KKT1b3": dict(norm="ln", ln_bias=True, lin_bias=True, lr=1e-3, act="kkt1"),              # 土台を B3（LN の β と線形層の bias あり）に
    "KKT1c": dict(norm="ln", ln_bias=False, lin_bias=False, lr=1e-3, act="kkt1", center_offset=True),  # φ − 1/(2α): sin² の広い分布での平均を引く
})


def snake_phi(z, a, kind):
    ia = a.reciprocal()
    s = z + torch.sin(a * z) ** 2 * ia                   # MLP・CNN と同じ演算の順
    if kind == "snake":
        return s
    th = 2.0 * a * z
    return torch.where(th < -2.0 * PI, z, torch.where(th > PI, z + ia, s))


def snake_dphi(z, a, kind):
    th = 2.0 * a * z
    s = 1.0 + torch.sin(th)
    if kind == "snake":
        return s
    inside = (th >= -2.0 * PI) & (th <= PI)
    return torch.where(inside, s, torch.ones_like(z))


class MLPKune(nn.Module):
    def __init__(s, arm):
        super().__init__()
        a = C.ARMS[arm]
        b = a["lin_bias"]
        s.up = nn.Linear(C.D, C.FF, bias=b)
        s.down = nn.Linear(C.FF, C.D, bias=b)
        s.act = a.get("act", "relu")
        s.snake = s.act in ("snake", "kkt1")
        s.alpha_lo = a.get("alpha_lo", SN_LO)
        s.center_offset = a.get("center_offset", False)
        if s.snake:
            s.register_buffer("V", torch.ones(C.FF))
            s.register_buffer("v_rate", torch.tensor(SN_BETA))      # 0 で凍結（SNAfrz1）

    def alpha(s):
        return (SN_C / s.V.sqrt()).clamp(s.alpha_lo, SN_HI)

    def phi(s, z):
        if s.act == "relu":
            return F.relu(z)
        if s.act == "gelu":
            return F.gelu(z)
        a = s.alpha()
        out = snake_phi(z, a, s.act)
        if s.center_offset:
            out = out - 0.5 * a.reciprocal()               # 定数の引き算なので φ′ と継ぎ目の連続性は変わらない
        return out

    def dphi(s, z):
        if s.act == "relu":
            return (z > 0).to(z.dtype)
        if s.act == "gelu":
            cdf = 0.5 * (1.0 + torch.erf(z / math.sqrt(2.0)))
            return cdf + z * torch.exp(-0.5 * z * z) / math.sqrt(2.0 * PI)
        return snake_dphi(z, s.alpha(), s.act)

    def forward(s, y, cap=None):
        z = s.up(y)
        if cap is not None:
            cap.append(z)
        h = s.phi(z)
        if s.snake and torch.is_grad_enabled():
            with torch.no_grad():
                var = z.detach().reshape(-1, C.FF).var(0, unbiased=False)
                s.V.mul_(1 - s.v_rate).add_(s.v_rate * var)
        return s.down(h)


C.MLP = MLPKune

# ---------------------------------------------------------------- V の較正（初期値の網・課題 1 の probe）
_orig_build = C.build_model


@torch.no_grad()
def build_model_calibrated(arm, box, seed, device):
    m = _orig_build(arm, box, seed, device)
    mlps = [blk.mlp for blk in m.h if getattr(blk.mlp, "snake", False)]
    if mlps and box == "M":
        probe = C.task_cpu_M(seed, 1, 1, 0.01)["probe"].to(device)
        cap = []
        m.body(probe, cap)
        for li, blk in enumerate(m.h):
            z = cap[2 * li + 1].reshape(-1, C.FF)
            blk.mlp.V.copy_(z.var(0, unbiased=False))
        m.v_calibrated = True
    return m


C.build_model = build_model_calibrated

# ---------------------------------------------------------------- グラフの暖機で動いた V を戻す
_orig_capture = C.Engine._capture


def _capture_keep_buffers(s):
    bufs = [b for n, b in s.m.named_buffers() if n.endswith(".V")]
    snap = [b.detach().clone() for b in bufs]
    _orig_capture(s)
    with torch.no_grad():
        for b, q in zip(bufs, snap):
            b.copy_(q)
    torch.cuda.synchronize()


C.Engine._capture = _capture_keep_buffers

# ---------------------------------------------------------------- SNAfrz1: 課題 2 の頭で α と V を固定
_orig_run_task = C.run_task


def run_task_freeze(model, eng, ctx, *, warmup_from, **kw):
    if C.ARMS[model.arm].get("freeze_after_task1") and warmup_from > 0:
        for blk in model.h:
            if getattr(blk.mlp, "snake", False):
                blk.mlp.v_rate.fill_(0.0)
    return _orig_run_task(model, eng, ctx, warmup_from=warmup_from, **kw)


C.run_task = run_task_freeze


# ---------------------------------------------------------------- 測定（本走の量に φ′ と α の量を足す）
@torch.no_grad()
def measure_kune(model, x):
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
        cos = t1 / (W.norm(dim=1) * gm.norm() + 1e-30)
        r = mu / (sd + 1e-12)
        rq = torch.quantile(r, torch.tensor([0.25, 0.5, 0.75], dtype=torch.float64, device=r.device))
        dp = mlp.dphi(z2).double()                            # 局所応答 φ′（トークン × unit）
        dpu = dp.mean(0)                                      # unit ごとの φ′ の平均
        resp = (dp >= 0.5).double().mean(0)                   # unit ごとの「応答する」割合（φ′ ≥ 0.5）
        dq = torch.quantile(dpu, torch.tensor([0.1, 0.5], dtype=torch.float64, device=r.device))
        agg = dict(neg_frac=float((z2 < 0).double().mean()), units_mean_neg=float((mu < 0).double().mean()),
                   r_q25=float(rq[0]), r_med=float(rq[1]), r_q75=float(rq[2]),
                   sunk=float((q >= 0.95).double().mean()), dead=float(dead.double().mean()),
                   decomp_err=float((mu - (t1 + t2 + t3)).abs().max()), mu_absmean=float(mu.abs().mean()),
                   phi1_mean=float(dp.mean()), phi1_unit_q10=float(dq[0]), phi1_unit_med=float(dq[1]),
                   units_phi1_lt01=float((dpu < 0.1).double().mean()),
                   resp_lt005=float((resp < 0.05).double().mean()), resp_005_03=float(((resp >= 0.05) & (resp < 0.3)).double().mean()),
                   resp_03_07=float(((resp >= 0.3) & (resp <= 0.7)).double().mean()), resp_gt07=float((resp > 0.7).double().mean()))
        if getattr(mlp, "snake", False):
            a = mlp.alpha().double()
            Wd = mlp.V.double().sqrt()
            raw = SN_C / Wd
            th = 2.0 * a[None, :] * z64
            agg.update(alpha_med=float(a.median()), two_alpha_W_med=float((2 * a * Wd).median()),
                       two_alpha_sd_med=float((2 * a * sd).median()),
                       alpha_clip_frac=float(((raw < mlp.alpha_lo) | (raw > SN_HI)).double().mean()),
                       alpha_below_lo=float((raw < SN_LO).double().mean()),   # 既定の下限 0.005 を割る割合（下限なしの腕でも数える）
                       frac_in_curve=float(((th >= -2.0 * PI) & (th <= PI)).double().mean()))
        units = dict(mu=mu, sd=sd, q=q, t_wgm=t1, t_wbeta=t2, t_b=t3, cos=cos)
        out.append(dict(agg=agg, units={k: v.float().cpu().numpy() for k, v in units.items()}, dead=dead.cpu().numpy()))
    return out


C.measure = measure_kune

# ---------------------------------------------------------------- 課題末の checkpoint に Adam の状態も残す
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
