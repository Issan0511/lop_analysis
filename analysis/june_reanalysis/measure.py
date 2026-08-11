"""B2/B3 用の再測定（forward のみ、学習は一切しない）。

`src/freeze.py:_restore` で ckpt を復元し、followup.py と**同一のセグメント走査**で
測定窓を再走する。入力乱数系列が一致するので、得られる µ̂ は npz の `mu_inter` と一致する。

各 run について集める:
  mu_raw, Sigma_raw   測定窓の経験平均・経験共分散（生入力 x_raw）      -> B3
  mu_in,  Sigma_in    同、学習器が実際に見る入力 x_in (= x_raw - 中心化) -> B3
  Edx                 E[δ·x_in]          （E[g_i] = 2 v_i E[δ·gate_i·x_in] の非ゲート版）
  Cov_delta_x         Cov(δ, x_in)       -> H-cov の主対象
  Cov_yhat_x          Cov(ŷ, x_in)
  Cov_y_x             Cov(y,  x_in)      （教師側; c_t の窓平均に対応）
  Eg_gated            E[δ·gate_i·x_in]   （= E[g_i]/(2 v_i)。整列の直接の担い手）
  per-period c_t      各周期の Cov(y, x_in) と Cov(δ, x_in)

注意: 中心化 (enc=centered) 系列では x_in ≠ x_raw。E[g] は x_in で作られるので
H-cov の検証は x_in 側で行うのが正しい。
"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))

from src.common import load_config, build_runs, group_runs           # noqa: E402
from src.freeze import SEG, _restore, _ema_toeplitz                  # noqa: E402

from . import common as C                                            # noqa: E402

P_MAX = 64


def measure(gkey, step, cfg, device="cpu"):
    """1 グループ × 1 ckpt を再測定。dict of numpy arrays を返す。"""
    exp, width = gkey
    path = os.path.join(C.SRC_RESULTS, "ckpts", f"{exp}_w{width}_step{step}.pt")
    if not os.path.exists(path):
        return None
    ckpt = torch.load(path, map_location=device, weights_only=False)
    runs, env, teacher, net, rm0, centered, period, d = _restore(gkey, ckpt, cfg, device)
    Cc = cfg["common"]
    R, h = len(runs), width
    alpha = cfg["condA"]["center_alpha"]
    cmask = centered[:, None].float()

    M_r = torch.minimum(Cc["freeze_min_periods"] * period,
                        torch.tensor(Cc["freeze_M_cap"])).to(device)
    M_max = int(M_r.max().item())
    decay_full, L_full = _ema_toeplitz(alpha, SEG, device)
    per_dev = period.to(device)
    t0 = env.t
    bin0 = t0 // per_dev
    ridx = torch.arange(R, device=device)

    z = lambda *s: torch.zeros(*s, device=device, dtype=torch.float64)
    s_xr, s_xxr = z(R, d), z(R, d, d)
    s_xi, s_xxi = z(R, d), z(R, d, d)
    s_del, s_del_x, s_y, s_y_x, s_yh, s_yh_x = z(R), z(R, d), z(R), z(R, d), z(R), z(R, d)
    s_gated = z(R, h, d)
    # 周期別
    p_n, p_y, p_yx, p_d, p_dx = z(R, P_MAX), z(R, P_MAX), z(R, P_MAX, d), \
        z(R, P_MAX), z(R, P_MAX, d)

    rm = rm0.clone()
    s = 0
    while s < M_max:
        Cs = min(SEG, M_max - s)
        if exp == "B":
            teacher.t = env.t
            teacher.maybe_resample()
        xseg = env.segment(Cs)                                    # [Cs,R,d]
        y = teacher(xseg)                                         # [Cs,R]

        decay, L = decay_full[:Cs], L_full[:Cs, :Cs]
        rm_prev = decay[:, None, None] * rm[None] + torch.einsum("cj,jrd->crd", L, xseg)
        x_in = xseg - cmask[None] * rm_prev
        Lend = alpha * (1 - alpha) ** torch.arange(Cs - 1, -1, -1, device=device).float()
        rm = (1 - alpha) ** Cs * rm + torch.einsum("j,jrd->rd", Lend, xseg)

        pre, a, yhat = net.forward_batch(x_in)
        delta = yhat - y
        gate = (pre > 0).float()                                  # [Cs,R,h]

        sidx = s + torch.arange(Cs, device=device)
        w = (sidx[:, None] < M_r[None, :]).double()                # [Cs,R]
        xr, xi = xseg.double(), x_in.double()
        dl, yy, yh = delta.double(), y.double(), yhat.double()

        s_xr += (xr * w[..., None]).sum(0)
        s_xxr += torch.einsum("cr,crd,cre->rde", w, xr, xr)
        s_xi += (xi * w[..., None]).sum(0)
        s_xxi += torch.einsum("cr,crd,cre->rde", w, xi, xi)
        s_del += (dl * w).sum(0)
        s_del_x += torch.einsum("cr,crd->rd", w * dl, xi)
        s_y += (yy * w).sum(0)
        s_y_x += torch.einsum("cr,crd->rd", w * yy, xi)
        s_yh += (yh * w).sum(0)
        s_yh_x += torch.einsum("cr,crd->rd", w * yh, xi)
        s_gated += torch.einsum("cr,crh,crd->rhd", w * dl, gate.double(), xi)

        b = ((t0 + s) // per_dev - bin0).clamp(0, P_MAX - 1)
        p_n[ridx, b] += w.sum(0)
        p_y[ridx, b] += (yy * w).sum(0)
        p_yx[ridx, b] += torch.einsum("cr,crd->rd", w * yy, xi)
        p_d[ridx, b] += (dl * w).sum(0)
        p_dx[ridx, b] += torch.einsum("cr,crd->rd", w * dl, xi)
        s += Cs

    M = M_r.double()
    np_ = lambda t: t.detach().cpu().numpy()

    def cov(sx, sxx, n):
        m = sx / n[:, None]
        return (sxx / n[:, None, None]) - m[:, :, None] * m[:, None, :], m

    Sig_r, mu_r = cov(s_xr, s_xxr, M)
    Sig_i, mu_i = cov(s_xi, s_xxi, M)
    Ed, Edx = s_del / M, s_del_x / M[:, None]
    Ey, Eyx = s_y / M, s_y_x / M[:, None]
    Eyh, Eyhx = s_yh / M, s_yh_x / M[:, None]

    pn = p_n.clamp_min(1)
    p_mu_ok = (p_n >= 0.5 * per_dev[:, None].double())
    p_cd = p_dx / pn[:, :, None] - (p_d / pn)[:, :, None] * (s_xi / M[:, None])[:, None, :]
    p_cy = p_yx / pn[:, :, None] - (p_y / pn)[:, :, None] * (s_xi / M[:, None])[:, None, :]

    return dict(
        run_ids=np.array([r["run_id"] for r in runs]),
        step=np.int64(step), d=np.int64(d),
        mu_raw=np_(mu_r), Sigma_raw=np_(Sig_r),
        mu_in=np_(mu_i), Sigma_in=np_(Sig_i),
        E_delta=np_(Ed), Edx=np_(Edx),
        Cov_delta_x=np_(Edx - Ed[:, None] * mu_i),
        Cov_y_x=np_(Eyx - Ey[:, None] * mu_i),
        Cov_yhat_x=np_(Eyhx - Eyh[:, None] * mu_i),
        Eg_gated=np_(s_gated / M[:, None, None]),
        period_cov_delta_x=np_(p_cd), period_cov_y_x=np_(p_cy),
        period_ok=np_(p_mu_ok), period_n=np_(p_n),
        centered=np_(centered), M=np_(M), period=np_(per_dev),
    )


def load_cfg():
    return load_config(os.path.join(C.SRC_RESULTS, "config_used.yaml"))


def cache_path(exp, width, step):
    return os.path.join(C.OUT, "_measure", f"meas_{exp}_w{width}_step{step}.npz")


def get(exp, width, step, cfg=None, force=False):
    """キャッシュ付き。初回は再測定して npz に保存する。"""
    p = cache_path(exp, width, step)
    if os.path.exists(p) and not force:
        z = dict(np.load(p, allow_pickle=True))
        z["run_ids"] = np.array([str(s) for s in z["run_ids"]])
        return z
    cfg = cfg or load_cfg()
    out = measure((exp, width), step, cfg)
    if out is None:
        return None
    os.makedirs(os.path.dirname(p), exist_ok=True)
    np.savez_compressed(p, **out)
    print(f"    measured -> {os.path.relpath(p, C.ROOT)}")
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", nargs="*", type=int, default=[100000, 1000000])
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg()
    torch.set_grad_enabled(False)
    for exp, width in C.GROUPS:
        for step in a.steps:
            print(f"=== measure {exp}_w{width} step={step}", flush=True)
            get(exp, width, step, cfg, force=a.force)
    print("MEASURE DONE")


if __name__ == "__main__":
    main()
