"""LoP 症状メトリクス ([J] Appendix B の定義 + [NEW] 符号一致率)。

固定 eval バッチ上で計測 (仕様書 §4)。
saturated の「勾配」はニューロン i のプリ活性勾配 |2 delta v_i 1[pre_i>0]| を使用
([J] の定義が入手不能な箇所は README に明記)。
"""
import torch


def compute_lop_metrics(net, x_eval, y_eval, cfg):
    """x_eval: [N,R,d], y_eval: [N,R] -> dict of per-run tensors [R]"""
    C = cfg["common"]
    N = x_eval.shape[0]
    with torch.no_grad():
        pre, a, yhat = net.forward_batch(x_eval)        # [N,R,h]
        delta = yhat - y_eval                           # [N,R]

        # 発散系列 (パラメータ NaN/Inf) は NaN を記録して除外 (発散も観測結果)
        finite = a.isfinite().all(dim=2).all(dim=0) & delta.isfinite().all(dim=0)  # [R]
        a = torch.where(finite[None, :, None], a, torch.zeros_like(a))
        pre = torch.where(finite[None, :, None], pre, torch.zeros_like(pre))
        delta = torch.where(finite[None, :], delta, torch.zeros_like(delta))

        # --- dead: |a| < tol がサンプルの tau_dead 割合超
        dead_i = ((a.abs() < C["dead_tol"]).float().mean(dim=0) > C["dead_tau"])  # [R,h]
        dead_frac = dead_i.float().mean(dim=1)

        # --- duplicate: 正規化活性ベクトルの内積 > tau_corr のペア割合
        An = a.permute(1, 2, 0)                          # [R,h,N]
        norms = An.norm(dim=2, keepdim=True)
        An = An / norms.clamp_min(1e-12)
        G = torch.einsum("rin,rjn->rij", An, An)         # [R,h,h]
        h = G.shape[1]
        iu = torch.triu_indices(h, h, offset=1, device=G.device)
        pair_cos = G[:, iu[0], iu[1]]                    # [R,P]
        dup_frac = (pair_cos > C["dup_tau"]).float().mean(dim=1)

        # --- saturated: |grad|/max(平均活性, eps) < tol がサンプルの sat_frac 超
        g_pre = (2.0 * delta)[..., None] * net.v * (pre > 0).float()   # [N,R,h]
        mean_act = a.abs().mean(dim=0)                                  # [R,h]
        ratio = g_pre.abs() / mean_act.clamp_min(1e-12)
        sat_i = ((ratio < C["sat_ratio_tol"]).float().mean(dim=0) > C["sat_frac"])
        sat_frac = sat_i.float().mean(dim=1)

        # --- 実効ランク (エントロピー) と stable rank (中心化済み)
        s = torch.linalg.svdvals(a.permute(1, 0, 2))     # [R,min(N,h)]
        p = s / s.sum(dim=1, keepdim=True).clamp_min(1e-12)
        eff_rank = torch.exp(-(p * (p.clamp_min(1e-12)).log()).sum(dim=1))
        a_c = a - a.mean(dim=0, keepdim=True)
        s2 = torch.linalg.svdvals(a_c.permute(1, 0, 2)) ** 2
        stable_rank = s2.sum(dim=1) ** 2 / (s2 ** 2).sum(dim=1).clamp_min(1e-24)

        # --- 符号一致率 ([NEW]): sign(w_i) と sign(w_j) の成分一致割合
        sgn = torch.sign(net.W)                          # [R,h,d]
        match = torch.einsum("rid,rjd->rij", sgn, sgn)   # 一致 d - 2*不一致
        d = net.W.shape[2]
        match = (match + d) / (2 * d)                    # 一致割合 [R,h,h]
        pair_match = match[:, iu[0], iu[1]]
        sign_match_mean = pair_match.mean(dim=1)
        sign_clone_frac = (pair_match >= C["sign_match_tau"]).float().mean(dim=1)

        # --- [NEW mu_sweep] 重み方向の多様性 D (W の行方向。E[g] ではない点に注意)
        Wn = net.W / net.W.norm(dim=2, keepdim=True).clamp_min(1e-12)   # u_i [R,h,d]
        ubar_all = Wn.mean(dim=1)                                        # [R,d]
        w_D_all = 1.0 - (ubar_all ** 2).sum(dim=1)                       # D = 1 - ||ubar||^2
        alive = ~dead_i                                                  # [R,h]
        n_alive = alive.float().sum(dim=1)                               # [R]
        ubar_al = (Wn * alive.float().unsqueeze(2)).sum(dim=1) / n_alive.clamp_min(1).unsqueeze(1)
        w_D_alive = torch.where(n_alive >= 2, 1.0 - (ubar_al ** 2).sum(dim=1),
                                torch.full_like(n_alive, float("nan")))
        Gu = torch.einsum("rid,rjd->rij", Wn, Wn)
        w_paircos_all = Gu[:, iu[0], iu[1]].mean(dim=1)                  # 符号付き平均 pairwise cos
        # [NEW norm_sweep] 生存ユニット限定 pairwise cos (D_alive とは独立経路で計算し、
        # 恒等式 D = (1-1/n)(1-cbar) の数値確認に使う)
        Am = alive.float()
        pairmask_al = Am[:, :, None] * Am[:, None, :]
        sumG_al = (Gu * pairmask_al).sum(dim=(1, 2))                     # 対角込み (対角和 = n_alive)
        npairs_al = (n_alive * (n_alive - 1.0)).clamp_min(1.0)
        w_paircos_alive = torch.where(n_alive >= 2, (sumG_al - n_alive) / npairs_al,
                                      torch.full_like(n_alive, float("nan")))
        rnorm = net.W.norm(dim=2)                                        # [R,h]
        w_rnorm_mean = rnorm.mean(dim=1)
        w_rnorm_alive = torch.where(n_alive >= 1,
                                    (rnorm * alive.float()).sum(dim=1) / n_alive.clamp_min(1),
                                    torch.full_like(n_alive, float("nan")))

        # eval バッチ上の損失 (公式同様 (yhat-y)^2 平均)
        eval_loss = (delta ** 2).mean(dim=0)

    out = dict(dead_frac=dead_frac, dup_frac=dup_frac, sat_frac=sat_frac,
               eff_rank=eff_rank, stable_rank=stable_rank,
               sign_match_mean=sign_match_mean, sign_clone_frac=sign_clone_frac,
               eval_loss=eval_loss,
               w_D_all=w_D_all, w_D_alive=w_D_alive, n_alive=n_alive,
               w_paircos_all=w_paircos_all, w_paircos_alive=w_paircos_alive,
               w_rnorm_mean=w_rnorm_mean, w_rnorm_alive=w_rnorm_alive)
    nan = torch.full_like(finite.float(), float("nan"))
    return {k: torch.where(finite, v.float(), nan) for k, v in out.items()}
