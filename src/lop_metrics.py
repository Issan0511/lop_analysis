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

        # --- 重み行ベクトルの pair |cos| 平均 ([NEW], Path B の整列時系列)
        Wn = net.W / net.W.norm(dim=2, keepdim=True).clamp_min(1e-12)
        wG = torch.einsum("rid,rjd->rij", Wn, Wn)        # [R,h,h]
        wcos_mean = wG[:, iu[0], iu[1]].abs().mean(dim=1)

        # --- 符号一致率 ([NEW]): sign(w_i) と sign(w_j) の成分一致割合
        sgn = torch.sign(net.W)                          # [R,h,d]
        match = torch.einsum("rid,rjd->rij", sgn, sgn)   # 一致 d - 2*不一致
        d = net.W.shape[2]
        match = (match + d) / (2 * d)                    # 一致割合 [R,h,h]
        pair_match = match[:, iu[0], iu[1]]
        sign_match_mean = pair_match.mean(dim=1)
        sign_clone_frac = (pair_match >= C["sign_match_tau"]).float().mean(dim=1)

        # --- SDE 分解 [methods_sde_0813] ①: eval バッチ上の drift / diffusion (W 勾配)
        #     snr_drift = |E[g]|^2 / tr C(w) がその run の drift 支配 / diffusion 支配の直接指標
        x_z = torch.where(finite[None, :, None], x_eval, torch.zeros_like(x_eval))
        gW = net.grads_batch(x_z, pre, a, delta)[0]         # [N,R,h,d]
        drift_sq_W = gW.mean(0).pow(2).sum((1, 2))          # |E[g]|^2   [R]
        trC_W = gW.var(0, unbiased=False).sum((1, 2))       # tr C(w)    [R]
        snr_drift = drift_sq_W / trC_W.clamp_min(1e-30)

        # --- [methods_sde_0813] ②: 重み行列 W 本体の有効ランクと特異値集中度
        #     (既存 eff_rank は活性 a のランク。整列による低ランク化を W 側で直接見る)
        Wz = torch.where(finite[:, None, None], net.W, torch.zeros_like(net.W))
        sw = torch.linalg.svdvals(Wz)                       # [R,min(h,d)]
        pw = sw / sw.sum(dim=1, keepdim=True).clamp_min(1e-12)
        eff_rank_W = torch.exp(-(pw * pw.clamp_min(1e-12).log()).sum(dim=1))
        top1_frac = sw[:, 0] ** 2 / (sw ** 2).sum(dim=1).clamp_min(1e-24)
        # srank(W) = ||W||_F^2 / ||W||_2^2 (実験(5) coupling_ab の指標①の厳密定義)
        stable_rank_W = (sw ** 2).sum(dim=1) / (sw[:, 0] ** 2).clamp_min(1e-24)

        # dead 行除外 srank (④→① 逆流の分離用): dead 行をゼロ化した W の srank。
        # ゼロ行の追加は非ゼロ特異値を変えないため部分行列の srank と等価。
        # 全行 dead の run は NaN。
        W_alive = torch.where(dead_i[:, :, None], torch.zeros_like(Wz), Wz)
        sa = torch.linalg.svdvals(W_alive)
        stable_rank_W_alive = (sa ** 2).sum(dim=1) / (sa[:, 0] ** 2).clamp_min(1e-24)
        any_alive = ~dead_i.all(dim=1)
        stable_rank_W_alive = torch.where(any_alive, stable_rank_W_alive,
                                          torch.full_like(stable_rank_W_alive, float("nan")))

        # --- [methods_sde_0813] ③: gate 開放率ベースの dead (Leaky 用)。
        #     Leaky では |a|<tol 基準の dead_frac が定義上ほぼ 0 になるため、
        #     ReLU 換算で dead 相当 (P(pre>0) < 1-dead_tau) のユニット割合を併記
        open_frac = (pre > 0).float().mean(dim=0)           # [R,h]
        neg_gate_frac = (open_frac < 1 - C["dead_tau"]).float().mean(dim=1)

        # eval バッチ上の損失 (公式同様 (yhat-y)^2 平均)
        eval_loss = (delta ** 2).mean(dim=0)

    out = dict(dead_frac=dead_frac, dup_frac=dup_frac, sat_frac=sat_frac,
               eff_rank=eff_rank, stable_rank=stable_rank, wcos_mean=wcos_mean,
               sign_match_mean=sign_match_mean, sign_clone_frac=sign_clone_frac,
               drift_sq_W=drift_sq_W, trC_W=trC_W, snr_drift=snr_drift,
               eff_rank_W=eff_rank_W, top1_frac=top1_frac, stable_rank_W=stable_rank_W,
               stable_rank_W_alive=stable_rank_W_alive,
               neg_gate_frac=neg_gate_frac,
               eval_loss=eval_loss)
    nan = torch.full_like(finite.float(), float("nan"))
    return {k: torch.where(finite, v.float(), nan) for k, v in out.items()}
