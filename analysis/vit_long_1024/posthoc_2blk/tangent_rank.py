"""体の接核（tangent kernel）の有効階数（Claude、1027）: 体は画像ごとに特徴を別々に動かせるか。

保存した課題末の重みで、head が読む CLS 特徴 h_m（最終 LN の後、384 次元）の方向微分 u·h_m を体のパラメータ θ（head 以外）で
微分した g_m = ∂(u·h_m)/∂θ を、2,500 枚から固定で選んだ 256 枚について求め、Gram 行列 G = [g_m·g_m'] の固有値の参加比
（有効階数）と、向きだけの Gram（cos）の参加比を書く。方向 u は 2 種:
  rand   固定の乱数の単位ベクトル 4 本（head がまだ無いときに読む向き）
  pc     その状態の 2,500 枚の中心化した特徴の上位 4 主成分（head が実際に読む向き）
比較のために特徴そのものの Gram（0 次、256 枚）の参加比も書く。

乱数ラベルの当てはめは 2,500 枚を別々に動かす必要がある。接核の有効階数が低いほど、同じ更新が多くの画像を同じ向きに
動かし、画像ごとの分けに使える方向が少ない（Lewandowski ら 2023 の「曲率の方向の数」と同じ量の、一次の版）。

    PYTHONPATH=$PWD VIT_LONG_PRODUCTION=production_2blk_rlfix.json \
        python analysis/vit_long_1024/posthoc_2blk/tangent_rank.py SL 100 <出力.json>
"""
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

from analysis.vit_long_1024 import common as C

SEED, N_IMG, N_DIR = 10252010, 256, 4
RUNS = Path.home() / 'Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/runs'


def cls_features(model, x):
    L = model.layers
    hh = L['patch_embed'](x)
    hh = torch.cat((model.cls_token.expand(hh.shape[0], -1, -1), hh), 1) + model.pos_embed
    for i in range(model.depth):
        hh = L[f'block_{i}'](hh)
    return L['norm'](hh)[:, 0]


def participation(e):
    e = e.clamp(min=0)
    return float(e.sum() ** 2 / (e ** 2).sum())


def eigvals(G):
    """固有値（降順）。収束しないときは対角に trace/m × 1e-9 を足して CPU で解き直す（R0 t30 で失敗した）。"""
    try:
        return torch.linalg.eigvalsh(G.double()).flip(0)
    except RuntimeError:
        m = G.shape[0]
        return torch.linalg.eigvalsh(G.double().cpu() + 1e-9 * float(G.trace()) / m * torch.eye(m, dtype=torch.float64)).flip(0).to(G.device)


def gram_stats(G, old=None):
    """old は前の課題（この状態が暗記したラベル）の 256 枚のラベル。同じ旧クラスの対と違う旧クラスの対で cos を分ける。"""
    e = eigvals(G)
    n = torch.diag(G).sqrt()
    cosG = G / (n[:, None] * n[None, :])
    ec = eigvals(cosG)
    off = cosG - torch.diag(torch.diag(cosG))
    m = G.shape[0]
    order = n.argsort()
    bulk = order[:int(0.9 * m)]                      # ノルムの下位 90%（大きい少数の画像を除いた本体）
    eb = eigvals(G[bulk][:, bulk])
    d = dict(pr=participation(e), pr_cos=participation(ec), pr_bulk90=participation(eb), top1_share=float(e[0] / e.sum()),
             top10_share=float(e[:10].sum() / e.sum()), mean_abs_cos=float(off.abs().sum() / (m * (m - 1))),
             mean_cos=float(off.sum() / (m * (m - 1))), norm_med=float(n.median()), norm_mean=float(n.mean()),
             norm_quantiles=[float(n.quantile(q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)],
             trace_share_top5pct=float((n[order[-int(0.05 * m):]] ** 2).sum() / (n ** 2).sum()),
             norms=[round(float(v), 4) for v in n],
             eig_quantiles=[float(e[int(q * (m - 1))] / e[0]) for q in (0.1, 0.25, 0.5, 0.75, 0.9)])
    if old is not None:
        same = (old[:, None] == old[None, :]) & ~torch.eye(m, dtype=torch.bool, device=G.device)
        diff = (old[:, None] != old[None, :])
        d['cos_same_old'] = float(cosG[same].mean())
        d['cos_diff_old'] = float(cosG[diff].mean())
        d['abs_cos_same_old'] = float(cosG[same].abs().mean())
        d['abs_cos_diff_old'] = float(cosG[diff].abs().mean())
    return d


def directional_gram(model, params, x, u):
    """g_m = ∂(u·h_m)/∂θ の Gram（fp16 で貯め、fp32 で積む）。"""
    rows = []
    for m in range(x.shape[0]):
        model.zero_grad(set_to_none=True)
        s = (cls_features(model, x[m:m + 1])[0] * u).sum()
        g = torch.autograd.grad(s, params)
        rows.append(torch.cat([t.reshape(-1) for t in g]).half())
    X = torch.stack(rows)
    G = torch.zeros(len(rows), len(rows), device=x.device)
    for i in range(0, len(rows), 32):
        Xi = X[i:i + 32].float()
        for j in range(0, len(rows), 32):
            G[i:i + 32, j:j + 32] = Xi @ X[j:j + 32].float().T
    del X, rows
    return G


def main(arm, t, out):
    C.set_determinism(4)
    device = 'cuda'
    model = C.build_arm_model(arm, SEED, device, 0.0, 2, 8.0)
    model.load_state_dict(torch.load(RUNS / arm / f'seed{SEED}' / f'ckpt/model_t{t:02d}.pt', map_location=device, weights_only=False))
    model.eval()
    params = [p for n, p in model.named_parameters() if not n.startswith('layers.out')]
    source = C.image_source('real')
    data, _ = C.task_dataset(source, SEED, t + 1, 10, 250, augment=False)
    x_all = C.eval_tensor(data.images, device)
    with torch.no_grad():
        H = torch.cat([cls_features(model, x_all[s:s + 128]) for s in range(0, x_all.shape[0], 128)])
    Hc = H - H.mean(0)
    _, S, Vh = torch.linalg.svd(Hc, full_matrices=False)
    pcs = Vh[:N_DIR]
    g = torch.Generator().manual_seed(0)
    idx = torch.randperm(x_all.shape[0], generator=g)[:N_IMG].to(device)
    x = x_all[idx]
    old = C.task_dataset(source, SEED, t, 10, 250, augment=False)[0].labels.to(device)[idx] if t > 0 else None
    rand = torch.randn(N_DIR, H.shape[1], generator=g).to(device)
    rand = rand / rand.norm(dim=1, keepdim=True)
    res = dict(arm=arm, task=t, n_img=N_IMG, n_params=int(sum(p.numel() for p in params)),
               feature_pr_2500=participation(S ** 2), feature_var_top4=float((S[:N_DIR] ** 2).sum() / (S ** 2).sum()))
    with torch.no_grad():
        Hs = H[idx]
        res['feature_gram'] = gram_stats(Hs @ Hs.T, old)
        res['feature_gram_centered'] = gram_stats((Hs - Hs.mean(0)) @ (Hs - Hs.mean(0)).T, old)
    t0 = time.monotonic()
    for name, U in (('rand', rand), ('pc', pcs)):
        res[name] = [gram_stats(directional_gram(model, params, x, U[k]), old) for k in range(N_DIR)]
    res['seconds'] = round(time.monotonic() - t0, 1)
    Path(out).write_text(json.dumps(res, indent=1))
    f = lambda key, k: sum(d[k] for d in res[key]) / N_DIR
    print(f"{arm} t{t}: 特徴の参加比 {res['feature_pr_2500']:.1f} | 接核 rand: pr {f('rand', 'pr'):.1f} pr_cos {f('rand', 'pr_cos'):.1f} "
          f"|cos| {f('rand', 'mean_abs_cos'):.3f} ‖g‖ {f('rand', 'norm_med'):.3g} | pc: pr {f('pc', 'pr'):.1f} pr_cos {f('pc', 'pr_cos'):.1f} "
          f"|cos| {f('pc', 'mean_abs_cos'):.3f} ‖g‖ {f('pc', 'norm_med'):.3g} ({res['seconds']} s)", flush=True)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3])
