"""保存状態（課題末）をまたぐ平均ベクトルの一貫性（Fable エージェント A、依頼 4 の A-4、1028）。

R0 と S の model_t{00..10,20,...,100} を CPU の float64 で読み、固定 probe（val 64 枚）で
FFN0・注意 0・注意 1・埋め込みの平均 patch 出力ベクトル、m2（最終 FFN の入力の token 平均）、各段の patch rms・CLS rms、FFN0 の開口を出す。
列 t1..t10（1 課題ずつ）と t10,20,...,40 / t10..100（10 課題ずつ）で、増分の一貫性 C = ‖X_end − X_start‖² / Σ‖ΔX_k‖²、
隣り合う増分の cos、増分と直前の位置の cos（半径成分）、ノルムの列を出す。
増分が互いに直交（乱歩）なら C = 1、同じ向きに積むなら C = 区間数、行って戻るなら C < 1。
増分の大きさが区間ごとに大きく違うと C は最大の増分に支配されるので、隣の増分の cos と半径成分の cos、始点と終点の cos を併記する。

    PYTHONPATH=$PWD nice -n 10 python analysis/vit_long_1024/posthoc_2blk/fable_r0form_coherence.py
"""
import json
import time
from pathlib import Path

import numpy as np
import torch

from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import C, ROOT, SEED, ckpt, images
from analysis.vit_long_1024.posthoc_2blk.fable_r0form_replay import NODES, now, progress, resources, rms_norm, run_nodes

OUT = ROOT / 'posthoc_r0_form'
TASKS = list(range(0, 11)) + list(range(20, 101, 10))
KEYS = ['ffn0_patch', 'attn0_patch', 'attn1_patch', 'e_patch', 'm2', 'm2_patch', 'ffn0_cls', 'attn1_cls', 'h1_patch', 'u2_patch']


@torch.no_grad()
def measure(arm, task, xp):
    m = C.build_arm_model(arm, SEED, 'cpu', 0., 2, 8.)
    m.load_state_dict(torch.load(ckpt(arm, task), map_location='cpu', weights_only=False))
    m = m.double().eval()
    nd, h = run_nodes(m, xp)
    row = {}
    for k in NODES:
        row[f'{k}_patch_rms'] = rms_norm(nd[k][:, 1:])
        row[f'{k}_cls_rms'] = rms_norm(nd[k][:, 0])
    vec = dict(ffn0_patch=nd['ffn0'][:, 1:].mean((0, 1)), attn0_patch=nd['attn0'][:, 1:].mean((0, 1)),
               attn1_patch=nd['attn1'][:, 1:].mean((0, 1)), e_patch=nd['e'][:, 1:].mean((0, 1)),
               m2=nd['u2'].mean((0, 1)), m2_patch=nd['u2'][:, 1:].mean((0, 1)),
               ffn0_cls=nd['ffn0'][:, 0].mean(0), attn1_cls=nd['attn1'][:, 0].mean(0),
               h1_patch=nd['h1'][:, 1:].mean((0, 1)), u2_patch=nd['u2'][:, 1:].mean((0, 1)))
    row['ffn0_open_patch'] = float((nd['z0'][:, 1:] > 0).double().mean())
    row['z1_dead'] = float(((nd['z1'].flatten(0, 1) > 0).double().mean(0) <= .005).double().mean())
    row['z1_mean'] = float(nd['z1'].mean())
    row['head_norm'] = float(m.layers['out'].weight.norm())
    b0 = m.layers['block_0'].layers
    row['ffn0_fc1_norm'] = float(b0['mlp'].layers['fc1'].weight.norm())
    row['ffn0_fc2_norm'] = float(b0['mlp'].layers['fc2'].weight.norm())
    row['ffn0_fc2_bias_norm'] = float(b0['mlp'].layers['fc2'].bias.norm())
    row['attn0_qkv_norm'] = float(b0['attn'].layers['qkv'].weight.norm())
    row['attn0_proj_norm'] = float(b0['attn'].layers['proj'].weight.norm())
    row['patch_embed_norm'] = float(sum(p.norm() ** 2 for p in m.layers['patch_embed'].parameters()) ** .5)
    # FFN0 の平均出力の分解: W2 ā + b2（ā は開いた値の平均）
    act0 = torch.relu(nd['z0'][:, 1:]).reshape(-1, nd['z0'].shape[-1])
    abar = act0.mean(0)
    row['ffn0_abar_norm'] = float(abar.norm())
    row['ffn0_abar_l1'] = float(abar.abs().sum())
    row['ffn0_act_rms_open'] = float(act0[act0 > 0].square().mean().sqrt()) if (act0 > 0).any() else 0.
    W2 = b0['mlp'].layers['fc2'].weight
    row['ffn0_W2abar_norm'] = float((W2 @ abar).norm())
    row['ffn0_gain'] = float((W2 @ abar).norm() / abar.norm()) if abar.norm() > 0 else 0.
    return row, {k: v.numpy() for k, v in vec.items()}


def coherence(seq):
    X = np.stack(seq)
    d = np.diff(X, axis=0)
    total = X[-1] - X[0]
    out = dict(C=float((total ** 2).sum() / (d ** 2).sum()), n=len(d), norms=[float(np.linalg.norm(v)) for v in X],
               step_norms=[float(np.linalg.norm(v)) for v in d],
               adjacent_cos=[float(d[i] @ d[i + 1] / (np.linalg.norm(d[i]) * np.linalg.norm(d[i + 1]) + 1e-30)) for i in range(len(d) - 1)],
               radial_cos=[float(d[i] @ X[i] / (np.linalg.norm(d[i]) * np.linalg.norm(X[i]) + 1e-30)) for i in range(len(d))],
               cos_total_vs_start=float(total @ X[0] / (np.linalg.norm(total) * np.linalg.norm(X[0]) + 1e-30)),
               cos_end_vs_start=float(X[-1] @ X[0] / (np.linalg.norm(X[-1]) * np.linalg.norm(X[0]) + 1e-30)))
    out['adjacent_cos_mean'] = float(np.mean(out['adjacent_cos'])) if out['adjacent_cos'] else None
    out['radial_cos_mean'] = float(np.mean(out['radial_cos']))
    return out


def main():
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    xp = images('cpu', train=False).double()
    start = time.monotonic()
    rows, vecs = {}, {}
    for arm in ['R0', 'S']:
        for t in TASKS:
            row, v = measure(arm, t, xp)
            rows[f'{arm}_t{t}'] = row
            vecs[f'{arm}_t{t}'] = v
            print(f'{arm} t{t:3d} ffn0_patch_rms {row["ffn0_patch_rms"]:9.1f} ffn0_mean {np.linalg.norm(v["ffn0_patch"]):9.1f} '
                  f'm2 {np.linalg.norm(v["m2"]):9.1f} open {row["ffn0_open_patch"]:.3f} gain {row["ffn0_gain"]:.2f} sec {time.monotonic() - start:.0f}', flush=True)
    spans = {'t1_10': list(range(1, 11)), 't0_10': list(range(0, 11)), 't10_40': [10, 20, 30, 40], 't10_60': [10, 20, 30, 40, 50, 60],
             't10_100': list(range(10, 101, 10)), 't40_100': list(range(40, 101, 10))}
    coh = {}
    for arm in ['R0', 'S']:
        for key in KEYS:
            for name, ts in spans.items():
                coh[f'{arm}_{key}_{name}'] = coherence([vecs[f'{arm}_t{t}'][key] for t in ts])
    np.savez(OUT / 'fable_coherence_vectors.npz', **{f'{k}__{kk}': vv for k, v in vecs.items() for kk, vv in v.items()})
    (OUT / 'fable_coherence.json').write_text(json.dumps(dict(rows=rows, coherence=coh, tasks=TASKS, spans=spans,
                                                              resources=resources(), seconds=time.monotonic() - start), ensure_ascii=False, indent=1))
    progress(f'一貫性（A-4）完了: {len(rows)} 状態、資源 {resources()}、{time.monotonic() - start:.0f} 秒。')


if __name__ == '__main__':
    main()
