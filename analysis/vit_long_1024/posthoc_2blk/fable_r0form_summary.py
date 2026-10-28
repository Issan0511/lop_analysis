"""回し直しの時系列を 3 相に割り、力の符号との相関と予測 A1〜A7 の判定量を出す（Fable エージェント A、依頼 4、1028）。

入力: posthoc_r0_form/fable_replay_*.json（fable_r0form_replay.py）、fable_coherence.json（fable_r0form_coherence.py）。
出力: posthoc_r0_form/fable_summary.json と、報告に貼る markdown の表（fable_summary_tables.md）。

相の切り方: 切替相 = 0 から暗記率が初めて 0.2 以上になる記録（t_rise）まで（固定 200 更新の値も併記）、当てはめ相 = t_rise から T90、
当てはめ後 = T90 から 5,000。尺度は log（自然対数）の差で比べる。

    PYTHONPATH=$PWD python analysis/vit_long_1024/posthoc_2blk/fable_r0form_summary.py
"""
import json
import math
from pathlib import Path

import numpy as np

from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import ROOT


def spearmanr(x, y, permutations=4000, seed=0):
    """Spearman の順位相関と並べ替え p 値（scipy なし）。"""
    def ranks(v):
        v = np.asarray(v, dtype=float)
        order = v.argsort()
        r = np.empty(len(v))
        r[order] = np.arange(len(v))
        # 同順位は平均順位
        for val in np.unique(v):
            idx = np.where(v == val)[0]
            if len(idx) > 1:
                r[idx] = r[idx].mean()
        return r
    rx, ry = ranks(x), ranks(y)
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(permutations):
        if abs(np.corrcoef(rx, rng.permutation(ry))[0, 1]) >= abs(rho):
            count += 1
    return rho, (count + 1) / (permutations + 1)

OUT = ROOT / 'posthoc_r0_form'
SCALES = ['e_patch_rms', 'attn0_patch_rms', 'ffn0_patch_rms', 'h1_patch_rms', 'attn1_patch_rms', 'u2_patch_rms',
          'u2_cls_rms', 'h2_cls_rms', 'm2_mean_norm', 'ffn0_patch_mean_norm', 'attn1_cls_rms']
MAIN = 'ffn0_patch_rms'


def load_runs():
    runs = {}
    for f in sorted(OUT.glob('fable_replay_*.json')):
        d = json.loads(f.read_text())
        if d['steps'] < 5000:
            continue
        frz = d.get('freeze_body_until', 0)
        runs[f'{d["arm"]}_t{d["task_end"]}_{d["adam"]}' + (f'_frz{frz}' if frz else '')] = d
    return runs


def at(records, step):
    for r in records:
        if r['step'] == step:
            return r
    raise KeyError(step)


def dlog(records, key, a, b):
    return math.log(at(records, b)[key]) - math.log(at(records, a)[key])


def phases(d):
    rec = d['records']
    t_rise = next((r['step'] for r in rec if r['acc'] >= .2), None)
    t90 = d['T90']
    end = rec[-1]['step']
    out = dict(t_rise=t_rise, T90=t90, end=end, acc_end=rec[-1]['acc'], acc_last1000=float(np.mean([r['acc'] for r in rec if r['step'] > end - 1000])))
    for key in SCALES:
        p = dict(total=dlog(rec, key, 0, end), first200=dlog(rec, key, 0, 200), first100=dlog(rec, key, 0, 100), first50=dlog(rec, key, 0, 50))
        p['switch'] = dlog(rec, key, 0, t_rise) if t_rise is not None else None
        fit_start = t_rise if t_rise is not None else 0
        if t90 is not None:
            p['fit'] = dlog(rec, key, fit_start, t90)
            p['post'] = dlog(rec, key, t90, end)
            p['post_share'] = p['post'] / p['total'] if abs(p['total']) > 1e-9 else None
        else:
            p['fit'] = dlog(rec, key, fit_start, end)
            p['post'] = None
            p['post_share'] = None
        # 切替直後の過渡: 最初の 200 更新の中で log 尺度が最大・最小になる更新とその値（更新 0 比）
        base = math.log(rec[0][key])
        inside = [(r['step'], math.log(r[key]) - base) for r in rec if 0 < r['step'] <= 200]
        p['peak_step'], p['peak_dlog'] = max(inside, key=lambda v: v[1])
        p['trough_step'], p['trough_dlog'] = min(inside, key=lambda v: v[1])
        p['first25'] = dlog(rec, key, 0, 25)
        # 当てはめ相の 1 更新あたりの log 増分（Adam の定歩幅なら一定のはず）
        fit_len = (t90 if t90 is not None else end) - fit_start
        p['fit_rate_per_1000'] = 1000 * p['fit'] / fit_len if fit_len > 0 else None
        p['post_rate_per_1000'] = 1000 * p['post'] / (end - t90) if (t90 is not None and end > t90) else None
        out[key] = p
    return out


def early_stats(d):
    e = d['early']
    if not e:
        return {}
    steps = [x['body_step_norm'] for x in e]
    g = [x['body_grad_norm'] for x in e]
    out = dict(body_grad_norm_step1=g[0], body_step_norm_step1=steps[0], body_step_norm_step10=steps[9], body_step_norm_step50=steps[49],
               body_step_norm_step200=steps[-1], sum_body_step_1_10=float(np.sum(steps[:10])), sum_body_step_1_50=float(np.sum(steps[:50])),
               sum_body_step_1_200=float(np.sum(steps)), head_norm_step1=e[0]['head_norm'], head_norm_step10=e[9]['head_norm'],
               head_norm_step50=e[49]['head_norm'], head_norm_step200=e[-1]['head_norm'])
    # 体の勾配が head の勾配の 1% を初めて超える更新
    out['step_body_grad_exceeds_1pct_head'] = next((x['step'] for x in e if x['body_grad_norm'] > .01 * x['head_grad_norm']), None)
    return out


def force_correlation(d, key=MAIN, horizon=50):
    """窓の始めの −∂L/∂α と、その後 horizon 更新の log 尺度の増分の Spearman。"""
    rec = {r['step']: r for r in d['records']}
    xs, ys, rows = [], [], []
    for s in d['scale_derivative']:
        t = s['step']
        if t + horizon not in rec or t not in rec:
            continue
        x = -s['dL_dalpha_ffn0']
        y = math.log(rec[t + horizon][key]) - math.log(rec[t][key])
        xs.append(x)
        ys.append(y)
        rows.append(dict(step=t, minus_dLdalpha=x, dlog=y, acc=s['acc'], correct=s['dL_dalpha_ffn0_correct'], wrong=s['dL_dalpha_ffn0_wrong'],
                         frac_negative=s['dL_dalpha_ffn0_frac_negative'], minus_dLdalpha_h=-s['dL_dalpha_h'],
                         dacc=rec[t + horizon]['acc'] - rec[t]['acc'], loss=rec[t]['loss'], misfit=1 - rec[t]['acc']))
    out = dict(n=len(xs))
    if len(xs) >= 5:
        rho, p = spearmanr(xs, ys)
        out.update(spearman=float(rho), p=float(p))
        # 崩れ・未当てはめの窓と成長: 窓の暗記率の変化 dacc、窓の始めの損失との Spearman。崩れ窓（dacc ≤ −0.1）・上昇窓（≥ +0.1）・平らな窓の平均 dlog
        r1, p1 = spearmanr([r['dacc'] for r in rows], ys)
        r2, p2 = spearmanr([r['loss'] for r in rows], ys)
        out.update(spearman_dacc=float(r1), p_dacc=float(p1), spearman_loss=float(r2), p_loss=float(p2))
        for name, cond in (('crash', lambda r: r['dacc'] <= -.1), ('rise', lambda r: r['dacc'] >= .1), ('flat', lambda r: abs(r['dacc']) < .1)):
            sel = [r['dlog'] for r in rows if cond(r)]
            out[f'mean_dlog_{name}'] = float(np.mean(sel)) if sel else None
            out[f'n_{name}'] = len(sel)
        for name, cond in (('unfit', lambda r: r['acc'] < .9), ('fit', lambda r: r['acc'] >= .9)):
            sel = [r['dlog'] for r in rows if cond(r)]
            out[f'mean_dlog_{name}'] = float(np.mean(sel)) if sel else None
            out[f'sum_dlog_{name}'] = float(np.sum(sel)) if sel else None
            out[f'n_{name}'] = len(sel)
            # 崩れ窓の ∂L/∂α の符号（成長側 = −∂L/∂α > 0 の窓の割合）
        crash = [r for r in rows if r['dacc'] <= -.1]
        out['crash_frac_force_growth'] = float(np.mean([r['minus_dLdalpha'] > 0 for r in crash])) if crash else None
        out['crash_mean_minus_dLdalpha'] = float(np.mean([r['minus_dLdalpha'] for r in crash])) if crash else None
        # 平らな窓だけ: 力の符号 −∂L/∂α と増分の Spearman と符号の一致（穏やかな漂いが尺度微分に従うか）
        flat = [r for r in rows if abs(r['dacc']) < .1]
        if len(flat) >= 5:
            r3, p3 = spearmanr([r['minus_dLdalpha'] for r in flat], [r['dlog'] for r in flat])
            out.update(spearman_flat_force=float(r3), p_flat_force=float(p3), n_flat_force=len(flat),
                       flat_sign_agreement=float(np.mean([(r['minus_dLdalpha'] > 0) == (r['dlog'] > 0) for r in flat if abs(r['minus_dLdalpha']) > 1e-6])),
                       flat_mean_force=float(np.mean([r['minus_dLdalpha'] for r in flat])), flat_mean_dlog=float(np.mean([r['dlog'] for r in flat])))
        # 当てはめ相（0.2 ≤ acc < 0.9）だけ
        sel = [(x, y) for x, y, r in zip(xs, ys, rows) if .2 <= r['acc'] < .9]
        if len(sel) >= 5:
            rho2, p2 = spearmanr([a for a, _ in sel], [b for _, b in sel])
            out.update(spearman_fit=float(rho2), p_fit=float(p2), n_fitphase=len(sel))
        sign_agree = float(np.mean([(x > 0) == (y > 0) for x, y in zip(xs, ys) if abs(x) > 1e-6]))
        out['sign_agreement'] = sign_agree
    out['rows'] = rows
    return out


def sign_table(d):
    """相ごとの ∂L/∂α の符号: 切替相（acc < 0.2）・当てはめ相（0.2–0.9）・当てはめ後（≥ 0.9）。"""
    out = {}
    for name, lo, hi in (('switch', -1, .2), ('fit', .2, .9), ('post', .9, 2)):
        sel = [s for s in d['scale_derivative'] if lo <= s['acc'] < hi]
        if not sel:
            out[name] = None
            continue
        out[name] = dict(n=len(sel), dL_dalpha_ffn0_mean=float(np.mean([s['dL_dalpha_ffn0'] for s in sel])),
                         frac_windows_growth=float(np.mean([s['dL_dalpha_ffn0'] < 0 for s in sel])),
                         dL_dalpha_h_mean=float(np.mean([s['dL_dalpha_h'] for s in sel])),
                         correct_mean=float(np.mean([s['dL_dalpha_ffn0_correct'] for s in sel if s['dL_dalpha_ffn0_correct'] is not None])),
                         wrong_mean=float(np.mean([s['dL_dalpha_ffn0_wrong'] for s in sel if s['dL_dalpha_ffn0_wrong'] is not None])))
    s0 = d['scale_derivative'][0]
    out['step0'] = dict(dL_dalpha_ffn0=s0['dL_dalpha_ffn0'], dL_dalpha_h=s0['dL_dalpha_h'], acc=s0['acc'])
    return out


def fmt(v, nd=3):
    if v is None:
        return '—'
    if isinstance(v, float):
        return f'{v:+.{nd}f}' if abs(v) < 100 else f'{v:,.0f}'
    return str(v)


def main():
    runs = load_runs()
    summary = {k: dict(phases=phases(d), early=early_stats(d), force=force_correlation(d), signs=sign_table(d), seconds=d['seconds'], resources=d['resources'])
               for k, d in runs.items()}
    coh = json.loads((OUT / 'fable_coherence.json').read_text()) if (OUT / 'fable_coherence.json').exists() else None
    lines = []
    lines.append('## 相ごとの log 尺度の増分（FFN0 出力の patch rms）\n')
    lines.append('| 走 | t_rise | T90 | 末の暗記率 | 最初の 25 | 過渡の山（更新: log） | 切替相（0→t_rise） | 最初の 200 | 当てはめ相 | 率/1,000 更新 | 当てはめ後 | 率/1,000 更新 | 当てはめ後の割合 | 合計 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k, s in summary.items():
        p = s['phases']
        m = p[MAIN]
        lines.append(f'| {k} | {p["t_rise"]} | {p["T90"]} | {p["acc_end"]:.3f} | {fmt(m["first25"])} | {m["peak_step"]}: {m["peak_dlog"]:+.3f} | {fmt(m["switch"])} | {fmt(m["first200"])} | {fmt(m["fit"])} | {fmt(m["fit_rate_per_1000"])} | {fmt(m["post"])} | {fmt(m["post_rate_per_1000"])} | {fmt(m["post_share"], 2)} | {fmt(m["total"])} |')
    lines.append('\n## 切替相（最初の 200 更新）の log 増分、段ごと\n')
    keys = ['e_patch_rms', 'attn0_patch_rms', 'ffn0_patch_rms', 'h1_patch_rms', 'attn1_patch_rms', 'u2_cls_rms', 'h2_cls_rms', 'm2_mean_norm']
    lines.append('| 走 | ' + ' | '.join(keys) + ' |')
    lines.append('|---|' + '---|' * len(keys))
    for k, s in summary.items():
        lines.append(f'| {k} | ' + ' | '.join(fmt(s['phases'][kk]['first200']) for kk in keys) + ' |')
    lines.append('\n## 合計（0→5,000）の log 増分、段ごと\n')
    lines.append('| 走 | ' + ' | '.join(keys) + ' |')
    lines.append('|---|' + '---|' * len(keys))
    for k, s in summary.items():
        lines.append(f'| {k} | ' + ' | '.join(fmt(s['phases'][kk]['total']) for kk in keys) + ' |')
    lines.append('\n## 最初の 200 更新の体の動き\n')
    lines.append('| 走 | 体の勾配ノルム（更新 1） | 体の 1 歩（更新 1 / 10 / 50 / 200） | 体の歩みの和（1–10 / 1–50 / 1–200） | head ノルム（1 / 10 / 50 / 200） | 体の勾配が head の 1% を超える更新 |')
    lines.append('|---|---|---|---|---|---|')
    for k, s in summary.items():
        e = s['early']
        lines.append(f'| {k} | {e["body_grad_norm_step1"]:.3g} | {e["body_step_norm_step1"]:.3g} / {e["body_step_norm_step10"]:.3g} / {e["body_step_norm_step50"]:.3g} / {e["body_step_norm_step200"]:.3g} | '
                     f'{e["sum_body_step_1_10"]:.3g} / {e["sum_body_step_1_50"]:.3g} / {e["sum_body_step_1_200"]:.3g} | {e["head_norm_step1"]:.3g} / {e["head_norm_step10"]:.3g} / {e["head_norm_step50"]:.3g} / {e["head_norm_step200"]:.3g} | {e["step_body_grad_exceeds_1pct_head"]} |')
    lines.append('\n## 力の符号 ∂L/∂α（FFN0 の patch 出力の倍率）と 50 更新先の log 増分の相関\n')
    lines.append('| 走 | 更新 0 の ∂L/∂α | 切替相の平均（窓数） | 当てはめ相の平均（窓数） | 当てはめ後の平均（窓数） | 当たり画像の平均（当てはめ相） | 外れ画像の平均（当てはめ相） | Spearman 全窓 | Spearman 当てはめ相 | 符号の一致 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|')
    for k, s in summary.items():
        sg, f = s['signs'], s['force']
        def cell(ph):
            return f'{sg[ph]["dL_dalpha_ffn0_mean"]:+.3g}（{sg[ph]["n"]}）' if sg.get(ph) else '—'
        fitc = f'{sg["fit"]["correct_mean"]:+.3g}' if sg.get('fit') else '—'
        fitw = f'{sg["fit"]["wrong_mean"]:+.3g}' if sg.get('fit') else '—'
        lines.append(f'| {k} | {sg["step0"]["dL_dalpha_ffn0"]:+.3g} | {cell("switch")} | {cell("fit")} | {cell("post")} | {fitc} | {fitw} | '
                     f'{fmt(f.get("spearman"), 2)} | {fmt(f.get("spearman_fit"), 2)}（{f.get("n_fitphase", 0)}） | {fmt(f.get("sign_agreement"), 2)} |')
    lines.append('\n## 崩れ・未当てはめの 50 更新窓と尺度の成長（FFN0 出力の patch rms の log 増分）\n')
    lines.append('崩れ窓 = 窓の中で暗記率が 0.1 以上落ちた窓、上昇窓 = 0.1 以上上がった窓、未当てはめ窓 = 窓の始めの暗記率 < 0.9。\n')
    lines.append('| 走 | Spearman（Δ暗記率, Δlog） | Spearman（窓始めの損失, Δlog） | 崩れ窓の平均 Δlog（数） | 上昇窓（数） | 平ら（数） | 未当てはめ窓の Δlog の和（数） | 当てはめ済み窓の和（数） | 崩れ窓で −∂L/∂α > 0 の割合 | 平らな窓: Spearman（−∂L/∂α, Δlog） | 平らな窓の符号の一致 | 平らな窓の −∂L/∂α の平均 / Δlog の平均 |')
    lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
    for k, s in summary.items():
        f = s['force']
        if 'spearman_dacc' not in f:
            continue
        lines.append(f'| {k} | {f["spearman_dacc"]:+.2f}（p {f["p_dacc"]:.3f}） | {f["spearman_loss"]:+.2f}（p {f["p_loss"]:.3f}） | {fmt(f["mean_dlog_crash"])}（{f["n_crash"]}） | {fmt(f["mean_dlog_rise"])}（{f["n_rise"]}） | '
                     f'{fmt(f["mean_dlog_flat"])}（{f["n_flat"]}） | {fmt(f["sum_dlog_unfit"])}（{f["n_unfit"]}） | {fmt(f["sum_dlog_fit"])}（{f["n_fit"]}） | {fmt(f["crash_frac_force_growth"], 2)} | '
                     f'{fmt(f.get("spearman_flat_force"), 2)}（p {f.get("p_flat_force", float("nan")):.3f}、{f.get("n_flat_force", 0)}） | {fmt(f.get("flat_sign_agreement"), 2)} | {fmt(f.get("flat_mean_force"), 4)} / {fmt(f.get("flat_mean_dlog"), 4)} |')
    if coh:
        lines.append('\n## 課題をまたぐ増分の一貫性 C（保存状態、固定 probe）\n')
        lines.append('直交する増分（乱歩）なら C = 1、同じ向きに積むなら C = 区間数、行って戻るなら C < 1。増分の大きさが偏ると C は最大の増分に支配される。\n')
        lines.append('| 腕 | ベクトル | 区間 | 区間数 n | C | C/n | 隣の増分の cos の平均 | 半径成分 cos の平均 | cos(終, 始) | ノルム（始→終） |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|')
        for arm in ['R0', 'S']:
            for key in ['ffn0_patch', 'attn0_patch', 'attn1_patch', 'm2', 'e_patch']:
                for span in ['t1_10', 't10_40', 't10_100', 't40_100']:
                    c = coh['coherence'][f'{arm}_{key}_{span}']
                    lines.append(f'| {arm} | {key} | {span} | {c["n"]} | {c["C"]:.2f} | {c["C"] / c["n"]:.2f} | {fmt(c["adjacent_cos_mean"], 2)} | {c["radial_cos_mean"]:+.2f} | {c["cos_end_vs_start"]:+.2f} | {c["norms"][0]:,.1f} → {c["norms"][-1]:,.1f} |')
        lines.append('\n## 保存状態の FFN0 の平均出力の分解（patch token）\n')
        lines.append('| 腕 | 課題末 | FFN0 出力 patch rms | 平均出力ノルム ‖W2 ā + b2‖ 相当 | ‖ā‖ | 開いた値の rms | 開口 | ‖W2 ā‖/‖ā‖（利得） | fc2 ノルム | fc1 ノルム | qkv0 ノルム | 埋め込みノルム |')
        lines.append('|---|---|---|---|---|---|---|---|---|---|---|---|')
        for arm in ['R0', 'S']:
            for t in [1, 5, 10, 20, 30, 40, 60, 100]:
                r = coh['rows'][f'{arm}_t{t}']
                lines.append(f'| {arm} | {t} | {r["ffn0_patch_rms"]:,.1f} | {r["ffn0_W2abar_norm"]:,.1f} | {r["ffn0_abar_norm"]:.2f} | {r["ffn0_act_rms_open"]:.2f} | {r["ffn0_open_patch"]:.3f} | {r["ffn0_gain"]:.2f} | {r["ffn0_fc2_norm"]:.1f} | {r["ffn0_fc1_norm"]:.1f} | {r["attn0_qkv_norm"]:.1f} | {r["patch_embed_norm"]:.1f} |')
    (OUT / 'fable_summary_tables.md').write_text('\n'.join(lines) + '\n')
    (OUT / 'fable_summary.json').write_text(json.dumps(dict(runs=summary), ensure_ascii=False, indent=1))
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
