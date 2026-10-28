"""posthoc_r0_late の回し直し・上界の JSON を集計して表（markdown）と fable_r0late_summary.json を出す（Fable エージェント B、1028）。"""
import json
import statistics as st
from pathlib import Path

ROOT = Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix')
OUT = ROOT / 'posthoc_r0_late'
PREV = ROOT / 'posthoc_r0'
GROUPS = ['block0', 'attn1', 'ffn1', 'head', 'body']


def load(name, prev=False):
    p = (PREV if prev else OUT) / f'{name}.json'
    return json.load(p.open()) if p.exists() else None


def trace_stats(d, steps=5000):
    r = d['records']
    T = d.get('T90') if 'summary' not in d else d['summary']['T90']
    accs = [(x['step'], x['acc']) for x in r]
    after = [(s, a) for s, a in accs if T is not None and s >= T]
    drops = [(s1, a0 - a1) for (s0, a0), (s1, a1) in zip(after, after[1:]) if a0 - a1 >= .1]
    tail = [a for s, a in accs if s > steps - 1000]
    below = [s for s, a in after if a < .9]
    at = {x['step']: x for x in r}
    t90row = at.get(T) if T is not None else None
    end = r[-1]
    return dict(T90=T, start=r[0]['step'], acc_start=r[0]['acc'], final=r[-1]['acc'], tail_mean=st.mean(tail) if tail else None,
                tail_min=min(tail) if tail else None, min_after=min([a for s, a in after]) if after else None,
                n_drops=len(drops), first_drop=drops[0][0] if drops else None,
                frac_below90_after=(len(below) / len(after)) if after else None, n_after=len(after),
                margin_t90=t90row.get('margin_median') if t90row else None, margin_end=end.get('margin_median'),
                margin_q10_end=end.get('margin_q10'), h_t90=t90row.get('h_norm_median') if t90row else None,
                h_end=end.get('h_norm_median'), margin_max=max([x.get('margin_median', 0) for x in r]))


def diag_stats(d, lo, hi):
    rows = [q for q in d.get('step_diagnostics', []) if lo <= q['step'] <= hi]
    if not rows:
        return None
    med = lambda k: st.median([q[k] for q in rows if k in q]) if any(k in q for q in rows) else None
    out = dict(n=len(rows), total=med('total'), head=med('head'), body=med('body'), cross=med('cross'),
               dmargin_abs_median=med('dmargin_abs_median'), dmargin_rms=med('dmargin_rms'), margin_before=med('margin_median_before'),
               flip_frac=med('flip_frac'), head_coord_max=med('head_coord_max'))
    g = [q['body_groups'] for q in rows if 'body_groups' in q]
    if g:
        out['groups'] = {k: st.median([x[k] for x in g]) for k in g[0]}
        out['n_groups'] = len(g)
    return out


def main():
    summary = {'runs': {}, 'diag': {}, 'bound': {}}
    names = {}
    for task in (60, 30):
        b = f'fable_R0_t{task}_hl001_'
        names[task] = [(b + 'base', 'head lr 1/100（分岐の元）'), (b + 'br_body', '届いた時点で体を全部凍結'),
                       (b + 'br_head', '届いた時点で head を凍結'), (b + 'br_ffn1', '最終 FFN だけ凍結'),
                       (b + 'br_attn1', '注意 1 だけ凍結'), (b + 'br_block0', 'ブロック 0（埋め込み含む）だけ凍結'),
                       (b + 'br_bl01', '届いた時点から体の lr 1/10'), (b + 'br_newhead1', '届いた時点の体を凍結・新 head（lr 1e-4）'),
                       (b + 'br_newhead001', '届いた時点の体を凍結・新 head（lr 1e-6）'), (b + 'bl001', '頭から head・体とも 1/100'),
                       (b + 'bl01', '頭から head 1/100・体 1/10')]
        for name, label in names[task]:
            d = load(name)
            if d is None:
                continue
            s = trace_stats(d)
            s.update(label=label, freeze=d.get('freeze'), new_head=d.get('new_head'), seconds=d.get('seconds'),
                     resources=d.get('resources'))
            summary['runs'][name] = s
            T = s['T90'] if s['T90'] is not None else s['start']
            summary['diag'][name] = dict(window=diag_stats(d, T, T + 400), pre=diag_stats(d, max(0, T - 100), T - 1),
                                         late=diag_stats(d, 4000, 5000), all_after=diag_stats(d, T, 5000))
        for name in ('codex_replay_R0_t%d_lr001_full' % task, 'codex_replay_R0_t%d_base' % task):
            d = load(name, prev=True)
            if d is not None:
                s = trace_stats(d)
                s['label'] = '依頼 3 の ' + ('head lr 1/100' if 'lr001' in name else '元の lr')
                summary['runs'][name] = s
        bnd = load(f'fable_bound_R0_t{task}_t90')
        if bnd:
            summary['bound'][task] = bnd['summary']
    (OUT / 'fable_r0late_summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=1, allow_nan=False))
    # markdown
    f = lambda v, n=3: ('—' if v is None else (f'{v:.{n}f}' if isinstance(v, float) else str(v)))
    print('| 走 | 条件 | 開始更新 | T90 | 末の暗記率 | 最後 1,000 の平均 | 最後 1,000 の最小 | 届いた後の最小 | 急落回数 | 最初の急落 | 届いた後に 0.9 未満の割合 | 余裕の中央値 T90→末（最大） | 余裕 q10 末 | ‖h‖ T90→末 |')
    print('|---|---|---|---|---|---|---|---|---|---|---|---|---|---|')
    for name, s in summary['runs'].items():
        print(f"| {name} | {s['label']} | {s['start']} | {f(s['T90'])} | {f(s['final'],4)} | {f(s['tail_mean'],4)} | {f(s['tail_min'],4)} | {f(s['min_after'],4)} | {s['n_drops']} | {f(s['first_drop'])} | {f(s['frac_below90_after'])} | {f(s['margin_t90'],2)}→{f(s['margin_end'],2)}（{f(s['margin_max'],2)}） | {f(s['margin_q10_end'],2)} | {f(s['h_t90'],0)}→{f(s['h_end'],0)} |")
    print()
    print('| 走 | 区間 | n | 一歩の logit 変化 rms: 全体 / head / 体 / 交差 | 体の群 block0 / attn1 / ffn1 | 余裕の中央値 | \\|Δ余裕\\| の中央値 | 入替 |')
    print('|---|---|---|---|---|---|---|---|')
    for name, dd in summary['diag'].items():
        for k, q in dd.items():
            if q is None:
                continue
            g = q.get('groups')
            gs = ' / '.join(f'{g[x]:.3f}' for x in ('block0', 'attn1', 'ffn1')) if g else '—'
            print(f"| {name} | {k} | {q['n']} | {f(q['total'])} / {f(q['head'],4)} / {f(q['body'])} / {f(q['cross'],4)} | {gs} | {f(q['margin_before'])} | {f(q['dmargin_abs_median'],4)} | {f(q['flip_frac'],4)} |")
    print()
    for task, b in summary['bound'].items():
        print(f"t{task} 届いた時点（更新 {b['step']}、暗記率 {b['acc_at_state']:.3f}）: 余裕の中央値 {b['margin']['median']:.3g}（q10 {b['margin']['q10']:.3g}）")
        print('| 群 | η | 上界 η‖∂m/∂θ‖₁ の中央値 | q90 | 上界/余裕の中央値 | 上界 > 余裕 の割合 | いまの歩幅での和の中央値 | その比 | Adam の歩幅 \\|Δθ\\| の中央値 / q90 |')
        print('|---|---|---|---|---|---|---|---|---|')
        for g in GROUPS:
            x, y = b[f'bound_{g}'], b[f'statebound_{g}']
            step = b['adam_step_abs'].get(g)
            ss = f"{step['median']:.2e} / {step['q90']:.2e}（η {step['eta']:.0e}）" if step else '—'
            print(f"| {g} | | {x['median']:.3g} | {x['q90']:.3g} | {x['ratio_median']:.3g} | {x['frac_ratio_gt1']:.2f} | {y['median']:.3g} | {y['ratio_median']:.3g} | {ss} |")
        print()


if __name__ == '__main__':
    main()
