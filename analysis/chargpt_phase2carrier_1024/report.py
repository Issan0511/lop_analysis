#!/usr/bin/env python3
"""Aggregate the carrier ledger. The independent replication unit is seed."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/chargpt_phase2carrier_1024'
TASKS = [1, 4, 5, 6, 9, 39, 50, 89, 100]
GROUPS = ['embedding', 'attention0', 'ln0', 'up0', 'down0', 'attention1', 'ln1', 'up1', 'downstream']
NAMES = {'embedding':'共有token埋め込み', 'attention0':'注意0＋前LN', 'ln0':'FFN0前LN',
         'up0':'FFN0 up', 'down0':'FFN0 down', 'attention1':'注意1＋前LN', 'ln1':'FFN1前LN',
         'up1':'FFN1 up', 'downstream':'対象より後段', 'own':'自unit up行', 'upstream':'上流合計',
         'full':'全経路', 'ffn0':'FFN0全体'}
TARGETS = ['mu', 'sd', 'mean', 'width', 'r']
METRICS = ['SGD', 'Adam_fixed']
SIGNALS = ['P2', 'P2ctx', 'switch', 'fit', 'switch_body', 'fit_body', 'switch_head', 'fit_head']


def dump(path, obj):
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def write_csv(path, rows):
    with path.open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def rho(x, y):
    if len(x)<3 or np.ptp(x)==0 or np.ptp(y)==0:
        return None
    def rank(v):
        _, ix, counts = np.unique(v, return_inverse=True, return_counts=True)
        return (np.cumsum(counts)-(counts-1)/2)[ix]
    return float(np.corrcoef(rank(x), rank(y))[0,1])


def nz(x, tol):
    x = np.array(x, copy=True)
    x[np.abs(x) <= tol*np.max(np.abs(x))] = 0
    return x


def compare(pred, actual, mask, tol):
    a, b = nz(pred, tol), nz(actual, tol)
    both = mask & (a != 0) & (b != 0)
    x, y = a[both], b[both]
    pos, neg = y > 0, y < 0
    ba = float(((x[pos]>0).mean()+(x[neg]<0).mean())/2) if pos.sum() >= 10 and neg.sum() >= 10 else None
    den = np.linalg.norm(pred[mask])*np.linalg.norm(actual[mask])
    return dict(n_base=int(mask.sum()), n_nonzero=int(both.sum()), n_sink=int(pos.sum()), n_float=int(neg.sum()),
                pred_zero=int((mask & (a==0)).sum()), actual_zero=int((mask & (b==0)).sum()),
                agreement=float((np.sign(x)==np.sign(y)).mean()) if len(x) else None,
                agreement_all=float((np.sign(a[mask])==np.sign(b[mask])).mean()) if mask.any() else None,
                ba=ba, rho=rho(pred[mask], actual[mask]),
                cos=float(pred[mask]@actual[mask]/den) if den else None)


def directions(raw, layer, metric, signal):
    prefix = f'L{layer}_{metric}_{signal}_'
    out = {'full': {k: raw[prefix+'full_'+k] for k in TARGETS}}
    if signal in ['P2', 'P2ctx', 'switch', 'fit'] or signal.startswith('P2tol'):
        out.update({g: {k: raw[prefix+g+'_'+k] for k in TARGETS} for g in GROUPS})
        out['own'] = out[f'up{layer}']
        parents = ['embedding', 'attention0', 'ln0'] if layer == 0 else GROUPS[:7]
        out['upstream'] = {k: sum(out[g][k] for g in parents) for k in TARGETS}
        if layer == 1:
            out['ffn0'] = {k: sum(out[g][k] for g in ['ln0','up0','down0']) for k in TARGETS}
        for k in TARGETS:
            a, b = out['full'][k], out['own'][k]+out['upstream'][k]
            err = np.max(np.abs(a-b))/max(np.max(np.abs(a)),np.max(np.abs(b)),1e-300)
            assert err <= 1e-8, (layer,metric,signal,k,err)
    return out


def statistics(v, alive):
    x = v['r']; z = nz(x,1e-10)
    dead = ~alive
    return dict(n_units=len(x), n_alive=int(alive.sum()),
                mean=float(x.mean()), median=float(np.median(x)),
                sink_fraction=float((z>0).mean()), float_fraction=float((z<0).mean()), zero_fraction=float((z==0).mean()),
                mu_mean=float(v['mu'].mean()), sd_mean=float(v['sd'].mean()),
                mean_term=float(v['mean'].mean()), width_term=float(v['width'].mean()),
                alive_mean=float(x[alive].mean()) if alive.any() else None,
                alive_sink=float((z[alive]>0).mean()) if alive.any() else None,
                dead_mean=float(x[dead].mean()) if dead.any() else None,
                dead_sink=float((z[dead]>0).mean()) if dead.any() else None,
                max_abs=float(np.max(np.abs(x))))


def main():
    rows, comparisons, winners, checks = [], [], [], []
    rawdir = OUT/'raw'
    if not rawdir.exists():
        rawdir = Path('/home/issan/Projects/obsidian-research-data/chargpt_phase2carrier_1024/results/chargpt_phase2carrier_1024/raw')
    for seed in range(10):
        for task in TASKS:
            path = rawdir/f'C3_seed{seed}_task{task}.npz'
            assert path.exists(), path
            meta = json.loads(path.with_suffix('.json').read_text())
            assert hashlib.sha256(path.read_bytes()).hexdigest() == meta['raw_sha256']
            checks.append(meta)
            with np.load(path) as raw:
                names = SIGNALS + sorted({k.split('_')[2] for k in raw.files if k.startswith('L0_SGD_P2tol')})
                for layer in [0, 1]:
                    alive = raw[f'L{layer}_alive'].astype(bool)
                    for metric in METRICS:
                        signals = {name: directions(raw,layer,metric,name) for name in names}
                        for name, routes in signals.items():
                            for route, vv in routes.items():
                                rows.append(dict(seed=seed,task=task,layer=layer,metric=metric,signal=name,route=route,
                                    r_median=float(np.median(raw[f'L{layer}_r'])), p_median=float(np.median(raw[f'L{layer}_p'])),
                                    **statistics(vv,alive)))
                            if name in ['P2','P2ctx','switch','fit']:
                                means = {g: float(routes[g]['r'].mean()) for g in GROUPS}
                                pos, neg = max(means,key=means.get), min(means,key=means.get)
                                winners.append(dict(seed=seed,task=task,layer=layer,metric=metric,signal=name,
                                    full_mean=float(routes['full']['r'].mean()),sink_carrier=pos if means[pos]>0 else 'none',
                                    sink_mean=means[pos],float_carrier=neg if means[neg]<0 else 'none',float_mean=means[neg]))
                        for pred in [s for s in names if s.startswith('P2')]:
                            for actual in ['switch','fit','switch_body','fit_body']:
                                for route in signals[actual]:
                                    for subset, mask in [('all',np.ones(512,dtype=bool)),('alive',alive)]:
                                        for tol in [1e-10,1e-8,1e-12]:
                                            cc = compare(signals[pred][route]['r'],signals[actual][route]['r'],mask,tol)
                                            comparisons.append(dict(seed=seed,task=task,layer=layer,metric=metric,
                                                predictor=pred,actual=actual,route=route,subset=subset,zero_tol=tol,**cc))
    write_csv(OUT/'carriers.csv',rows)
    write_csv(OUT/'comparisons.csv',comparisons)
    write_csv(OUT/'winners.csv',winners)
    write_csv(OUT/'states.csv',[r for r in rows if r['route']=='full'])
    rowindex = {(r['seed'],r['task'],r['layer'],r['metric'],r['signal'],r['route']):r for r in rows}
    compindex = {(r['seed'],r['task'],r['layer'],r['metric'],r['predictor'],r['actual'],r['route'],r['subset'],r['zero_tol']):r for r in comparisons}
    summary=[]
    for task in TASKS:
        for layer in [0,1]:
            for metric in METRICS:
                for signal in ['P2','P2ctx','switch','fit','switch_head','fit_head']:
                    routes = ['full','own','upstream'] if not signal.endswith('_head') else ['full']
                    for route in routes:
                        rr=[rowindex[s,task,layer,metric,signal,route] for s in range(10)]
                        cols=['mean','median','sink_fraction','mean_term','width_term','n_alive','dead_mean','dead_sink']
                        out={}
                        for col in cols:
                            vs=[r[col] for r in rr if r[col] is not None]
                            out[col+'_range']=[min(vs),max(vs)] if vs else None
                        summary.append(dict(task=task,layer=layer,metric=metric,signal=signal,route=route,
                            seeds_mean_sinks=sum(r['mean']>0 for r in rr),seeds_median_sinks=sum(r['median']>0 for r in rr),**out))
    dump(OUT/'summary.json',summary)
    dump(OUT/'judge.json',dict(method='signed means, medians and unit fractions; no posthoc support threshold',
                             totals=[r for r in summary if r['route']=='full']))
    # Human report helpers: all numbers come from the preceding CSV rows.
    def sel(task,layer,metric,signal,route='full'):
        return [rowindex[s,task,layer,metric,signal,route] for s in range(10)]
    def rng(rr,col='mean',fmt='.3g'):
        vv=[r[col] for r in rr if r[col] is not None]
        return f'{min(vv):{fmt}}～{max(vv):{fmt}}' if vv else '未定義'
    def npos(rr):return sum(r['mean']>0 for r in rr)
    def comp(task,layer,metric,route='full',actual='switch',pred='P2'):
        return [compindex[s,task,layer,metric,pred,actual,route,'all',1e-10] for s in range(10)]
    text=['# C3：相2予測と実CEの全経路・担い手の実測','',
          '新しい学習なし。C3-2・10seed・指定した9課題末の90状態。位置は固定一様probe、勾配は現課題chain上。主はL0、L1も併記。',
          '正のSはΔr=−ηS<0（沈む）。switchは独立一様新ラベルへのCE期待、fitは現課題の真の確率QへのCE期待。実ミニバッチやmomentum込みAdam更新の再生ではない。',
          '主集計は全512unit。閉じたunitも上流で動くため除かない。表の範囲はseed0–9の最小～最大。各行はその課題末であり、窓全体の集計ではない。',
          'P2はfree-head LS模型のiid旧ラベル平均、P2ctxは同文脈を共有する旧ラベル平均。どちらも実網の学習済みHを条件として使う。量の係数はfitせず、方向と同一信号内の担い手を比較する。','']
    for layer in [0,1]:
        for metric in METRICS:
            text += [f'## L{layer} / {metric}：切替の合計は沈むか','',
              '| 課題末 | 実CE全体の平均S | 沈むseed数 | 自unit行の平均S | 上流合計の平均S | 相2平均が沈むseed数 | 全体の符号一致 | 全体のBA |',
              '|---:|---:|---:|---:|---:|---:|---:|---:|']
            for task in TASKS:
                aa,oo,uu=sel(task,layer,metric,'switch'),sel(task,layer,metric,'switch','own'),sel(task,layer,metric,'switch','upstream')
                pp=sel(task,layer,metric,'P2');cc=comp(task,layer,metric)
                text.append(f'| {task} | {rng(aa)} | {npos(aa)}/10 | {rng(oo)} | {rng(uu)} | {npos(pp)}/10 | {rng(cc,"agreement",".3f")} | {rng(cc,"ba",".3f")} |')
            text += ['','BAは実測の正・負が各10unit以上の場合だけ。全seedで定義されないときは範囲が定義可能seedに限られる。全数と分母はcomparisons.csv。','']
    text += ['## 全体の平均・中央値・unit割合は別に見る','',
              '| 層 | 課題末 | switch平均S | 中央値S | 沈むunit割合 | 生存unit数 | 閉じたunitの沈む割合 |',
              '|---:|---:|---:|---:|---:|---:|---:|']
    for layer in [0,1]:
        for task in TASKS:
            aa=sel(task,layer,'Adam_fixed','switch')
            text.append(f'| L{layer} | {task} | {rng(aa)} | {rng(aa,"median")} | {rng(aa,"sink_fraction",".3f")} | {rng(aa,"n_alive",".0f")} | {rng(aa,"dead_sink",".3f")} |')
    for layer in [0,1]:
        text += ['',f'## L{layer} / Adam固定分母：担い手の内訳','',
                 '| 課題末 | 担い手 | 実CE平均S | 相2平均S | 実CEで沈めるseed数 | 相2で沈めるseed数 | unit符号一致 | Spearman |',
                 '|---:|---|---:|---:|---:|---:|---:|---:|']
        groups=['embedding','attention0','ln0','up0'] if layer==0 else ['embedding','attention0','ffn0','attention1','ln1','up1']
        for task in [1,9,39,100]:
            for group in groups:
                aa,pp=sel(task,layer,'Adam_fixed','switch',group),sel(task,layer,'Adam_fixed','P2',group)
                cc=comp(task,layer,'Adam_fixed',group)
                text.append(f'| {task} | {NAMES[group]} | {rng(aa)} | {rng(pp)} | {npos(aa)}/10 | {npos(pp)}/10 | {rng(cc,"agreement",".3f")} | {rng(cc,"rho",".3f")} |')
    text += ['','## 平均の項と幅の項（Adam固定分母）','',
       'S_r=S_mu/s−mu*S_sd/s²。表は各unitで分解してから平均した値。中央値の項は足せないため、帳簿には平均を用いる。','',
       '| 層 | 課題末 | 信号 | 平均の項 | 幅の項 | 合計S_r |','|---:|---:|---|---:|---:|---:|']
    for layer in [0,1]:
        for task in [1,9,39,100]:
            for signal in ['switch','P2']:
                rr=sel(task,layer,'Adam_fixed',signal)
                text.append(f'| L{layer} | {task} | {signal} | {rng(rr,"mean_term")} | {rng(rr,"width_term")} | {rng(rr)} |')
    text += ['','## 現課題の勾配と共有head直接経路（Adam固定分母）','',
          '相2を照合した切替の期待と、現課題をさらに学ぶ勾配を混ぜない。共有head直接経路はembeddingを動かし、上流経路の一部をなす。','',
          '| 層 | 課題末 | fit全体の平均S | fitで沈むseed数 | P2対fitのBA | switchのhead直接平均S | fitのhead直接平均S |',
          '|---:|---:|---:|---:|---:|---:|---:|']
    for layer in [0,1]:
        for task in TASKS:
            rr=sel(task,layer,'Adam_fixed','fit');cc=comp(task,layer,'Adam_fixed',actual='fit')
            hh=sel(task,layer,'Adam_fixed','switch_head');fh=sel(task,layer,'Adam_fixed','fit_head')
            text.append(f'| L{layer} | {task} | {rng(rr)} | {npos(rr)}/10 | {rng(cc,"ba",".3f")} | {rng(hh)} | {rng(fh)} |')
    text += ['','## 経路ごとの方向対応（課題100末、全512unit、seed別）','',
       '| 層 | metric | seed | 全体BA | 全体rho | own BA | 上流BA | context全体BA |',
       '|---:|---|---:|---:|---:|---:|---:|---:|']
    def val(v):return f'{v:.3f}' if v is not None else '未定義'
    for layer in [0,1]:
        for metric in METRICS:
            cc,oo,uu,ct=[comp(100,layer,metric,route=route,pred=pred) for route,pred in [('full','P2'),('own','P2'),('upstream','P2'),('full','P2ctx')]]
            for s in range(10):
                text.append(f'| L{layer} | {metric} | {s} | {val(cc[s]["ba"])} | {val(cc[s]["rho"])} | {val(oo[s]["ba"])} | {val(uu[s]["ba"])} | {val(ct[s]["ba"])} |')
    maxcheck=max(float(v) for cc in checks for v in cc['checks'].values())
    refs=[cc['reference'] for cc in checks if cc['reference']]
    maxref=max(float(v) for cc in refs for v in cc['checks'].values())
    finite=[r for cc in checks for r in (cc['finite'] or [])]
    finite_best=[]
    for cc in checks:
        if not cc['finite']:continue
        for key in {(r['signal'],r['layer']) for r in cc['finite']}:
            finite_best.append(min(r['error'] for r in cc['finite'] if (r['signal'],r['layer'])==key))
    audit=dict(states=len(checks),max_carrier_gradient_closure=maxcheck,reference_states=len(refs),max_reference_error=maxref,
               worst_best_finite_error=max(finite_best),ranks=sorted(set(r['stats']['rank'] for r in checks)),
               all_bias_absent=all(not any(k.endswith('.bias') for k in cc['parameter_groups']) for cc in checks),
               independent_seed_count=10,all_unit_count=512)
    dump(OUT/'audit.json',audit)
    text += ['','## 検算と限界','',
       f'- {len(checks)}状態のcarrier和・全経路JVP・平均/幅・own解析式・共有headの勾配帳簿：最大相対誤差 {maxcheck:.3g}。',
       f'- 既存{len(refs)}状態のr・生存集合・own P2/switchを照合：最大相対誤差 {maxref:.3g}。',
       f'- 先行3状態の正規化方向の有限差分：各対象で最も小さい誤差を選んだ最大 {max(finite_best):.3g}。全試行はgate.json。',
       f'- 全状態のrank：{audit["ranks"]}。各parameterは一つの群だけへ所属。biasは無し。対象より後段の直接JVPは厳密0。',
       '- 固定状態の一次の押し。有限100更新や長期の寄与率ではなく、担い手がなぜ学習で形成されたかの介入証明でもない。',
       '- 相2は自由headの旧ラベル平均を学習済みHに当てた模型。実網は共有headであり、直接headの項を省略せず、差を明示した。',
       '- 一部の経路で対応しても全体の対応とは限らない。meanの符号と沈むunitの割合は区別する。','']
    (OUT/'summary.md').write_text('\n'.join(text))
    print(json.dumps(audit,indent=2))


if __name__ == '__main__':
    main()
