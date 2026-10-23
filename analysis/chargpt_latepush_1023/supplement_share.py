#!/usr/bin/env python3
"""POST HOC: disclose 0/0 share missingness, without changing registered results."""
import csv
import json
from pathlib import Path
import numpy as np
from judge import WINDOWS, table, write_csv

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT/'results/chargpt_latepush_1023'


def main():
    rows=[]
    for arm in ['C3','ReLUcap']:
        for seed in range(10):
            with np.load(OUT/'raw'/f'{arm}_seed{seed}.npz') as raw:
                for t in [0]+list(range(1,11))+[25]+list(range(39,51))+list(range(89,101)):
                    for layer in [0,1]:
                        prefix=f't{t}_L{layer}_'
                        alive=raw[prefix+'alive']
                        load,res=raw[prefix+'load'],raw[prefix+'res']
                        denom=np.abs(load)+np.abs(res)
                        valid=alive & (denom>0)
                        v=float(np.median(np.abs(load[valid])/denom[valid])) if valid.any() else None
                        rows.append(dict(arm=arm,seed=seed,task=t,layer=layer,n_alive=int(alive.sum()),
                                         n_nonzero_push=int(valid.sum()),n_zero_push=int((alive & (denom==0)).sum()),
                                         share_load_nonzero=v))
    write_csv(OUT/'posthoc_nonzero_share_states.csv',rows)
    windows=[]
    for arm in ['C3','ReLUcap']:
        for layer in [0,1]:
            for seed in range(10):
                for win,tasks in WINDOWS.items():
                    rr=[r for r in rows if r['arm']==arm and r['layer']==layer and r['seed']==seed and r['task'] in tasks]
                    vv=[r['share_load_nonzero'] for r in rr]
                    windows.append(dict(arm=arm,layer=layer,seed=seed,window=win,
                                        share_load_nonzero=float(np.median(vv)) if all(v is not None for v in vv) else None,
                                        states_with_zero_push=sum(r['n_zero_push']>0 for r in rr),
                                        zero_push_units=sum(r['n_zero_push'] for r in rr),
                                        n_alive_units=sum(r['n_alive'] for r in rr)))
    write_csv(OUT/'posthoc_nonzero_share_windows.csv',windows)
    counts={}
    wrows=[]
    for arm in ['C3','ReLUcap']:
        counts[arm]={}
        for layer in [1,0]:
            rr=[w for w in windows if w['arm']==arm and w['layer']==layer]
            by={(r['seed'],r['window']):r for r in rr}
            decreases=sum(by[s,'late']['share_load_nonzero']<by[s,'early']['share_load_nonzero'] for s in range(10))
            counts[arm][str(layer)]=dict(share_decreases_nonzero=decreases)
            for win in WINDOWS:
                ww=[r for r in rr if r['window']==win]
                vals=[r['share_load_nonzero'] for r in ww]
                wrows.append([arm,f'L{layer}',win,f'{min(vals):.6f}–{max(vals):.6f}',
                              sum(r['states_with_zero_push'] for r in ww),sum(r['zero_push_units'] for r in ww)])
    d=dict(status='POST_HOC',trigger='1476 alive unit-state entries have zero in both component pushes; 425 state/layer shares are undefined',
           rule='Exclude only alive units with |S_load|+|S_res|=0 from share, keeping all original s metrics and judges unchanged',counts=counts)
    (OUT/'posthoc_nonzero_share.json').write_text(json.dumps(d,indent=2,ensure_ascii=False)+'\n')
    lines=['## 事後補助：押しが両方ゼロの unit を分担比からのみ除外', '',
           '登録 share_load に0/0が出たため、分担比の定義域（分母>0）に限った中央値を、結果を見た後の補助量として計算。登録 states.csv・判定・欠測の扱いは変更していない。', '',
           table(['arm','layer','window','share_load_nonzero (10 seed 範囲)','0/0を含む状態数','除外unit×状態数'],wrows),'']
    for arm in counts:
        for layer,c in counts[arm].items():
            lines.append(f"- {arm} L{layer}: この事後補助の share が早期→後期で低下する seed は {c['share_decreases_nonzero']}/10。")
    lines += ['', '生存判定は現課題の probe、押しの勾配は別の一様 probe を使うので、前者で生きていることから後者の非ゼロ勾配は保証されない。実際に両成分の押しがゼロの unit が観測された。ゼロ押しは s の分母には登録どおり含まれ、S>0 には数えない。', '',
              '事後補助の範囲だけを登録 share_load の結果として引用しない。出典は posthoc_nonzero_share_states.csv・posthoc_nonzero_share_windows.csv・posthoc_nonzero_share.json。']
    p=OUT/'summary.md';text=p.read_text().split('\n## 事後補助：')[0]
    p.write_text(text.rstrip()+'\n\n'+'\n'.join(lines)+'\n')
    print(json.dumps(d,ensure_ascii=False))


if __name__=='__main__':
    main()
