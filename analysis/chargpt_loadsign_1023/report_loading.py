#!/usr/bin/env python3
import csv,json
from report import OUT,ARMS,WINDOWS

rows=list(csv.DictReader((OUT/'loading_sources_windows.csv').open()))
def rng(arm,win,k):
    vals=[float(r[k]) for r in rows if r['arm']==arm and r['window']==win]
    return f'{min(vals):.4f}–{max(vals):.4f}'
text=['# 装填側の分解（探索的な第三段階）','',
      '予測誤差側のself/otherと区別する。本表はFを作る現課題FFN活性をunit別に分解したもの。',
      'T=Gamma P U、G=T^T T、B=(Y-u)^T(a/sigma)、F_r=(Y-u)^T(Gamma P r/sigma)。F=B T^T+F_r。',
      'unit i のdirect読み出しは κB_i G_ii + κΣ_{j≠i}B_j G_ji + κF_r T_i。3項に同じ元の誤差を通す。',
      'exactは最終LNの分散微分も各装填成分に分けて足す。自己成分も補題のΣKφφ′そのものではない。','']
for arm in ARMS:
    text += [f'## {arm}','', '| 量 | 早期 | 中期 | 後期 |','|---|---|---|---|']
    keys=[f's_{s}_{l}_K' for l in ['direct','exact'] for s in ['self','other','residual']]
    keys += ['s_self_direct_absK','s_self_exact_absK']
    keys += [f'share_{s}_{l}' for l in ['direct','exact'] for s in ['self','other','residual']]
    for k in keys:text.append(f'| {k} | {rng(arm,"early",k)} | {rng(arm,"middle",k)} | {rng(arm,"late",k)} |')
    text+=['']
p=json.loads((OUT/'loading_sources_provenance.json').read_text())
text += ['## 検算','',f'700状態のF再現誤差{p["max_F_error"]:.3g}、direct/exactのK/absK全4通りの押しの再現誤差{p["max_push_error"]:.3g}。符号割合も完全一致。']
(OUT/'loading_sources_summary.md').write_text('\n'.join(text)+'\n')
