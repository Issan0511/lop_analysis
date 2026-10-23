#!/usr/bin/env python3
import csv
import json
import numpy as np
from report import OUT, WINDOWS, ARMS


def main():
    rows=list(csv.DictReader((OUT/'components_states.csv').open()))
    fields=[k for k in rows[0] if k not in ['arm','seed','task']]
    wins=[]
    for arm in ARMS:
        for seed in range(10):
            for win,tasks in WINDOWS.items():
                rr=[r for r in rows if r['arm']==arm and int(r['seed'])==seed and int(r['task']) in tasks]
                assert len(rr)==len(tasks)
                wins.append(dict(arm=arm,seed=seed,window=win,**{k:float(np.median([float(r[k]) for r in rr])) for k in fields}))
    with (OUT/'components_windows.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(wins[0]),lineterminator='\n');w.writeheader();w.writerows(wins)
    def rng(arm,win,k):
        vals=[r[k] for r in wins if r['arm']==arm and r['window']==win]
        return f'{min(vals):.4f}–{max(vals):.4f}'
    text=['# direct 成分の発生源の分解（探索的な第二段階）','',
          '第一段階の結果を見てから spec §7 を登録し、測定した。元の判定は変更しない。',
          'self は自 unit の forward 出力、other は他 FFN unit の forward 出力、residual は固定残差入力が作る誤差信号。上流の更新ではない。',
          'direct = self + other + residual が厳密に成立。最終 LN の分散微分はこの表には含めない。',
          '各値は seed 内の窓中央値の seed 間の範囲。大きさの分担は絶対値で正規化した帳簿であり、因果的な寄与率ではない。','']
    for arm in ARMS:
        text += [f'## {arm}','', '| 量 | 早期 | 中期 | 後期 |','|---|---|---|---|']
        for e in ['original','loaderror']:
            for k in [f's_{e}_{part}' for part in ['self','other','residual','direct','self_plus_other','self_plus_residual']]+[
                f'{e}_self_abs_share',f'{e}_other_abs_share',f'{e}_residual_abs_share',f'{e}_self_positive_direct_nonpositive']:
                text.append(f'| {k} | {rng(arm,"early",k)} | {rng(arm,"middle",k)} | {rng(arm,"late",k)} |')
        text += ['']
    pv=json.loads((OUT/'components_provenance.json').read_text())
    text+=['## 検算','',f'700 状態の第一段階 direct との最大相対誤差 {pv["max_stage1_error"]:.3g}、softmax 恒等式 {pv["max_softmax_error"]:.3g}、token ごとの加法分解 {pv["max_token_sum_error"]:.3g}。',
           'load-only error + abs(K) の self 非負性も全状態で確認。']
    (OUT/'components_summary.md').write_text('\n'.join(text)+'\n')


if __name__=='__main__':main()
