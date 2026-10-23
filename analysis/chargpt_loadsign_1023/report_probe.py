#!/usr/bin/env python3
import csv,json
from report import OUT,ARMS,WINDOWS

rows=list(csv.DictReader((OUT/'probe_control_windows.csv').open()))
def rng(a,w,k):
    v=[float(r[k]) for r in rows if r['arm']==a and r['window']==w]
    return f'{min(v):.4f}–{max(v):.4f}'
text=['# 入力probeの対照と旧課題の確信度（探索的第四段階）','',
      '元の勾配probeは独立一様文字列。対照はFと同じ現課題の文字列。新課題への期待では両方とも誤差p-uを使う。',
      '旧課題CEの誤差p-y_oldとは異なる。旧課題で正答すればp-y_oldは消えるが、切替後のp-uは一般に消えない。',
      '箱Mは、直前2文字ごとのDirichlet遷移分布から次の文字を毎回サンプルする。固定ラベル暗記ではない。',
      'accuracy/p_true/p_max/CEは現課題probe全128位置、H_trueは生成連鎖の定常条件付きentropy。最初の位置の情報量の差があるためCE-Hを厳密な同一標本差とは呼ばない。',
      '各値はseed内の窓中央値の10seed間範囲。','']
for a in ARMS:
    text += [f'## {a}','', '| 量 | 早期 | 中期 | 後期 |','|---|---|---|---|']
    for k in ['uniform_s_load','s_switch_load','uniform_s_full','s_switch_full','s_old_full','accuracy','mean_p_true','mean_p_max','entropy','CE','H_true','old_over_switch_full_abs']:
        text.append(f'| {k} | {rng(a,"early",k)} | {rng(a,"middle",k)} | {rng(a,"late",k)} |')
    text+=['']
j=json.loads((OUT/'probe_control_judge.json').read_text());g=json.loads((OUT/'probe_control_gate.json').read_text())
text += ['## 判定と検算','',
         f'入力を揃えた後期s_loadが>=.75かつ元より>=.20増えるseed: C3 {j["n_restored"]["C3"]}/10、ReLUcap {j["n_restored"]["ReLUcap"]}/10。探索的対照。',
         f'gate8状態で新ラベル期待full/loadと旧課題CE fullを自動微分と照合、最大相対誤差{g["max_error"]:.3g}。']
(OUT/'probe_control_summary.md').write_text('\n'.join(text)+'\n')
