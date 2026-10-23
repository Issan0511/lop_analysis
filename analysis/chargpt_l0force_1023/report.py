#!/usr/bin/env python3
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/chargpt_l0force_1023'
rows=list(csv.DictReader((OUT/'states.csv').open()))
def num(r,k):return float(r[k]) if r.get(k) not in ['',None] else float('nan')
def ran(rr,k):
    a=np.array([num(r,k) for r in rr]);a=a[np.isfinite(a)]
    if not len(a):return '欠測'
    return f'{a.min():.4g}–{a.max():.4g}'
def count(rr,k):return sum(r[k]=='True' for r in rr)
def select(arm,task,step,metric='SGD'):return [r for r in rows if r['arm']==arm and int(r['task'])==task and int(r['step'])==step and r['metric']==metric]
states=[(9,10000),(39,10000),(99,10000),(100,100),(100,10000)]
judges=[]
for arm in ['C3','ReLUcap']:
 for t,u in states:
  for metric in ['SGD','Adam_fixed']:
   rr=select(arm,t,u,metric)
   if not rr:continue
   for error in ['fit','switch']:
    n=count(rr,f'alive_{error}_self_carries');judges.append(dict(arm=arm,task=t,step=u,metric=metric,error=error,support_seeds=n,n_seeds=len(rr),verdict='DIRECT_SELF_CARRIES' if n>=8 else 'NOT_ESTABLISHED'))
(OUT/'judge.json').write_text(json.dumps(judges,ensure_ascii=False,indent=2)+'\n')
lines=['# L0の自己出力抑制と実際の勾配','',
'**解釈訂正（2026-10-23）: 登録した自己項除去の基準は、残りを無視できるかを検定していない。数値と判定は保持するが、論文H1の支持やL0沈降の原因同定とは扱わない。詳細はconclusion.md。**','',
'事後に起案・計測前にspec固定。2腕×10 seed×5保存時点=100状態、新しい学習なし。表は同一時点のseed範囲。課題100更新100と課題100末を分ける。',
'','S>0が沈める向き。muは平均前活性、r=mu/s。SGDの一次変化は-ηS。固定uniform probeで位置を測り、現課題のchain入力から位置0を除いた期待CE勾配を得る。',
'','## 登録した帳簿基準（H1支持とは別）','',
'|腕|時点|計量|期待CEの基準達成seed|一様target切替の基準達成seed|','|---|---|---|---:|---:|']
for arm in ['C3','ReLUcap']:
 for t,u in states:
  for metric in ['SGD','Adam_fixed']:
   rr=select(arm,t,u,metric)
   if rr:lines.append(f'|{arm}|task{t}/u{u}|{metric}|{count(rr,"alive_fit_self_carries")}/10|{count(rr,"alive_switch_self_carries")}/10|')
lines+=['','基準: 期待CE（またはswitch）でrを沈めるalive unit≥32、その80%以上でselfも沈める側、その50%以上はselfを引くと沈める側でなくなる、を8/10seed。この基準は残りが期待値で消えることを検査しておらず、達成をH1支持とは読まない。','',
'## 自己項と合計の方向（SGD、alive）','','|腕|時点|alive数|selfのmu沈み割合|selfのr沈み割合|期待CEのr沈み割合|switchのr沈み割合|self絶対値分担（期待CE帳簿）|self除去で沈みを失う割合|','|---|---|---:|---:|---:|---:|---:|---:|---:|']
for arm in ['C3','ReLUcap']:
 for t,u in states:
  rr=select(arm,t,u)
  lines.append('|'+ '|'.join([arm,f'{t}/{u}']+[ran(rr,k) for k in ['n_alive','alive_self_mu_s','alive_self_r_s','alive_fit_r_s','alive_switch_r_s','alive_fit_self_share','alive_fit_self_removal']])+'|')
for metric in ['SGD','Adam_fixed']:
 lines+=['',f'## 各項のS_rのunit中央値（{metric}、alive）','','|腕|時点|self|他L0unit出力|残差出力|最終LN分散|上位FFN枝|上位attention枝|正解分布の項|期待CE合計|','|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
 for arm in ['C3','ReLUcap']:
  for t,u in states:
   rr=select(arm,t,u,metric)
   if rr:lines.append('|'+ '|'.join([arm,f'{t}/{u}']+[ran(rr,'alive_'+k+'_r_med') for k in ['self','other0','residual','LNrad','upper_ffn','upper_attn','negative_target','fit']])+'|')
lines+=['','各項の中央値の和は合計の中央値ではない。打消しを含み、絶対値分担は因果寄与率ではない。Adam_fixedは保存分母固定で、momentの更新・新バッチを含む実Adam一歩ではない。','',
'## 自信と真の分布','','|腕|時点|H(q)|KL(q||p)|H(p)|max pの平均|真の偏り方向の予測倍率α|','|---|---|---:|---:|---:|---:|---:|']
for arm in ['C3','ReLUcap']:
 for t,u in states:
  rr=select(arm,t,u);lines.append('|'+ '|'.join([arm,f'{t}/{u}']+[ran(rr,k) for k in ['Hq','KL','Hp','pmax','alpha']])+'|')
lines+=['','α=〈p-u,q-u〉/‖q-u‖²の位置平均。予測の方向違いもあり、αだけを校正や自信と同一視しない。H(q)>0でもp=qなら期待CE勾配p-qは0。switch=p-u=(q-u)+(p-q)の分解は正確。','',
'## 検算','']
gate=json.loads((OUT/'gate.json').read_text());checkkeys=[k for k in rows[0] if k.startswith('check_')]
for k in checkkeys:lines.append(f'- {k}: 最大 {max(num(r,k) for r in rows):.3g}')
lines.append(f'- 独立CE autograd、forward再現、r有限差分をseed0両腕3時点で照合。有限差分最大相対誤差 {max(r.get("finite_difference_rel",0) for r in gate["rows"]):.3g}。')
lines+=['','## 読める範囲','',
'selfはL0→最終LNの残差直通の自己出力であり、上位注意・FFNを経由して戻る同じunitの信号まで分離した量ではない。Hは定義を固定した厳密PSD secantで、非線形の源への配分は一意ではない。NOT_ESTABLISHEDは全経路の自己機構の否定ではない。',
'','この帳簿は保存点での損失勾配の加法分解。端点間で起きた全Adam更新の積分ではなく、上流パラメータが変えて運ぶL0の入力平均も含めない。新しい学習0、optimizer生成0。']
(OUT/'summary.md').write_text('\n'.join(lines)+'\n')
print('report',len(rows),'rows')
