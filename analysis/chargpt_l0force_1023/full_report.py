#!/usr/bin/env python3
"""Canonical summary for the preregistered complete downstream self ledger."""
import csv,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2];OUT=ROOT/'results/chargpt_l0force_1023/full_self'
rows=list(csv.DictReader((OUT/'states.csv').open()))
def n(r,k):return float(r[k]) if r.get(k) else np.nan
def ran(rr,k):
 a=np.array([n(r,k) for r in rr]);a=a[np.isfinite(a)]
 return f'{a.min():.4g}–{a.max():.4g}' if len(a) else '欠測'
def subset(a,t,m):return [r for r in rows if r['arm']==a and r['task']==str(t) and r['metric']==m]
judges=[]
for a in ['C3','ReLUcap']:
 for t in [39,100]:
  for m in ['SGD','Adam_fixed']:
   rr=subset(a,t,m)
   for e in ['fit','switch']:
    k=sum(r[e+'_self_carries']=='True' for r in rr)
    judges.append(dict(arm=a,task=t,metric=m,error=e,support_seeds=k,total=10,verdict='FULL_LOCAL_SELF_CARRIES' if k>=8 else 'NOT_ESTABLISHED'))
(OUT/'judge.json').write_text(json.dumps(judges,ensure_ascii=False,indent=2)+'\n')
lines=['# 全後段を通るL0自身の出力から戻る力','',
'**解釈訂正（2026-10-23）: FULL_LOCAL_SELF_CARRIESは登録した算術的な符号除去基準の名称であり、他項が期待値で消えることや論文H1の支持を意味しない。自己項の正値・除去時の反転だけから自己形が原因とした初版の結論を撤回する。数値・登録判定は保持。詳細は親ディレクトリのconclusion.md。**','',
'主解析の後に追補登録、追加測定前に判定固定。新しい学習なし。両腕10seed×課題末39/100=40状態、各状態のuniform-alive unitから勾配を見ず32unitを抽出。合計1280unit-state。位置は固定uniform、誤差は現課題chainの位置1..127。S_r>0が沈める側。',
'','## 結果（32unit内集約、10seedの範囲）','',
'|腕|課題末|計量|自己項の沈み割合|期待CE合計の沈み割合|自己項の絶対値分担|自己項除去で沈みを失う割合|期待CE基準達成seed|切替基準達成seed|','|---|---|---|---:|---:|---:|---:|---:|---:|']
for a in ['C3','ReLUcap']:
 for t in [39,100]:
  for m in ['SGD','Adam_fixed']:
   rr=subset(a,t,m)
   line=[a,str(t),m]+[ran(rr,k) for k in ['self_r_s','fit_r_s','fit_self_share','fit_self_removal']]+[str(sum(r[e+'_self_carries']=='True' for r in rr))+'/10' for e in ['fit','switch']]
   lines.append('|'+ '|'.join(line)+'|')
lines+=['','自己項除去の割合の分母は、同じ状態の選択unitのうち期待CE合計が沈めるもの。基準達成はその数≥8、selfも沈める割合≥.8、self除去で非正になる割合≥.5のseed。8/10seed以上でFULL_LOCAL_SELF_CARRIES。この登録ラベルはH1支持を意味しない。','',
'## 符号の事後内訳（ゼロと浮かせる側を区別）','',
'割合の低いseedを見てから追加した記述集計で、Q1は変更しない。各行は10seed×32unit。unitを独立seedとは数えない。','','|腕|課題末|計量|沈める|ゼロ|浮かせる|','|---|---|---|---:|---:|---:|']
manifest_path=OUT.parent/'backup_manifest.json'
manifest=json.loads(manifest_path.read_text())['files'] if manifest_path.exists() else []
for a in ['C3','ReLUcap']:
 for t in [39,100]:
  for metric in ['SGD','Adam_fixed']:
   values=[]
   for seed in range(10):
    p=OUT/'raw'/f'{a}_seed{seed}_t{t}.npz'
    if not p.exists():p=Path(next(r['backup'] for r in manifest if r['source'].endswith('/full_self/raw/'+p.name)))
    with np.load(p) as data:values.append(data[metric+'_self_r'])
   v=np.concatenate(values);lines.append(f'|{a}|{t}|{metric}|{int((v>0).sum())}|{int((v==0).sum())}|{int((v<0).sum())}|')
lines+=['',
'## 力の内訳（S_rのunit中央値）','','|腕|課題末|計量|自己出力|他L0unitの出力|局所線形化の残り|正解分布の項|期待CE合計|','|---|---|---|---:|---:|---:|---:|---:|']
for a in ['C3','ReLUcap']:
 for t in [39,100]:
  for m in ['SGD','Adam_fixed']:
   rr=subset(a,t,m);lines.append('|'+ '|'.join([a,str(t),m]+[ran(rr,k+'_r_med') for k in ['self','other0','residual','negative_target','fit']])+'|')
lines+=['','各項の中央値は足し合わせられない。絶対値分担は各unitの|self|/(|self|+|other|+|residual|+|target|)を取ってから中央値。符号付きの正味更新に対する寄与率でも、学習軌道の因果寄与率でもない。','',
'## 相殺の規模（|S_r|のunit中央値）','','|腕|課題末|計量|自己項|switch|target|期待CE合計|','|---|---|---|---:|---:|---:|---:|']
for a in ['C3','ReLUcap']:
 for t in [39,100]:
  for m in ['SGD','Adam_fixed']:
   rr=subset(a,t,m);lines.append('|'+ '|'.join([a,str(t),m]+[ran(rr,k+'_r_abs') for k in ['self','switch','target','fit']])+'|')
lines+=['','期待CEの微小な正味勾配からselfを引くと符号が変わっても、長期沈降の大きさをselfが支配した証拠にはならない。保存点の大きな相殺を併記する。','',
'## seed別判定（SGD）','','|腕|課題末|seed|沈むunit数/32|selfも沈む割合|self除去で非正|基準達成|','|---|---|---:|---:|---:|---:|---|']
for r in rows:
 if r['metric']=='SGD':lines.append('|'+ '|'.join([r['arm'],r['task'],r['seed'],r['fit_sink_n'],f'{n(r,"fit_self_positive"):.4g}',f'{n(r,"fit_self_removal"):.4g}',r['fit_self_carries']])+'|')
lines+=['','## 定義と独立検算','',
'activation aの後の全計算をf(a)とし、保存点のJacobianをJとする。各unitの全token列だけを残したa^(i)からf_i=J a^(i)を求め、[J^T H f_i]_{:,i}をそのunit自身の出力から戻る信号とした。Hは主解析と同じ厳密PSD secant。後段注意の時間混合、FFN、LN分散微分を全て含む。',
'','全L0の局所出力はJ a。したがってswitch=自己+(全L0−自己)+(f−J a)が正確に閉じる。fitはそこから真の条件付き分布q-uの信号を引く。全512unitの中の自己以外をotherに含め、選択32unitだけで残りを作っていない。',
'','J^T H JがPSDでもtoken間交差があるため、各unitの位置押しは無条件正ではない。正値を仮定せず、実際のS_rを計算した。']
gate=json.loads((OUT/'gate.json').read_text())
for k in ['direct_self_independent','secant','H_one','jvp_fd','bilinear','source','reference']:
 vv=[float(r[k]) for r in gate['rows']]+[n(r,k) for r in rows];lines.append(f'- {k}: 最大 {max(vv):.3g}')
lines+=['','## 解釈の範囲','',
'この検査は「回転」という運動の記述ではなく、その向きに働く損失勾配の信号源を計算したもの。ただし局所線形化とsecantによる出力源の配分で、独立した長期介入ではない。非線形の有限unit除去と同じではなく、全期間の沈降を自己項だけが作ったと証明したものでもない。',
'','ReLUcapにも同じ自己出力抑制はあり得る。したがってselfが存在することと、C3だけが長期に深く沈むことの説明は分ける。正解分布と他unitの打消し、後段の配置、入力変化、実Adam履歴が正味の動きを決める。',
'','相2の残差化特徴模型（固定特徴/二乗誤差/当てはめ完了など）を検証したものではない。今回の自己項に相2を仮定する必要はないが、相2の併存を排除もしない。']
(OUT/'summary.md').write_text('\n'.join(lines)+'\n');print('full report',len(rows),'rows')
