"""Final report only after both 50-task arms finish. Parent results are read-only."""
from common import *
import scientific as S
import launch as L

def main():
 parts={n:json.loads((CHECK/(n+'.json')).read_text()) for n in REQUIRED}
 if not L.qualified(parts):raise RuntimeError('checks missing, stale or failed')
 frames={};provs={}
 for arm in ('ref','H','C','CH'):
  folder=BASE/arm if arm in ('ref','H') else ROOT/'results/relu_doors_0919'/arm
  frames[arm]=df(folder);provs[arm]=json.loads((folder/'provenance.json').read_text())
  if arm in HASHES:assert sha(folder/'per_task.csv')==HASHES[arm]
  else:
   p=provs[arm];start=json.loads((folder/'provenance_start.json').read_text())
   assert p['run_id']==RUN and p['prereg_commit']==PREREG and p['launch_provenance']==start
   assert p['git_hash']==start['git_hash'] and not p['dirty_src_analysis']
  p=provs[arm]
  assert p['R']==10 and p['seeds']==list(range(10)) and p['conds']==['raw']
  assert p['n_tasks']==50 and p['epochs_per_task']==400 and p['doors']==E.DOORS[arm]
  assert p['lr']==.001 and p['door_beta']==.01
 for arm in ('H','C','CH'):
  assert provs[arm]['data_sha256']==provs['ref']['data_sha256']
  assert provs[arm]['subset_sha256']==provs['ref']['subset_sha256']
 v=S.verdict(frames,qualified=True,hashes_ok=True)
 v.update({'run_id':RUN,'prereg_commit':PREREG,'independent_audit':False,
           'implementation_commit':provs['H']['git_hash'],'reference_policy':'fresh R10 ref; qualified old R10 C/CH'})
 write(BASE/'verdict.json',v)
 pd.DataFrame([{'key':k,'label':label} for k,label in v['labels'].items()]).to_csv(BASE/'verdict.csv',index=False)
 per=[]
 for arm,d in frames.items():
  for seed in range(10):
   z=d[d.seed==seed]
   per.append({'arm':arm,'seed':seed,'window_t31_50':z[z.task.between(31,50)].online_acc.mean(),
    'memo_t31_50':z[z.task.between(31,50)].memo_acc.mean(),
    'drop_t1_10_minus_t41_50':z[z.task.between(1,10)].online_acc.mean()-z[z.task.between(41,50)].online_acc.mean()})
 pd.DataFrame(per).to_csv(BASE/'per_seed.csv',index=False)
 layer=[]
 for arm,d in frames.items():
  for _,r in d[d.task.isin([1,50])].iterrows():
   cols=['seed','task','dead_frac_l1','dead_frac_l2','zbar_l1','zsd_l1','zbar_l2','zsd_l2','sink_ratio_l2','gate_zero_frac_l2','bias_over_sd_l2','r_a1']
   row={'arm':arm,**{c:r[c] for c in cols}}
   row['ratio_of_medians_l2']=r.zbar_l2/r.zsd_l2 if r.zsd_l2!=0 else None
   layer.append(row)
 pd.DataFrame(layer).to_csv(BASE/'layer_table.csv',index=False)
 # Read m, exact-zero z, and the numerator/denominator of r_a1 from full R10 snapshots.
 torch.set_num_threads(2);dev=E.H.setup('cuda');cifar=E.RC.Cifar10();snaprows=[]
 for arm in ('ref','H'):
  for t in (1,50):
   zz=E.replay_stack(BASE/arm,arm,[(s,'raw') for s in range(10)],t,dev,cifar)
   for seed in range(10):
    a=zz[1][seed];mu=a.mean(0);den=(a-mu).square().sum(1).mean().sqrt()
    with np.load(E.snapshot_path(BASE/arm,arm,'raw',seed,t)) as sn:
     snaprows.append({'arm':arm,'seed':seed,'task':t,'a1_mean_norm':float(mu.norm()),'a1_rms_deviation':float(den),
      'z1_exact_zero_count':int((zz[0][seed]==0).sum()),'z2_exact_zero_count':int((zz[2][seed]==0).sum()),
      'm1_mean':float(sn['m1'].mean()) if 'm1' in sn else None,'m2_mean':float(sn['m2'].mean()) if 'm2' in sn else None})
 pd.DataFrame(snaprows).to_csv(BASE/'snapshot_diagnostics.csv',index=False)
 predictions=[]
 for person,probs in [('Issa',[None,None,None]),('Codex',[.8,.8,.7])]:
  for key,pred,p in zip(['M_H','N_C','R_LAYER'],['H_ALONE_FAILS','C_NEEDED','LAYER_CORRESPONDENCE'],probs):
   hit=(v['labels'][key]==pred) if v['status']=='COMPLETE' else None
   predictions.append({'person':person,'key':key,'prediction':pred,'probability':p,'actual':v['labels'][key],
    'hit':hit,'binary_assertion_brier':(p-int(hit))**2 if p is not None and hit is not None else None})
 pd.DataFrame(predictions).to_csv(BASE/'prediction_score.csv',index=False)
 manifest={'run_id':RUN,'prereg_commit':PREREG,'C_CH_qualified':True,'sources':{},'checks':{n:sha(CHECK/(n+'.json')) for n in REQUIRED}}
 for arm in frames:
  folder=BASE/arm if arm in ('ref','H') else ROOT/'results/relu_doors_0919'/arm
  manifest['sources'][arm]={'path':str(folder.relative_to(ROOT)),'csv_sha256':sha(folder/'per_task.csv'),'provenance_sha256':sha(folder/'provenance.json'),'launch_commit':provs[arm]['git_hash']}
 write(BASE/'reuse_manifest.json',manifest)
 lines=['# relu_doors_h_ref_0920 — 新規refでH単独を判定','',f"状態: **{v['status']}**。独立監査なし。",'',
 '| 腕 | t31–50 onlineのseed中央値 | seed範囲 | 窓≥0.5 |','|---|---:|---:|---:|']
 for arm in ('ref','H','C','CH'):
  w=np.array(v.get('windows',{}).get(arm,[]))
  if len(w):lines.append(f'| {arm} | {np.median(w):.10f} | {w.min():.10f}–{w.max():.10f} | {int((w>=.5).sum())}/10 |')
 lines+=['','## 登録判定','']+[f"- **{k}: {label}**" for k,label in v['labels'].items()]
 if v['status']=='COMPLETE':
  lines += ['',f"H−ref対応差の中央値: {v['median_d_HR']:.10f}。H−CH: {v['median_d_HC']:.10f}。",
   f"H−CHの順位区間[d₂,d₉]: {v['I_HC']}。固定帯δ={S.DELTA}。",
   f"層別条件: `{json.dumps(v['layer_conditions'])}`。H第1層: {v['R1_H']}。",
   'C_NEEDEDはCH水準の維持にCが要る、C_REDUNDANTは登録帯での非劣性。TIEを同等性とは解釈しない。']
 lines+=['','## 予測','']
 for who in ('Issa','Codex'):
  rows=[r for r in predictions if r['person']==who]
  lines.append(f"- {who}: {sum(r['hit'] for r in rows)}/3" if v['status']=='COMPLETE' else f'- {who}: 未採点')
  if who=='Codex' and v['status']=='COMPLETE':lines.append(f"  3つの登録命題のbinary Brier平均: {np.mean([r['binary_assertion_brier'] for r in rows]):.6f}（多クラスBrierではない）。")
 lines+=['','## 検査・開示','',
 'C/CHを旧起動コードと新実装でR10×2tasks×400epochs照合して再利用資格を確認。ref/Hは各R10を初期状態から50tasks、新規run配下へ直列実行した。',
 '全12必須検査と列挙した変異を確認。空・欠落・古いPASSは収集器で拒否。sourceと参照hashを集計時にも照合。',
 '検査attempt1はr_a1の独立計算をseed一括のnormにしていたため数ULP不一致で停止。seedごとの親の演算順へ修正。学習コードと許容誤差は変更していない。旧検査出力は保存。',
 'H検査はseed200–209。本走H0–9をprobeしていない。診断gate(z>0)とclampの訓練微分(z=0で1)を区別する。',
 f'登録commit: `{PREREG}`。本走起動commit: `{provs["H"]["git_hash"]}`。起動時git状態を記録し、終了時の状態で差し替えていない。',
 f'実測本走秒数: ref={provs["ref"]["wall_clock_s"]:.3f}、H={provs["H"]["wall_clock_s"]:.3f}。',
 '前走のR20→R10不一致を見た後の新規登録。既知C/CHから作った帯・旧ref/Cから作ったθ1を維持し、新refで再推定していない。',
 '予測は前登録から継承。本走中は進捗・生存・資源のみ確認、性能による打切りや判定の変更はしていない。',
 '数値と判定の正本は本summaryとverdict.json/verdict.csv。全seedはper_seed.csv、t1/t50の層別量はlayer_table.csv。',
 '新規snapshotからのm、r_a1分子分母、厳密z=0の数はsnapshot_diagnostics.csv。欠損/未定義の比を0に埋めていない。',
 '生データと検査ログの退避先はbackup_manifest.json。単一Codexによる実装・自己検査で独立監査なし。']
 (BASE/'summary.md').write_text('\n'.join(lines)+'\n')
 print(json.dumps({'status':v['status'],'labels':v['labels']}))
if __name__=='__main__':main()
