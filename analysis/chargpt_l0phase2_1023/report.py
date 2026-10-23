#!/usr/bin/env python3
"""Fixed, non-selected unit comparisons and preregistered judgments."""
import json,subprocess
import measure as Q
np,OUT,ROOT=Q.np,Q.OUT,Q.ROOT

def read(kind,arm,seed,task,step=10000):return np.load(Q.file_path(kind,arm,seed,task,step))
def rng(xs,digits=3):
    a=[float(v) for v in xs if v is not None and np.isfinite(float(v))]
    if not a:return 'NA'
    lo,hi=min(a),max(a)
    return f'{lo:.{digits}f}–{hi:.{digits}f}'
def ft(x):return 'NA' if x is None else f'{x:.3f}'
def table(headers,rows):
    return '\n'.join(['| '+' | '.join(map(str,headers))+' |','|'+'---|'*len(headers)]+['| '+' | '.join(map(str,r))+' |' for r in rows])

def main():
    import csv
    full=list(csv.DictReader((OUT/'full_states.csv').open()))
    scalar=list(csv.DictReader((OUT/'states.csv').open()))
    def cv(rows):
        for r in rows:
            for k,v in r.items():
                if v=='':r[k]=None
                else:
                    try:r[k]=float(v)
                    except ValueError:pass
        return rows
    full,scalar=cv(full),cv(scalar)
    seedrows=[];motion=[];j={};sensitivity=[]
    for arm in ['C3','ReLUcap']:
        for seed in range(10):
            both=[read('full',arm,seed,t) for t in [39,100]]
            pred=np.concatenate([r['q_1e-10'] for r in both])
            sr=dict(arm=arm,seed=seed)
            for er in ['fit','switch']:
                for target in [er,'canonical_'+er,'allparam_'+er]:
                    cr=Q.compare(pred,np.concatenate([r[target] for r in both]))
                    sr.update({target+'_'+k:v for k,v in cr.items()})
                    sr[target+'_pass']=cr['n']>=32 and cr['rho'] is not None and cr['rho']>.5 and cr['balanced_accuracy'] is not None and cr['balanced_accuracy']>.65
            seedrows.append(sr)
            for task in [39,100]:
                a=read('full',arm,seed,task)
                for tol in [1e-8,1e-12]:
                    cr=Q.compare(a['q_1e-10'],a[f'q_{tol:g}'])
                    sensitivity.append(dict(arm=arm,seed=seed,task=task,tol=tol,n_main=int(np.isfinite(a['q_1e-10']).sum()),n_alternative=int(np.isfinite(a[f'q_{tol:g}']).sum()),**cr))
            for oldtask,oldstep,newtask,newstep,label in [(39,10000,100,10000,'39_to_100'),(99,10000,100,100,'head100'),(100,100,100,10000,'body100')]:
                old=read('scalar',arm,seed,oldtask,oldstep);new=read('scalar',arm,seed,newtask,newstep)
                wnew=new['W'];mu=wnew@old['m'];s=np.sqrt(np.einsum('ij,jk,ik->i',wnew,old['cov'],wnew))
                targets={'actual':old['r']-new['r'],'ownW':old['r']-mu/s}
                for probe in ['chain','uniform']:
                    for metric in ['r','adam_r']:
                        key=probe+'_'+metric+'_tol1e-10'
                        if key not in old:continue
                        for name,target in targets.items():
                            cr=Q.compare(old[key],target,old['alive'])
                            motion.append(dict(arm=arm,seed=seed,interval=label,probe=probe,metric=metric,target=name,delta_r_median=Q.med(-target[old['alive']]),**cr,
                                               passed=cr['rho'] is not None and cr['rho']>.5))
        late=[r for r in full if r['arm']==arm and r['task']==100 and r['tol']==1e-10]
        n=sum(r['n_valid']>=16 and r['q_positive'] is not None and r['q_positive']>=.75 for r in late)
        j[arm+'_J1']=dict(count=n,out_of=10,passed=n>=8,identifiable_seeds=sum(r['n_valid']>=16 for r in late))
        for e in ['fit','switch','canonical_fit','canonical_switch','allparam_fit','allparam_switch']:
            n=sum(r[e+'_pass'] for r in seedrows if r['arm']==arm)
            j[arm+'_J2_'+e]=dict(count=n,out_of=10,passed=n>=8)
        for label in ['39_to_100','head100','body100']:
            for name in ['actual','ownW']:
                sel=[r for r in motion if r['arm']==arm and r['interval']==label and r['probe']=='chain' and r['metric']=='r' and r['target']==name]
                n=sum(r['passed'] for r in sel);j[arm+'_J3_'+label+'_'+name]=dict(count=n,out_of=10,passed=n>=8)
        for task in [39,100]:
            rr=[r for r in full if r['arm']==arm and r['task']==task and r['tol']==1e-10]
            n=sum(r['joint_fit_ratio']<=.1 for r in rr)
            j[f'{arm}_joint_t{task}']=dict(count=n,out_of=10,passed=n>=8)
    predictions=[]
    for label,p in [('C3_J1',.65),('C3_J2_fit',.15),('C3_J2_switch',.35),('C3_joint_t100',.30)]+[(f'C3_J3_{k}_actual',.25) for k in ['39_to_100','head100','body100']]:
        y=int(j[label]['passed']);predictions.append(dict(test=label,probability=p,outcome=y,brier=(p-y)**2))
    Q.csvwrite(OUT/'seed_comparisons.csv',seedrows);Q.csvwrite(OUT/'motion.csv',motion);Q.csvwrite(OUT/'sensitivity.csv',sensitivity)
    Q.dump(OUT/'judge.json',dict(judgments=j,predictions=predictions,mean_brier=np.mean([p['brier'] for p in predictions]),
         scope='Conditional least-squares geometry. No causal identification of actual GPT training.',training_steps=0))
    text=['# L0 の相殺後の押しを相2と照合した結果','',
      '保存状態のみ。C3/ReLUcap各10seed。特徴上100状態、後段の注意・FFN・LNを含む40状態。spec登録: 160259ae。',
      '', '## 主判定', '',table(['検査','C3 成立seed','判定'],[[k,str(j['C3_'+k]['count'])+'/10','○' if j['C3_'+k]['passed'] else '×'] for k in ['J1','J2_switch','J2_fit','J3_39_to_100_actual','J3_head100_actual','J3_body100_actual']]),'',
      'J1は他unitを除いた局所線形モデルの期待値の符号。自己項正値だけとは異なるが、それだけでは実際の合計が沈む理由にならない。J2は実際の合計に対する順位相関>0.5かつbalanced accuracy>0.65を8/10seed。fitとswitchは別問題として判定した。', '', '## 後段を通した相殺後の予測（seed範囲）','']
    rows=[]
    for arm in ['C3','ReLUcap']:
        for task in [39,100]:
            rr=[r for r in full if r['arm']==arm and r['task']==task and r['tol']==1e-10]
            rows.append([arm,task,rng([r['n_valid'] for r in rr],0),rng([r['q_positive'] for r in rr]),rng([r['energy_median'] for r in rr]),rng([r['fit_rho'] for r in rr]),rng([r['switch_rho'] for r in rr]),rng([r['joint_fit_ratio'] for r in rr])])
    text += [table(['腕','課題末','識別可/32','予測が沈める割合','残存energy比中央値','ρ 現課題合計','ρ 独立切替合計','共同fit残差比'],rows),'',
      '残存energy比は各unitの出力方向のうち他unit＋固定残差で表せない割合。共同fit残差比は ||Fᵀ(p−q)||/||Fᵀ(p−1/V)||。0.1以下は必要条件の目安であり、他の理論仮定を保証しない。', '',
      '## seed別：課題39/100末を結合したunit順位相関と符号精度','']
    text += [table(['腕','seed','ρ fit','BA fit','ρ switch','BA switch','ρ 全parameter fit','ρ 全parameter switch'],[[r['arm'],r['seed'],ft(r['fit_rho']),ft(r['fit_balanced_accuracy']),ft(r['switch_rho']),ft(r['switch_balanced_accuracy']),ft(r['allparam_fit_rho']),ft(r['allparam_switch_rho'])] for r in seedrows]),'',
      'own-WはL0の当該up行への合計勾配。全parameterは入力を変える上流parameterも含めたSGDのr全微分。Adam固定分母の特徴上の結果はstates.csv/motion.csvに別記。', '',
      '## 特徴上の予測：何を除くと符号が変わるか','']
    rr=[]
    for arm in ['C3','ReLUcap']:
        for task in [9,39,100]:
            base=[r for r in scalar if r['arm']==arm and r['task']==task and r['step']==10000 and r['probe']=='chain' and r['metric']=='r']
            vals=[]
            for proj in ['constant','constant_z','tol1e-10']:
                vals.append(rng([r['q_positive'] for r in base if r['projection']==proj]))
            fullr=[r for r in base if r['projection']=='tol1e-10']
            rr.append([arm,task,*vals,rng([r['n_valid'] for r in fullr],0),rng([r['fit_rho'] for r in fullr]),rng([r['switch_rho'] for r in fullr])])
    text += [table(['腕','末','定数だけ','定数+z','定数+全他unit','識別可数','ρ fit','ρ switch'],rr),'',
      'この表は非線形後段を通していない特徴上の理論診断であり、上の出力空間での測定と区別する。', '', '## 保存点間の沈みとの照合','']
    rr=[]
    for arm in ['C3','ReLUcap']:
        for interval in ['39_to_100','head100','body100']:
            for target in ['actual','ownW']:
                sel=[r for r in motion if r['arm']==arm and r['interval']==interval and r['probe']=='chain' and r['metric']=='r' and r['target']==target]
                rr.append([arm,interval,target,rng([r['rho'] for r in sel]),str(sum(r['passed'] for r in sel))+'/10',rng([r['delta_r_median'] for r in sel])])
    text += [table(['腕','区間','実変化/入力固定W変化','ρ(予測,-Δr)','ρ>0.5','Δr 中央値'],rr),'',
      '区間開始点の予測を、同じunitの終点変化と比較した。途中の力は保存されていないので積分の再現・因果寄与率にはしない。', '', '## probeと数値検査','']
    for e in ['fit','switch']:
        rr=[r for r in full if r['tol']==1e-10]
        text.append(f'- 8系列と既存64系列の実際のown-W押し: {e}の符号一致 {rng([r["probe_"+e+"_agreement"] for r in rr])}、Spearman {rng([r["probe_"+e+"_rho"] for r in rr])}。')
    g=json.loads((OUT/'gate.json').read_text())
    text += [f'- 小行列leave-one-outとの相対差 {g["algebra"]["leave_one_out_relative"]:.3g}。重複列は識別不能として扱う検査に合格。非Gaussianのp<1/2反例でQ={g["algebra"]["nonGaussian_counterexample"]:.3g}。',
      f'- 全40状態の既存full値との相対差最大 {max(r[e+"_reference"] for r in full for e in ["fit","switch"]):.3g}。JVPと独立autogradの内積相対差最大 {max(r[e+"_bilinear"] for r in full for e in ["fit","switch"]):.3g}。',
      f'- Gram閾値1e-8/1e-12で共通識別unitの符号一致 {rng([r["agreement"] for r in sensitivity])}。識別数の変化はsensitivity.csv。', '',
      '## 適用の限界', '',
      'Poutは全512unitの現在の出力接線と固定残差を取り除く。任意の128次元down方向を学習し直すモデルではなく、現在の出力方向の振幅を共同fitする局所二乗損失モデルの期待値である。CE、課題依存の特徴、Markovラベル、Adam履歴はこの等分散・独立ラベル平均の定理では扱われない。joint-fit必要条件の成否も別記した。', '',
      '時間相関する勾配noiseと読み出しの更新が作る駆動は、途中のbatch/残差相関が保存されていないため、この保存点からは実測していない。正のnoise成分の式だけを原因確認の代用にしない。', '',
      '## 登録予測の採点', '',table(['検査','事前P','成否','Brier'],[[p['test'],p['probability'],p['outcome'],f'{p["brier"]:.4f}'] for p in predictions]),'',
      f'平均Brier: {np.mean([p["brier"] for p in predictions]):.4f}。閾値・予測を変更していない。', '',
      'コード: analysis/chargpt_l0phase2_1023/。checkpointを更新していない。生配列とログはbackup_manifest.json参照。']
    (OUT/'summary.md').write_text('\n'.join(text)+'\n')
    (OUT/'judge.md').write_text('# 登録判定\n\n'+table(['判定','成立seed/10','結果'],[[k,v['count'],'PASS' if v['passed'] else 'NOT_MET'] for k,v in j.items()])+'\n')
    print(json.dumps(j,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
