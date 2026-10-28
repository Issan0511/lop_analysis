"""Finish the growth-law comparisons using saved outputs only; NumPy on CPU."""
import json
import math
from pathlib import Path
import numpy as np

OUT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_qk_formation')

def forecast(rows,states,degree):
    train=[x for x in rows if 20<=x['t0'] and x['t1']<=50]
    mid=np.array([(x['t0']+x['t1'])/2 for x in train])
    params={}
    for key in ('radial_energy','step_energy'):
        y=np.array([x[key]/x['tasks'] for x in train])
        params[key]=np.polyfit(mid,y,degree).tolist()
    energy=states['50']['a_fixed']**2;rec=[]
    for t in range(60,101,10):
        er=10*float(np.polyval(params['radial_energy'],t-5))
        eq=10*float(np.polyval(params['step_energy'],t-5))
        old=energy;energy+=er+eq
        pred=math.sqrt(energy) if energy>0 else None
        actual=states[str(t)]['a_fixed']
        rec.append(dict(t=t,predicted=pred,actual=actual,relative_error=pred/actual-1 if pred else None,
                        radial_energy=er,step_energy=eq,negative_step_prediction=eq<0))
    errs=[abs(x['relative_error']) for x in rec if x['relative_error'] is not None]
    return dict(degree=degree,coefficients=params,calibration_intervals=[[x['t0'],x['t1']] for x in train],
                rows=rec,heldout_mape=float(np.mean(errs)),max_ape=max(errs))

def main():
    s=json.loads((OUT/'codex_summary.json').read_text());cp=s['checkpoints']
    cpu=json.loads((OUT/'codex_finish_cpu.json').read_text())
    r=dict(reference=cpu['reference'],forecasts={},activity={},temperature={},scores=[])
    for arm in ('SL','R0L'):
        r['forecasts'][arm]={name:forecast(cpu['checkpoints'][arm],cp['states'][arm],degree) for name,degree in [('constant_energy',0),('affine_energy',1)]}
    for name,d in s['replays'].items():
        a=np.load(OUT/f'{name}.npz');v=a['values'];fields=a['fields'].tolist()
        cq=v[:,fields.index('q_c')];ck=v[:,fields.index('k_c')]
        c=np.sqrt((cq*cq+ck*ck)/2)
        l=1e-4*384*math.sqrt(2)*float(c.sum())
        dw=math.hypot(d['end']['walk']['q']['net_norm'],d['end']['walk']['k']['net_norm'])
        rho=dw/l
        r['activity'][name]=dict(sum_c=float(c.sum()),c_mean=float(c.mean()),c_rms=float(np.sqrt(np.mean(c*c))),
                effective_steps=float(c.sum()**2/(c*c).sum()),qk_path_length=l,qk_net_norm=dw,temporal_alignment=rho,
                tail_c_fraction=float(c[500:].sum()/c.sum()),
                F_change_common=cpu['replays'][name]['total']['delta_F'],F_change_start=d['end']['a_fixed']-d['start']['a_fixed'],
                windows=[dict(lo=lo,hi=hi,sum_c=float(c[lo:hi].sum()),mean_c=float(c[lo:hi].mean()),path_length=1e-4*384*math.sqrt(2)*float(c[lo:hi].sum())) for lo,hi in [(0,100),(100,500),(500,1500)]])
    train=['codex_replay_SL_t10_shared_temp1','codex_replay_R0L_t10_zero_temp1']
    L=np.array([r['activity'][x]['qk_path_length'] for x in train])
    V=np.array([r['activity'][x]['qk_net_norm'] for x in train])
    F=np.array([r['activity'][x]['F_change_common'] for x in train])
    rho=float(L@V/(L@L));slope=float(L@F/(L@L))
    for name,d in r['activity'].items():
        d['predicted_net_norm']=rho*d['qk_path_length']
        d['net_relative_error']=d['predicted_net_norm']/d['qk_net_norm']-1
        d['predicted_F_change_simple']=slope*d['qk_path_length']
        d['F_error_simple']=d['predicted_F_change_simple']-d['F_change_common']
    r['activity_calibration']=dict(training=train,rho=rho,F_per_path=slope,
        note='Post hoc constant-directional-factor test calibrated at task10; not the task20–50 time extrapolation.')
    for temp in (1,4):
        name=f'codex_replay_SL_t100_zero_temp{temp}'
        d=s['replays'][name];u=cpu['replays'][name]['total'];old=d['end']['delta_start']
        f0=d['start']['a_fixed'];f1=d['end']['a_fixed'];den=f0+f1
        rad=old['a_energy_radial'];step=old['a_energy_step'];pa=(rad/(2*f0))**2
        r['temperature'][temp]=dict(dA=old['dA'],da_start=old['da_fixed'],da_common=u['da'],
                delta_F_start=f1-f0,radial_start=rad/den,step_start=step/den,
                perpendicular_start=(step-pa)/den,parallel_start=pa/den,
                delta_F_common=u['delta_F'],radial_common=u['radial_norm'],step_common=u['step_norm'],
                perpendicular_common=u['perpendicular_norm'],parallel_common=u['parallel_norm'],
                radial_cos_common=u['radial_cos'])
    for arm in ('SL','R0L'):
        inc=[d for d in cp['increments'][arm] if d['t0']>=10]
        r.setdefault('lagged_rows',{})[arm]=[dict(t0=d['t0'],t1=d['t1'],saturation=cp['states'][arm][str(d['t0'])]['saturation'],
                    dA=d['dA'],Q_pr=d['Q_pr'],alignment=d['alignment_correct']) for d in inc]
    # Evaluate the original preregistered N1--N10 without changing their criteria.
    ver=json.loads((OUT/'codex_verification.json').read_text())
    diff=cp['log_difference'];fraction=(diff['T']+diff['I'])/sum(diff.values())
    shared=s['replays']['codex_replay_SL_t10_shared_temp1'];zero=s['replays']['codex_replay_SL_t10_zero_temp1']
    sw=shared['windows'];Sshare=sw[1]['delta_S']/(shared['end']['S']-shared['start']['S'])
    late=s['signals'];qsl=late['SL']['late']['common_task101_labels_q_energy']['8'];qrl=late['R0L']['late']['common_task101_labels_q_energy']['8']
    ksl=late['SL']['late']['common_task101_labels_k_energy']['8'];krl=late['R0L']['late']['common_task101_labels_k_energy']['8']
    r['scores']=[
      dict(id='N1',verdict='支持',evidence=f"六因子の最大相対誤差 {ver['identity_max_relative_errors']['six_factor']:.3g}"),
      dict(id='N2',verdict='支持',evidence=f"保持T＋入力Iは腕差の{100*fraction:.2f}%"),
      dict(id='N3',verdict='支持',evidence='独立更新の見積もりはSL10 queryで0.554、実測2.095。時間交差項を要する'),
      dict(id='N4',verdict='支持（符号は測定）',evidence=f"尺度対称性の最大誤差{ver['max_gauge_error']:.3g}。R0Lでγの平均変位は負"),
      dict(id='N5',verdict='支持',evidence=f"SL10共有のS増加の{Sshare*100:.1f}%が101〜500更新。終点共有/head0={shared['end']['S']/zero['end']['S']:.4f}"),
      dict(id='N6',verdict='支持（変位の大きさ）',evidence=f"score/4のdA比{r['temperature'][4]['dA']/r['temperature'][1]['dA']:.4f}、開始tokenのda比{r['temperature'][4]['da_start']/r['temperature'][1]['da_start']:.4f}"),
      dict(id='N7',verdict='支持（記述相関）',evidence=f"SLのSpearmanは飽和率対dA={s['correlations']['SL']['dA']['spearman']:.3f}、対PR(ΔQ)={s['correlations']['SL']['Q_pr']['spearman']:.3f}。課題番号との分離不可"),
      dict(id='N8',verdict='外れ（逆方向）',evidence=f"共通ラベルの上位8方向cos²はq: SL {qsl:.4f}<R0L {qrl:.4f}、k: {ksl:.4f}<{krl:.4f}"),
      dict(id='N9',verdict='外れ（逆方向）',evidence=f"head0の100更新後対元headのcos²はSL10 {s['head_comparisons']['10']['zero100_to_original']['cos2']:.4f}、SL100 {s['head_comparisons']['100']['zero100_to_original']['cos2']:.4f}"),
      dict(id='N10',verdict='支持（固定ξ10に限定）',evidence=f"t20〜100の指数はSL {s['fits']['SL']['a_fixed']['all']['alpha']:.4f}、R0L {s['fits']['R0L']['a_fixed']['all']['alpha']:.4f}")]
    dest=OUT/'codex_finish_analysis.json'
    with dest.open('x') as f:json.dump(r,f,indent=2,allow_nan=False)
    print(json.dumps({'activity_calibration':r['activity_calibration'],
          'forecast_mape':{a:{k:v['heldout_mape'] for k,v in d.items()} for a,d in r['forecasts'].items()},
          'temperature':r['temperature']},indent=2))

if __name__=='__main__':main()
