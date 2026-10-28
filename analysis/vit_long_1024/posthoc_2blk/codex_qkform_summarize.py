"""Lightweight aggregation; no checkpoint/model loading and no training."""
import json
import math
from pathlib import Path
import numpy as np

OUT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_qk_formation')
CASES=[('SL',10,'shared',1),('SL',10,'zero',1),('SL',60,'shared',1),('SL',100,'shared',1),
       ('R0L',10,'zero',1),('R0L',100,'zero',1),('SL',100,'zero',1),('SL',100,'zero',4)]

def rankdata(x):
    x=np.asarray(x)
    return np.array([.5*(np.sum(x<v)+np.sum(x<=v)-1)+1 for v in x],dtype=float)

def corr(x,y):
    if len(x)<3 or np.std(x)<1e-12 or np.std(y)<1e-12:return None
    return float(np.corrcoef(rankdata(x),rankdata(y))[0,1])

def partial_time(x,y,t):
    design=np.stack([np.ones(len(t)),rankdata(t)],axis=1)
    x=rankdata(x);y=rankdata(y)
    x=x-design@np.linalg.lstsq(design,x,rcond=None)[0]
    y=y-design@np.linalg.lstsq(design,y,rcond=None)[0]
    if np.std(x)<1e-10 or np.std(y)<1e-10:return None
    return float(np.corrcoef(x,y)[0,1])

def powerfit(t,y,train):
    t=np.array(t,dtype=float);y=np.array(y,dtype=float);train=np.array(train,dtype=bool)
    alpha,intercept=np.polyfit(np.log(t[train]/20),np.log(y[train]),1)
    pred=np.exp(intercept)*(t/20)**alpha
    return dict(alpha=float(alpha),at20=float(np.exp(intercept)),tasks=t.tolist(),actual=y.tolist(),predicted=pred.tolist(),
                mape=float(np.mean(abs(pred/y-1))),max_ape=float(np.max(abs(pred/y-1))),
                heldout_mape=float(np.mean(abs(pred[~train]/y[~train]-1))) if (~train).any() else None)

def head_overlap(W,V):
    W=W-W.mean(0);V=V-V.mean(0)
    def bas(A):
        _,s,v=np.linalg.svd(A,full_matrices=False)
        return v[s>s[0]*1e-6].T
    U,Z=bas(W),bas(V)
    s=np.clip(np.linalg.svd(U.T@Z,compute_uv=False),0,1)
    return dict(cos2=float(np.mean(s*s)),angles_deg=np.rad2deg(np.arccos(s)).tolist(),dimensions=[U.shape[1],Z.shape[1]])

def main():
    cp=json.loads((OUT/'codex_checkpoints.json').read_text())
    r=dict(checkpoints=cp,replays={},fits={},correlations={},head_comparisons={},energy={})
    for arm in ('SL','R0L'):
        states=cp['states'][arm];t=list(range(20,101,10))
        r['fits'][arm]={}
        for key in ('a_fixed','S','a_own','A_norm'):
            y=[states[str(k)][key] for k in t]
            r['fits'][arm][key]={'all':powerfit(t,y,[True]*len(t)),
                                'early':powerfit(t,y,[k<=50 for k in t])}
        inc=[v for v in cp['increments'][arm] if v['t0']>=10]
        r['energy'][arm]={k:sum(v[k] for v in inc) for k in ('A_energy_radial','A_energy_step','a_energy_radial','a_energy_step','Q_energy_radial','Q_energy_step','K_energy_radial','K_energy_step')}
        sat=[states[str(d['t0'])]['saturation'] for d in inc]
        ts=[d['t0'] for d in inc]
        r['correlations'][arm]={k:dict(spearman=corr(sat,[d[k] for d in inc]),
                                                   partial_task=partial_time(sat,[d[k] for d in inc],ts))
                                   for k in ('dA','Q_pr','alignment_correct','alignment_legacy','radial_a_cos')}
        # Norm-square drift plus separately fitted R,G,T: an empirical time model, not an autonomous feedback law.
        tt=np.array(t,dtype=float);train=tt<=50
        qcoef=np.polyfit(tt[train],[states[str(int(k))]['Q_norm']**2 for k in tt[train]],1)
        kcoef=np.polyfit(tt[train],[states[str(int(k))]['K_norm']**2 for k in tt[train]],1)
        pred=np.sqrt(np.maximum(np.polyval(qcoef,tt),0)*np.maximum(np.polyval(kcoef,tt),0))/8
        params={}
        for key in ('R','G','T'):
            fit=powerfit(t,[states[str(k)]['factors'][key] for k in t],train)
            pred*=np.array(fit['predicted']);params[key]={k:fit[k] for k in ('alpha','at20')}
        y=np.array([states[str(k)]['a_fixed'] for k in t])
        r['fits'][arm]['factor_model']=dict(q_norm2_slope_intercept=qcoef.tolist(),k_norm2_slope_intercept=kcoef.tolist(),
                                          factors=params,tasks=t,predicted=pred.tolist(),actual=y.tolist(),
                                          heldout_mape=float(np.mean(abs(pred[~train]/y[~train]-1))))
    for arm,t,head,temp in CASES:
        tag=f'codex_replay_{arm}_t{t}_{head}_temp{temp}'
        path=OUT/f'{tag}.json'
        if not path.exists():continue
        d=json.loads(path.read_text())
        if not d.get('complete'):continue
        a=np.load(path.with_suffix('.npz'));v=a['values'];fields=a['fields'].tolist()
        values={k:v[:,j] for j,k in enumerate(fields)}
        geo={x['step']:x for x in d['geometry']}
        windows=[]
        for lo,hi in ((0,100),(100,500),(500,1500)):
            mask=(values['step']>lo)&(values['step']<=hi)
            w=dict(lo=lo,hi=hi,mean_loss=float(values['batch_loss'][mask].mean()),
                   q_c_mean=float(values['q_c'][mask].mean()),q_c_rms=float(np.sqrt(np.mean(values['q_c'][mask]**2))),
                   k_c_mean=float(values['k_c'][mask].mean()),k_c_rms=float(np.sqrt(np.mean(values['k_c'][mask]**2))),
                   gamma_raw_gradient_mean=float(values['gamma_gradient_mean'][mask].mean()),
                   gamma_u_mean=float(values['gamma_u_mean'][mask].mean()),
                   gamma_mean_change=float(values['gamma_update_mean'][mask].sum()),
                   gamma_energy_radial=float(values['gamma_update_radial'][mask].sum()),
                   gamma_energy_steps=float(values['gamma_step_sq'][mask].sum()),
                   delta_S=geo[hi]['S']-geo[lo]['S'],delta_a_fixed=geo[hi]['a_fixed']-geo[lo]['a_fixed'])
            for key in ('q','k'):
                denom=float(values[f'{key}_step_sq'][mask].sum())
                if lo==0:net=geo[hi]['walk'][key]['net_norm']
                else:
                    one=np.load(OUT/f'{tag}_step{lo}_small.npz')[key.upper()]
                    two=np.load(OUT/f'{tag}_step{hi}_small.npz')[key.upper()]
                    net=float(np.linalg.norm((two-one).astype('float64')))
                w[f'{key}_net_norm']=net;w[f'{key}_independent_norm']=math.sqrt(denom)
                w[f'{key}_coherence']=net*net/denom
                w[f'{key}_norm2_radial']=float(values[f'{key}_radial'][mask].sum())
                w[f'{key}_norm2_steps']=denom
            steps=[x for x in d['geometry'] if lo<x['step']<=hi]
            w['firstorder_qk']=sum(x['delta25']['a_firstorder_qk'] for x in steps)
            w['firstorder_gamma']=sum(x['delta25']['a_firstorder_gamma'] for x in steps)
            w['firstorder_total']=w['firstorder_qk']+w['firstorder_gamma']
            w['exact_radial_norm_growth']=sum(x['delta25']['a_energy_radial']/(geo[x['step']]['a_fixed']+geo[x['step']-25]['a_fixed']) for x in steps)
            w['exact_step_norm_growth']=sum(x['delta25']['a_energy_step']/(geo[x['step']]['a_fixed']+geo[x['step']-25]['a_fixed']) for x in steps)
            arc=1e-4*384*np.sum(np.sqrt(values['q_c'][mask]**2+values['k_c'][mask]**2))
            w['qk_path_length']=float(arc);w['sensitivity_weighted_alignment']=w['firstorder_qk']/arc
            di=[x for x in d['diagnostics'] if lo<x['step']<=hi and x['score_grad_energy']>0]
            for key in ('score_top5_energy','q_signal_pr','k_signal_pr','gradient_Q_pr','gradient_K_pr'):
                va=[x[key] for x in di if key in x]
                w[key]=float(np.mean(va)) if va else None
            windows.append(w)
        end=geo[1500]
        comparison=next((i for i in cp['increments'][arm] if i['t0']==t),None)
        if comparison is None:comparison=next(i for i in cp['increments'][arm] if i['t1']==t)
        r['replays'][tag]=dict(arm=arm,t=t,head=head,temperature=temp,windows=windows,
                              start=geo[0],end=end,records=d['records'],geometry=d['geometry'],
                              ten_task_q_projection=end['walk']['q']['net_norm']*math.sqrt(10),
                              ten_task_k_projection=end['walk']['k']['net_norm']*math.sqrt(10),
                              decade_comparison=comparison,
                              max_gauge_error=float(max(values['gauge_relative_error'])),
                              max_perstep_adam_error=float(max(values['adam_update_error'])),
                              integrated_adam_error=end['integrated_adam_relative_error'],
                              manual_forward_error=d['manual_forward_error'],rss_mb=d['rss_mb'],cuda_peak_mb=d['cuda_peak_mb'])
    for t in (10,100):
        shared=f'codex_replay_SL_t{t}_shared_temp1';zero=f'codex_replay_SL_t{t}_zero_temp1'
        if shared not in r['replays'] or zero not in r['replays']:continue
        a=np.load(OUT/f'{shared}.npz')['original_head']
        s=np.load(OUT/f'{shared}_step100_small.npz')['head']
        z=np.load(OUT/f'{zero}_step100_small.npz')['head']
        r['head_comparisons'][t]=dict(zero100_to_original=head_overlap(a,z),zero100_to_shared100=head_overlap(s,z),
                                      shared100_to_original=head_overlap(a,s),
                                      S_shared_over_zero=r['replays'][shared]['end']['S']/r['replays'][zero]['end']['S'])
    sigpath=OUT/'codex_signals.json'
    if sigpath.exists():
        sig=json.loads(sigpath.read_text());r['signals']={};r['signal_correlations']={}
        for arm in ('SL','R0L'):
            r['signals'][arm]={}
            for regime in ('early','late'):
                points=[d for d in sig['adjacent'][arm] if (1<=d['t0']<10 if regime=='early' else d['t0']>=10)]
                agg={}
                for mode in sig['modes']:
                    for kind in ('q','k'):
                        for weighting in ('energy','image_normalized'):
                            key=f'{mode}_{kind}_{weighting}'
                            agg[key]={k:float(np.mean([v['overlaps'][key][str(k)]['cos2'] for v in points if key in v['overlaps']])) for k in (4,8,16)}
                r['signals'][arm][regime]=agg
            r['signals'][arm]['longlag']=sig['longlag'][arm]
            points=[d for d in sig['adjacent'][arm] if d['t0']>=10]
            persistence=[np.mean([v['overlaps'][f'common_task101_labels_{kind}_energy']['8']['cos2'] for kind in ('q','k')]) for v in points]
            inc=[v for v in cp['increments'][arm] if v['t0']>=10]
            ret=[math.log(cp['states'][arm][str(v['t1'])]['factors']['T']/cp['states'][arm][str(v['t0'])]['factors']['T']) for v in inc]
            r['signal_correlations'][arm]=dict(persistence_to_retention_change=corr(persistence,ret),
                    persistence_to_alignment=corr(persistence,[d['alignment_correct'] for d in inc]),
                    persistence_to_A_step=corr(persistence,[d['dA'] for d in inc]),
                    mean_head_cos2=float(np.mean([x['cos2'] for x in sig['head_adjacent'][arm] if x['t0']>=10])),
                    mean_feature_cos2=float(np.mean([x['cos2'] for x in sig['feature_adjacent'][arm] if x['t0']>=10])))
    (OUT/'codex_summary.json').write_text(json.dumps(r,indent=2,allow_nan=False))
    print(json.dumps({'completed_replays':len(r['replays']),'signals':bool(r.get('signals')),
                      'fits':{a:{k:r['fits'][a][k]['all']['alpha'] for k in ('a_fixed','S')} for a in r['fits']}},indent=2))

if __name__=='__main__':main()
