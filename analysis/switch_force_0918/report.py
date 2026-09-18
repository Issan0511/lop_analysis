"""Independent NumPy endpoint checks, registered seed inference and figures."""
from pathlib import Path
import csv, hashlib, json, subprocess, sys
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
RAW=Path('/home/issan/Projects/obsidian-research-data/switch_force_0918')
CP=Path('/home/issan/Projects/obsidian-research-data/zero_attraction_0913/training/ckpts')
OUT=ROOT/'results/switch_force_0918'
ARMS=['LR_a0p1_q0','LR_a0p7_q0']; STEPS=[200000,1000000,5000000]
VERIFICATION={}

def verify(name,a,b,tol=1e-10):
    a,b=np.broadcast_arrays(a,b); error=np.abs(a-b); bound=tol*(1+np.abs(b))
    s=VERIFICATION.setdefault(name,dict(count=0,failed=0,max_abs=0.))
    s['count']+=a.size; s['failed']+=int(np.sum(~np.isfinite(error)|(error>bound))); s['max_abs']=max(s['max_abs'],float(np.max(error)))

def ci(v,alpha=.05):
    v=np.asarray(v,dtype=float)
    m=float(v.mean()); sd=float(v.std(ddof=1)); hw=float(stats.t.ppf(1-alpha/2,len(v)-1)*sd/np.sqrt(len(v)))
    return dict(mean=m,lo=m-hw,hi=m+hw,sd=sd,n=len(v))

def csvout(name,rows):
    with (OUT/name).open('w',newline='') as f:
        wr=csv.DictWriter(f,list(rows[0])); wr.writeheader();wr.writerows(rows)

def main():
    primary=[]; windows=[]; window_summary=[]; frozen=[]; frozen_seed=[]; rawdata={}
    for arm in ARMS:
        for step in STEPS:
            ff=np.load(RAW/f'{arm}_{step}_frozen.npz')
            for mode,idx in [('old',0),('input_only',1),('target_only',2),('new',3)]:
                for key in ['force','bias','radial','tangent','h','hself','hrest','common','correlation','loss','h_pos','h_neg','hs_pos','hs_neg','hr_pos','hr_neg']:
                    v=ff[key][:,idx].mean(axis=(0,2))
                    frozen.append(dict(arm=arm,step=step,mode=mode,metric=key,**ci(v)))
                    for seed,val in enumerate(v): frozen_seed.append(dict(arm=arm,step=step,mode=mode,metric=key,seed=seed,value=val))
            for key in ['input_shapley','target_shapley','factorial_interaction','change_residual','change_gate','change_residual_gate_interaction','change_input_vector','input_jump']:
                v=ff[key].mean(axis=(0,2)); frozen.append(dict(arm=arm,step=step,mode='change',metric=key,**ci(v)))
            delta=ff['force'][:,3]-ff['force'][:,0]
            verify('frozen_factor_allocation',delta,ff['input_shapley']+ff['target_shapley'])
            verify('frozen_gradient_product',delta,sum(ff[k] for k in ['change_residual','change_gate','change_residual_gate_interaction','change_input_vector']))
            for dtype in ['float64','float32']:
                d=np.load(RAW/f'{arm}_{step}_{dtype}_trajectory.npz'); rawdata[arm,step,dtype]=d
                w=d['W']; w0=d['W_start']; mu=np.concatenate([d['mu_new']]*2)
                proj=np.einsum('trhd,rd->trh',w-w0,mu)
                verify('W_endpoint_'+dtype,proj,d['cum_W_actual'])
                verify('projection_saved_'+dtype,proj,d['projection_change'])
                verify('bias_endpoint_'+dtype,d['b']-d['b_start'],d['cum_b_actual'])
                verify('radial_tangent_'+dtype,proj,d['cum_radial']+d['cum_tangent'])
                verify('source_sum_'+dtype,proj,sum(d['cum_'+k] for k in ['self_pos','self_neg','rest_pos','rest_neg','W_round']))
                r=np.linalg.norm(w,axis=-1); r0=np.linalg.norm(w0,axis=-1)
                c=np.einsum('trhd,rd->trh',w,mu)/r; c0=np.einsum('rhd,rd->rh',w0,mu)/r0
                length=(r-r0)*(c+c0)/2; direction=(c-c0)*(r+r0)/2
                verify('endpoint_length_'+dtype,length,d['length']);verify('endpoint_direction_'+dtype,direction,d['direction'])
                verify('endpoint_geometry_'+dtype,proj,length+direction)
                wc=w-w.mean(-1,keepdims=True); wc0=w0-w0.mean(-1,keepdims=True)
                normdiff=(wc*wc).sum(-1)-(wc0*wc0).sum(-1)
                verify('centered_norm_endpoint_'+dtype,normdiff,d['cum_R']-d['cum_E']+d['cum_J']+d['cum_norm_round'])
                jump=np.einsum('rhd,rd->rh',w0[:10],d['mu_new']-d['mu_old'])
                verify('boundary_jump_'+dtype,jump,d['input_jump'])
                zold=np.einsum('rhd,rd->rh',w0[:10],d['mu_old'])+d['b_start'][:10]
                znew=np.einsum('trhd,rd->trh',w[:,:10],d['mu_new'])+d['b'][:,:10]
                verify('boundary_plus_learning_'+dtype,znew-zold,jump+proj[:,:10]+d['b'][:,:10]-d['b_start'][:10])
                for it,t in enumerate(d['time']):
                    for mode,sl in [('switch',slice(0,10)),('control',slice(10,20))]:
                        vals=dict(W=proj[it,sl].mean(-1),bias=(d['b'][it,sl]-d['b_start'][sl]).mean(-1),
                            loss=d['loss'][it,sl],cos_change=(d['cosine'][it,sl]-d['cosine'][0,sl]).mean(-1),
                            length=length[it,sl].mean(-1),direction=direction[it,sl].mean(-1),
                            centered_norm_change=normdiff[it,sl].mean(-1),
                            fraction_units_down=(proj[it,sl]<0).mean(-1),
                            oldmean_W=d['oldmean_projection_change'][it,sl].mean(-1))
                        for k in ['radial','tangent','self_pos','self_neg','rest_pos','rest_neg','pos','neg','down','up','common_mu','input_correlation','centered_transport','rowmean_transport','E','R','J','W_round']:
                            vals[k]=d['cum_'+k][it,sl].mean(-1)
                        vals['z_learning']=vals['W']+vals['bias']
                        vals['input_jump']=jump.mean(-1) if mode=='switch' else np.zeros(10)
                        vals['z_own_task_change']=(vals['W']+vals['bias']+vals['input_jump']) if mode=='switch' else vals['oldmean_W']+vals['bias']
                        for seed in range(10):windows.append(dict(arm=arm,step=step,dtype=dtype,mode=mode,t=int(t),seed=seed,**{k:float(v[seed]) for k,v in vals.items()}))
                        for k,v in vals.items():window_summary.append(dict(arm=arm,step=step,dtype=dtype,mode=mode,t=int(t),metric=k,**ci(v)))
                    if t in [100,10000]:
                        v=(proj[it,:10]-proj[it,10:]).mean(-1); s=ci(v,.05/12)
                        label='SWITCH_ADDS_DOWNWARD' if s['hi']<0 else 'SWITCH_ADDS_UPWARD' if s['lo']>0 else 'UNRESOLVED'
                        primary.append(dict(arm=arm,step=step,dtype=dtype,t=int(t),label=label,switch_mean=proj[it,:10].mean(),control_mean=proj[it,10:].mean(),negative_seeds=int(np.sum(v<0)),**s))
    csvout('primary.csv',primary);csvout('window_seed.csv',windows);csvout('window_summary.csv',window_summary)
    csvout('frozen_summary.csv',frozen);csvout('frozen_seed.csv',frozen_seed)
    verification=dict(checks=VERIFICATION,all_pass=all(v['failed']==0 for v in VERIFICATION.values()))
    (OUT/'independent_verification.json').write_text(json.dumps(verification,indent=2))
    # All selected windows are shown, without choosing by outcome.
    fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True)
    for ir,arm in enumerate(ARMS):
        for ic,step in enumerate(STEPS):
            d=rawdata[arm,step,'float64']; ax=axes[ir,ic]; tt=d['time']
            for sl,label,color,style in [(slice(0,10),'Switch','#2563eb','-'),(slice(10,20),'No switch','#e07825','--')]:
                v=d['projection_change'][:,sl].mean(-1); m=v.mean(-1); se=v.std(-1,ddof=1)/np.sqrt(10)
                ax.plot(tt,m,label=label,color=color,linestyle=style,lw=2)
                ax.fill_between(tt,m-stats.t.ppf(.975,9)*se,m+stats.t.ppf(.975,9)*se,color=color,alpha=.13)
            ax.axhline(0,color='0.45',lw=.7);ax.set_xscale('symlog',linthresh=10)
            ax.set_title(f'a={0.1 if ir==0 else 0.7}; after {step//10000} tasks')
            ax.grid(alpha=.2)
            if ic==0:ax.set_ylabel('Weight-only change in mean response')
            if ir==1:ax.set_xlabel('SGD updates after branch point')
    axes[0,0].legend(frameon=False)
    fig.suptitle('Task switch vs matched continuation: fixed new-task mean input',fontsize=14)
    fig.text(.5,.012,'10 seeds; bands are descriptive 95% t intervals. Each panel fixes its own new-task mean. No bias or input-jump contribution.',ha='center',fontsize=9)
    fig.tight_layout(rect=(0,.04,1,.94));fig.savefig(OUT/'transport.png',dpi=180);fig.savefig(OUT/'transport.pdf');plt.close(fig)
    # Source levels, followed by changes; no percentages relative to small net values.
    def fs(arm,step,mode,key):return next(r for r in frozen if r['arm']==arm and r['step']==step and r['mode']==mode and r['metric']==key)['mean']
    def ws(arm,step,t,key,mode='switch'):
        return next(r for r in window_summary if r['arm']==arm and r['step']==step and r['dtype']=='float64' and r['t']==t and r['mode']==mode and r['metric']==key)['mean']
    lines=['# CondA task switch: mean-input force and actual transport','',
        'Protocol: specs/spec_switch_force_0918.md, committed 355529e before these runs. Existing checkpoints and prior post-hoc analyses informed the question. Prospective measurement protocol on previously studied models, not an unseen-data confirmation.','',
        'Two leaky arms, 3 trained checkpoints, 10 seeds, 15 exhaustive possible switches, 32 exact support patterns; paired 10000-step switch/no-switch continuations in float64 and float32. MSE, lr=.005. No original RNG replay.','',
        '## Registered paired endpoints (float64)','',
        'New-task mean held fixed for both trajectories; weight-only response change. Bonferroni two-sided t intervals across 12 comparisons (family alpha .05), n=10 seeds. Negative paired effect alone is not absolute sinking.','',
        '| a | prior tasks | updates | switched | control | difference | adjusted interval | label |','|---|---:|---:|---:|---:|---:|---|---|']
    for r in primary:
        if r['dtype']=='float64':lines.append(f"| {r['arm']} | {r['step']//10000} | {r['t']} | {r['switch_mean']:.7g} | {r['control_mean']:.7g} | {r['mean']:.7g} | [{r['lo']:.7g}, {r['hi']:.7g}] | {r['label']} |")
    lines+=['','## Frozen switches: expected one-step force, all 15 flips','',
        'Each entry averages support patterns and units/flips within seed, then 10 seeds. Projection reference is the new mean for both old and new conditions. Source decomposition is algebraic, not a removal intervention.','',
        '| arm | prior tasks | old total | new total | old self | new self | old rest | new rest | input Shapley | target Shapley |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for step in STEPS:
            selfv=lambda mode:fs(arm,step,mode,'hs_pos')+fs(arm,step,mode,'hs_neg')
            restv=lambda mode:fs(arm,step,mode,'hr_pos')+fs(arm,step,mode,'hr_neg')
            v=[fs(arm,step,'old','force'),fs(arm,step,'new','force'),selfv('old'),selfv('new'),restv('old'),restv('new'),fs(arm,step,'change','input_shapley'),fs(arm,step,'change','target_shapley')]
            lines.append(f"| {arm} | {step//10000} | "+' | '.join(f'{q:.7g}' for q in v)+' |')
    lines+=['','## Full-task switched trajectory: separate input jump, W and bias','',
        '| arm | prior tasks | input jump | W learning | b learning | total z change | radial W | tangent W | endpoint length | endpoint direction |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for step in STEPS:
            vals=[ws(arm,step,10000,k) for k in ['input_jump','W','bias','z_own_task_change','radial','tangent','length','direction']]
            lines.append(f"| {arm} | {step//10000} | "+' | '.join(f'{q:.7g}' for q in vals)+' |')
    lines+=['','## Full-task source cancellation','',
        '| arm | prior tasks | self pos | self neg | rest pos | rest neg | net W | positive-branch total | negative-branch total |','|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for arm in ARMS:
        for step in STEPS:
            vals=[ws(arm,step,10000,k) for k in ['self_pos','self_neg','rest_pos','rest_neg','W','pos','neg']]
            lines.append(f"| {arm} | {step//10000} | "+' | '.join(f'{q:.7g}' for q in vals)+' |')
    lines+=['','## Verification and limits','',f"Independent endpoint verification: {'PASS' if verification['all_pass'] else 'FAIL'}.",
        'The first preflight failed because scalar torch.where branches made diagnostic leak constants float32; production update was correct. Original failed output is retained. Corrected code uses dtype-preserving branches; no thresholds changed. Main checks are in checks_main.json.',
        'Primary inference concerns the added effect of a switch relative to no switch, at 100 and 10000 updates. Other windows, source allocations, branch breakdowns and geometry are descriptive/exploratory. The paired trajectories use one random flip per seed; all-flip robustness is tested only for frozen instantaneous forces. Results do not prove indefinite drift or transfer to CE/Adam.',
        'Tables are seed means, not universal unit behavior. A whole-population mean can be driven by subsets. Tangential projection contributions are exact stepwise ledgers, not pure finite rotations. Endpoint length/direction allocation is symmetric and exact.',
        'Hybrid input/target interventions are frozen diagnostics. Shapley allocations symmetrize two ordering choices, not a unique physical separation. The switch changes input, offset and teacher targets together. The environment-induced mean jump is not a weight update.',
        'Native float32 sensitivity results, including all registered interval labels, are retained in primary.csv. Full unrounded seed/window data and exhaustive-factor estimates are in window_seed.csv and frozen_seed.csv.','']
    (OUT/'summary.md').write_text('\n'.join(lines))
    print(json.dumps(dict(independent_pass=verification['all_pass'],primary=[r for r in primary if r['dtype']=='float64']),indent=2))
    if not verification['all_pass']:raise SystemExit(1)

if __name__=='__main__':main()
