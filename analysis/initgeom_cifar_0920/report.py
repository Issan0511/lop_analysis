"""Registered A5 complete-only L1 and conditional-L2 reporting."""
import argparse,csv,json,math
from pathlib import Path
import numpy as np
from analysis.initgeom_cifar_0920 import run as R,stats as S


def validate(out,production=True):
    out=Path(out);marker=json.loads((out/'complete.json').read_text());ident=json.loads((out/'input_manifest.json').read_text())
    assert marker['identity']==ident and ident['run_id']==R.RUN and ident['source_sha256']==R.sources()
    seeds=list(range(20,40)) if production else [100,101]
    assert ident['seeds']==seeds and marker['seeds']==seeds and ident['mode']==('production' if production else 'check'),'INCOMPLETE or wrong role'
    assert ident['conditions']==list(S.CONDITIONS) and ident['alias']=={'gamma100':'raw'} and ident['layers']==[1,2]
    assert ident['images']==1200 and ident['units']==100 and ident['device']=='cpu'
    assert R.sha(out/'state.json')==marker['state_sha256']
    state=json.loads((out/'state.json').read_text());assert state['seeds']==seeds and state['identity']==ident
    assert set(marker['seed_markers'])=={str(s) for s in seeds}
    records=[]
    for seed in seeds:
        p=out/f'seed{seed}/complete.json';assert R.sha(p)==marker['seed_markers'][str(seed)]
        rec=json.loads(p.read_text());assert rec['seed']==seed and rec['identity']==ident
        expected={f'seed{seed}/initial.pt',f'seed{seed}/prediction_manifest.json'}
        for c in S.CONDITIONS:expected|={f'seed{seed}/{c}_{s}' for s in ('layer1.json','layer2.json','layer2_prediction.json','raw.npz')}
        assert {f['path'] for f in rec['files']}==expected,'missing/foreign condition artifact'
        R.verify_files(out,rec)
        wanted={f'seed{seed}/{c}_layer{l}.json' for c in S.CONDITIONS for l in (1,2)}
        assert set(rec['records'])==wanted and len(rec['records'])==16
        pred=json.loads((out/f'seed{seed}/prediction_manifest.json').read_text());assert pred['conditions']==list(S.CONDITIONS) and pred['seed']==seed and pred['source_sha256']==ident['source_sha256']
        for c in S.CONDITIONS:
            for l in (1,2):
                item=json.loads((out/f'seed{seed}/{c}_layer{l}.json').read_text())
                assert (item['seed'],item['condition'],item['layer'])==(seed,c,l)
                assert item['source_sha256']==ident['source_sha256'] and item['P_sha256']==rec['P_sha256']
                for branch in ('main','bias0'):
                    d=item[branch];assert d['images']==1200 and d['units']==100 and d['threshold']==1189
                    assert 0<=d['f_lower']<=d['f']<=d['f_upper']<=1
                    for key in ('positive','negative','zero','zmean','zsd'):
                        assert len(d[key])==100 and all(math.isfinite(x) for x in d[key])
                    assert all(p+n+z==1200 for p,n,z in zip(d['positive'],d['negative'],d['zero']))
                if l==1:assert item['main']['input_stats']==pred['statistics'][c]
                else:
                    lp=json.loads((out/f'seed{seed}/{c}_layer2_prediction.json').read_text());assert item['main']['input_stats']==lp['statistics'] and lp['P_sha256']==rec['P_sha256']
                records.append(item)
    assert len(records)==len(seeds)*8*2
    return records


def csv_write(p,rows):
    with Path(p).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def evaluate(records):
    assert len(records)==20*8*2
    keys=[(r['seed'],r['condition'],r['layer']) for r in records]
    assert len(set(keys))==len(keys) and set(keys)=={(s,c,l) for s in range(20,40) for c in S.CONDITIONS for l in (1,2)}
    decisions=[];layerlabels={}
    for layer in (1,2):
        labels={}
        for c in S.CONDITIONS:
            rows=sorted([r for r in records if r['layer']==layer and r['condition']==c],key=lambda r:r['seed'])
            ds=[r['main'] for r in rows];gs=[d['input_stats'] for d in ds]
            low=float(np.median([d['f_lower'] for d in ds]));high=float(np.median([d['f_upper'] for d in ds]));obs=float(np.median([d['f'] for d in ds]))
            if any(g['status']=='UNDEFINED_INPUT' for g in gs):band=None;label='UNDEFINED'
            else:
                ps=[g['p'] for g in gs];band=S.median_band(ps);label=S.condition_label(low,high,band)
                # Numerical p uncertainty must not change the discrete-band decision.
                alt=[S.median_band([g[k] for g in gs]) for k in ('p_numeric_lower','p_numeric_upper')]
                if any(S.condition_label(low,high,b)!=label for b in alt):label='NUMERIC_UNRESOLVED'
            labels[c]=label;decisions.append(dict(layer=layer,condition=c,label=label,observed_median=obs,observed_low=low,observed_high=high,band=band))
        layerlabels[str(layer)]=S.family(labels,layer)
    dial=[]
    for layer in (1,2):
        by={c:sorted([r['main'] for r in records if r['condition']==c and r['layer']==layer],key=lambda d:d['f']) for c in S.DIAL}
        intervals=[(float(np.median([d['f_lower'] for d in by[c]])),float(np.median([d['f_upper'] for d in by[c]]))) for c in S.DIAL]
        comparisons=[]
        lookup={(r['seed'],r['condition']):r['main']['f'] for r in records if r['layer']==layer}
        for a,b in zip(S.DIAL[:-1],S.DIAL[1:]):
            delta=[lookup[s,b]-lookup[s,a] for s in range(20,40)]
            comparisons.append(dict(left=a,right=b,paired_seed_differences=delta,interval=S.paired_interval(delta)))
        dial.append(dict(layer=layer,label=S.monotone(intervals),median_intervals=intervals,adjacent=comparisons,role='registered secondary' if layer==1 else 'REPORT_ONLY'))
    return dict(families=layerlabels,conditions=decisions,dial=dial,alias={'gamma100':'raw'},interpretation='compatibility under an approximate binomial model, not proof/equivalence; L2 conditional on measured a1; two separate families')


def report(src,out):
    records=validate(src);verdict=evaluate(records);out=Path(out);out.mkdir(parents=True,exist_ok=True)
    R.put(out/'verdict.json',verdict)
    flat=[];units=[]
    for r in records:
        for kind in ('main','bias0'):
            d=r[kind];g=d['input_stats'];row=dict(seed=r['seed'],condition=r['condition'],layer=r['layer'],variant=kind,f=d['f'],f_lower=d['f_lower'],f_upper=d['f_upper'],r=g['r'],p=g['p'],p_bias=g['p_bias'],mu_norm=g['mu_norm'],trSigma=g['trSigma'],effective_rank=g['effective_rank'],gaussian_proxy=d['gaussian_proxy_fraction'],positive_units=d['positive_units'],negative_units=d['negative_units'],all_zero_units=d['all_zero_units'],uncertain_values=d['uncertain_values'])
            flat.append(row)
            for i in range(100):
                unit=dict(seed=r['seed'],condition=r['condition'],layer=r['layer'],variant=kind,unit=i)
                for k in ('positive','negative','zero','positive_sure','negative_sure','positive_possible','negative_possible','one_sided','guaranteed','possible','zmean','zsd','projection_mean','projection_sd','bias_over_sd','gaussian_proxy'):unit[k]=d[k][i]
                units.append(unit)
    csv_write(out/'per_seed.csv',flat);csv_write(out/'per_unit.csv',units)
    display=[dict(r,alias_of=None) for r in flat]
    display += [dict(r,condition='gamma100',alias_of='raw') for r in flat if r['condition']=='raw']
    csv_write(out/'display_rows.csv',display)
    cond={(r['layer'],r['condition']):r['label'] for r in verdict['conditions']}
    claims=[('raw_L1',.75,cond[1,'raw']=='PREDICTED'),('std_C_L1',.90,all(cond[1,c]=='PREDICTED' for c in ('std','C'))),('L1_all',.55,verdict['families']['1']=='L1_ALL_COMPATIBLE'),('dial_monotone',.90,verdict['dial'][0]['label']=='MONOTONE_SAMPLE_MEDIANS'),('L2_conditional_all',.35,verdict['families']['2']=='L2_CONDITIONAL_ALL_COMPATIBLE')]
    unresolved=any(r['label'] in ('UNDEFINED','NUMERIC_UNRESOLVED','DIVERGED') for r in verdict['conditions'])
    R.put(out/'predictions.json',dict(author='Codex',Issa=None,claims=[dict(name=n,p=p,outcome=v,brier=None if unresolved else (p-int(v))**2,unscored_reason='some numerical/undefined outcomes unresolved' if unresolved else None) for n,p,v in claims]))
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4),layout='constrained')
    for layer,ax in zip((1,2),axes):
        rows=[r for r in verdict['conditions'] if r['layer']==layer]
        for j,r in enumerate(rows):
            if r['band']:ax.plot([j,j],[r['band']['low'],r['band']['high']],lw=6,color='lightgray')
            ax.plot(j,r['observed_median'],'o',color='black')
        ax.set_xticks(range(8),S.CONDITIONS,rotation=35,ha='right');ax.set_ylabel('Median one-sided fraction');ax.set_title('Layer 1' if layer==1 else 'Layer 2: conditional on a1')
    fig.savefig(out/'bands.png',dpi=160);fig.savefig(out/'bands.pdf');plt.close(fig)
    (out/'summary.md').write_text('# A5\n\n'+verdict['families']['1']+'\n\n'+verdict['families']['2']+'\n\n'+verdict['interpretation']+'\n')
    return verdict
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--src',required=True);p.add_argument('--out',required=True);a=p.parse_args();print(report(a.src,a.out))
