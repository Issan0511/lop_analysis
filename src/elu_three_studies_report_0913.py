"""Cross-study audit and descriptive figures; no new hypothesis tests."""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/elu_three_studies_0913'
def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def checkclose(x,y):assert abs(float(x)-float(y))<1e-10,(x,y)
def tci(x):
    x=np.asarray(x);h=4.30265273*x.std(ddof=1)/math.sqrt(3)
    return float(x.mean()),float(h)
def main():
    OUT.mkdir(parents=True,exist_ok=True)
    ad=ROOT/'results/elu_reserve_0913';bd=ROOT/'results/elu_depth_width_0913';cd=ROOT/'results/elu_environment_0913'
    # Recalculate A from individual recorded curve points.
    auc={};acount=0
    for p in sorted(ad.glob('*_curve.csv')):
        rr=read(p);assert len(rr)==50,p
        first=rr[0];k=(first['arm'],int(first['seed']),int(first['checkpoint']),first['branch'])
        tasks=sorted(set(int(r['task']) for r in rr))
        assert tasks==list(range(k[2]+1,k[2]+11))
        areas=[]
        for task in tasks:
            r=sorted([x for x in rr if int(x['task'])==task],key=lambda x:int(x['step']))
            xx=[int(x['step']) for x in r];assert xx==[0,20,100,300,625]
            areas.append(float(np.trapezoid([float(x['ce']) for x in r],xx)/625))
        auc[k]=float(np.mean(areas));acount+=len(rr)
    assert len(auc)==72 and acount==3600
    arows=read(ad/'contrasts.csv')
    for r in arows:
        a,s,t=r['arm'],int(r['seed']),int(r['checkpoint'])
        checkclose(r['target_minus_matched'],auc[a,s,t,'TARGET']-auc[a,s,t,'MATCHED'])
        checkclose(r['shallow_minus_matched'],auc[a,s,t,'SHALLOW']-auc[a,s,t,'SHALLOW_MATCHED'])
    aprovs=[json.loads(p.read_text()) for p in ad.glob('*_provenance.json')]
    assert len(aprovs)==6
    assert all(all(v==0 for q in p['exact_anchor_maxabs'].values() for v in q.values()) for p in aprovs)
    assert all(p['code_sha']==digest(ROOT/'src/elu_reserve_0913.py') for p in aprovs)
    # Recalculate B late levels and registered width/depth/activation contrasts.
    bmeans={};bcount=0
    for p in sorted((bd/'raw').glob('*_rows.csv')):
        rr=read(p);bcount+=len(rr)
        for cell in ['ref','n5_d1','n10_d1','n5_d4','n10_d4']:
            e=[r for r in rr if r['cell']==cell and int(r['step'])==625]
            assert sorted(int(r['task']) for r in e)==list(range(21,121)),(p,cell,len(e))
            late=[float(r['test_acc']) for r in e if int(r['task'])>=101]
            first=e[0];bmeans[first['activation'],int(first['seed']),cell]=float(np.mean(late))
    assert len(bmeans)==30
    bver=read(bd/'verdict.csv');braw={}
    for d in ['d1','d4']:
        for a in ['ELU1','LR']:
            braw[a,d]=[bmeans[a,s,'n5_'+d]-bmeans[a,s,'n10_'+d] for s in range(3)]
        vals=np.array(braw['ELU1',d])-np.array(braw['LR',d])
        reported=next(r for r in bver if r['metric']=='late_test_acc' and r['contrast']=='H_ELU_minus_LR_'+d)
        assert np.allclose(vals,json.loads(reported['seed_values']),rtol=0,atol=1e-10)
    bprovs=[json.loads(p.read_text()) for p in (bd/'raw').glob('*_provenance.json')]
    assert len(bprovs)==6 and all(p['code_sha256']==digest(ROOT/'src/elu_depth_width_0913.py') for p in bprovs)
    assert all(p['checks']['anchor']['maxabs']==0 for p in bprovs)
    # Recalculate C from per-task online accuracy, treating seed as replicate.
    cr=read(cd/'rows.csv');assert len(cr)==1200
    lop={};clevel={}
    for s in range(3):
        for env in ['PM','RL']:
            for a in ['ELU1','LR']:
                for iv in ['ref','wclamp']:
                    rr=[r for r in cr if int(r['seed'])==s and r['env']==env and r['act']==a and r['iv']==iv]
                    assert sorted(int(r['task']) for r in rr)==list(range(1,51))
                    early=np.mean([float(r['online_acc']) for r in rr if 11<=int(r['task'])<=20])
                    late=np.mean([float(r['online_acc']) for r in rr if 41<=int(r['task'])<=50])
                    lop[s,env,a,iv]=100*(early-late);clevel[s,env,a,iv]=[float(early),float(late)]
    cdiff={(env,a):[lop[s,env,a,'ref']-lop[s,env,a,'wclamp'] for s in range(3)] for env in ['PM','RL'] for a in ['ELU1','LR']}
    cv=read(cd/'verdict.csv')
    for (env,a),vals in cdiff.items():
        rr=next(r for r in cv if r['contrast']=='D_'+env+'_'+a)
        assert np.allclose(vals,[float(rr['seed'+str(s)]) for s in range(3)],rtol=0,atol=1e-10)
    cp=json.loads((cd/'provenance.json').read_text())
    assert cp['code_sha256']==digest(ROOT/'src/elu_environment_0913.py')
    assert cp['checks']['warmup_pair_maxabs']==0 and cp['checks']['clamp_norm_rel_max']<2e-5
    import torch
    cck=torch.load(cd/'checkpoint.pt',map_location='cpu',weights_only=False)
    assert float(cck['t'])==300000
    audit=dict(status='PASS',A=dict(branches=72,curve_rows=acount,original_anchors_exact=True,auc_recomputed=True),
               B=dict(cells=30,curve_rows=bcount,original_anchors_exact=True,accuracy_contrasts_recomputed=True),
               C=dict(trajectories=24,task_rows=1200,updates_each=300000,warmup_exact=True,degradation_contrasts_recomputed=True),
               all_running_code_hashes_match=True,no_new_hypothesis_tests=True,
               source_sha256={str(p.relative_to(ROOT)):digest(p) for p in [ad/'verdict.csv',bd/'verdict.csv',cd/'verdict.csv']})
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2))
    # Descriptive plot; uncertainty bars are the registered across-seed t interval.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.facecolor':'white'})
    fig,axs=plt.subplots(1,3,figsize=(15,4.7),layout='constrained')
    colors={'ELU1':'#d66535','LR':'#3273a8'}
    ax=axs[0]
    for ai,a in enumerate(['ELU1','LR']):
        for j,(label,field) in enumerate([('Shallow + small W','target_minus_matched'),('Shallow 25','shallow_minus_matched')]):
            vals=[float(r[field]) for r in arows if r['arm']==a and int(r['checkpoint'])==100]
            xx=j+(ai-.5)*.28
            ax.scatter(np.full(3,xx)+np.linspace(-.035,.035,3),vals,color=colors[a],s=28,alpha=.8,label=a if j==0 else None)
            ax.plot([xx-.075,xx+.075],[np.mean(vals)]*2,color=colors[a],lw=2)
    ax.axhline(0,color='#aaaaaa',lw=1);ax.set_xticks([0,1],['Shallow + small W','Shallow 25'])
    ax.set_title('A  Prospective update freezing');ax.set_ylabel('Excess CE area vs matched control')
    ax.legend(frameon=False);ax.text(.02,.98,'Checkpoint 100; 10 new tasks\nDots: 3 seeds',va='top',transform=ax.transAxes,fontsize=8)
    ax=axs[1]
    for a in ['ELU1','LR']:
        for d,style in [('d1','-'),('d4','--')]:
            means=[];err=[]
            for n in ['n5','n10']:
                vals=[bmeans[a,s,n+'_'+d] for s in range(3)]
                mm,hh=tci(vals);means.append(mm);err.append(hh)
            ax.errorbar([5,10],means,yerr=err,color=colors[a],linestyle=style,marker='o',capsize=3,label=a+' mean '+('-1' if d=='d1' else '-4'))
    ax.set_xticks([5,10]);ax.set_xlabel('Controlled centered W norm')
    ax.set_ylabel('Late accuracy (%)');ax.set_title('B  Width x mean depth')
    ax.legend(frameon=False,fontsize=8);ax.text(.02,.02,'Tasks 101-120; bars: t95% across 3 seeds',transform=ax.transAxes,fontsize=8)
    ax=axs[2]
    for ai,a in enumerate(['ELU1','LR']):
        for j,env in enumerate(['PM','RL']):
            vals=cdiff[env,a];mm,hh=tci(vals);xx=j+(ai-.5)*.25
            ax.errorbar(xx,mm,yerr=hh,color=colors[a],fmt='o',capsize=3,label=a if j==0 else None)
            ax.scatter(np.full(3,xx)+np.linspace(-.035,.035,3),vals,color=colors[a],s=15,alpha=.65)
    ax.axhline(0,color='#aaaaaa',lw=1);ax.set_xticks([0,1],['Permuted','Random label'])
    ax.set_ylabel('Degradation removed by W clamp (pp)');ax.set_title('C  Matched environment comparison')
    ax.legend(frameon=False);ax.text(.02,.98,'Same 1200 images; 6000 updates/task\n50 tasks; 3 seeds',va='top',transform=ax.transAxes,fontsize=8)
    fig.savefig(OUT/'three_studies.png',dpi=160)
    print(json.dumps(audit,indent=2))
    print('A t100:',json.dumps([r for r in arows if int(r['checkpoint'])==100]))
    print('B registered:',json.dumps([r for r in bver if r['metric']=='late_test_acc']))
    print('C registered:',json.dumps(cv))
if __name__=='__main__':main()
