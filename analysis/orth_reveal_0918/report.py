"""Independent NumPy endpoints; registered pooled-seed tests and descriptive supplements."""
from pathlib import Path
import csv,json
import numpy as np
from scipy.stats import t as student
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2]
RAW=Path('/home/issan/Projects/obsidian-research-data/orth_reveal_0918')
OUT=ROOT/'results/orth_reveal_0918'
PARENT=RAW.parent/'switch_force_0918'
ARMS=['LR_a0p1_q0','LR_a0p7_q0'];AGES=[200000,1000000,5000000]
CHECKS={}
def check(name,a,b):
    a,b=np.broadcast_arrays(a,b);err=np.abs(a-b);bound=1e-10*(1+np.abs(b))
    row=CHECKS.setdefault(name,dict(count=0,failed=0,max_abs=0.))
    row['count']+=int(err.size);row['failed']+=int(np.sum(~np.isfinite(err)|(err>bound)))
    row['max_abs']=max(row['max_abs'],float(np.max(err)))
def stats(x,primary=False):
    x=np.asarray(x,dtype=float)
    if not np.isfinite(x).all():return dict(mean=None,lo=None,hi=None,label='NOT_ESTIMABLE',n=int(np.isfinite(x).sum()))
    m=float(x.mean());h=float(student.ppf(1-.05/(36 if primary else 2),9)*x.std(ddof=1)/np.sqrt(10))
    return dict(mean=m,lo=m-h,hi=m+h,label='POSITIVE' if m-h>0 else 'NEGATIVE' if m+h<0 else 'UNRESOLVED',n=10)
def ratio(a,b):return np.divide(a,b,out=np.full(np.broadcast_shapes(np.shape(a),np.shape(b)),np.nan),where=b>0)
def energy(q):return np.sum(q*q,axis=(0,2)) # patterns,seeds,units

def measure_C(q,ds,mask):
    return ratio(-np.sum(q*ds*mask,axis=(0,2)),np.sum(q*q*mask,axis=(0,2)))
def write(name,rows):
    if not rows:return
    with (OUT/name).open('w') as f:
        w=csv.DictWriter(f,list(rows[0]));w.writeheader();w.writerows(rows)
def get_prim(d):
    q=d['q_B'];u=d['u_whole'];du=d['W'][-1]-d['W_Bstart'];xb=np.concatenate([d['x_B']]*2,1)
    ds=np.einsum('rhd,srd->srh',du,xb)
    cp=[];cm=[];kk=[]
    for k in [0,10]:
        cp.append(measure_C(q,ds[:,k:k+10],q>0));cm.append(measure_C(q,ds[:,k:k+10],q<0))
        kk.append(ratio(-np.sum(du[k:k+10]*u,axis=(1,2)),np.sum(u*u,axis=(1,2))))
    return dict(S=(cp[0]-cm[0])-(cp[1]-cm[1]),Kdiff=kk[0]-kk[1],Cplus_B=cp[0],Cminus_B=cm[0],Cplus_control=cp[1],Cminus_control=cm[1],K_B=kk[0],K_control=kk[1])

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    primary=[];secondary=[];seeds=[];curves=[];precision=[];nullrows=[]
    def add(table,arm,age,coord,metric,values,prim=False):
        table.append(dict(arm=arm,age=age,coordinate=coord,metric=metric,**stats(values,prim)))
        for seed,x in enumerate(values):seeds.append(dict(arm=arm,age=age,coordinate=coord,metric=metric,seed=seed,value=float(x)))
    for arm in ARMS:
      for age in AGES:
        g=np.load(RAW/f'{arm}_{age}_geometry.npz');d=np.load(RAW/f'{arm}_{age}_float64_B.npz');f=np.load(RAW/f'{arm}_{age}_float32_B.npz')
        w0=g['w0'];w1=g['w1'];xa=g['x_A'];a=.1 if '0p1' in arm else .7
        for coord in ['whole','centered']:
            wstart=w0 if coord=='whole' else w0-w0.mean(-1,keepdims=True)
            wend=w1 if coord=='whole' else w1-w1.mean(-1,keepdims=True)
            D=wend-wstart;alpha=np.sum(wstart*D,-1)/np.sum(wstart*wstart,-1)
            p=alpha[...,None]*wstart;u=D-p;key=lambda k:coord+'_'+k
            for k,val in [('delta',D),('alpha',alpha),('p',p),('u',u)]:check('geometry_'+k,val,g[key(k)])
            qa=np.einsum('rhd,srd->srh',u,xa);qb=np.einsum('rhd,fsrd->fsrh',u,g[key('B_x')])
            check('q_A',qa,g[key('q_A')]);check('q_B',qb,g[key('B_q')])
            ea=energy(qa);eb=np.sum(qb*qb,axis=(0,1,3))/15
            G=.5*np.log(ratio(eb,ea))
            if coord=='whole':add(primary,arm,age,coord,'G',G,True)
            add(secondary,arm,age,coord,'RMS_B_over_A',np.sqrt(ratio(eb,ea)))
            r0=np.sum(wstart*wstart,axis=(1,2));uu=np.sum(u*u,axis=(1,2))
            add(secondary,arm,age,coord,'A_alpha_norm_weighted',np.sum(wstart*D,axis=(1,2))/r0)
            add(secondary,arm,age,coord,'A_fraction_alpha_negative',(alpha<0).mean(1))
            add(secondary,arm,age,coord,'A_radial_norm_budget',np.sum((2*alpha+alpha*alpha)*np.sum(wstart*wstart,-1),1))
            add(secondary,arm,age,coord,'A_u_norm2',uu)
            add(secondary,arm,age,coord,'A_norm2_change',np.sum(wend*wend-wstart*wstart,axis=(1,2)))
            add(secondary,arm,age,coord,'A_D_norm2',np.sum(D*D,axis=(1,2)))
            add(secondary,arm,age,coord,'A_normalized_visibility',ea/32/uu)
            add(secondary,arm,age,coord,'A_q_RMS',np.sqrt(ea/3200))
            add(secondary,arm,age,coord,'B_q_RMS',np.sqrt(eb/3200))
            add(secondary,arm,age,coord,'q_mean_A',qa.mean((0,2)))
            add(secondary,arm,age,coord,'q_mean_B',qb.mean((0,1,3)))
            add(secondary,arm,age,coord,'q_mean_jump',qb.mean((0,1,3))-qa.mean((0,2)))
            add(secondary,arm,age,coord,'B_q_positive_energy_fraction',np.sum(qb*qb*(qb>0),axis=(0,1,3))/15/eb)
            # Fresh SVD, independent projector and cancellation audit.
            xx=xa if coord=='whole' else xa-xa.mean(-1,keepdims=True)
            P=[];ranks=[]
            for seed in range(10):
                _,s,v=np.linalg.svd(xx[:,seed],full_matrices=False);r=int(np.sum(s>1e-10*s[0]));P.append(v[:r].T@v[:r]);ranks.append(r)
            P=np.stack(P);ns=lambda v:v-np.einsum('rhi,rij->rhj',v,P)
            nd,np_,nu=ns(D),ns(p),ns(u)
            check('null_sum',nd,np_+nu);check('saved_null_delta',nd,g[key('delta_null')]);check('saved_null_u',nu,g[key('u_null')])
            add(secondary,arm,age,coord,'u_null_energy_fraction',np.sum(nu*nu,axis=(1,2))/uu)
            add(secondary,arm,age,coord,'D_null_relative_norm',np.sqrt(np.sum(nd*nd,axis=(1,2))/np.sum(D*D,axis=(1,2))))
            for seed in range(10):nullrows.append(dict(arm=arm,age=age,coordinate=coord,dtype='float64',seed=seed,rank=ranks[seed],D_null_rel_norm=float(np.linalg.norm(nd[seed])/np.linalg.norm(D[seed])),u_null_energy_fraction=float(np.sum(nu[seed]**2)/uu[seed])))
            # Native actual null drift is not zeroed out.
            pa=np.load(PARENT/f'{arm}_{age}_float32_trajectory.npz');df=pa['W'][-1,:10]-pa['W_start'][:10]
            if coord=='centered':df-=df.mean(-1,keepdims=True)
            # Float32 supports differ slightly; use their own stored A support.
            xf=f['x_A'];xf=xf if coord=='whole' else xf-xf.mean(-1,keepdims=True)
            for seed in range(10):
                _,s,v=np.linalg.svd(xf[:,seed],full_matrices=False);rank=int(np.sum(s>1e-10*s[0]));proj=v[:rank].T@v[:rank]
                nf=df[seed]-df[seed]@proj
                nullrows.append(dict(arm=arm,age=age,coordinate=coord,dtype='float32',seed=seed,rank=rank,D_null_rel_norm=float(np.linalg.norm(nf)/np.linalg.norm(df[seed])),u_null_energy_fraction=None))
            # Functional ablation, coordinate component removed at fixed remaining model.
            za=g['z_A'];zb=g[key('B_z')]
            phi=lambda z:np.where(z>0,z,a*z)
            aa=phi(za)-phi(za-qa);ab=phi(zb)-phi(zb-qb)
            check('functional_A',aa,g[key('act_A')]);check('functional_B',ab,g[key('B_act')])
            add(secondary,arm,age,coord,'functional_RMS_B_over_A',np.sqrt(ratio(np.sum(ab*ab,axis=(0,1,3))/15,energy(aa))))
            v=g['v1'];vaa=aa*v;vab=ab*v
            add(secondary,arm,age,coord,'unit_output_contrib_RMS_B_over_A',np.sqrt(ratio(np.sum(vab*vab,axis=(0,1,3))/15,energy(vaa))))
            # Exact total output ablation sums units before squaring.
            add(secondary,arm,age,coord,'network_output_ablation_RMS_B_over_A',np.sqrt(ratio(np.sum(vab.sum(-1)**2,axis=(0,1))/15,np.sum(vaa.sum(-1)**2,axis=0))))
            for field in ['delta','p']:
                af=g[key(field+'_A')];bf=g[key('B_'+field)]
                add(secondary,arm,age,coord,field+'_RMS_B_over_A',np.sqrt(ratio(np.sum(bf*bf,axis=(0,1,3))/15,energy(af))))
                add(secondary,arm,age,coord,field+'_mean_jump',bf.mean((0,1,3))-af.mean((0,2)))
            add(secondary,arm,age,coord,'A_p_u_cross_energy',2*np.sum(g[key('p_A')]*qa,axis=(0,2))/32)
            add(secondary,arm,age,coord,'B_p_u_cross_energy',2*np.sum(g[key('B_p')]*qb,axis=(0,1,3))/15/32)
        for dtype,data in [('float64',d),('float32',f)]:
            res=get_prim(data)
            for metric in ['S','Kdiff']:
                if dtype=='float64':add(primary,arm,age,'whole',metric,res[metric],True)
                precision.append(dict(arm=arm,age=age,dtype=dtype,metric=metric,**stats(res[metric],True)))
            for metric in ['Cplus_B','Cminus_B','Cplus_control','Cminus_control','K_B','K_control']:
                if dtype=='float64':add(secondary,arm,age,'whole',metric,res[metric])
            # G native: full 15 flips geometry reconstructed from own saved A supports.
            if dtype=='float32':
                uf=data['u_whole'];cpw=np.load(PARENT/f'{arm}_{age}_float32_trajectory.npz')
                # Independent x reconstruction from flip bits with original constant-dose formula.
                fa=g['flip_A'];bits=np.array([[int(c) for c in f'{i:05b}'] for i in range(32)])
                qa=np.einsum('rhd,srd->srh',uf,data['x_A']);eas=energy(qa);ebs=np.zeros(10)
                for flipid in range(15):
                    fb=fa.astype('float64');fb[:,flipid]=1-fb[:,flipid];k=fb.sum(-1);target=3.041
                    offset=.05*(k+2.5-np.sqrt((k+2.5)**2-20*(k+1.25-target**2)))
                    raw=np.concatenate([np.broadcast_to(fb,(32,10,15)),np.broadcast_to(bits[:,None],(32,10,5))],-1).astype('float32')
                    xb=(raw-offset.astype('float32')[None,:,None]).astype('float32').astype('float64')
                    
                    selected=data['B_flip_index']==flipid
                    if selected.any():check('native_B_support',xb[:,selected],data['x_B'][:,selected])
                    qb=np.einsum('rhd,srd->srh',uf,xb);ebs+=energy(qb)/15
                precision.append(dict(arm=arm,age=age,dtype=dtype,metric='G',**stats(.5*np.log(ratio(ebs,eas)),True)))
            for coord in ['whole','centered']:
                u=data['u_'+coord];q=data['q_B' if coord=='whole' else 'q_B_centered'];den=np.sum(u*u,axis=(1,2));z=data['z_Bstart']
                masks={'qpos':q>0,'qneg':q<0,'qpos_zpos':(q>0)&(z>0),'qpos_zneg':(q>0)&(z<=0),'qneg_zpos':(q<0)&(z>0),'qneg_zneg':(q<0)&(z<=0),'newly_activated':(z>0)&(z-q<=0)}
                for it,t in enumerate(data['time']):
                    dw=data['W'][it]-data['W_Bstart'];dx=np.einsum('rhd,srd->srh',dw,np.concatenate([data['x_B']]*2,1))
                    check('receiver_field_'+dtype,dx,data['response_delta'][it]);check('receiver_bias_'+dtype,dx+(data['b'][it]-data['b_Bstart'])[None],data['response_delta_with_bias'][it])
                    inner=np.sum(dw*np.concatenate([u]*2,0),-1)
                    check('inner_'+dtype,inner,data[coord+'_inner'][it]);check('source_sum_'+dtype,inner,data[coord+'_source_positive'][it]+data[coord+'_source_negative'][it])
                    if dtype=='float32':continue
                    for branch,k in [('B',0),('control',10)]:
                        dd=dx[:,k:k+10];ddbias=data['response_delta_with_bias'][it,:,k:k+10]
                        measures={'K':-inner[k:k+10].sum(-1)/den,
                           'K_source_positive':-data[coord+'_source_positive'][it,k:k+10].sum(-1)/den,
                           'K_source_negative':-data[coord+'_source_negative'][it,k:k+10].sum(-1)/den,
                           'mean_B_delta':(data['mean_B'][it,k:k+10]-data['mean_B'][0,k:k+10]).mean(-1)}
                        for name,mask in masks.items():
                            measures['C_'+name]=measure_C(q,dd,mask);measures['Cbias_'+name]=measure_C(q,ddbias,mask)
                            if t==10000:
                                add(secondary,arm,age,coord,branch+'_'+name+'_energy_fraction',np.sum(q*q*mask,axis=(0,2))/energy(q))
                                add(secondary,arm,age,coord,branch+'_'+name+'_count_fraction',mask.mean((0,2)))
                        for name,val in measures.items():
                            curves.append(dict(arm=arm,age=age,coordinate=coord,branch=branch,time=int(t),metric=name,**stats(val)))
                            if t==10000:add(secondary,arm,age,coord,branch+'_'+name,val)
        print(arm,age,'report complete',flush=True)
    write('primary.csv',primary);write('secondary.csv',secondary);write('seed_endpoints.csv',seeds);write('curves.csv',curves);write('precision.csv',precision);write('nullspace_audit.csv',nullrows)
    (OUT/'report_checks.json').write_text(json.dumps(dict(checks=CHECKS,all_pass=all(v['failed']==0 for v in CHECKS.values())),indent=2))
    # All six cells and all three registered outcomes, no selection.
    fig,axs=plt.subplots(1,3,figsize=(13.4,5.1),layout='constrained')
    colors=['#167d9a','#b76b2a'];labels=[]
    for ax,metric,title in zip(axs,['G','S','Kdiff'],['Exposure: log(RMS B / RMS A)','Positive-side counteraction: S','Tagged direction loss: Kdiff']):
        for j,row in enumerate([r for r in primary if r['metric']==metric]):
            m,lo,hi=[row[k] for k in ['mean','lo','hi']]
            ax.errorbar(m,5-j,xerr=[[m-lo],[hi-m]],fmt='o',color=colors[j//3],capsize=3)
            if metric=='G':labels.append(f"a={'.1' if j<3 else '.7'}, age {row['age']//10000} tasks")
        ax.axvline(0,color='#555555',lw=1);ax.set_title(title,fontsize=11);ax.grid(axis='x',alpha=.2);ax.set_yticks(range(6));ax.set_yticklabels(labels[::-1] if metric=='G' else ['']*6);ax.set_xlabel('Mean and family-adjusted 95% interval')
    fig.suptitle('A orthogonal increment → B exposure → selective counteraction\nCondA SGD, lr=.005, 10 seeds; 18-comparison Bonferroni intervals',fontsize=13)
    fig.savefig(OUT/'primary.png',dpi=180);plt.close(fig)
    if any(v['failed'] for v in CHECKS.values()):raise SystemExit('independent report checks failed')
    print(json.dumps(primary),flush=True)
if __name__=='__main__':main()
