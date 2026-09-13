"""sigma_decomp_posthoc_0910 — 事後解析（事前登録なし）。
前活性の分散を W の二つの成分に厳密分解する:
  σ_inv²_i = κ‖W̃_i‖² + Var(S)·m_i² + cross,   κ = v̄ − c̄（画素分散平均 − 画素共分散平均）, S = 総インク
置換の期待値では ΣW̃=0 により cross と v̄/c̄ 以外の項が消える。データ定数のみで自由パラメータ無し。
入力: boundary_groups_0908 の t100 checkpoint (LR/SNA)、elu_growth_0909 / linear_growth_0910 の units.npz。
変異対照: (a) 参照置換を恒等 1 本にする（cross が消えない）、(b) 単位間で z̄_i を入れ替える（オフセット回帰が壊れる）。
"""
import sys,json,numpy as np,torch,pandas as pd
from pathlib import Path
sys.path.insert(0,'src'); import pmnist_boundary_host_0908 as H
H.DATA_DIR=Path('/home/issan/Projects/claude/proj_004_drift/data/mnist')
OUT=Path('results/sigma_decomp_posthoc_0910'); OUT.mkdir(exist_ok=True)
torch.set_num_threads(4); H.setup('cpu'); mn=H.Mnist(torch.device('cpu')); x=mn.test_x.double()
C=torch.cov(x.T,correction=0); v=C.diag().mean().item(); c=((C.sum()-C.diag().sum())/(784*783)).item()
S=x.sum(1); ES=S.mean().item(); VS=S.var(unbiased=False).item(); kap=v-c; gam=VS/ES**2
const=dict(vbar=v,cbar=c,kappa=kap,ES=ES,VarS=VS,sdS=VS**.5,gamma=gam,mnist_sha=mn.sha256)
print(f"data: vbar={v:.5f} cbar={c:.5f} kappa={kap:.5f} E[S]={ES:.2f} sd(S)={VS**.5:.2f} gamma=Var(S)/E[S]^2={gam:.4f}")
g=torch.Generator().manual_seed(20260909); refs=[torch.randperm(784,generator=g) for _ in range(8)]
def decomp(W,b,refs):
  m=W.mean(1,keepdim=True);Wt=W-m
  zr=torch.stack([x[:,r]@W.T+b for r in refs]); var_i=zr.var(1,unbiased=False).mean(0)
  width=torch.stack([(x[:,r]@Wt.T).var(0,unbiased=False) for r in refs]).mean(0)
  cross=torch.stack([2*m[:,0]*((x[:,r]@Wt.T-(x[:,r]@Wt.T).mean(0))*(S-ES)[:,None]).mean(0) for r in refs]).mean(0)
  off=(m[:,0]**2)*VS; n2=(Wt*Wt).sum(1)
  return dict(sigma2=var_i.mean().item(),width=width.mean().item(),offset=off.mean().item(),cross=cross.mean().item(),
    resid=(var_i-width-off-cross).abs().max().item(),k_i_mean=(width/n2).mean().item(),k_i_sd=(width/n2).std().item(),
    off_over_m2ES2=(off/(m[:,0]**2*ES**2)).mean().item(),zbar=zr.mean().item(),bmean=b.mean().item(),
    mES_over_zbar_median=(m[:,0]*ES/zr.mean((0,1))).median().item())
ck=[]
for arm in ['LR','SNA']:
  for s in range(3):
    p=torch.load(f'results/boundary_groups_0908/{arm}_none_s{s}_task100.pt',map_location='cpu',weights_only=False)['state']['params']
    W=p[0].double();b=p[1].double()
    r=decomp(W,b,refs); r.update(arm=arm,seed=s,kind='8refperms'); ck.append(r)
    r=decomp(W,b,[torch.arange(784)]); r.update(arm=arm,seed=s,kind='MUT_identity_perm'); ck.append(r)
ck=pd.DataFrame(ck); pd.set_option('display.width',300); pd.set_option('display.precision',4); pd.set_option('display.max_columns',30)
print("\n== exact decomposition on t100 checkpoints ==");print(ck.set_index(['arm','seed','kind']))
ck.to_csv(OUT/'checkpoint_decomp.csv',index=False)
runs={'LIN':'linear_growth_0910/LIN','LR':'elu_growth_0909/LR_none','ELU03':'elu_growth_0909/ELU03_none','ELU1':'elu_growth_0909/ELU1_none','SNA':'elu_growth_0909/SNA_none'}
rows=[]
for arm,base in runs.items():
  for s in range(3):
    d=np.load(f'results/{base}_s{s}_units.npz'); cn,zb,sd=d['cnorm_i'],d['zbar_i'],d['sd_i']
    tasks=d['tasks'] if 'tasks' in d else np.arange(1,121)
    i20=int(np.where(tasks==20)[0][0]); i120=int(np.where(tasks==120)[0][0]); T=tasks>=20
    X=np.stack([cn[T].ravel()**2, zb[T].ravel()**2],1); y=sd[T].ravel()**2
    def fit(X,y):
      th,*_=np.linalg.lstsq(X,y,rcond=None); return th, 1-((y-X@th)**2).sum()/((y-y.mean())**2).sum()
    (a,gg),r2=fit(X,y)
    yp=kap*X[:,0]+gam*X[:,1]; r2th=1-((y-yp)**2).sum()/((y-y.mean())**2).sum()
    rng=np.random.default_rng(s); zsh=np.stack([rng.permutation(z) for z in zb[T]]).ravel()   # MUT: shuffle zbar across units within each task
    (am,gm),r2m=fit(np.stack([X[:,0],zsh**2],1),y)
    s2=(sd[i120]**2).mean(); n2=(cn[i120]**2).mean(); off=gam*(zb[i120]**2).mean()
    rows.append(dict(arm=arm,seed=s,a_fit=a,g_fit=gg,R2_fit=r2,R2_theory_0param=r2th,R2_MUT_shuffle_zbar=r2m,g_MUT=gm,
      N120=cn[i120].mean(),sig120=np.sqrt(s2),s2_over_N2=s2/n2,width_term=kap*n2,offset_term=off,pred_s2=kap*n2+off,act_s2=s2,
      off_frac=off/s2,zbar120=zb[i120].mean(),rms_zbar_i=np.sqrt((zb[i120]**2).mean()),sig_width=np.sqrt(kap*n2),depth_over_sigwidth=zb[i120].mean()/np.sqrt(kap*n2),
      N20=cn[i20].mean(),sig20=np.sqrt((sd[i20]**2).mean())))
df=pd.DataFrame(rows); df.to_csv(OUT/'unit_fit.csv',index=False)
med=df.groupby('arm').median(numeric_only=True).drop(columns='seed').loc[['LIN','LR','ELU03','ELU1','SNA']]
print("\n== per-unit fit on logged units (t>=20), seed median ==");print(med.T)
med.T.to_csv(OUT/'unit_fit_median.csv')
json.dump(const,open(OUT/'constants.json','w'),indent=1)
