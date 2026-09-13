"""Post-hoc (unregistered) re-analysis: is the upper edge of the preactivation
distribution pinned at the kink (0) while the mean sinks and the width grows?

No new learning.  Reads only committed outputs:
  results/task20to100_0908/{arm}.npz            per-unit zbar_i / within_i, tasks 20..100, current perm
  results/elu_growth_0909/{arm}_rows.csv        pooled pos_frac at task ends (phase 'end'), tasks 1..120
  results/width_depth_intervention_0909/*       one-shot width/depth kicks at task 20, tasks 20..40
Writes results/edge_pin_posthoc_0909/{occupancy_by_task.csv,invariants.csv,setpoints.csv,
intervention.csv,summary.md,fig_edge_pin.png}.
"""
from pathlib import Path
import csv,json,hashlib,math
import numpy as np
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results/edge_pin_posthoc_0909'
T20=ROOT/'results/task20to100_0908';ELU=ROOT/'results/elu_growth_0909';WDI=ROOT/'results/width_depth_intervention_0909'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def Phi(x):return .5*(1+np.vectorize(math.erf)(x/math.sqrt(2)))
def rows(path):return list(csv.DictReader(open(path)))
def csvwrite(path,rr):
 keys=[];[keys.append(k) for r in rr for k in r if k not in keys]
 with open(path,'w',newline='') as f:w=csv.DictWriter(f,keys);w.writeheader();[w.writerow({k:r.get(k) for k in keys}) for r in rr]
def md(rr,cols,fmt='{:.3f}'):
 out=['| '+' | '.join(cols)+' |','|'+'---|'*len(cols)]
 for r in rr:out.append('| '+' | '.join(fmt.format(r[c]) if isinstance(r[c],float) else str(r[c]) for c in cols)+' |')
 return '\n'.join(out)
# ---------------- 1. task20to100: per-unit means, current perm ----------------
ARMS20=['LR_none','SNA_none','LR_l2','SNA_l2'];occ=[];inv=[]
for arm in ARMS20:
 for s in range(3):
  d=np.load(T20/f'{arm}_s{s}.npz');t=d['task'];zm=d['zmean'];sd=np.sqrt(d['within'])
  for i,tt in enumerate(t):
   m=zm[i];w=sd[i];top=m+2*w
   occ.append(dict(arm=arm,seed=s,task=int(tt),mean=float(m.mean()),sdB=float(m.std()),sdW=float(w.mean()),
    u_pos=float((m>0).mean()),p_pos_gauss=float(Phi(m/w).mean()),q90_zbar=float(np.quantile(m,.9)),
    top2W_mean=float(top.mean()),top2W_q90=float(np.quantile(top,.9)),
    r_B=float(-m.mean()/m.std()),r_W=float(-m.mean()/w.mean())))
csvwrite(OUT.mkdir(parents=True,exist_ok=True) or OUT/'occupancy_by_task.csv',occ)
# invariance table: window t40..100 (fixed), seed-wise CV then median; also t20 and t100 levels
for arm in ARMS20:
 for q in ['u_pos','p_pos_gauss','q90_zbar','r_B','r_W','top2W_mean','mean','sdB','sdW']:
  lv20=[];lv100=[];cv=[];sdw=[]
  for s in range(3):
   rr=[r for r in occ if r['arm']==arm and r['seed']==s];x=np.array([r[q] for r in rr if 40<=r['task']<=100])
   lv20.append([r[q] for r in rr if r['task']==20][0]);lv100.append([r[q] for r in rr if r['task']==100][0])
   sdw.append(float(x.std()));cv.append(float(x.std()/abs(x.mean())) if abs(x.mean())>1e-9 else float('nan'))
  inv.append(dict(arm=arm,quantity=q,t20=float(np.median(lv20)),t100=float(np.median(lv100)),
   ratio_100_over_20=float(np.median(np.array(lv100)/np.array(lv20))) if min(abs(np.array(lv20)))>1e-9 else float('nan'),
   sd_t40_100=float(np.median(sdw)),cv_t40_100=float(np.median(cv))))
csvwrite(OUT/'invariants.csv',inv)
# ---------------- 2. elu_growth: exact pooled pos_frac at task ends ----------------
ELUARMS=['LR_none','ELU03_none','ELU1_none','SNA_none','ELU1_l2'];setp=[];series={}
for arm in ELUARMS:
 per=[]
 for s in range(3):
  rr=[r for r in rows(ELU/f'{arm}_s{s}_rows.csv') if r['phase']=='end']
  g=lambda k,t:[float(r[k]) for r in rr if int(r['task'])==t][0]
  late=np.array([float(r['pos_frac']) for r in rr if 40<=int(r['task'])<=100])
  per.append(dict(p20=g('pos_frac',20),p100=g('pos_frac',100),p_late_mean=float(late.mean()),p_late_sd=float(late.std()),
   p_late_min=float(late.min()),p_late_max=float(late.max()),z20=g('zbar_cur',20),z100=g('zbar_cur',100),s20=g('sigma_cur',20),s100=g('sigma_cur',100),
   zi20=g('zbar_inv',20),zi100=g('zbar_inv',100),si20=g('sigma_inv',20),si100=g('sigma_inv',100)))
  series[(arm,s)]=[(int(r['task']),float(r['pos_frac'])) for r in rr]
 setp.append(dict(arm=arm,**{k:float(np.median([p[k] for p in per])) for k in per[0]},
  **{k+'_seeds':';'.join(f'{p[k]:.3f}' for p in per) for k in ['p20','p_late_mean','p100']}))
csvwrite(OUT/'setpoints.csv',setp)
# ---------------- 3. width/depth kicks: does the kick move occupancy, is the level ratio restored ----------------
kick=[]
for arm in ['LR_none','SNA_none']:
 for iv in ['k0.7','k1.0','k1.4','bplus','bminus']:
  per=[]
  for s in range(3):
   u=np.load(WDI/f'{arm}_s{s}_units.npz');rr=rows(WDI/f'{arm}_s{s}_rows.csv')
   end={}
   for r in rr:
    if r['intervention']!=iv:continue
    if (r['phase']=='post' and r['task']=='20') or (r['phase']=='train' and r['step']=='625'):
     end[int(r['task'])]=(float(r['zbar_inv']),float(r['sigma_inv']),float(r['pos_frac']))
   per.append(dict(u_pos_inv_t20=float((u[f'{iv}_zbar_i_t20']>0).mean()),u_pos_inv_t40=float((u[f'{iv}_zbar_i_t40']>0).mean()),
    r_t20=-end[20][0]/end[20][1],r_t25=-end[25][0]/end[25][1],r_t40=-end[40][0]/end[40][1],
    p_cur_t20=end[20][2],p_cur_t21=end[21][2],p_cur_t40=end[40][2],z_t20=end[20][0],z_t40=end[40][0],s_t20=end[20][1],s_t40=end[40][1]))
  kick.append(dict(arm=arm,iv=iv,**{k:float(np.median([p[k] for p in per])) for k in per[0]},
   u_pos_inv_t20_seeds=';'.join(f'{p["u_pos_inv_t20"]:.3f}' for p in per),r_t40_seeds=';'.join(f'{p["r_t40"]:.2f}' for p in per)))
# level-ratio restoration for kappa arms, seed-wise against k1.0: needed dz = r_ref(20)*dsigma(20+); realised dz at t40
rest=[]
for arm in ['LR_none','SNA_none']:
 for s in range(3):
  rr=rows(WDI/f'{arm}_s{s}_rows.csv');E={}
  for r in rr:
   if (r['phase']=='post' and r['task']=='20') or (r['phase']=='train' and r['step']=='625'):E[(r['intervention'],int(r['task']))]=(float(r['zbar_inv']),float(r['sigma_inv']))
  zr20,sr20=E[('k1.0',20)];rref=-zr20/sr20
  for iv in ['k0.7','k1.4']:
   z20,s20=E[(iv,20)];dsig=s20-sr20;need=-rref*dsig
   for t in (30,40):
    dz=(E[(iv,t)][0]-z20)-(E[('k1.0',t)][0]-zr20)
    rest.append(dict(arm=arm,seed=s,iv=iv,task=t,dsigma_20plus=dsig,dz_needed_level=need,dz_realised=dz,restoration=dz/need))
csvwrite(OUT/'intervention.csv',kick+[dict(arm=r['arm'],iv=r['iv'],seed=r['seed'],task=r['task'],restoration=r['restoration'],dz_needed_level=r['dz_needed_level'],dz_realised=r['dz_realised']) for r in rest])
# ---------------- figure ----------------
import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
C={'LR_none':'#0072B2','ELU03_none':'#009E73','ELU1_none':'#E69F00','SNA_none':'#D55E00','ELU1_l2':'#333333'}
L={'LR_none':'leaky 0.1','ELU03_none':'ELU α=0.3','ELU1_none':'ELU α=1','SNA_none':'Snake (adaptive)','ELU1_l2':'ELU α=1 + L2 (control)'}
fig,ax=plt.subplots(1,3,figsize=(13.5,3.9));fig.subplots_adjust(wspace=.32,left=.05,right=.99,bottom=.16,top=.86)
for arm in ELUARMS:
 for s in range(3):
  tt,pp=zip(*series[(arm,s)]);ax[0].plot(tt,pp,color=C[arm],lw=1.2 if s==0 else .8,alpha=1 if s==0 else .45,ls='--' if arm=='ELU1_l2' else '-',label=L[arm] if s==0 else None)
ax[0].set_xlabel('task');ax[0].set_ylabel('p⁺ = P(z>0), task end, current perm');ax[0].set_title('Occupancy above the kink is held',loc='left',fontsize=10)
ax[0].legend(fontsize=7.5,frameon=False,loc='upper right');ax[0].set_ylim(0,.8)
for s in range(3):
 rr=[r for r in occ if r['arm']=='LR_none' and r['seed']==s];t=[r['task'] for r in rr]
 for k,c,lab in [('mean','#0072B2','mean of z̄ᵢ'),('q90_zbar','#D55E00','q90 of z̄ᵢ'),('top2W_mean','#009E73','mean of z̄ᵢ+2σᵢ')]:
  ax[1].plot(t,[r[k] for r in rr],color=c,lw=1.2 if s==0 else .8,alpha=1 if s==0 else .45,label=lab if s==0 else None)
 ax[1].plot(t,[r['mean']-r['sdB'] for r in rr],color='#0072B2',lw=.6,ls=':',alpha=.5,label='mean ± between-unit sd' if s==0 else None)
 ax[1].plot(t,[r['mean']+r['sdB'] for r in rr],color='#0072B2',lw=.6,ls=':',alpha=.5)
ax[1].axhline(0,color='#888',lw=.8);ax[1].set_xlabel('task');ax[1].set_ylabel('preactivation (leaky, no WD)');ax[1].set_title('q90 of unit means sits at 0; the mean sinks',loc='left',fontsize=10);ax[1].legend(fontsize=7.5,frameon=True,framealpha=.9,edgecolor='none',loc='lower left')
for s in range(3):
 rr=rows(WDI/f'LR_none_s{s}_rows.csv')
 for iv,c,lab in [('k0.7','#009E73','κ=0.7 (narrow)'),('k1.0','#0072B2','κ=1 (reference)'),('k1.4','#D55E00','κ=1.4 (wide)')]:
  pts=[(int(r['task']),-float(r['zbar_inv'])/float(r['sigma_inv'])) for r in rr if r['intervention']==iv and ((r['phase']=='post' and r['task']=='20') or (r['phase']=='train' and r['step']=='625'))]
  ax[2].plot(*zip(*pts),color=c,lw=1.2 if s==0 else .8,alpha=1 if s==0 else .45,label=lab if s==0 else None)
ax[2].set_xlabel('task');ax[2].set_ylabel('−z̄/σ (inv), task end');ax[2].set_title('One-shot width kick: level ratio relaxes back',loc='left',fontsize=10);ax[2].legend(fontsize=7.5,frameon=False)
for a in ax:a.spines[['top','right']].set_visible(False);a.grid(axis='y',color='#eee',lw=.6)
fig.savefig(OUT/'fig_edge_pin.png',dpi=160)
# ---------------- summary ----------------
S=['# edge_pin_posthoc_0909 — 上端の張り付き（事後・未登録）','',
 '新規学習なし。committed 出力の再集計のみ: `task20to100_0908`（ユニット別 z̄ᵢ・within、現 perm、t20–100）、`elu_growth_0909`（pooled pos_frac、タスク終端、現 perm）、`width_depth_intervention_0909`（一発介入）。',
 '窓ラベル: 「late」= t40–100 の固定窓。seed 内で集計してから 3 seed の中央値。','',
 '## 1. pooled 占有率 p⁺ = P(z>0)（elu_growth_0909、probe 512×100 ユニット、現 perm、タスク終端）','',
 md(setp,['arm','p20','p_late_mean','p_late_sd','p100','z20','z100','s20','s100']),'',
 'seed 別 p20 / p_late / p100: '+' ; '.join(f"{r['arm']}: {r['p20_seeds']} / {r['p_late_mean_seeds']} / {r['p100_seeds']}" for r in setp),'',
 '## 2. どの量が一定か（task20to100_0908、t40–100 の seed 内 sd と CV の 3 seed 中央値）','',
 md([r for r in inv if r['arm'] in ('LR_none','SNA_none','LR_l2')],['arm','quantity','t20','t100','ratio_100_over_20','sd_t40_100','cv_t40_100']),'',
 '`u_pos` = frac(z̄ᵢ>0)、`p_pos_gauss` = mean_i Φ(z̄ᵢ/σᵢ)（ガウス近似の pooled 占有率）、`q90_zbar` = ユニット平均の 90% 点、`r_B` = −mean/sd_between、`r_W` = −mean/mean(σ_within)、`top2W_mean` = mean_i(z̄ᵢ+2σᵢ)。','',
 '## 3. 一発介入（width_depth_intervention_0909）の再読み','',
 md(kick,['arm','iv','u_pos_inv_t20','u_pos_inv_t40','p_cur_t20','p_cur_t21','p_cur_t40','r_t20','r_t25','r_t40']),'',
 '幅腕 κ は star_i を厳密保存するので介入直後（t20 post）の u_pos_inv は κ 3 水準で同値（seed 別: '+' ; '.join(f"{r['arm']} {r['iv']}: {r['u_pos_inv_t20_seeds']}" for r in kick if r['iv'].startswith('k'))+'）。**占有率は一発介入では一度も揺らされていない。**','',
 '水準比 r=−z̄/σ の復元（必要量 = r_ref(20)·Δσ(20⁺)、実現量 = 参照との追加 Δz̄）:','',
 md(rest,['arm','seed','iv','task','dsigma_20plus','dz_needed_level','dz_realised','restoration']),'',
 '## 図','`fig_edge_pin.png`: (左) p⁺ の時系列 5 腕、(中) 素 leaky の mean / q90 / mean+2σ_within、(右) κ 腕の −z̄/σ の緩和。','',
 '## 出所 SHA','']
prov={}
for p in sorted(list(T20.glob('*.npz'))+list(ELU.glob('*_rows.csv'))+list(WDI.glob('*_units.npz'))+list(WDI.glob('*_rows.csv'))):prov[str(p.relative_to(ROOT))]=sha(p)
S.append(f'{len(prov)} files; code sha256 {sha(Path(__file__))}')
(OUT/'summary.md').write_text('\n'.join(S));(OUT/'provenance.json').write_text(json.dumps(dict(inputs=prov,code_sha256=sha(Path(__file__)),scope='post-hoc re-analysis of committed outputs; no new learning'),indent=1))
print('\n'.join(S[:40]))
