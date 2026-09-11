"""Verdicts for spec_gradient_factors_0910."""
from pathlib import Path
from math import erf,exp,sin,cos,sqrt
import json
import numpy as np, pandas as pd
from src import boundary_gradient_0908 as G
ROOT=G.ROOT;OUT=ROOT/'results/gradient_factors_0910'
VT=(100,120)
REF=['LR','LR001','LR03','ELU1','LIN','SN02','SN06','SN15']
BL=['BL001','BL010','BL050','BL100']
GP={'LR':('leaky',.1),'LR001':('leaky',.01),'LR03':('leaky',.3),'ELU1':('elu',1.),
    'LIN':('lin',0.),'SN02':('snake',.2),'SN06':('snake',.6),'SN15':('snake',1.5)}

def md(df):
    h='| arm | '+' | '.join(map(str,df.columns))+' |'
    s='|---|'+'---:|'*len(df.columns)
    return '\n'.join([h,s]+[f'| {i} | '+' | '.join(f'{x:g}' for x in r)+' |' for i,r in df.iterrows()])
def spear(a,b):
    return float(np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1])
def gauss_rms(kind,p,zb,s):
    """RMS phi' under a normal z -- the §0 estimate, reproduced for V2."""
    if kind=='lin':return 1.
    pp=.5*(1+erf(zb/(s*sqrt(2))))
    if kind=='leaky':return sqrt(pp+p*p*(1-pp))
    if kind=='snake':
        a=p;E1=exp(-2*a*a*s*s)*sin(2*a*zb);E2=.5*(1-exp(-8*a*a*s*s)*cos(4*a*zb))
        return sqrt(max(1+2*E1+E2,1e-12))
    m=exp(2*zb+2*s*s)*.5*(1+erf((-zb-2*s*s)/(s*sqrt(2))))
    return sqrt(max(pp+p*p*m,1e-12))

def main():
    df=pd.concat([pd.read_csv(f) for f in sorted(OUT.glob('*_rows.csv'))],ignore_index=True)
    v=df[df.task.isin(VT)].dropna(subset=['gw2'])
    g=v.groupby('arm').mean(numeric_only=True)
    end=df[df.task==120].groupby('arm').N.mean()
    out=['# gradient_factors_0910 summary\n',
         "spec: `specs/spec_gradient_factors_0910.md`（事前登録 commit `f84b881`、追補1 `db2add8`、"
         "判定値は未読で起動）\n実装: `src/gradient_factors_0910.py` / 集計 `src/gradient_factors_report_0910.py`\n"
         "scope: 120 タスク新規学習（CPU・1 スレッド）、12 腕 × 3 seed = 36 走。判定は t=100,120。\n"]

    ck=[json.loads(p.read_text())['checks'] for p in sorted(OUT.glob('*_provenance.json'))]
    def mx(k,f=max):
        vals=[c[k] for c in ck if k in c and c[k] is not None];return f(vals) if vals else None
    out.append('## 0. 検査\n')
    out.append('| # | 検査 | 最悪値 | 変異対照 |');out.append('|---|---|---:|---:|')
    out.append(f"| C1 | 4 因子の積 = autograd の ‖∇W1‖² | {mx('ident'):.3g} | "
               f"ゲートを外す {mx('mut_nogate',min):.3g} |")
    out.append(f"| C2 | R が K のスペクトル内（Rayleigh） | 違反 {sum(c.get('rayleigh',0) for c in ck)} 件 | — |")
    bl=[c for c in ck if c.get('arm','').startswith('BL') or 'fwd_gap' in c]
    fw=[c['mut_fwd'] for c in ck if c.get('mut_fwd') not in (None,float('inf'))]
    out.append(f"| C3 | `BL` 腕で前向きゲートを使うと壊れる | — | {min(fw):.3g}〜{max(fw):.3g} |")
    c4=[c['c4'] for c in ck if c.get('c4') is not None]
    if c4:out.append(f"| C4 | `graw` が step_persistence_0910 と一致 | **{max(c4):.3g}** | "
                     f"{min(c['c4_mutctl'] for c in ck if c.get('c4_mutctl')):.3g} |")
    g1=[c for c in ck if 'g1_maxabs' in c]
    if g1:out.append(f"| G1 | committed checkpoint {g1[0]['g1_n']} 点 | maxabs **{max(c['g1_maxabs'] for c in g1):.3g}** | "
                     f"{max(c.get('g1_mutctl',0) for c in g1):.3g} |")
    b0=[c for c in ck if 'b0' in c]
    if b0:out.append(f"| B0 | **`BL010` が `LR` と一致** | maxabs **{max(c['b0'] for c in b0):.3g}** | "
                     f"`BL050` と {min(c['b0_mutctl'] for c in b0):.3g} |")
    c5=[c['c5'] for c in ck if 'c5' in c]
    if c5:out.append(f"| C5 | 測定の非侵襲 | **{max(c5):.3g}** | {min(c['c5_mutctl'] for c in ck if 'c5_mutctl' in c):.3g} |")
    out.append(f"| C6 | φ′ が autograd と一致 | {mx('dphi'):.3g} | {mx('dphi_mutctl',min):.3g} |")

    # ---- V1
    R=g.loc[REF]
    L={k:np.log(R[k]) for k in ('gw2','g2','e2','xi','R')}
    tot=L['gw2'];f={k:float(np.cov(L[k],tot)[0,1]/np.var(tot,ddof=1)) for k in ('g2','e2','xi','R')}
    V1=('GATE_WITH_COMPENSATION' if f['g2']>=.6 and f['e2']<=0 else
        'GATE' if f['g2']>=.6 else 'ERROR' if f['e2']>=.6 else 'SPLIT')
    # ---- V2
    est=np.array([gauss_rms(*GP[a],R.loc[a].zbar,R.loc[a].sigma) for a in REF])
    exact=np.sqrt(R.phi2_probe.values)
    sp2=spear(est,exact);rel=float(np.abs(est/exact-1).max())
    V2='OK' if (sp2>=.9 and rel<=.25) else 'BIASED'
    # ---- V3
    B=g.loc[BL];rmsb=np.sqrt(B.phi2_probe.values)
    s_i=spear(rmsb,B.graw.values);s_ii=spear(B.graw.values,B.rho.values)
    order=np.argsort(rmsb);mono=bool(np.all(np.diff(end.loc[BL].values[order])<=0))
    V3=('GATE_CAUSAL' if (s_i>=.8 and s_ii<=-.8 and mono) else
        'PARTIAL' if s_i>=.8 else 'NO_EFFECT')
    # ---- V4
    b_,a_=np.polyfit(np.log(R.graw),np.log(R.rho),1)
    res=np.log(B.rho)-(a_+b_*np.log(B.graw));mres=float(res.abs().max())
    V4='ON' if mres<=.25 else 'OFF'
    # ---- V5
    sw=float(B.sigma.max()/B.sigma.min()-1);dz=float(B.zbar.max()-B.zbar.min())
    V5='HELD' if (sw<=.25 and dz<=1.0) else 'DRIFTED'
    out.append('\n## 1. 判定\n')
    out.append('| | ラベル | 予測 | 実測 |');out.append('|---|---|---|---|')
    out.append(f"| V1 腕差はどの因子か | **`{V1}`** | `GATE_WITH_COMPENSATION` | "
               f"f_g2 **{f['g2']:+.2f}** / f_e2 **{f['e2']:+.2f}** / f_ξ {f['xi']:+.2f} / f_R {f['R']:+.2f} |")
    out.append(f"| V2 正規仮定は公正か | **`{V2}`** | `BIASED` | Spearman {sp2:+.2f}・最大相対誤差 {100*rel:.0f}% |")
    out.append(f"| V3 ゲートだけの介入 | **`{V3}`** | `GATE_CAUSAL` | "
               f"(i) {s_i:+.2f} / (ii) {s_ii:+.2f} / (iii) 単調減 {mono} |")
    out.append(f"| V4 介入腕は直線に乗るか | **`{V4}`** | `ON` | 最大残差 {mres:.3f}（参照 sd 0.082） |")
    out.append(f"| V5 ループは切れたか | **`{V5}`** | `DRIFTED` | σ 幅 {100*sw:.0f}%・Δz̄ {dz:.2f} |")
    hit=sum([V1=='GATE_WITH_COMPENSATION',V2=='BIASED',V3=='GATE_CAUSAL',V4=='ON',V5=='DRIFTED'])
    out.append(f"\n**Claude: {hit}/5 的中。**\n")

    out.append('## 2. 参照 8 腕（t=100,120 × 3 seed・因子は幾何平均）\n')
    t=g.loc[REF][['graw','gw2','g2','e2','xi','R','rho','phi2_probe','sigma','zbar','posfrac']].copy()
    t.insert(len(t.columns),'N120',end.loc[REF]);t.insert(8,'RMSφ′実測',np.sqrt(g.loc[REF].phi2_probe))
    t.insert(9,'RMSφ′正規',est)
    out.append(t.round(5).pipe(md))
    out.append('\n## 3. 介入 4 腕（前向きは全部 leaky a=0.1）\n')
    b=g.loc[BL][['graw','g2','e2','xi','R','rho','phi2_probe','phi2_probe_fwd','sigma','zbar','acc']].copy()
    b.insert(0,'a_b',[.01,.1,.5,1.]);b['N120']=end.loc[BL]
    b['線からの残差']=res
    out.append(b.round(5).pipe(md))
    (OUT/'summary.md').write_text('\n'.join(out)+'\n')
    print('\n'.join(out))

if __name__=='__main__':main()
