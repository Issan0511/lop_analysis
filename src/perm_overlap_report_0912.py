"""Verdicts for spec_perm_overlap_0912."""
import json
import numpy as np, pandas as pd
from src import boundary_gradient_0908 as G
ROOT=G.ROOT;OUT=ROOT/'results/perm_overlap_0912'
VT=(100,120);KS=[('K000',0.),('K010',.10),('K025',.25),('K050',.50),('K100',1.)]
ACTS=['LR','SN06']

def md(df):
    h='| cell | '+' | '.join(map(str,df.columns))+' |';s='|---|'+'---:|'*len(df.columns)
    return '\n'.join([h,s]+[f'| {i} | '+' | '.join(f'{x:g}' for x in r)+' |' for i,r in df.iterrows()])
def spear(a,b):return float(np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1])

def integrate(D,c,N0,n):
    """dN^2/dt = D^2 - 2 c D N, Euler over n tasks."""
    N2=N0*N0
    for _ in range(n):
        N=np.sqrt(max(N2,1e-12));N2=max(N2+D*D-2*c*D*N,1e-12)
    return np.sqrt(N2)

def main():
    df=pd.concat([pd.read_csv(f) for f in sorted(OUT.glob('*_rows.csv'))],ignore_index=True)
    df['cell']=df.arm
    v=df[df.task.isin(VT)].dropna(subset=['rho'])
    g=v.groupby('cell').mean(numeric_only=True)
    g['c']=-g.cos;g['D']=np.sqrt(g.D2);g['Nstar']=g.D/(2*g.c)
    g['N120']=df[df.task==120].groupby('cell').N.mean()
    g['N20']=df[df.task==20].groupby('cell').N.mean()
    sh=df[df.task>=100].groupby('cell')[['shock_ce','shock_acc','f']].mean()
    for cN in sh.columns:g[cN]=sh[cN]
    a20=df[df.task==20].groupby('cell').acc.mean();a120=df[df.task==120].groupby('cell').acc.mean()
    g['acc_drop']=100*(a20-a120)
    late=df[(df.task>=60)&df.rho.notna()]
    g['p_late']=late.groupby('cell').apply(lambda s:np.polyfit(np.log(s.task),np.log(s.N),1)[0],include_groups=False)
    g['model']=[integrate(g.loc[i,'D'],g.loc[i,'c'],g.loc[i,'N20'],100) for i in g.index]
    order=[f'{k}_{a}' for a in ACTS for k,_ in KS]
    g=g.reindex([o for o in order if o in g.index])
    g['k']=[dict(KS)[i.split('_')[0]] for i in g.index]
    g['act']=[i.split('_')[1] for i in g.index]

    ck={json.loads(p.read_text())['arm']+'_s'+str(json.loads(p.read_text())['seed']):
        json.loads(p.read_text())['checks'] for p in OUT.glob('*_provenance.json')}
    allck=list(ck.values())
    mx=lambda k,f=max:(f([c[k] for c in allck if k in c and c[k] is not None]) if any(k in c for c in allck) else None)
    out=['# perm_overlap_0912 summary\n',
         "spec: `specs/spec_perm_overlap_0912.md`（事前登録 `d9aa0ef`、追補1、判定値は未読で起動）\n"
         "実装: `src/perm_overlap_0912.py` / 集計 `src/perm_overlap_report_0912.py`\n"
         "scope: 120 タスク新規学習（CPU・1 スレッド）、5 水準 × 2 活性化 × 3 seed = 30 走。判定は t=100,120。\n",
         '## 0. 検査\n','| # | 検査 | 結果 | 変異対照 |','|---|---|---:|---:|']
    # P4: gd/gb consumption identical across arms at the same seed
    p4=[];p3=[]
    for s in range(3):
        idx={n:c['idx50'] for n,c in ck.items() if n.endswith(f'_s{s}')}
        orr={n:c['order50'] for n,c in ck.items() if n.endswith(f'_s{s}')}
        p4.append(len(set(idx.values()))==1 and len(set(orr.values()))==1)
        h={n:c['t1_perm_hash'] for n,c in ck.items() if n.endswith(f'_s{s}')}
        p3.append(len(set(h.values()))==1)
    out+= [f"| G1 | `K100_LR` が committed checkpoint {mx('g1_n')} 点 | maxabs **{mx('g1_maxabs'):.3g}** | {mx('g1_mutctl'):.3g} |",
           f"| C2 | `K100_LR` の D²/cos/ρ/e2 = `gradient_factors_0910` `LR` | **{mx('c2'):.3g}** | {mx('c2_mutctl',min):.3g} |",
           f"| P1 | perm が全タスク妥当 | {'PASS' if all(c['valid'] for c in allck) else 'FAIL'} | — |",
           f"| P2 | 固定率 f が狙い ±0.02 | 最大ずれ {mx('f_err'):.4f} | — |",
           f"| P3 | タスク 1 の perm が全腕で同一（seed 別） | {'PASS' if all(p3) else 'FAIL'} | k=0 腕の t2=t1 / k=1 腕の t2≠t1 を個別確認 |",
           f"| P4 | `gd`・`gb` の消費が k に依らない（t=50） | {'PASS' if all(p4) else 'FAIL'} | — |",
           f"| P5 | k=1 のみ宿主経路（host_path） | {'PASS' if all((c['host_path']==120)==(ck_n.split('_')[0]=='K100') for ck_n,c in ck.items()) else 'FAIL'} | — |",
           f"| C1 | 測定の非侵襲（切替 CE 含む） | **{mx('c1'):.3g}** | {mx('c1_mutctl',min):.3g} |",
           f"| C3 | 4 因子の積 = autograd | {mx('ident'):.3g} | ゲート除去 {mx('mut_nogate',min):.3g} |",
           f"\nRayleigh 違反 {sum(c.get('rayleigh',0) for c in allck)} 件、addgap ≤ {mx('addgap'):.3g}。\n"]

    L=g[g.act=='LR']
    c0,c1=L.loc['K000_LR','c'],L.loc['K100_LR','c']
    V1=('SIGN_FLIPS' if (c0<=-0.01 and c1>=0.03) else 'NO_RESPONSE' if abs(c1-c0)<0.01
        else 'MONOTONE_NO_FLIP' if spear(L.k,L.c)>=0.8 else 'PARTIAL')
    gaps=[abs(g.loc[f'{k}_LR','c']-g.loc[f'{k}_SN06','c']) for k,_ in KS]
    V2='SAME_CURVE' if max(gaps)<=0.010 else ('DIFFERENT' if max(gaps)>=0.020 else 'PARTIAL')
    s3=spear(g.shock_ce,g.c)
    V3='MEDIATES' if s3>=.8 else ('NO' if s3<.5 else 'WEAK')
    p0=L.loc['K000_LR','p_late']
    V4='SUPER_SQRT' if p0>=.55 else ('SUB_SQRT' if p0<=.45 else 'SQRT')
    rel=(g.model/g.N120-1).abs();V5='HOLDS' if (rel.max()<=.10 and spear(g.model,g.N120)>=.9) else 'FAILS'
    sw=spear(g.N120,g.acc_drop);sk=spear(g.k,g.acc_drop)
    V6=('WIDTH_TRACKS' if sw<=-.7 else 'NO' if sw>-.4 else 'PARTIAL')
    if V6=='WIDTH_TRACKS' and abs(sk)>=abs(sw)-.1:V6='CONFOUNDED'
    pred=dict(V1='SIGN_FLIPS',V2='SAME_CURVE',V3='MEDIATES',V4='SUPER_SQRT',V5='HOLDS',V6='WIDTH_TRACKS')
    got=dict(V1=V1,V2=V2,V3=V3,V4=V4,V5=V5,V6=V6)
    out+=['## 1. 判定\n','| | ラベル | 予測 | 実測 |','|---|---|---|---|',
      f"| V1 c は k で反転するか | **`{V1}`** | `SIGN_FLIPS` | c(k=0) **{c0:+.4f}** / c(k=1) **{c1:+.4f}** |",
      f"| V2 2 活性化で同じ曲線か | **`{V2}`** | `SAME_CURVE` | |Δc| 最大 **{max(gaps):.4f}**（{', '.join(f'{x:.4f}' for x in gaps)}） |",
      f"| V3 切替ショックが媒介か | **`{V3}`** | `MEDIATES` | Spearman(切替CE, c) **{s3:+.2f}** |",
      f"| V4 c<0 で成長則 | **`{V4}`** | `SUPER_SQRT` | p_late(k=0) **{p0:.3f}** |",
      f"| V5 帳簿の再現 | **`{V5}`** | `HOLDS` | 最大相対誤差 {100*rel.max():.1f}%・Spearman {spear(g.model,g.N120):+.2f} |",
      f"| V6 幅が LoP を並べるか | **`{V6}`** | `WIDTH_TRACKS` | Spearman(N120, 低下) {sw:+.2f} / **(k, 低下) {sk:+.2f}** |",
      f"\n**Claude: {sum(pred[k]==got[k] for k in pred)}/6 的中。**\n",'## 2. 主要数値（t=100,120 × 3 seed）\n']
    t=g[['k','f','shock_ce','shock_acc','c','D','D2','rho','S2','e2','Nstar','N120','model','p_late','acc_drop','sigma','zbar']]
    out.append(t.round(4).pipe(md))
    (OUT/'summary.md').write_text('\n'.join(out)+'\n');print('\n'.join(out))
if __name__=='__main__':main()
