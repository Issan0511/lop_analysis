"""Verdicts for spec_step_persistence_0910."""
from pathlib import Path
import json,glob,os
import numpy as np, pandas as pd
from src import boundary_gradient_0908 as G
ROOT=G.ROOT;OUT=ROOT/'results/step_persistence_0910'
VT=(100,120);LAGS=[1,2,3,4,5,10,20,50,100,200,400]
ARMORD=['LR','LR001','LR03','ELU1','LIN','SN02','SN06','SN15']

def md(df):
    h='| arm | '+' | '.join(map(str,df.columns))+' |'
    s='|---|'+'---:|'*len(df.columns)
    b=[f'| {i} | '+' | '.join(f'{x:g}' for x in r)+' |' for i,r in df.iterrows()]
    return '\n'.join([h,s]+b)

def spearman(a,b):
    ra=pd.Series(a).rank();rb=pd.Series(b).rank();return float(np.corrcoef(ra,rb)[0,1])

def load():
    rs=[]
    for f in sorted(OUT.glob('*_rows.csv')):
        d=pd.read_csv(f);rs.append(d)
    return pd.concat(rs,ignore_index=True)

def main():
    df=load();v=df[df.task.isin(VT)].dropna(subset=['rho'])
    g=v.groupby('arm').mean(numeric_only=True)
    g=g.reindex([a for a in ARMORD if a in g.index])
    out=[]
    out.append('# step_persistence_0910 summary\n')
    prov=json.loads(next(OUT.glob('LR_s0_provenance.json')).read_text())
    out.append(f"spec: `specs/spec_step_persistence_0910.md`（事前登録 commit `45f82c6`、判定値は未読で起動）\n"
               f"実装: `src/step_persistence_0910.py` / 集計 `src/step_persistence_report_0910.py`\n"
               f"scope: 120 タスク新規学習（CPU・1 スレッド）、8 腕 × 3 seed = 24 走。判定は t=100,120。\n")

    # --- checks
    ck=[json.loads(p.read_text())['checks'] for p in sorted(OUT.glob('*_provenance.json'))]
    def mx(k):
        vals=[c[k] for c in ck if k in c and c[k] is not None]
        return (max(vals),min(vals)) if vals else (None,None)
    out.append('## 0. 検査\n')
    out.append('| # | 検査 | 最悪値 | 変異対照 |')
    out.append('|---|---|---:|---:|')
    rowsck=[('C1','Σ_s ũ_s == W̃_end − W̃_start','c1',None),
            ('C2','G(1)=1 かつ G(625)=D²/S²','c2',None),
            ('C4','部分標本が X を再現','c4','c4_mutctl'),
            ('C5','Adam の一歩の形','c5','c5_mutctl'),
            ('C6','既登録 S²・D² と一致','c6','c6_mutctl'),
            ('C8',"φ′ が autograd と一致",'dphi','dphi_mutctl')]
    for tag,name,k,km in rowsck:
        a,_=mx(k)
        if a is None:continue
        m=''
        if km:
            vals=[c[km] for c in ck if km in c and c[km] is not None]
            m=f'{min(vals):.3g}' if vals else ''
        out.append(f'| {tag} | {name} | {a:.3g} | {m} |')
    g1=[c for c in ck if 'g1_maxabs' in c]
    if g1:
        out.append(f"| G1 | `LR` が committed checkpoint {g1[0]['g1_n']} 点を再現 | "
                   f"maxabs **{max(c['g1_maxabs'] for c in g1):.3g}** | "
                   f"{max(c.get('g1_mutctl',0) for c in g1):.3g} |")
    c7=[c['c7'] for c in ck if 'c7' in c]
    if c7:
        out.append(f"| C7 | 測定の非侵襲 | **{max(c7):.3g}** | "
                   f"{min(c['c7_mutctl'] for c in ck if 'c7_mutctl' in c):.3g} |")
    cs=[c for c in ck if not c.get('c3',True)]
    out.append(f"\nC3（Cauchy–Schwarz G(b) ≤ b）: {'PASS' if not cs else 'FAIL'}（{len(ck)} 走）\n")

    # --- V1
    out.append('## 1. 判定\n')
    xl=g.X_long_frac.median()
    V1='LONG_MEMORY' if xl>=0.60 else ('SHORT_MEMORY' if xl<=0.35 else 'MIXED')
    # --- V2
    lr,sn=g.loc['LR'],g.loc['SN06']
    dX=lr.X-sn.X
    flong=((lr.X*lr.X_long_frac)-(sn.X*sn.X_long_frac))/dX if dX!=0 else np.nan
    V2='LONG' if flong>=0.65 else ('SHORT' if flong<=0.35 else 'BOTH')
    # --- V3
    cv=lambda s:float(s.std()/s.mean())
    cvk=cv(g.kap2)
    V3='CONFIRMED' if cvk<=0.10 else ('REOPENED' if cvk>=0.25 else 'V3_PARTIAL')
    # --- V4
    sp=spearman(g.gcorr,g.rho)
    V4='GATE_EXPLAINS' if (abs(sp)>=0.7 and sp<0) else ('GATE_FAILS' if abs(sp)<0.5 else 'V4_PARTIAL')
    # --- V5
    cvg,cvs=cv(g.graw),cv(g.S2)
    ratio=cvg/cvs
    V5='SCALE_KILLED' if ratio>=2.0 else ('SCALE_SURVIVES' if ratio<1.2 else 'V5_PARTIAL')
    out.append('| 判定 | ラベル | 予測 | 実測 |')
    out.append('|---|---|---|---|')
    out.append(f'| V1 整列はどのラグに住むか | **`{V1}`** | `LONG_MEMORY` | X_long/X 中央値 **{xl:.3f}** |')
    out.append(f'| V2 腕差（LR 対 SN06）の在処 | **`{V2}`** | `LONG` | f_long **{flong:.3f}** |')
    out.append(f'| V3 一歩の予算は腕不変か | **`{V3}`** | `CONFIRMED` | κ2 の腕間 CV **{100*cvk:.1f}%** |')
    out.append(f'| V4 ゲート回転説 | **`{V4}`** | `GATE_FAILS` | Spearman(gcorr, ρ) **{sp:+.2f}** |')
    out.append(f'| V5 Adam は生勾配の尺度を殺すか | **`{V5}`** | `SCALE_KILLED` | CV(graw)/CV(S²) **{ratio:.2f}** |')
    hit=sum([V1=='LONG_MEMORY',V2=='LONG',V3=='CONFIRMED',V4=='GATE_FAILS',V5=='SCALE_KILLED'])
    out.append(f'\n**Claude: {hit}/5 的中。**\n')

    # --- tables
    out.append('## 2. 主要数値（t=100,120 × 3 seed 平均）\n')
    cols=['D2','S2','rho','X_long_frac','kap1','kap2','graw','gcorr','N','sigma','zbar']
    t=g[cols].copy()
    t.columns=['D²','S²','ρ','X_long/X','κ1','κ2','‖∇W1‖','gcorr','N','σ','z̄']
    out.append(t.round(4).pipe(md))
    out.append('\n### ブロック因子 G(b)（1=拡散、b=弾道）\n')
    bl=g[[f'G{b}' for b in (1,5,25,125,625)]].copy()
    bl.columns=['G(1)','G(5)','G(25)','G(125)','G(625)=ρ']
    out.append(bl.round(3).pipe(md))
    out.append('\n### ラグ相関 A(τ)\n')
    la=g[[f'A{l}' for l in LAGS]].copy();la.columns=[f'τ={l}' for l in LAGS]
    out.append(la.round(4).pipe(md))

    out.append('\n## 3. 読み\n')
    out.append(f'- ρ の腕間の並びは gcorr（ゲートの回転）と Spearman {sp:+.2f}。')
    out.append(f'- 平均ペア cos = (ρ−1)/624: '+'、'.join(f'{a} {(g.loc[a].rho-1)/624:.4f}' for a in g.index)+'。')
    (OUT/'summary.md').write_text('\n'.join(out)+'\n')
    print('\n'.join(out))

if __name__=='__main__':main()
