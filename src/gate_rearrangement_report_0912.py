"""Verdicts for spec_gate_rearrangement_0912."""
import json
import numpy as np, pandas as pd
from src import boundary_gradient_0908 as G
ROOT=G.ROOT;OUT=ROOT/'results/gate_rearrangement_0912'
VT=(100,120)
ACT=['LR001','LR','LR03','ELU1','SN02','SN06','SN15']
LEAKY=['LR001','LR','LR03'];SNAKE=['SN02','SN06','SN15']

def md(df):
    h='| '+df.index.name+' | '+' | '.join(map(str,df.columns))+' |';s='|---|'+'---:|'*len(df.columns)
    return '\n'.join([h,s]+[f'| {i} | '+' | '.join(f'{x:g}' for x in r)+' |' for i,r in df.iterrows()])
def spear(a,b):return float(np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1])

def main():
    df=pd.concat([pd.read_csv(f) for f in sorted(OUT.glob('*_rows.csv'))],ignore_index=True)
    v=df[df.task.isin(VT)].dropna(subset=['rho'])
    cell=v.groupby(['actname','k']).mean(numeric_only=True)
    cell['c']=-cell.cos
    gcell=v[v.task>=100].groupby(['actname','k'])[['gate_corr','gate_l1','shock_ce']].mean()
    for col in gcell.columns:cell[col]=gcell[col]
    cell['N120']=df[df.task==120].groupby(['actname','k']).N.mean()
    S=pd.DataFrame({'S':[cell.loc[(a,.1),'c']/cell.loc[(a,1.),'c'] for a in ACT],
                    'c01':[cell.loc[(a,.1),'c'] for a in ACT],
                    'c00':[cell.loc[(a,0.),'c'] for a in ACT],
                    'c1':[cell.loc[(a,1.),'c'] for a in ACT],
                    'gate_corr01':[cell.loc[(a,.1),'gate_corr'] for a in ACT],
                    'gate_l101':[cell.loc[(a,.1),'gate_l1'] for a in ACT],
                    'nz':[cell.loc[(a,1.),'nz'] for a in ACT]},index=ACT)
    S.index.name='act'
    ck=[json.loads(p.read_text())['checks'] for p in sorted(OUT.glob('*_provenance.json'))]
    mx=lambda k,f=max:(f([c[k] for c in ck if k in c and c[k] is not None]) if any(k in c and c[k] is not None for c in ck) else None)
    prov=[json.loads(p.read_text()) for p in sorted(OUT.glob('*_provenance.json'))]
    p3n=[len({q['checks']['idx50'] for q in prov if q['seed']==sd}|{q['checks']['order50'] for q in prov if q['seed']==sd}) for sd in range(3)]
    p3='PASS' if all(n==2 for n in p3n) else 'FAIL'
    out=['# gate_rearrangement_0912 summary\n',
         "spec: `specs/spec_gate_rearrangement_0912.md`（事前登録 `08da078`、追補1、判定値は未読で起動）\n"
         "実装: `src/gate_rearrangement_0912.py` / 集計 `src/gate_rearrangement_report_0912.py`\n"
         "scope: 120 タスク新規学習、活性化 7 種 × k∈{0, 0.1, 1} × 3 seed = 63 走。判定は t=100,120。\n",
         '## 0. 検査\n','| # | 検査 | 結果 | 変異対照 |','|---|---|---:|---:|',
         f"| G1 | `K100_LR` が committed checkpoint {mx('g1_n')} 点 | maxabs **{mx('g1_maxabs'):.3g}** | {mx('g1_mutctl'):.3g} |",
         f"| C1 | `perm_overlap_0912` と共有セルが一致 | **{mx('c1'):.3g}** | 別 k と {mx('c1_mutctl',min):.3g} |",
         f"| P1 | perm 全タスク妥当 | {'PASS' if all(c['valid'] for c in ck) else 'FAIL'} | — |",
         f"| P2 | 固定率 f が狙い ±0.02 | 最大ずれ {mx('f_err'):.4f} | — |",
         f"| P4 | 手計算 = torch.corrcoef | {mx('p4'):.3g} | 自己相関 {mx('p4_self'):.3g}・対を壊すと {mx('p4_shuf'):.3g} |",
         f"| C2 | 測定の非侵襲 | **{mx('c2'):.3g}** | {mx('c2_mutctl',min):.3g} |",
         f"| C3 | 4 因子の積 = autograd | {mx('ident'):.3g} | ゲート除去 {mx('mut_nogate',min):.3g} |",
         f"\nRayleigh 違反 {sum(c.get('rayleigh',0) for c in ck)} 件。"
         f"P3（`gd`/`gb` の消費が k・活性化に依らない）: {p3}（seed ごとに 1 値、実測 {p3n} 通り）。"
         f"P5 は追補 1 で報告のみ: 帰無 gc の中央値 {np.median([c['null_gc'] for c in ck if c.get('null_gc') is not None]):.4f}。\n"]
    s1=spear(S.gate_corr01,S.S)
    V1='EXPLAINS' if s1<=-.8 else ('FAILS' if abs(s1)<.5 else 'PARTIAL')
    sn=S.loc[SNAKE,'S'].values
    V2='MONOTONE' if (sn[0]<sn[1]<sn[2]) else 'NOT'
    lk=S.loc[LEAKY,'S'].values
    V3=('NO_SATURATION' if (lk<=.5).all() else 'SATURATES' if (lk>=.8).any() else 'PARTIAL')
    se=S.loc['ELU1','S']
    V4='LIKE_LEAKY' if se<=.5 else ('LIKE_SNAKE' if se>=.8 else 'BETWEEN')
    fl=cell.reset_index()
    sg,ss=spear(fl.gate_corr,fl.c),spear(fl.shock_ce,fl.c)
    V5=('GATE_WINS' if abs(sg)>=abs(ss)+.15 else 'SHOCK_WINS' if abs(ss)>=abs(sg)+.15 else 'TIE')
    pred=dict(V1='EXPLAINS',V2='MONOTONE',V3='NO_SATURATION',V4='LIKE_LEAKY',V5='GATE_WINS')
    got=dict(V1=V1,V2=V2,V3=V3,V4=V4,V5=V5)
    out+=['## 1. 判定\n','| | ラベル | 予測 | 実測 |','|---|---|---|---|',
      f"| V1 組み換えが飽和を説明するか | **`{V1}`** | `EXPLAINS` | Spearman(gate_corr, S) **{s1:+.2f}** |",
      f"| V2 Snake 族で α に単調か | **`{V2}`** | `MONOTONE` | S = {' / '.join(f'{x:.2f}' for x in sn)} |",
      f"| V3 傾き族は飽和しないか | **`{V3}`** | `NO_SATURATION` | S = {' / '.join(f'{x:.2f}' for x in lk)} |",
      f"| V4 ELU はどちら側か | **`{V4}`** | `LIKE_LEAKY` | S(ELU1) **{se:.2f}** |",
      f"| V5 c を並べるのは | **`{V5}`** | `GATE_WINS` | gate_corr {sg:+.2f} / 切替CE {ss:+.2f}（21 セル） |",
      f"\n**Claude: {sum(pred[k]==got[k] for k in pred)}/5 的中。**\n",
      '## 2. 飽和比と組み換え（活性化ごと）\n',S.round(4).pipe(md),
      '\n## 3. 全 21 セル\n']
    t=cell.reset_index()
    t['cell']=t.actname+'_k'+t.k.astype(str);t=t.set_index('cell')
    t.index.name='cell'
    out.append(t[['k','gate_corr','gate_l1','shock_ce','c','D2','rho','e2','N120','sigma','zbar','acc']].round(4).pipe(md))
    (OUT/'summary.md').write_text('\n'.join(out)+'\n');print('\n'.join(out))
if __name__=='__main__':main()
