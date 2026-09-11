"""Verdicts for spec_h_sign_0912."""
import json
import numpy as np, pandas as pd
from src import boundary_gradient_0908 as G
ROOT=G.ROOT;OUT=ROOT/'results/h_sign_0912'
VT=(100,120);ORD=['HM03','HM01','H00','HP01','HP03','HP10'];HELD=ORD[:5]
EPS={'HM03':-.03,'HM01':-.01,'H00':0.,'HP01':.01,'HP03':.03,'HP10':.10}

def md(df):
    h='| arm | '+' | '.join(map(str,df.columns))+' |';s='|---|'+'---:|'*len(df.columns)
    return '\n'.join([h,s]+[f'| {i} | '+' | '.join(f'{x:g}' for x in r)+' |' for i,r in df.iterrows()])
def spear(a,b):return float(np.corrcoef(pd.Series(a).rank(),pd.Series(b).rank())[0,1])

def main():
    df=pd.concat([pd.read_csv(f) for f in sorted(OUT.glob('*_rows.csv'))],ignore_index=True)
    v=df[df.task.isin(VT)]
    g=v.groupby('arm').mean(numeric_only=True).reindex(ORD)
    g['E_rel']=v.assign(E_rel=v.E/v.r_scale).groupby('arm').E_rel.mean()
    g['c']=-g.cos;g['e2N2']=g.e2*g.N2
    g['N120']=df[df.task==120].groupby('arm').N.mean()
    ck=[json.loads(p.read_text())['checks'] for p in sorted(OUT.glob('*_provenance.json'))]
    mx=lambda k,f=max:(f([c[k] for c in ck if k in c]) if any(k in c for c in ck) else None)
    out=['# h_sign_0912 summary\n',
         "spec: `specs/spec_h_sign_0912.md`（事前登録 `6518619`、追補1 同日、判定値は未読で起動）\n"
         "実装: `src/h_sign_0912.py` / 集計 `src/h_sign_report_0912.py`\n"
         "scope: 120 タスク新規学習（CPU・1 スレッド）、6 腕 × 3 seed = 18 走。判定は t=100,120。\n",
         '## 0. 検査\n','| # | 検査 | 最悪値 | 変異対照 |','|---|---|---:|---:|',
         f"| G1 | `H00` が committed checkpoint {mx('g1_n')} 点 | maxabs **{mx('g1_maxabs'):.3g}** | {mx('g1_mutctl'):.3g} |",
         f"| B0 | **`H00` ≡ `LR`**（120 タスク） | maxabs **{mx('b0'):.3g}** | `HP01` と {mx('b0_mutctl',min):.3g} |",
         f"| C1 | φ′_ε = autograd | {mx('c1'):.3g} | {mx('c1_mutctl',min):.3g} |",
         f"| C2 | h_ε = zφ′−φ | {mx('c2'):.3g} | ELU の h {mx('c2_mutctl',min):.3g} |",
         f"| C3 | Euler E = Σe·h | {mx('euler'):.3g} | Snake の h {mx('euler_mut',min):.3g} |",
         f"| C4 | `H00` の graw/e2/ρ = `gradient_factors_0910` `LR` | **{mx('c4'):.3g}** | {mx('c4_mutctl',min):.3g} |",
         f"| C5 | 4 因子の積 = autograd | {mx('ident'):.3g} | ゲート除去 {mx('mut_nogate',min):.3g} |",
         f"| C6 | 測定の非侵襲 | **{mx('c6'):.3g}** | {mx('c6_mutctl',min):.3g} |",
         f"| C7 | `H00` の E/r_scale ≡ 0 | {mx('e_h00'):.3g} | `HP01` {mx('e_hp01'):.3g} |",
         f"\nRayleigh 違反 {sum(c.get('rayleigh',0) for c in ck)} 件、addgap ≤ {mx('addgap'):.3g}。\n"]
    Ep,Em=g.loc['HP03','E_rel'],g.loc['HM03','E_rel']
    V1=('FOLLOWS_H' if (Ep>=5e-4 and Em<=-5e-4) else 'OPPOSES_H' if (Ep<=-5e-4 and Em>=5e-4)
        else 'NO_FORCE' if (abs(Ep)<5e-4 or abs(Em)<5e-4) else 'MIXED')
    h=g.loc[HELD];e=[EPS[a] for a in HELD]
    sN=spear(e,h.N120);gap=g.loc['HP03','N120']-g.loc['HM03','N120']
    V2=('INWARD_SHRINKS' if (sN<=-.8 and gap<=-.3) else 'NO_RESPONSE' if abs(gap)<.15
        else 'OUTWARD_SHRINKS' if (sN>=.8 and gap>=.3) else 'PARTIAL')
    sc,sr=spear(e,h.c),spear(e,h.rho)
    V3=('VIA_EROSION' if (sc>=.8 and abs(sr)<.5) else 'VIA_INJECTION' if (sr<=-.8 and abs(sc)<.5)
        else 'BOTH' if (sc>=.8 and sr<=-.8) else 'NEITHER')
    gp,gm=g.loc['HP03','g2']/g.loc['H00','g2'],g.loc['HM03','g2']/g.loc['H00','g2']
    V4='HELD' if all(.8<=x<=1.25 for x in (gp,gm)) else 'MOVED'
    se,sei=spear(e,h.e2),spear(e,h.e2N2)
    V5=('NA' if V2!='INWARD_SHRINKS' else 'E2_OPPOSITE' if se>=.5 else 'E2_ALIGNED' if se<=-.5 else 'E2_FLAT')
    da=100*(g.loc['HP03','acc']-g.loc['H00','acc'])
    V6='FREE' if da>=-.3 else 'COSTS' if da<=-1. else 'SMALL_COST'
    pred={'V1':'FOLLOWS_H','V2':'INWARD_SHRINKS','V3':'VIA_EROSION','V4':'HELD','V5':'E2_OPPOSITE','V6':'FREE'}
    got={'V1':V1,'V2':V2,'V3':V3,'V4':V4,'V5':V5,'V6':V6}
    out+=['## 1. 判定\n','| | ラベル | 予測 | 実測 |','|---|---|---|---|',
      f"| V1 狙った符号の力 | **`{V1}`** | `FOLLOWS_H` | E/r_scale HP03 **{Ep:+.2e}** / HM03 **{Em:+.2e}** |",
      f"| V2 幅の応答 | **`{V2}`** | `INWARD_SHRINKS` | Spearman(ε,N120) {sN:+.2f}・gap {gap:+.3f} |",
      f"| V3 帳簿のどの項か | **`{V3}`** | `VIA_EROSION` | Spearman(ε,c) {sc:+.2f} / (ε,ρ) {sr:+.2f} |",
      f"| V4 ゲート保持 | **`{V4}`** | `HELD` | g2 比 HP03 {gp:.3f} / HM03 {gm:.3f}（HP10 {g.loc['HP10','g2']/g.loc['H00','g2']:.3f}） |",
      f"| V5 誤差の向き | **`{V5}`** | `E2_OPPOSITE` | Spearman(ε,e2) {se:+.2f} / (ε,e2·N²) {sei:+.2f} |",
      f"| V6 精度 | **`{V6}`** | `FREE` | acc(HP03)−acc(H00) = {da:+.2f} pt |",
      f"\n**Claude: {sum(pred[k]==got[k] for k in pred)}/6 的中。**\n",
      '## 2. 主要数値（t=100,120 × 3 seed）\n']
    t=g[['E_rel','r_W1b1','r_W2','N120','c','rho','D2','S2','e2','e2N2','g2','graw','w2col','rad1','rad2','sigma','zbar','posfrac','acc']].copy()
    t.insert(0,'ε',[EPS[a] for a in ORD])
    out.append(t.round(5).pipe(md))
    (OUT/'summary.md').write_text('\n'.join(out)+'\n');print('\n'.join(out))
if __name__=='__main__':main()
