#!/usr/bin/env python3
"""Post-hoc (not registered) for l1_push_split_0915: split each task into the first 1000 updates after
the input/label switch ("early") and updates 1000-6000 ("late"), and decompose the change of z̄ in each
phase into the sample groups P/B/V.  Reads results/l1_push_split_0915/logs/chunks.npy.
    python3 -m analysis.l1_push_split_0915_phases
"""
from __future__ import annotations
import csv
import numpy as np
from src import l1_push_split_0915 as L
from src import l1_push_split_0915_report as R

OUT=R.OUT; G=("P","B","V"); EARLY=1000
DB=((0.0,3.0),(-1.0,0.0),(-2.0,-1.0),(-3.0,-2.0),(-6.0,-3.0),(-12.0,-6.0),(-np.inf,-12.0))
ZB=((0.0,np.inf),(-1.0,0.0),(-2.0,-1.0),(-4.0,-2.0),(-8.0,-4.0),(-16.0,-8.0),(-np.inf,-16.0))

def per_task(D,j):
    """(T, 100) arrays: task-start depth, early/late sums of dz_g, dzA, tv."""
    T=D.T;out={k:np.zeros((T,100)) for k in ["zst"]+[f"{ph}{q}" for ph in ("e","l") for q in ("P","B","V","A","tv")]}
    for t in range(1,T+1):
        cs=np.where(D.task==t)[0];out["zst"][t-1]=D.zs[t-1,j]
        for c in cs:
            ph="e" if D.end[c]<=EARLY else "l"
            assert D.start[c]>=EARLY or D.end[c]<=EARLY
            for g in G:out[f"{ph}{g}"][t-1]+=D.ch[c,j,:,R.C["dz"+g]]
            out[f"{ph}A"][t-1]+=D.ch[c,j,:,R.C["dzA"]];out[f"{ph}tv"][t-1]+=D.ch[c,j,:,R.C["tv"]]
    return out

def table(D,act,env,bins,rel=True):
    zc=L.ZC.get(act,0.0) if rel else 0.0
    P=[per_task(D,D.j(s,env,act)) for s in range(3)]
    cat=lambda k:np.concatenate([p[k][1:].ravel() for p in P])       # tasks 2..T (after a switch)
    x=cat("zst")-zc
    rows=[];lines=[f"|{'d' if rel else 'z̄'}（タスク開始時）|unit-task|前半 Δz̄|うち P/B/V|前半 上/下|後半 Δz̄|うち P/B/V|後半 上/下|",
                  "|---|---:|---:|---|---:|---:|---|---:|"]
    for lo,hi in bins:
        m=(x>=lo)&(x<hi)
        if m.sum()==0:continue
        e={g:float(cat("e"+g)[m].mean()) for g in G};l={g:float(cat("l"+g)[m].mean()) for g in G}
        eA,lA=float(cat("eA")[m].mean()),float(cat("lA")[m].mean())
        etv,ltv=float(cat("etv")[m].sum()),float(cat("ltv")[m].sum());eAs,lAs=float(cat("eA")[m].sum()),float(cat("lA")[m].sum())
        eud=(etv+eAs)/(etv-eAs) if etv>eAs else float("inf");lud=(ltv+lAs)/(ltv-lAs) if ltv>lAs else float("inf")
        rows.append(dict(act=act,env=env,lo=lo,hi=hi,n=int(m.sum()),early=eA,eP=e["P"],eB=e["B"],eV=e["V"],early_updown=eud,
                         late=lA,lP=l["P"],lB=l["B"],lV=l["V"],late_updown=lud))
        lines.append(f"|[{lo:g},{hi:g})|{int(m.sum())}|{eA:+.3f}|{e['P']:+.2f} / {e['B']:+.2f} / {e['V']:+.2f}|{eud:.2f}|"
                     f"{lA:+.3f}|{l['P']:+.2f} / {l['B']:+.2f} / {l['V']:+.2f}|{lud:.2f}|")
    return lines,rows

def burst200(D):
    """RL, t2-t10: the first 200-update chunk after the label switch versus the rest of the task."""
    L_=["## RL: 切替直後の 200 更新（沈み）と残り 5800 更新（戻し）を、タスクごとに P/B/V へ割る","",
        "値はユニットあたりの Δz̄ の和（3 seed または指定 seed のユニット平均）。d₀ はタスク開始時の z̄ − z_c の中央値（谷の無い活性化は z̄）。",""]
    for act,groups in (("GELU",(((0,1,2),"s0–2"),)),("SILU",(((0,1),"s0–1"),((2,),"s2"))),("LR",(((0,1,2),"s0–2"),)),("ELU1",(((0,1,2),"s0–2"),))):
        for seeds,lab in groups:
            js=[D.j(s,"RL",act) for s in seeds];zc=L.ZC.get(act,0.0)
            L_+=[f"### {act} {lab}","","|task|d₀|最初の 200 更新 P / B / V|計|占有 P/B/V|残り P / B / V|計|","|---|---:|---|---:|---|---|---:|"]
            for t in (2,3,4,5,6,8,10):
                cs=np.where(D.task==t)[0];c0=cs[0];rest=cs[1:]
                assert D.start[c0]==0 and D.end[c0]==200,("chunking changed",t,D.end[c0])
                b={g:float(np.mean([D.ch[c0,j,:,R.C["dz"+g]].mean() for j in js])) for g in G}
                r={g:float(np.mean([D.ch[rest,j,:,R.C["dz"+g]].sum(0).mean() for j in js])) for g in G}
                occ={g:float(np.mean([D.ch[c0,j,:,R.C["n"+g]].mean()/(D.n[c0]*16) for j in js])) for g in G}
                d0=float(np.median(np.concatenate([D.zs[t-1,j]-zc for j in js])))
                L_.append(f"|{t}|{d0:+.2f}|{b['P']:+.2f} / {b['B']:+.2f} / {b['V']:+.2f}|{sum(b.values()):+.2f}|"
                          f"{occ['P']:.2f}/{occ['B']:.2f}/{occ['V']:.2f}|{r['P']:+.2f} / {r['B']:+.2f} / {r['V']:+.2f}|{sum(r.values()):+.2f}|")
            L_.append("")
    return L_

def main():
    D=R.Data(OUT);M=["# l1_push_split_0915 事後（登録外）: タスク切替直後の 1000 更新と残り 5000 更新",
       "","各タスク（t2–150）の Δz̄ を、切替直後の 1000 更新（前半）と 1000–6000 更新（後半）に分け、それぞれを P/B/V に割った平均（ユニット・タスクあたり、Adam を通した分解）。",
       "上/下 = 上向きの総移動 ÷ 下向きの総移動（1 に近いほど行き来が多い）。t1 は切替が無いので除く。",""]
    M+=burst200(D)
    allrows=[]
    for act in ("GELU","SILU"):
        for env in ("RL","PM","K784","K107"):
            lines,rows=table(D,act,env,DB);allrows+=rows;M+=[f"## {act} {env}",""]+lines+[""]
    for act in ("LR","ELU1","R"):
        for env in ("RL","PM"):
            lines,rows=table(D,act,env,ZB,rel=False);allrows+=rows;M+=[f"## {act} {env}（z̄ で分ける・B∪V は名目 −1 の分割）",""]+lines+[""]
    (OUT/"posthoc_phases_0915.md").write_text("\n".join(M)+"\n",encoding="utf-8")
    with open(OUT/"posthoc_phases_0915.csv","w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    print("\n".join(M))

if __name__=="__main__":main()
