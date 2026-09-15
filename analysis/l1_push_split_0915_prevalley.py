#!/usr/bin/env python3
"""Post-hoc (not registered) for l1_push_split_0915, RL arms.  Units that START a task before the valley
(P-unit: z̄>0, B-unit: z_c<z̄<=0; monotone activations use 0 / nominal -1): is the task's net Δz̄ finished in
the first chunk after the label switch (200 updates for t<=10, 100 for t>=11)?
  (1) cumulative Δz̄ at chunk edges by start position;
  (2) the same units split by whether they crossed the valley within the first chunk, with the P/B/V
      decomposition of first chunk and rest, per-unit coherence |Δz̄|/Σ|Δz̄|, sample occupancy, raw force R/I;
  (3) V-units by start depth: Adam bias denominator per update and coherence.
Reads logs/chunks.npy (falls back to the backup path in backup_manifest.json).
    python3 -m analysis.l1_push_split_0915_prevalley
"""
from __future__ import annotations
import json
import numpy as np
from src import l1_push_split_0915 as L
from src import l1_push_split_0915_report as R

OUT=R.OUT; C=R.C; G=("P","B","V")
GROUPS=(("GELU",(0,1,2),"s0–2"),("SILU",(0,1),"s0–1"),("SILU",(2,),"s2"),("LR",(0,1,2),"s0–2"),("ELU1",(0,1,2),"s0–2"),("R",(0,1,2),"s0–2"))
RANGES=((2,10),(11,50),(51,150))
EDGES={True:(200,400,1000,2000,6000),False:(100,300,1000,3000,6000)}
DEPTH=((-3,0),(-6,-3),(-12,-6),(-24,-12),(-np.inf,-24))

class Data(R.Data):
    def __init__(self,out):
        if not (out/"logs/chunks.npy").exists():                      # raw log was moved out of the repo (§4 of CLAUDE.md)
            import shutil,tempfile;from pathlib import Path
            src=[e for e in json.load(open(out/"backup_manifest.json"))["files"] if e["source"].endswith("chunks.npy")][0]["backup"]
            tmp=Path(tempfile.mkdtemp());(tmp/"logs").mkdir();(tmp/"logs/chunks.npy").symlink_to(src)
            (tmp/"logs/zstart.npy").symlink_to(out/"logs/zstart.npy");shutil.copy(out/"chunk_meta.csv",tmp/"chunk_meta.csv");out=tmp
        super().__init__(out)

def collect(D,act,seeds,t_lo,t_hi):
    """Per unit-task arrays over tasks t_lo..t_hi (RL).  cum/zedge are (N, len(EDGES)); *g are (N,3)."""
    zc=L.ZC.get(act,L.ZC_NOMINAL);edges=EDGES[t_hi<=L.FINE_TASKS]
    o={k:[] for k in ("zst","d0","cum","zedge","first","rest","z1","zend","tv1","tvr","occ1","fR1","fI1","firstg","restg","den1","denr","n1","nr")}
    for s in seeds:
        j=D.j(s,"RL",act)
        for t in range(t_lo,t_hi+1):
            cs=np.where(D.task==t)[0];ch=D.ch[cs,j].astype(np.float64);dzA=ch[:,:,C["dzA"]];cumA=np.cumsum(dzA,0)
            idx=[int(np.where(D.end[cs]==e)[0][0]) for e in edges]
            o["zst"].append(D.zs[t-1,j]);o["d0"].append(D.zs[t-1,j]-zc);o["cum"].append(cumA[idx].T);o["zedge"].append(ch[idx,:,C["zend"]].T)
            o["first"].append(dzA[0]);o["rest"].append(dzA[1:].sum(0));o["z1"].append(ch[0,:,C["zend"]]);o["zend"].append(ch[-1,:,C["zend"]])
            o["tv1"].append(ch[0,:,C["tv"]]);o["tvr"].append(ch[1:,:,C["tv"]].sum(0))
            o["occ1"].append(np.stack([ch[0,:,C["n"+g]] for g in G],1)/(D.n[cs[0]]*16))
            o["fR1"].append(np.stack([ch[0,:,C["fR"+g]] for g in G],1));o["fI1"].append(np.stack([ch[0,:,C["fI"+g]] for g in G],1))
            o["firstg"].append(np.stack([ch[0,:,C["dz"+g]] for g in G],1));o["restg"].append(np.stack([ch[1:,:,C["dz"+g]].sum(0) for g in G],1))
            o["den1"].append(ch[0,:,C["denb"]]);o["denr"].append(ch[1:,:,C["denb"]].sum(0))
            o["n1"].append(np.full(100,D.n[cs[0]]));o["nr"].append(np.full(100,D.n[cs[1:]].sum()))
    return {k:np.concatenate(v) for k,v in o.items()},zc,edges

def classes(U,act,zc):
    zst=U["zst"];thr=zc if act in L.ZC else 0.0
    return thr,(("P-unit z̄>0",zst>0),(f"B-unit ({zc:.2f},0]" if act in L.ZC else "(−1,0]",(zst>zc)&(zst<=0)),("V-unit ≤z_c" if act in L.ZC else "≤−1",zst<=zc))

def coh(x,tv):return float(np.median(np.abs(x)/np.maximum(tv,1e-30)))
def pbv(a):return f"{a[0]:+.2f} / {a[1]:+.2f} / {a[2]:+.2f}"

def table1(D):
    M=["## 1. 開始時の位置別: 累積 Δz̄（各区切り）","","値はユニット・タスクあたりの平均。一貫性 = |Σ Δz̄| ÷ Σ|Δz̄|（ユニット・タスクごとに取った中央値、1 なら一方向）。den = Adam のバイアス分母 √v̂+ε の更新平均。",""]
    for act,seeds,slab in GROUPS:
        for t_lo,t_hi in RANGES:
            U,zc,edges=collect(D,act,seeds,t_lo,t_hi);thr,cls=classes(U,act,zc)
            M+=[f"### {act} {slab} t{t_lo}–{t_hi}（区切り {edges}）","",
                "|開始時の位置|unit-task|累積 Δz̄|中央値 最初 / 全体|最初 P/B/V|残り P/B/V|一貫性 最初 / 残り|den/更新 最初 / 残り|"+("谷越え率" if act in L.ZC else "z̄≤0 率")+"（各区切り）|",
                "|---|---:|---|---|---|---|---|---|---|"]
            for lab,m in cls:
                n=int(m.sum())
                if n==0:M.append(f"|{lab}|0||||||||");continue
                f,r=U["first"][m],U["rest"][m]
                M.append(f"|{lab}|{n}|{' / '.join(f'{c:+.2f}' for c in U['cum'][m].mean(0))}|{np.median(f):+.2f} / {np.median(f+r):+.2f}|{pbv(U['firstg'][m].mean(0))}|{pbv(U['restg'][m].mean(0))}|"
                         f"{coh(f,U['tv1'][m]):.2f} / {coh(r,U['tvr'][m]):.2f}|{U['den1'][m].sum()/U['n1'][m].sum():.1e} / {U['denr'][m].sum()/U['nr'][m].sum():.1e}|"
                         f"{' / '.join(f'{c:.2f}' for c in (U['zedge'][m]<=thr).mean(0))}|")
            M.append("")
    return M

def table2(D):
    M=["## 2. 最初の区切りで谷を越えたか否かで分ける","","符号一致 = 最初の区切りの Δz̄ と全体の Δz̄ の符号が同じ unit-task の割合。標本占有は最初の区切りの更新×標本を P/B/V で数えた割合。生の力 = −Σφ′u をバイアス勾配で見た和（減らせ u>0 / 増やせ u≤0）。",""]
    H=["|開始時の位置|unit-task|Δz̄ 最初 / 残り|中央値 全体÷最初|符号一致|一貫性 最初 / 残り|標本占有 P/B/V|最初 P/B/V|残り P/B/V|最初 生の力 減らせ/増やせ P・B・V|越え率 最初 → 末|","|---|---:|---|---:|---:|---|---|---|---|---|---|"]
    for act,seeds,slab,(t_lo,t_hi) in (("GELU",(0,1,2),"s0–2",(2,10)),("SILU",(0,1),"s0–1",(2,10)),("SILU",(0,1),"s0–1",(11,50)),("SILU",(2,),"s2",(2,10)),
                                       ("LR",(0,1,2),"s0–2",(2,10)),("LR",(0,1,2),"s0–2",(11,150)),("ELU1",(0,1,2),"s0–2",(2,10)),("R",(0,1,2),"s0–2",(2,10))):
        U,zc,_=collect(D,act,seeds,t_lo,t_hi);thr,cls=classes(U,act,zc);c1=U["z1"]<=thr
        M+=[f"### {act} {slab} t{t_lo}–{t_hi}（最初の区切り = {200 if t_hi<=L.FINE_TASKS else 100} 更新）",""]+H
        for lab,m in cls[:2]:
            for sub,mm in ((lab,m),("　うち最初で越えた",m&c1),("　うち越えず",m&~c1)):
                n=int(mm.sum())
                if n==0:M.append(f"|{sub}|0||||||||||");continue
                f,r=U["first"][mm],U["rest"][mm];full=f+r
                q=full/np.where(np.abs(f)>0,f,np.nan);ratio=float(np.nanmedian(q)) if np.isfinite(q).any() else float("nan");same=float(np.mean(np.sign(f)==np.sign(full)))
                occ=U["occ1"][mm].mean(0);fR=U["fR1"][mm].mean(0);fI=U["fI1"][mm].mean(0)
                M.append(f"|{sub}|{n}|{f.mean():+.2f} / {r.mean():+.2f}|{ratio:+.2f}|{same:.2f}|{coh(f,U['tv1'][mm]):.2f} / {coh(r,U['tvr'][mm]):.2f}|"
                         f"{occ[0]:.2f}/{occ[1]:.2f}/{occ[2]:.2f}|{pbv(U['firstg'][mm].mean(0))}|{pbv(U['restg'][mm].mean(0))}|"
                         f"{fR[0]:+.1f}/{fI[0]:+.1f}・{fR[1]:+.1f}/{fI[1]:+.1f}・{fR[2]:+.1f}/{fI[2]:+.1f}|{float((U['z1'][mm]<=thr).mean()):.2f} → {float((U['zend'][mm]<=thr).mean()):.2f}|")
        M.append("")
    return M

def table3(D):
    M=["## 3. 谷の奥のユニット（開始時の深さ別・t11–150）: Adam のバイアス分母・一貫性・往復","",
       "|活性化|d₀ = z̄−z_c|unit-task|den/更新 最初 / 残り|Δz̄ 最初 / 残り|一貫性 最初 / 残り|Σ\\|Δz̄\\|/更新 残り|","|---|---|---:|---|---|---|---:|"]
    for act,seeds in (("GELU",(0,1,2)),("SILU",(2,))):
        U,zc,_=collect(D,act,seeds,11,150);d0=U["d0"]
        for lo,hi in DEPTH:
            m=(d0>=lo)&(d0<hi)
            if m.sum()==0:continue
            M.append(f"|{act}|[{lo:g},{hi:g})|{int(m.sum())}|{U['den1'][m].sum()/U['n1'][m].sum():.1e} / {U['denr'][m].sum()/U['nr'][m].sum():.1e}|"
                     f"{U['first'][m].mean():+.2f} / {U['rest'][m].mean():+.2f}|{coh(U['first'][m],U['tv1'][m]):.2f} / {coh(U['rest'][m],U['tvr'][m]):.2f}|{U['tvr'][m].sum()/U['nr'][m].sum():.1e}|")
    return M+[""]

def main():
    D=Data(OUT)
    M=["# l1_push_split_0915 事後（登録外）: 谷の手前から始まるユニットの正味は切替直後で済むか","",
       "RL の腕。開始時の位置はタスク開始時の z̄（GELU z_c=−0.75・SiLU z_c=−1.28、谷の無い活性化は 0 と名目 −1 で分ける）。最初の区切り = 切替直後の 200 更新（t≤10）／100 更新（t≥11）。",""]
    M+=table1(D)+table2(D)+table3(D)
    (OUT/"posthoc_prevalley_0915.md").write_text("\n".join(M)+"\n",encoding="utf-8");print("\n".join(M))

if __name__=="__main__":main()
