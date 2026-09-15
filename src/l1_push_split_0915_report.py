#!/usr/bin/env python3
"""Verdict and report for l1_push_split_0915 (spec_l1_push_split_0915.md).
Kept apart from the training module so the training file's hash in provenance.json stays valid.
    python3 -m src.l1_push_split_0915_report [--out-dir results/l1_push_split_0915]
"""
from __future__ import annotations
import argparse, csv, io, json, math, subprocess
from pathlib import Path
import numpy as np
from src import relu_gelu_silu_rl_0914 as B
from src import perm_dial_rl_0915 as DIAL
from src import l1_push_split_0915 as L

ROOT=B.ROOT; NAME=L.NAME; OUT=ROOT/f"results/{NAME}"
EXT150_RESULT="fca473d"; DIAL_RESULT="b0f0bdd"
C={k:i for i,k in enumerate(L.SAVE)}
G=("P","B","V")
BAND=(-3.0,0.0); SUB=((-1.0,0.0),(-2.0,-1.0),(-3.0,-2.0))
DBINS=((1.0,3.0),(0.0,1.0),(-1.0,0.0),(-2.0,-1.0),(-3.0,-2.0),(-6.0,-3.0),(-12.0,-6.0),(-np.inf,-12.0))
ZBINS=((1.0,np.inf),(0.0,1.0),(-1.0,0.0),(-2.0,-1.0),(-4.0,-2.0),(-8.0,-4.0),(-16.0,-8.0),(-np.inf,-16.0))
EXT_MODELS=[dict(seed=s,env=e,act=a) for s in range(3) for e in ("RL","PM") for a in B.ACTS]

def gitbytes(commit,path):return subprocess.check_output(["git","show",f"{commit}:{path}"],cwd=ROOT)
def npzdict(b):
    z=np.load(io.BytesIO(b) if isinstance(b,bytes) else b);return {k:z[k] for k in z.files}
def lab(lo,hi):return f"[{lo:g},{hi:g})"
def fmt(x,p=3):return "—" if x is None or (isinstance(x,float) and not math.isfinite(x)) else f"{x:+.{p}f}"
def sci(x):return "—" if x is None or not math.isfinite(x) else f"{x:+.2e}"

class Data:
    def __init__(self,out):
        self.out=out
        self.ch=np.load(out/"logs/chunks.npy")                       # (K, M, 100, len(SAVE))
        meta=list(csv.DictReader(open(out/"chunk_meta.csv")))
        self.task=np.array([int(r["task"]) for r in meta]);self.start=np.array([int(r["start"]) for r in meta])
        self.end=np.array([int(r["end"]) for r in meta]);self.n=(self.end-self.start).astype(np.float64)
        self.zs=np.load(out/"logs/zstart.npy");self.T=self.zs.shape[0]
        assert self.ch.shape[0]==len(meta) and self.ch.shape[1]==len(L.MODELS)
        self.zbeg=np.empty(self.ch.shape[:3],np.float64)
        for c in range(len(meta)):
            self.zbeg[c]=self.zs[self.task[c]-1] if self.start[c]==0 else self.ch[c-1,:,:,C["zend"]]
        self.last={t:int(np.where(self.task==t)[0].max()) for t in range(1,self.T+1)}
    def j(self,seed,env,act):return L.MODELS.index(dict(seed=seed,env=env,act=act))
    def col(self,j,name):return self.ch[:,j,:,C[name]].astype(np.float64)
    def d(self,j):return self.zbeg[:,j,:]-L.ZC.get(L.MODELS[j]["act"],L.ZC_NOMINAL)
    def okD(self,j):return self.ch[:,j,:,C["ratioD"]]<=1
    def okF(self,j):return self.ch[:,j,:,C["ratioF"]]<=1

def rates(D,js,masks,kind="dz"):
    """Pooled per-1000-update rates of the three groups over (model j, mask) pairs.  kind 'dz' (Adam-exact D) or 'f' (raw force)."""
    num={g:0. for g in G};steps=0.;cnt=0
    extra=dict(tv=0.,net=0.,occ={g:0. for g in G},fR={g:0. for g in G},fI={g:0. for g in G})
    for j,m in zip(js,masks):
        if not m.any():continue
        nn=np.broadcast_to(D.n[:,None],m.shape)
        steps+=float(nn[m].sum());cnt+=int(m.sum())
        for g in G:
            if kind=="dz":num[g]+=float(D.col(j,"dz"+g)[m].sum())
            else:num[g]+=float((D.col(j,"fR"+g)+D.col(j,"fI"+g))[m].sum())
            extra["occ"][g]+=float(D.col(j,"n"+g)[m].sum())
            extra["fR"][g]+=float(D.col(j,"fR"+g)[m].sum());extra["fI"][g]+=float(D.col(j,"fI"+g)[m].sum())
        extra["tv"]+=float(D.col(j,"tv")[m].sum());extra["net"]+=float(D.col(j,"dzA")[m].sum())
    if steps==0:return None
    r={g:1000*num[g]/steps for g in G}
    r["net"]=sum(r[g] for g in G);r["n"]=cnt;r["steps"]=steps
    r["occ"]={g:extra["occ"][g]/(steps*B.BATCH) for g in G}
    r["fR"]={g:1000*extra["fR"][g]/steps for g in G};r["fI"]={g:1000*extra["fI"][g]/steps for g in G}
    tv,net=extra["tv"],extra["net"]
    r["tv"]=1000*tv/steps;r["netA"]=1000*net/steps;r["updown"]=(tv+net)/(tv-net) if tv>net else float("inf")
    r["rho"]=-(r["B"]+r["V"])/r["P"] if r["P"]!=0 else float("nan")
    return r

def band_mask(D,j,lo,hi,kind="dz"):
    d=D.d(j);return (d>=lo)&(d<hi)&(D.okD(j) if kind=="dz" else D.okF(j))

# ------------------------------------------------------------------ gates
def gate_g1(D,out):
    new=npzdict(out/"units.npz")
    ext=npzdict(gitbytes(EXT150_RESULT,"results/relu_gelu_silu_rl_ext150_0914/units.npz"))
    dia=npzdict(gitbytes(DIAL_RESULT,"results/perm_dial_rl_0915/units.npz"))
    ekeys=sorted({k.rsplit("_t",1)[0] for k in ext});cn=0.;cn_n=0;other={}
    for j,m in enumerate(L.MODELS):
        if m["env"] in ("RL","PM"):ref,jj,keys=ext,EXT_MODELS.index(m),ekeys
        else:ref,jj,keys=dia,DIAL.MODELS.index(dict(seed=m["seed"],k=int(m["env"][1:]),act=m["act"])),list(DIAL.KEEP_UNITS)
        for t in range(1,D.T+1):
            for k in keys:
                dd=float(np.abs(new[f"{k}_t{t}"][j].astype(np.float64)-ref[f"{k}_t{t}"][jj].astype(np.float64)).max())
                if k=="cnorm_l1":cn=max(cn,dd);cn_n+=100
                else:other[k]=max(other.get(k,0.),dd)
    rows=list(csv.DictReader(open(out/"rows.csv")))
    er={(int(r["seed"]),r["env"],r["act"],int(r["task"])):r for r in csv.DictReader(io.StringIO(
        gitbytes(EXT150_RESULT,"results/relu_gelu_silu_rl_ext150_0914/rows.csv").decode()))}
    dr={(int(r["seed"]),int(r["k"]),r["act"],int(r["task"])):r for r in csv.DictReader(io.StringIO(
        gitbytes(DIAL_RESULT,"results/perm_dial_rl_0915/rows.csv").decode()))}
    rw=0.;rn=0
    for r in rows:
        s,e,a,t=int(r["seed"]),r["env"],r["act"],int(r["task"])
        o=er[s,e,a,t] if e in ("RL","PM") else dr[s,int(e[1:]),a,t]
        for k in ("online_acc","train_ce"):rw=max(rw,abs(float(r[k])-float(o[k])));rn+=1
    exp_rows=len(L.MODELS)*D.T*2;exp_cn=len(L.MODELS)*D.T*100
    ok=cn==0 and rw==0 and rn==exp_rows and cn_n==exp_cn
    return dict(verdict="BIT_IDENTICAL" if ok else "DIFFERS",cnorm_max=cn,cnorm_values=cn_n,rows_max=rw,rows_values=rn,
                eval_stats_max=other)

def gate_g2(D):
    rD=D.ch[...,C["ratioD"]];rF=D.ch[...,C["ratioF"]]
    return dict(exceed_D=int((rD>1).sum()),exceed_F=int((rF>1).sum()),total=int(rD.size),
                max_ratio_D=float(np.nanmax(rD)),max_ratio_F=float(np.nanmax(rF)))

# ------------------------------------------------------------------ verdicts
def q1q2(D,act,kind="dz"):
    per=[]
    for s in range(3):
        j=D.j(s,"RL",act);r=rates(D,[j],[band_mask(D,j,*BAND,kind)],kind);per.append(r)
    signs=[r["V"] for r in per]
    q1="V_DOWN" if all(x<0 for x in signs) else "V_UP" if all(x>0 for x in signs) else "V_MIXED"
    sink=[]
    for r in per:
        g=min(G,key=lambda g:r[g]);sink.append(g if r[g]<0 else "NONE")
    q2=f"SINK_BY_{sink[0]}" if sink[0]!="NONE" and all(x==sink[0] for x in sink) else "SINK_MIXED"
    return q1,q2,per,sink

def q3(D,kind="dz"):
    avg={}
    for act in ("GELU","SILU"):
        js=[D.j(s,"RL",act) for s in range(3)];sub=[]
        for lo,hi in SUB:sub.append(rates(D,js,[band_mask(D,j,lo,hi,kind) for j in js],kind))
        avg[act]={g:float(np.mean([r[g] for r in sub if r is not None])) for g in G}
        a=avg[act];a["net"]=a["P"]+a["B"]+a["V"];a["rho"]=-(a["B"]+a["V"])/a["P"];a["sub"]=sub
    g_,s_=avg["GELU"],avg["SILU"]
    pre=g_["P"]<0 and s_["P"]<0 and g_["net"]<0 and s_["net"]<0 and g_["net"]<s_["net"]
    if not pre:return "Q3_NOT_APPLICABLE",avg,None,None
    Lp=math.log(g_["P"]/s_["P"]);Lr=math.log((1-g_["rho"])/(1-s_["rho"]))
    return ("GAP_FROM_RESISTANCE" if Lr>Lp else "GAP_FROM_PUSH"),avg,Lp,Lr

# ------------------------------------------------------------------ report tables
def table_bins(D,act,env):
    js=[D.j(s,env,act) for s in range(3)];L_=[]
    L_.append(f"|d = z̄ − z_c|unit-chunk|r_P|r_B|r_V|net|ρ|占有 P/B/V|上/下 総量|F_P 減/増|F_B 減/増|F_V 減/増|")
    L_.append("|---|---:|---:|---:|---:|---:|---:|---|---:|---|---|---|")
    out=[]
    for lo,hi in DBINS:
        r=rates(D,js,[band_mask(D,j,lo,hi) for j in js])
        if r is None:L_.append(f"|{lab(lo,hi)}|0|||||||||||");continue
        out.append(dict(act=act,env=env,lo=lo,hi=hi,n=r["n"],rP=r["P"],rB=r["B"],rV=r["V"],net=r["net"],rho=r["rho"],
                        occP=r["occ"]["P"],occB=r["occ"]["B"],occV=r["occ"]["V"],updown=r["updown"],
                        **{f"fR{g}":r["fR"][g] for g in G},**{f"fI{g}":r["fI"][g] for g in G}))
        L_.append(f"|{lab(lo,hi)}|{r['n']}|{fmt(r['P'])}|{fmt(r['B'])}|{fmt(r['V'])}|{fmt(r['net'])}|{fmt(r['rho'],2)}|"
                  f"{r['occ']['P']:.2f}/{r['occ']['B']:.2f}/{r['occ']['V']:.2f}|{r['updown']:.3f}|"
                  f"{sci(r['fR']['P'])} / {sci(r['fI']['P'])}|{sci(r['fR']['B'])} / {sci(r['fI']['B'])}|{sci(r['fR']['V'])} / {sci(r['fI']['V'])}|")
    return L_,out

def table_monotone(D,act,env):
    js=[D.j(s,env,act) for s in range(3)];L_=["|z̄|unit-chunk|r_P|r_B∪V|net|占有 P|F_P 減/増|F_B∪V 減/増|","|---|---:|---:|---:|---:|---:|---|---|"]
    for lo,hi in ZBINS:
        ms=[(D.zbeg[:,j,:]>=lo)&(D.zbeg[:,j,:]<hi)&D.okD(j) for j in js];r=rates(D,js,ms)
        if r is None:continue
        L_.append(f"|{lab(lo,hi)}|{r['n']}|{fmt(r['P'])}|{fmt(r['B']+r['V'])}|{fmt(r['net'])}|{r['occ']['P']:.2f}|"
                  f"{sci(r['fR']['P'])} / {sci(r['fI']['P'])}|{sci(r['fR']['B']+r['fR']['V'])} / {sci(r['fI']['B']+r['fI']['V'])}|")
    return L_

def table_crossing(D,act,seeds=(0,1,2),tasks=(1,2,3),group=3):
    L_=["|task|更新|median z̄ 末|r_P|r_B|r_V|net|占有 P/B/V|","|---|---|---:|---:|---:|---:|---:|---|"]
    js=[D.j(s,"RL",act) for s in seeds]
    for t in tasks:
        idx=np.where(D.task==t)[0]
        for k in range(0,len(idx),group):
            cs=idx[k:k+group]
            ms=[]
            for j in js:
                m=np.zeros(D.ch.shape[0:1]+(100,),bool);m[cs]=D.okD(j)[cs];ms.append(m)
            r=rates(D,js,ms)
            zend=np.median(np.concatenate([D.ch[cs[-1],j,:,C["zend"]] for j in js]))
            L_.append(f"|{t}|{D.start[cs[0]]}–{D.end[cs[-1]]}|{zend:.2f}|{fmt(r['P'])}|{fmt(r['B'])}|{fmt(r['V'])}|{fmt(r['net'])}|"
                      f"{r['occ']['P']:.2f}/{r['occ']['B']:.2f}/{r['occ']['V']:.2f}|")
    return L_

def table_crossings(D,act,env):
    """unit-chunks whose z̄ crosses z_c inside the chunk: sums of D_g (not rates)."""
    zc=L.ZC[act];res=[]
    for direction in ("up","down"):
        tot={g:0. for g in G};net=0.;cnt=0
        for s in range(3):
            j=D.j(s,env,act);zb=D.zbeg[:,j,:];ze=D.ch[:,j,:,C["zend"]].astype(np.float64)
            m=((zb<zc)&(ze>=zc) if direction=="up" else (zb>=zc)&(ze<zc))&D.okD(j)
            cnt+=int(m.sum())
            for g in G:tot[g]+=float(D.col(j,"dz"+g)[m].sum())
            net+=float(D.col(j,"dzA")[m].sum())
        res.append((direction,cnt,tot,net))
    return res

def table_kick(D,act,env):
    zc=L.ZC[act];rows_=[]
    for s in range(3):
        j=D.j(s,env,act)
        for t in range(2,D.T+1):
            pre=D.ch[D.last[t-1],j,:,C["zend"]].astype(np.float64);st=D.zs[t-1,j,:].astype(np.float64)
            end=D.ch[D.last[t],j,:,C["zend"]].astype(np.float64);cs=np.where(D.task==t)[0]
            within={g:D.ch[cs,j,:,C["dz"+g]].astype(np.float64).sum(0) for g in G}
            rows_.append((pre-zc,st-pre,end-st,within))
    dpre=np.concatenate([r[0] for r in rows_]);J=np.concatenate([r[1] for r in rows_]);W=np.concatenate([r[2] for r in rows_])
    Wg={g:np.concatenate([r[3][g] for r in rows_]) for g in G}
    L_=["|前タスク末の d|unit-task|跳び 平均|跳び sd|タスク内 平均|うち P|うち B|うち V|","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for lo,hi in DBINS:
        m=(dpre>=lo)&(dpre<hi)
        if not m.any():continue
        L_.append(f"|{lab(lo,hi)}|{int(m.sum())}|{fmt(float(J[m].mean()))}|{float(J[m].std()):.3f}|{fmt(float(W[m].mean()))}|"
                  f"{fmt(float(Wg['P'][m].mean()))}|{fmt(float(Wg['B'][m].mean()))}|{fmt(float(Wg['V'][m].mean()))}|")
    return L_,float(np.abs(J).max())

def report(out_dir=OUT):
    out=B.output_path(out_dir);D=Data(out)
    g1=gate_g1(D,out);g2=gate_g2(D);tag="" if g1["verdict"]=="BIT_IDENTICAL" else " (TRAJECTORY_DIFFERS)"
    V=[dict(section="G1",verdict=g1["verdict"],value=g1["cnorm_max"],detail=json.dumps({k:v for k,v in g1.items() if k!="verdict"})),
       dict(section="G2",verdict="EXCEED" if g2["exceed_D"] or g2["exceed_F"] else "CLOSED",detail=json.dumps(g2))]
    res={}
    for act in ("GELU","SILU"):
        q1_,q2_,per,sink=q1q2(D,act);res[act]=(q1_,q2_,per,sink)
        V.append(dict(section="Q1",arm=act,verdict=q1_+tag,detail=" ".join(f"{r['V']:+.4g}" for r in per)))
        V.append(dict(section="Q2",arm=act,verdict=q2_+tag,detail=" ".join(sink)))
    q3_,avg,Lp,Lr=q3(D);V.append(dict(section="Q3",verdict=q3_+tag,detail=json.dumps(dict(L_push=Lp,L_res=Lr))))
    resF={act:q1q2(D,act,"f") for act in ("GELU","SILU")};q3F=q3(D,"f")
    for act in ("GELU","SILU"):
        V.append(dict(section="Q1_F_report",arm=act,verdict=resF[act][0],detail=" ".join(f"{r['V']:+.4g}" for r in resF[act][2])))
        V.append(dict(section="Q2_F_report",arm=act,verdict=resF[act][1],detail=" ".join(resF[act][3])))
    V.append(dict(section="Q3_F_report",verdict=q3F[0],detail=json.dumps(dict(L_push=q3F[2],L_res=q3F[3]))))
    B.csvwrite(out/"verdict.csv",V)

    M=[f"# {NAME} report","",
       f"G1: **{g1['verdict']}**（cnorm_l1 max|Δ| {g1['cnorm_max']:.3g} / {g1['cnorm_values']} 値、rows max|Δ| {g1['rows_max']:.3g} / {g1['rows_values']} 値）  ",
       "評価統計（報告のみ）max|Δ|: "+", ".join(f"{k} {v:.2g}" for k,v in sorted(g1["eval_stats_max"].items())),"",
       f"G2: 閉包の超過 D {g2['exceed_D']} / F {g2['exceed_F']}（全 {g2['total']} ユニット・チャンク、最大比 D {g2['max_ratio_D']:.3g}・F {g2['max_ratio_F']:.3g}）","",
       "## 判定（RL・帯 d ∈ [−3, 0)・率は 1000 更新あたりの Δz̄）","",
       "|活性化|Q1|Q2|seed 別 r_P / r_B / r_V（net）|","|---|---|---|---|"]
    for act in ("GELU","SILU"):
        q1_,q2_,per,sink=res[act]
        M.append(f"|{act}|**{q1_}{tag}**|**{q2_}{tag}**|"+"<br>".join(f"s{s}: {fmt(r['P'])} / {fmt(r['B'])} / {fmt(r['V'])}（{fmt(r['net'])}）" for s,r in enumerate(per))+"|")
    M+=["",f"**Q3: {q3_}{tag}**"+(f"（L_push = {Lp:+.3f}、L_res = {Lr:+.3f}）" if Lp is not None else ""),"",
        "|活性化|r_P|r_B|r_V|net|ρ|","|---|---:|---:|---:|---:|---:|"]
    for act in ("GELU","SILU"):
        a=avg[act];M.append(f"|{act}|{fmt(a['P'])}|{fmt(a['B'])}|{fmt(a['V'])}|{fmt(a['net'])}|{fmt(a['rho'],3)}|")
    M+=["","### 同じ判定を生の力 F（Adam を通さない）で（報告のみ）","",
        f"Q1: GELU {resF['GELU'][0]}・SiLU {resF['SILU'][0]}／Q2: GELU {resF['GELU'][1]}・SiLU {resF['SILU'][1]}／Q3: {q3F[0]}"+
        (f"（L_push {q3F[2]:+.3f}・L_res {q3F[3]:+.3f}）" if q3F[2] is not None else ""),""]
    binrows=[]
    M+=["## 深さ別の 3 群（報告のみ）","","r は 1000 更新あたりの Δz̄（Adam を通した分解）、F は 1000 更新あたりのバイアスへの力（上向き正、減 = u>0、増 = u≤0）、上/下 総量 = (TV+net)/(TV−net)。",""]
    for act in ("GELU","SILU"):
        for env in ("RL","PM","K784","K107"):
            t_,o_=table_bins(D,act,env);binrows+=o_;M+=[f"### {act} {env}",""]+t_+[""]
    B.csvwrite(out/"binned.csv",binrows)
    M+=["## 谷の無い活性化（z̄ 別・P 対 B∪V・報告のみ）",""]
    for act in ("LR","ELU1","R"):
        for env in ("RL","PM","K784"):
            if act=="R" and env=="K784":continue
            M+=[f"### {act} {env}",""]+table_monotone(D,act,env)+[""]
    M+=["## RL の谷越え（t1–3・600 更新ごと・3 seed プール・報告のみ）",""]
    for act in ("GELU","SILU"):M+=[f"### {act}",""]+table_crossing(D,act)+[""]
    M+=["### SiLU seed 別（t1–3）",""]
    for s in range(3):M+=[f"#### SiLU seed {s}",""]+table_crossing(D,"SILU",seeds=(s,))+[""]
    M+=["## チャンク内で z_c を跨いだユニット・チャンク（D_g の和・報告のみ）","","|活性化|環境|向き|件数|Σ D_P|Σ D_B|Σ D_V|Σ 実際|","|---|---|---|---:|---:|---:|---:|---:|"]
    for act in ("GELU","SILU"):
        for env in ("RL","PM","K784","K107"):
            for direction,cnt,tot,net in table_crossings(D,act,env):
                M.append(f"|{act}|{env}|{'上' if direction=='up' else '下'}|{cnt}|{fmt(tot['P'])}|{fmt(tot['B'])}|{fmt(tot['V'])}|{fmt(net)}|")
    M+=["","## 蹴り: タスク境界の跳びとタスク内の学習（報告のみ）",""]
    for act in ("GELU","SILU"):
        for env in ("RL","PM","K784","K107"):
            t_,jmax=table_kick(D,act,env);M+=[f"### {act} {env}（|跳び| の最大 {jmax:.3g}）",""]+t_+[""]
    (out/"summary.md").write_text("\n".join(M)+"\n",encoding="utf-8")
    print("\n".join(M[:40]),flush=True)

if __name__=="__main__":
    ap=argparse.ArgumentParser();ap.add_argument("--out-dir",default=f"results/{NAME}");a=ap.parse_args()
    report(a.out_dir)
