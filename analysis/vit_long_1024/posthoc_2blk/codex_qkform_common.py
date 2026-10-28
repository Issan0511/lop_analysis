"""Shared, bounded-memory diagnostics for the second q/k formation request."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix.json')
import hashlib
import json
import math
import resource
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.nn import functional as F
from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk.qk_projected import tokens

ROOT = Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix')
OUT = ROOT / 'posthoc_qk_formation'
SEED = 10252010
TASKS = list(range(11)) + list(range(20,101,10))
D,H,DH = 384,6,64

def setup():
    C.set_determinism(2)
    torch.cuda.set_per_process_memory_fraction(.40)
    assert os.environ.get('VIT_LONG_PRODUCTION') == 'production_2blk_rlfix.json'

def stamp():
    return datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds')

def progress(message):
    with (OUT/'PROGRESS.md').open('a') as f:
        f.write(f'\n## {stamp()}\n\n{message}\n')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def track(path):
    dest=OUT/'codex_input_hashes.json'
    rec=json.loads(dest.read_text()) if dest.exists() else {}
    key=str(path)
    if key not in rec:
        rec[key]={'sha256':sha(path),'bytes':Path(path).stat().st_size,'first_read':stamp()}
        dest.write_text(json.dumps(rec,indent=2))

def ckpt(arm,t,kind='model'):
    return ROOT/'runs'/arm/f'seed{SEED}/ckpt/{kind}_t{t:02d}.pt'

def load_model(arm,t):
    path=ckpt(arm,t);track(path)
    model=C.build_arm_model(arm,SEED,'cuda',0.,2,8.)
    model.load_state_dict(torch.load(path,map_location='cpu',weights_only=False))
    model.eval()
    return model

def resource_record():
    out=dict(rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
             cuda_peak_mb=torch.cuda.max_memory_allocated()/2**20,
             cuda_reserved_mb=torch.cuda.max_memory_reserved()/2**20)
    assert out['rss_mb'] < 3072, out
    assert out['cuda_reserved_mb'] < 7600, out
    return out

def write_new(path, data):
    with Path(path).open('x') as f:json.dump(data,f,indent=2,allow_nan=False)

def sq(x):return float(x.double().square().sum())

def norm(x):return math.sqrt(sq(x))

def inner(x,y):return float((x.double()*y.double()).sum())

def pr(M):
    s=torch.linalg.svdvals(M.double())
    return float(s.sum().square()/s.square().sum().clamp_min(1e-100))

def cos(x,y):return inner(x,y)/max(norm(x)*norm(y),1e-100)

@torch.no_grad()
def parts(model):
    block=model.layers['block_0'].layers
    qkv=block['attn'].layers['qkv'];ln=block['norm1']
    Q=qkv.weight[:D].detach().reshape(H,DH,D).clone()
    K=qkv.weight[D:2*D].detach().reshape(H,DH,D).clone()
    g=ln.weight.detach().clone();beta=ln.bias.detach().clone()
    B=K.transpose(-2,-1)@Q/8
    A=B*g[None,:,None]*g[None,None,:]
    b=(K.transpose(-2,-1)@(Q@beta+qkv.bias[:D].reshape(H,DH))[...,None]).squeeze(-1)*g/8
    return dict(Q=Q,K=K,g=g,beta=beta,B=B,A=A,b=b)

def moment(xi):
    x=xi.reshape(-1,D)
    return x.T@x/len(x),x.mean(0)

def effective(A,Cov,b=None,mean=None):
    v=(A*(A@Cov)).double().sum()/H
    if b is not None:
        v=v+b.double().square().sum()/H+2*((A@mean)*b).double().sum()/H
    return math.sqrt(max(float(v),0.))

@torch.no_grad()
def snapshot(model,x,refmom=None):
    p=parts(model);xi=tokens(model,x);cc,mu=moment(xi)
    cfix=cc if refmom is None else refmom[0]
    Q,K,A,B,b=p['Q'],p['K'],p['A'],p['B'],p['b']
    P=norm(Q)*norm(K)/8
    an,bn=norm(A),norm(B)
    af=effective(A,cfix);ao=effective(A,cc);S=effective(A,cc,b,mu)
    fac=dict(P=P,R=bn/P,G=an/bn,T=af/an,I=ao/af,J=S/ao)
    block=model.layers['block_0'].layers;att=block['attn'];ln=block['norm1']
    z=xi*ln.weight+ln.bias
    q,k,v=att.layers['qkv'](z).reshape(len(x),65,3,H,DH).permute(2,0,3,1,4).unbind(0)
    scores=(q@k.transpose(-2,-1))/8
    attn=scores.softmax(-1)
    ap=A@xi.reshape(-1,D).T+b[...,None]
    direct=(K.transpose(-2,-1)[None]@q.transpose(-2,-1))*p['g'][None,None,:,None]/8
    reconstructed=ap.reshape(H,D,len(x),65).permute(2,0,1,3)
    identity=norm(direct-reconstructed)/max(norm(direct),1e-100)
    scores_center=scores-scores.mean(-1,keepdim=True)
    bscore=torch.einsum('hd,bnd->bhn',b,xi)
    bscore=bscore-bscore.mean(-1,keepdim=True)
    ccpatch,mupatch=moment(xi[:,1:])
    rec=dict(factors=fac,S=S,a_fixed=af,a_own=ao,A_norm=an,B_norm=bn,
             b_norm=norm(b),Q_norm=norm(Q),K_norm=norm(K),
             gamma_mean=float(p['g'].mean()),gamma_rms=norm(p['g'])/math.sqrt(D),
             gamma_anisotropy=fac['G']/(sq(p['g'])/D),
             saturation=float((attn.max(-1).values>.99).float().mean()),
             entropy=float(-(attn*attn.clamp_min(1e-30).log()).sum(-1).mean()),
             score_rms=math.sqrt(sq(scores_center)/scores.numel()),
             bscore_rms=math.sqrt(sq(bscore)/bscore.numel()),
             S_other=effective(A,ccpatch,b,mupatch),identity_relative_l2=identity,
             factor_relative_error=abs(math.prod(fac.values())/S-1))
    return rec,p,(cc,mu),xi

def delta_record(p0,p1,cm,ranks=True):
    dq,dk=p1['Q']-p0['Q'],p1['K']-p0['K']
    da=p1['A']-p0['A']
    g=p0['g']
    tq=(p0['K'].transpose(-2,-1)@dq)/8*g[None,:,None]*g[None,None,:]
    tk=(dk.transpose(-2,-1)@p0['Q'])/8*g[None,:,None]*g[None,None,:]
    t2=(dk.transpose(-2,-1)@dq)/8*g[None,:,None]*g[None,None,:]
    tg=da-tq-tk-t2
    actual=[];legacy=[]
    for h in range(H):
        den=max(norm(p0['K'][h])*norm(dq[h]),1e-100)
        actual.append(norm(p0['K'][h].T@dq[h])/den)
        legacy.append(norm(p0['K'][h]@dq[h].T)/den)
    va0=effective(p0['A'],cm);vda=effective(da,cm)
    rad=float((p0['A']*(da@cm)).double().sum())/H
    dg=p1['g']-g
    tglinear=p0['B']*(dg[None,:,None]*g[None,None,:]+g[None,:,None]*dg[None,None,:])
    def project(v):return float((p0['A']*(v@cm)).double().sum())/(H*max(va0,1e-100))
    rec=dict(dQ=norm(dq),dK=norm(dk),dA=norm(da),da_fixed=vda,
                alignment_correct=float(np.mean(actual)),alignment_legacy=float(np.mean(legacy)),
                radial_A_cos=cos(da,p0['A']),radial_a_cos=rad/max(va0*vda,1e-100),
                A_Q_term=norm(tq),A_K_term=norm(tk),A_quadratic_term=norm(t2),A_gamma_term=norm(tg),
                A_energy_radial=2*inner(p0['A'],da),A_energy_step=sq(da),
                a_energy_radial=2*rad,a_energy_step=vda*vda,
                Q_energy_radial=2*inner(p0['Q'],dq),Q_energy_step=sq(dq),
                K_energy_radial=2*inner(p0['K'],dk),K_energy_step=sq(dk),
                a_firstorder_qk=project(tq+tk),a_firstorder_gamma=project(tglinear),
                a_exact_norm_change=effective(p1['A'],cm)-va0)
    if ranks:rec.update(Q_pr=pr(dq.reshape(D,D)),K_pr=pr(dk.reshape(D,D)))
    return rec

def attention_forward(model,x,temperature=1.):
    """Same node-level forward as codex_bodyrank_decompose, exposing q/k."""
    L=model.layers
    z=L['patch_embed'](x)
    z=torch.cat((model.cls_token.expand(len(x),-1,-1),z),1)+model.pos_embed
    nodes={}
    for i in range(model.depth):
        b=L[f'block_{i}'].layers
        a=b['norm1'](z);att=b['attn'];B,N,D_=a.shape
        q,k,v=att.layers['qkv'](a).reshape(B,N,3,H,DH).permute(2,0,3,1,4).unbind(0)
        score=(q@k.transpose(-2,-1))/8/(temperature if i==0 else 1.)
        prob=score.softmax(-1)
        z=z+att.layers['proj']((prob@v).transpose(1,2).reshape(B,N,D_))
        mlp=b['mlp'].layers
        z=z+mlp['fc2'](mlp['act'](mlp['fc1'](b['norm2'](z))))
        if i==0:nodes=dict(q=q,k=k,score=score,prob=prob,ln=a)
    h=L['norm'](z)[:,0]
    return L['out'](h),nodes,h

def basis_overlap(U,V):
    if U.shape[1]==0 or V.shape[1]==0:
        return dict(cos2=None,angles_deg=[])
    s=torch.linalg.svdvals(U.double().T@V.double()).clamp(0,1)
    return dict(cos2=float(s.square().mean()),angles_deg=(torch.acos(s)*180/math.pi).cpu().tolist())

def row_basis(W):
    u,s,vh=torch.linalg.svd(W.double(),full_matrices=False)
    rank=int((s>s[0]*1e-6).sum()) if float(s[0])>0 else 0
    return vh[:rank].T
