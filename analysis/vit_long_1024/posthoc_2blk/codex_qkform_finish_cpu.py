"""CPU-only saved-state differences for finishing Q5; no model replay or CUDA."""
import os
os.environ.setdefault('VIT_LONG_PRODUCTION','production_2blk_rlfix.json')
import hashlib
import json
import math
import resource
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo
import numpy as np
import torch
from torch.nn import functional as TF
from analysis.vit_long_1024 import common as C

ROOT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix')
OUT=ROOT/'posthoc_qk_formation'
REPO=Path('/home/issan/Projects/claude/wt/vit_long_1024')
SEED=10252010
D,H,DH=384,6,64

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def state(arm,t):
    return torch.load(ROOT/'runs'/arm/f'seed{SEED}/ckpt/model_t{t:02d}.pt',map_location='cpu',weights_only=False,mmap=True)

def op(sd):
    p='layers.block_0.layers.'
    w=sd[p+'attn.layers.qkv.weight'].double()
    g=sd[p+'norm1.weight'].double()
    return (w[:D].reshape(H,DH,D)*g,w[D:2*D].reshape(H,DH,D)*g)

def op_small(path):
    with np.load(path) as d:
        g=torch.from_numpy(d['gamma'].astype('float64'))
        return torch.from_numpy(d['Q'].astype('float64'))*g,torch.from_numpy(d['K'].astype('float64'))*g

def dot(a,b,cov):
    q1,k1=a;q0,k0=b
    return float(((k1@k0.transpose(-1,-2))*((q1@cov)@q0.transpose(-1,-2))).sum()/(64*H))

def diff(a,b,cov):
    e0=dot(a,a,cov);e1=dot(b,b,cov);cross=dot(a,b,cov)
    f0,f1=math.sqrt(e0),math.sqrt(e1)
    radial=2*(cross-e0)
    step=max(e1+e0-2*cross,0.)
    parallel=(radial/(2*f0))**2
    perpendicular=max(step-parallel,0.)
    den=f0+f1
    return dict(F0=f0,F1=f1,delta_F=f1-f0,radial_energy=radial,step_energy=step,
                parallel_energy=parallel,perpendicular_energy=perpendicular,
                radial_norm=radial/den,step_norm=step/den,
                parallel_norm=parallel/den,perpendicular_norm=perpendicular/den,
                da=math.sqrt(step),radial_cos=radial/(2*f0*math.sqrt(step)) if step>0 else 0.,
                identity_absolute_error=abs((radial+step)/den-(f1-f0)))

@torch.no_grad()
def main():
    torch.set_num_threads(2)
    assert not torch.cuda.is_initialized()
    dest=OUT/'codex_finish_cpu.json'
    if dest.exists():raise FileExistsError(dest)
    initial_files=[p for p in OUT.iterdir() if p.is_file() and p.name!='PROGRESS.md' and not p.name.startswith('codex_finish')]
    initial_files+=list((REPO/'analysis/vit_long_1024/posthoc_2blk').glob('codex_qkform_*.py'))
    initial_hash={str(p):sha(p) for p in initial_files}
    summary=json.loads((OUT/'codex_summary.json').read_text());cp=summary['checkpoints']
    data,_=C.task_dataset(C.image_source('real'),SEED,11,10,250,augment=False)
    idx=torch.tensor(cp['idx']);x=C.eval_tensor(data.images[idx.numpy()],'cpu')
    result=dict(timestamp=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(timespec='seconds'),
                device='cpu',reference='each arm task10, same 32 images and all 65 tokens as codex_checkpoints.json',
                checkpoints={},replays={},max_reference_relative_error=0.)
    start=time.monotonic()
    for arm in ('SL','R0L'):
        sd=state(arm,10)
        patch=TF.conv2d(x,sd['layers.patch_embed.layers.proj.weight'],sd['layers.patch_embed.layers.proj.bias'],stride=8).flatten(2).transpose(1,2)
        z=torch.cat((sd['cls_token'].expand(len(x),-1,-1),patch),1)+sd['pos_embed']
        xi=(z-z.mean(-1,keepdim=True))/(z.var(-1,unbiased=False,keepdim=True)+1e-5).sqrt()
        xi=xi.reshape(-1,D).double();cov=xi.T@xi/len(xi)
        del sd,patch,z,xi
        states={};previous=None;previous_t=None;rows=[]
        for t in sorted(map(int,cp['states'][arm])):
            sd=state(arm,t);parts=op(sd);del sd
            f=math.sqrt(dot(parts,parts,cov))
            err=abs(f/cp['states'][arm][str(t)]['a_fixed']-1)
            result['max_reference_relative_error']=max(result['max_reference_relative_error'],err)
            states[t]=parts
            if previous is not None:
                d=diff(previous,parts,cov);d.update(t0=previous_t,t1=t,tasks=t-previous_t)
                for key in ('delta_F','radial_energy','step_energy','radial_norm','step_norm'):
                    d[key+'_per_task']=d[key]/d['tasks']
                rows.append(d)
            previous=parts;previous_t=t
        result['checkpoints'][arm]=rows
        for name,old in summary['replays'].items():
            if old['arm']!=arm:continue
            snaps={0:states[old['t']]}
            for step in (100,500,1500):snaps[step]=op_small(OUT/f'{name}_step{step}_small.npz')
            windows=[]
            for lo,hi in ((0,100),(100,500),(500,1500)):
                d=diff(snaps[lo],snaps[hi],cov);d.update(lo=lo,hi=hi);windows.append(d)
            total=diff(snaps[0],snaps[1500],cov)
            result['replays'][name]=dict(windows=windows,total=total)
        print(f'{arm}: saved endpoint differences completed on CPU',flush=True)
        del states,previous,cov
    result['seconds']=time.monotonic()-start
    result['rss_mb']=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
    result['cuda_initialized']=torch.cuda.is_initialized()
    result['input_files_checked']=len(initial_hash)
    result['changed_inputs']=[p for p,v in initial_hash.items() if sha(p)!=v]
    assert result['max_reference_relative_error']<1e-5,result
    assert result['rss_mb']<3072 and not result['cuda_initialized']
    assert not result['changed_inputs']
    with dest.open('x') as f:json.dump(result,f,indent=2,allow_nan=False)
    with (OUT/'codex_finish_input_hashes.json').open('x') as f:json.dump(initial_hash,f,indent=2)
    with (OUT/'PROGRESS.md').open('a') as f:
        f.write(f'\n## {result["timestamp"]} — CPU差分の完了\n\nξ10を全比較で固定した課題末38区間と回し直し24窓を再集計した。既存Fとの最大相対差は{result["max_reference_relative_error"]:.3g}。RSSは{result["rss_mb"]:.0f}MiB。CUDA初期化は無い。検査した既存{len(initial_hash)}ファイルの変更は0件。\n')
    print(json.dumps({k:result[k] for k in ('seconds','rss_mb','cuda_initialized','max_reference_relative_error','input_files_checked','changed_inputs')},indent=2))

if __name__=='__main__':main()
