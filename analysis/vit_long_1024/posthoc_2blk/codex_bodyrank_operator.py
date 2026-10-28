"""Saved-state exact block0 query-key operator and input decomposition.

No training. P19 was registered before this measurement. Cross-state component
mixing is a static algebraic diagnostic, not a trained-network counterfactual.
"""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import json
import math
import resource
import torch
from analysis.vit_long_1024 import common as C
from analysis.vit_long_1024.posthoc_2blk.codex_bodyrank_decompose import ROOT,OUT,SEED


@torch.no_grad()
def main():
    C.set_determinism(2)
    torch.cuda.set_per_process_memory_fraction(.15)
    dest=OUT/'operator_decomposition.json'
    if dest.exists():raise FileExistsError(dest)
    data,_=C.task_dataset(C.image_source('real'),SEED,1,10,250,augment=False)
    idx=torch.randperm(2500,generator=torch.Generator().manual_seed(0))[:256]
    states=[('SL',10),('SL',100),('R0L',10),('R0L',100)]
    cached={};records={}
    for arm,t in states:
        name=f'{arm}{t}'
        model=C.build_arm_model(arm,SEED,'cuda',0.,2,8.)
        model.load_state_dict(torch.load(ROOT/'runs'/arm/f'seed{SEED}/ckpt/model_t{t:02d}.pt',map_location='cpu',weights_only=False))
        model.eval()
        n1=model.layers['block_0'].layers['norm1']
        att=model.layers['block_0'].layers['attn']
        qkv=att.layers['qkv']
        wq=qkv.weight[:384].reshape(6,64,384)
        wk=qkv.weight[384:768].reshape(6,64,384)
        bq=qkv.bias[:384].reshape(6,64)
        Q=wq*n1.weight
        K=wk*n1.weight
        A=K.transpose(-2,-1)@Q*att.scale
        b=torch.einsum('hde,hd->he',K,torch.einsum('hde,e->hd',wq,n1.bias)+bq)*att.scale
        qF=Q.square().sum((-1,-2)).mean().sqrt()
        kF=K.square().sum((-1,-2)).mean().sqrt()
        aF=A.square().sum((-1,-2)).mean().sqrt()
        xx=[];maxerr=0.
        for s in range(0,256,32):
            image=C.eval_tensor(data.images[idx[s:s+32].numpy()],'cuda')
            x=model.layers['patch_embed'](image)
            x=torch.cat((model.cls_token.expand(len(x),-1,-1),x),1)+model.pos_embed
            xi=(x-x.mean(-1,keepdim=True))/(x.var(-1,unbiased=False,keepdim=True)+n1.eps).sqrt()
            qq=qkv(n1(x))[:,:,:384].reshape(len(x),65,6,64).permute(0,2,1,3)
            direct=torch.einsum('bhnd,hde->bhne',qq,wk)*att.scale*n1.weight
            reconstructed=torch.einsum('hde,bne->bhnd',A,xi)+b[None,:,None,:]
            maxerr=max(maxerr,float((direct-reconstructed).norm()/direct.norm()))
            xx.append(xi.cpu())
        sv=torch.linalg.svdvals(A)
        records[name]=dict(q_affine_frob_rms=float(qF),k_affine_frob_rms=float(kF),operator_frob_rms=float(aF),
                           qk_alignment=float(aF/(qF*kF*att.scale)),operator_stable_rank=(A.square().sum((-1,-2))/sv[:,0].square()).cpu().tolist(),
                           gamma_rms=float(n1.weight.square().mean().sqrt()),bias_rms=float(b.square().sum(-1).mean().sqrt()),identity_relative_l2=maxerr)
        cached[name]=(A,b,torch.cat(xx))
        del model
    mixes={}
    for operator,(A,b,_) in cached.items():
        mixes[operator]={}
        for source,(_,_,xi) in cached.items():
            squares=[]
            for s in range(0,len(xi),32):
                aa=torch.einsum('hde,bne->bhnd',A,xi[s:s+32,1:].to('cuda'))+b[None,:,None,:]
                squares.append(aa.square().sum(-1).cpu())
            v=torch.cat(squares)
            mixes[operator][source]=dict(rms=float(v.mean().sqrt()),geomean=float(v.clamp_min(1e-30).log().mean().mul(.5).exp()))
        records[operator]['input_alignment_with_bias']=mixes[operator][operator]['rms']/records[operator]['operator_frob_rms']
    changes={}
    for arm in ['SL','R0L']:
        a,b=f'{arm}10',f'{arm}100'
        l00=math.log(mixes[a][a]['rms']);l01=math.log(mixes[a][b]['rms'])
        l10=math.log(mixes[b][a]['rms']);l11=math.log(mixes[b][b]['rms'])
        op=.5*((l10-l00)+(l11-l01));inp=.5*((l01-l00)+(l11-l10))
        changes[arm]=dict(log_total=l11-l00,log_operator=op,log_input=inp,operator_fraction=op/(l11-l00),
                          exact_log_identity_error=op+inp-(l11-l00))
    result=dict(states=records,mixes=mixes,changes=changes,rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,
                cuda_peak_mb=torch.cuda.max_memory_allocated()/2**20)
    dest.write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()
