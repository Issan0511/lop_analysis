"""Static input-writer decomposition of attention1, after failed CLS hypothesis."""
import numpy as np
import torch
from torch.nn import functional as F
from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import OUT,ckpt,dump,images,load_model,parts,progress,resources,setup


@torch.no_grad()
def measure(m,x):
    sums={};count=0
    att=m.layers['block_1'].layers['attn'].layers
    wv=att['qkv'].weight[768:].double().reshape(6,64,384)
    bv=att['qkv'].bias[768:].double()
    wp=att['proj'].weight.double();bp=att['proj'].bias.double()
    for s in range(0,len(x),8):
        nd=parts(m,x[s:s+8],True)
        A=nd['A1'].double();alpha=A.mean(2)
        n=len(A);count+=n
        for name in ['e','attn0','ffn0','h1']:
            source=nd[name].double()
            read=torch.einsum('bhk,bkd->bhd',alpha,source)
            sums[name]=sums.get(name,0)+read.sum(0)
        sums['m2']=sums.get('m2',0)+nd['y2'].double().mean(1).sum(0)
        sums['a1']=sums.get('a1',0)+nd['attn1'].double().mean(1).sum(0)
        # Attention-weighted key residual norm vs all-key rms.
        h1=nd['h1'].double()
        sums['h1_energy']=sums.get('h1_energy',0)+h1.square().sum(-1).mean(1).sum()
        sums['read_energy']=sums.get('read_energy',0)+(alpha*h1.square().sum(-1)[:,None,:]).sum(-1).mean(1).sum()
        sums['keymass_entropy']=sums.get('keymass_entropy',0)+(-(alpha*alpha.clamp_min(1e-300).log()).sum(-1)).sum()
        sums['peak_keymass']=sums.get('peak_keymass',0)+alpha.max(-1).values.sum()
        sums['amax']=sums.get('amax',0)+A.max(-1).values.mean((1,2)).sum()
    means={k:v/count for k,v in sums.items()}
    m2=means['m2'];direction=m2/m2.norm()
    vv=torch.einsum('hed,hd->he',wv,means['h1']).reshape(-1)
    pv=wp@vv
    mn=means['a1'].norm()
    contrib={}
    for k in ['e','attn0','ffn0']:
        output=wp@torch.einsum('hed,hd->he',wv,means[k]).reshape(-1)
        contrib[k]=dict(to_m2=float(output@direction),norm=float(output.norm()))
    contrib['value_bias']=dict(to_m2=float((wp@bv)@direction),norm=float((wp@bv).norm()))
    contrib['proj_bias']=dict(to_m2=float(bp@direction),norm=float(bp.norm()))
    rec=pv+wp@bv+bp
    # Exact factorization including direction, cancellation and bias ratio.
    z=means['h1'].norm();vn=wv.norm();pn=wp.norm()
    vf=vv.norm()/(vn*z);pf=pv.norm()/(pn*vv.norm())
    bf=mn/pv.norm()
    factors=dict(input_read_norm=float(z),value_fro=float(vn),proj_fro=float(pn),
                 value_alignment=float(vf),proj_alignment=float(pf),bias_ratio=float(bf))
    product=np.prod(list(factors.values()))
    return dict(contrib=contrib,factors=factors,attention_mean_norm=float(mn),
                identity_relative=float((rec-means['a1']).norm()/mn),product_relative=float(abs(product/float(mn)-1)),
                h1_rms=float(means['h1_energy'].sqrt()),read_key_rms=float(means['read_energy'].sqrt()),
                keymass_entropy=float(means['keymass_entropy']/6),peak_keymass=float(means['peak_keymass']/6),
                amax=float(means['amax']),m2_norm=float(m2.norm()))


def main():
    setup();x=images(train=False);rows=[]
    for arm in ['R0','S']:
        for t in sorted(int(p.stem.split('_t')[1]) for p in ckpt(arm,0).parent.glob('model_t*.pt')):
            m=load_model(arm,t);r=measure(m,x);r.update(arm=arm,task=t);rows.append(r);del m
    dump(OUT/'codex_transport.json',dict(rows=rows,resources=resources()))
    progress('Q1の追加: attention1の入力書き手と6因子の積を保存状態40件で測定した。codex_transport.json。')


if __name__=='__main__':main()
