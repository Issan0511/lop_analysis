"""Expected uniform-label CE: exact PSD secant identity, numerical integral.

CPU, cached training features. No optimizer or new training.
"""
import numpy as np
import torch
from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import C,OUT,SEED,ckpt,dump,resources,setup,summary


def secant(logits,nodes):
    q,w=np.polynomial.legendre.leggauss(nodes)
    B=torch.zeros(len(logits),10,10,dtype=torch.float64)
    for t,weight in zip((q+1)/2,w/2):
        p=(float(t)*logits).softmax(-1)
        B+=float(weight)*(torch.diag_embed(p)-p[:,:,None]*p[:,None,:])
    return B


def main():
    setup('cpu');torch.set_grad_enabled(False);results=[]
    for arm in ['R0','S']:
        for task in [10,30,60]:
            arr=np.load(OUT/f'codex_features_{arm}_t{task}.npz')
            h=torch.from_numpy(arr['h']).double();z=torch.from_numpy(arr['z']).double();x=torch.from_numpy(arr['y2']).double()
            units=np.load(OUT/f'codex_units_{arm}_t{task}.npz');m=torch.from_numpy(units['m'])
            sd=torch.load(ckpt(arm,task),map_location='cpu',weights_only=False)
            inds=C.loss_classes(arm,SEED,task,10,250)
            w=sd['layers.out.weight'][inds].double();b=sd['layers.out.bias'][inds].double()
            U=sd['layers.block_1.layers.mlp.layers.fc2.weight'].double();V=w@U
            logits=h@w.T+b;err=logits.softmax(-1)-.1
            for nquad in [16,32,64,128,256]:
                B=secant(logits,nquad)
                delta=(B@logits[:,:,None]).squeeze(-1)-err
                error=float(delta.abs().max())
                if error<1e-9:break
            assert error<1e-7,(arm,task,error)
            total=torch.zeros(3072,dtype=torch.float64);selfterm=total.clone();bias_self=total.clone()
            for start in range(0,len(h),64):
                sl=slice(start,start+64)
                curvature=torch.einsum('nck,cj,kj->nj',B[sl],V,V)
                gate=(z[sl]>0);a=z[sl].relu();radial=x[sl]@m
                g=(err[sl]@V)*gate
                total+=(g*radial[:,None]).sum(0)
                selfterm+=(a*curvature*radial[:,None]).sum(0)
                bias_self+=(a*curvature).sum(0)
            total/=len(h);selfterm/=len(h);bias_self/=len(h)
            alive=(z>0).any(0);cross=total-selfterm
            def stat(v):
                x=v[alive].numpy()
                return dict(**summary(x),sinking_fraction=float((x>0).mean()))
            results.append(dict(arm=arm,task=task,alive=int(alive.sum()),quad_nodes=nquad,
                                probability_identity_max_abs=error,total=stat(total),self=stat(selfterm),cross=stat(cross),
                                self_bias=stat(bias_self),
                                self_over_total_l2=float(selfterm.norm()/total.norm()),
                                median_abs_self_over_abs_cross=float((selfterm[alive].abs()/cross[alive].abs().clamp_min(1e-30)).median())))
    dump(OUT/'codex_ce_self.json',dict(rows=results,resources=resources()))
    print('CE self/cross complete',results)


if __name__=='__main__':main()
