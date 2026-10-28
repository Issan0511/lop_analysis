"""Q1 writers, Q2 exact endpoint ledgers, Q3 static mean counterfactuals.

No optimization. Fixed validation probe (64 images), full training set (2500).
Model forward float32; statistics and algebraic ledgers float64.
"""
import argparse
import json
import time
import numpy as np
import torch
from torch.nn import functional as F
from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import (
    C, OUT, REPO, ROOT, SEED, ckpt, dump, images, load_model, logit_stats,
    parts, progress, resources, setup, sha, summary)


@torch.no_grad()
def probe_stats(model, x, arm, task):
    arrays = {}
    # Retain only this probe, no graph; at most 64*65*3072 activations.
    for start in range(0, len(x), 8):
        nd = parts(model, x[start:start+8], True)
        att = model.layers['block_1'].layers['attn']
        A, V = nd['A1'], nd['V1']
        b, h, n, d = V.shape
        cmix = (A[..., :1] * V[:, :, :1, :]).transpose(1, 2).reshape(b, n, h*d)
        nd['attn1_cls'] = F.linear(cmix, att.layers['proj'].weight, None)
        nd['attn1_patch'] = nd['attn1'] - nd['attn1_cls'] - att.layers['proj'].bias
        for name in ['e','attn0','ffn0','h1','attn1','u1','u2','y2','h2','h','z0','z1','act0','attn1_cls','attn1_patch']:
            arrays.setdefault(name, []).append(nd[name].detach().cpu().numpy())
        for name, val in [('cls_attention', A[..., 0]), ('amax', A.max(-1).values),
                          ('cls_winner', (A.argmax(-1)==0).float()),
                          ('value_cls_norm', V[:, :, 0].norm(dim=-1)),
                          ('value_patch_norm', V[:, :, 1:].norm(dim=-1))]:
            arrays.setdefault(name, []).append(val.cpu().numpy())
        del nd
    a = {k:np.concatenate(v) for k,v in arrays.items()}
    del arrays
    m = a['y2'].astype('float64').mean((0,1))
    mc = a['y2'][:,0].astype('float64').mean(0)
    mn = np.linalg.norm(m)
    source = {k:a[k].astype('float64').mean((0,1)) for k in ['e','attn0','ffn0','attn1','attn1_cls','attn1_patch']}
    source['proj_bias'] = model.layers['block_1'].layers['attn'].layers['proj'].bias.detach().cpu().double().numpy()
    contrib = {k:float(v@m/mn) for k,v in source.items()}
    fc = model.layers['block_1'].layers['mlp'].layers['fc1']
    w = fc.weight.detach().cpu().double().numpy()
    bias = fc.bias.detach().cpu().double().numpy()
    cos = (w@m)/(np.linalg.norm(w,axis=1)*mn)
    unit = {'w':w, 'bias':bias, 'm':m, 'm_cls':mc, 'cos':cos,
            **{f'mean_{k}':v for k,v in source.items()}}
    zz = {}
    cf = []
    for label, z in [('all',a['z1'].reshape(-1,w.shape[0])), ('cls',a['z1'][:,0])]:
        z = z.astype('float64')
        mu, sd, op = z.mean(0),z.std(0,ddof=1),(z>0).mean(0)
        unit.update({f'mu_{label}':mu, f'sd_{label}':sd, f'open_{label}':op})
        zz[label] = dict(mean=float(mu.mean()), median=float(np.median(mu)),
                         width=float(np.median(sd)), r=float(np.median(mu/np.maximum(sd,1e-12))),
                         dead=float((op<=.005).mean()), alive_any=int((op>0).sum()), dead_zero=float((op==0).mean()))
        # Global all-token mean is removed in BOTH token populations.
        for k in [0., .1, .25, .5, 1.]:
            zcf = z + (k-1)*(w@m)[None,:]
            opened = (zcf>0).mean(0)
            cf.append(dict(population=label,k=k,dead=float((opened<=.005).mean()),
                           alive_any=int((opened>0).sum()),open_fraction=float((zcf>0).mean())))
    norms = {}
    for name in ['e','u1','ffn0','h1','attn1','u2','h2']:
        z = a[name].astype('float64')
        norms[name] = dict(cls_rms=float(np.sqrt((z[:,0]**2).sum(-1).mean())),
                           patch_rms=float(np.sqrt((z[:,1:]**2).sum(-1).mean())),
                           mean_norm=float(np.linalg.norm(z.mean((0,1)))))
    weights = {}
    for i in range(2):
        att = model.layers[f'block_{i}'].layers['attn'].layers
        mlp = model.layers[f'block_{i}'].layers['mlp'].layers
        for key in ['q','k','v']:
            j = ['q','k','v'].index(key)
            weights[f'{key}{i}'] = float(att['qkv'].weight[j*384:(j+1)*384].norm())
        weights[f'proj{i}']=float(att['proj'].weight.norm())
        weights[f'fc1_{i}']=float(mlp['fc1'].weight.norm())
        weights[f'fc2_{i}']=float(mlp['fc2'].weight.norm())
    aw = a['cls_attention'].astype('float64')
    attstats = dict(cls_mass=float(aw.mean()), cls_mass_by_head=aw.mean((0,2)).tolist(),
                    cls_mass_cls_query=float(aw[:,:,0].mean()),
                    cls_mass_patch_query=float(aw[:,:,1:].mean()),
                    cls_winner=float(a['cls_winner'].mean()),amax_mean=float(a['amax'].mean()),
                    amax_median=float(np.median(a['amax'])),saturated=float((a['amax']>.99).mean()),
                    cls_value_norm_rms=float(np.sqrt((a['value_cls_norm']**2).mean())),
                    patch_value_norm_rms=float(np.sqrt((a['value_patch_norm']**2).mean())))
    mn_reconstructed = sum(source[k] for k in ['e','attn0','ffn0','attn1'])
    att_error=np.linalg.norm(source['attn1']-source['attn1_cls']-source['attn1_patch']-source['proj_bias'])
    row = dict(arm=arm,task=task,m2_norm=float(mn), m2_energy=float(mn**2/np.square(a['y2'].astype('float64')).sum(-1).mean()),
               contrib=contrib,attention1=attstats,norms=norms,weights=weights,z=zz,
               cos=summary(cos),bias=summary(bias),wm=summary(w@m),
               ffn0_open=float((a['z0']>0).mean()),ffn0_open_cls=float((a['z0'][:,0]>0).mean()),
               ffn0_mean_activation_norm=float(np.linalg.norm(a['act0'].astype('float64').mean((0,1)))),
               counterfactual=cf,att_reconstruction_relative=float(att_error/max(np.linalg.norm(source['attn1']),1e-30)),
               residual_reconstruction_relative=float(np.linalg.norm(mn_reconstructed-m)/max(mn,1e-30)),
               probe_logit=logit_stats(torch.from_numpy(a['h']).cuda(),model.layers['out'],C.loss_classes(arm,SEED,max(task,1),10,250)))
    return row,unit


@torch.no_grad()
def train_stats(model, x, arm, task, m):
    hs=[]; ys=[]; zs=[]
    for s in range(0,len(x),32):
        nd=parts(model,x[s:s+32])
        hs.append(nd['h']);ys.append(nd['y2'][:,0]);zs.append(nd['z1'][:,0])
    h=torch.cat(hs);y2=torch.cat(ys);z=torch.cat(zs)
    rows=C.loss_classes(arm,SEED,max(task,1),10,250)
    old=C.task_labels(len(x),SEED,max(task,1),10).cuda()
    out=logit_stats(h,model.layers['out'],rows,old)
    if arm in ['R0','S']:
        H=h.double(); Y=y2.double(); Z=z.double()
        head=model.layers['out']; W=head.weight[rows].double(); b=head.bias[rows].double()
        U=model.layers['block_1'].layers['mlp'].layers['fc2'].weight.double()
        p=F.linear(H,W,b).softmax(-1)
        vv=W@U
        mm=torch.as_tensor(m,device='cuda'); radial=Y@mm
        nextlabels=C.task_labels(len(x),SEED,task+1,10).cuda()
        components={}
        for name,target in [('uniform',torch.full_like(p,.1)),('old',F.one_hot(old,10).double()),('next',F.one_hot(nextlabels,10).double())]:
            dz=((p-target)@vv)*(Z>0)
            projected=-(dz*radial[:,None]).mean(0)
            common_only=-dz.mean(0)*mm.square().sum()
            common_activation=-((p-target).mean(0)@vv)
            components[name]={'minus_grad_w_dot_m':summary(projected.cpu().numpy()),
                              'common_input_term':summary(common_only.cpu().numpy()),
                              'residual_input_term':summary((projected-common_only).cpu().numpy()),
                              'minus_grad_common_activation':summary(common_activation.cpu().numpy()),
                              'minus_grad_bias':summary((-dz.mean(0)).cpu().numpy()),
                              'alive_minus_grad_w_dot_m':summary(projected[(Z>0).any(0)].cpu().numpy()) if bool((Z>0).any()) else None}
        out['gradient_diagnostic']=components
        out['label_counts_old']=torch.bincount(old,minlength=10).tolist()
        out['label_counts_next']=torch.bincount(nextlabels,minlength=10).tolist()
        out['gradient_note']='Fixed saved head; R0 actual reset makes current body gradient zero at step 0. Saved Adam may still move.'
        # Compact full training features make all authorized linear probes cheap.
        if task in [10,20,30,40,60,100]:
            np.savez_compressed(OUT/f'codex_features_{arm}_t{task}.npz',h=h.cpu().numpy(),
                                y2=y2.cpu().numpy(),z=z.cpu().numpy())
    return out


def ledger(old, new):
    w0,w1=old['w'],new['w'];dw=w1-w0
    n0=np.linalg.norm(w0,axis=1);n1=np.linalg.norm(w1,axis=1)
    stretch=(n1-n0)[:,None]*w0/n0[:,None]
    rotate=n1[:,None]*(w1/n1[:,None]-w0/n0[:,None])
    result={}
    for name,mkey in [('all','m'),('cls','m_cls')]:
        m0,m1=old[mkey],new[mkey];dm=m1-m0
        terms={'weight':dw@m0,'input':w0@dm,'cross':dw@dm,'bias':new['bias']-old['bias'],
               'stretch':stretch@m0,'rotation':rotate@m0}
        delta=w1@m1+new['bias']-w0@m0-old['bias']
        resid=delta-sum(terms[k] for k in ['weight','input','cross','bias'])
        s0=np.maximum(old[f'sd_{name}'],1e-12);s1=np.maximum(new[f'sd_{name}'],1e-12)
        mu0=w0@m0+old['bias'];mu1=w1@m1+new['bias']
        rterms={k:v/s1 for k,v in terms.items()}
        rterms['width']=mu0*(1/s1-1/s0)
        result[name]=dict(delta=summary(delta),terms={k:summary(v) for k,v in terms.items()},
                         delta_r=summary(mu1/s1-mu0/s0),r_terms={k:summary(v) for k,v in rterms.items()},
                         identity_max_abs=float(abs(resid).max()),
                         r_identity_max_abs=float(abs(mu1/s1-mu0/s0-sum(rterms[k] for k in ['weight','input','cross','bias','width'])).max()))
    return result


def main(args):
    setup();torch.set_grad_enabled(False)
    dest=OUT/'codex_checkpoints.json'
    if dest.exists():raise FileExistsError(dest)
    inputs=[p for a in args.arms for p in ckpt(a,0).parent.glob('*.pt') if p.name.startswith(('model','adam'))]
    inputs+=list((REPO/'analysis/vit_long_1024/posthoc_2blk').glob('*.py'))
    inputs+=[OUT/'common_mode_R0.json',OUT/'common_mode_S.json']
    dump(OUT/'codex_input_hashes.json',{str(p):sha(p) for p in inputs})
    xp=images(train=False);xt=images(train=True)
    rows=[];ledgers=[];start=time.monotonic()
    for arm in args.arms:
        previous=None;prev_t=None
        tasks=sorted(int(p.stem.split('_t')[1]) for p in ckpt(arm,0).parent.glob('model_t*.pt'))
        for task in tasks:
            model=load_model(arm,task)
            row,units=probe_stats(model,xp,arm,task)
            row['train']=train_stats(model,xt,arm,task,units['m'])
            if arm in ['R0','S']:
                np.savez_compressed(OUT/f'codex_units_{arm}_t{task}.npz',**units)
                if previous is not None:
                    ledgers.append(dict(arm=arm,start=prev_t,end=task,task_span=task-prev_t,**ledger(previous,units)))
                previous=units;prev_t=task
            rows.append(row)
            dump(dest,dict(rows=rows,ledgers=ledgers,resources=resources(),seconds=time.monotonic()-start))
            print(f"{arm} t{task}: m2={row['m2_norm']:.3g} cls_mass={row['attention1']['cls_mass']:.3f} cls_write={row['contrib']['attn1_cls']:.3g} a1_write={row['contrib']['attn1']:.3g}",flush=True)
            del model
    progress(f'Q1保存状態測定とQ2端点帳簿が完了。{len(rows)}状態。codex_checkpoints.json。資源: {resources()}。')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--arms',nargs='+',default=['R0','S','R0L','SL'])
    main(p.parse_args())
