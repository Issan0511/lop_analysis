"""Read-only radial loss derivatives on cached full training features (CPU)."""
import numpy as np
import torch
from torch.nn import functional as F
from analysis.vit_long_1024.posthoc_2blk.codex_r0_common import C,OUT,SEED,ckpt,dump,resources,setup


def main():
    setup('cpu');torch.set_grad_enabled(False);rows=[]
    states=__import__('json').loads((OUT/'codex_checkpoints.json').read_text())['rows']
    populations=[]
    for r in states:
        if r['arm'] not in ['R0','S']:continue
        a=np.load(OUT/f'codex_units_{r["arm"]}_t{r["task"]}.npz')
        m,mc=a['m'],a['m_cls']
        populations.append(dict(arm=r['arm'],task=r['task'],all_mean_norm=float(np.linalg.norm(m)),
            cls_mean_norm=float(np.linalg.norm(mc)),mean_cosine=float(m@mc/(np.linalg.norm(m)*np.linalg.norm(mc))),
            cls_u2_rms=r['norms']['u2']['cls_rms'],cls_after_ffn_rms=r['norms']['h2']['cls_rms']))
    dump(OUT/'codex_mean_populations.json',populations)
    for arm in ['R0','S']:
        for t in [10,20,30,40,60,100]:
            arrays=np.load(OUT/f'codex_features_{arm}_t{t}.npz')
            h=torch.from_numpy(arrays['h']).double()
            x=torch.from_numpy(arrays['y2']).double()
            state=torch.load(ckpt(arm,t),map_location='cpu',weights_only=False)
            inds=C.loss_classes(arm,SEED,t,10,250)
            w=state['layers.out.weight'][inds].double();b=state['layers.out.bias'][inds].double()
            wh=h@w.T;l=wh+b;p=l.softmax(-1)
            old=C.task_labels(len(h),SEED,t,10);nxt=C.task_labels(len(h),SEED,t+1,10)
            out=dict(arm=arm,task=t)
            mx=x.mean(0);mf=(h-x).mean(0);mh=h.mean(0)
            out['final_ffn_cancellation']=dict(input_mean_norm=float(mx.norm()),
                output_mean_norm=float(mf.norm()),feature_mean_norm=float(mh.norm()),
                cancel_fraction=float(-(mf@mx)/mx.square().sum()),
                cls_input_rms=float(x.square().sum(1).mean().sqrt()),
                cls_output_rms=float(h.square().sum(1).mean().sqrt()))
            for name,y in [('old',old),('next',nxt),('uniform',None)]:
                e=p-(torch.full_like(p,.1) if y is None else F.one_hot(y,10))
                deriv=(e*wh).sum(1)
                out[name]=dict(mean=float(deriv.mean()),median=float(deriv.median()),
                               growth_side_fraction=float((deriv<0).double().mean()))
            rows.append(out)
    dump(OUT/'codex_radial.json',dict(rows=rows,resources=resources()))
    print('radial derivatives completed')


if __name__=='__main__':main()
