"""Q4: fixed 256-image q/k cotangent subspaces at all saved task ends."""
import time
from analysis.vit_long_1024.posthoc_2blk.codex_qkform_common import *

def eigensystem(cov,total,n):
    centered=cov-total[:,None]*total[None,:]/n
    ev,U=torch.linalg.eigh(centered.double())
    ev=ev.flip(0).clamp_min(0);U=U.flip(1)
    valid=bool(ev[0]>1e-50)
    return dict(eigenvalues=ev.cpu().numpy(),basis=U[:,:32].cpu().numpy(),
                pr=float(ev.sum().square()/ev.square().sum().clamp_min(1e-100)),valid=valid)

def compare(a,b):
    a=torch.as_tensor(a);b=torch.as_tensor(b)
    return {k:basis_overlap(a[:,:k],b[:,:k]) for k in (4,8,16)}

def main():
    setup();dest=OUT/'codex_signals.json'
    if dest.exists():raise FileExistsError(dest)
    data,_=C.task_dataset(C.image_source('real'),SEED,11,10,250,augment=False)
    idx=torch.randperm(2500,generator=torch.Generator().manual_seed(0))[:256]
    X=C.eval_tensor(data.images[idx.numpy()],'cuda')
    common_y=C.task_labels(2500,SEED,101,10)[idx].to('cuda')
    result=dict(timestamp=stamp(),images=256,idx=idx.tolist(),space='q/k concatenated 6 heads in 384 output coordinates; tokens pooled; centered covariance',
                modes=['trained_labels','common_task101_labels'],states={},adjacent={},longlag={},head_adjacent={},feature_adjacent={})
    start=time.monotonic()
    for arm in ('SL','R0L'):
        result['states'][arm]={};result['adjacent'][arm]=[];result['head_adjacent'][arm]=[];result['feature_adjacent'][arm]=[]
        previous=None;previous_t=None;anchor=None;prevhead=None;prevfeat=None
        for t in TASKS:
            model=load_model(arm,t)
            rows=torch.tensor(C.loss_classes(arm,SEED,max(t,1),10,250),device='cuda')
            trained_y=C.task_labels(2500,SEED,max(t,1),10)[idx].to('cuda')
            cov={};tot={};n=0;concentrations={};losses={};features=[]
            for mode in result['modes']:
                concentrations[mode]=[];losses[mode]=[]
                for kind in ('q','k'):
                    for weighting in ('energy','image_normalized'):
                        key=f'{mode}_{kind}_{weighting}'
                        cov[key]=torch.zeros(D,D,device='cuda',dtype=torch.float64)
                        tot[key]=torch.zeros(D,device='cuda',dtype=torch.float64)
            for s in range(0,256,16):
                logits,nodes,h=attention_forward(model,X[s:s+16])
                features.append(h.detach().cpu())
                for m,mode in enumerate(result['modes']):
                    labels=trained_y if m==0 else common_y
                    loss=F.cross_entropy(logits.index_select(1,rows),labels[s:s+16],reduction='sum')
                    gq,gk,gs=torch.autograd.grad(loss,(nodes['q'],nodes['k'],nodes['score']),retain_graph=m==0)
                    losses[mode].append(float(loss.detach()))
                    en=gs.detach().square().flatten(1)
                    top=en.topk(math.ceil(en.shape[1]*.05),dim=1).values.sum(1)
                    concentrations[mode].append((float(top.sum()),float(en.sum())))
                    for kind,g in (('q',gq),('k',gk)):
                        z=g.detach().permute(0,2,1,3).reshape(len(g),65,D).double()
                        for weighting in ('energy','image_normalized'):
                            zz=z if weighting=='energy' else z/z.square().sum((1,2),keepdim=True).sqrt().clamp_min(1e-100)
                            zz=zz.reshape(-1,D)
                            key=f'{mode}_{kind}_{weighting}'
                            cov[key]+=zz.T@zz;tot[key]+=zz.sum(0)
                n+=16*65
                del logits,nodes,h,gq,gk,gs,loss
            systems={k:eigensystem(cov[k],tot[k],n) for k in cov}
            arrays={}
            for key,sy in systems.items():
                arrays[f'{key}_basis']=sy['basis'];arrays[f'{key}_eigenvalues']=sy['eigenvalues']
            feats=torch.cat(features).double();fc=feats-feats.mean(0)
            ev,U=torch.linalg.eigh(fc.T@fc);fb=U[:,-16:].flip(1)
            W=model.layers['out'].weight[rows].detach().cpu().double();W=W-W.mean(0)
            hb=row_basis(W)
            arrays.update(features=feats.numpy(),feature_basis=fb.numpy(),head_centered=W.numpy())
            np.savez_compressed(OUT/f'codex_signals_{arm}_t{t}.npz',**arrays)
            rec=dict(signals={k:{'pr':v['pr'],'valid':v['valid']} for k,v in systems.items()},
                     loss={k:sum(v)/256 for k,v in losses.items()},
                     score_top5_energy={k:sum(x for x,y in v)/max(sum(y for x,y in v),1e-100) for k,v in concentrations.items()},
                     feature_pr=float(ev.sum().square()/ev.square().sum().clamp_min(1e-100)),
                     feature_head_energy=float((fc@hb).square().sum()/fc.square().sum().clamp_min(1e-100)))
            result['states'][arm][t]=rec
            if previous is not None:
                vals={k:compare(previous[k]['basis'],v['basis']) for k,v in systems.items() if v['valid'] and previous[k]['valid']}
                result['adjacent'][arm].append(dict(t0=previous_t,t1=t,overlaps=vals))
                result['head_adjacent'][arm].append(dict(t0=previous_t,t1=t,**basis_overlap(prevhead,hb)))
                result['feature_adjacent'][arm].append(dict(t0=previous_t,t1=t,**basis_overlap(prevfeat,fb[:,:8])))
            if t==10:anchor=systems
            if t==100:
                result['longlag'][arm]={k:compare(anchor[k]['basis'],v['basis']) for k,v in systems.items() if v['valid'] and anchor[k]['valid']}
            previous=systems;previous_t=t;prevhead=hb;prevfeat=fb[:,:8]
            del model,cov,tot,features
            print(f'{arm} t{t} fixed256 signals measured; {time.monotonic()-start:.1f}s',flush=True)
    result.update(resource_record());result['seconds']=time.monotonic()-start
    write_new(dest,result)
    progress('Q4の固定256枚・全40保存状態のq/k逆伝播部分空間を測定した。元課題ラベルと共通課題101ラベルを分けた。主角は4・8・16次元で保存した。画像ごとの信号を正規化した対照も保存した。')

if __name__=='__main__':main()
