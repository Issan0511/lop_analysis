"""Depth-two SL baseline; restart only from atomically completed task boundaries."""
import argparse
import json
import platform
import shutil
import time
from pathlib import Path
import common as T
C, torch, np, F=T.C, T.torch, T.np, T.F


def run(args):
    T.init()
    if not args.smoke:
        assert args.seed in T.SEEDS and args.steps==500 and args.data=='real' and args.engine=='compile'
        assert 1<=args.until_task<=60
    head, dirty=T.git_state()
    assert not dirty, 'Commit source/spec before running: '+dirty
    original=T.original_reference()
    hashes=T.own_hashes()
    out=args.output or T.ARCHIVE/'runs'/f'seed{args.seed}'
    out.mkdir(parents=True, exist_ok=True); (out/'ckpt').mkdir(exist_ok=True); (out/'moments').mkdir(exist_ok=True)
    cfg=dict(seed=args.seed, steps=args.steps, model_config=T.CONFIG, data=args.data, engine=args.engine,
             batch=128, lr=C.LR, labels=10, images_per_class=25, arm='SL', dropout=0., augment=False,
             optimizer=dict(name='Adam', betas=[.9,.999], eps=1e-8, weight_decay=0., fused=True),
             smoke=args.smoke)
    source=C.image_source(args.data)
    model=T.build(args.seed); optimizer=C.build_optimizer(model)
    forward=T.train_forward(model,args.engine)
    classes=torch.tensor(C.loss_classes('SL',args.seed,1,10,25),device='cuda')
    probe_names=C.probe_names(source,args.seed)
    xp=C.eval_tensor(source.val(probe_names),'cuda')
    cp=out/'checkpoint.pt'; done=0; rows=[]; manifest=[]; data_records=[]
    if cp.exists():
        state=torch.load(cp,map_location='cpu',weights_only=False)
        assert state['config']==cfg and state['source_hashes']==hashes
        assert state['original_source_hashes']==original['source_hashes']
        model.load_state_dict(state['model']); optimizer.load_state_dict(state['optimizer'])
        done=state['task']; rows=state['rows']; manifest=state['manifest']; data_records=state['data_records']
        if done<args.until_task and args.engine=='compile': T.warm_compile(model,forward,classes)
        T.set_rng(state['rng'])
        print(json.dumps(dict(resumed_after_task=done, seed=args.seed)),flush=True)
    else:
        T.reset_head(model)
        T.atomic_json(out/'provenance.json',dict(config=cfg,git_hash=head,source_hashes=hashes,
            original=original,probe_names=probe_names,torch=torch.__version__,cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(),python=platform.python_version(),
            parameters=sum(p.numel() for p in model.parameters()),started=time.time()))
    begin=time.monotonic()
    for task in range(done+1,args.until_task+1):
        if (out/'STOP').exists() or (T.ARCHIVE/'STOP').exists():
            T.atomic_json(out/'status.json',dict(status='paused_by_file',completed=task-1));return
        free=shutil.disk_usage(out).free
        if free<25*1024**3: raise RuntimeError(f'Free disk below 25 GiB: {free}')
        data,plan=C.task_dataset(source,args.seed,task,10,25,augment=False)
        x=C.eval_tensor(data.images,'cuda'); y=data.labels.to('cuda')
        data_records.append(dict(task=task,plan=plan,paths=data.paths,
                                 images_sha=T.array_sha(data.images),labels_sha=T.array_sha(data.labels.numpy())))
        task_arrays={}; task_rows=[]; before=time.monotonic()
        def state_at(step):
            return dict(model=T.cpu_tree(model.state_dict()),optimizer=T.cpu_tree(optimizer.state_dict()),
                        rng=T.get_rng(),task=task,step=step,config=cfg,source_hashes=hashes,
                        original_source_hashes=original['source_hashes'],git_hash=head)
        def record(step):
            for population, xx, yy in [('train',x,y),('validation',xp,None)]:
                rr, aa=T.observe(model,xx,classes,yy)
                task_rows.extend(dict(seed=args.seed,task=task,step=step,population=population,**r) for r in rr)
                task_arrays.update({f's{step}_{population}_{k}':v for k,v in aa.items()})
            if step in T.snapshot_steps(task):
                path=out/'ckpt'/f't{task:02d}_s{step:03d}.pt'
                T.atomic_save(path,state_at(step))
                manifest.append(dict(path=str(path.relative_to(out)),bytes=path.stat().st_size,sha256=T.sha256(path)))
        record(0); model.train()
        for step,batch in enumerate(T.PairedBatches(len(y),128,args.steps,args.seed,task),1):
            idx=torch.tensor([i for i,_ in batch],device='cuda')
            optimizer.zero_grad(set_to_none=True)
            logits=forward(x.index_select(0,idx)).index_select(1,classes)
            loss=F.cross_entropy(logits,y.index_select(0,idx))
            if not bool(torch.isfinite(loss)): raise FloatingPointError((task,step,'loss'))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True,foreach=True)
            optimizer.step()
            if step in (1,5,20) or step%25==0 or step==args.steps: record(step)
            if step%100==0:
                T.atomic_json(out/'status.json',dict(status='running',seed=args.seed,task=task,step=step,updated=time.time()))
        assert step==args.steps
        assert all(bool(torch.isfinite(p).all()) for p in model.parameters())
        rows.extend(task_rows)
        path=out/'moments'/f't{task:02d}.npz'
        np.savez_compressed(path,**task_arrays)
        manifest.append(dict(path=str(path.relative_to(out)),bytes=path.stat().st_size,sha256=T.sha256(path)))
        state=state_at(args.steps)
        state.update(rows=rows,manifest=manifest,data_records=data_records)
        T.atomic_save(cp,state)
        T.write_rows(out/'states.csv',rows)
        T.atomic_json(out/'manifest.json',manifest);T.atomic_json(out/'data.json',data_records)
        last=next(r for r in task_rows if r['population']=='train' and r['layer']==1 and r['step']==args.steps)
        status=dict(status='task_complete',seed=args.seed,completed=task,task_seconds=time.monotonic()-before,
                    process_seconds=time.monotonic()-begin,ce=last['ce'],accuracy=last['accuracy'],updated=time.time())
        T.atomic_json(out/'status.json',status);print(json.dumps(status),flush=True)
    T.atomic_json(out/'stage_complete.json',dict(status='complete',seed=args.seed,through_task=max(done,args.until_task),
        requested_until=args.until_task,seconds=time.monotonic()-begin,git_hash=head))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--seed',type=int,required=True);ap.add_argument('--until-task',type=int,default=60)
    ap.add_argument('--steps',type=int,default=500);ap.add_argument('--data',choices=['real','synthetic'],default='real')
    ap.add_argument('--engine',choices=['eager','compile'],default='compile')
    ap.add_argument('--smoke',action='store_true');ap.add_argument('--output',type=Path)
    args=ap.parse_args()
    run(args)
