"""Input/recording gate, and compiled uninterrupted-versus-resumed gate."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import common as T
torch,np,C=T.torch,T.np,T.C


def basic(out):
    T.init(); reference=T.original_reference()
    m=T.build(97).eval()
    assert m.depth==2
    assert sum(isinstance(a,torch.nn.LayerNorm) for a in m.modules())==5
    for i in range(2):
        mlp=m.layers[f'block_{i}'].layers['mlp'].layers
        assert mlp['fc1'].weight.shape==(1536,384)
        assert torch.equal(mlp['act'](torch.tensor([-1.,0.,1.],device='cuda')),torch.tensor([0.,0.,1.],device='cuda'))
    source=C.RealImages();data,plan=C.task_dataset(source,0,10,10,25,augment=False)
    x=C.eval_tensor(data.images,'cuda');y=data.labels.to('cuda')
    for k,((xx,yy),batch) in enumerate(zip(T.train_loader(data,0,10,4,0),T.PairedBatches(250,128,4,0,10))):
        idx=torch.tensor([i for i,_ in batch],device='cuda')
        assert torch.equal(xx,x[idx].cpu()) and torch.equal(yy,y[idx].cpu())
    assert k==3
    saved={};hooks=[]
    for i in range(2):
        hooks.append(C.fc1(m,i).register_forward_hook(lambda module,args,z,i=i:saved.__setitem__(i,z.detach().clone())))
    with torch.no_grad(): pred=m(x[:32])
    for h in hooks:h.remove()
    with torch.no_grad(): trace,blocks=C.trace(m,x[:32])
    assert torch.equal(pred,trace)
    for i,(_,z) in enumerate(blocks):assert torch.equal(z,saved[i])
    before=T.digest_tree(m.state_dict());rng=T.digest_tree(T.get_rng())
    classes=torch.tensor(C.loss_classes('SL',0,10,10,25),device='cuda')
    rr,aa=T.observe(m,x,classes,y)
    assert T.digest_tree(m.state_dict())==before and T.digest_tree(T.get_rng())==rng
    out.mkdir(parents=True,exist_ok=True)
    T.atomic_json(out/'basic.json',dict(status='passed',model_depth=2,layernorms=5,
        loader_batches_equal=4,trace_logits_and_preactivations_bit_equal=True,recording_state_rng_unchanged=True,
        original_git=reference['git_hash'],source_hashes=T.own_hashes(),git_hash=T.git_state()[0]))
    print('basic gate passed',flush=True)


def resume(out):
    out.mkdir(parents=True,exist_ok=True)
    script=Path(__file__).with_name('train.py')
    for name,until in [('continuous',2),('resumed',1),('resumed',2)]:
        command=[sys.executable,'-u',str(script),'--seed','97','--until-task',str(until),'--steps','25',
                 '--data','synthetic','--smoke','--output',str(out/name)]
        print(json.dumps(dict(starting=name,until=until)),flush=True)
        with (out/f'{name}_{until}.log').open('w') as log:
            subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,check=True,cwd=T.ROOT)
    a=torch.load(out/'continuous/checkpoint.pt',map_location='cpu',weights_only=False)
    b=torch.load(out/'resumed/checkpoint.pt',map_location='cpu',weights_only=False)
    for key in ('model','optimizer','rng','rows','data_records'):
        assert T.digest_tree(a[key])==T.digest_tree(b[key]),key
    for task in (1,2):
        with np.load(out/f'continuous/moments/t{task:02d}.npz') as aa,np.load(out/f'resumed/moments/t{task:02d}.npz') as bb:
            assert aa.files==bb.files
            for key in aa.files:assert np.array_equal(aa[key],bb[key]),(task,key)
    T.atomic_json(out/'resume.json',dict(status='passed',engine='compile',tasks=2,steps=25,
        bit_equal=['model','optimizer','rng','rows','data_records','all_moments'],git_hash=T.git_state()[0]))
    print('compiled resume gate passed',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--mode',choices=['basic','resume'],required=True)
    ap.add_argument('--output',type=Path,default=T.ARCHIVE/'gate')
    args=ap.parse_args()
    (basic if args.mode=='basic' else resume)(args.output)
