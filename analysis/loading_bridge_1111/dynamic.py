"""Registered acute-gradient controls and standardized body recovery (spec_loading_bridge_1111)."""
import os
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
os.environ.setdefault('VIT_LONG_PRODUCTION', 'production_2blk_rlfix.json')
import argparse, copy, hashlib, json, math, subprocess, time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from src import stage1_ctrl_1111 as M

ROOT = Path(__file__).resolve().parents[2]
DATA = Path('/home/issan/Projects/obsidian-research-data')
RAW_M = DATA/'stage1_ctrl_1111/results/stage1_ctrl_1111/raw'
RAW_V = DATA/'vit_long_1024_2blk_rlfix/runs'
POINTS = (0, 1, 5, 25, 50, 100, 200)
ARMS = ('S','P','H10','T','G','TG','H0')
RECOVER = ('S','P','H10','TG')

class ForwardBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, c, b):
        ctx.b = b
        return x * c
    @staticmethod
    def backward(ctx, grad):
        return grad * ctx.b, None, None

def gains(arm):
    return {'T':(.1,1.), 'G':(1.,.1), 'TG':(.1,.1)}.get(arm,(1.,1.))

def loss_for(box,logits,y):
    if isinstance(box,MLP):
        return (logits.logsumexp(1)-(logits*F.one_hot(y,10).float()).sum(1)).mean()
    return F.cross_entropy(logits,y)

def own(seed,t):
    h=hashlib.sha256(f'loading_bridge_1111|P|{seed}|{t}'.encode()).digest()
    return torch.Generator().manual_seed(int.from_bytes(h[:8],'little')%(2**63-1))

def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for v in iter(lambda:f.read(2**20),b''): h.update(v)
    return h.hexdigest()

def provenance(sources,args):
    return dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
                script_sha256=sha(__file__), spec_sha256=sha(ROOT/'specs/spec_loading_bridge_1111.md'),
                sources={str(p):sha(p) for p in sources}, torch=torch.__version__, numpy=np.__version__,
                device=args.model, args=vars(args), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))

def save(p,obj):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(obj,indent=1,allow_nan=False));q.replace(p)

def med(v): return float(torch.quantile(v.double().flatten(),.5))

@torch.no_grad()
def head_stats(h,W,b,yold,ynew,c=1.,back=1.):
    h=h.double();W=W.double();b=b.double();z=h@W.T+b;zc=z-z.mean(1,keepdim=True)
    lp=(c*z).log_softmax(1);p=lp.exp();K=W.shape[0]
    gh=back*(p-1/K)@W
    hc=h-h.mean(0)
    return dict(ce=float(F.nll_loss(lp,ynew)),acc=float((p.argmax(1)==ynew).double().mean()),
                old_acc=float((p.argmax(1)==yold).double().mean()),entropy=float(-(p*lp).sum(1).mean()),
                confidence=float(p.max(1).values.mean()),head_centered_norm=float((W-W.mean(0)).norm()),
                feature_centered_rms=float(hc.square().mean().sqrt()),feature_mean_norm=float(h.mean(0).norm()),
                logit_centered_rms=float((c*zc).square().mean().sqrt()),
                old_margin=float((c*zc).gather(1,yold[:,None]).mean()),new_margin=float((c*zc).gather(1,ynew[:,None]).mean()),
                raw_old_margin=float(zc.gather(1,yold[:,None]).mean()),raw_new_margin=float(zc.gather(1,ynew[:,None]).mean()),
                expected_feature_grad_rms=float(gh.square().mean().sqrt()),
                expected_feature_radial=float((gh*h).sum(1).mean()))

@torch.no_grad()
def intervene(arm,box,seed,t):
    W,b,rows=box.head()
    if arm=='P':
        h=box.features(); old=W[rows].clone();mu=h.mean(0)
        from analysis.loading_bridge_1111.static_probe import rng
        gen=rng('P',seed,t,'MLP' if isinstance(box,MLP) else 'ViT',getattr(box,'act','SL'))
        idx=torch.stack([torch.randperm(len(rows),generator=gen) for j in range(old.shape[1])],1).to(W.device)
        new=old.gather(0,idx);W[rows]=new;b[rows]+= (old-new)@mu
        st=box.opt.state.get(W)
        if st:
            for k in ('exp_avg','exp_avg_sq'): st[k][rows]=st[k][rows].gather(0,idx)
        assert torch.equal(old.sort(0).values,new.sort(0).values)
    elif arm=='H10':
        W.mul_(.1);b.mul_(.1)
    elif arm=='H0':
        W.zero_();b.zero_();box.opt.state.pop(W,None);box.opt.state.pop(b,None)
    elif arm=='Q':
        assert isinstance(box,MLP)
        M.intervene('Q',box.P,box.opt,box.ctx,own(seed,t),M.initial(seed,(784,100,10))[2],'adam')

class MLP:
    def __init__(self,seed,act,tasks):
        self.seed,self.act=seed,act;self.X,subset=M.load_X(seed);self.ctx=M.Ctx(self.X,act)
        self.source=RAW_M/f'states_adam_lr0.001_{act}_s{seed}.npz';saved=np.load(self.source)
        assert np.array_equal(subset,saved['subset'])
        self.P=M.new_params(M.initial(seed,(784,100,10)));self.opt=M.make_opt('adam',self.P,.001)
        gl,gb=M.stream('env_labels_0913',seed),M.stream('env_batch_0913',seed)
        self.labels={};self.orders={};self.bases={};self.replay_checks=[]
        for t in range(1,max(tasks)+2):
            y=torch.randint(10,(1200,),generator=gl);order=M.order_for(gb,4000)
            self.labels[t]=y;self.orders[t]=order
            if t>max(tasks): break
            Y=F.one_hot(y,10).float()
            for idx in order:
                lg=self.forward(self.X[idx]);loss=(lg.logsumexp(1)-(lg*Y[idx]).sum(1)).mean()
                self.opt.zero_grad(set_to_none=True);loss.backward();self.opt.step()
            err=M.maxabs(self.P,[saved[f'end{q}'][t-1] for q in range(4)])
            assert err==0.,(act,seed,t,err)
            assert np.array_equal(y.numpy(),saved['y'][t-1])
            self.replay_checks.append(dict(task=t,maxabs=err))
            if t in tasks:self.bases[t]=M.snapshot(self.P,self.opt)
        self.n=1200;self.eval_every=100;self.recovery_limit=4000
    def reset(self,t):
        self.P,self.opt=M.restore(self.bases[t],'adam',.001);self.t=t
        self.old=self.labels[t];self.new=self.labels[t+1]
    def parameters(self):return self.P
    def body(self):return self.P[:2]
    def head(self):return self.P[2],self.P[3],torch.arange(10)
    def forward(self,x):return M.activ(x@self.P[0].T+self.P[1],self.act)@self.P[2].T+self.P[3]
    def features(self):return M.activ(self.X@self.P[0].T+self.P[1],self.act).detach()
    def batches(self,n,start=0):
        for idx in self.orders[self.t+1][start:n]:yield self.X[idx],self.new[idx]
    def reset_all(self):
        with torch.no_grad():self.P[2].zero_();self.P[3].zero_()
        self.opt=M.make_opt('adam',self.P,.001)
    @torch.no_grad()
    def measure(self,arm,units=True):
        c,b=gains(arm);out=head_stats(self.features(),self.P[2],self.P[3],self.old,self.new,c,b)
        if units:
            u=M.unit_stats(self.P,self.ctx);r=u[0]/np.maximum(u[1],1e-30)
            z=(self.X@self.P[0].T+self.P[1]).double();s=z.std(0,unbiased=False).clamp_min(1e-30)
            out['layers']=[dict(m_med=float(np.median(u[0])),s_med=float(np.median(u[1])),r_med=float(np.median(r)),
                                opening_mean=float(u[2].mean()),dphi_mean=float(u[3].mean()),
                                upper_max_med=med(z.max(0).values),upper_over_s_med=med(z.max(0).values/s),
                                upper_q99_med=med(torch.quantile(z,.99,dim=0)),body_weight_norm=float(self.P[0].norm()))]
            upper=z.max(0).values
            out['unit_arrays']=dict(m=u[0].tolist(),s=u[1].tolist(),r=r.tolist(),pplus=u[2].tolist(),dphi=u[3].tolist(),
                                    upper=upper.tolist(),upper_over_s=(upper/s).tolist(),q99=torch.quantile(z,.99,dim=0).tolist(),
                                    has_positive=(upper>0).tolist())
        return out

class ViT:
    def __init__(self,seed,args):
        from analysis.vit_long_1024 import common as C
        from analysis.joudaki_vit_battle_0919.model import train_forward
        from analysis.joudaki_vit_battle_0919.data import PairedBatches
        self.C=C;C.set_determinism(2);self.seed=seed;self.t=100
        self.model=C.build_arm_model('SL',seed,'cuda',0.,2,8.)
        d=RAW_V/'SL'/f'seed{seed}'/'ckpt'
        self.sources=[d/'model_t100.pt',d/'adam_t100.pt']
        self.base=torch.load(self.sources[0],map_location='cuda',weights_only=False)
        self.adam=torch.load(self.sources[1],map_location='cuda',weights_only=False)
        assert self.adam['task']==100
        source=C.image_source('real');data,_=C.task_dataset(source,seed,101,10,250,augment=False)
        self.X=C.eval_tensor(data.images,'cuda');self.old=C.task_labels(2500,seed,100,10).cuda();self.new=data.labels.cuda()
        self.rows=torch.tensor(C.loss_classes('SL',seed,101,10,250),device='cuda')
        self.order=[torch.tensor([i for i,_ in batch],device='cuda') for batch in PairedBatches(2500,128,1500,seed,101)]
        self.n=2500;self.eval_every=25;self.recovery_limit=1500
        self.compiled=train_forward(self.model,args.engine);self.reset(100)
    def reset(self,t):
        from analysis.joudaki_vit_battle_0919.run import set_rng
        self.model.load_state_dict(self.base);self.opt=self.C.build_optimizer(self.model)
        self.opt.load_state_dict(copy.deepcopy(self.adam['optimizer']));set_rng(self.adam['rng']);self.model.train()
    def parameters(self):return list(self.model.parameters())
    def head(self):
        h=self.model.layers['out'];return h.weight,h.bias,self.rows
    def body(self):
        W,b,_=self.head();return [p for p in self.parameters() if p is not W and p is not b]
    def forward(self,x):return self.compiled(x)[:,self.rows]
    @torch.no_grad()
    def features(self):
        from analysis.vit_long_1024.posthoc_2blk.scale_counterfactual import cls_features
        self.model.eval();h=torch.cat([cls_features(self.model,x) for x in self.X.split(128)]);self.model.train();return h
    def batches(self,n,start=0):
        for idx in self.order[start:n]:yield self.X[idx],self.new[idx]
    def reset_all(self):
        W,b,_=self.head()
        with torch.no_grad():W.zero_();b.zero_()
        self.opt=self.C.build_optimizer(self.model)
    @torch.no_grad()
    def measure(self,arm,units=True):
        c,b=gains(arm);W,hb,rows=self.head();out=head_stats(self.features(),W[rows],hb[rows],self.old,self.new,c,b)
        if units:
            self.model.eval();_,zs=self.C.trace(self.model,self.X[:128]);self.model.train()
            out['layers']=[]
            for _,z in zs:
                z=z.flatten(0,1).double();m=z.mean(0);s=z.std(0,unbiased=False)
                out['layers'].append(dict(m_med=med(m),s_med=med(s),r_med=med(m/s.clamp_min(1e-30)),
                                          opening_mean=float((z>0).double().mean()),dphi_mean=float((z>0).double().mean())))
        return out

@torch.no_grad()
def normdiff(ps,old):return torch.stack([(p-q).square().sum() for p,q in zip(ps,old)]).sum().sqrt()

def acute(box,arm,steps):
    c,b=gains(arm);body=box.body();hp=list(box.head()[:2]);start=[p.detach().clone() for p in body]
    prev=[p.detach().clone() for p in body];prevh=[p.detach().clone() for p in hp]
    rec=[dict(step=0,**box.measure(arm),body_displacement=0.)];track=[]
    for step,(x,y) in enumerate(box.batches(steps),1):
        lg=box.forward(x);loss=loss_for(box,ForwardBackward.apply(lg,c,b),y)
        box.opt.zero_grad(set_to_none=True);loss.backward();box.opt.step()
        with torch.no_grad():
            db=normdiff(body,prev);dh=normdiff(hp,prevh)
            gb=torch.stack([p.grad.square().sum() for p in body if p.grad is not None]).sum().sqrt()
            vb=torch.stack([box.opt.state[p]['exp_avg_sq'].sum() for p in body]).sum()/sum(p.numel() for p in body)
            vals=torch.stack([loss.detach(),db,dh,gb,vb.sqrt()]).tolist()
            track.append(dict(step=step,**dict(zip(('loss','d_body','d_head','g_body','v_body'),vals))))
            for p,q in zip(body,prev):q.copy_(p)
            for p,q in zip(hp,prevh):q.copy_(p)
        if step in POINTS or step==steps:
            rec.append(dict(step=step,**box.measure(arm),body_displacement=float(normdiff(body,start))))
    return dict(snapshots=rec,track=track)

def recovery(box,limit):
    box.reset_all();rows=[dict(step=0,**box.measure('S',units=False))];reach=None
    for step,(x,y) in enumerate(box.batches(limit),1):
        loss=loss_for(box,box.forward(x),y);box.opt.zero_grad(set_to_none=True);loss.backward();box.opt.step()
        if step%box.eval_every==0 or step==limit:
            r=dict(step=step,**box.measure('S',units=False));rows.append(r)
            if r['acc']>=.90:reach=step;break
    return dict(T90=reach,censored=reach is None,steps_run=step,curve=rows)

def natural_continue(box,start):
    rows=[]
    for step,(x,y) in enumerate(box.batches(4000,start),start+1):
        loss=loss_for(box,box.forward(x),y);box.opt.zero_grad(set_to_none=True);loss.backward();box.opt.step()
        if step in (1000,4000):rows.append(dict(step=step,**box.measure('S')))
    return rows

def run_box(box,tasks,args,prov):
    for t in tasks:
        arms=(ARMS+('Q',) if isinstance(box,MLP) else ARMS) if not args.arms else tuple(args.arms.split(','))
        names=('B0',)+arms if args.recovery else arms
        for arm in names:
            key=f'{args.model}_{getattr(box,"act","SL")}_s{box.seed}_t{t}_{arm}'
            path=Path(args.out)/(key+'.json')
            if path.exists():
                old=json.loads(path.read_text());assert old.get('complete') and old['provenance']['script_sha256']==prov['script_sha256']
                for k in ('steps','recovery','natural','engine'):
                    assert old['provenance']['args'][k]==getattr(args,k),(path,'incompatible existing run',k)
                continue
            started=time.monotonic();box.reset(t);out=dict(model=args.model,activation=getattr(box,'act','SL'),seed=box.seed,t_end=t,arm=arm,provenance=prov)
            if arm!='B0':
                intervene(arm,box,box.seed,t);out['acute']=acute(box,arm,args.steps)
            if args.natural and isinstance(box,MLP) and box.act=='ELU' and arm in ('S','Q'):
                sp=M.snapshot(box.P,box.opt)
                out['natural']=natural_continue(box,args.steps)
                box.P,box.opt=M.restore(sp,'adam',.001)
            if args.recovery and (arm=='B0' or arm in RECOVER):out['recovery']=recovery(box,box.recovery_limit)
            out['elapsed_s']=time.monotonic()-started;out['complete']=True;save(path,out)
            print(key,round(out['elapsed_s'],1),out.get('recovery',{}).get('T90'),flush=True)

def synthetic():
    x=torch.randn(7,10,dtype=torch.float64,requires_grad=True);y=torch.arange(7)%10
    gs=[];vs=[]
    for arm in ('S','T','G','TG'):
        c,b=gains(arm);z=ForwardBackward.apply(x,c,b);vs.append(z.detach());gs.append(torch.autograd.grad(F.cross_entropy(z,y),x)[0])
    assert torch.equal(vs[0],vs[2]) and torch.equal(vs[1],vs[3])
    assert torch.allclose(gs[0]*.1,gs[2],atol=1e-14) and torch.allclose(gs[1]*.1,gs[3],atol=1e-14)
    print('synthetic forward/backward factor checks PASS')

def main():
    a=argparse.ArgumentParser();a.add_argument('--model',choices=['mlp','vit'],default='mlp');a.add_argument('--seeds',default='0');a.add_argument('--act',default='R');a.add_argument('--tasks',default='1,10');a.add_argument('--steps',type=int,default=200);a.add_argument('--recovery',action='store_true');a.add_argument('--natural',action='store_true');a.add_argument('--arms',default='');a.add_argument('--out',required=True);a.add_argument('--engine',choices=['compile','eager'],default='compile');a.add_argument('--synthetic',action='store_true');args=a.parse_args()
    if args.synthetic:synthetic();return
    for seed in map(int,args.seeds.split(',')):
        started=time.monotonic()
        if args.model=='mlp':
            tasks=list(map(int,args.tasks.split(',')));box=MLP(seed,args.act,tasks);sources=[box.source]
        else:tasks=[100];box=ViT(seed,args);sources=box.sources
        prov=provenance(sources,args);prov['setup_elapsed_s']=time.monotonic()-started
        if args.model=='mlp':prov['replay_checks']=box.replay_checks
        run_box(box,tasks,args,prov)
        del box
        if args.model=='vit':torch.cuda.empty_cache()

if __name__=='__main__':main()
