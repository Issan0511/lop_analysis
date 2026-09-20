"""A2: exact C-arm training with full1200, every-update read-only observations."""
from __future__ import annotations
import argparse, hashlib, json, os, subprocess, sys, time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='2'
import numpy as np
import torch
import torch.nn.functional as F
from src import relu_doors_0919 as E
from src.cifar_interventions_0920 import Engine as Base, cpu, tree_hash, save_pt, load_pt, DATA
from analysis.cifar_ledger_0920.replay import put, sha, save_npz, setup, exclusive, environment
from analysis.drive_cifar_c_0920 import numerics as N
RUN='drive_cifar_c_0920'


def inputs(seeds,device):
    E.RC.DATA_DIR=DATA;cifar=E.RC.Cifar10()
    return torch.stack([E.slot_inputs(cifar,s,'raw',device,center=True) for s in seeds]),cifar


def sources():
    paths=['src/drive_cifar_c_0920.py','src/relu_doors_0919.py','src/cifar_interventions_0920.py',
           'src/rlcifar_mlp_battle_0918.py','src/pmnist_0905.py','src/pmnist_rlcifar_0907.py','src/pmnist_rlmnist_0906.py',
           'analysis/cifar_ledger_0920/replay.py','analysis/cifar_ledger_0920/ledger.py','specs/spec_drive_cifar_c_0920.md']
    paths += [str(p.relative_to(ROOT)) for p in sorted((ROOT/'analysis'/RUN).glob('*.py'))]
    return {p:sha(ROOT/p) for p in paths}


class Engine(Base):
    def __init__(self,seeds,X,state=None,graph=True,observe=True):
        super().__init__(seeds,X,graph=False)
        self.act=E.make_act('C');self.act.init_state(self.R,self.device,'C')
        self.observe=observe;self.graph_enabled=graph
        self.oldY=torch.zeros_like(self.Y)
        self.conf=torch.zeros(self.R,100,101,device=self.device,dtype=torch.float64)
        self.hist=torch.zeros_like(self.conf)
        self.agg=torch.zeros(self.R,100,len(N.KEYS),device=self.device,dtype=torch.float64)
        self.maxima=torch.zeros_like(self.agg)
        self.fail_step=torch.full((),-1,device=self.device,dtype=torch.long)
        self.cg=None
        if state is not None:self.restore(state)
        self.capture()

    def mutable_tensors(self):
        return super().mutable_tensors()+[self.oldY,self.conf,self.hist,self.agg,self.maxima,self.fail_step]

    def state(self):
        d=super().state()
        d.update(cpu({k:getattr(self,k) for k in ('oldY','conf','hist','agg','maxima','fail_step')}))
        d['observe']=self.observe
        return d

    def restore(self,st):
        assert st['observe']==self.observe
        super().restore(st)
        with torch.no_grad():
            for k in ('oldY','conf','hist','agg','maxima','fail_step'):getattr(self,k).copy_(st[k])

    def step(self):
        xb,yb=self.X[self.ar,self.ids],self.Y[self.ar,self.ids]
        z1,h,z2,a2,logits=E.forward(self.P,xb,self.act,train=True)
        loss=F.cross_entropy(logits.reshape(-1,10),yb.reshape(-1),reduction='none').view(self.R,16).mean(1)
        hit=(logits.detach().argmax(-1)==yb).float().mean(1)
        self.acc_sum.add_(hit);self.last_hit.copy_(hit)
        grads=torch.autograd.grad(loss.sum(),self.P)
        with torch.no_grad():
            if self.observe:
                before=N.augmented(self.P[2],self.P[3]);mprev=N.augmented(self.m[2],self.m[3])
                f0=N.full_features(self.P,self.X)
                dec=N.decompose(h,z2,logits,self.P[4],self.oldY[self.ar,self.ids],yb)
            self.ce_sum.add_(loss.detach())
            bad=~torch.isfinite(loss)
            self.bad_step.copy_(torch.where((self.bad_step<0)&bad,self.step_t,self.bad_step))
            self.step_t.add_(1)
            for p,g,m,v in zip(self.P,grads,self.m,self.v):
                m.mul_(.9).add_(g,alpha=1-.9)
                v.mul_(.999).addcmul_(g,g,value=1-.999)
                p.sub_(.001*(m*self.inv1)/((v*self.inv2).sqrt()+1e-8))
            self.act.post_update(self.P,.001);self.act.update(z1.detach(),z2.detach())
            if self.observe:
                f1=N.full_features(self.P,self.X)
                d,cm,hm=N.measure(before,N.augmented(self.P[2],self.P[3]),mprev,N.augmented(self.m[2],self.m[3]),N.augmented(self.v[2],self.v[3]),N.augmented(grads[2],grads[3]),f0,f1,dec,self.inv1,self.inv2,self.conf,self.hist)
                self.conf.copy_(cm);self.hist.copy_(hm)
                a=N.pack(d);self.agg.add_(a);self.maxima.copy_(torch.maximum(self.maxima,a.abs()))
                fail=(d['fail']>0).any()|(d['nonfinite']>0).any()
                self.fail_step.copy_(torch.where((self.fail_step<0)&fail,self.step_t,self.fail_step))
                self.last_measure=d

    def begin_task(self,t):
        assert not self.in_task
        self.oldY.copy_(self.Y)
        self.Y.copy_(torch.stack([E.RC.task_labels(self.g_lab[s]) for s in self.seeds]).to(self.device))
        if t==1:self.oldY.copy_(self.Y) # t1 is not in any scientific denominator.
        self.task=t;self.epoch=0;self.in_task=True
        self.acc_sum.zero_();self.ce_sum.zero_();self.step_t.zero_();self.bad_step.fill_(-1)
        self.conf.zero_();self.hist.copy_(N.augmented(self.m[2],self.m[3]))
        self.agg.zero_();self.maxima.zero_();self.fail_step.fill_(-1)
        self.order_chain='';self.label_hash=tree_hash(self.Y)

    def draw_order(self,expected=None):
        order=torch.stack([torch.randperm(1200,generator=self.g_batch[s]) for s in self.seeds])
        if expected is not None:assert torch.equal(order,expected.cpu()),'audit order/RNG mismatch'
        self.order_chain=hashlib.sha256(self.order_chain.encode()+order.cpu().numpy().tobytes()).hexdigest()
        return order.to(self.device)

    def train_epoch(self,order=None):
        self.agg.zero_();self.maxima.zero_()
        order=self.draw_order(order)
        for j in range(75):self.one_update(order[:,j*16:(j+1)*16])
        self.epoch+=1
        return order

    def rows(self,epochs):
        with torch.no_grad():m,_,_=E.evaluate(self.P,self.X,self.Y,self.act,list(range(self.R)))
        return [dict(arm='C',cond='raw',seed=s,slot=r,lr=.001,task=self.task,iv='none',online_acc=float(self.acc_sum[r])/(epochs*75),memo_acc=m[r]['acc'],**m[r]) for r,s in enumerate(self.seeds)]


def identity(seeds,X,cifar,epochs,tasks,mode):
    return dict(run_id=RUN,mode=mode,seeds=list(seeds),R=len(seeds),epochs=epochs,tasks=tasks,
                source_sha256=sources(),X_sha256=tree_hash(X),data_sha256=cifar.sha256,
                subset_sha256={str(s):tree_hash(E.RC.subset_idx(s)) for s in seeds})


def run(out,seeds,epochs=400,tasks=5,mode='check',resume=False,graph=True,epoch_limit=None):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    assert len(seeds)==10 and len(set(seeds))==10
    assert (mode=='check' and seeds==list(range(100,110))) or (mode=='production' and seeds==list(range(10)) and epochs==400 and tasks==5)
    device=setup();X,cifar=inputs(seeds,device);ident=identity(seeds,X,cifar,epochs,tasks,mode)
    manifest=out/'input_manifest.json';ck=out/'checkpoint.pt'
    if manifest.exists():
        assert resume and json.loads(manifest.read_text())==ident,'source/input/config mismatch or overwrite'
        st=load_pt(ck)
        assert st['identity']==ident
    else:
        assert not any(out.iterdir()),'nonempty output has no identity'
        st=None;put(manifest,ident)
        put(out/'provenance_start.json',dict(identity=ident,git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),dirty=subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),pid=os.getpid(),environment=environment()))
    engine=Engine(seeds,X,state=st['engine'] if st else None,graph=graph)
    rows=st['rows'] if st else [];files=st['files'] if st else []
    # Shards newer than the atomic checkpoint are uncommitted work and recomputed.
    def checkpoint():save_pt(ck,dict(identity=ident,engine=engine.state(),rows=rows,files=files))
    started=time.monotonic();epoch_times=[];done=0
    for t in range(engine.task if engine.in_task else engine.task+1,tasks+1):
        if not engine.in_task:engine.begin_task(t)
        while engine.epoch<epochs:
            start=engine.state();ep=engine.epoch+1;t0=time.monotonic()
            order=engine.train_epoch();torch.cuda.synchronize();epoch_times.append(dict(task=t,epoch=ep,seconds=time.monotonic()-t0))
            if int(engine.fail_step)>=0 or bool((engine.bad_step>=0).any()):
                # Save replay-complete fixture before any abort, then recover the exact first failing update.
                save_pt(out/'failure_epoch.pt',dict(start=start,order=order,end=engine.state(),identity=ident))
                replay=Engine(seeds,X,state=start,graph=False)
                replay.draw_order(order)
                for j in range(75):
                    before=replay.state();replay.one_update(order[:,j*16:(j+1)*16])
                    if int(replay.fail_step)>=0:
                        save_pt(out/'first_failure.pt',dict(before=before,after=replay.state(),ids=order[:,j*16:(j+1)*16],measure=replay.last_measure,identity=ident));break
                put(out/'CHECK_FAILED.json',dict(task=t,epoch=ep,step=int(engine.fail_step)));raise RuntimeError('CHECK_FAILED: saved first_failure.pt')
            if ep in (1,epochs):
                p=out/'audit'/f't{t}_e{ep:03}.pt';save_pt(p,dict(start=start,order=order,end=engine.state(),identity=ident));files.append(dict(path=str(p.relative_to(out)),sha256=sha(p),kind='audit'))
            agg=engine.agg.cpu().numpy();mx=engine.maxima.cpu().numpy()
            for group,sl in [('primary',slice(0,5)),('calibration',slice(5,10))]:
                p=out/group/f't{t}_e{ep:03}.npz';save_npz(p,dict(sum=agg[sl],max=mx[sl],seeds=np.array(seeds[sl]),task=t,epoch=ep,keys=np.array(N.KEYS),steps=75,order_hash=engine.order_chain,label_hash=engine.label_hash,old_label_hash=tree_hash(engine.oldY)))
                files.append(dict(path=str(p.relative_to(out)),sha256=sha(p),kind=group))
            done+=1
            if engine.epoch==epochs:
                rows.extend(engine.rows(epochs));engine.in_task=False
            checkpoint()
            if (out/'STOP').exists() or (epoch_limit and done>=epoch_limit):return dict(status='STOPPED',task=t,epoch=ep)
        print(f'completed task {t}/{tasks}; no scientific summaries opened',flush=True)
    E.H.write_csv(out/'per_task.csv',rows)
    files.append(dict(path='per_task.csv',sha256=sha(out/'per_task.csv'),kind='host_rows'))
    put(out/'cost.json',dict(seconds=time.monotonic()-started,epochs=epoch_times,full_images=1200,every_update=True,max_cuda_bytes=torch.cuda.max_memory_allocated(),check_only=mode=='check'))
    put(out/'complete.json',dict(identity=ident,files=files,tc=engine.tc,task=engine.task,epochs=epochs,checkpoint_sha256=sha(ck)))
    put(out/'provenance_end.json',dict(status='COMPLETE',utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),tc=engine.tc))
    return dict(status='COMPLETE',tc=engine.tc)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',required=True);ap.add_argument('--mode',choices=['check','production'],default='check')
    ap.add_argument('--epochs',type=int,default=400);ap.add_argument('--tasks',type=int,default=2);ap.add_argument('--resume',action='store_true');ap.add_argument('--checks');ap.add_argument('--production-go',action='store_true')
    a=ap.parse_args()
    if a.mode=='production':
        assert a.production_go and a.checks,'production needs a separate explicit GO and validated checks'
        from analysis.drive_cifar_c_0920.checks import validate_checks
        validate_checks(json.loads(Path(a.checks).read_text()))
        assert a.tasks==5 and a.epochs==400
    with exclusive():print(run(a.out,list(range(10)) if a.mode=='production' else list(range(100,110)),a.epochs,a.tasks,a.mode,a.resume))
if __name__=='__main__':main()
