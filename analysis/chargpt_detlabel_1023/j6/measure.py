#!/usr/bin/env python3
"""Exact final-FFN loading-gradient diagnostics, with no parameter updates.

J6（spec_chargpt_detlabel_1023 §6.5）の第一段階。写し元:
  analysis/chargpt_loadsign_1023/measure.py  git blob 324ef6d04bf01eea9b31600b8fad893fdad11753（commit 6aa4c490）
分解（task_fields・calculate）は変えない。変えた所は §6.5 の 4 つだけ:
  1. checkpoint の読み元を seed10 の束（tar）から <run_dir>/seed<k>/ckpts/task<t>.pt に替えた（Inputs.load）。
     それに伴い ROOT・OUT・provenance の置き場を替えた（ROOT = $DL、OUT = results/chargpt_detlabel_1023/j6/<腕>/）。
  2. 現課題の鎖 probe を腕の α で作る（load10.data の ChainM(seed, t, 0.01) を腕の α に。Inputs.data）。
     π は (r4) の pi.npz から読み（read_pi、読み口はこの関数 1 つ）、無ければ ChainM で作る。
     pi.npz を使う腕は gate で 2 状態（seed 0 の課題末 1・100）の π が ChainM の π とビットで一致することを確かめる。
  3. 前回（latepush）の unit 配列・states.csv・κ との照合を外した。autograd との照合と恒等式の検算は残す。
  4. 腕を（模型の腕, データの α）の組で持つ（arms.py。J6_ARM で選ぶ）。
足した所（分解に触れない）: run の --seeds/--tasks（検査で部分集合を回すため。既定は全部）、part の json に
  checkpoint の sha256 と π の出所を残す。
"""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
import argparse
import csv
import datetime
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import platform
import subprocess
import time

import arms as ARMS_MOD

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
LABEL, ARM, OUT = ARMS_MOD.current()
MODEL_ARM, ALPHA, RUN_DIR = ARM['model_arm'], ARM['alpha'], ARM['run_dir']
sp = importlib.util.spec_from_file_location('latepush', ROOT/'analysis/chargpt_latepush_1023/measure.py')
P = importlib.util.module_from_spec(sp); sp.loader.exec_module(P)
A, C, torch, np = P.A, P.C, P.torch, P.np
dump, write_csv, relmax = P.dump, P.write_csv, P.relative_max
GATE_TASKS = [1, 9, 50, 100]
PI_CHECK_TASKS = [1, 100]


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(4<<20), b''): h.update(b)
    return h.hexdigest()


def read_pi(seed, task):
    """(r4) の pi.npz から課題 task の π を読む。無ければ None。

    pi.npz の形は compute_W が決める。決まるまでは次の 2 つの形を受け、どちらでもなければ止める:
      (a) 課題ごとの鍵 't<task>'（または 'task<task>'）に π（4,225 値または 65×65）。任意で '<鍵>_iters'・'<鍵>_rule_sha'。
      (b) 'tasks'（課題番号の列）と 'pi'（[課題, 4,225] または [課題, 65, 65]）。任意で 'pi_iters'・'rule_sha'・'pi_sha'
          （2026-10-23 夜の src/chargpt_detlabel_1023.py の下書きはこの形: tasks・pi・rule_sha・pi_sha）。
      rule_sha（sha256(P.tobytes())）と pi_sha（sha256(π.tobytes())）があれば照合する。
    返り値 = dict(pi=65×65 float64, iters=int|None, rule_sha=str|None, path, sha256)。
    """
    p = RUN_DIR/f'seed{seed}'/'pi.npz'
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=False)
    iters = rule = pisha = None
    for key in (f't{task}', f'task{task}'):
        if key in z.files:
            pi = z[key]
            if f'{key}_iters' in z.files: iters = int(z[f'{key}_iters'])
            if f'{key}_rule_sha' in z.files: rule = str(z[f'{key}_rule_sha'])
            if f'{key}_pi_sha' in z.files: pisha = str(z[f'{key}_pi_sha'])
            break
    else:
        if 'tasks' in z.files and 'pi' in z.files:
            idx = np.flatnonzero(z['tasks'] == task)
            if len(idx) != 1:
                return None
            i = int(idx[0]); pi = z['pi'][i]
            if 'pi_iters' in z.files: iters = int(z['pi_iters'][i])
            if 'rule_sha' in z.files: rule = str(z['rule_sha'][i])
            if 'pi_sha' in z.files: pisha = str(z['pi_sha'][i])
        else:
            raise RuntimeError(f'STOP: unknown pi.npz layout {p} keys={z.files[:8]}')
    pi = np.ascontiguousarray(pi, dtype=np.float64).reshape(C.V, C.V)
    assert pi.dtype == np.float64 and np.isfinite(pi).all()
    assert pisha is None or pisha == hashlib.sha256(pi.tobytes()).hexdigest(), ('pi_sha mismatch', p, task)
    return dict(pi=pi, iters=iters, rule_sha=rule, path=str(p), sha256=sha(p))


def make_chain(seed, task, alpha, pi=None, iters=None):
    """C.ChainM そのもの。pi を渡すと、定常分布の反復（C.stationary）だけを渡した値に置き換える。"""
    if pi is None:
        return C.ChainM(seed, task, alpha)
    orig = C.stationary
    C.stationary = lambda P_: (pi, -1 if iters is None else iters)
    try:
        return C.ChainM(seed, task, alpha)
    finally:
        C.stationary = orig


class Inputs:
    def __init__(self):
        self.cache = {}
        self.pi_used = {}       # (seed, task) -> 'pi.npz' | 'ChainM'
        self.pi_checks = []     # π のビット一致の検査の記録
        self.ckpt_sha = {}

    def ckpt(self, seed, task):
        return RUN_DIR/f'seed{seed}'/'ckpts'/f'task{task}.pt'

    def load(self, arm, seed, task):
        p = self.ckpt(seed, task)
        sd = torch.load(p, map_location='cpu', weights_only=True)
        self.ckpt_sha[(seed, task)] = sha(p)
        C.L = 2
        m = C.GPT(arm, 'M'); m.load_state_dict(sd); m.eval().double()
        for b in m.h: b.mlp.track_v = False
        assert m.wte.weight is m.head.weight
        return m

    def data(self, seed, t, check_pi=False):
        """load10.data と同じ（probe の stream・本数・切り方）。ChainM の α だけを腕の α にする。"""
        if (seed, t) not in self.cache:
            got = read_pi(seed, t)
            if got is None:
                chain = make_chain(seed, t, ALPHA); self.pi_used[seed, t] = 'ChainM'
            else:
                chain = make_chain(seed, t, ALPHA, got['pi'], got['iters']); self.pi_used[seed, t] = 'pi.npz'
                rule = hashlib.sha256(chain.P.tobytes()).hexdigest()
                assert got['rule_sha'] is None or got['rule_sha'] == rule, ('rule_sha mismatch', seed, t)
                if check_pi:
                    ref = C.ChainM(seed, t, ALPHA)
                    same = bool(np.array_equal(ref.pi, got['pi'])) and ref.pi.tobytes() == got['pi'].tobytes()
                    rec = dict(seed=seed, task=t, alpha=ALPHA, bit_equal=same, rule_sha=rule,
                               pi_sha256=hashlib.sha256(got['pi'].tobytes()).hexdigest(),
                               chainm_pi_sha256=hashlib.sha256(ref.pi.tobytes()).hexdigest(),
                               iters_npz=got['iters'], iters_chainm=int(ref.pi_iters),
                               max_abs_diff=float(np.abs(ref.pi-got['pi']).max()), pi_npz=got['path'], pi_npz_sha256=got['sha256'])
                    self.pi_checks.append(rec)
                    assert same, ('STOP: pi.npz differs from ChainM', rec)
            seq = chain.sample(C.M_PROBE, C.cpu_gen(seed, 'probe', t, 'chain'))
            ev = chain.sample(C.M_EVAL, C.cpu_gen(seed, 'probe', t, 'eval'))
            self.cache[seed, t] = (seq[:, :C.T], seq[:, 1:], ev, chain.H)
        return self.cache[seed, t]

    def clear(self):
        self.cache.clear(); A.UCACHE.clear()


def median(x):
    return float(np.median(x.detach().numpy())) if x.numel() else float('nan')


def cosine(a, b):
    return (a*b).sum(-1)/(a.norm(dim=-1)*b.norm(dim=-1)).clamp_min(1e-300)


def cent(x): return x-x.mean(-1, keepdim=True)


@torch.no_grad()
def task_fields(m, x, labels):
    hs, ys, zs = [], [], []
    for _, xx in A.chunks(x):
        cap = []; h = m.body(xx, cap)
        hs.append(h.reshape(-1,C.D))
        ys.append(m.h[1].n2(cap[2]).reshape(-1,C.D))
        zs.append(cap[3].reshape(-1,C.FF))
    h, y, z = map(torch.cat, [hs,ys,zs])
    vc = m.head.weight-m.head.weight.mean(0)
    fm = (torch.nn.functional.one_hot(labels.reshape(-1),C.V).to(A.DT)-1/C.V).T@h
    kappa = (vc*fm).sum()/fm.square().sum()
    return y.mean(0), (z>0).any(0), y, vc, kappa*fm, float(kappa)


@torch.no_grad()
def calculate(m, xu, ybar, alive, vl):
    """Algebraic pullback through final LN, preserving original forward state."""
    mlp = m.h[1].mlp
    U = mlp.down.weight; Uc = U-U.mean(0,keepdim=True)
    pushes = {f'{e}_{l}_{k}':torch.zeros(C.FF,dtype=A.DT)
              for e in ['original','loaderror'] for l in ['exact','direct'] for k in ['K','absK']}
    extra = {n:torch.zeros(C.FF,dtype=A.DT) for n in ['radial','constantK','projection_cos2']}
    active = torch.zeros(C.FF,dtype=torch.bool)
    stats = {}
    def add(name, vals): stats[name] = stats.get(name, 0.)+float(vals.sum())/N
    N = xu.numel()
    for _, xx in A.chunks(xu):
        cap=[]; h=m.body(xx,cap).reshape(-1,C.D)
        r=cap[2].reshape(-1,C.D); z=cap[3].reshape(-1,C.FF)
        y=m.h[1].n2(r); gate=mlp.dphi(z); active |= (gate!=0).any(0)
        ffn=mlp.down(mlp.phi(z)); t=r+ffn
        c=cent(t); sigma=(c.square().mean(-1,keepdim=True)+m.nf.eps).sqrt(); q=c/sigma
        assert relmax(h-m.nf.weight*q,h)<=1e-12
        k=y@ybar
        logits=m.head(h); plogits=h@vl.T
        prob=logits.softmax(-1); pload=plogits.softmax(-1)
        err=prob-1/C.V; errload=pload-1/C.V
        qU=q@U
        for ename, e in [('original',err),('loaderror',errload)]:
            b=m.nf.weight*(e@vl)
            d=cent(b)/sigma
            direct=d@U
            radial=-qU*(q*b).mean(-1,keepdim=True)/sigma
            exact=direct+radial
            for lname, pull in [('exact',exact),('direct',direct)]:
                for kname, weight in [('K',k),('absK',k.abs())]:
                    pushes[f'{ename}_{lname}_{kname}']+=(weight[:,None]*gate*pull).sum(0)/N
            if ename=='original':
                extra['radial']+=(k[:,None]*gate*radial).sum(0)/N
                extra['constantK']+=(gate*exact).sum(0)/N
        extra['projection_cos2']+=(qU.square()/(q.square().sum(-1,keepdim=True)*Uc.square().sum(0)).clamp_min(1e-300)).sum(0)/N
        add('K_uniform_positive', (k>0).double()); add('K_uniform_negative',(k<0).double())
        add('sigma_mean',sigma.flatten())
        tc,rc,fc=cent(t),cent(r),cent(ffn)
        add('t_norm_mean',tc.norm(dim=-1)); add('residual_norm_mean',rc.norm(dim=-1)); add('ffn_norm_mean',fc.norm(dim=-1))
        add('ffn_over_residual_mean',fc.norm(dim=-1)/rc.norm(dim=-1).clamp_min(1e-300))
        add('ffn_residual_cos_mean',cosine(fc,rc)); add('error_cos_mean',cosine(err,errload))
        add('error_loadlogit_dot_mean',(err*plogits).sum(-1))
        add('error_loadlogit_dot_positive',((err*plogits).sum(-1)>0).double())
        add('entropy_mean',-(prob*prob.clamp_min(1e-300).log()).sum(-1))
        add('loaderror_entropy_mean',-(pload*pload.clamp_min(1e-300).log()).sum(-1))
    s=pushes['original_exact_K']; direct=pushes['original_direct_K']; radial=extra['radial']
    add_error=relmax(s-direct-radial,s,direct,radial)
    assert add_error<=1e-10
    rows={f's_{n}':A.sink(v,alive) for n,v in pushes.items()}
    rows.update({f'abs_{n}':median(v[alive].abs()) for n,v in pushes.items()})
    denom=direct[alive].abs()+radial[alive].abs(); nz=denom>0
    rows.update(stats)
    rows.update(s_radial=A.sink(radial,alive), s_constantK=A.sink(extra['constantK'],alive),
                n_gradient_active=int(active.sum()), n_alive_gradient_active=int((active&alive).sum()),
                s_nonzero=A.sink(s,alive&(s!=0)), n_zero=int((alive&(s==0)).sum()),
                cancellation=median(s[alive][nz].abs()/denom[nz]), cancellation_missing=int((~nz).sum()),
                direct_positive_to_nonpositive=float(((direct[alive]>0)&(s[alive]<=0)).double().mean()),
                direct_nonpositive_to_positive=float(((direct[alive]<=0)&(s[alive]>0)).double().mean()),
                projection_cos2_median=median(extra['projection_cos2'][alive]),additivity_error=add_error)
    return rows, {**pushes, **extra, 'alive':alive, 'gradient_active':active}


def state(inputs,arm,seed,task,autograd=False):
    m=inputs.load(MODEL_ARM,seed,task)
    x,labels,_,_=inputs.data(seed,task,check_pi=autograd and seed==0 and task in PI_CHECK_TASKS)
    ybar,alive,y,vc,vl,kappa=task_fields(m,x,labels)
    row, raw=calculate(m,A.uniform(seed),ybar,alive,vl)
    s=raw['original_exact_K']
    agerr=None
    if autograd:
        g=P.fixed_error_gradients(m,A.uniform(seed),[vl])[0][1]
        ags=(g*ybar).sum(1)
        agerr=relmax(s-ags,ags)
        assert agerr<=1e-10,(arm,seed,task,'autograd',agerr)
    key=dict(arm=arm,seed=seed,task=task,layer=1,n_alive=int(alive.sum()),kappa=kappa,
             K_task_positive=float(((y@ybar)>0).double().mean()),
             v_norm=float(vc.norm()),vL_norm=float(vl.norm()),W_norm=float(m.h[1].mlp.up.weight.norm()),
             v_over_W=float(vc.norm()/m.h[1].mlp.up.weight.norm()),
             autograd_error=agerr)
    return {**key,**row},raw


def gate(inputs):
    rows=[]
    for t in GATE_TASKS:
        r,_=state(inputs,LABEL,0,t,autograd=True); rows.append(r)
    dump(OUT/'gate.json',dict(passed=True,states=len(rows),arm=LABEL,model_arm=MODEL_ARM,alpha=ALPHA,run_dir=str(RUN_DIR),rows=rows,
        max_autograd_error=max(r['autograd_error'] for r in rows),
        pi_source={f'{s}_{t}':v for (s,t),v in inputs.pi_used.items()},pi_checks=inputs.pi_checks,
        source_module=C.__file__,source_sha256=sha(C.__file__),
        code_sha256=sha(__file__),arms_sha256=sha(HERE/'arms.py')))
    print('gate passed',flush=True)


def run(inputs,seeds,tasks):
    g=json.loads((OUT/'gate.json').read_text()); assert g['passed'] and g['code_sha256']==sha(__file__)
    rows=[]; start=time.monotonic()
    for seed in seeds:
        part=OUT/'parts'/f'{LABEL}_seed{seed}.csv'; rawpath=OUT/'raw'/f'{LABEL}_seed{seed}.npz'
        done=part.with_suffix('.json')
        if done.exists():
            d=json.loads(done.read_text()); assert d['code_sha256']==sha(__file__) and d['sha256']==sha(part) and d['raw_sha256']==sha(rawpath)
            assert d['tasks']==list(tasks)
            rows.extend(list(csv.DictReader(part.open()))); continue
        rr=[]; arrays={}; ts=time.monotonic(); timings=[]
        for t in tasks:
            t0=time.monotonic()
            row,raw=state(inputs,LABEL,seed,t); rr.append(row)
            arrays.update({f't{t}_{k}':v.numpy() for k,v in raw.items()})
            timings.append(dict(task=t,seconds=time.monotonic()-t0))
        write_csv(part,rr); rawpath.parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(rawpath,**arrays)
        dump(done,dict(sha256=sha(part),raw_sha256=sha(rawpath),code_sha256=sha(__file__),seconds=time.monotonic()-ts,
                       tasks=list(tasks),timings=timings,
                       pi_source={str(t):inputs.pi_used[seed,t] for t in tasks},
                       ckpt_sha256={str(t):inputs.ckpt_sha[seed,t] for t in tasks}))
        rows.extend(rr)
        print(f'{LABEL} seed{seed}: {time.monotonic()-ts:.1f}s',flush=True)
        inputs.clear()
    write_csv(OUT/'states.csv',rows)
    dump(OUT/'provenance.json',dict(git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
         script_sha256=sha(__file__),spec_sha256=sha(ROOT/'specs/spec_chargpt_detlabel_1023.md'),
         previous_script_sha256=sha(ROOT/'analysis/chargpt_latepush_1023/measure.py'),
         arm=LABEL,model_arm=MODEL_ARM,alpha=ALPHA,run_dir=str(RUN_DIR),seeds=list(seeds),tasks=list(tasks),
         training_steps=0,optimizer_created=False,
         host=platform.node(),torch=torch.__version__,numpy=np.__version__,threads=torch.get_num_threads(),device='cpu',dtype='float64',
         finished=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=len(rows),seconds=time.monotonic()-start))


def main():
    p=argparse.ArgumentParser(); p.add_argument('mode',choices=['gate','run'])
    p.add_argument('--seeds',type=int,nargs='+',default=list(range(10)))
    p.add_argument('--tasks',type=int,nargs='+',default=list(A.TASKS)); args=p.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    inputs=Inputs()
    if args.mode=='gate': gate(inputs)
    else: run(inputs,args.seeds,args.tasks)


if __name__=='__main__': main()
