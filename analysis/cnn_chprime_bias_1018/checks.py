#!/usr/bin/env python3
"""Independent CPU synthetic checks and adversarial mutations (no evaluation seeds).

CUDA is a separate gate. This file never starts a production or reduced CPU run.
"""
from pathlib import Path
import argparse, copy, importlib.util, json, sys, time
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src import cnn_chprime_bias_1018 as M
p=importlib.util.spec_from_file_location('cnn_width_verdict',Path(__file__).with_name('verdict.py'))
V=importlib.util.module_from_spec(p);p.loader.exec_module(V)


class SyntheticCifar:
    """Analytic images, explicit synthetic source; no CIFAR result is measured here."""
    train_u8=(torch.arange(8*3072).reshape(8,3072)%251).to(torch.uint8)
    sha256={'synthetic':'not a production dataset'}
    def images(self, idx, device):
        return (((idx[:,None]*37+torch.arange(3072)[None,:]*13)%251).float()/255).to(device)


def bundle(device='cpu',enabled=True,R=1):
    return M.Bundle([('CHf_B' if enabled else 'CHf',100+s) for s in range(R)],SyntheticCifar(),
                    torch.device(device),graph=False,enabled=enabled)


def host(device='cpu',R=1):
    return M.CF.Bundle([('CHf',100+s) for s in range(R)],SyntheticCifar(),torch.device(device),graph=False)


def same(a,b):
    if isinstance(a,torch.Tensor):
        if a.dtype!=b.dtype or a.shape!=b.shape: return False
        return a.detach().contiguous().cpu().numpy().tobytes()==b.detach().contiguous().cpu().numpy().tobytes()
    if isinstance(a,dict): return a.keys()==b.keys() and all(same(a[k],b[k]) for k in a)
    if isinstance(a,(list,tuple)): return len(a)==len(b) and all(same(x,y) for x,y in zip(a,b))
    return a==b


def train_state(b):
    state=copy.deepcopy(b.full_state());state.pop('kappa',None);state.pop('enabled',None)
    return state


@torch.no_grad()
def reference(P,xb,act,stale=False,skip=None):
    """Separate NumPy float64 variance, independently written sequential affine path."""
    R,B=xb.shape[:2];x=xb.permute(1,0,2,3,4).reshape(B,3*R,32,32)
    saved=[]
    if stale:
        _,ins=M.CF.forward(P,xb,act,with_inputs=True)
    for l in range(4):
        inp=ins[l] if stale else x
        w,b=P[2*l:2*l+2]
        if l<2:
            q=F.conv2d(inp,w.reshape(16*R,-1,5,5),padding=2,groups=R)
            a=q.detach().cpu().double().numpy().reshape(B,R,16,-1)
            centered=a-a.mean(axis=0,keepdims=True)
            variance=np.mean(centered*centered,axis=0)
            width=np.sqrt(variance.mean(axis=-1))
        else:
            q=torch.bmm(inp,w.transpose(1,2))
            a=q.detach().cpu().double().numpy()
            centered=a-a.mean(axis=1,keepdims=True)
            width=np.sqrt(np.mean(centered*centered,axis=1))
        before=b.detach().cpu().numpy().copy()
        expected=np.clip(before.astype(np.float64),-.2*width,.2*width).astype(np.float32)
        if l!=skip: b.copy_(torch.tensor(expected,device=b.device))
        saved.append(dict(width=width,before=before,after=b.detach().cpu().numpy().copy()))
        if l<2:
            z=q+b.reshape(1,16*R,1,1)
            x=F.max_pool2d(z.clamp_min(0),2,2)-act.m[l].reshape(1,16*R,16//(2**l),16//(2**l))
            if l==1: x=x.reshape(B,R,1024).transpose(0,1)
        else: x=(q+b[:,None]).clamp_min(0)-act.m[l][:,None]
    return saved


def varied_bias(b):
    with torch.no_grad():
        for l in range(4): b.P[2*l+1].copy_(torch.linspace(-4,4,M.CN.WIDTHS[l],device=b.device))
        # Different widths, with stable exact-zero and extremely small channels.
        b.P[0][:,0].zero_();b.P[0][:,1].mul_(1e-20);b.P[0][:,2].mul_(.01)
        for i in range(len(b.m)):
            b.m[i].fill_(.013);b.v[i].fill_(.003)
        b.tc.fill_(11)


def rows(kept):
    return [dict(arm='CHf_B',seed=s,task=t,online_acc=.5 if s<10+kept else .499,diverged=False)
            for s in M.SEEDS for t in range(1,31)]


def run(device='cpu'):
    torch.set_num_threads(1);M.H.setup(device)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    checks={};start=time.time()
    def put(name,ok,mutations=None,**data):
        checks[name]=dict(pass_=bool(ok),mutations=mutations or {},**data)
        checks[name]['pass']=checks[name].pop('pass_')
        print(name,checks[name]['pass'],flush=True)
    # Image variation with large spatial means, 100x and 1e-25 widths and exact zero.
    v=torch.tensor([-1.,1.,-1.,1.],dtype=torch.float64)
    channel=torch.stack((v,100*v,1e-25*v,0*v),dim=1)
    q=(channel[:,:,None]+torch.tensor([0.,1e3],dtype=torch.float64)).reshape(4,4,1,2).float()
    # Use the generic channel calculation on real 16 channels, with the tail zero.
    q16=torch.zeros(4,16,1,2,device=device);q16[:,:4]=q.to(device)
    # A tiny value cannot survive a +1000 offset in f32. Give it zero spatial mean;
    # the narrow-channel fixture must contain actual representable image variation.
    q16[:,2,0,0]=(v*1e-25).float().to(device);q16[:,2,0,1]=(v*1e-25).float().to(device)
    q16[:,3].zero_()
    expected=np.sqrt(q16.cpu().double().numpy().var(axis=0).mean(axis=(1,2)))[None,:]
    got=M.image_width(q16,True,1).cpu().numpy()
    # Welford/numpy float64 operations: conservative gamma_16 width error,
    # scaled by actual width, with exact zeros checked separately.
    bound=16*np.finfo(np.float64).eps*np.abs(expected)
    ok=np.all(np.abs(got-expected)<=bound) and got[0,2]>0 and got[0,3]==0
    mixed=q16.double().var((0,2,3),correction=0).sqrt().cpu().numpy()[None,:]
    median=np.broadcast_to(np.median(expected),expected.shape)
    put('S-scale',ok,{'image_position_mixed':not np.all(np.abs(mixed-expected)<=bound),
                     'shared_layer_median':not np.all(np.abs(median-expected)<=bound)},
        actual=got.tolist(),expected=expected.tolist(),absolute_bound=bound.tolist(),tolerance_basis='gamma16*f64 width; zero separate')
    width=torch.tensor([[1.,.01,0.,1e-25]],dtype=torch.float64,device=device)
    bias=torch.tensor([[.1,-1.,.2,-1e-23]],device=device);bits=bias[:,0].clone()
    before,hit=M.clip_bias(bias,width)
    ex=np.clip(before.cpu().double().numpy(),-.2*width.cpu().numpy(),.2*width.cpu().numpy()).astype(np.float32)
    put('S-B-scalar',np.array_equal(bias.cpu().numpy(),ex) and torch.equal(bits,bias[:,0]) and bias[0,2]==0,
        {'fixed_absolute_cap':not np.array_equal(before.clamp(-.2,.2).cpu().numpy(),ex),
         'epsilon_width_floor':not np.array_equal(np.clip(before.cpu().numpy(),-.2*np.maximum(width.cpu().numpy(),1e-8),.2*np.maximum(width.cpu().numpy(),1e-8)).astype(np.float32),ex)},
        max_relative_ratio=float((bias.double().abs()/width)[torch.isfinite(bias.double()/width)].max()))
    # Disabled path, original P183 step, RNG and all optimizer/EMA state.
    b=bundle(device,enabled=False);h=host(device);b.new_labels();h.new_labels();b.new_order();h.new_order()
    for j in range(3): b.step(j);h.step(j)
    put('S-P183',same(train_state(b),train_state(h)) and not same(torch.tensor([0.]),torch.tensor([-0.])),
        {'signed_zero_byte_change':not same(torch.tensor([0.]),torch.tensor([-0.])),'wrong_mode_channel':not same(b.act.m[0].shape,torch.zeros(1,16).shape)},R=1,seeds=[100],steps=3,production_R10_pending=device=='cpu')
    # Actual training-order comparison with nonzero bias and nonzero moments.
    b=bundle(device);varied_bias(b);b.new_labels();b.new_order()
    h=host(device);h.load_full_state(train_state(b));h.order.copy_(b.order);h.Y.copy_(b.Y)
    h.step(0)
    P=[p.detach().clone() for p in h.P];act=copy.deepcopy(h.act)
    ref=reference(P,b.X[b.ar,b.order[:,:16]],act)
    b.step(0)
    expected_state=train_state(h);expected_state['P']=[p.detach().cpu().clone() for p in P]
    real=train_state(b)
    unchanged=same(real['m'],expected_state['m']) and same(real['v'],expected_state['v']) and same(real['tc'],expected_state['tc'])
    exactP=same(real['P'],expected_state['P'])
    widths=np.concatenate([row['s'][0].cpu().numpy().ravel() for row in b.trace])
    wr=np.concatenate([row['width'].ravel() for row in ref])
    allowed=32*np.finfo(float).eps*np.abs(wr)
    serror=float(np.max(np.abs(widths-wr)))
    # Build real altered candidate implementations, not just scalar expected labels.
    skipP=[p.detach().clone() for p in h.P];reference(skipP,b.X[b.ar,b.order[:,:16]],h.act,skip=3)
    staleP=[p.detach().clone() for p in h.P];reference(staleP,b.X[b.ar,b.order[:,:16]],h.act,stale=True)
    mom_mut=copy.deepcopy(real);mom_mut['m'][1].mul_(.5)
    put('S-B-state',unchanged and exactP and np.all(np.abs(widths-wr)<=allowed),
        {'omit_f2':not same(real['P'],[p.cpu() for p in skipP]),'shrink_moment':not same(real['m'],mom_mut['m'])},
        width_max_error=serror,tolerance_basis='gamma32*f64 width; bias clip rounds f32',native_optimizer_unchanged=unchanged)
    put('S-order',same(real['act'],expected_state['act']) and not same(real['P'],[p.cpu() for p in staleP]),
        {'all_widths_before_correction':not same(real['P'],[p.cpu() for p in staleP]),
         'extra_forward_updates_EMA':not same(real['act'],(lambda a: (a.update([M.CF.forward(b.P,b.X[:,:16],a)[2*l] for l in range(4)]),a.state())[1])(copy.deepcopy(b.act)))})
    act=M.CF.FeatDoors(1,'feat',torch.device(device))
    z=torch.arange(4*16*32*32,device=device,dtype=torch.float32).reshape(4,16,32,32)%17-8
    proper=F.max_pool2d(z.clamp_min(0),2,2).mean(0).reshape(1,-1)
    native=act.out_mean(z,0);wrong=F.max_pool2d(z.clamp_min(0),2,2).mean((0,2,3)).reshape(1,-1)
    zs=[z,torch.ones(4,16,16,16,device=device),torch.ones(1,4,100,device=device),torch.ones(1,4,100,device=device)]
    act.update(zs);saved=act.m[0].clone();act.update(zs)
    expected2=.99*saved+.01*proper
    put('S-Hprime',torch.equal(proper,native) and act.m[0].shape==(1,4096) and torch.equal(act.m[0],expected2),
        {'spatial_axis_averaged':wrong.shape!=native.shape,'reset_each_task':not torch.equal(act.m[0],.01*proper)},beta=.01,initial_zero=True)
    z0=torch.tensor([-1.,0.,1.],requires_grad=True,device=device);z0.clamp_min(0).sum().backward()
    native_match=torch.equal(z0.grad,(z0>=0).to(z0.dtype))
    # Stable streaming population variance, spatial confound and zero division.
    accumulator=M.Moments()
    fixture=q16.reshape(4,1,16,-1).permute(1,0,2,3).double()
    accumulator.add(fixture[:,:2]);accumulator.add(fixture[:,2:])
    statsok=torch.allclose(accumulator.var(),fixture.var(1,correction=0),rtol=32*np.finfo(float).eps,atol=0)
    r=M.ratio(torch.tensor([0.,1.,-1.]),torch.zeros(3))
    put('S-moments',statsok and native_match and torch.isnan(r[0]) and torch.isposinf(r[1]) and torch.isneginf(r[2]),
        {'gate_zgt0_called_native':not torch.equal(z0.grad,(z0>0).to(z0.dtype)),
         'mixed_variance_named_image_variance':bool((mixed[0,:2]-expected[0,:2]>1).any()),
         'ratio_of_medians':np.median(np.array([1.,10.,100.])/np.array([1.,2.,1000.]))!=np.median([1.,10.,100.])/np.median([1.,2.,1000.])},
        native_derivative_at_zero=float(z0.grad[1]))
    # Logger is allowed to allocate, but cannot mutate P/Adam/EMA or any RNG.
    b=bundle(device);b.new_labels();b.new_order();b.step(0)
    saved=copy.deepcopy(b.full_state());order=b.order.clone();labels=b.Y.clone()
    torch_rng=torch.get_rng_state().clone()
    before=train_state(b)
    d,units=M.diagnostics(b.P,b.X,b.Y,b.act,chunk=20)
    logclean=same(before,train_state(b)) and torch.equal(torch_rng,torch.get_rng_state())
    for j in (1,2): b.step(j)
    resume=bundle(device);resume.load_full_state(saved);resume.order.copy_(order);resume.Y.copy_(labels)
    for j in (1,2): resume.step(j)
    bad=copy.deepcopy(saved);bad['kappa']=.3
    try: resume.load_full_state(bad);reject=False
    except ValueError: reject=True
    bad2=copy.deepcopy(before);g=bad2['g_lab'][100];bad2['g_lab'][100]=torch.roll(g,1)
    put('S-resume',logclean and same(train_state(b),train_state(resume)) and reject,
        {'resume_changes_kappa':reject,'logger_consumes_rng':not same(before,bad2)},
        all_layer_position_keys=all(f'r_position_{tag}' in units for tag in M.CN.SITES))
    boundaries={str(k):V.classify(rows(k))['label'] for k in (0,1,2,8,9,10)}
    miss=rows(9);miss.pop();dup=rows(9)+[rows(9)[0]]
    diverged=rows(9);diverged[0]['diverged']=True
    nonfinite=rows(9);nonfinite[0]['online_acc']=float('nan')
    # Real run() writes failure/provenance when stopping before complete rows.
    import tempfile
    with tempfile.TemporaryDirectory() as temp:
        parent=Path(temp);runout=parent/'CHf_B';runout.mkdir()
        M.json_write(runout/'provenance.json',dict(status='DIVERGED'))
        M.json_write(runout/'failure.json',dict(status='DIVERGED',reason='synthetic numeric stop'))
        evidence=V.report(runout,parent/'CHf',parent/'checks.json')
        divergent_file_precedence=evidence['primary']['label']=='DIVERGED'
    expected={'0':'FAIL','1':'FAIL','2':'PARTIAL','8':'PARTIAL','9':'RESCUED','10':'RESCUED'}
    put('S-verdict',boundaries==expected and V.classify(miss)['label']=='INCOMPLETE' and V.classify(dup)['label']=='INCOMPLETE' and V.classify(diverged)['label']=='DIVERGED' and V.classify(nonfinite)['label']=='DIVERGED' and divergent_file_precedence,
        {'strict_gt_point5':V.classify(rows(9))['label']!='FAIL','eight_rescued':V.classify(rows(8))['label']!='RESCUED',
         'drop_missing_seed':V.classify(miss)['label']=='INCOMPLETE','numeric_stop_called_incomplete':divergent_file_precedence},boundaries=boundaries)
    reuse=M.reuse_audit(ROOT/'results/cnn_center_feat_1011/CHf')
    put('S-reference',reuse['qualified'] is False and any('INCOMPATIBLE_PRECISION' in p for p in reuse['problems']) and len(reuse['hashes'])==75,
        {'ignore_precision':reuse['qualified'] is not True,'seed_exchange':[['CHf',s] for s in range(0,10)]!=[['CHf',s] for s in M.SEEDS]},
        recorded_issues=reuse['problems'],verified_artifact_count=len(reuse['hashes']))
    all_pass=all(v['pass'] and all(v['mutations'].values()) for v in checks.values())
    return dict(run_id=M.RUN,scope='CPU_SYNTHETIC' if device=='cpu' else 'GPU_SYNTHETIC',all_pass=all_pass,checks=checks,
                sources=M.source_hashes(),torch=torch.__version__,seconds=time.time()-start,
                pending=['CUDA R10 P183 equality','CUDA capture rewind and graph/eager equality','GPU timing seeds100–102','independent review'])

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,default=ROOT/'results'/M.RUN/'checks.json');ap.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    args=ap.parse_args()
    if args.device=='cuda' and not torch.cuda.is_available():
        result=dict(status='BLOCKED_CUDA',all_pass=False,sources=M.source_hashes())
    else: result=run(args.device)
    M.json_write(args.out,result)
    print('all_pass',result['all_pass']);sys.exit(0 if result['all_pass'] else 2)
