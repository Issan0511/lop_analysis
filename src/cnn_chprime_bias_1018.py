#!/usr/bin/env python3
"""P183 CHf + current-image-width bias projection; no production CPU fallback.

Run: python src/cnn_chprime_bias_1018.py run --out results/cnn_chprime_bias_1018/CHf_B
Preflight is read-only except for its explicit output directory. Synthetic checks live
in analysis/cnn_chprime_bias_1018/checks.py. All historical imports remain unchanged.
"""
from __future__ import annotations
import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F
from src import cnn_center_feat_1011 as CF
from src import cnn_center_relu_1011 as CR
from src import sna_cnn_cause_1009 as E
from src import rlcifar_cnn_0908 as CN
from src import pmnist_rlcifar_0907 as RC
from src import pmnist_0905 as H

RUN = 'cnn_chprime_bias_1018'
SPEC = ROOT / 'specs/spec_cnn_chprime_bias_1018.md'
COMMON = ROOT / 'specs/spec_six_runs_common_1018.md'
KAPPA = 0.2
SEEDS = list(range(10, 20))
PROBES = (1, 10, 100, 1000)
FILES = [Path(__file__), SPEC, COMMON, *[Path(m.__file__) for m in (CF, CR, E, CN, RC, H)]]


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1048576), b''): h.update(b)
    return h.hexdigest()


def json_write(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, default=str, allow_nan=False) + '\n')


def source_hashes():
    files = list(FILES)
    # Include all transitively imported repository Python modules, plus the analysis.
    for module in list(sys.modules.values()):
        path = getattr(module, '__file__', None)
        if path and Path(path).suffix == '.py' and Path(path).is_relative_to(ROOT): files.append(Path(path))
    files.extend((ROOT / 'analysis' / RUN).glob('*.py'))
    return {str(p.relative_to(ROOT)): sha(p) for p in sorted(set(files))}


def finite_json(v):
    if isinstance(v, dict): return {k: finite_json(x) for k, x in v.items()}
    if isinstance(v, list): return [finite_json(x) for x in v]
    if isinstance(v, (float, np.floating)) and not np.isfinite(v): return None
    return v


def ratio(num, den):
    """No epsilon: +/-inf for nonzero/0 and NaN for 0/0 are retained in NPZ."""
    return num / den


def image_layout(q, conv, R):
    return q.reshape(q.shape[0], R, CN.CHANNELS, -1).permute(1, 0, 2, 3) if conv else q


def image_width(q, conv, R):
    x = image_layout(q, conv, R).double()
    var = x.var(dim=1, correction=0)
    return var.mean(-1).sqrt() if conv else var.sqrt()


@torch.no_grad()
def clip_bias(b, width, kappa=KAPPA):
    cap = kappa * width
    # The where retains every in-bound bit, including signed zero. Bound rounding
    # to float32 can exceed the real cap by <= half an ulp; retain exact f64 widths.
    candidate = b.clone()
    hit = b.double().abs() > cap
    b.copy_(torch.where(hit, b.double().clamp(-cap, cap).to(b.dtype), b))
    return candidate, hit


@torch.no_grad()
def project_bias(P, xb, act, kappa=KAPPA):
    """Sequential extra forward AFTER Adam and the host EMA update; no EMA/RNG calls.

    q is computed without bias, never by subtracting b from a large z. Downstream
    receives q + corrected b (extra-forward roundoff versus native fused affine is
    measured by independent checks). W/f3/optimizer state are not accessed for writes.
    """
    R, B = xb.shape[:2]
    value = xb.transpose(0, 1).reshape(B, R * 3, 32, 32)
    trace = []
    for l in range(4):
        w, b = P[2*l:2*l+2]
        if l < 2:
            q = F.conv2d(value, w.reshape(R * CN.CHANNELS, -1, CN.KERNEL, CN.KERNEL),
                         None, padding=CN.PAD, groups=R)
        else:
            q = torch.bmm(value, w.transpose(1, 2))
        width = image_width(q, l < 2, R)
        before, hit = clip_bias(b, width, kappa)
        good = torch.isfinite(q).flatten(1).all(1) if l >= 2 else (
            torch.isfinite(q).reshape(B, R, -1).all(2).all(0))
        good = good & torch.isfinite(width).all(1) & torch.isfinite(before).all(1)
        trace.append(dict(s=width, before=before, after=b.clone(), hit=hit, finite=good))
        if l < 2:
            z = q + b.reshape(1, R * CN.CHANNELS, 1, 1)
            value = F.max_pool2d(act.phi(z, l), CN.POOL, CN.POOL) - act.sub_map(l, act.m)
            if l == 1: value = value.reshape(B, R, CN.FLAT).transpose(0, 1)
        else:
            z = q + b[:, None, :]
            value = act.phi(z, l) - act.m[l][:, None, :]
    return trace


class Bundle(CF.Bundle):
    def __init__(self, slots, cifar, device, lr=1e-3, epochs=400, graph=True, enabled=True):
        super().__init__([('CHf', seed) for _, seed in slots], cifar, device,
                         lr=lr, epochs=epochs, graph=graph)
        self.slots, self.arm, self.enabled = slots, slots[0][0], enabled
        self.trace = [{key: torch.zeros(CF.SPE, self.R, width, dtype=dtype, device=device)
                       for key, dtype in (('s', torch.float64), ('before', torch.float32),
                                          ('after', torch.float32), ('hit', torch.bool))}
                      for width in CN.WIDTHS]

    def step(self, j):
        super().step(j)  # verbatim P183 forward -> native gradient -> Adam -> EMA
        if self.enabled:
            idx = self.order[:, j*CF.BATCH:(j+1)*CF.BATCH]
            trace = project_bias(self.P, self.X[self.ar, idx], self.act)
            for l, row in enumerate(trace):
                for key in self.trace[l]: self.trace[l][key][j].copy_(row[key])
                self.bad.logical_or_(~row['finite'])

    def _state_tensors(self):
        # P183 capture calls the virtual method, therefore it rewinds trace buffers
        # as well as P, Adam, tc, accuracy, bad, EMA and EMA update counters.
        return (*super()._state_tensors(), *(v for row in self.trace for v in row.values()))

    def full_state(self):
        state = super().full_state()
        return state | {'kappa': KAPPA, 'enabled': self.enabled}

    def load_full_state(self, state):
        if state.get('kappa') != KAPPA or state.get('enabled') != self.enabled:
            raise ValueError('checkpoint intervention differs')
        super().load_full_state(state)


class Moments:
    """Parallel Welford over images: avoid E[q²]-E[q]² loss for tiny widths."""
    def __init__(self): self.n, self.mean, self.m2 = 0, None, None
    def add(self, x):
        x = x.double()
        n = x.shape[1]
        mean = x.mean(1)
        m2 = ((x - mean[:, None]) ** 2).sum(1)
        if self.n == 0: self.n, self.mean, self.m2 = n, mean, m2
        else:
            delta = mean - self.mean
            self.m2 += m2 + delta.square() * (self.n*n/(self.n+n))
            self.mean += delta * (n/(self.n+n))
            self.n += n
    def var(self): return self.m2/self.n


@torch.no_grad()
def diagnostics(P, X, Y, act, chunk=20):
    """P183 inherited scalar/unit columns + per-position image-axis population moments."""
    rows, old = CF.diagnostics(P, X, Y, act, chunk=chunk)
    R = X.shape[0]
    zm, qm, dm, inputs = ([Moments() for _ in range(4)] for _ in range(4))
    zzero, azero, positive = [None]*4, [None]*4, [None]*4
    for i in range(0, X.shape[1], chunk):
        out, ins = CF.forward(P, X[:, i:i+chunk], act, with_inputs=True)
        bsz = out[4].shape[1]
        for l in range(4):
            z = image_layout(out[2*l], l < 2, R)
            if l < 2:
                q = F.conv2d(ins[l], P[2*l].reshape(R*CN.CHANNELS, -1, 5, 5),
                             None, padding=CN.PAD, groups=R)
                q = image_layout(q, True, R)
                inp = F.unfold(ins[l], 5, padding=CN.PAD).reshape(bsz, R, CF.D_IN[l], -1).permute(1, 0, 2, 3)
            else:
                q = torch.bmm(ins[l], P[2*l].transpose(1, 2))
                inp = ins[l]
            zm[l].add(z); qm[l].add(q); inputs[l].add(inp)
            # P183 phi is torch.clamp(min=0), whose native derivative at zero is 1.
            dm[l].add((z >= 0).double())
            for accum, val in ((zzero, z == 0), (azero, z <= 0), (positive, z > 0)):
                total = val.double().sum(1)
                accum[l] = total if accum[l] is None else accum[l] + total
    arrays = dict(old)
    for l, tag in enumerate(CN.SITES):
        mu, sd = zm[l].mean, zm[l].var().sqrt()
        amu, qsd = qm[l].mean, qm[l].var().sqrt()
        b = P[2*l+1].detach().double()
        conv = l < 2
        s = qm[l].var().mean(-1).sqrt() if conv else qsd
        cm = mu.mean(-1) if conv else mu
        ca = amu.mean(-1) if conv else amu
        mixed = (zm[l].var().mean(-1) + mu.var(-1, correction=0)).sqrt() if conv else sd
        data = dict(m_position=mu, s_position=sd, r_position=ratio(mu,sd),
                    q_mean_position=amu, q_s_position=qsd, native_d_mean=dm[l].mean,
                    native_d_var=dm[l].var(), exact_z0=zzero[l]/zm[l].n,
                    strict_activation0=azero[l]/zm[l].n,
                    dead_position=positive[l]==0, positive_fixed_position=positive[l]==zm[l].n,
                    b=b, full_image_width=s, mixed_image_position_sd=mixed,
                    b_over_s=ratio(b,s), a_over_s=ratio(ca,s), m_over_s=ratio(cm,s),
                    mean_input=inputs[l].mean, ema=act.m[l].detach(),
                    channel_width_rank=s.argsort(1),
                    identity_q_plus_b_residual=cm-ca-b,
                    w_norm=P[2*l].detach().reshape(R,CN.WIDTHS[l],-1).double().norm(dim=-1))
        for key, value in data.items(): arrays[f'{key}_{tag}'] = value.cpu().numpy()
        for r in range(R):
            rows[r].update({f'width_min_{tag}':float(s[r].min()),
                            f'width_med_{tag}':float(s[r].median()),
                            f'width_max_{tag}':float(s[r].max()),
                            f'zero_width_n_{tag}':int((s[r]==0).sum()),
                            f'native_dvar_mean_{tag}':float(dm[l].var()[r].mean())})
    return rows, arrays


def stream_hashes(B):
    return {key: {str(seed): hashlib.sha256(g.get_state().cpu().numpy().tobytes()).hexdigest()
                  for seed,g in getattr(B,key).items()} for key in ('g_lab','g_batch')}


def tensor_hash(t): return hashlib.sha256(t.detach().cpu().numpy().tobytes()).hexdigest()


def write_trace(B, path):
    arrays = {f'{key}_{tag}': val.cpu().numpy() for tag,row in zip(CN.SITES,B.trace) for key,val in row.items()}
    arrays['order'] = B.order.cpu().numpy()
    for tag,row in zip(CN.SITES,B.trace):
        s,b0,b1=row['s'],row['before'].double(),row['after'].double()
        cap=KAPPA*s
        # Absolute error uses float32 ulp at the actual rounded bias, including
        # subnormals. No denominator floor weakens tiny-channel tests.
        rounded=row['after'].abs()
        ulp=(torch.nextafter(rounded,torch.full_like(rounded,float('inf')))-rounded).double()
        violation=(b1.abs()>cap+ulp/2) & torch.isfinite(s)
        if bool(violation.any()): raise RuntimeError('CHECK_FAILED: projection exceeds rounded cap')
        for key,value in dict(fire_count=row['hit'].sum(-1),removed_l1=(b0-b1).abs().sum(-1),
                              max_ratio_before=torch.where(s==0,torch.where(b0==0,0.,float('inf')),ratio(b0.abs(),s)).amax(-1),
                              max_ratio_after=torch.where(s==0,torch.where(b1==0,0.,float('inf')),ratio(b1.abs(),s)).amax(-1),
                              zero_width_count=(s==0).sum(-1)).items():
            arrays[f'{key}_{tag}']=value.cpu().numpy()

    np.savez_compressed(path, **arrays)
    # Finite flags are checked before any subsequent epoch and never turned into accuracy zero.
    if bool(B.bad.any()): raise FloatingPointError('DIVERGED: nonfinite loss/q/width/bias')


def reuse_audit(path):
    """Qualify primary CSV/provenance/source blobs and all archived mechanism inputs."""
    path = Path(path)
    problems, unavailable, hashes = [], [], {}
    try:
        p = json.loads((path/'provenance.json').read_text())
        expected = {'run_id':'cnn_center_feat_1011','arm':'CHf','epochs':400,'lr':1e-3,'n_tasks':30,'h_mode':'feat'}
        for key,value in expected.items():
            if p.get(key)!=value: problems.append(f'metadata:{key}')
        if p.get('slots') != [['CHf',s] for s in SEEDS]: problems.append('seed_or_slot_order')
        if p.get('tf32_cudnn') is not False or p.get('tf32_matmul') is not False:
            problems.append('INCOMPATIBLE_PRECISION: new spec TF32 off; P183 recorded cudnn TF32 on')
        for key,source in (('engine_sha256',Path(CF.__file__)),('parent_engine_sha256',Path(CR.__file__))):
            if p.get(key)!=sha(source): problems.append('source:'+key)
        for git in p.get('gits',[]):
            for source,key in (('src/cnn_center_feat_1011.py','engine_sha256'),('src/cnn_center_relu_1011.py','parent_engine_sha256')):
                blob=subprocess.run(['git','show',git['git_hash']+':'+source],cwd=ROOT,capture_output=True)
                if blob.returncode or hashlib.sha256(blob.stdout).hexdigest()!=p[key]: problems.append('blob:'+source)
        rows = list(csv.DictReader((path/'per_task.csv').open()))
        seen = set()
        for row in rows:
            k=(int(row['seed']),int(row['task']))
            if k in seen: problems.append('duplicate:'+str(k))
            seen.add(k)
            if row['arm']!='CHf' or not np.isfinite(float(row['online_acc'])) or row['diverged'].lower()=='true': problems.append('invalid:'+str(k))
        if seen!={(s,t) for s in SEEDS for t in range(1,31)}: problems.append('missing_or_extra_cells')
        manifest=json.loads((ROOT/'results/cnn_center_feat_1011/backup_manifest.json').read_text())
        archived={r['source']:r for r in manifest['files']}
        sources=['results/cnn_center_feat_1011/CHf/units.npz','results/cnn_center_feat_1011/CHf/state.pt']
        sources += [f'results/cnn_center_feat_1011/CHf/ckpt/CHf_seed{s}_t{t:02d}.pt' for s in SEEDS for t in CF.CKPT_TASKS]
        for source in sources:
            entry=archived.get(source)
            file=Path(entry['backup']) if entry else ROOT/source
            if not file.exists(): problems.append('missing:'+source)
            else:
                hs=sha(file); hashes[str(file)]=hs
                if entry and hs!=entry['sha256']: problems.append('hash:'+source)
        for name in ('per_task.csv','provenance.json','t0.csv'):
            hashes[str(path/name)]=sha(path/name)
        unavailable=['new per-position image-axis derivative and q-width diagnostics: NOT_AVAILABLE in historical P183']
    except (OSError,KeyError,ValueError) as err: problems.append(str(err))
    return dict(qualified=not problems,problems=problems,hashes=hashes,not_available=unavailable)


def preflight(out, baseline):
    out.mkdir(parents=True,exist_ok=True)
    avail=torch.cuda.is_available()
    reuse=reuse_audit(baseline)
    record=dict(run_id=RUN, timestamp=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),
                status='READY' if avail else 'BLOCKED_CUDA', cuda_available=avail,
                torch=torch.__version__, cuda_build=torch.version.cuda,
                nvidia_devices=[str(p) for p in Path('/dev').glob('nvidia*')],
                sources=source_hashes(), spec_hash=sha(SPEC), common_hash=sha(COMMON),
                git_hash=CF.git_state()['git_hash'], reference=reuse,
                prediction=dict(label='RESCUED',attribution='Codex formalization of Issa instruction; common §8'),
                protocol=dict(seeds=SEEDS,tasks=30,epochs=400,batch=16,steps_per_task=30000,lr=1e-3,
                              kappa=KAPPA,R=10,tf32_cudnn=False,tf32_matmul=False),
                uncompressed_projection_trace_bytes=30*400*75*10*sum(CN.WIDTHS)*(8+4+4+1),
                disk_free_bytes=shutil.disk_usage(out).free,
                pending=['CUDA disabled-B equality R10 against P183','CUDA graph/eager equality and warmup rewind',
                         'GPU seed100–102 timing/extra-forward rounding','independent review',
                         'same-precision baseline qualification or fresh CHf run'])
    json_write(out/'preflight.json',record)
    if not (out/'provenance.json').exists(): json_write(out/'provenance.json',record)
    shutil.copyfile(SPEC,out/'spec.md')
    shutil.copyfile(COMMON,out/'common_spec.md')
    return record


def run(out, arm, baseline, checks):
    pf=preflight(out,baseline)
    if not pf['cuda_available']: return 'BLOCKED_CUDA'
    ck=json.loads(Path(checks).read_text())
    if ck.get('all_pass') is not True or ck.get('sources')!=source_hashes():
        raise RuntimeError('CHECK_FAILED: missing/stale CPU checks')
    # CPU checks do not certify CUDA. Keep this explicitly fail-closed until the
    # independent GPU checker writes a source-bound release record.
    release=out.parent/'gpu_release.json'
    if not release.exists(): raise RuntimeError('CHECK_FAILED: GPU release record absent; see preflight.pending')
    approval=json.loads(release.read_text())
    if approval.get('all_pass') is not True or approval.get('sources')!=source_hashes() or not approval.get('independent_reviewer'):
        raise RuntimeError('CHECK_FAILED: GPU release incomplete/stale')
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    reserve=2*pf['uncompressed_projection_trace_bytes'] if arm=='CHf_B' else 8*2**30
    if pf['disk_free_bytes'] < reserve:
        raise RuntimeError(f'BLOCKED_STORAGE: need {reserve} free bytes including diagnostics/checkpoints')
    device=H.setup('cuda')
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cuda.matmul.allow_tf32=False
    cifar=RC.Cifar10()
    B=Bundle([(arm,s) for s in SEEDS],cifar,device,enabled=arm=='CHf_B')
    (out/'ckpt').mkdir(exist_ok=True);(out/'diagnostics').mkdir(exist_ok=True);(out/'projection').mkdir(exist_ok=True)
    config=pf['protocol'] | {'arm':arm,'sources':pf['sources'],'data_sha256':cifar.sha256}
    rows=[];curves=np.zeros((10,30,400),np.float32);start=1;resume=[]
    if (out/'state.pt').exists():
        st=torch.load(out/'state.pt',map_location='cpu',weights_only=False)
        if st['config']!=config: raise RuntimeError('resume config/source mismatch')
        B.load_full_state(st['state']);rows=st['rows'];curves=st['curves'];start=st['task']+1;resume=st['resume']+[start]
    provenance=pf | dict(status='RUNNING',config=config,resume=resume,initial_parameter_hashes=[tensor_hash(p) for p in B.P],
                        subsets={str(s):CN.subset_idx(s).tolist() for s in SEEDS},
                        derivative='native torch.clamp: 1 at z=0; conventional gate z>0 stored separately',
                        legacy_sd='P183 diagnostic sample SD mixes images and positions; new *_position correction=0',
                        execution='eager epochs 1/2/14 for exact update probes, captured epochs otherwise; identical R10 ops',
                        projection_order='native training, Adam, pre-update-z EMA, sequential bias-free q -> clip -> q+b',
                        divergence_check='stop at first host synchronization (each step in probe epochs, otherwise epoch)',
                        roundoff='extra q+b path differs from fused bias; independent GPU release records residuals',
                        reused_reference=False, reference_status='PENDING_FRESH_CHf' if not pf['reference']['qualified'] else 'QUALIFIED',
                        initial_identity='t0' if start==1 else f'resume task {start}; hashes describe restored state')
    json_write(out/'provenance.json',provenance)
    units={}
    def snapshot(t,update,label):
        before=stream_hashes(B)
        dg,un=diagnostics(B.P,B.X,B.Y,B.act)
        if before!=stream_hashes(B): raise RuntimeError('logger consumed training RNG')
        np.savez_compressed(out/'diagnostics'/f't{t:02d}_u{update:05d}_{label}.npz',**un)
        return dg,un
    if start==1:
        ev=CF.evaluate(B.P,B.X,B.Y,B.act);dg,un=snapshot(0,0,'init')
        H.write_csv(out/'t0.csv',[dict(arm=arm,seed=s,task=0,**ev[r],**dg[r]) for r,s in enumerate(SEEDS)])
        units[0]=un
    begin=time.time()
    try:
        for t in range(start,31):
            B.new_labels();B.act.hupd.zero_();snapshot(t,0,'start')
            stream_before=stream_hashes(B);label_hash=tensor_hash(B.Y)
            sce,sacc=CF.switch_eval(B.P,B.X,B.Y,B.act)
            ep_acc=torch.zeros(10,400,device=device)
            order_hashes=[]
            for e in range(400):
                if e in (0,1,13):
                    B.new_order();B.acc_ep.zero_()
                    for j in range(CF.SPE):
                        B.step(j)
                        if bool(B.bad.any()): raise FloatingPointError('nonfinite step in eager epoch')
                        update=e*CF.SPE+j+1
                        if update in PROBES: snapshot(t,update,'switch')
                else: B.run_epoch()
                ep_acc[:,e].copy_(B.acc_ep)
                order_hashes.append(tensor_hash(B.order))
                if B.enabled: write_trace(B,out/'projection'/f't{t:02d}_e{e+1:03d}.npz')
                if bool(B.bad.any()): raise FloatingPointError('nonfinite epoch')
            ep_acc/=CF.SPE;curves[:,t-1]=ep_acc.cpu().numpy()
            online=ep_acc.double().mean(1);ev=CF.evaluate(B.P,B.X,B.Y,B.act);dg,un=snapshot(t,30000,'end')
            units[t]=un
            for r,s in enumerate(SEEDS):
                cur=curves[r,t-1];hit=np.flatnonzero(cur>=.99)
                rows.append(dict(arm=arm,seed=s,task=t,online_acc=float(online[r]),switch_ce=float(sce[r]),
                                 switch_acc=float(sacc[r]),diverged=False,ep_first99=int(hit[0]+1) if hit.size else -1,
                                 acc_ep1=float(cur[0]),acc_ep10=float(cur[9]),**ev[r],**dg[r]))
            if t in CF.CKPT_TASKS:
                for r,s in enumerate(SEEDS):
                    E.save_atomic(B.run_state(r)|dict(kappa=KAPPA,enabled=B.enabled),out/'ckpt'/f'{arm}_seed{s}_t{t:02d}.pt')
            H.write_csv(out/'per_task.csv',rows);np.save(out/'curves.npy',curves)
            # P183-compatible units archive; merge earlier snapshots after restart.
            prior={}
            for tt in range(t+1):
                fn=out/'diagnostics'/('t00_u00000_init.npz' if tt==0 else f't{tt:02d}_u30000_end.npz')
                if fn.exists():
                    with np.load(fn) as z: prior[tt]={k:z[k] for k in z.files}
            CF.save_units(out/'units.npz',prior,30,10)
            E.save_atomic(dict(config=config,state=B.full_state(),rows=rows,curves=curves,task=t,resume=resume),out/'state.pt')
            json_write(out/f'stream_t{t:02d}.json',dict(label_hash=label_hash,batch_order_hashes=order_hashes,streams_before=stream_before,streams_after=stream_hashes(B)))
            print(f'task {t:2d} complete; elapsed {time.time()-begin:.1f}s',flush=True)
    except RuntimeError as err:
        json_write(out/'failure.json',dict(status='CHECK_FAILED',reason=str(err),task=t,epoch=e+1))
        provenance['status']='CHECK_FAILED';json_write(out/'provenance.json',provenance);return 'CHECK_FAILED'
    except FloatingPointError as err:
        json_write(out/'failure.json',dict(status='DIVERGED',reason=str(err),task=t,epoch=e+1))
        provenance['status']='DIVERGED';json_write(out/'provenance.json',provenance);return 'DIVERGED'
    provenance.update(status='COMPLETE',wall_seconds=time.time()-begin,final_streams=stream_hashes(B),
                      artifact_hashes={str(p.relative_to(out)):sha(p) for p in sorted(out.rglob('*')) if p.is_file() and p.name!='provenance.json'})
    json_write(out/'provenance.json',provenance)
    return 'COMPLETE'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('mode',choices=('preflight','run'))
    ap.add_argument('--out',type=Path,default=ROOT/'results'/RUN/'CHf_B')
    ap.add_argument('--baseline',type=Path,default=ROOT/'results/cnn_center_feat_1011/CHf')
    ap.add_argument('--checks',type=Path,default=ROOT/'results'/RUN/'checks.json')
    ap.add_argument('--arm',choices=('CHf','CHf_B'),default='CHf_B')
    args=ap.parse_args()
    result=preflight(args.out,args.baseline)['status'] if args.mode=='preflight' else run(args.out,args.arm,args.baseline,args.checks)
    print(result)
    return 0 if result in ('READY','COMPLETE') else 2

if __name__=='__main__': sys.exit(main())
