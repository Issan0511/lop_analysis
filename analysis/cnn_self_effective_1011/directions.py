#!/usr/bin/env python3
"""Posthoc c1 parameter geometry from quantized probes; CPU, no training.

Raw patch covariance and pool/gate effective Gram are deliberately separate.
All reported parameter states are least-squares estimates, not checkpoints.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '4')
os.environ.setdefault('OMP_NUM_THREADS', '4')
import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

SOURCE_REPO = Path('/home/issan/Projects/claude/proj_004_drift')
sys.path.insert(0, str(SOURCE_REPO))
from src import rlcifar_cnn_0908 as CN
from src import pmnist_rlcifar_0907 as RC

TASKS = [20, 30, 40, 50]
INTERVALS = [(20, 30), (30, 40), (40, 50), (20, 50)]
BANDS = [(0, 1), (1, 5), (5, 20), (20, 75)]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def median(v):
    return float(np.sort(v)[(len(v)-1)//2])


def f(v):
    return float(v)


def arrays(v):
    if isinstance(v, dict):
        return {k: arrays(x) for k, x in v.items()}
    if isinstance(v, (tuple, list)):
        return [arrays(x) for x in v]
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, np.generic):
        return v.item()
    return v


def reconstruct(seed, cifar, out):
    source = SOURCE_REPO / f'results/rlcifar_cnn_0908/LR/hist/LR_seed{seed}.npz'
    d = np.load(source)
    x = CN.images(cifar, CN.subset_idx(seed), torch.device('cpu')).double()
    xp = F.pad(x, (2, 2, 2, 2))
    A = np.stack([xp[int(i), :, int(p[0]):int(p[0])+5, int(p[1]):int(p[1])+5]
                  .reshape(-1).numpy() for i, p in zip(d['img_idx'], d['pos_idx'])])
    A = np.column_stack([A, np.ones(len(A))])
    Y = d['s_c1'].astype(np.float64).transpose(1, 0, 2).reshape(len(A), -1)
    coef, _, rank, sing = np.linalg.lstsq(A, Y, rcond=None)
    train = np.linalg.lstsq(A[:384], Y[:384], rcond=None)[0]
    W = coef[:-1].T.reshape(-1, 16, 75)
    b = coef[-1].reshape(-1, 16)
    W384 = train[:-1].T.reshape(-1, 16, 75)
    b384 = train[-1].reshape(-1, 16)
    total = torch.zeros(75, dtype=torch.float64)
    cross = torch.zeros((75, 75), dtype=torch.float64)
    n = 0
    for xx in x.split(32):
        p = F.unfold(xx, 5, padding=2).transpose(1, 2).reshape(-1, 75)
        total += p.sum(0)
        cross += p.T @ p
        n += len(p)
    mu = (total/n).numpy()
    cov = (cross/n).numpy() - np.outer(mu, mu)
    eig, Q = np.linalg.eigh(cov)
    eig, Q = eig[::-1], Q[:, ::-1]
    c = W @ Q
    means = W @ mu + b
    var = (c*c*eig).sum(2)
    norms = np.linalg.norm(W, axis=2)
    with (SOURCE_REPO/'results/rlcifar_cnn_0908/LR/per_task.csv').open() as ff:
        rows = {int(r['task']): r for r in csv.DictReader(ff) if int(r['seed']) == seed}
    stats = []
    for j, task in enumerate(d['ckpt_tasks']):
        task = int(task)
        stats.append(dict(task=task, mean=means[j], variance=var[j], norm=norms[j],
                          saved_mean=d['m_c1'][task-1],
                          norm_median_saved=f(rows[task]['w_norm_c1']),
                          norm_median=median(norms[j]),
                          sd_median=median(np.sqrt(var[j])),
                          sd_median_saved=f(rows[task]['zsd_c1'])))
    validation = dict(rank=int(rank), condition=f(sing[0]/sing[-1]),
                      fit_rmse=f(np.sqrt(np.mean((A@coef-Y)**2))),
                      holdout_rmse=f(np.sqrt(np.mean((A[384:]@train-Y[384:])**2))),
                      mean_max_abs_error=f(abs(means-d['m_c1'][d['ckpt_tasks']-1]).max()),
                      norm_median_max_relative_error=max(abs(s['norm_median']/s['norm_median_saved']-1) for s in stats),
                      sd_median_max_relative_error=max(abs(s['sd_median']/s['sd_median_saved']-1) for s in stats))
    assert rank == 76
    assert validation['norm_median_max_relative_error'] < .01, validation
    assert validation['sd_median_max_relative_error'] < .01, validation
    # Independent sensitivity to discarding one quarter of quantized probes.
    # This is not a confidence interval or a source-checkpoint comparison.
    c384 = W384 @ Q
    sensitivity = []
    lookup = {int(t): i for i, t in enumerate(d['ckpt_tasks'])}
    for t0, t1 in INTERVALS:
        i, j = lookup[t0], lookup[t1]
        dn = c[j]**2-c[i]**2
        dn384 = c384[j]**2-c384[i]**2
        dv = (dn*eig).sum(1)
        dv384 = (dn384*eig).sum(1)
        dm = (W[j]-W[i])@mu+b[j]-b[i]
        dm384 = (W384[j]-W384[i])@mu+b384[j]-b384[i]
        sensitivity.append(dict(task0=t0, task1=t1,
            shrink_sign_agreement=int(((dv<0)==(dv384<0)).sum()),
            rise_sign_agreement=int(((dm>0)==(dm384>0)).sum()),
            high5_norm_sign_agreement=int(((dn[:,:5].sum(1)<0)==(dn384[:,:5].sum(1)<0)).sum()),
            low55_norm_sign_agreement=int(((dn[:,20:].sum(1)>0)==(dn384[:,20:].sum(1)>0)).sum()),
            delta_variance_max_abs_difference=f(abs(dv-dv384).max()),
            delta_mean_max_abs_difference=f(abs(dm-dm384).max()),
            band_delta_norm2_max_abs_difference=max(f(abs((dn[:,lo:hi]-dn384[:,lo:hi]).sum(1)).max()) for lo,hi in BANDS)))
    np.savez_compressed(out/f'directions_states_seed{seed}.npz', W=W, b=b, mu=mu,
                        covariance=cov, eigenvalues=eig, eigenvectors=Q,
                        tasks=d['ckpt_tasks'], mean=means, variance=var, seed=seed,
                        W384=W384, b384=b384,
                        source_sha256=sha(source), source=str(source))
    return dict(seed=seed, source=str(source), source_sha256=sha(source),
                validation=validation, reconstruction_sensitivity=sensitivity,
                states=stats, x=x, W=W, b=b,
                mu=mu, cov=cov, eig=eig, Q=Q, tasks=d['ckpt_tasks'])


def raw_directions(state):
    W, b, mu, Q, eig = [state[k] for k in ['W', 'b', 'mu', 'Q', 'eig']]
    lookup = {int(t): i for i, t in enumerate(state['tasks'])}
    c = W@Q
    mupc = mu@Q
    out = []
    for t0, t1 in INTERVALS:
        i, j = lookup[t0], lookup[t1]
        da = c[j]-c[i]
        linear = 2*c[i]*da
        square = da*da
        dn = linear+square
        dv = dn*eig
        dm = da*mupc
        rows = []
        for h in range(16):
            bands = []
            for lo, hi in BANDS:
                sl = slice(lo, hi)
                bands.append(dict(pc=f'{lo+1}-{hi}',
                                  input_variance_fraction=f(eig[sl].sum()/eig.sum()),
                                  norm2_before=f((c[i,h,sl]**2).sum()),
                                  norm2_after=f((c[j,h,sl]**2).sum()),
                                  delta_norm2=f(dn[h,sl].sum()),
                                  delta_norm2_linear=f(linear[h,sl].sum()),
                                  delta_norm2_step2=f(square[h,sl].sum()),
                                  variance_before=f((c[i,h,sl]**2*eig[sl]).sum()),
                                  variance_after=f((c[j,h,sl]**2*eig[sl]).sum()),
                                  delta_variance=f(dv[h,sl].sum()),
                                  delta_mean=f(dm[h,sl].sum())))
            row = dict(channel=h+1, task0=t0, task1=t1,
                       mean_before=f(W[i,h]@mu+b[i,h]), mean_after=f(W[j,h]@mu+b[j,h]),
                       variance_before=f((c[i,h]**2*eig).sum()),
                       variance_after=f((c[j,h]**2*eig).sum()),
                       norm_before=f(np.linalg.norm(W[i,h])), norm_after=f(np.linalg.norm(W[j,h])),
                       delta_mean=f(dm[h].sum()+b[j,h]-b[i,h]), delta_bias=f(b[j,h]-b[i,h]),
                       delta_mu_path=f((W[j,h]-W[i,h])@mu),
                       delta_norm2=f(dn[h].sum()), delta_variance=f(dv[h].sum()), bands=bands)
            row['shrink'] = row['delta_variance'] < 0
            row['rise'] = row['delta_mean'] > 0
            rows.append(row)
        closure = dict(norm2_max_error=f(abs(dn.sum(1)-((W[j]**2).sum(1)-(W[i]**2).sum(1))).max()),
                       variance_max_error=f(abs(dv.sum(1)-np.einsum('hi,ij,hj->h', W[j],state['cov'],W[j])+np.einsum('hi,ij,hj->h',W[i],state['cov'],W[i])).max()),
                       mean_max_error=f(abs(dm.sum(1)+b[j]-b[i] - ((W[j]@mu+b[j])-(W[i]@mu+b[i]))).max()))
        assert max(closure.values()) < 1e-10
        summary = dict(shrink_channels=sum(r['shrink'] for r in rows),
                       rise_channels=sum(r['rise'] for r in rows),
                       shrink_and_rise_channels=sum(r['shrink'] and r['rise'] for r in rows),
                       norm_grows_channels=sum(r['delta_norm2'] > 0 for r in rows),
                       high5_norm_decreases_channels=int((dn[:,:5].sum(1)<0).sum()),
                       low55_norm_increases_channels=int((dn[:,20:].sum(1)>0).sum()),
                       mean_delta_variance=f(dv.sum(1).mean()),
                       mean_delta_norm2=f(dn.sum(1).mean()),
                       mean_delta_mean=f((dm.sum(1)+b[j]-b[i]).mean()),
                       bands=[dict(pc=band['pc'], **{k:f(np.mean([r['bands'][bi][k] for r in rows]))
                                                    for k in band if k != 'pc'})
                              for bi, band in enumerate(rows[0]['bands'])])
        out.append(dict(task0=t0, task1=t1, closure=closure, summary=summary, channels=rows))
    return out


def effective_directions(state, n_images, out):
    """Exact local derivative of pooled LeakyReLU w.r.t. augmented c1 filter.

    G is uncentered E[Psi Psi^T], not input covariance. Each channel, task has
    its own fixed pool winners and gates. Endpoints are compared in the START
    task's eigenbasis, so changes cannot arise from merely rotating the basis.
    """
    x = state['x'][:n_images]
    pp = F.unfold(x, 5, padding=2).transpose(1,2).contiguous()
    lookup = {int(t):i for i,t in enumerate(state['tasks'])}
    theta = np.concatenate([state['W'], state['b'][...,None]], axis=2)
    vals, winners, gates = {}, {}, {}
    Gs, qs, lambdas, mus = [], [], [], []
    for task in TASKS:
        i=lookup[task]
        wt=torch.from_numpy(state['W'][i].reshape(16,3,5,5))
        bt=torch.from_numpy(state['b'][i])
        z=F.conv2d(x,wt,bt,padding=2)
        # Leaky ReLU is monotonic; winners from z equal winners from phi(z).
        zw, ix=F.max_pool2d(z,2,2,return_indices=True)
        g=torch.where(zw>0,1.,.1)
        vals[task]=(zw*g).permute(1,0,2,3).reshape(16,-1).numpy()
        winners[task]=ix.reshape(n_images,16,-1)
        gates[task]=g.reshape(n_images,16,-1)
        gs, qe, le, me = [],[],[],[]
        for h in range(16):
            selected=pp.gather(1,winners[task][:,h,:,None].expand(-1,-1,75)).reshape(-1,75)
            aug=torch.cat([selected,torch.ones(len(selected),1,dtype=torch.float64)],1)
            psi=aug*gates[task][:,h].reshape(-1,1)
            G=(psi.T@psi/len(psi)).numpy()
            m=psi.mean(0).numpy()
            l,q=np.linalg.eigh(G)
            l,q=l[::-1],q[:,::-1]
            assert l.min()>-1e-12
            assert abs((psi@torch.from_numpy(theta[i,h])).numpy()-vals[task][h]).max()<1e-10
            gs.append(G);qe.append(q);le.append(l);me.append(m)
        Gs.append(gs);qs.append(qe);lambdas.append(le);mus.append(me)
    Gs,qs,lambdas,mus=map(np.asarray,[Gs,qs,lambdas,mus])
    np.savez_compressed(out/f'directions_effective_geometry_seed{state["seed"]}.npz',
                        tasks=TASKS, G=Gs, eigenvectors=qs, eigenvalues=lambdas,
                        effective_mean=mus, n_images=n_images,
                        cifar_indices=CN.subset_idx(state['seed'])[:n_images].numpy())
    results=[]
    for t0,t1 in INTERVALS:
        i,j=lookup[t0],lookup[t1]
        ti=TASKS.index(t0)
        rows=[]
        for h in range(16):
            q,l,G=qs[ti,h],lambdas[ti,h],Gs[ti,h]
            a=theta[i,h]@q
            da=(theta[j,h]-theta[i,h])@q
            sel=pp.gather(1,winners[t0][:,h,:,None].expand(-1,-1,75)).reshape(-1,75)
            aug=torch.cat([sel,torch.ones(len(sel),1,dtype=torch.float64)],1)
            psi=aug*gates[t0][:,h].reshape(-1,1)
            tang=(psi@torch.from_numpy(theta[j,h]-theta[i,h])).numpy()
            actual=vals[t1][h]-vals[t0][h]
            rem=actual-tang
            te=l*da*da
            dn=2*a*da+da*da
            bands=[]
            for lo,hi in [(0,1),(1,5),(5,20),(20,76)]:
                sl=slice(lo,hi)
                bands.append(dict(pc=f'{lo+1}-{hi}',gram_trace_fraction=f(l[sl].sum()/l.sum()),
                                  delta_norm2=f(dn[sl].sum()),
                                  delta_norm2_linear=f((2*a[sl]*da[sl]).sum()),
                                  delta_norm2_step2=f((da[sl]**2).sum()),
                                  tangent_energy=f(te[sl].sum()),
                                  tangent_energy_fraction=f(te[sl].sum()/te.sum()),
                                  fixed_geometry_output_second_moment_change=f((l[sl]*dn[sl]).sum())))
            row=dict(channel=h+1, output_mean_before=f(vals[t0][h].mean()),
                     output_mean_after=f(vals[t1][h].mean()),
                     output_variance_before=f(vals[t0][h].var()),
                     output_variance_after=f(vals[t1][h].var()),
                     mean_actual_change=f(actual.mean()), mean_tangent_change=f(tang.mean()),
                     mean_remainder=f(rem.mean()), actual_change_mse=f(np.mean(actual**2)),
                     tangent_energy=f(te.sum()), remainder_mse=f(np.mean(rem**2)),
                     tangent_remainder_cross=f(2*np.mean(tang*rem)),
                     pool_winner_changed_fraction=f((winners[t0][:,h]!=winners[t1][:,h]).double().mean()),
                     winner_gate_changed_fraction=f((gates[t0][:,h]!=gates[t1][:,h]).double().mean()),
                     bands=bands)
            assert abs(row['tangent_energy']-np.mean(tang**2))<1e-10
            assert abs(row['actual_change_mse']-row['tangent_energy']-row['remainder_mse']-row['tangent_remainder_cross'])<1e-10
            rows.append(row)
        results.append(dict(task0=t0,task1=t1,channels=rows,
                            summary={k:f(np.mean([r[k] for r in rows])) for k in rows[0]
                                     if k not in ['channel','bands']}))
    return dict(seed=state['seed'],n_images=n_images,samples_per_channel=n_images*256,
                geometry='uncentered Gram of phi_prime(z_winner)*(patch_winner,1); fixed start-task basis per interval',
                intervals=results)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--seeds',type=int,nargs='+',default=[0])
    ap.add_argument('--gate-images',type=int,default=128)
    ap.add_argument('--gate-seeds',type=int,nargs='*',default=[0])
    ap.add_argument('--out',type=Path,default=Path(__file__).resolve().parents[2]/'results/cnn_self_effective_1011')
    args=ap.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    cifar=RC.Cifar10()
    start=time.monotonic()
    records=[]
    for seed in args.seeds:
        state=reconstruct(seed,cifar,args.out)
        raw=raw_directions(state)
        result={k:state[k] for k in ['seed','source','source_sha256','validation','reconstruction_sensitivity','states']}
        result['raw_geometry']=dict(patches=1200*1024, dimension=75,
                                    eigenvalues=state['eig'], intervals=raw)
        if seed in args.gate_seeds and args.gate_images:
            result['effective_geometry']=effective_directions(state,args.gate_images,args.out)
        (args.out/f'directions_seed{seed}.json').write_text(json.dumps(arrays(result),indent=2))
        records.append(dict(seed=seed,validation=result['validation'],
                            intervals=[dict(task0=r['task0'],task1=r['task1'],**r['summary']) for r in raw]))
        print(json.dumps(dict(seed=seed,elapsed_seconds=time.monotonic()-start,
                              validation=result['validation'],intervals=records[-1]['intervals'])),flush=True)
    summary=dict(method='Least-squares c1 parameter estimates from 512 float16 probes; full fixed raw patch covariance; no retraining',
                 source_run='rlcifar_cnn_0908/LR',cifar_sha256=cifar.sha256,
                 bands='PC1, PC2-5, PC6-20, PC21-75 of decreasing raw input variance',
                 seeds=records,elapsed_seconds=time.monotonic()-start)
    (args.out/'directions_summary.json').write_text(json.dumps(arrays(summary),indent=2))


if __name__=='__main__':
    main()
