#!/usr/bin/env python3
"""On real CUDA: R10 host equality, graph rewind/replay, short seed100–102 probes.

Never CPU fallback. No evaluation performance is read. An independent review file
bound to the same sources is mandatory for release; invoking this script does not
supply one automatically. CPU and GPU checks remain separate records.
"""
from pathlib import Path
import argparse, copy, json, sys, time
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[2];sys.path.insert(0,str(ROOT))
sys.path.insert(0,str(Path(__file__).parent))
from src import cnn_chprime_bias_1018 as M
from checks import same, train_state


def check(out,review):
    if not torch.cuda.is_available():
        result=dict(status='BLOCKED_CUDA',all_pass=False,sources=M.source_hashes())
        M.json_write(out,result);return result
    device=M.H.setup('cuda');torch.set_num_threads(1)
    torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
    cifar=M.RC.Cifar10()
    slots=[('CHf',s) for s in range(100,110)]
    host=M.CF.Bundle(slots,cifar,device,graph=False)
    child=M.Bundle(slots,cifar,device,graph=False,enabled=False)
    host.new_labels();child.new_labels();host.new_order();child.new_order()
    for j in range(M.CF.SPE): host.step(j);child.step(j)
    disabled=same(train_state(host),train_state(child))
    del host,child
    eager=M.Bundle([('CHf_B',s) for s in range(100,110)],cifar,device,graph=False)
    graphed=M.Bundle([('CHf_B',s) for s in range(100,110)],cifar,device,graph=True)
    eager.new_labels();graphed.new_labels();eager.new_order();graphed.new_order()
    prior=copy.deepcopy(graphed.full_state());gprior=M.stream_hashes(graphed)
    traceprior=[{k:v.clone() for k,v in row.items()} for row in graphed.trace]
    graphed.capture()
    rewind=same(prior,graphed.full_state()) and same(traceprior,graphed.trace) and gprior==M.stream_hashes(graphed)
    eager.epoch_body();graphed.cg.replay();torch.cuda.synchronize()
    equality=same(train_state(eager),train_state(graphed)) and same(eager.trace,graphed.trace)
    del eager,graphed
    # Explicit implementation/cost probes only, never kappa calibration. Use real
    # 1200-image subsets but three epochs, seeds100–102 in their own R3 bundle.
    probes={};roundoff={}
    for enabled in (False,True):
        b=M.Bundle([('CHf_B' if enabled else 'CHf',s) for s in range(100,103)],cifar,device,graph=False,enabled=enabled)
        b.new_labels();torch.cuda.synchronize();start=time.perf_counter()
        for _ in range(3): b.run_epoch()
        torch.cuda.synchronize();elapsed=time.perf_counter()-start
        probes['B_width' if enabled else 'CHf']=dict(R=3,seeds=[100,101,102],epochs=3,updates=225,seconds=elapsed,seconds_per_step=elapsed/225)
        if enabled:
            idx=b.order[:,:M.CF.BATCH];xb=b.X[b.ar,idx]
            # Compare q+b to native bias-fused affine at each layer with identical
            # inputs. Bound gamma(fan_in+2)*sum|w*x| plus subnormal terms.
            out,ins=M.CF.forward(b.P,xb,b.act,with_inputs=True)
            for l,tag in enumerate(M.CN.SITES):
                R=b.R;w=b.P[2*l];bias=b.P[2*l+1]
                if l<2:
                    q=torch.nn.functional.conv2d(ins[l],w.reshape(R*16,-1,5,5),None,padding=2,groups=R)
                    extra=q+bias.reshape(1,R*16,1,1)
                    magnitude=torch.nn.functional.conv2d(ins[l].abs(),w.abs().reshape(R*16,-1,5,5),None,padding=2,groups=R)+bias.abs().reshape(1,R*16,1,1)
                else:
                    q=torch.bmm(ins[l],w.transpose(1,2));extra=q+bias[:,None]
                    magnitude=torch.bmm(ins[l].abs(),w.abs().transpose(1,2))+bias.abs()[:,None]
                k=2*(M.CF.D_IN[l]+2);eps=2**-24;gamma=k*eps/(1-k*eps)
                residual=(out[2*l]-extra).abs().double();bound=gamma*magnitude.double()+k*np.finfo(np.float32).smallest_subnormal
                roundoff[tag]=dict(max_abs=float(residual.max()),max_bound=float(bound.max()),pass_=bool((residual<=bound).all()),
                                   bound_basis='two affine paths: gamma_(2*(fan_in+2))*sum_abs_products + underflow bound')
        del b
    try: independent=json.loads(review.read_text())
    except (OSError,ValueError): independent={}
    reviewed=independent.get('approved') is True and independent.get('sources')==M.source_hashes() and bool(independent.get('reviewer'))
    checks=dict(R10_disabled_host_bit_exact=disabled,capture_rewind=rewind,graph_eager_bit_exact=equality,
                extra_forward_roundoff=all(v['pass_'] for v in roundoff.values()),independent_review=reviewed)
    result=dict(status='COMPLETE' if all(checks.values()) else 'CHECK_FAILED',all_pass=all(checks.values()),
                sources=M.source_hashes(),checks=checks,timing=probes,roundoff=roundoff,
                independent_reviewer=independent.get('reviewer'),review_sha256=M.sha(review) if review.exists() else None,
                device=torch.cuda.get_device_name(),torch=torch.__version__,tf32_cudnn=False,tf32_matmul=False)
    M.json_write(out,result);return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,default=ROOT/'results'/M.RUN/'gpu_release.json')
    p.add_argument('--review',type=Path,default=ROOT/'results'/M.RUN/'independent_review.json')
    a=p.parse_args();r=check(a.out,a.review);print(r['status']);sys.exit(0 if r['all_pass'] else 2)
