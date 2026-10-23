"""Synthetic KKA throughput and bitwise consistency at 1/2/4 GPU processes."""
import os
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '2'
import sys, time, json, hashlib, traceback
from pathlib import Path
import multiprocessing as mp
REPO = Path('/home/issan/Projects/claude/wt/joudaki_vit_battle_0919')
OUT = Path('/home/issan/Projects/obsidian-research-data/joudaki_vit_battle_0919/preflight/parallel')
sys.path.insert(0, str(REPO))

def worker(index, barrier, queue):
    try:
        import torch
        from torch.nn import functional as F
        from analysis.joudaki_vit_battle_0919.model import make_model, train_forward, prepare_noise, update_adaptive
        torch.set_num_threads(4)
        torch.backends.cudnn.benchmark=False
        torch.backends.cudnn.deterministic=True
        torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False
        torch.use_deterministic_algorithms(True)
        model=make_model('KKA', 100)
        fn=train_forward(model, 'compile')
        x=torch.randn(128,3,64,64,device='cuda')
        y=torch.randint(5,(128,),device='cuda')
        opt=torch.optim.Adam(model.parameters(),lr=1e-4,fused=True)
        def step():
            opt.zero_grad(set_to_none=True)
            prepare_noise(model,128)
            loss=F.cross_entropy(fn(x)[:,:5],y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),float('inf'),error_if_nonfinite=True,foreach=True)
            opt.step()
            update_adaptive(model)
        for _ in range(10): step()
        torch.cuda.synchronize()
        barrier.wait(timeout=240)
        start=time.monotonic()
        for _ in range(200): step()
        torch.cuda.synchronize()
        end=time.monotonic()
        # Keep every CUDA context alive until every process finishes its measurement.
        barrier.wait(timeout=240)
        digest=hashlib.sha256()
        for name,tensor in model.state_dict().items():
            if isinstance(tensor,torch.Tensor):
                digest.update(name.encode()); digest.update(tensor.detach().cpu().numpy().tobytes())
        queue.put(dict(worker=index,start=start,end=end,seconds=end-start,
                       ms_per_step=(end-start)*5,peak_gb=torch.cuda.max_memory_allocated()/1e9,
                       parameters_and_ema_sha256=digest.hexdigest(),status='ok'))
    except BaseException:
        queue.put(dict(worker=index,status='error',traceback=traceback.format_exc()))
        try: barrier.abort()
        except Exception: pass
        raise

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    context=mp.get_context('spawn')
    summaries=[]
    for count in [1,2,4]:
        print(json.dumps(dict(event='start',parallel=count)),flush=True)
        barrier=context.Barrier(count)
        queue=context.Queue()
        processes=[context.Process(target=worker,args=(i,barrier,queue)) for i in range(count)]
        for p in processes: p.start()
        rows=[queue.get(timeout=420) for _ in processes]
        for p in processes: p.join(timeout=30)
        summary=dict(parallel=count,workers=rows)
        if all(r['status']=='ok' for r in rows) and all(p.exitcode==0 for p in processes):
            elapsed=max(r['end'] for r in rows)-min(r['start'] for r in rows)
            summary.update(status='ok',wall_seconds=elapsed,total_steps=count*200,
                           aggregate_steps_per_second=count*200/elapsed,
                           equivalent_ms_per_step=elapsed*1000/(count*200),
                           same_digest=len({r['parameters_and_ema_sha256'] for r in rows})==1)
        else: summary['status']='failed'
        summaries.append(summary)
        (OUT/f'parallel_{count}.json').write_text(json.dumps(summary,indent=2)+'\n')
        print(json.dumps({k:v for k,v in summary.items() if k!='workers'}),flush=True)
        if summary['status']!='ok': break
    if summaries[0]['status']=='ok':
        base=summaries[0]['aggregate_steps_per_second']
        original=summaries[0]['workers'][0]['parameters_and_ema_sha256']
        for summary in summaries:
            if summary['status']=='ok':
                summary['speedup_over_one']=summary['aggregate_steps_per_second']/base
                summary['bit_equal_to_one']=all(r['parameters_and_ema_sha256']==original for r in summary['workers'])
    (OUT/'summary.json').write_text(json.dumps(summaries,indent=2)+'\n')
    print('COMPLETE',flush=True)

if __name__=='__main__': main()
