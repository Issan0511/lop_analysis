"""Forward-only input-set check of the original 100->500 learning interval."""
import argparse
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import measure as T
M,torch,np,pd=T.M,T.torch,T.np,T.pd
ROOT,DATA=T.ROOT,T.DATA


@torch.no_grad()
def collect(m,x,checks):
    frames=T.frame(m,x,checks)
    values={}
    for i in range(6):
        z=frames[i]['z']
        mu,sd=z.mean(0),z.std(0,unbiased=True)
        for key,value in dict(mu=mu,sd=sd,r=mu/(sd+1e-12),valid=sd>1e-8,
                              open=(z>0).double().mean(0)).items():
            values[f'L{i+1}_{key}']=value.cpu().numpy()
    return values,frames


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output',type=Path,default=ROOT/'results/vit_transport_1025')
    ap.add_argument('--raw-output',type=Path,default=DATA/'vit_transport_1025/raw')
    args=ap.parse_args()
    source_root=DATA/'vit_ffn_ablation_1025/source'
    sys.path.insert(0,str(source_root))
    from analysis.vit_long_1024 import common as C
    from analysis.joudaki_vit_battle_0919.model import make_model
    assert Path(C.__file__).resolve().is_relative_to(source_root)
    C.set_determinism(threads=2)
    torch.cuda.set_per_process_memory_fraction(.45)
    source=C.RealImages()
    args.output.mkdir(parents=True,exist_ok=True);args.raw_output.mkdir(parents=True,exist_ok=True)
    rows=[];drifts=[];transport=[];checks=[];provenance=[];begin=time.monotonic()
    for seed in range(5):
        archive=DATA/'vit_long_1024/_gigafile_dl'/f'vit_long_1024_{"vn" if seed<3 else "il"}_raw_seed{seed}.tar'
        xp=C.eval_tensor(source.val(C.probe_names(source,seed)),'cuda')
        with tarfile.open(archive,'r:') as tar:
            def read(name):return tar.extractfile(f'vit_long_1024/runs/SL/seed{seed}/{name}').read()
            prov=json.loads(read('provenance.json'))
            for name,digest in prov['source_hashes'].items():assert M.sha((source_root/name).read_bytes())==digest
            manifest={r['path']:r for r in json.loads(read('manifest.json'))}
            m=make_model('R',seed,'cuda',prov['config']['model_config']).eval()
            for task in (10,60):
                old,_=C.task_dataset(source,seed,task,10,25,augment=False)
                nxt,_=C.task_dataset(source,seed,task+1,10,25,augment=False)
                inputs=dict(train=C.eval_tensor(old.images,'cuda'),next=C.eval_tensor(nxt.images,'cuda'),validation=xp)
                snapshots={};allframes={};arrays={};numeric={}
                for stage in ('mid','end'):
                    name=f'ckpt/{"mid_" if stage=="mid" else "model_"}t{task:02d}.pt'
                    payload=read(name);digest=M.sha(payload);assert digest==manifest[name]['sha256']
                    m.load_state_dict(torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True))
                    provenance.append(dict(seed=seed,task=task,stage=stage,archive=str(archive),member=name,sha256=digest))
                    for population,x in inputs.items():
                        values,frames=collect(m,x,numeric);snapshots[stage,population]=values
                        allframes[stage,population]=frames
                        arrays.update({f'{stage}_{population}_{k}':v for k,v in values.items()})
                        if population=='validation':
                            with np.load(DATA/'vit_phases_1025/raw'/f'SL_seed{seed}_t{task}_{stage}.npz') as prior:
                                err=max(np.max(np.abs(values[f'L{i}_{key}']-prior[f'L{i}_{key}'])) for i in range(1,7) for key in ('mu','sd','r'))
                            assert err<2e-5
                            checks.append(dict(seed=seed,task=task,stage=stage,prior_max_abs=float(err)))
                        for i in range(1,7):
                            good=values[f'L{i}_valid'];r=values[f'L{i}_r'];op=values[f'L{i}_open']
                            rows.append(dict(seed=seed,task=task,stage=stage,population=population,n_images=len(x),layer=i,
                                             r_mean=float(r[good].mean()),r_median=float(np.median(r[good])),
                                             mu_mean=float(values[f'L{i}_mu'][good].mean()),sd_mean=float(values[f'L{i}_sd'][good].mean()),
                                             open_mean=float(op.mean()),closed_unit_fraction=float((op==0).mean())))
                for population in inputs:
                    a,b=snapshots['mid',population],snapshots['end',population]
                    terms=T.split_step(allframes['mid',population],allframes['end',population],numeric).cpu().numpy()
                    for i in range(1,7):
                        mu0,mu1=a[f'L{i}_mu'],b[f'L{i}_mu'];s0,s1=a[f'L{i}_sd']+1e-12,b[f'L{i}_sd']+1e-12
                        valid=a[f'L{i}_valid']&b[f'L{i}_valid']
                        dm=.5*(1/s0+1/s1)*(mu1-mu0);ds=.5*(mu0+mu1)*(1/s1-1/s0)
                        dr=b[f'L{i}_r']-a[f'L{i}_r'];assert np.abs((dm+ds-dr)[valid]).max()<1e-10
                        drifts.append(dict(seed=seed,task=task,population=population,layer=i,delta_mean=float(dr[valid].mean()),
                                           delta_median=float(np.median(dr[valid])),decreased_fraction=float((dr[valid]<0).mean()),
                                           mean_channel=float(dm[valid].mean()),width_channel=float(ds[valid].mean()),
                                           sd_ratio_median=float(np.median(s1[valid]/s0[valid]))))
                        tr=dict(seed=seed,task=task,population=population,layer=i,delta_mean=float(dr[valid].mean()))
                        for j,key in enumerate(('own_mean','own_width','upstream_mean','upstream_width')):
                            tr[key]=float(terms[i-1,j,valid].mean())
                        transport.append(tr)
                        arrays[f'transport_{population}_L{i}_terms']=terms[i-1]
                checks.append(dict(seed=seed,task=task,**numeric))
                np.savez_compressed(args.raw_output/f'probes_seed{seed}_t{task}.npz',**arrays)
                M.csv_write(args.output/'probe_states.csv',rows);M.csv_write(args.output/'probe_drifts.csv',drifts)
                M.csv_write(args.output/'probe_transport.csv',transport)
                M.dump(args.output/'probe_checks.json',checks);M.dump(args.output/'probe_provenance.json',provenance)
                print(json.dumps(dict(seed=seed,task=task,seconds=round(time.monotonic()-begin,1))),flush=True)
                del inputs,snapshots,allframes,frames,arrays
            del m
            torch.cuda.empty_cache()
    M.dump(args.output/'probe_run.json',dict(status='complete',states=20,seconds=time.monotonic()-begin,
        git_hash=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()))


if __name__=='__main__':main()
