#!/usr/bin/env python3
"""Infrastructure-interruption recovery; same cells and unchanged training code."""
import concurrent.futures, datetime, hashlib, json, os, shutil, subprocess, sys, time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
import demand_allocation_1111 as D

def main():
    raw=ROOT/'results/demand_allocation_1111/raw'
    interrupted=raw.parent/'interrupted_attempt'
    start=time.monotonic()
    source=D.sha(ROOT/'src/demand_allocation_1111.py')
    engine=D.sha(ROOT/'src/stage1_ctrl_1111.py')
    assert source=='a68bbb120a0e52183accf0d58d8476c4b31f901215be83acd98d344faafd0e15'
    assert engine=='84c9289ab18ccdd0499379a0e78cb6fe662cbf4a9d5875ebf6dac4e8b1d9f778'
    jobs=[];todo=[];moved=[]
    for act in ['R','ELU','SILU','GELU','LR']:
        for seed in range(3):
            p=raw/f'{act}_s{seed}';prov=p/'provenance.json'
            if prov.exists():
                j=json.loads(prov.read_text())
                assert j['source_sha256']==source and j['engine_sha256']==engine
                assert len(j['baseline_maxabs'])==25 and max(j['baseline_maxabs'])==0
                jobs.append(j)
            else:
                todo.append((act,seed))
                if p.exists():
                    interrupted.mkdir(exist_ok=True)
                    dest=interrupted/p.name
                    assert not dest.exists()
                    moved.append(dict(source=str(p),preserved=str(dest)))
                    shutil.move(p,dest)
    assert len(jobs)==11 and todo==[('GELU',2),('LR',0),('LR',1),('LR',2)],(len(jobs),todo)
    record=dict(tier='infrastructure recovery; no selected results',original_command_exit=143,
                cause='unknown; no experiment traceback observed',preserved=moved,
                completed_cells=[(j['act'],j['seed']) for j in jobs],rerun_cells=todo,
                source_sha256=source,engine_sha256=engine,
                started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    (raw.parent/'recovery.json').write_text(json.dumps(record,indent=2))
    with concurrent.futures.ProcessPoolExecutor(max_workers=2) as pool:
        fs=[pool.submit(D.run_job,a,s,str(raw)) for a,s in todo]
        for f in concurrent.futures.as_completed(fs):jobs.append(f.result())
    launch=json.loads((raw/'launch.json').read_text())
    elapsed=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(launch['started_utc'].replace('Z','+00:00'))).total_seconds()
    (raw/'complete.json').write_text(json.dumps(dict(jobs=jobs,seconds=elapsed,recovery_seconds=time.monotonic()-start),indent=2))
    record.update(completed_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),all_cells=15)
    (raw.parent/'recovery.json').write_text(json.dumps(record,indent=2))
    print('ALL 15 JOBS COMPLETE after recorded infrastructure recovery',flush=True)
    subprocess.run([sys.executable,str(ROOT/'analysis/demand_allocation_1111/summarize.py'),'--raw',str(raw),'--out',str(raw.parent)],check=True)

if __name__=='__main__':main()
