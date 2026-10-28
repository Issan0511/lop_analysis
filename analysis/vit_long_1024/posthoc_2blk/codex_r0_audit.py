"""Final read-only input audit, replay accounting, resources, provenance."""
import hashlib
import json
import platform
import argparse
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo

REPO=Path('/home/issan/Projects/claude/wt/vit_long_1024')
OUT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_r0')
RESULT=REPO/'results/vit_long_1024_2blk_rlfix'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()


def main():
    before=json.loads((OUT/'codex_input_hashes.json').read_text())
    changed=[];checked=0
    for name,value in before.items():
        if Path(name).name.startswith('codex_r0_'):continue
        checked+=1
        if sha(name)!=value:changed.append(name)
    assert not changed,changed
    replay={};checks=[];memory=[]
    for path in sorted(OUT.glob('codex_replay_*.json')):
        d=json.loads(path.read_text());replay[path.stem]=d
        assert 0<d['steps']<=5000
        assert d['records'][-1]['step']==d['steps']
        assert d['next_task']==d['task_end']+1
        hits=[r['step'] for r in d['records'] if r['correct']>=2250]
        actual=hits[0] if hits else None
        assert actual==d['T90'],(path,actual,d['T90'])
        for r in d['records']:
            assert abs(r['acc']-r['correct']/2500)<1e-7
        checks.append(dict(file=path.name,steps=d['steps'],T90=actual,final_acc=d['records'][-1]['acc']))
        memory.append(d['resources'])
    expected=['codex_replay_R0_t10_base']
    expected+=[f'codex_replay_{a}_t{t}_{m}' for a,t in [('R0',20),('R0',30),('R0',40),('R0',60),('S',30),('S',60)] for m in ['base','lr01','lr001','scale01']]
    expected+=[f'codex_replay_R0_t{t}_{m}_full' for t,m in [(30,'lr01'),(30,'lr001'),(60,'lr001'),(60,'scale01')]]
    assert set(expected)==set(replay),(set(expected)-set(replay),set(replay)-set(expected))
    lp=json.loads((OUT/'codex_linear_probes.json').read_text())
    assert len(lp['results'])==18
    for r in lp['results']:
        assert r['steps']==5000
        hs=[v['step'] for v in r['records'] if v['correct']>=2250]
        assert r['T90']==(hs[0] if hs else None)
    matched=json.loads((OUT/'codex_frozen_matched.json').read_text())
    assert len(matched['results'])==12
    for r in matched['results']:
        assert r['steps']==5000 and r['next_task']==r['task_end']+1 and r['loaded_adam']
        hs=[v['step'] for v in r['records'] if v['correct']>=2250]
        assert r['T90']==(hs[0] if hs else None)
        assert r['reset_head']==(r['arm']=='R0')
    prefixes=[]
    for t,m in [(30,'lr01'),(30,'lr001'),(60,'lr001'),(60,'scale01')]:
        short=replay[f'codex_replay_R0_t{t}_{m}'];long=replay[f'codex_replay_R0_t{t}_{m}_full']
        prefix=long['records'][:len(short['records'])]
        discrepancy=max(abs(a['acc']-b['acc']) for a,b in zip(short['records'],prefix))
        assert discrepancy==0,(t,m,discrepancy)
        prefixes.append(dict(task=t,mode=m,checked_evaluations=len(prefix),max_acc_difference=discrepancy))
    for name in ['codex_checkpoints.json','codex_transport.json','codex_verification.json','codex_linear_probes.json','codex_radial.json','codex_ce_self.json','codex_frozen_matched.json']:
        d=json.loads((OUT/name).read_text());memory.append(d['resources'])
    maxrss=max(r['rss_mib'] for r in memory)
    maxgpu=max(r.get('cuda_alloc_mib',0) for r in memory)
    maxreserved=max(r.get('cuda_reserved_mib',0) for r in memory)
    assert maxrss<=3072 and maxreserved<=8192
    result=dict(timestamp=datetime.now(ZoneInfo('Asia/Tokyo')).isoformat(),
                checked_existing_files=checked,changed_existing_files=changed,replay_count=len(replay),
                linear_probe_count=18,matched_frozen_count=12,repeat_prefix_checks=prefixes,
                max_rss_mib=maxrss,max_cuda_alloc_mib=maxgpu,max_cuda_reserved_mib=maxreserved,
                checks=checks,python=platform.python_version(),missing_task25=True,
                fresh_adam_tasks=[20,40],single_task_max_steps=5000,
                caveat='Allocator peaks exclude CUDA context. Serial children inherited nice 10. No git command, kill, or vault write used.')
    (OUT/'codex_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    write_manifest()
    print({k:v for k,v in result.items() if k!='checks'})


def write_manifest():
    files=list(OUT.glob('codex_*'))+[OUT/'PROGRESS.md',RESULT/'posthoc_r0_collapse_codex.md']
    files+=list((REPO/'analysis/vit_long_1024/posthoc_2blk').glob('codex_r0_*.py'))
    files+=list(RESULT.glob('codex_r0_*.png'))
    manifest=[]
    for p in sorted(files):
        if p.is_file() and p.name!='codex_manifest.json':
            manifest.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
    (OUT/'codex_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
    print('manifest updated',len(manifest))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--manifest-only',action='store_true');args=p.parse_args()
    if args.manifest_only:write_manifest()
    else:main()
