"""Read-only input audit and self-contained output manifest for this analysis."""
import ast
import hashlib
import json
import platform
import sys
from importlib.metadata import version
from datetime import datetime
from pathlib import Path

WORK=Path('/home/issan/Projects/claude/wt/vit_long_1024')
OUT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_codex_bodyrank')


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def main():
    scripts=sorted((WORK/'analysis/vit_long_1024/posthoc_2blk').glob('codex_bodyrank_*.py'))
    for p in scripts:ast.parse(p.read_text())
    inputs=json.loads((OUT/'input_checkpoint_manifest.json').read_text())
    differences=[]
    for row in inputs:
        p=Path(row['path'])
        if sha(p)!=row['sha256']:differences.append(str(p))
    replay_count=0;errors=[];rss=[];gpu=[]
    for p in sorted(OUT.glob('*.json')):
        if p.name in ('audit.json','output_manifest.json'):continue
        d=json.loads(p.read_text())
        if not isinstance(d,dict):continue
        if 'rss_mb' in d:rss.append((d['rss_mb'],p.name))
        if 'cuda_peak_mb' in d:gpu.append((d['cuda_peak_mb'],p.name))
        if not ('records' in d and 'T90' in d):continue
        replay_count+=1
        reaches=[]
        for v in d['records']:
            count=v.get('correct',round(v['acc']*2500))
            if count>=2250:reaches.append(v['step'])
        target=min(reaches) if reaches else None
        if target!=d['T90']:errors.append(dict(file=p.name,reported=d['T90'],from_counts=target))
        if d['steps']>5000:errors.append(dict(file=p.name,steps=d['steps']))
        if not p.with_suffix('.npz').exists():errors.append(dict(file=p.name,error='missing final per-image margins'))
    result=dict(time=datetime.now().astimezone().isoformat(),script_ast_count=len(scripts),input_count=len(inputs),
                changed_inputs=differences,replay_count=replay_count,replay_audit_errors=errors,
                max_rss_mb=max(rss) if rss else None,max_cuda_peak_mb=max(gpu) if gpu else None)
    (OUT/'audit.json').write_text(json.dumps(result,indent=2))
    source_paths=[
        'analysis/vit_long_1024/common.py',
        'analysis/vit_long_1024/production_2blk_rlfix.json',
        'analysis/joudaki_vit_battle_0919/upstream/vit.py',
        'analysis/joudaki_vit_battle_0919/model.py',
        'analysis/joudaki_vit_battle_0919/data.py',
        'analysis/vit_long_1024/posthoc_2blk/tangent_rank.py',
    ]
    provenance=dict(time=result['time'],python=platform.python_version(),executable=sys.executable,
                    packages={p:version(p) for p in ('torch','numpy','matplotlib')},seed=10252010,
                    cuda_allocator_fraction=.15,threads=2,source_files=[dict(path=str(WORK/p),sha256=sha(WORK/p)) for p in source_paths])
    adam_inputs=[OUT.parent/'runs'/arm/'seed10252010'/'ckpt'/f'adam_t{t:02d}.pt' for arm in ('SL','R0L') for t in (10,60,100)]
    provenance['optimizer_inputs_hash_at_completion']=[dict(path=str(p),sha256=sha(p)) for p in adam_inputs]
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2))
    print(json.dumps(result,indent=2))
    assert not differences and not errors
    assert max(rss)[0]<3072 and max(gpu)[0]<8192
    manifest()


def manifest():
    scripts=sorted((WORK/'analysis/vit_long_1024/posthoc_2blk').glob('codex_bodyrank_*.py'))
    files=scripts+sorted((WORK/'results/vit_long_1024_2blk_rlfix').glob('codex_bodyrank_*.png'))
    files.append(WORK/'results/vit_long_1024_2blk_rlfix/posthoc_body_rank_codex.md')
    files+=sorted(p for p in OUT.iterdir() if p.is_file() and p.name!='output_manifest.json')
    records=[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in files]
    (OUT/'output_manifest.json').write_text(json.dumps(records,indent=2))


if __name__=='__main__':
    if '--manifest-only' in sys.argv:manifest()
    else:main()
