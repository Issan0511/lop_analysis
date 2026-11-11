"""Validate frozen execution provenance and all saved source forks, without training."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from analysis.readout_separation_1111 import summarize as S
from analysis.readout_separation_1111 import summarize_probe as P

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root',type=Path,default=S.OUT);a=ap.parse_args()
    records=S.load_records(a.root/'raw');S.design_check(records)
    docs=P.read_all(a.root/'probes')
    probes={(d['model'],d['seed']):d for d in docs}
    sources={};codes={};fork_provenance={};runtime_git=set();legacy=0;states=0
    for r in records:
        assert not r.get('smoke',False)
        prov=r['provenance'];key=(r['model'],r['seed'])
        assert prov['spec_sha256']==S.sha(S.ROOT/'specs/spec_readout_separation_1111.md')
        for path,value in prov['code_hashes'].items():
            assert S.sha(S.ROOT/path)==value,('Changed execution source',path)
            assert codes.get(path,value)==value
            codes[path]=value
        for path,value in prov['sources'].items():
            assert sources.get(path,value)==value
            sources[path]=value
        p=probes[key]
        pp=p['provenance'];runtime_git.add(prov['git_hash'])
        for k in ('git_hash','spec_sha256','sources','inputs_sha256'):
            assert pp[k]==prov[k],('Probe/continuation provenance mismatch',k)
        assert pp['new_labels_sha256']==prov['labels_sha256']
        required={f'analysis/readout_separation_1111/{name}.py' for name in ('ops','run','probe','launch')}
        assert required<=set(pp['helpers']) and required<=set(prov['code_hashes'])
        assert prov['probe_sha256']==p['_sha256']
        fixed={k:prov[k] for k in ('batch_sha256','labels_sha256','inputs_sha256','source_state_sha256','source_metadata_sha256')}
        assert fork_provenance.get(key,fixed)==fixed
        fork_provenance[key]=fixed
        for path,value in p['provenance']['helpers'].items():
            assert S.sha(S.ROOT/path)==value,('Changed probe source',path)
        assert r['fork_hashes']==p['source_fingerprints']
        assert r['post_hashes']['body']==r['fork_hashes']['body']
        empty=hashlib.sha256(b'').hexdigest()
        assert r['post_hashes']['head_adam']==empty
        if r['arm']['body_memory']=='keep':
            assert r['post_hashes']['body_adam']==r['fork_hashes']['body_adam']
        else:assert r['post_hashes']['body_adam']==empty
        if r['arm']['family']!='physical':
            assert r['pre_release'] and r['pre_release']['step']==200
        check=r['checks']['legacy_KC']
        if check:
            legacy+=1
            assert S.sha(check['path'])==check['sha256']
            old={q['step']:q for q in json.loads(Path(check['path']).read_text())['curve']}
            pairs=[(q,old[q['step']]) for q in r['curve'] if q['step'] in old]
            x=np.array([[q['acc'],q['ce']] for q,_ in pairs])
            y=np.array([[q['acc'],q['ce']] for _,q in pairs])
            assert np.array_equal(x,y) if r['model']=='mlp' else np.allclose(x,y,rtol=2e-5,atol=2e-5)
        source=(prov['source_state_file'],prov['source_metadata_file'])
        if key not in sources:
            assert S.sha(source[0])==prov['source_state_sha256']
            assert S.sha(source[1])==prov['source_metadata_sha256']
            sources[key]=source;states+=1
    for path,value in list(sources.items()):
        if isinstance(path,str):assert S.sha(path)==value,('Changed archived input',path)
    assert legacy==30 and states==15 and len(runtime_git)==1
    record=dict(complete=True,curves=len(records),frozen_source_states=states,legacy_controls=legacy,
                all_fork_fingerprints_match=True,runtime_git_hash=next(iter(runtime_git)),execution_code_hashes=codes,
                code_sha256=S.sha(__file__))
    (a.root/'integrity_validation.json').write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record))

if __name__=='__main__':main()
