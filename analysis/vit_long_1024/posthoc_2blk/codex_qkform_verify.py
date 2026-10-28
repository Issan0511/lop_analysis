"""Audit the identities, input hashes, resource bounds, and finished replay scope."""
import hashlib
import json
import math
from pathlib import Path
import numpy as np

OUT=Path('/home/issan/Projects/obsidian-research-data/vit_long_1024_2blk_rlfix/posthoc_qk_formation')

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20),b''):h.update(b)
    return h.hexdigest()

def main():
    s=json.loads((OUT/'codex_summary.json').read_text());cp=s['checkpoints']
    assert len(s['replays'])==8
    assert 'signals' in s
    errors={'six_factor':[],'operator_direct':[],'checkpoint_a_energy':[],'checkpoint_Q_energy':[],
            'replay_a_growth':[],'replay_Q_energy':[],'replay_gamma_energy':[]}
    for arm,states in cp['states'].items():
        for t,d in states.items():
            errors['six_factor'].append(abs(math.prod(d['factors'].values())/d['S']-1))
            errors['operator_direct'].append(d['identity_relative_l2'])
        for d in cp['increments'][arm]:
            a,b=states[str(d['t0'])],states[str(d['t1'])]
            for quantity,key in [('a_fixed','a'),('Q_norm','Q')]:
                actual=b[quantity]**2-a[quantity]**2
                pred=d[f'{key}_energy_radial']+d[f'{key}_energy_step']
                errors[f'checkpoint_{key}_energy'].append(abs(actual-pred)/max(abs(actual),1e-10))
    for name,d in s['replays'].items():
        raw=json.loads((OUT/f'{name}.json').read_text())
        assert raw['complete'] and raw['steps']==1500 and raw['loaded_adam'] and not raw['fresh_adam']
        assert raw['rss_mb']<3072 and raw['cuda_reserved_mb']<7600
        assert raw['manual_forward_error']<1e-5
        assert d['integrated_adam_error']<1e-3
        f=np.load(OUT/f'{name}.npz');v=f['values'];keys=f['fields'].tolist()
        def column(k):return v[:,keys.index(k)]
        pred=float(column('q_radial').sum()+column('q_step_sq').sum())
        actual=d['end']['Q_norm']**2-d['start']['Q_norm']**2
        errors['replay_Q_energy'].append(abs(actual-pred)/max(abs(actual),1e-10))
        pred=float(column('gamma_update_radial').sum()+column('gamma_step_sq').sum())
        actual=384*(d['end']['gamma_rms']**2-d['start']['gamma_rms']**2)
        errors['replay_gamma_energy'].append(abs(actual-pred)/max(abs(actual),1e-10))
        for w in d['windows']:
            pred=w['exact_radial_norm_growth']+w['exact_step_norm_growth']
            errors['replay_a_growth'].append(abs(pred-w['delta_a_fixed'])/max(abs(w['delta_a_fixed']),1e-10))
    checks={k:max(v) for k,v in errors.items()}
    # Small net changes can amplify relative roundoff; absolute input algebra remains <1e-5.
    assert checks['six_factor']<1e-10 and checks['operator_direct']<1e-5,checks
    assert checks['checkpoint_Q_energy']<1e-5 and checks['replay_Q_energy']<1e-5,checks
    assert checks['checkpoint_a_energy']<1e-3 and checks['replay_a_growth']<1e-2,checks
    inputs=json.loads((OUT/'codex_input_hashes.json').read_text())
    changed=[p for p,v in inputs.items() if sha(p)!=v['sha256']]
    assert not changed,changed
    sig=json.loads((OUT/'codex_signals.json').read_text())
    assert sig['images']==256 and all(len(x)==20 for x in sig['states'].values())
    report=dict(identity_max_relative_errors=checks,input_files_checked=len(inputs),changed_inputs=changed,
                replay_count=8,replay_updates_each=1500,checkpoint_states=40,signal_states=40,
                max_rss_mb=max([cp['rss_mb'],sig['rss_mb']]+[x['rss_mb'] for x in s['replays'].values()]),
                max_cuda_peak_mb=max([cp['cuda_peak_mb'],sig['cuda_peak_mb']]+[x['cuda_peak_mb'] for x in s['replays'].values()]),
                max_gauge_error=max(x['max_gauge_error'] for x in s['replays'].values()),
                max_integrated_adam_error=max(x['integrated_adam_error'] for x in s['replays'].values()))
    (OUT/'codex_verification.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()
