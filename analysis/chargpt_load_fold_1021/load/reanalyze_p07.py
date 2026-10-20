#!/usr/bin/env python3
"""Audit correction: paired current-position gradients; original arrays are retained."""
import csv, hashlib, json, time
from pathlib import Path
import numpy as np
import torch
import measure as M
O = M.OUT

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def next_gradient(m, x, y):
    # Full 128-token context; exclude only the first prediction from the loss.
    m.zero_grad(set_to_none=True)
    for i, xx in M.chunks(x):
        (M.ce(m(xx)[:, 1:], y[i:i+len(xx), 1:]) * len(xx)/len(x)).backward()
    return {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
            for n, p in m.named_parameters()}

def sink(g, alive):
    return float((g[alive] > 0).double().mean()) if alive.any() else float('nan')

def main():
    start = time.time()
    pv = json.loads((O/'provenance.json').read_text())
    # Verify the inputs and previous outputs before changing any measurements.
    for name, sha in pv['outputs'].items():
        assert digest(O/name) == sha, name
    for name, meta in pv['inputs'].items():
        assert digest(Path(name)) == meta['sha256'], name
    audit_hash = digest(Path(__file__).with_name('codex_audit.md'))
    rows = list(csv.DictReader((O/'table_units.csv').open()))
    states = list(csv.DictReader((O/'table_head.csv').open()))
    last = {(b,a): max(int(r['task']) for r in states if r['box']==b and r['arm']==a and r['stage']=='end') for b,a in M.ARMS}
    xu = M.C.uniform_probe_M(M.SEED, 'cpu')
    results = []
    for idx, state in enumerate(states):
        b,a,t,s = state['box'],state['arm'],int(state['task']),state['stage']
        m = M.load(b,a,t,s); l = len(m.h)-1
        x,_,_,_ = M.data(max(1,t)); xn,yn,_,_ = M.data(t+1)
        representative = s=='end' and (t in (1,last[b,a]) or (b=='M' and a in ('SiLU','GELU') and t==3))
        path = O/f'units_{b}_{a}_t{t}{"_u100" if s=="u100" else ""}.npz'
        with np.load(path) as z:
            payload = {k:z[k].copy() for k in z.files}
        y,z = M.moments(m,x)[-1]; ybar=y.mean(0); alive=(z>0).any(0)
        assert np.array_equal(alive.numpy(),payload[f'L{l}_alive'])
        assert np.allclose(z.mean(0).numpy(),payload[f'L{l}_m'],rtol=0,atol=1e-12)
        gu = M.gradient(m,xu); gn = next_gradient(m,xn,yn)
        ownu=(gu[f'h.{l}.mlp.up.weight']*ybar).sum(1)
        ownn=(gn[f'h.{l}.mlp.up.weight']*ybar).sum(1)
        assert np.allclose(ownu.numpy(),payload[f'L{l}_own'],rtol=1e-11,atol=1e-12)
        r=dict(box=b,arm=a,task=t,stage=s,n_alive=int(alive.sum()),
               uniform_own_sink=sink(ownu,alive),fixed_next_own_sink=sink(ownn,alive),
               fixed_next_minus_uniform=sink(ownn,alive)-sink(ownu,alive),
               full_recomputed=int(representative))
        payload['p07_uniform_own_last']=ownu.numpy()
        payload['p07_fixed_next_own_last']=ownn.numpy()
        payload['p07_fixed_alive_last']=alive.numpy()
        if representative:
            fu=M.full(m,x,gu); fn=M.full(m,x,gn)
            assert np.allclose(fu.numpy(),payload['G_full_last'],rtol=1e-11,atol=1e-12)
            r.update(p07_uniform_full_sink=sink(fu,alive),fixed_next_full_sink=sink(fn,alive),fixed_next_full_minus_uniform=sink(fn,alive)-sink(fu,alive))
            payload['p07_uniform_full_last']=fu.numpy()
            payload['p07_fixed_next_full_last']=fn.numpy()
        original=next(q for q in rows if (q['box'],q['arm'],int(q['task']),q['stage'],int(q['layer']))==(b,a,t,s,l))
        r['legacy_next_minus_uniform']=float(original['next_minus_uniform'])
        for key,value in r.items():
            if key not in ('box','arm','task','stage','n_alive'):
                original[key]=value
        np.savez_compressed(path,**payload)
        results.append(r)
        M.writecsv('table_p07.csv',results)
        print(f'{idx+1}/{len(states)} {b}/{a} t{t} {s} own_delta={r["fixed_next_minus_uniform"]:.9f} full={representative} elapsed={time.time()-start:.1f}s',flush=True)
    M.writecsv('table_units.csv',rows)
    # Reclassify the existing numerical evaluations, without pretending to rerun them.
    ck=json.loads((O/'checks.json').read_text())
    for r in ck['checks']:
        if r['check']=='recorded_eval_rounding':
            r['registered_rounding_pass']=r['abs_error']<=r['csv_rounding_bound']
            r['float32_adjusted_pass']=r['abs_error']<=r['tolerance']
            r['dtype_evidence']=r['dtype_evidence'].replace('user-authorized','parent (Claude, V13 session)-directed')
    ev=[r for r in ck['checks'] if r['check']=='recorded_eval_rounding']
    ck['evaluation_criteria']=dict(registered_rounding=dict(passed=sum(r['registered_rounding_pass'] for r in ev),total=len(ev),criterion='.9g rounding bound only'),float32_adjusted=dict(passed=sum(r['float32_adjusted_pass'] for r in ev),total=len(ev),criterion='2e-6 * abs(recorded) + .9g rounding bound'),change_authority='親（Claude、V13 セッション）の指示で変更',reclassified_existing_evaluations=True)
    ck['registered_all_pass']=all(r.get('registered_rounding_pass',r['pass_']) for r in ck['checks'])
    ck['all_pass_criterion']='float32-adjusted evaluation tolerance; not original registration'
    note='recorded_eval_rounding only: rtol 2e-6 + .9g rounding bound; 親（Claude、V13 セッション）の指示で変更'
    ck['provenance']['rounding_exception']=note
    pv['rounding_exception']=note
    pv['source_head_note']='計算時の source HEAD は 6c80d308d0763ef1f88a8a1a78425159fb4744c9。spec の f71be184 との差は対象2ソースでは chargpt_cl_1019_kune.py の更新100の Adam 状態保存1行のみ。forward・勾配・鎖生成は不変（独立監査と git diff で確認）。'
    pv['p07_reanalysis']=dict(script_sha256=digest(Path(__file__)),measure_script_sha256=digest(Path(M.__file__)),source_head_at_reanalysis=M.provenance()['source_head'],own_states=len(results),full_states=sum(r['full_recomputed'] for r in results),definition='current m_k and A_k fixed; uniform loss all 128 positions; next-task CE excludes first position (127 targets), full input context retained',full_selection='each arm task1 and last end state, plus M/SiLU and M/GELU task3 end; chosen before recomputation',legacy='next_minus_uniform and next_own_last/next_alive_last unchanged (next position/set, 128-position CE)',audit_sha256=audit_hash,elapsed_seconds=time.time()-start)
    assert digest(Path(__file__).with_name('codex_audit.md'))==audit_hash
    (O/'checks.json').write_text(json.dumps(ck,indent=2,ensure_ascii=False)+'\n')
    (O/'checks.log').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in ck['checks']))
    (O/'provenance.json').write_text(json.dumps(pv,indent=2,ensure_ascii=False)+'\n')
    print('P0-7 recalculation complete',flush=True)

if __name__=='__main__': main()
